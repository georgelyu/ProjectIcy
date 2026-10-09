"""Bounded interface correction for a prescribed, fixed-pose full melt.

This experiment leaves the production step unchanged. The reference projection
uses NumPy/SciPy; a Taichi kernel synchronizes the corrected phase with the real
LBM populations. It validates total water volume, not moving-boundary physics,
energy conservation, or time-dependent stability.

Run: python -m iceflow2d_v3.benchmarks.check_simple_melt_volume
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage
import taichi as ti

from iceflow2d_v3.lattice import (
    D2Q9_DIRECTION_COUNT,
    mixture_density,
    momentum_equilibrium,
    phase_equilibrium,
)
from iceflow2d_v3.simulator import IceFlow2D
from iceflow2d_v3.test_iceflow import _falling_config


_CROSS = ndimage.generate_binary_structure(2, 1)


def correct_full_melt_volume(phase, fluid, melted_cells, target_water_cells):
    """Seed melt water, then adjust its connected water region at its surface.

    Inputs are host arrays in either consistent axis order. ``melted_cells``
    must describe cells vacated by a prescribed full melt at a fixed pose; it
    must not contain cells merely exposed by rigid motion. A free surface and
    enough fluid space are required. Return a new f32 phase array and metrics.
    """
    original = np.asarray(phase)
    fluid = np.asarray(fluid, dtype=bool)
    seeds = np.asarray(melted_cells, dtype=bool)
    if (
        original.ndim != 2
        or fluid.shape != original.shape
        or seeds.shape != original.shape
    ):
        raise ValueError("phase, fluid, and melted_cells must share one 2D shape")
    if not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
        raise ValueError("phase must be finite and bounded by zero and one")
    if not seeds.any() or np.any(seeds & ~fluid):
        raise ValueError("melted_cells must select newly available fluid cells")
    target = float(target_water_cells)
    if not np.isfinite(target) or not 0 <= target <= np.count_nonzero(fluid):
        raise ValueError("target water volume exceeds available fluid capacity")

    phi = original.astype(np.float64, copy=True)
    phi[seeds] = 1.0
    seeded_volume = float(phi[fluid].sum(dtype=np.float64))
    labels, _ = ndimage.label(fluid & (phi > 0), structure=_CROSS)
    seed_labels = np.unique(labels[seeds])
    component = fluid & np.isin(labels, seed_labels[seed_labels > 0])
    outside_volume = float(phi[fluid & ~component].sum(dtype=np.float64))
    if target < outside_volume - 1e-10:
        raise ValueError("target would require removing water from another component")

    adjustable = np.zeros_like(fluid)
    layers = 0
    for _ in range(sum(phi.shape)):
        residual = target - float(phi[fluid].sum(dtype=np.float64))
        if abs(residual) <= 1e-10:
            break
        partial = component & (phi > 0) & (phi < 1)
        if residual < 0:
            air_neighbors = ndimage.binary_dilation(
                fluid & (phi == 0), structure=_CROSS
            )
            band = component & (phi > 0) & (partial | air_neighbors)
            capacities = phi[band]
        else:
            new_air = (
                fluid
                & (phi == 0)
                & ndimage.binary_dilation(component & (phi > 0), structure=_CROSS)
            )
            band = partial | new_air
            capacities = 1.0 - phi[band]
        capacity = float(capacities.sum(dtype=np.float64))
        if capacity <= 0:
            raise ValueError("no reachable gas-water interface has correction capacity")
        fraction = min(1.0, abs(residual) / capacity)
        if residual < 0:
            phi[band] *= 1.0 - fraction
        else:
            phi[band] += fraction * (1.0 - phi[band])
            component |= band
        adjustable |= band
        layers += 1
    else:
        raise RuntimeError("interface correction did not converge")

    result = phi.astype(np.float32)
    # Closing one interface value after the f32 cast avoids a coherent rounding
    # error across a flat interface. No bulk or unrelated water cell is used.
    for _ in range(4):
        residual = target - float(result[fluid].sum(dtype=np.float64))
        if abs(residual) <= 1e-7:
            break
        values = result.astype(np.float64)
        eligible = adjustable & (values + residual >= 0) & (values + residual <= 1)
        indices = np.flatnonzero(eligible)
        if not indices.size:
            raise RuntimeError("no interface cell can absorb the rounding residual")
        selected = indices[np.argmin(np.abs(values.flat[indices]))]
        result.flat[selected] = np.float32(values.flat[selected] + residual)
    final_volume = float(result[fluid].sum(dtype=np.float64))
    if abs(final_volume - target) > 1e-7:
        raise RuntimeError("f32 water-volume residual exceeds 1e-7 cells")
    return result, {
        "seeded_water_cells": seeded_volume,
        "interface_correction_cells": target - seeded_volume,
        "interface_layers": layers,
        "adjusted_interface_cells": int(adjustable.sum()),
        "final_water_cells": final_volume,
        "residual_cells": final_volume - target,
    }


@ti.kernel
def _apply_phase(
    simulation: ti.template(), corrected: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    """Lift equilibrium parts and close the phase zeroth moment on the GPU."""
    for i, j in simulation.water_phase:
        if simulation._is_fluid_cell(i, j):
            old_phi = simulation.water_phase[i, j]
            new_phi = corrected[i, j]
            if new_phi != old_phi:
                velocity = simulation.momentum_velocity_lattice[i, j]
                phase_velocity = (
                    velocity + 0.5 * simulation.fluid_acceleration_lattice[i, j]
                )
                pressure = (
                    simulation.pressure_lattice[i, j]
                    - simulation.hydrostatic_reference_pressure[i, j]
                )
                old_rho = mixture_density(
                    old_phi,
                    simulation._water_density_lattice,
                    simulation._air_density_lattice,
                )
                new_rho = mixture_density(
                    new_phi,
                    simulation._water_density_lattice,
                    simulation._air_density_lattice,
                )
                for q in range(D2Q9_DIRECTION_COUNT):
                    simulation.momentum_populations[i, j, q] += momentum_equilibrium(
                        q, pressure, new_rho, velocity
                    ) - momentum_equilibrium(q, pressure, old_rho, velocity)
                    simulation.phase_populations[i, j, q] += phase_equilibrium(
                        q, new_phi, phase_velocity
                    ) - phase_equilibrium(q, old_phi, phase_velocity)
                moving_sum = ti.cast(0.0, ti.f64)
                for q in range(1, D2Q9_DIRECTION_COUNT):
                    moving_sum += ti.cast(simulation.phase_populations[i, j, q], ti.f64)
                simulation.phase_populations[i, j, 0] = ti.cast(
                    ti.cast(new_phi, ti.f64) - moving_sum, ti.f32
                )
                simulation.water_phase[i, j] = new_phi


def run_case(placement, rho_ice):
    config = _falling_config(
        rho_ice=rho_ice,
        ice_base_y_cells=8.0 if placement == "water" else 30.0,
        ice_initial_angle=0.0,
        gravity=(0.0, 0.0),
        sigma=0.0,
        phase_warmup_steps=0,
    )
    simulation = IceFlow2D(config)
    wall = simulation.wall_mask.to_numpy().astype(bool)
    ice_before = simulation.solid_mask.to_numpy().astype(bool)
    phase_before = simulation.water_phase.to_numpy()
    initial_water = float(phase_before[~wall & ~ice_before].sum(dtype=np.float64))
    initial_mass = float(
        simulation.thermal.body_solid_mass.to_numpy().sum(dtype=np.float64)
    )
    simulation.thermal._body_heat_delta.fill(0)
    simulation.thermal.body_sensible_energy.fill(
        2
        * simulation.thermal._initial_body_cell_mass
        * simulation.thermal._latent_heat_j_kg
    )
    simulation.thermal._apply_body_heat_and_phase_change()
    simulation._refresh_eroded_body()
    final_mass = float(
        simulation.thermal.body_solid_mass.to_numpy().sum(dtype=np.float64)
    )
    fluid = ~wall & (simulation.solid_mask.to_numpy() == 0)
    melted_cells = ice_before & fluid
    expected_increment = (initial_mass - final_mass) / (config.rho_water * config.dx**2)
    target = initial_water + expected_increment
    refilled = simulation.water_phase.to_numpy()
    before_projection = float(refilled[fluid].sum(dtype=np.float64))
    start = perf_counter()
    corrected, metrics = correct_full_melt_volume(refilled, fluid, melted_cells, target)
    projection_ms = (perf_counter() - start) * 1000
    _apply_phase(simulation, corrected)
    # Recover from populations to check that a later recovery does not undo the
    # correction. No collision, flow time step, or thermal evolution is run.
    simulation._update_streamed_macroscopic_fields()
    simulation._update_pressure()
    recovered = simulation.water_phase.to_numpy()
    np.testing.assert_array_equal(recovered, corrected)
    actual_water = float(recovered[fluid].sum(dtype=np.float64))
    populations = simulation.phase_populations.to_numpy()
    population_error = float(
        np.max(
            np.abs(populations.sum(axis=-1, dtype=np.float64)[fluid] - recovered[fluid])
        )
    )
    assert abs(actual_water - target) <= 1e-7
    assert population_error < 1e-7
    assert final_mass == 0
    assert np.all((recovered[fluid] >= 0) & (recovered[fluid] <= 1))
    if placement == "water":
        # Shrinkage is represented at the pool surface, not as artificial air
        # or reduced density at the submerged former ice location.
        np.testing.assert_array_equal(
            recovered[melted_cells], np.ones(melted_cells.sum())
        )
    else:
        # The isolated melt droplet must not take its mass from the old pool.
        old_pool = (~wall & ~ice_before) & (phase_before > 0)
        np.testing.assert_array_equal(recovered[old_pool], phase_before[old_pool])
    # Thermal synchronization is intentionally not claimed as conservative in
    # this isolated mass experiment. The production solver is unchanged.
    row = {
        "placement": placement,
        "rho_ice_kg_m3": rho_ice,
        "rho_water_kg_m3": config.rho_water,
        "initial_ice_cells": int(ice_before.sum()),
        "initial_water_cells": initial_water,
        "original_increment_cells": before_projection - initial_water,
        "required_increment_cells": expected_increment,
        "corrected_increment_cells": actual_water - initial_water,
        "population_zeroth_moment_max_error": population_error,
        "reference_projection_ms_excluding_transfers_and_compilation": projection_ms,
        **metrics,
    }
    return row, (refilled, recovered, ice_before, wall)


def check_projection_edges():
    # Negative correction larger than one interface layer, positive expansion,
    # invalid capacity, and unchanged inputs on a failed correction.
    for density in (0.5, 0.917, 1.0, 1.2):
        phase = np.zeros((16, 16), dtype=np.float32)
        fluid = np.ones_like(phase, dtype=bool)
        seeds = np.zeros_like(fluid)
        seeds[4:12, 4:12] = True
        result, _ = correct_full_melt_volume(phase, fluid, seeds, 64 * density)
        assert abs(result.sum(dtype=np.float64) - 64 * density) <= 1e-7
        assert result.min() >= 0 and result.max() <= 1
        np.testing.assert_array_equal(phase, np.zeros_like(phase))
    try:
        correct_full_melt_volume(phase, fluid, seeds, 257)
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible target was accepted")
    try:
        correct_full_melt_volume(np.ones_like(phase), fluid, seeds, 200)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a sealed all-water domain was given an artificial interface"
        )


def render_comparison(pictures, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    for row_index, placement in enumerate(("water", "air")):
        before, after, ice, wall = pictures[placement]
        for col_index, phase in enumerate((before, after)):
            axis = axes[row_index, col_index]
            data = np.ma.masked_where(wall.T, phase.T)
            artist = axis.imshow(
                data,
                origin="lower",
                vmin=0,
                vmax=1,
                cmap="Blues",
                interpolation="nearest",
            )
            axis.contour(
                ice.T.astype(float),
                levels=[0.5],
                colors=["#e87518"],
                linewidths=1.2,
                linestyles="dashed",
            )
            axis.set_title(
                f"{placement.capitalize()}: {'original refill' if col_index == 0 else 'volume corrected'}"
            )
            axis.set_xlabel("x / dx")
            axis.set_ylabel("y / dx")
    fig.colorbar(artist, ax=axes, shrink=0.7, label="Water fraction")
    fig.suptitle(
        "64-cell ice, density ratio 0.917; fixed-pose full melt\nOrange outline: former ice; no flow evolution",
        fontsize=12,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/simple_melt_volume"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_projection_edges()
    rows, pictures = [], {}
    for placement, density in (
        ("water", 500.0),
        ("water", 917.0),
        ("water", 1000.0),
        ("air", 917.0),
    ):
        row, arrays = run_case(placement, density)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if density == 917:
            pictures[placement] = arrays
    output = {
        "scope": "Fixed-pose complete melt; volume correction only; no production-step changes or energy/dynamic validation.",
        "algorithm": "Seed melt location, select its water component, remove/add interface volume in proportion to bounded capacity, close f32 rounding residual.",
        "implementation": "NumPy/SciPy reference projection and Taichi population synchronization; timing excludes transfers and compilation and is not a production GPU benchmark.",
        "rows": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    render_comparison(pictures, args.output_dir / "comparison.png")
    print(
        "All volume, bounds, locality, capacity, and population checks passed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
