"""Output, diagnostics, and visualization support for the coupled falling-ice run.

This module is deliberately independent of Taichi so geometry and reporting
contracts remain testable on hosts without a CUDA runtime.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import IceFlowConfig
from .config import LatticeScales


DEFAULT_SCENARIO_LABEL = "falling-ice melting"


@dataclass(slots=True)
class CoupledSnapshot:
    """Host copy of one synchronized LBM/thermal state."""

    physical_time_s: float
    thermal_time_s: float
    lbm_steps: int
    thermal_substeps: int
    temperature_c: np.ndarray
    liquid_fraction: np.ndarray
    enthalpy_j_m3: np.ndarray
    water_phase: np.ndarray
    solid: np.ndarray
    sdf: np.ndarray
    phase_change_material: np.ndarray
    velocity_lattice: np.ndarray
    physical_velocity_lattice: np.ndarray
    solid_volume_cells: float
    sharp_geometry_volume_cells: float
    solid_area_m2: float
    melted_fraction: float
    equivalent_square_side_m: float
    equivalent_uniform_melt_depth_m: float
    generated_water_volume_cells: float
    phase_change_contraction_cells: float
    water_volume_target_cells: float
    water_volume_current_cells: float
    water_volume_residual_cells: float
    total_enthalpy_j_m: float
    boundary_heat_input_j_m: float
    energy_residual_j_m: float


@dataclass(frozen=True, slots=True)
class VelocitySequenceSummary:
    """Files and the fixed scale used by the velocity visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_max_m_s: float
    color_max_m_s: float
    quiver_stride_cells: int


@dataclass(frozen=True, slots=True)
class VorticitySequenceSummary:
    """Files and the fixed scale used by the vorticity visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_abs_max_s_1: float
    color_abs_max_s_1: float
    quiver_stride_cells: int


@dataclass(frozen=True, slots=True)
class TemperatureSequenceSummary:
    """Files and the fixed scale used by the temperature visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_min_c: float
    observed_max_c: float
    color_min_c: float
    color_max_c: float


@dataclass(frozen=True, slots=True)
class VisualizationLimits:
    """Color and vector scales shared by every case in one invocation."""

    velocity_max_m_s: float
    vorticity_abs_max_s_1: float
    temperature_min_c: float
    temperature_max_c: float


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _shared_visualization_limits(
    args: argparse.Namespace,
    temperatures_c: list[float] | tuple[float, ...],
) -> VisualizationLimits:
    """Resolve plot limits once so every temperature case uses the same scale."""

    if not temperatures_c:
        raise ValueError("at least one water temperature is required")
    temperatures = [float(value) for value in temperatures_c]
    if not all(math.isfinite(value) for value in temperatures):
        raise ValueError("visualization temperatures must be finite")
    temperature_min = float(args.melting_temperature_c)
    if not math.isfinite(temperature_min):
        raise ValueError("melting_temperature_c must be finite")
    temperature_max = max(temperatures)
    if temperature_max < temperature_min:
        raise ValueError(
            "temperature visualization maximum must not be below "
            "the melting temperature"
        )
    velocity_max = _positive(
        "velocity_visualization_max_m_s",
        args.velocity_visualization_max_m_s,
    )
    vorticity_max = _positive(
        "vorticity_visualization_max_s_1",
        args.vorticity_visualization_max_s_1,
    )
    return VisualizationLimits(
        velocity_max_m_s=velocity_max,
        vorticity_abs_max_s_1=vorticity_max,
        temperature_min_c=temperature_min,
        temperature_max_c=temperature_max,
    )


def _requested_snapshot_times(end_time_s: float, interval_s: float) -> list[float]:
    end = _positive("end_time_s", end_time_s)
    interval = _positive("output_interval_s", interval_s)
    times = [0.0]
    count = int(math.floor(end / interval + 1.0e-12))
    times.extend(index * interval for index in range(1, count + 1))
    if not math.isclose(times[-1], end, rel_tol=0.0, abs_tol=1.0e-12):
        times.append(end)
    return times


def _snapshot_step_targets(
    config: IceFlowConfig, end_time_s: float, interval_s: float
) -> list[int]:
    scales = LatticeScales.from_iceflow_config(config)
    update_interval = int(config.thermal.update_interval_lbm_steps)
    targets = [0]
    for requested_time in _requested_snapshot_times(end_time_s, interval_s)[1:]:
        updates = math.ceil(requested_time / (scales.dt_s * update_interval) - 1.0e-12)
        target = max(update_interval, int(updates) * update_interval)
        if target > targets[-1]:
            targets.append(target)
    return targets


def _capture_snapshot(
    simulation,
    *,
    initial_total_enthalpy_j_m: float,
    initial_solid_volume_cells: float,
    thermal_totals=None,
) -> CoupledSnapshot:
    """Synchronize diagnostics and copy Taichi's ``(nx, ny)`` fields to host."""

    solid_volume = float(simulation.phase_change_solid_volume_cells())
    sharp_geometry_volume = float(simulation.phase_change_geometry_volume_cells())
    initial_solid = float(initial_solid_volume_cells)
    melted_solid = max(0.0, initial_solid - solid_volume)
    density_ratio = float(simulation.config.rho_ice / simulation.config.rho_water)
    melted_fraction = 0.0 if initial_solid <= 0.0 else melted_solid / initial_solid
    dx = float(simulation.config.dx)
    solid_area = solid_volume * dx * dx
    initial_side = math.sqrt(initial_solid * dx * dx)
    equivalent_side = math.sqrt(max(0.0, solid_area))

    total_enthalpy = (
        float(thermal_totals.total_energy_j_m)
        if thermal_totals is not None and hasattr(thermal_totals, "total_energy_j_m")
        else float(simulation.thermal.total_enthalpy_j_m(simulation.wall_mask))
    )
    boundary_heat = float(simulation.thermal.boundary_heat_input_j_m[None])
    energy_residual = total_enthalpy - float(initial_total_enthalpy_j_m) - boundary_heat
    target = float(simulation.water_volume_target[None])
    current = float(simulation.water_volume_current[None])

    # Taichi uses (nx, ny); files and Matplotlib use conventional (ny, nx).
    def scalar_host(field, dtype=None):
        values = np.asarray(field.to_numpy(), dtype=dtype)
        return np.ascontiguousarray(values.T)

    sample_thermal = getattr(simulation, "sample_thermal_fields", None)
    if sample_thermal is not None:
        thermal_fields = {
            name: np.ascontiguousarray(values.T)
            for name, values in sample_thermal().items()
        }
    else:
        thermal_fields = {
            "temperature_c": scalar_host(simulation.temperature, np.float64),
            "liquid_fraction": scalar_host(simulation.liquid_fraction, np.float32),
            "enthalpy_j_m3": scalar_host(simulation.thermal_enthalpy, np.float64),
            "phase_change_material": scalar_host(
                simulation.phase_change_material, np.int8
            ),
        }

    momentum_velocity = np.asarray(
        simulation.momentum_velocity_lattice.to_numpy(), dtype=np.float32
    )
    force = np.asarray(
        simulation.fluid_acceleration_lattice.to_numpy(), dtype=np.float32
    )
    # The physical velocity in the forced LBM is the momentum velocity plus
    # the standard half-force correction.  Keep the legacy raw-u field while
    # storing this physical field for visualization and thermal postprocessing.
    physical_velocity = momentum_velocity + np.float32(0.5) * force
    momentum_velocity = np.ascontiguousarray(np.transpose(momentum_velocity, (1, 0, 2)))
    physical_velocity = np.ascontiguousarray(np.transpose(physical_velocity, (1, 0, 2)))
    return CoupledSnapshot(
        physical_time_s=float(simulation.physical_time_s),
        thermal_time_s=float(simulation.thermal.time_s),
        lbm_steps=int(simulation.steps),
        thermal_substeps=int(simulation.thermal.steps),
        temperature_c=thermal_fields["temperature_c"],
        liquid_fraction=thermal_fields["liquid_fraction"],
        enthalpy_j_m3=thermal_fields["enthalpy_j_m3"],
        water_phase=scalar_host(simulation.water_phase, np.float32),
        solid=scalar_host(simulation.solid_mask, np.int8),
        sdf=scalar_host(simulation.body_signed_distance_m, np.float32),
        phase_change_material=thermal_fields["phase_change_material"],
        velocity_lattice=momentum_velocity,
        physical_velocity_lattice=physical_velocity,
        solid_volume_cells=solid_volume,
        sharp_geometry_volume_cells=sharp_geometry_volume,
        solid_area_m2=solid_area,
        melted_fraction=melted_fraction,
        equivalent_square_side_m=equivalent_side,
        equivalent_uniform_melt_depth_m=0.5 * (initial_side - equivalent_side),
        generated_water_volume_cells=density_ratio * melted_solid,
        phase_change_contraction_cells=(1.0 - density_ratio) * melted_solid,
        water_volume_target_cells=target,
        water_volume_current_cells=current,
        water_volume_residual_cells=current - target,
        total_enthalpy_j_m=total_enthalpy,
        boundary_heat_input_j_m=boundary_heat,
        energy_residual_j_m=energy_residual,
    )


HISTORY_COLUMNS = (
    "physical_time_s",
    "thermal_time_s",
    "lbm_steps",
    "thermal_substeps",
    "solid_volume_cells",
    "sharp_geometry_volume_cells",
    "solid_area_m2",
    "melted_fraction",
    "equivalent_square_side_m",
    "equivalent_uniform_melt_depth_m",
    "generated_water_volume_cells",
    "phase_change_contraction_cells",
    "water_volume_target_cells",
    "water_volume_current_cells",
    "water_volume_residual_cells",
    "total_enthalpy_j_m",
    "boundary_heat_input_j_m",
    "energy_residual_j_m",
)


def _history_row(item: CoupledSnapshot) -> tuple[object, ...]:
    return (
        f"{item.physical_time_s:.17g}",
        f"{item.thermal_time_s:.17g}",
        item.lbm_steps,
        item.thermal_substeps,
        f"{item.solid_volume_cells:.17g}",
        f"{item.sharp_geometry_volume_cells:.17g}",
        f"{item.solid_area_m2:.17g}",
        f"{item.melted_fraction:.17g}",
        f"{item.equivalent_square_side_m:.17g}",
        f"{item.equivalent_uniform_melt_depth_m:.17g}",
        f"{item.generated_water_volume_cells:.17g}",
        f"{item.phase_change_contraction_cells:.17g}",
        f"{item.water_volume_target_cells:.17g}",
        f"{item.water_volume_current_cells:.17g}",
        f"{item.water_volume_residual_cells:.17g}",
        f"{item.total_enthalpy_j_m:.17g}",
        f"{item.boundary_heat_input_j_m:.17g}",
        f"{item.energy_residual_j_m:.17g}",
    )


def snapshot_arrays(
    config: IceFlowConfig, snapshots: list[CoupledSnapshot]
) -> dict[str, np.ndarray]:
    """Build an archive payload without an intermediate compressed file."""
    if not snapshots:
        raise ValueError("at least one snapshot is required")
    x_m = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
    y_m = (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dx
    return dict(
        x_m=x_m,
        y_m=y_m,
        physical_time_s=np.asarray([item.physical_time_s for item in snapshots]),
        thermal_time_s=np.asarray([item.thermal_time_s for item in snapshots]),
        lbm_steps=np.asarray([item.lbm_steps for item in snapshots], dtype=np.int64),
        temperature_c=np.stack([item.temperature_c for item in snapshots]),
        liquid_fraction=np.stack([item.liquid_fraction for item in snapshots]),
        enthalpy_j_m3=np.stack([item.enthalpy_j_m3 for item in snapshots]),
        water_phase=np.stack([item.water_phase for item in snapshots]),
        solid=np.stack([item.solid for item in snapshots]),
        sdf=np.stack([item.sdf for item in snapshots]),
        phase_change_material=np.stack(
            [item.phase_change_material for item in snapshots]
        ),
        velocity_lattice=np.stack([item.velocity_lattice for item in snapshots]),
        physical_velocity_lattice=np.stack(
            [item.physical_velocity_lattice for item in snapshots]
        ),
        solid_volume_cells=np.asarray(
            [item.solid_volume_cells for item in snapshots], dtype=np.float64
        ),
        sharp_geometry_volume_cells=np.asarray(
            [item.sharp_geometry_volume_cells for item in snapshots],
            dtype=np.float64,
        ),
        solid_area_m2=np.asarray(
            [item.solid_area_m2 for item in snapshots], dtype=np.float64
        ),
        melted_fraction=np.asarray(
            [item.melted_fraction for item in snapshots], dtype=np.float64
        ),
        generated_water_volume_cells=np.asarray(
            [item.generated_water_volume_cells for item in snapshots], dtype=np.float64
        ),
        phase_change_contraction_cells=np.asarray(
            [item.phase_change_contraction_cells for item in snapshots],
            dtype=np.float64,
        ),
        water_volume_target_cells=np.asarray(
            [item.water_volume_target_cells for item in snapshots], dtype=np.float64
        ),
        water_volume_current_cells=np.asarray(
            [item.water_volume_current_cells for item in snapshots], dtype=np.float64
        ),
        total_enthalpy_j_m=np.asarray(
            [item.total_enthalpy_j_m for item in snapshots], dtype=np.float64
        ),
        boundary_heat_input_j_m=np.asarray(
            [item.boundary_heat_input_j_m for item in snapshots], dtype=np.float64
        ),
        energy_residual_j_m=np.asarray(
            [item.energy_residual_j_m for item in snapshots], dtype=np.float64
        ),
    )


def write_fields_npz(
    path: Path, config: IceFlowConfig, snapshots: list[CoupledSnapshot]
) -> None:
    np.savez_compressed(path, **snapshot_arrays(config, snapshots))


def _static_wall_mask(config: IceFlowConfig) -> np.ndarray:
    """Return the plotting mask for the fixed container walls."""

    wall = np.zeros((config.ny, config.nx), dtype=bool)
    boundary = int(config.boundary_cells)
    wall[:boundary, :] = True
    wall[-boundary:, :] = True
    wall[:, :boundary] = True
    wall[:, -boundary:] = True
    return wall


def _physical_water_velocity(
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical velocity and the visible liquid-water mask for one frame."""

    velocity = np.asarray(snapshot.physical_velocity_lattice, dtype=np.float64)
    expected_vector_shape = (config.ny, config.nx, 2)
    if velocity.shape != expected_vector_shape:
        raise ValueError(
            f"velocity frame {frame_index} has shape {velocity.shape}, "
            f"expected {expected_vector_shape}"
        )
    if not np.isfinite(velocity).all():
        raise ValueError(f"velocity frame {frame_index} contains NaN or infinity")

    solid = np.asarray(snapshot.solid)
    water_phase = np.asarray(snapshot.water_phase)
    expected_scalar_shape = (config.ny, config.nx)
    if (
        solid.shape != expected_scalar_shape
        or water_phase.shape != expected_scalar_shape
    ):
        raise ValueError(
            f"velocity frame {frame_index} masks must have shape "
            f"{expected_scalar_shape}"
        )
    water = (~_static_wall_mask(config)) & (solid == 0) & (water_phase >= 0.5)
    return velocity * scales.velocity_scale_m_s, water


def _scaled_quiver_components(
    velocity_m_s: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    *,
    display_max_m_s: float,
    reference_arrow_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical velocity to plot-space arrows using one shared scale."""

    maximum = _positive("display_max_m_s", display_max_m_s)
    reference_length = _positive("reference_arrow_length_m", reference_arrow_length_m)
    sampled = velocity_m_s[np.ix_(y_indices, x_indices)]
    factor = reference_length / maximum
    return sampled[..., 0] * factor, sampled[..., 1] * factor


def _masked_centered_axis_derivative(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    spacing_m: float,
    axis: int,
) -> np.ndarray:
    """Apply a centered derivative only when both neighbors are valid water."""

    scalar = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if scalar.shape != mask.shape or scalar.ndim != 2:
        raise ValueError(
            "masked derivative values and mask must be matching 2-D arrays"
        )
    if axis not in (0, 1):
        raise ValueError("masked derivative axis must be 0 or 1")
    spacing = _positive("spacing_m", spacing_m)

    forward_value = np.roll(scalar, -1, axis=axis)
    backward_value = np.roll(scalar, 1, axis=axis)
    forward_valid = np.roll(mask, -1, axis=axis)
    backward_valid = np.roll(mask, 1, axis=axis)
    forward_boundary = [slice(None), slice(None)]
    backward_boundary = [slice(None), slice(None)]
    forward_boundary[axis] = -1
    backward_boundary[axis] = 0
    forward_valid[tuple(forward_boundary)] = False
    backward_valid[tuple(backward_boundary)] = False

    derivative = np.full(scalar.shape, np.nan, dtype=np.float64)
    centered = mask & forward_valid & backward_valid
    derivative[centered] = (forward_value[centered] - backward_value[centered]) / (
        2.0 * spacing
    )
    return derivative


def _water_vorticity_s_1(
    velocity_m_s: np.ndarray,
    visible_water: np.ndarray,
    *,
    spacing_m: float,
) -> np.ndarray:
    r"""Return :math:`\omega_z=\partial_x v-\partial_y u` in inverse seconds."""

    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    water = np.asarray(visible_water, dtype=bool)
    if velocity.shape != (*water.shape, 2) or water.ndim != 2:
        raise ValueError("velocity and visible-water mask shapes are incompatible")
    if not np.isfinite(velocity).all():
        raise ValueError("velocity contains NaN or infinity")
    dv_dx = _masked_centered_axis_derivative(
        velocity[..., 1], water, spacing_m=spacing_m, axis=1
    )
    du_dy = _masked_centered_axis_derivative(
        velocity[..., 0], water, spacing_m=spacing_m, axis=0
    )
    vorticity = dv_dx - du_dy
    vorticity[~water] = np.nan
    return vorticity


def _contains_contour(values: np.ndarray, level: float) -> bool:
    finite = np.asarray(values)[np.isfinite(values)]
    return bool(finite.size and np.min(finite) < level < np.max(finite))


@dataclass(frozen=True, slots=True)
class _FlowPlotGeometry:
    extent_m: tuple[float, float, float, float]
    x_indices: np.ndarray
    y_indices: np.ndarray
    sample_x_m: np.ndarray
    sample_y_m: np.ndarray
    reference_arrow_length_m: float
    stride_cells: int


def _flow_plot_geometry(config: IceFlowConfig) -> _FlowPlotGeometry:
    extent_m = (
        0.0,
        config.nx * config.dx,
        0.0,
        config.ny * config.dx,
    )
    x_m = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
    y_m = (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dx
    stride = max(1, int(math.ceil(max(config.nx, config.ny) / 24.0)))
    boundary = int(config.boundary_cells)
    x_indices = np.arange(boundary, config.nx - boundary, stride)
    y_indices = np.arange(boundary, config.ny - boundary, stride)
    sample_x, sample_y = np.meshgrid(x_m[x_indices], y_m[y_indices])
    return _FlowPlotGeometry(
        extent_m=extent_m,
        x_indices=x_indices,
        y_indices=y_indices,
        sample_x_m=sample_x,
        sample_y_m=sample_y,
        reference_arrow_length_m=max(
            0.75 * stride * config.dx,
            1.0e-12,
        ),
        stride_cells=stride,
    )


def _draw_flow_boundaries(
    axis: object,
    snapshot: CoupledSnapshot,
    extent_m: tuple[float, float, float, float],
    *,
    water_surface_color: str,
) -> None:
    signed_distance = getattr(snapshot, "sdf", None)
    if signed_distance is not None and _contains_contour(signed_distance, 0.0):
        axis.contour(
            signed_distance,
            levels=(0.0,),
            colors="cyan",
            linewidths=1.2,
            origin="lower",
            extent=extent_m,
        )
    else:
        solid = np.asarray(snapshot.solid)
        if _contains_contour(solid, 0.5):
            axis.contour(
                solid,
                levels=(0.5,),
                colors="cyan",
                linewidths=1.2,
                origin="lower",
                extent=extent_m,
            )
    water_phase = np.asarray(snapshot.water_phase)
    if _contains_contour(water_phase, 0.5):
        axis.contour(
            water_phase,
            levels=(0.5,),
            colors=water_surface_color,
            linestyles="--",
            linewidths=0.9,
            origin="lower",
            extent=extent_m,
        )


class _StreamingGif:
    """Append PNG frames to a GIF while retaining at most one decoded frame."""

    def __init__(self, path: Path, *, duration_s: float = 0.8) -> None:
        self.path = path
        self._duration_ms = int(round(1000.0 * _positive("duration_s", duration_s)))
        self._stream = path.open("wb")
        self._frame_count = 0
        self._closed = False

    def append_png(self, path: Path) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed GIF stream")
        from PIL import GifImagePlugin, Image

        with Image.open(path) as source:
            frame = source.convert("RGB").convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=256,
            )
        try:
            frame_info = {
                "duration": self._duration_ms,
                "disposal": 2,
                "optimize": False,
            }
            if self._frame_count == 0:
                frame_info["loop"] = 0
                for block in GifImagePlugin._get_global_header(frame, frame_info):
                    self._stream.write(block)
            else:
                frame_info["include_color_table"] = True
            GifImagePlugin._write_frame_data(
                self._stream,
                frame,
                (0, 0),
                frame_info,
            )
            self._stream.flush()
            self._frame_count += 1
        finally:
            frame.close()

    def close(self) -> None:
        if not self._closed:
            if self._frame_count > 0:
                self._stream.write(b";")
            self._stream.close()
            if self._frame_count == 0:
                self.path.unlink(missing_ok=True)
            self._closed = True


def _render_velocity_frame(
    path: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_max_m_s: float,
    geometry: _FlowPlotGeometry,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
) -> float:
    """Render one velocity frame and return its observed water-speed maximum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    velocity_m_s, visible_water = _physical_water_velocity(
        config, scales, snapshot, frame_index=frame_index
    )
    speed_m_s = np.hypot(velocity_m_s[..., 0], velocity_m_s[..., 1])
    frame_observed_max = (
        float(np.max(speed_m_s[visible_water])) if np.any(visible_water) else 0.0
    )
    plotted_speed = np.ma.masked_where(~visible_water, speed_m_s)
    normalization = Normalize(vmin=0.0, vmax=color_max_m_s)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    speed_image = axis.imshow(
        plotted_speed,
        origin="lower",
        extent=geometry.extent_m,
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="equal",
    )

    sampled_mask = visible_water[np.ix_(geometry.y_indices, geometry.x_indices)]
    arrow_u, arrow_v = _scaled_quiver_components(
        velocity_m_s,
        geometry.x_indices,
        geometry.y_indices,
        display_max_m_s=color_max_m_s,
        reference_arrow_length_m=geometry.reference_arrow_length_m,
    )
    sampled_speed = np.hypot(arrow_u, arrow_v)
    arrow_mask = sampled_mask & (sampled_speed > 0.0)
    arrow_u = np.ma.masked_where(~arrow_mask, arrow_u)
    arrow_v = np.ma.masked_where(~arrow_mask, arrow_v)
    if frame_observed_max > 0.0:
        quiver = axis.quiver(
            geometry.sample_x_m,
            geometry.sample_y_m,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            color="white",
            width=0.004,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.quiverkey(
            quiver,
            X=0.92,
            Y=0.04,
            U=geometry.reference_arrow_length_m,
            label=f"{color_max_m_s:.3e} m/s",
            labelpos="W",
            labelcolor="white",
            coordinates="axes",
        )
    else:
        axis.text(
            0.97,
            0.04,
            "all water velocities = 0",
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    _draw_flow_boundaries(
        axis,
        snapshot,
        geometry.extent_m,
        water_surface_color="white",
    )
    figure.colorbar(
        speed_image,
        ax=axis,
        label="water speed |u| (m/s), shared across all cases",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        f"Water velocity during {scenario_label}\n"
        f"T_water,0={config.thermal.initial_water_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return frame_observed_max


class VelocitySequenceStream:
    """Write velocity PNG/GIF frames incrementally with constant frame memory."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        scales: LatticeScales,
        *,
        display_max_m_s: float,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
    ) -> None:
        self.config = config
        self.scales = scales
        self.color_max_m_s = _positive("display_max_m_s", display_max_m_s)
        self.scenario_label = str(scenario_label)
        self.geometry = _flow_plot_geometry(config)
        self.frame_directory = output_dir / "velocity_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "velocity_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_max_m_s = 0.0
        self._closed = False
        self._summary: VelocitySequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed velocity sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        frame_observed_max = _render_velocity_frame(
            frame_path,
            self.config,
            self.scales,
            snapshot,
            frame_index=self.frame_count,
            color_max_m_s=self.color_max_m_s,
            geometry=self.geometry,
            scenario_label=self.scenario_label,
        )
        self._gif.append_png(frame_path)
        self.observed_max_m_s = max(
            self.observed_max_m_s,
            frame_observed_max,
        )
        self.frame_count += 1
        return frame_path

    def close(self) -> VelocitySequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = VelocitySequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_max_m_s=self.observed_max_m_s,
                color_max_m_s=self.color_max_m_s,
                quiver_stride_cells=self.geometry.stride_cells,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def _render_vorticity_frame(
    path: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_abs_max_s_1: float,
    quiver_speed_max_m_s: float,
    geometry: _FlowPlotGeometry,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
) -> float:
    """Render one vorticity frame and return its observed absolute maximum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    velocity_m_s, visible_water = _physical_water_velocity(
        config, scales, snapshot, frame_index=frame_index
    )
    vorticity_s_1 = _water_vorticity_s_1(
        velocity_m_s,
        visible_water,
        spacing_m=config.dx,
    )
    finite_vorticity = np.isfinite(vorticity_s_1)
    frame_observed_abs_max = (
        float(np.max(np.abs(vorticity_s_1[finite_vorticity])))
        if np.any(finite_vorticity)
        else 0.0
    )
    speed_m_s = np.hypot(velocity_m_s[..., 0], velocity_m_s[..., 1])
    frame_observed_velocity_max = (
        float(np.max(speed_m_s[visible_water])) if np.any(visible_water) else 0.0
    )
    plotted_vorticity = np.ma.masked_invalid(vorticity_s_1)
    normalization = Normalize(
        vmin=-color_abs_max_s_1,
        vmax=color_abs_max_s_1,
    )
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    vorticity_image = axis.imshow(
        plotted_vorticity,
        origin="lower",
        extent=geometry.extent_m,
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="equal",
    )

    sampled_mask = finite_vorticity[np.ix_(geometry.y_indices, geometry.x_indices)]
    arrow_u, arrow_v = _scaled_quiver_components(
        velocity_m_s,
        geometry.x_indices,
        geometry.y_indices,
        display_max_m_s=quiver_speed_max_m_s,
        reference_arrow_length_m=geometry.reference_arrow_length_m,
    )
    sampled_speed = np.hypot(arrow_u, arrow_v)
    arrow_mask = sampled_mask & (sampled_speed > 0.0)
    arrow_u = np.ma.masked_where(~arrow_mask, arrow_u)
    arrow_v = np.ma.masked_where(~arrow_mask, arrow_v)
    if frame_observed_velocity_max > 0.0:
        quiver = axis.quiver(
            geometry.sample_x_m,
            geometry.sample_y_m,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            color="black",
            edgecolor="white",
            linewidth=0.35,
            width=0.004,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.quiverkey(
            quiver,
            X=0.92,
            Y=0.04,
            U=geometry.reference_arrow_length_m,
            label=f"{quiver_speed_max_m_s:.3e} m/s",
            labelpos="W",
            labelcolor="black",
            coordinates="axes",
        )
    else:
        axis.text(
            0.97,
            0.04,
            "all water velocities = 0",
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    _draw_flow_boundaries(
        axis,
        snapshot,
        geometry.extent_m,
        water_surface_color="black",
    )
    figure.colorbar(
        vorticity_image,
        ax=axis,
        label=(
            r"water vorticity $\omega_z=\partial_x v-\partial_y u$ "
            r"(s$^{-1}$), shared across all cases"
        ),
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        f"Water vorticity during {scenario_label}\n"
        f"T_water,0={config.thermal.initial_water_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return frame_observed_abs_max


class VorticitySequenceStream:
    """Write vorticity PNG/GIF frames incrementally with constant frame memory."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        scales: LatticeScales,
        *,
        display_abs_max_s_1: float,
        velocity_display_max_m_s: float,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
    ) -> None:
        self.config = config
        self.scales = scales
        self.color_abs_max_s_1 = _positive(
            "display_abs_max_s_1",
            display_abs_max_s_1,
        )
        self.quiver_speed_max_m_s = _positive(
            "velocity_display_max_m_s",
            velocity_display_max_m_s,
        )
        self.scenario_label = str(scenario_label)
        self.geometry = _flow_plot_geometry(config)
        self.frame_directory = output_dir / "vorticity_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "vorticity_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_abs_max_s_1 = 0.0
        self._closed = False
        self._summary: VorticitySequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed vorticity sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        frame_observed_abs_max = _render_vorticity_frame(
            frame_path,
            self.config,
            self.scales,
            snapshot,
            frame_index=self.frame_count,
            color_abs_max_s_1=self.color_abs_max_s_1,
            quiver_speed_max_m_s=self.quiver_speed_max_m_s,
            geometry=self.geometry,
            scenario_label=self.scenario_label,
        )
        self._gif.append_png(frame_path)
        self.observed_abs_max_s_1 = max(
            self.observed_abs_max_s_1,
            frame_observed_abs_max,
        )
        self.frame_count += 1
        return frame_path

    def close(self) -> VorticitySequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = VorticitySequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_abs_max_s_1=self.observed_abs_max_s_1,
                color_abs_max_s_1=self.color_abs_max_s_1,
                quiver_stride_cells=self.geometry.stride_cells,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def _temperature_field_for_plot(
    config: IceFlowConfig,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Return temperature masked to the water/ice thermal domain."""

    expected_shape = (config.ny, config.nx)
    temperature_c = np.asarray(snapshot.temperature_c, dtype=np.float64)
    water_phase = np.asarray(snapshot.water_phase)
    phase_change_material = np.asarray(snapshot.phase_change_material)
    if (
        temperature_c.shape != expected_shape
        or water_phase.shape != expected_shape
        or phase_change_material.shape != expected_shape
    ):
        raise ValueError(
            f"temperature frame {frame_index} fields must have shape {expected_shape}"
        )

    visible = (~_static_wall_mask(config)) & (
        (water_phase >= 0.5) | (phase_change_material != 0)
    )
    if not np.any(visible):
        raise ValueError(f"temperature frame {frame_index} has no visible cells")
    if not np.isfinite(temperature_c[visible]).all():
        raise ValueError(
            f"temperature frame {frame_index} contains NaN or infinity in "
            "the water/ice domain"
        )
    return np.ma.masked_where(~visible, temperature_c), visible


def _render_temperature_frame(
    path: Path,
    config: IceFlowConfig,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_min_c: float,
    color_max_c: float,
    scenario_label: str = DEFAULT_SCENARIO_LABEL,
) -> tuple[float, float]:
    """Render one temperature frame and return its observed range."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plotted_temperature, visible = _temperature_field_for_plot(
        config,
        snapshot,
        frame_index=frame_index,
    )
    visible_temperature = np.asarray(snapshot.temperature_c, dtype=np.float64)[visible]
    observed_min_c = float(np.min(visible_temperature))
    observed_max_c = float(np.max(visible_temperature))
    extent_m = (
        0.0,
        config.nx * config.dx,
        0.0,
        config.ny * config.dx,
    )
    colormap = plt.get_cmap("inferno").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    temperature_image = axis.imshow(
        plotted_temperature,
        origin="lower",
        extent=extent_m,
        cmap=colormap,
        norm=Normalize(vmin=color_min_c, vmax=color_max_c),
        interpolation="nearest",
        aspect="equal",
    )
    _draw_flow_boundaries(
        axis,
        snapshot,
        extent_m,
        water_surface_color="white",
    )
    figure.colorbar(
        temperature_image,
        ax=axis,
        label="temperature (°C), shared across all cases",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        f"Temperature during {scenario_label}\n"
        f"T_water,0={config.thermal.initial_water_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return observed_min_c, observed_max_c


class TemperatureSequenceStream:
    """Write temperature PNG/GIF frames incrementally with fixed limits."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        *,
        temperature_min_c: float,
        temperature_max_c: float,
        scenario_label: str = DEFAULT_SCENARIO_LABEL,
    ) -> None:
        color_min_c = float(temperature_min_c)
        color_max_c = float(temperature_max_c)
        if not math.isfinite(color_min_c) or not math.isfinite(color_max_c):
            raise ValueError("temperature visualization limits must be finite")
        if color_max_c < color_min_c:
            raise ValueError("temperature_max_c must not be below temperature_min_c")
        self.config = config
        self.scenario_label = str(scenario_label)
        self.color_min_c = color_min_c
        self.color_max_c = color_max_c
        self.frame_directory = output_dir / "temperature_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "temperature_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_min_c = math.inf
        self.observed_max_c = -math.inf
        self._closed = False
        self._summary: TemperatureSequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed temperature sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        observed_min_c, observed_max_c = _render_temperature_frame(
            frame_path,
            self.config,
            snapshot,
            frame_index=self.frame_count,
            color_min_c=self.color_min_c,
            color_max_c=self.color_max_c,
            scenario_label=self.scenario_label,
        )
        self._gif.append_png(frame_path)
        self.observed_min_c = min(self.observed_min_c, observed_min_c)
        self.observed_max_c = max(self.observed_max_c, observed_max_c)
        self.frame_count += 1
        return frame_path

    def close(self) -> TemperatureSequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = TemperatureSequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_min_c=self.observed_min_c,
                observed_max_c=self.observed_max_c,
                color_min_c=self.color_min_c,
                color_max_c=self.color_max_c,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _TerminalProgress:
    """Small dependency-free progress display with optional run details."""

    def __init__(
        self,
        total,
        *,
        enabled,
        stream=None,
        width=24,
        label="Simulating rigid ice",
        unit="frames",
    ):
        self.total = max(0, int(total))
        self.enabled = bool(enabled) and self.total > 0
        self.stream = stream if stream is not None else sys.stderr
        self.width = max(1, int(width))
        self.label = str(label)
        self.unit = str(unit)
        self.status = ""
        self.completed = 0
        self._started_at = None
        self._last_line_length = 0
        self._closed = False

    def __enter__(self):
        if self.enabled:
            self._started_at = time.monotonic()
            self._render(self._started_at)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def advance(self, amount=1):
        increment = int(amount)
        if increment < 0:
            raise ValueError("progress increment must be non-negative")
        self.completed = min(self.total, self.completed + increment)
        if self.enabled:
            self._render(time.monotonic())

    def set_status(self, status):
        """Replace the trailing status text and render it immediately."""

        self.status = str(status)
        if self.enabled and self._started_at is not None:
            self._render(time.monotonic())

    def close(self):
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, now):
        started_at = self._started_at if self._started_at is not None else now
        elapsed = max(0.0, now - started_at)
        fraction = self.completed / self.total
        filled = min(self.width, int(fraction * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.completed / elapsed if self.completed > 0 and elapsed > 0 else 0.0
        eta = (self.total - self.completed) / rate if rate > 0 else None
        line = (
            f"{self.label} [{bar}] {fraction:6.1%} | "
            f"{self.completed}/{self.total} {self.unit} | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
        )
        if self.status:
            line += f" | {self.status}"
        padding = " " * max(0, self._last_line_length - len(line))
        self.stream.write(f"\r{line}{padding}")
        self.stream.flush()
        self._last_line_length = len(line)
