"""Read-only operator water-volume accounting for the W=5 four-ice case.

No volume projection or source correction is enabled. Extra GPU reductions
observe actual populations and actual refill values; production kernels remain
unchanged. ``legacy`` may terminate at the existing instability guard.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config, parse_args
from iceflow2d_v3.lattice import (
    inside_grid,
    lattice_direction,
    opposite_direction_index,
)
from iceflow2d_v3.simulator import IceFlow2D


NAMES = (
    "collision_zeroth_rounding",
    "moving_wall_phase_flux",
    "interior_stream_residual",
    "macro_f32_rounding",
    "clip_negative_added",
    "clip_above_one_removed_signed",
    "motion_exposed_cells",
    "motion_covered_cells",
    "motion_exposed_water",
    "motion_covered_water",
    "motion_exposed_air_cells",
    "motion_exposed_interface_cells",
    "motion_exposed_water_cells",
    "melt_exposed_cells",
    "melt_covered_cells",
    "melt_exposed_water",
    "melt_covered_water",
    "melt_exposed_air_cells",
    "melt_exposed_interface_cells",
    "melt_exposed_water_cells",
)


@ti.data_oriented
class AuditedIceFlow2D(IceFlow2D):
    def __init__(self, config):
        self._audit_enabled = False
        super().__init__(config)
        self.audit_sums = ti.field(ti.f64, shape=len(NAMES))
        self._audit_enabled = True

    def _advance_fluid(self):
        # Keep the production call sequence, inserting only read-only kernels.
        # Taichi kernel descriptors must be bound through self, not super().
        self._collide_momentum()
        if self.config.phase_momentum_coupling == "legacy":
            self._collide_phase()
        self._stream()
        self._audit_stream()
        self._update_streamed_macroscopic_fields()
        self._audit_macro()
        self._update_pressure()
        self._audit_bounds()
        self._bound_water_phase()

    def _refresh_fluid_domain(self, *, resolve_contact):
        super()._refresh_fluid_domain(resolve_contact=resolve_contact)
        if self._audit_enabled:
            self._audit_geometry(0 if resolve_contact else 1)

    @ti.kernel
    def _audit_stream(self):
        collision = ti.cast(0.0, ti.f64)
        moving_wall = ti.cast(0.0, ti.f64)
        interior = ti.cast(0.0, ti.f64)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                incoming = ti.cast(0.0, ti.f64)
                outgoing = ti.cast(0.0, ti.f64)
                wall = ti.cast(0.0, ti.f64)
                for q in range(9):
                    post = ti.cast(self.phase_stream_buffer[i, j, q], ti.f64)
                    incoming += ti.cast(self.phase_populations[i, j, q], ti.f64)
                    outgoing += post
                    direction = lattice_direction(q)
                    ni, nj = i + direction.x, j + direction.y
                    if inside_grid(ni, nj, self.nx, self.ny):
                        if self.solid_mask[ni, nj] != 0:
                            # This is the value actually returned by streaming,
                            # including its f32 arithmetic, not a predicted flux.
                            reflected = ti.cast(
                                self.phase_populations[
                                    i, j, opposite_direction_index(q)
                                ],
                                ti.f64,
                            )
                            wall += reflected - post
                collision += outgoing - ti.cast(self.water_phase[i, j], ti.f64)
                moving_wall += wall
                interior += incoming - outgoing - wall
        self.audit_sums[0] += collision
        self.audit_sums[1] += moving_wall
        self.audit_sums[2] += interior

    @ti.kernel
    def _audit_macro(self):
        rounding = ti.cast(0.0, ti.f64)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                population_sum = ti.cast(0.0, ti.f64)
                for q in range(9):
                    population_sum += ti.cast(self.phase_populations[i, j, q], ti.f64)
                rounding += ti.cast(self.water_phase[i, j], ti.f64) - population_sum
        self.audit_sums[3] += rounding

    @ti.kernel
    def _audit_bounds(self):
        added = ti.cast(0.0, ti.f64)
        removed_signed = ti.cast(0.0, ti.f64)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                value = ti.cast(self.water_phase[i, j], ti.f64)
                if value < 0:
                    added -= value
                elif value > 1:
                    removed_signed += 1.0 - value
        self.audit_sums[4] += added
        self.audit_sums[5] += removed_signed

    @ti.kernel
    def _audit_geometry(self, kind: ti.template()):
        opened = ti.cast(0.0, ti.f64)
        covered = ti.cast(0.0, ti.f64)
        added_water = ti.cast(0.0, ti.f64)
        covered_water = ti.cast(0.0, ti.f64)
        air = ti.cast(0.0, ti.f64)
        interface = ti.cast(0.0, ti.f64)
        water = ti.cast(0.0, ti.f64)
        for i, j in self.water_phase:
            if self.wall_mask[i, j] == 0:
                old_solid = self.previous_solid_mask[i, j] != 0
                new_solid = self.solid_mask[i, j] != 0
                phi = ti.cast(self.water_phase[i, j], ti.f64)
                if old_solid and not new_solid:
                    opened += 1.0
                    added_water += phi
                    if phi <= 0.001:
                        air += 1.0
                    elif phi >= 0.999:
                        water += 1.0
                    else:
                        interface += 1.0
                elif not old_solid and new_solid:
                    # Production refill preserves phi in newly covered cells;
                    # this is precisely the old fluid value being excluded.
                    covered += 1.0
                    covered_water += phi
        self.audit_sums[ti.static(6 + 7 * kind)] += opened
        self.audit_sums[ti.static(7 + 7 * kind)] += covered
        self.audit_sums[ti.static(8 + 7 * kind)] += added_water
        self.audit_sums[ti.static(9 + 7 * kind)] += covered_water
        self.audit_sums[ti.static(10 + 7 * kind)] += air
        self.audit_sums[ti.static(11 + 7 * kind)] += interface
        self.audit_sums[ti.static(12 + 7 * kind)] += water


@ti.kernel
def measure_state(s: ti.template()) -> ti.types.vector(8, ti.f64):
    amount = ti.cast(0.0, ti.f64)
    cutoff_amount = ti.cast(0.0, ti.f64)
    solid_count = ti.cast(0.0, ti.f64)
    maximum_speed = ti.cast(0.0, ti.f64)
    minimum_phase = ti.cast(1e30, ti.f64)
    maximum_phase = ti.cast(-1e30, ti.f64)
    invalid = ti.cast(0.0, ti.f64)
    for i, j in s.water_phase:
        if s.solid_mask[i, j] != 0:
            solid_count += 1.0
        if s._is_fluid_cell(i, j):
            phase = ti.cast(s.water_phase[i, j], ti.f64)
            velocity = s._physical_velocity_at(i, j)
            speed = ti.cast(ti.abs(velocity.x) + ti.abs(velocity.y), ti.f64)
            amount += phase
            if phase > s.config.water_phase_cutoff:
                cutoff_amount += ti.min(1.0, ti.max(0.0, phase))
            ti.atomic_min(minimum_phase, phase)
            ti.atomic_max(maximum_phase, phase)
            ti.atomic_max(maximum_speed, speed)
            if (
                ti.math.isnan(phase)
                or ti.math.isinf(phase)
                or ti.math.isnan(speed)
                or ti.math.isinf(speed)
            ):
                invalid += 1.0
    return ti.Vector(
        [
            amount,
            cutoff_amount,
            solid_count,
            maximum_speed,
            minimum_phase,
            maximum_phase,
            invalid,
            0.0,
        ],
        dt=ti.f64,
    )


def snapshot(s, initial=None, *, partial_step=False):
    values = np.asarray(measure_state(s))
    ledger = dict(zip(NAMES, s.audit_sums.to_numpy().tolist()))
    groups = {
        "collision": ledger["collision_zeroth_rounding"],
        "moving_wall": ledger["moving_wall_phase_flux"],
        "interior_stream": ledger["interior_stream_residual"],
        "macro_recovery": ledger["macro_f32_rounding"],
        "bounds": ledger["clip_negative_added"]
        + ledger["clip_above_one_removed_signed"],
        "motion_geometry": ledger["motion_exposed_water"]
        - ledger["motion_covered_water"],
        "melt_geometry": ledger["melt_exposed_water"] - ledger["melt_covered_water"],
    }
    mass = float(s.body_mass_lattice.array().sum(dtype=np.float64))
    row = {
        "step": s.steps,
        "time_s": s.steps * s.scales.dt_s,
        "partial_next_step": partial_step,
        "water_cells": float(values[0]),
        "water_cutoff_cells": float(values[1]),
        "solid_cells": float(values[2]),
        "max_l1": float(values[3]),
        "phase_min": float(values[4]),
        "phase_max": float(values[5]),
        "invalid_cells": int(values[6]),
        "ice_mass_water_equivalent_cells": mass,
        "groups": groups,
        "ledger": ledger,
    }
    if initial is not None:
        delta = row["water_cells"] - initial["water_cells"]
        total0 = initial["water_cells"] + initial["ice_mass_water_equivalent_cells"]
        net_opened = sum(
            ledger[f"{kind}_exposed_cells"] - ledger[f"{kind}_covered_cells"]
            for kind in ("motion", "melt")
        )
        row.update(
            water_increase_cells=delta,
            predicted_water_increase_cells=sum(groups.values()),
            ledger_closure_residual_cells=delta - sum(groups.values()),
            geometric_cell_closure_residual=net_opened
            - (initial["solid_cells"] - row["solid_cells"]),
            total_mass_error_percent=(row["water_cells"] + mass - total0)
            / total0
            * 100,
            target_water_cells=total0 - mass,
        )
    return row


def self_check(coupling):
    from iceflow2d_v3.test_iceflow import _falling_config

    config = _falling_config(
        phase_warmup_steps=5,
        ice_base_y_cells=20.0,
        ice_initial_angle=0.0,
        phase_momentum_coupling=coupling,
    )
    plain = IceFlow2D(config)
    audited = AuditedIceFlow2D(config)
    initial = snapshot(audited)
    plain.step(128)
    audited.step(128)
    errors = {}
    for name in (
        "water_phase",
        "phase_populations",
        "momentum_populations",
        "body_mass_lattice",
        "solid_mask",
    ):
        a, b = getattr(plain, name).to_numpy(), getattr(audited, name).to_numpy()
        errors[name] = float(
            np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        )
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=2e-7)
    row = snapshot(audited, initial)
    assert abs(row["ledger_closure_residual_cells"]) < 1e-8
    assert row["geometric_cell_closure_residual"] == 0
    fluid = (audited.wall_mask.to_numpy() == 0) & (audited.solid_mask.to_numpy() == 0)
    phase = audited.water_phase.to_numpy()[fluid]
    velocity = audited.physical_velocity_lattice.to_numpy()[fluid]
    assert row["phase_min"] == float(phase.min())
    assert row["phase_max"] == float(phase.max())
    np.testing.assert_allclose(
        row["max_l1"], np.abs(velocity).sum(axis=-1).max(), rtol=1e-6
    )
    print(
        json.dumps(
            {
                "variant": coupling,
                "self_check_state_max_differences": errors,
                "self_check_closure": row["ledger_closure_residual_cells"],
            }
        ),
        flush=True,
    )


def fixed_controls(width, output):
    """Force all four blocks to melt, without flow or motion, in pure phases."""
    output.mkdir(parents=True, exist_ok=False)
    base = create_config(parse_args(["--interface-width", str(width)]))
    results = []
    for placement, center_y, expected in (("water", 40.0, 3600.0), ("air", 120.0, 0.0)):
        config = replace(
            base,
            phase_warmup_steps=0,
            ice_bodies=tuple(
                replace(body, center_cells=(body.center_cells[0], center_y))
                for body in base.ice_bodies
            ),
        )
        s = AuditedIceFlow2D(config)
        initial = snapshot(s)
        assert initial["solid_cells"] == 3600
        s.thermal._body_heat_delta.fill(0)
        s.thermal.body_sensible_energy.fill(
            2 * s.thermal._initial_body_cell_mass * s.thermal._latent_heat_j_kg
        )
        s.thermal._apply_body_heat_and_phase_change()
        after_material_water = float(measure_state(s)[0])
        s._refresh_eroded_body()
        final = snapshot(s, initial)
        assert final["solid_cells"] == 0
        assert final["ice_mass_water_equivalent_cells"] == 0
        assert abs(after_material_water - initial["water_cells"]) < 1e-8
        # Refill averages f32 neighbors; non-power-of-two stencil sizes can
        # introduce a few ulps even when every neighbor is exactly one.
        assert abs(final["water_increase_cells"] - expected) < 1e-3, final
        results.append(
            {
                "placement": placement,
                "width": width,
                "prescribed_full_melt": True,
                "no_flow_or_motion_steps": True,
                "phase_warmup_steps": 0,
                "initial": initial,
                "after_material_water_cells": after_material_water,
                "last": final,
            }
        )
    (output / "fixed_controls.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", choices=("legacy", "consistent_flux"), default="consistent_flux"
    )
    p.add_argument("--end", type=float, default=5.0)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--output", type=Path)
    p.add_argument("--sample-steps", type=int, default=1200)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--fixed-controls", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check(args.variant)
        return
    if args.output is None:
        p.error("--output is required unless --self-check is used")
    if args.fixed_controls:
        fixed_controls(args.width, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    config = create_config(
        parse_args(
            [
                "--interface-width",
                str(args.width),
                "--phase-momentum-coupling",
                args.variant,
            ]
        )
    )
    sources = [
        Path(__file__),
        *(
            Path(__file__).parents[1] / name
            for name in (
                "simulator.py",
                "config.py",
                "lattice.py",
                "stability_options.py",
                "body_state.py",
                "contact.py",
                "thermal_d2q5.py",
                "viscosity.py",
            )
        ),
    ]
    manifest = {
        "variant": args.variant,
        "config": config.to_dict(),
        "end_s": args.end,
        "instrumentation": "Read-only GPU operator reductions, no mass correction; warm-up excluded from ledger.",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    s = AuditedIceFlow2D(config)
    s._thermal_stability_check_enabled = True
    initial = snapshot(s)
    end_step = round(args.end / s.scales.dt_s)
    started = time.monotonic()
    rows, error = [], None
    with (args.output / "budget.jsonl").open("w") as out:
        while True:
            row = snapshot(s, initial, partial_step=error is not None)
            row["elapsed_s"] = time.monotonic() - started
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if len(rows) == 1 or s.steps % 12000 == 0 or error or s.steps >= end_step:
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in (
                                "step",
                                "time_s",
                                "water_increase_cells",
                                "ice_mass_water_equivalent_cells",
                                "ledger_closure_residual_cells",
                                "groups",
                                "elapsed_s",
                            )
                        }
                    ),
                    flush=True,
                )
            if error or s.steps >= end_step:
                break
            try:
                s.step(min(args.sample_steps, end_step - s.steps))
            except RuntimeError as exc:
                error = str(exc)
    np.savez_compressed(
        args.output / "final.npz",
        phi=s.water_phase.to_numpy(),
        solid=s.solid_mask.to_numpy(),
        velocity=s.physical_velocity_lattice.to_numpy(),
        body_mass=s.body_mass_lattice.array(),
        body_center=s.body_center.to_numpy(),
    )
    summary = {
        "completed": error is None and s.steps == end_step,
        "error": error,
        "variant": args.variant,
        "width": args.width,
        "initial": initial,
        "last": rows[-1],
        "max_sampled_abs_ledger_residual_cells": max(
            abs(r["ledger_closure_residual_cells"]) for r in rows
        ),
        "max_sampled_abs_mass_error_percent": max(
            abs(r["total_mass_error_percent"]) for r in rows
        ),
        "elapsed_s": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
