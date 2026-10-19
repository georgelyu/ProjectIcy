"""Reproduce the single-ice failure and replay matched-state causal controls.

Only scalar/point diagnostics are written. Checkpoints remain in RAM.
Production solver kernels and example defaults are unchanged.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    PointProbe,
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.examples.dambreak_single_ice_melting_2d import create_config
from iceflow2d_v3.lattice import (
    cross_product_2d,
    inside_grid,
    lattice_direction,
    lattice_weight,
    mixture_density,
    momentum_equilibrium,
)
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.phase_reconstruction import PhaseReconstruction


@ti.data_oriented
class SingleProbe(PointProbe):
    def __init__(self, config):
        ensure_taichi_cuda()
        self.control_mode = ti.field(ti.i32, shape=())
        super().__init__(config)
        self.probe_points_host = np.array(
            [
                (439, 3),
                (438, 3),
                (440, 3),
                (439, 4),
                (439, 5),
                (437, 3),
                (441, 3),
                (439, 2),
                (438, 4),
                (440, 4),
                (439, 3),
            ],
            dtype=np.int32,
        )
        self.probe_points.from_numpy(self.probe_points_host)

    @ti.kernel
    def _control_gradients(self):
        for i, j in self.water_phase:
            if self.control_mode[None] == 3 or (
                self.control_mode[None] == 1 and self.phase_gradient_status[i, j] != 1
            ):
                self.source_gradient_lattice[i, j] = ti.Vector([0.0, 0.0])
            if (
                (
                    self.control_mode[None] == 7
                    or self.control_mode[None] == 9
                    or self.control_mode[None] == 10
                )
                and j == 3
                and i >= 416
                and self._is_fluid_cell(i, j)
            ):
                phi = self.water_phase[i, j]
                before, after = phi, phi
                if self._is_fluid_cell(i - 1, j):
                    before = self.water_phase[i - 1, j]
                if self._is_fluid_cell(i + 1, j):
                    after = self.water_phase[i + 1, j]
                gradient = ti.Vector([0.5 * (after - before), 0.0])
                if self.control_mode[None] != 10:
                    self.source_gradient_lattice[i, j] = gradient
                if self.control_mode[None] != 9:
                    self.phase_gradient_lattice[i, j] = gradient

    def _prepare_phase_gradients(self):
        super()._prepare_phase_gradients()
        self._control_gradients()

    @ti.func
    def _fit_boundary_phase(self, i, j, quadratic):
        use_quadratic = quadratic
        if self.control_mode[None] == 8 and j == 3 and i >= 416:
            use_quadratic = True
        return PhaseReconstruction._fit_boundary_phase(self, i, j, use_quadratic)

    @ti.func
    def _phase_flux_force_density(self, i, j):
        force = ti.Vector.zero(ti.f32, 2)
        if self.control_mode[None] != 2:
            for axis, sign in ti.static(ti.ndrange(2, 2)):
                direction = 2 * sign - 1
                di, dj = 0, 0
                if ti.static(axis == 0):
                    di = direction
                else:
                    dj = direction
                index = self._stability_neighbor(i, j, di, dj)
                if inside_grid(index.x, index.y, self.nx, self.ny):
                    if self._is_fluid_cell(index.x, index.y):
                        mass = 0.5 * (
                            self.phase_mass_flux[i, j][axis]
                            + self.phase_mass_flux[index.x, index.y][axis]
                        )
                        velocity = 0.5 * (
                            self.stability_velocity[i, j]
                            + self.stability_velocity[index.x, index.y]
                        )
                        force -= direction * mass * velocity
        return force

    @ti.kernel
    def _positive_phase_collision(self):
        for i, j in self.water_phase:
            if self.control_mode[None] == 4 and self._is_fluid_cell(i, j):
                phi = self.water_phase[i, j]
                alpha = 1.0
                for q in range(9):
                    low = lattice_weight(q) * phi
                    high = self.phase_stream_buffer[i, j, q]
                    if high < 0:
                        alpha = ti.min(alpha, low / (low - high + 1e-30))
                for q in range(9):
                    low = lattice_weight(q) * phi
                    self.phase_stream_buffer[i, j, q] = low + alpha * (
                        self.phase_stream_buffer[i, j, q] - low
                    )

    def _collide_phase(self, *, gradients_prepared=False):
        super()._collide_phase(gradients_prepared=gradients_prepared)
        self._positive_phase_collision()

    @ti.func
    def _cut_link_fraction(self, i, j, ni, nj):
        fraction = IceFlow2D._cut_link_fraction(self, i, j, ni, nj)
        if self.control_mode[None] == 6:
            fraction = 0.5
        return fraction

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        prepared = IceFlow2D._prepare_momentum_cut_links(self, i, j, raw_post)
        if self.control_mode[None] == 5:
            density = mixture_density(
                self.water_phase[i, j],
                self._water_density_lattice,
                self._air_density_lattice,
            )
            for q in range(1, 9):
                c = lattice_direction(q)
                ni, nj = i + c.x, j + c.y
                if (
                    inside_grid(ni, nj, self.nx, self.ny)
                    and self.solid_mask[ni, nj] == 1
                ):
                    b = self.solid_body_id[ni, nj]
                    eta = self._cut_link_fraction(i, j, ni, nj)
                    direction = ti.cast(c, ti.f32)
                    point = self._cell_point(i, j) + eta * direction
                    ub = self._body_velocity_at(point, b)
                    reflected = raw_post[q] - 6 * lattice_weight(
                        q
                    ) * density * direction.dot(ub)
                    correction = ti.cast(
                        (-direction - ub) * (prepared[q] - reflected), ti.f64
                    )
                    prepared[q] = reflected
                    ti.atomic_add(self.hydrodynamic_impulse[b], correction)
                    ti.atomic_add(
                        self.hydrodynamic_torque[b],
                        cross_product_2d(
                            ti.cast(point - self.body_center[b], ti.f64), correction
                        ),
                    )
        return prepared

    @ti.kernel
    def reset_gap(self, populations: ti.i32):
        for i, j in self.water_phase:
            if i >= 416 and j == 3 and self._is_fluid_cell(i, j):
                pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                velocity = self._physical_velocity_at(i, j)
                density = mixture_density(
                    self.water_phase[i, j],
                    self._water_density_lattice,
                    self._air_density_lattice,
                )
                for q in range(9):
                    if populations != 0:
                        self.momentum_populations[i, j, q] = momentum_equilibrium(
                            q, pressure, density, velocity
                        )
                    else:
                        self.momentum_populations[i, j, q] += momentum_equilibrium(
                            q, 0.0, density, velocity
                        ) - momentum_equilibrium(q, pressure, density, velocity)
                if populations == 0:
                    self.pressure_lattice[i, j] = self.hydrostatic_reference_pressure[
                        i, j
                    ]


def record(s, attempted_step, stage):
    summary = state_summary(s)
    s.probe_points_host[-1] = summary["peak_cell"]
    s.probe_points.from_numpy(s.probe_points_host)
    points = s.point_rows()
    fields = {
        name: getattr(s, name).to_numpy()
        for name in (
            "phase_gradient_lattice",
            "source_gradient_lattice",
            "phase_gradient_status",
            "phase_donor_mask",
            "phase_reconstruction_coefficients",
            "phase_mass_flux",
            "previous_solid_mask",
            "phase_wall_geometry",
            "phase_stream_buffer",
            "momentum_stream_buffer",
            "refill_state",
        )
    }
    for point in points:
        cell = tuple(point["cell"])
        for name, field in fields.items():
            point[name] = field[cell].tolist()
        point["donor_count"] = int(point["phase_donor_mask"]).bit_count()
    gap = {}
    for name in (
        "water_phase",
        "pressure_lattice",
        "solid_mask",
        "previous_solid_mask",
    ):
        field = getattr(s, name).to_numpy()
        gap[name] = field[410:448, 3].tolist()
    gap["velocity"] = s.physical_velocity_lattice.to_numpy()[410:448, 3].tolist()
    return dict(
        attempted_step=attempted_step,
        stage=stage,
        state=summary,
        points=points,
        gap_line_x_start=410,
        gap_line=gap,
    )


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=21600)
    parser.add_argument("--end", type=int, default=22660)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[22000, 22550])
    parser.add_argument("--control-steps", type=int, default=600)
    parser.add_argument("--detail-start", type=int, default=22590)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "fixed_pose",
            "stationary_pose",
            "no_thermal",
            "zero_boundary_source",
            "zero_all_source",
            "no_flux_force",
            "positive_phase_populations",
            "postcollision_bounceback",
            "halfway_links",
            "reset_gap_pressure",
            "reset_gap_populations",
            "zero_gap_normal_wall_velocity",
            "tangent_gradient",
            "tangent_source_only",
            "tangent_phase_only",
            "quadratic_gap",
        ],
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s = SingleProbe(create_config())
    s._thermal_stability_check_enabled = True
    started = time.monotonic()
    s.step(args.start)
    print(
        json.dumps(
            dict(
                event="ready",
                elapsed_s=time.monotonic() - started,
                state=state_summary(s),
            )
        ),
        flush=True,
    )
    checkpoints, trace, failure = {}, [], None
    for step in range(args.start + 1, args.end + 1):
        if s.steps in args.checkpoints:
            checkpoints[s.steps] = (device_snapshot(s), host_snapshot(s))
        detailed = step >= args.detail_start or step % 20 == 0

        def capture(stage):
            if detailed:
                trace.append(record(s, step, stage))

        try:
            capture("before")
            s._collide_momentum()
            capture("collision")
            s._stream()
            capture("stream")
            s._update_streamed_macroscopic_fields()
            capture("macros")
            s._bound_water_phase()
            capture("bounded_phase")
            s._prepare_phase_gradients()
            s._update_pressure()
            capture("pressure")
            s._advance_body()
            capture("body")
            s._advance_thermal()
            s.steps += 1
            capture("thermal")
        except RuntimeError as exc:
            failure = dict(step=step, error=str(exc), state=state_summary(s))
            break
        if step % 120 == 0:
            print(json.dumps(dict(event="trace", state=state_summary(s))), flush=True)
    save(out / "trace.json", trace)
    save(out / "failure.json", failure)
    print(json.dumps(dict(event="trace_end", failure=failure)), flush=True)
    results = []
    for checkpoint, snapshot in checkpoints.items():
        for mode in args.modes:
            restore(s, snapshot)
            s.control_mode[None] = {
                "zero_boundary_source": 1,
                "no_flux_force": 2,
                "zero_all_source": 3,
                "positive_phase_populations": 4,
                "postcollision_bounceback": 5,
                "halfway_links": 6,
                "tangent_gradient": 7,
                "quadratic_gap": 8,
                "tangent_source_only": 9,
                "tangent_phase_only": 10,
            }.get(mode, 0)
            if mode in ("reset_gap_pressure", "reset_gap_populations"):
                s.reset_gap(int(mode == "reset_gap_populations"))
            if mode == "stationary_pose":
                s.body_velocity.fill(0)
                s.body_angular_velocity.fill(0)
            history, error = [], None
            for k in range(args.control_steps):
                try:
                    old_velocity = None
                    if mode == "zero_gap_normal_wall_velocity":
                        old_velocity = s.body_velocity.field.to_numpy()
                        velocity = old_velocity.copy()
                        velocity[:, 1] = 0
                        s.body_velocity.field.from_numpy(velocity)
                    s._advance_fluid()
                    if old_velocity is not None:
                        s.body_velocity.field.from_numpy(old_velocity)
                    if mode not in ("fixed_pose", "stationary_pose"):
                        s._advance_body()
                    if mode not in ("fixed_pose", "stationary_pose", "no_thermal"):
                        s._advance_thermal()
                    s.steps += 1
                    if k % 10 == 0 or k == args.control_steps - 1:
                        state = state_summary(s)
                        history.append(state)
                        if state["speed_l1_max"] > 1 or not np.isfinite(
                            state["speed_l1_max"]
                        ):
                            break
                except RuntimeError as exc:
                    error = str(exc)
                    break
            result = dict(
                checkpoint=checkpoint,
                mode=mode,
                completed_steps=s.steps - checkpoint,
                error=error,
                history=history,
                last=state_summary(s),
            )
            results.append(result)
            save(out / "matched_controls.json", results)
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "history"}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
