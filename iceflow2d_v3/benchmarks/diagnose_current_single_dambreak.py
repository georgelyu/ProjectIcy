"""Replay a recorded single-ice failure, with scalar-only causal diagnostics.

The failure JSON supplies the physical configuration; current example defaults
are deliberately not used to infer the failed geometry. No solver is modified.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    PointProbe,
    device_snapshot,
)
from iceflow2d_v3.examples import dambreak_single_ice_melting_2d as example
from iceflow2d_v3.lattice import (
    cross_product_2d,
    inside_grid,
    lattice_direction,
    lattice_weight,
    mixture_density,
)
from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.stability_options import StabilityOptions


@ti.data_oriented
class Probe(PointProbe):
    def __init__(self, config):
        ensure_taichi_cuda()
        self.diagnostic_mode = ti.field(ti.i32, shape=())
        super().__init__(config)

    @ti.func
    def _phase_flux_force_density(self, i, j):
        force = ti.Vector.zero(ti.f32, 2)
        if self.diagnostic_mode[None] != 1:
            force = StabilityOptions._phase_flux_force_density(self, i, j)
        return force

    @ti.func
    def _fluid_force_and_gradient(self, i, j):
        from_base = IceFlow2D._fluid_force_and_gradient(self, i, j)
        if self.diagnostic_mode[None] == 3 and self.water_phase[i, j] < 0.01:
            from_base[2], from_base[3] = 0.0, 0.0
        return from_base

    @ti.func
    def _cut_link_fraction(self, i, j, ni, nj):
        fraction = IceFlow2D._cut_link_fraction(self, i, j, ni, nj)
        if self.diagnostic_mode[None] == 4:
            fraction = 0.5
        return fraction

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        prepared = IceFlow2D._prepare_momentum_cut_links(self, i, j, raw_post)
        if self.diagnostic_mode[None] == 5:
            rho = mixture_density(
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
                    ) * rho * direction.dot(ub)
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
    def _limit_phase_for_diagnosis(self):
        for i, j in self.water_phase:
            if self.diagnostic_mode[None] == 2 and self._is_fluid_cell(i, j):
                phi = self.water_phase[i, j]
                alpha = 1.0
                for q in range(9):
                    low = lattice_weight(q) * phi
                    high = self.phase_stream_buffer[i, j, q]
                    if high < 0.0:
                        alpha = ti.min(alpha, low / (low - high + 1e-30))
                for q in range(9):
                    low = lattice_weight(q) * phi
                    self.phase_stream_buffer[i, j, q] = low + alpha * (
                        self.phase_stream_buffer[i, j, q] - low
                    )

    def _collide_phase(self, *, gradients_prepared=False):
        super()._collide_phase(gradients_prepared=gradients_prepared)
        self._limit_phase_for_diagnosis()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summary(s, detail=False):
    u = s.physical_velocity_lattice.to_numpy()
    phi = s.water_phase.to_numpy()
    solid = s.solid_mask.to_numpy()
    active = (s.wall_mask.to_numpy() == 0) & (solid == 0)
    speed = np.abs(u).sum(axis=-1)
    peak = np.unravel_index(np.argmax(np.where(active, speed, -np.inf)), phi.shape)
    air, water = active & (phi < 0.5), active & (phi >= 0.5)
    result = dict(
        step=s.steps,
        time_s=s.physical_time_s,
        max_l1=float(speed[peak]),
        max_l2=float(np.linalg.norm(u[active], axis=-1).max()),
        peak=[int(x) for x in peak],
        peak_phi=float(phi[peak]),
        air_max=float(speed[air].max()),
        water_max=float(speed[water].max()),
        phi_min=float(phi[active].min()),
        phi_max=float(phi[active].max()),
        water_amount=float(phi[active].sum(dtype=np.float64)),
        center=s.body_center.to_numpy().tolist(),
        angle=s.body_angle.to_numpy().tolist(),
    )
    if detail:
        i, j = peak
        s.probe_points_host[:] = [(i, j)] + [
            (int(np.clip(i + di, 0, s.nx - 1)), int(np.clip(j + dj, 0, s.ny - 1)))
            for di, dj in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
                (0, -2),
                (0, 2),
            )
        ]
        s.probe_points.from_numpy(s.probe_points_host)
        result["points"] = s.point_rows()
        for name in (
            "phase_gradient_lattice",
            "phase_gradient_status",
            "phase_donor_mask",
            "phase_mass_flux",
            "previous_solid_mask",
        ):
            field = getattr(s, name).to_numpy()
            for point in result["points"]:
                point[name] = field[tuple(point["cell"])].tolist()
        result["solid_neighborhood"] = solid[
            max(0, i - 3) : i + 4, max(0, j - 3) : j + 4
        ].tolist()
        result["phi_neighborhood"] = phi[
            max(0, i - 3) : i + 4, max(0, j - 3) : j + 4
        ].tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure",
        type=Path,
        default=Path(
            "iceflow2d_v3/outputs/dambreak_single_ice_melting_2d/run_failure.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end", type=int, default=16800)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[15000, 16000, 16400]
    )
    parser.add_argument("--control-steps", type=int, default=2400)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--interface-width", type=float)
    parser.add_argument("--width", type=float)
    parser.add_argument("--reference-velocity", type=float)
    parser.add_argument("--collision", choices=["cm_mrt", "cm_hrr", "rm_mrt"])
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--detail-start", type=int, default=16500)
    parser.add_argument("--detail-end", type=int, default=16650)
    parser.add_argument(
        "--control-modes",
        nargs="+",
        default=[
            "baseline",
            "no_flux",
            "positive_phase",
            "zero_air_source",
            "frozen_pose",
            "stationary_pose",
            "no_thermal",
        ],
    )
    args = parser.parse_args()
    if not args.checkpoints or min(args.checkpoints) < 1:
        parser.error("checkpoints must contain positive step numbers")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(args.failure.read_text())["configuration"]
    width = (
        int(recorded["resolution"][0] * recorded["water_width_fraction"])
        - recorded["boundary_cells"]
    ) * recorded["dx"]
    center = recorded["ice_bodies"][0]["center_cells"][0] * recorded["dx"]
    config = example.create_config(
        example.parse_args(
            [
                "--water-column-width-m",
                str(args.width or width),
                "--ice-centers-x-m",
                str(center),
                "--no-plot",
                "--output-dir",
                str(out),
            ]
        )
    )
    overrides = {
        k: v
        for k, v in {
            "phase_normal_epsilon": args.epsilon,
            "interface_width": args.interface_width,
            "reference_velocity": args.reference_velocity,
            "momentum_collision_model": args.collision,
        }.items()
        if v is not None
    }
    config = replace(config, **overrides)
    actual = json.loads(json.dumps(config.to_dict()))
    diffs = {
        k: [recorded[k], v]
        for k, v in actual.items()
        if k != "output_dir" and recorded[k] != v
    }
    sources = [
        Path("iceflow2d_v3") / name
        for name in (
            "simulator.py",
            "phase_reconstruction.py",
            "stability_options.py",
            "config.py",
            "examples/dambreak_four_ice_melting_2d.py",
        )
    ]
    save(
        out / "manifest.json",
        dict(
            configuration=config.to_dict(),
            differences_from_recorded_failure=diffs,
            source_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
        ),
    )
    s = Probe(config)
    s._thermal_stability_check_enabled = True
    checkpoints, trace, failure = {}, [], None
    started = time.monotonic()
    initial = summary(s)
    trace.append(dict(stage="initial", **initial))
    try:
        for step in range(1, args.end + 1):
            if (
                step < min(args.checkpoints)
                or not args.detail_start <= step <= args.detail_end
            ):
                s.step()
            else:
                detailed = args.detail_start <= step <= args.detail_end
                stages = [
                    ("collision", s._collide_momentum),
                    ("stream", s._stream),
                    ("macros", s._update_streamed_macroscopic_fields),
                    ("bounded", s._bound_water_phase),
                    ("gradient", s._prepare_phase_gradients),
                    ("pressure", s._update_pressure),
                    ("body", s._advance_body),
                    ("thermal", s._advance_thermal),
                ]
                for stage, operation in stages:
                    operation()
                    if detailed and stage not in ("stream", "gradient"):
                        trace.append(
                            dict(attempted_step=step, stage=stage, **summary(s, True))
                        )
                s.steps += 1
            if not args.no_controls and s.steps in args.checkpoints:
                checkpoints[s.steps] = (device_snapshot(s), host_snapshot(s))
            if s.steps % 100 == 0:
                row = summary(s, s.steps >= 16000)
                trace.append(dict(stage="completed", **row))
                if s.steps % 1200 == 0 or s.steps >= 16000:
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k != "points" and not k.endswith("neighborhood")
                            }
                        ),
                        flush=True,
                    )
    except RuntimeError as exc:
        failure = dict(
            attempted_step=s.steps + 1, error=str(exc), state=summary(s, True)
        )
    save(out / "trace.json", trace)
    save(out / "failure.json", failure)
    result = dict(
        status="failed" if failure else "completed",
        initial=initial,
        last=summary(s),
        failure=failure,
        elapsed_s=time.monotonic() - started,
    )
    save(out / "result.json", result)
    print(
        json.dumps(
            dict(
                event="baseline_done",
                status=result["status"],
                last=result["last"],
                error=None if failure is None else failure["error"],
                elapsed_s=result["elapsed_s"],
            )
        ),
        flush=True,
    )
    if args.no_controls:
        return
    controls = []
    for checkpoint, snapshot in checkpoints.items():
        for mode in args.control_modes:
            restore(s, snapshot)
            s.diagnostic_mode[None] = {
                "no_flux": 1,
                "positive_phase": 2,
                "zero_air_source": 3,
                "halfway_links": 4,
                "postcollision_bounceback": 5,
            }.get(mode, 0)
            if mode == "stationary_pose":
                s.body_velocity.fill(0.0)
                s.body_angular_velocity.fill(0.0)
            history, error = [], None
            for k in range(args.control_steps):
                try:
                    s._advance_fluid()
                    if mode not in ("frozen_pose", "stationary_pose"):
                        s._advance_body()
                    if mode not in ("no_thermal", "frozen_pose", "stationary_pose"):
                        s._advance_thermal()
                    s.steps += 1
                    if k % 20 == 0 or k + 1 == args.control_steps:
                        row = summary(s)
                        history.append(row)
                        if row["max_l1"] > 1 or not np.isfinite(row["max_l1"]):
                            break
                except RuntimeError as exc:
                    error = str(exc)
                    break
            result = dict(
                checkpoint=checkpoint,
                mode=mode,
                steps=s.steps - checkpoint,
                error=error,
                history=history,
                last=summary(s, True),
            )
            controls.append(result)
            save(out / "controls.json", controls)
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("history", "last")}
                    | {
                        "last": {
                            k: v
                            for k, v in result["last"].items()
                            if k != "points" and not k.endswith("neighborhood")
                        }
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
