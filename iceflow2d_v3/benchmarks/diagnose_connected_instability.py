"""Matched-state diagnostics of connected_ls; save scalar/point records only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    BoundaryExperiment,
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.body_state import BodyField
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config
from iceflow2d_v3.lattice import inside_grid
from iceflow2d_v3.simulator import ensure_taichi_cuda


@ti.data_oriented
class ConnectedProbe(BoundaryExperiment):
    def __init__(self, config):
        ensure_taichi_cuda()
        self.control_mode = ti.field(ti.i32, shape=())
        super().__init__(config)
        self.probe_points_host = np.array(
            [
                (176, 11),
                (175, 11),
                (177, 11),
                (176, 10),
                (176, 12),
                (175, 10),
                (175, 12),
                (177, 10),
                (177, 12),
                (175, 16),
                (176, 8),
            ],
            dtype=np.int32,
        )
        self.probe_points.from_numpy(self.probe_points_host)

    @ti.kernel
    def _control_gradients(self):
        for i, j in self.water_phase:
            if self.control_mode[None] == 2 or (
                self.control_mode[None] == 1 and self.phase_gradient_status[i, j] != 1
            ):
                self.source_gradient_lattice[i, j] = ti.Vector([0.0, 0.0])

    def _prepare_phase_gradients(self):
        super()._prepare_phase_gradients()
        self._control_gradients()

    @ti.func
    def _phase_flux_force_density(self, i, j):
        force = ti.Vector.zero(ti.f32, 2)
        if self.control_mode[None] != 3:
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


def record(s, step, stage):
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
        )
    }
    for point in points:
        index = tuple(point["cell"])
        for name, field in fields.items():
            point[name] = field[index].tolist()
        point["donor_count"] = int(point["phase_donor_mask"]).bit_count()
    return dict(attempted_step=step, stage=stage, state=summary, points=points)


def host_snapshot(s):
    return dict(
        steps=s.steps,
        pending=s._thermal_lbm_steps_pending,
        thermal={
            k: v
            for k, v in vars(s.thermal).items()
            if isinstance(v, (int, float, bool))
        },
    )


def restore(s, snapshot):
    fields, host = snapshot
    for prefix, name, values in fields:
        owner = s if prefix == "flow" else s.thermal
        field = getattr(owner, name)
        if isinstance(field, BodyField):
            field = field.field
        field.from_numpy(values)
    s.steps, s._thermal_lbm_steps_pending = host["steps"], host["pending"]
    for name, value in host["thermal"].items():
        setattr(s.thermal, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=8000)
    parser.add_argument("--end", type=int, default=8290)
    parser.add_argument("--control-steps", type=int, default=400)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s = ConnectedProbe(create_config())
    s._thermal_stability_check_enabled = True
    start = time.monotonic()
    s.step(args.start)
    print(
        json.dumps(
            dict(
                event="ready", elapsed=time.monotonic() - start, state=state_summary(s)
            )
        ),
        flush=True,
    )
    snapshots = {args.start: (device_snapshot(s), host_snapshot(s))}
    trace = []
    failure = None
    for step in range(args.start + 1, args.end + 1):
        if s.steps in (8200, 8240, 8260, 8270, 8275, 8280):
            snapshots[s.steps] = (device_snapshot(s), host_snapshot(s))
        detailed = step >= 8230 or step % 10 == 0

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
            failure = dict(step=step, error=str(exc))
            break
        if step % 20 == 0:
            print(json.dumps(dict(event="trace", state=state_summary(s))), flush=True)
            (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out / "trace_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
    print(json.dumps(dict(event="trace_end", failure=failure)), flush=True)
    results = []
    for checkpoint in (8200, 8270):
        if checkpoint not in snapshots:
            continue
        for mode in (
            "baseline",
            "zero_boundary_source",
            "zero_all_source",
            "no_flux_force",
            "postcollision_bounceback",
            "fixed_pose",
            "stationary_pose",
            "no_thermal",
        ):
            restore(s, snapshots[checkpoint])
            s.control_mode[None] = {
                "zero_boundary_source": 1,
                "zero_all_source": 2,
                "no_flux_force": 3,
            }.get(mode, 0)
            s.boundary_experiment_mode[None] = int(mode == "postcollision_bounceback")
            if mode == "stationary_pose":
                s.body_velocity.fill(0)
                s.body_angular_velocity.fill(0)
            history = []
            result = dict(
                checkpoint=checkpoint, mode=mode, completed_steps=0, history=history
            )
            for k in range(args.control_steps):
                try:
                    s._advance_fluid()
                    if mode not in ("fixed_pose", "stationary_pose"):
                        s._advance_body()
                    if mode not in ("fixed_pose", "stationary_pose", "no_thermal"):
                        s._advance_thermal()
                    s.steps += 1
                    result["completed_steps"] += 1
                    if k % 5 == 0 or k == args.control_steps - 1:
                        state = state_summary(s)
                        history.append(state)
                        if state["speed_l1_max"] > 1 or not np.isfinite(
                            state["speed_l1_max"]
                        ):
                            result["stopped_at_speed_above_one"] = True
                            break
                except RuntimeError as exc:
                    result["error"] = str(exc)
                    break
            results.append(result)
            (out / "matched_controls.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "history"}
                    | dict(last_state=state_summary(s))
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
