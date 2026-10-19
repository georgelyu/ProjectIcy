"""Test zero-valued solid/wall samples as a replacement gradient operator.

Production kernels/configuration are unchanged. Run both Cartesian central
differences and the original bulk D2Q9 formulas with zero nonfluid samples.
Compare fresh runs and identical pre-instability checkpoints in three cases.
For legacy phase reconstruction, zero_cartesian changes the density/source
gradient to Cartesian differences while retaining its existing D2Q9 phase
compression formula. zero_d2q9 retains both original bulk formulas throughout.
Only scalar and point records are written; snapshots remain in RAM.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as four
from iceflow2d_v3.examples import dambreak_single_ice_melting_2d as single
from iceflow2d_v3.lattice import inside_grid, lattice_direction, lattice_weight
from iceflow2d_v3.simulator import ensure_taichi_cuda


MODES = {"production": 0, "zero_cartesian": 1, "zero_d2q9": 2}


@ti.data_oriented
class GradientTrial(IceFlow2D):
    def __init__(self, config, mode=0):
        ensure_taichi_cuda()
        self.gradient_trial_mode = ti.field(ti.i32, shape=())
        self.gradient_trial_mode[None] = mode
        super().__init__(config)

    @ti.func
    def _zero_sample(self, i, j):
        value = 0.0
        if inside_grid(i, j, self.nx, self.ny) and self._is_fluid_cell(i, j):
            value = self.water_phase[i, j]
        return value

    @ti.func
    def _phase_neighbor(self, i, j, sx, sy, distance):
        value = IceFlow2D._phase_neighbor(self, i, j, sx, sy, distance)
        if self.gradient_trial_mode[None] != 0:
            value = self._zero_sample(i + distance * sx, j + distance * sy)
        return value

    @ti.func
    def _replacement_gradients(self, i, j):
        phase, source = ti.Vector.zero(ti.f32, 2), ti.Vector.zero(ti.f32, 2)
        if self.gradient_trial_mode[None] == 1:
            source = ti.Vector(
                [
                    0.5 * (self._zero_sample(i + 1, j) - self._zero_sample(i - 1, j)),
                    0.5 * (self._zero_sample(i, j + 1) - self._zero_sample(i, j - 1)),
                ]
            )
            phase = source
        else:
            phi = self.water_phase[i, j]
            for q in range(1, 9):
                c = lattice_direction(q)
                first = self._zero_sample(i + c.x, j + c.y)
                second = self._zero_sample(i + 2 * c.x, j + 2 * c.y)
                vector = lattice_weight(q) * ti.cast(c, ti.f32)
                phase += vector * (
                    2 * (first - phi) + 0.5 * (4 * first - second - 3 * phi)
                )
                source += 3 * vector * (first - phi)
        return ti.Vector([phase.x, phase.y, source.x, source.y])

    @ti.kernel
    def _replace_gradients(self):
        for i, j in self.water_phase:
            if self.gradient_trial_mode[None] != 0:
                value = ti.Vector.zero(ti.f32, 4)
                if self._is_fluid_cell(i, j):
                    value = self._replacement_gradients(i, j)
                self.phase_gradient_lattice[i, j] = ti.Vector([value[0], value[1]])
                self.source_gradient_lattice[i, j] = ti.Vector([value[2], value[3]])

    def _prepare_phase_gradients(self):
        IceFlow2D._prepare_phase_gradients(self)
        if self.config.phase_gradient_scheme == "connected_ls":
            self._replace_gradients()

    @ti.func
    def _phase_central_gradient(self, i, j):
        gradient = IceFlow2D._phase_central_gradient(self, i, j)
        if self.gradient_trial_mode[None] != 0:
            value = self._replacement_gradients(i, j)
            gradient = ti.Vector([value[2], value[3]])
        return gradient


def run(s, steps, out, label):
    start = s.steps
    history = [state_summary(s)]
    error = None
    began = time.monotonic()
    try:
        while s.steps - start < steps:
            s.step(min(120, steps - (s.steps - start)))
            history.append(state_summary(s))
    except RuntimeError as exc:
        error = str(exc)
        history.append(state_summary(s))
    result = dict(
        label=label,
        start_step=start,
        requested_steps=steps,
        completed=s.steps - start == steps,
        completed_steps=s.steps - start,
        error=error,
        last=history[-1],
        history=history,
        peak_sampled_speed_l1=max(r["speed_l1_max"] for r in history),
        elapsed_s=time.monotonic() - began,
    )
    (out / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "history"}),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/zero_solid_phase_gradient_controls"),
    )
    parser.add_argument("--end-time-s", type=float, default=0.25)
    parser.add_argument("--replay-steps", type=int, default=2400)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["single_connected", "four_connected", "four_legacy"],
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cases = {
        "single_connected": (single.create_config(), 22627),
        "four_connected": (four.create_config(), 8276),
        "four_legacy": (
            replace(four.create_config(), phase_gradient_scheme="legacy"),
            8544,
        ),
    }
    results = []
    for case in args.cases:
        config, checkpoint = cases[case]
        print(
            json.dumps(
                dict(event="checkpoint_begin", case=case, checkpoint=checkpoint)
            ),
            flush=True,
        )
        source = IceFlow2D(config)
        source._thermal_stability_check_enabled = True
        source.step(checkpoint)
        snapshot = (device_snapshot(source), host_snapshot(source))
        target = GradientTrial(config)
        target._thermal_stability_check_enabled = True
        for mode in MODES:
            restore(target, snapshot)
            target.gradient_trial_mode[None] = MODES[mode]
            result = run(target, args.replay_steps, out, f"{case}_replay_{mode}")
            result.update(case=case, mode=mode, experiment="matched_checkpoint")
            results.append(result)
            (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        if case == "single_connected":
            target.water_phase.fill(1.0)
            target._prepare_phase_geometry()
            sanity = []
            for mode in MODES:
                target.gradient_trial_mode[None] = MODES[mode]
                target._prepare_phase_gradients()
                sanity.append(
                    dict(
                        mode=mode,
                        cell=[40, 3],
                        phi=1.0,
                        phase_gradient=target.phase_gradient_lattice[40, 3]
                        .to_numpy()
                        .tolist(),
                        source_gradient=target.source_gradient_lattice[40, 3]
                        .to_numpy()
                        .tolist(),
                    )
                )
            (out / "uniform_water_wall_check.json").write_text(
                json.dumps(sanity, indent=2) + "\n"
            )
            print(
                json.dumps(dict(event="uniform_wall_check", results=sanity)), flush=True
            )
        for mode in ("zero_cartesian", "zero_d2q9"):
            trial = GradientTrial(config, mode=MODES[mode])
            trial._thermal_stability_check_enabled = True
            steps = round(args.end_time_s / trial.scales.dt_s)
            result = run(trial, steps, out, f"{case}_from_initial_{mode}")
            result.update(
                case=case,
                mode=mode,
                experiment="from_initial",
                requested_time_s=args.end_time_s,
                configuration=config.to_dict(),
            )
            results.append(result)
            (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
