"""Causal controls at the exact newly opened cell, without production edits.

One-off phase smoothing preserves the sum of phi in a local connected row.
It is a diagnostic intervention, not a production remapping implementation.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.diagnose_data_priority_four import visual_state
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    PointProbe,
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config
from iceflow2d_v3.lattice import (
    mixture_density,
    momentum_equilibrium,
    phase_equilibrium,
)


@ti.data_oriented
class OpeningProbe(PointProbe):
    @ti.kernel
    def patch_row(
        self,
        phase: ti.types.ndarray(),
        phase_mode: ti.i32,
        reset_pressure: ti.i32,
        reinitialize_g: ti.i32,
    ):
        for i in range(148, 162):
            j = 3
            if self._is_fluid_cell(i, j):
                old_phi = self.water_phase[i, j]
                new_phi = old_phi
                if phase_mode == 1:
                    new_phi = phase[i - 148]
                old_rho = mixture_density(
                    old_phi, self._water_density_lattice, self._air_density_lattice
                )
                new_rho = mixture_density(
                    new_phi, self._water_density_lattice, self._air_density_lattice
                )
                old_pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                new_pressure = old_pressure
                if reset_pressure != 0:
                    new_pressure = 0.0
                velocity = (
                    self.momentum_velocity_lattice[i, j]
                    + 0.5 * self.fluid_acceleration_lattice[i, j]
                )
                for q in range(9):
                    if phase_mode != 0:
                        self.phase_populations[i, j, q] = phase_equilibrium(
                            q, new_phi, velocity
                        )
                    new_g = momentum_equilibrium(q, new_pressure, new_rho, velocity)
                    if reinitialize_g != 0:
                        self.momentum_populations[i, j, q] = new_g
                    else:
                        self.momentum_populations[i, j, q] += (
                            new_g
                            - momentum_equilibrium(q, old_pressure, old_rho, velocity)
                        )
                self.water_phase[i, j] = new_phi
                self.pressure_lattice[i, j] = (
                    new_pressure + self.hydrostatic_reference_pressure[i, j]
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--modes", nargs="+", default=None)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s = OpeningProbe(create_config())
    s._thermal_stability_check_enabled = True
    started = time.monotonic()
    s.step(8520)
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
    snapshot = (device_snapshot(s), host_snapshot(s))
    original = s.water_phase.to_numpy()[148:162, 3].astype(float)
    solid = s.solid_mask.to_numpy()[148:162, 3]
    assert not solid.any()
    x = np.arange(148, 162, dtype=float)
    lo, hi = 140.0, 170.0
    wet = float(original.max())
    dry = float(original.min())
    for _ in range(100):
        center = (lo + hi) / 2
        values = dry + (wet - dry) * 0.5 * (1 - np.tanh(2 * (x - center) / 5))
        if values.sum() > original.sum():
            hi = center
        else:
            lo = center
    smooth = (
        dry + (wet - dry) * 0.5 * (1 - np.tanh(2 * (x - (lo + hi) / 2) / 5))
    ).astype(np.float32)
    patch = dict(
        i_range=[148, 161],
        j=3,
        old_phi=original.tolist(),
        smoothed_phi=smooth.tolist(),
        old_sum=float(original.sum()),
        new_sum=float(smooth.sum(dtype=float)),
        interface_width=5.0,
    )
    (out / "smoothing_intervention.json").write_text(json.dumps(patch, indent=2) + "\n")
    results = []
    for mode in args.modes or (
        "baseline",
        "fixed_pose",
        "stationary_pose",
        "phase_reinit_only",
        "pressure_reset",
        "momentum_reinit_only",
        "smooth_phi",
        "smooth_and_pressure",
        "smooth_and_momentum_reinit",
    ):
        restore(s, snapshot)
        if mode == "stationary_pose":
            s.body_velocity.fill(0)
            s.body_angular_velocity.fill(0)
        smooth_mode = mode.startswith("smooth")
        phase_mode = 1 if smooth_mode else 2 if mode == "phase_reinit_only" else 0
        pressure_mode = int(mode in ("pressure_reset", "smooth_and_pressure"))
        reinit_g = int(mode in ("momentum_reinit_only", "smooth_and_momentum_reinit"))
        if phase_mode or pressure_mode or reinit_g:
            s.patch_row(smooth, phase_mode, pressure_mode, reinit_g)
            s._update_streamed_macroscopic_fields()
            s._prepare_phase_gradients()
        history = []
        error = None
        first_bad = None
        for k in range(args.steps):
            try:
                step = s.steps + 1
                s._collide_momentum()
                s._stream()
                s._update_streamed_macroscopic_fields()
                before = state_summary(s)
                if before["speed_l1_max"] > 1 or not np.isfinite(
                    before["speed_l1_max"]
                ):
                    first_bad = dict(
                        attempted_step=step, stage="macros_before_body", state=before
                    )
                    break
                s._bound_water_phase()
                s._prepare_phase_gradients()
                s._update_pressure()
                if mode not in ("fixed_pose", "stationary_pose"):
                    s._advance_body()
                if mode not in ("fixed_pose", "stationary_pose"):
                    s._advance_thermal()
                s.steps += 1
                if k < 5 or k % 20 == 0 or k == args.steps - 1:
                    history.append(state_summary(s))
            except RuntimeError as exc:
                error = str(exc)
                break
        result = dict(
            mode=mode,
            checkpoint=8520,
            completed_steps=s.steps - 8520,
            error=error,
            first_bad=first_bad,
            last=state_summary(s),
            history=history,
        )
        results.append(result)
        (out / "controls.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True
        )
        if smooth_mode and s.steps > 8600:
            visual_state(s, s.steps, mode, out)


if __name__ == "__main__":
    main()
