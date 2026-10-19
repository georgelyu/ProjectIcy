"""Separate refill discontinuities and gradient closure at one opening event."""

import argparse
import json
from pathlib import Path

import taichi as ti

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
    lattice_weight,
    mixture_density,
    momentum_equilibrium,
    phase_equilibrium,
)
from iceflow2d_v3.simulator import ensure_taichi_cuda


@ti.data_oriented
class GapControls(PointProbe):
    def __init__(self, config):
        ensure_taichi_cuda()
        self.gap_mode = ti.field(ti.i32, shape=())
        super().__init__(config)

    @ti.kernel
    def _tangent_gradient(self):
        for i, j in self.water_phase:
            if (
                self.gap_mode[None] == 1
                and i == 176
                and j > 3
                and j < 33
                and self._is_fluid_cell(i, j)
            ):
                phi = self.water_phase[i, j]
                before, after = phi, phi
                if self._is_fluid_cell(i, j - 1):
                    before = self.water_phase[i, j - 1]
                if self._is_fluid_cell(i, j + 1):
                    after = self.water_phase[i, j + 1]
                gradient = ti.Vector([0.0, 0.5 * (after - before)])
                self.phase_gradient_lattice[i, j] = gradient
                self.source_gradient_lattice[i, j] = gradient

    def _prepare_phase_gradients(self):
        super()._prepare_phase_gradients()
        self._tangent_gradient()

    @ti.kernel
    def patch_pressure(self):
        for i, j in self.water_phase:
            if (
                i == 176 and j >= 3 and j <= 33 or i == 175 and j == 3
            ) and self._is_fluid_cell(i, j):
                old_pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                for q in range(9):
                    factor = 3 * lattice_weight(q)
                    if q == 0:
                        factor -= 3
                    self.momentum_populations[i, j, q] -= factor * old_pressure
                self.pressure_lattice[i, j] = self.hydrostatic_reference_pressure[i, j]

    @ti.kernel
    def patch_phase(self):
        for j in range(4, 10):
            i = 176
            old_phi = self.water_phase[i, j]
            lo = self.water_phase[i, 3]
            hi = self.water_phase[i, 10]
            # Bounded smooth interpolation across the six newly exposed nodes.
            a = 0.5 * (1.0 - ti.tanh(2.0 * (j - 6.5) / 5.0))
            a0 = 0.5 * (1.0 - ti.tanh(2.0 * (3 - 6.5) / 5.0))
            a1 = 0.5 * (1.0 - ti.tanh(2.0 * (10 - 6.5) / 5.0))
            phi = hi + (lo - hi) * (a - a1) / (a0 - a1)
            pressure = (
                self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
            )
            velocity = self._physical_velocity_at(i, j)
            old_rho = mixture_density(
                old_phi, self._water_density_lattice, self._air_density_lattice
            )
            rho = mixture_density(
                phi, self._water_density_lattice, self._air_density_lattice
            )
            for q in range(9):
                self.momentum_populations[i, j, q] += momentum_equilibrium(
                    q, pressure, rho, velocity
                ) - momentum_equilibrium(q, pressure, old_rho, velocity)
                self.phase_populations[i, j, q] = phase_equilibrium(q, phi, velocity)
            self.water_phase[i, j] = phi

    @ti.kernel
    def _positive_phase_collision(self):
        for i, j in self.water_phase:
            if self.gap_mode[None] == 2 and self._is_fluid_cell(i, j):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    s = GapControls(create_config())
    s._thermal_stability_check_enabled = True
    s.step(8276)
    snapshot = (device_snapshot(s), host_snapshot(s))
    results = []
    for mode in (
        "baseline",
        "pressure_reset",
        "phase_smooth",
        "both",
        "tangent_gradient",
        "smooth_and_tangent",
        "positive_phase_populations",
        "pressure_and_positive_phase",
    ):
        restore(s, snapshot)
        s.gap_mode[None] = (
            1
            if mode in ("tangent_gradient", "smooth_and_tangent")
            else 2
            if "positive_phase" in mode
            else 0
        )
        if mode in ("pressure_reset", "both", "pressure_and_positive_phase"):
            s.patch_pressure()
        if mode in ("phase_smooth", "both", "smooth_and_tangent"):
            s.patch_phase()
        history = []
        result = dict(mode=mode, checkpoint=8276, history=history)
        try:
            for k in range(600):
                s.step()
                if k < 20 or k % 10 == 0 or k == 599:
                    state = state_summary(s)
                    history.append(state)
                    if state["speed_l1_max"] > 1:
                        break
        except RuntimeError as exc:
            result["error"] = str(exc)
        result["completed_steps"] = s.steps - 8276
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "history"}
                | dict(last=state_summary(s))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
