"""Periodic planar-layer translation: phase/momentum consistency diagnostic."""

# ruff: noqa: E741 -- Taichi grouped grid indices.
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig  # noqa: E402
from iceflow3d.lattice import direction, weight, Q, phase_equilibrium, density  # noqa: E402


@ti.data_oriented
class PeriodicLayer(IceFlow3D):
    @ti.func
    def _wrapped(self, I):
        shape = ti.Vector([self.nx, self.ny, self.nz])
        return (I + shape) % shape

    @ti.func
    def _stress_neighbor_velocity(self, I, c):
        return self.velocity_lattice[self._wrapped(I + c)]

    @ti.func
    def _hrr_cubic_correction(self, I):
        derivative = ti.Vector.zero(ti.f32, 3)
        for q in range(Q):
            c = direction(q)
            J = self._wrapped(I + c)
            derivative += (
                3
                * weight(q)
                * ti.cast(c, ti.f32)
                * (
                    density(self.water_phase[J], self.rho_air)
                    * self.velocity_lattice[J] ** 3
                )
            )
        return -0.5 * derivative

    @ti.kernel
    def _forces(self, with_flux: ti.template()):
        for I in ti.grouped(self.water_phase):
            force = ti.Vector.zero(ti.f32, 3)
            if ti.static(with_flux):
                for a, sign in ti.static(ti.ndrange(3, 2)):
                    c = ti.Vector.zero(ti.i32, 3)
                    c[a] = 2 * sign - 1
                    J = self._wrapped(I + c)
                    force -= (
                        (2 * sign - 1)
                        * 0.25
                        * (self.phase_mass_flux[I][a] + self.phase_mass_flux[J][a])
                        * (self.velocity_lattice[I] + self.velocity_lattice[J])
                    )
            self.force_density[I] = force

    @ti.kernel
    def _gradients(self):
        for I in ti.grouped(self.water_phase):
            gradient = ti.Vector.zero(ti.f32, 3)
            for q in range(Q):
                c = direction(q)
                gradient += (
                    3
                    * weight(q)
                    * (self.water_phase[self._wrapped(I + c)] - self.water_phase[I])
                    * ti.cast(c, ti.f32)
                )
            self.phase_gradient_lattice[I] = gradient
            self.capillary_stress[I] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def seed(self, phase: ti.template()):
        for I in ti.grouped(self.water_phase):
            if ti.static(phase):
                for q in range(Q):
                    self.h[I, q] = phase_equilibrium(
                        q, self.water_phase[I], self.velocity_lattice[I]
                    )
            else:
                self._set_equilibrium(
                    I, self.water_phase[I], 0.0, self.velocity_lattice[I]
                )

    @ti.kernel
    def stream(self, phase: ti.template()):
        for I in ti.grouped(self.water_phase):
            for q in range(Q):
                J = self._wrapped(I + direction(q))
                self.h[J, q] = self.h_post[I, q]
                if ti.static(not phase):
                    self.f[J, q] = self.f_post[I, q]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=5)
    parser.add_argument("--speed", type=float, default=0.01)
    parser.add_argument("--density-ratio", type=float, default=800)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-flux", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = IceFlowConfig(
        nx=8,
        ny=128,
        nz=8,
        domain_size_m=(8 / 3000, 128 / 3000, 8 / 3000),
        boundary_cells=1,
        water_width=3,
        water_height=10,
        floor_water_depth_cells=5,
        ice_bodies=(IceBodyConfig((4, 6, 4), (4, 4, 4)),),
        phase_warmup_steps=0,
        gas_state_repair=False,
        gravity=(0, 0, 0),
        surface_tension_n_m=0.0,
        backend=args.backend,
        rho_air=1000 / args.density_ratio,
        interface_width=args.width,
        density_gradient_source_limit=0,
        phase_flux_acceleration_limit=0,
    )
    s = PeriodicLayer(config)
    s.solid_body_id.fill(-1)
    y = np.arange(s.ny) + 0.5
    profile = 0.5 * (
        np.tanh(2 * (y - 32) / args.width) - np.tanh(2 * (y - 96) / args.width)
    )
    phi = np.broadcast_to(profile[None, :, None], s.shape).astype(np.float32).copy()
    u0 = np.array([0, args.speed, 0])
    s.water_phase.from_numpy(phi)
    s.velocity_lattice.from_numpy(
        np.broadcast_to(u0, (*s.shape, 3)).astype(np.float32).copy()
    )
    s.seed(True)
    for _ in range(2000):
        s._gradients()
        s._collide_phase()
        s.stream(True)
        s._recover_phase()
    saved_h = s.h.to_numpy()
    s.force_density.fill(0)
    s.pressure_lattice.fill(0)
    s.seed(False)
    s.h.from_numpy(saved_h)
    s.water_volume.fill(s.cell_volume)
    s.water_energy.fill(s.rho_cp * s.cell_volume * (90 - s.tm))
    initial = float(s.water_phase.to_numpy().sum(dtype=float))
    rows = []
    for step in range(args.steps + 1):
        if step % 500 == 0 or step == args.steps:
            u = s.velocity_lattice.to_numpy()
            row = {
                "step": step,
                "max_velocity_error": float(np.linalg.norm(u - u0, axis=-1).max()),
                "max_l1_speed": float(abs(u).sum(-1).max()),
                "phase_relative_error": float(
                    s.water_phase.to_numpy().sum(dtype=float) / initial - 1
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(
                json.dumps(
                    {
                        "args": {**vars(args), "output": str(args.output)},
                        "samples": rows,
                    },
                    indent=2,
                )
            )
            if row["max_l1_speed"] > 0.5 or not np.isfinite(u).all():
                break
        if step == args.steps:
            break
        s._gradients()
        s._collide_phase()
        s._forces(not args.no_flux)
        s._collide_momentum()
        s.stream(False)
        s._recover_phase()
        s._gradients()
        s._recover_momentum()


if __name__ == "__main__":
    main()
