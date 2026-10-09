"""Isolate phase/momentum transport in a periodic translating planar layer.

No solids, gravity, surface tension or thermal updates are present. The exact
velocity stays uniform. Uses production collision, recovery and phase bounds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import taichi as ti

from iceflow2d_v3.lattice import lattice_direction
from iceflow2d_v3.simulator import IceFlow2D
from iceflow2d_v3.test_iceflow import _falling_config


@ti.data_oriented
class TranslationProbe(IceFlow2D):
    @ti.func
    def _stability_neighbor(self, i, j, di, dj):
        return ti.Vector([(i + di + self.nx) % self.nx, (j + dj + self.ny) % self.ny])

    @ti.func
    def _is_fluid_cell(self, i, j):
        return True

    @ti.func
    def _phase_neighbor(self, i, j, step_x, step_y, distance):
        return self.water_phase[
            (i + step_x * distance + self.nx * 2) % self.nx,
            (j + step_y * distance + self.ny * 2) % self.ny,
        ]

    @ti.func
    def _fluid_force_and_gradient(self, i, j):
        grad = self._phase_central_gradient(i, j)
        return ti.Vector([0.0, 0.0, grad.x, grad.y])

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        return raw_post

    @ti.kernel
    def _stream(self):
        for i, j in self.water_phase:
            for q in range(9):
                c = lattice_direction(q)
                ni = (i - c.x + self.nx) % self.nx
                nj = (j - c.y + self.ny) % self.ny
                self.momentum_populations[i, j, q] = self.momentum_stream_buffer[
                    ni, nj, q
                ]
                self.phase_populations[i, j, q] = self.phase_stream_buffer[ni, nj, q]

    @ti.kernel
    def stream_momentum(self):
        for i, j, q in self.momentum_populations:
            c = lattice_direction(q)
            self.momentum_populations[i, j, q] = self.momentum_stream_buffer[
                (i - c.x + self.nx) % self.nx, (j - c.y + self.ny) % self.ny, q
            ]

    @ti.kernel
    def stream_phase(self):
        for i, j in self.water_phase:
            for q in range(9):
                c = lattice_direction(q)
                self.phase_populations[i, j, q] = self.phase_stream_buffer[
                    (i - c.x + self.nx) % self.nx, (j - c.y + self.ny) % self.ny, q
                ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--density-ratio", type=float, default=800.0)
    p.add_argument("--speed", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument(
        "--momentum-collision-model",
        choices=("cm_mrt", "rm_mrt", "cm_hrr"),
        default="cm_mrt",
    )
    p.add_argument("--hrr-kinetic-blend", type=float, default=0.0)
    p.add_argument(
        "--phase-momentum-coupling",
        choices=("legacy", "consistent_flux"),
        default="legacy",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = _falling_config(
        resolution=(32, 128),
        dx=1 / 3000,
        interface_width=args.width,
        phase_warmup_steps=0,
        sigma=0.0,
        gravity=(0.0, 0.0),
        rho_air=1000.0 / args.density_ratio,
        reference_velocity=4.0,
        viscosity_water=3.141732579601075e-7,
        viscosity_air=1.4506451972115707e-5,
        momentum_collision_model=args.momentum_collision_model,
        hrr_kinetic_blend=args.hrr_kinetic_blend,
        phase_momentum_coupling=args.phase_momentum_coupling,
    )
    s = TranslationProbe(c)
    y = np.arange(s.ny) + 0.5
    phi = (
        np.broadcast_to(
            0.5
            * (np.tanh(2 * (y - 32) / args.width) - np.tanh(2 * (y - 96) / args.width)),
            (s.nx, s.ny),
        )
        .astype(np.float32)
        .copy()
    )
    weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
    directions = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    u0 = np.array([0.0, args.speed], dtype=np.float32)
    cu = directions @ u0
    velocity_poly = 3 * cu + 4.5 * cu**2 - 1.5 * (u0 @ u0)
    s.water_phase.from_numpy(phi)
    s.momentum_velocity_lattice.from_numpy(np.broadcast_to(u0, (s.nx, s.ny, 2)).copy())
    s.fluid_acceleration_lattice.fill(0.0)
    s.pressure_lattice.fill(0.0)
    s.reference_pressure_by_row.fill(0.0)
    s.phase_populations.from_numpy(
        (phi[..., None] * weights * (1 + velocity_poly)).astype(np.float32)
    )
    # Let the phase-only transport settle with the exact velocity prescribed.
    for _ in range(2000):
        s._collide_phase()
        s.stream_phase()
        s._update_streamed_phase()
        s._bound_water_phase()
    phi = s.water_phase.to_numpy()
    rho = 1 / args.density_ratio + (1 - 1 / args.density_ratio) * phi
    s.momentum_populations.from_numpy(
        (rho[..., None] * weights * velocity_poly).astype(np.float32)
    )
    s._update_pressure()
    rows = []
    for step in range(args.steps + 1):
        if step % 20 == 0 or step == args.steps:
            u = s.momentum_velocity_lattice.to_numpy()
            ph = s.water_phase.to_numpy()
            row = dict(
                step=step,
                max_speed=float(np.linalg.norm(u, axis=-1).max()),
                max_velocity_error=float(np.linalg.norm(u - u0, axis=-1).max()),
                phi_min=float(ph.min()),
                phi_max=float(ph.max()),
            )
            rows.append(row)
            if not np.isfinite(u).all() or row["max_speed"] > 1.0:
                break
        if step < args.steps:
            s._advance_fluid()
    summary = dict(
        width=args.width,
        config=c.to_dict(),
        density_ratio=args.density_ratio,
        speed=args.speed,
        last=rows[-1],
        completed=rows[-1]["step"] == args.steps,
        peak_error=max(r["max_velocity_error"] for r in rows),
    )
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output / "final.npz",
        phi=s.water_phase.to_numpy(),
        u=s.momentum_velocity_lattice.to_numpy(),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
