"""Periodic planar-interface relaxation/advection with unchanged five-cell W.

Reproduce the historical 0.01 versus 1e-8 comparison using the actual phase
collision/recovery kernels. The strict purity assertions apply to 1e-8,
independently of the production default. No global volume correction is used.
"""

# ruff: noqa: E741
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig  # noqa: E402
from iceflow3d.lattice import Q, direction, weight  # noqa: E402


@ti.data_oriented
class PhaseProbe(IceFlow3D):
    @ti.func
    def _phase_neighbor(self, I, c):
        shape = ti.Vector(self.shape)
        return (I + c + shape) % shape

    @ti.kernel
    def seed(self):
        for I in ti.grouped(self.water_phase):
            self._set_phase_equilibrium(I, self.water_phase[I], self.velocity_lattice[I])

    @ti.kernel
    def periodic_gradient(self):
        for I in ti.grouped(self.water_phase):
            shape = ti.Vector([self.nx, self.ny, self.nz])
            gradient = ti.Vector.zero(ti.f32, 3)
            for q in range(Q):
                c = direction(q)
                gradient += (
                    3
                    * weight(q)
                    * ti.cast(c, ti.f32)
                    * (self.water_phase[(I + c + shape) % shape] - self.water_phase[I])
                )
            self.phase_gradient_lattice[I] = gradient



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    n = 96
    config = IceFlowConfig(
        nx=n,
        ny=12,
        nz=12,
        domain_size_m=(0.096, 0.012, 0.012),
        boundary_cells=1,
        water_width=20,
        water_height=8,
        floor_water_depth_cells=4,
        ice_bodies=(IceBodyConfig((60, 5, 6), (4, 4, 4)),),
        phase_warmup_steps=0,
        backend="cuda",
    )
    x = np.arange(n) + 0.5
    records = []
    purity_reference_epsilon = 1e-8
    for epsilon in (0.01, purity_reference_epsilon):
        s = PhaseProbe(replace(config, phase_normal_epsilon=epsilon))
        s.solid_body_id.fill(-1)
        for speed in (0.0, 0.025):
            phi = 0.5 + 0.5 * np.tanh(2 * (n / 4 - abs(x - n / 2)) / 5)
            s.water_phase.from_numpy(
                np.broadcast_to(phi[:, None, None], s.shape).astype(np.float32)
            )
            s.velocity_lattice.fill([speed, 0, 0])
            s.phase_clip_volume.fill(0)
            s.seed()
            initial = float(s.water_phase.to_numpy().sum(dtype=np.float64))
            for step in range(args.steps):
                s.periodic_gradient()
                s._collide_phase()
                s._recover_phase()
            profile = s.water_phase.to_numpy().mean(axis=(1, 2), dtype=np.float64)
            distance = n / 4 - abs((x - speed * args.steps) % n - n / 2)
            expected = 0.5 + 0.5 * np.tanh(2 * distance / 5)
            water, air = profile[distance > 12], profile[distance < -12]
            middle = (profile > 0.1) & (profile < 0.9)
            # logit(phi) = 4*distance/W for the prescribed tanh profile.
            slope = np.polyfit(
                distance[middle], np.log(profile[middle] / (1 - profile[middle])), 1
            )[0]
            row = dict(
                normal_epsilon=epsilon,
                velocity_lattice=speed,
                steps=args.steps,
                water_bulk_median=float(np.median(water)),
                air_bulk_median=float(np.median(air)),
                water_bulk_max_error=float(np.max(1 - water)),
                air_bulk_max_error=float(np.max(air)),
                fitted_interface_width_cells=float(4 / slope),
                relative_volume_change=float(profile.sum() * s.ny * s.nz / initial - 1),
                mean_absolute_profile_error=float(abs(profile - expected).mean()),
                clipped_volume_cells=float(s.phase_clip_volume[None]),
                profile=profile.tolist(),
            )
            records.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "profile"}), flush=True
            )
    fixed = [r for r in records if r["normal_epsilon"] == purity_reference_epsilon]
    passed = all(
        r["water_bulk_max_error"] < 2e-4
        and r["air_bulk_max_error"] < 2e-4
        and abs(r["fitted_interface_width_cells"] / config.interface_width - 1) < 0.05
        and abs(r["relative_volume_change"]) < 1e-3
        for r in fixed
    )
    args.output.write_text(json.dumps(dict(passed=passed, cases=records), indent=2))
    assert passed, "phase transport does not retain pure phases / prescribed width"


if __name__ == "__main__":
    main()
