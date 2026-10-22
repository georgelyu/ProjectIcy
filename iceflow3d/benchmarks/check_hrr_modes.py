"""Check a Fourier mode against actual collision/stream/recovery kernels."""

# ruff: noqa: E741 -- Taichi grouped grid indices.
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-root", type=Path)
parser.add_argument("--legacy", action="store_true")
parser.add_argument("--speed", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--bulk", type=float, default=1.0)
parser.add_argument("--cubic", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(
    0, str((args.source_root or Path(__file__).resolve().parents[2]).resolve())
)

from iceflow3d import IceFlow3D, IceFlowConfig, IceBodyConfig  # noqa: E402
from iceflow3d.lattice import Q, REST, direction, weight  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "hrr_spectrum", Path(__file__).with_name("hrr_spectrum.py")
)
theory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory)


@ti.data_oriented
class PeriodicProbe(IceFlow3D):
    @ti.func
    def _stress_neighbor_velocity(self, I, c):
        shape = ti.Vector([self.nx, self.ny, self.nz])
        return self.velocity_lattice[(I + c + shape) % shape]

    @ti.func
    def _regularized_stress(self, I, rho, tau):
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        for q in range(Q):
            if q != REST:
                c = direction(q)
                shape = ti.Vector([self.nx, self.ny, self.nz])
                J = (I + c + shape) % shape
                gradient += (
                    3
                    * weight(q)
                    * self.velocity_lattice[J].outer_product(ti.cast(c, ti.f32))
                )
        return -rho * tau / 3 * (gradient + gradient.transpose())

    @ti.kernel
    def seed_mode(self):
        for I in ti.grouped(self.water_phase):
            self._set_equilibrium(
                I, 0.0, self.pressure_lattice[I], self.velocity_lattice[I]
            )

    @ti.kernel
    def stream_periodic(self):
        for I in ti.grouped(self.water_phase):
            for q in range(Q):
                c = direction(q)
                shape = ti.Vector([self.nx, self.ny, self.nz])
                J = (I + c + shape) % shape
                self.f[J, q] = self.f_post[I, q]


def main():
    n = 24
    options = {}
    if "momentum_collision_model" in IceFlowConfig.__dataclass_fields__:
        options["momentum_collision_model"] = "cm_hrr"
    if "hrr_cubic_correction" in IceFlowConfig.__dataclass_fields__:
        options["hrr_cubic_correction"] = args.cubic
    config = IceFlowConfig(
        nx=n,
        ny=n,
        nz=n,
        domain_size_m=(0.008, 0.008, 0.008),
        boundary_cells=1,
        water_width=6,
        water_height=10,
        floor_water_depth_cells=5,
        ice_bodies=(IceBodyConfig((16, 6, 12), (4, 4, 4)),),
        phase_warmup_steps=0,
        gas_state_repair=False,
        backend="cuda",
        cm_acoustic_relaxation_rate=args.bulk,
        **options,
    )
    s = PeriodicProbe(config)
    u0 = np.full(3, args.speed / np.sqrt(3))
    k = np.full(3, -2 * np.pi / n)
    matrix = theory.population_symbol(
        u0, s.nu_air, k, args.bulk, not args.legacy, not args.legacy, args.cubic
    )[0]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    selected = np.argmax(np.abs(eigenvalues))
    eigenvector = eigenvectors[:, selected]
    phase = np.einsum("i,ixyz->xyz", k, np.indices(s.shape))
    s.solid_body_id.fill(-1)
    s.water_phase.fill(0)
    s.phase_gradient_lattice.fill(0)
    s.force_density.fill(0)
    s.pressure_lattice.fill(0)
    s.velocity_lattice.from_numpy(
        np.broadcast_to(u0, (*s.shape, 3)).astype(np.float32).copy()
    )
    s.seed_mode()
    baseline = s.f.to_numpy().copy()
    perturbation = (
        1e-5 * s.rho_air * np.real(np.exp(1j * phase)[..., None] * eigenvector)
    )
    s.f.from_numpy((baseline + perturbation).astype(np.float32))
    s._recover_momentum()
    left_values, left_vectors = np.linalg.eig(matrix.T)
    left = left_vectors[:, np.argmin(abs(left_values - eigenvalues[selected]))]
    left /= left @ eigenvector

    def measure():
        population = (s.f.to_numpy() - baseline) / s.rho_air
        coefficient = (population * np.exp(-1j * phase)[..., None]).mean(axis=(0, 1, 2))
        return float(abs(left @ coefficient))

    initial = measure()
    for _ in range(args.steps):
        s._collide_momentum()
        s.stream_periodic()
        s._recover_momentum()
    final = measure()
    result = dict(
        legacy=args.legacy,
        speed=args.speed,
        steps=args.steps,
        viscosity=s.nu_air,
        bulk=args.bulk,
        cubic=args.cubic,
        predicted_per_step=float(abs(eigenvalues[selected])),
        measured_per_step=(final / initial) ** (1 / args.steps),
        initial=initial,
        final=final,
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
