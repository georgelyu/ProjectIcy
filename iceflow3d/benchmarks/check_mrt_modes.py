"""Measure physical shear-wave decay using the actual CM-MRT CUDA kernels.

The uniform periodic probe isolates collision, streaming and recovery. It
does not use HRR, velocity-gradient stress, phase projection or gas repair.
"""

# ruff: noqa: E741 -- Taichi grouped grid indices.

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig  # noqa: E402
from iceflow3d.benchmarks.mrt_spectrum import shear_mode  # noqa: E402
from iceflow3d.lattice import Q, direction  # noqa: E402
from iceflow3d.viscosity import (  # noqa: E402
    air_dynamic_viscosity_pa_s,
    water_dynamic_viscosity_pa_s,
)


@ti.data_oriented
class PeriodicProbe(IceFlow3D):
    @ti.kernel
    def stream_periodic(self):
        for I in ti.grouped(self.water_phase):
            for q in range(Q):
                shape = ti.Vector([self.nx, self.ny, self.nz])
                self.f[(I + direction(q) + shape) % shape, q] = self.f_post[I, q]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--amplitude", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = 24
    config = IceFlowConfig(
        nx=n,
        ny=12,
        nz=12,
        domain_size_m=(0.024, 0.012, 0.012),
        boundary_cells=1,
        water_width=6,
        water_height=8,
        floor_water_depth_cells=4,
        ice_bodies=(IceBodyConfig((16, 5, 6), (4, 4, 4)),),
        phase_warmup_steps=0,
        momentum_collision_model="cm_mrt",
        backend="cuda",
    )
    s = PeriodicProbe(config)
    s.solid_body_id.fill(-1)
    s.phase_gradient_lattice.fill(0)
    s.force_density.fill(0)
    phase = 2 * np.pi * np.indices(s.shape)[0] / n
    fourier = np.exp(-1j * phase)
    cases = []
    for phi, temperature in (
        (1.0, 90),
        (1.0, 20),
        (1.0, 0),
        (0.0, 20),
        (0.1, 90),
        (0.5, 90),
        (0.99, 90),
    ):
        name = "water" if phi == 1 else ("air" if phi == 0 else "mixture")
        mu = phi * water_dynamic_viscosity_pa_s(temperature) + (
            1 - phi
        ) * air_dynamic_viscosity_pa_s(config.thermal.initial_air_temperature_c)
        density = phi * config.rho_water + (1 - phi) * config.rho_air
        s.water_phase.fill(phi)
        s.water_volume.fill(phi * s.cell_volume)
        s.water_energy.fill(phi * s.cell_volume * s.rho_cp * (temperature - s.tm))
        s.velocity_lattice.fill(0)
        s.pressure_lattice.fill(0)
        nu = mu / density * s.scales.dt_s / s.scales.dx_m**2
        value, vector = shear_mode(nu, n, config.cm_acoustic_relaxation_rate)
        amplitude = args.amplitude
        perturbation = (
            density
            / config.rho_water
            * amplitude
            * np.real(np.exp(1j * phase)[..., None] * vector)
        )
        s.f.from_numpy(perturbation.astype(np.float32))
        s._recover_momentum()
        samples = []
        for step in range(args.steps + 1):
            if step % 1000 == 0 or step == args.steps:
                velocity = s.velocity_lattice.to_numpy()
                assert np.isfinite(velocity).all()
                coefficient = abs((velocity[..., 1] * fourier).mean())
                samples.append((step, float(coefficient)))
            if step < args.steps:
                s._collide_momentum()
                s.stream_periodic()
                s._recover_momentum()
        steps, amplitudes = np.array(samples).T
        slope = np.polyfit(steps, np.log(amplitudes), 1)[0]
        measured = -slope / (2 * np.pi / n) ** 2
        theoretical = -np.log(abs(value)) / (2 * np.pi / n) ** 2
        row = dict(
            phase=name,
            water_fraction=phi,
            temperature_c=temperature,
            mu_pa_s=mu,
            nu_lattice=nu,
            predicted_shear_nu=float(theoretical),
            measured_shear_nu=float(measured),
            ratio=float(measured / nu),
            samples=samples,
        )
        cases.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        configuration=config.to_dict(),
        steps=args.steps,
        wavelength_cells=n,
        amplitude_lattice=args.amplitude,
        cases=cases,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).resolve().parents[1].glob("*.py")
        },
    )
    # At these very small viscosities the f32 kinetic evolution must resolve
    # less than 1% decay in the hot-water case. Record errors explicitly.
    result["passed"] = all(abs(row["ratio"] - 1) < 0.05 for row in cases)
    args.output.write_text(json.dumps(result, indent=2))
    assert result["passed"], "measured molecular shear viscosity differs by >=5%"


if __name__ == "__main__":
    main()
