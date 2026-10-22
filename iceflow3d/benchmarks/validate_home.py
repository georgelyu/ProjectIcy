"""CUDA acceptance: HOME molecular viscosity, Archimedes force, thermal lift.

The eigenmode reference uses an independent NumPy linearized 10-moment
operator. The buoyancy reference uses displaced volume, NOT solver forces.
All probes retain W=5 and the production dt and viscosity constitutive law.
"""

# ruff: noqa: E741
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig  # noqa: E402
from iceflow3d.checkpoint import source_hashes  # noqa: E402
from iceflow3d.lattice import DIRECTIONS, WEIGHTS  # noqa: E402
from iceflow3d.viscosity import air_dynamic_viscosity_pa_s, water_dynamic_viscosity_pa_s  # noqa: E402


@ti.data_oriented
class PeriodicHome(IceFlow3D):
    @ti.func
    def _home_neighbor(self, I, c):
        shape = ti.Vector(self.shape)
        return (I + c + shape) % shape

    @ti.func
    def _phase_neighbor(self, I, c):
        shape = ti.Vector(self.shape)
        return (I + c + shape) % shape


def shear_eigenmode(nu, n, bulk):
    c, w = np.array(DIRECTIONS), np.array(WEIGHTS)
    hermite = np.array(
        [
            c[:, 0] ** 2 - 1 / 3,
            c[:, 1] ** 2 - 1 / 3,
            c[:, 2] ** 2 - 1 / 3,
            c[:, 0] * c[:, 1],
            c[:, 0] * c[:, 2],
            c[:, 1] * c[:, 2],
        ]
    ).T
    reconstruction = w[:, None] * np.column_stack(
        [np.ones(27), 3 * c, 4.5 * hermite * np.array([1, 1, 1, 2, 2, 2])]
    )
    recovery = np.row_stack([np.ones(27), c.T, hermite.T])
    collision = np.eye(10)
    rate = 1 / (0.5 + 3 * nu)
    collision[4:, 4:] *= 1 - rate
    collision[4:7, 4:7] += (rate - bulk) / 3
    stream = np.exp(-2j * np.pi * c[:, 0] / n)
    operator = (recovery * stream) @ reconstruction @ collision
    values, vectors = np.linalg.eig(operator)
    candidates = np.where(np.abs(vectors[2]) > 0.5 * np.max(np.abs(vectors), axis=0))[0]
    index = min(
        candidates, key=lambda i: abs(values[i] - np.exp(-nu * (2 * np.pi / n) ** 2))
    )
    return values[index], vectors[:, index] / vectors[2, index]


def viscosity(steps, dx, amplitude=1e-4):
    n = 24
    cfg = IceFlowConfig(
        nx=n,
        ny=10,
        nz=10,
        domain_size_m=(n * dx, 10 * dx, 10 * dx),
        boundary_cells=1,
        water_width=6,
        water_height=7,
        floor_water_depth_cells=3,
        ice_bodies=(IceBodyConfig((16, 5, 5), (2, 2, 2)),),
        phase_warmup_steps=0,
        backend="cuda",
    )
    s = PeriodicHome(cfg)
    s.solid_body_id.fill(-1)
    s.home_acceleration.fill(0)
    phase = 2 * np.pi * np.indices(s.shape)[0] / n
    fourier = np.exp(-1j * phase)
    rows = []
    for phi, T in (
        (1.0, 90.0),
        (1.0, 20.0),
        (1.0, 0.0),
        (0.0, 20.0),
        (0.1, 90.0),
        (0.5, 90.0),
        (0.99, 90.0),
    ):
        rho = phi * cfg.rho_water + (1 - phi) * cfg.rho_air
        mu = phi * water_dynamic_viscosity_pa_s(T) + (
            1 - phi
        ) * air_dynamic_viscosity_pa_s(cfg.thermal.initial_air_temperature_c)
        nu = mu / rho * s.scales.dt_s / dx**2
        val, vec = shear_eigenmode(nu, n, cfg.cm_acoustic_relaxation_rate)
        moments = (amplitude * np.real(np.exp(1j * phase)[..., None] * vec)).astype(
            np.float32
        )
        s.home.from_numpy(moments)
        s.velocity_lattice.from_numpy(moments[..., 1:4].copy())
        s.water_phase.fill(phi)
        s.water_volume.fill(phi * s.cell_volume)
        s.water_energy.fill(phi * s.cell_volume * s.rho_cp * (T - s.tm))
        samples = []
        for i in range(steps + 1):
            if i % 1000 == 0 or i == steps:
                u = s.velocity_lattice.to_numpy()
                assert np.isfinite(u).all()
                samples.append([i, float(abs((u[..., 1] * fourier).mean()))])
            if i < steps:
                s._collide_momentum()
                s._stream_home()
        x, y = np.array(samples).T
        measured = -np.polyfit(x, np.log(y), 1)[0] / (2 * np.pi / n) ** 2
        row = dict(
            water_fraction=phi,
            temperature_c=T,
            mu_pa_s=mu,
            nu_lattice=nu,
            expected_discrete_nu=float(-np.log(abs(val)) / (2 * np.pi / n) ** 2),
            measured_nu=float(measured),
            ratio=float(measured / nu),
            samples=samples,
        )
        row["passed"] = abs(row["ratio"] - 1) < 0.05
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(
        dx_m=dx, dt_s=s.scales.dt_s, cases=rows, passed=all(r["passed"] for r in rows)
    )


def buoyancy():
    cfg = IceFlowConfig(
        nx=30,
        ny=30,
        nz=24,
        domain_size_m=(0.03, 0.03, 0.024),
        boundary_cells=1,
        water_width=29,
        water_height=25,
        floor_water_depth_cells=0,
        ice_bodies=(IceBodyConfig((15, 13, 12), (8, 8, 8)),),
        phase_warmup_steps=0,
        surface_tension_n_m=0,
        backend="cuda",
    )
    s = IceFlow3D(cfg)
    ids = s.solid_body_id.to_numpy()
    fluid = ids == -1
    volume = 8**3
    rows = []
    # Fluid can be uniform air, water, or a prescribed smooth mixture.
    for phi in (0.0, 1.0, 0.37):
        s.water_phase.from_numpy(np.where(fluid, phi, 0).astype("f"))
        s.water_volume.from_numpy(np.where(fluid, phi * s.cell_volume, 0).astype("f"))
        s.water_energy.from_numpy(
            (
                s.water_volume.to_numpy()
                * s.rho_cp
                * cfg.thermal.initial_water_temperature_c
            ).astype("f")
        )
        s._build_reference()
        s._gradients()
        s._forces(False)
        s._seed_populations()
        s._collide_momentum()
        s.hydrodynamic_force.fill(0)
        s.hydrodynamic_torque.fill(0)
        s._stream_home()
        actual = s.hydrodynamic_force.to_numpy()[0].astype(float)
        expected = -volume * (phi + (1 - phi) * s.rho_air) * np.array(s.gravity)
        torque = s.hydrodynamic_torque.to_numpy()[0].astype(float)
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        row = dict(
            water_fraction=phi,
            force_n=(actual * s.scales.force_n).tolist(),
            expected_force_n=(expected * s.scales.force_n).tolist(),
            relative_error=error,
            torque_lattice=torque.tolist(),
            passed=error < 2e-4,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Restore pure water, test the TOTAL acceleration after adding ice gravity.
    s.water_phase.from_numpy(np.where(fluid, 1, 0).astype("f"))
    s._build_reference()
    s._gradients()
    s._forces(False)
    s._seed_populations()
    s._collide_momentum()
    s.hydrodynamic_force.fill(0)
    s.hydrodynamic_torque.fill(0)
    s._stream_home()
    s.rigid.velocity.fill(0)
    s.rigid.omega.fill(0)
    s.rigid.advance()
    actual = s.rigid.velocity[0]
    expected = (1 - cfg.rho_water / cfg.rho_ice) * np.array(s.gravity)
    acceleration_error = float(
        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    )
    # Boussinesq lift of water ten kelvin hotter, independent of ice force.
    s.water_volume.from_numpy(np.where(fluid, s.cell_volume, 0).astype("f"))
    s.water_energy.from_numpy(
        (
            s.water_volume.to_numpy()
            * s.rho_cp
            * (cfg.thermal.initial_water_temperature_c + 10)
        ).astype("f")
    )
    s.velocity_lattice.fill(0)
    s.pressure_lattice.fill(0)
    s._forces(False)
    I = (5, 8, 5)
    expected_thermal = (
        -cfg.thermal.thermal_expansion_water_1_k * 10 * np.array(s.gravity)
    )
    actual_thermal = s.force_density.to_numpy()[I].astype(float)
    thermal_error = float(
        np.linalg.norm(actual_thermal - expected_thermal)
        / np.linalg.norm(expected_thermal)
    )

    # A block intersecting the diffuse free surface: choose the level by
    # integrating the prescribed W=5 tanh density through the solid volume.
    # This checks that normalized g does not mistake pressure for rho/3.
    def submerged_fraction(level):
        W = cfg.interface_width
        return 0.5 + W / 32 * (
            np.log(np.cosh(2 * (level - 9) / W)) - np.log(np.cosh(2 * (level - 17) / W))
        )

    target = (cfg.rho_ice - cfg.rho_air) / (cfg.rho_water - cfg.rho_air)
    lo, hi = 9.0, 24.0
    for _ in range(64):
        level = 0.5 * (lo + hi)
        if submerged_fraction(level) < target:
            lo = level
        else:
            hi = level
    profile = 0.5 + 0.5 * np.tanh(
        2 * (level - np.arange(s.ny) - 0.5) / cfg.interface_width
    )
    s.water_phase.from_numpy(np.where(fluid, profile[None, :, None], 0).astype("f"))
    s.water_volume.from_numpy((s.water_phase.to_numpy() * s.cell_volume).astype("f"))
    s.water_energy.from_numpy(
        (
            s.water_volume.to_numpy()
            * s.rho_cp
            * cfg.thermal.initial_water_temperature_c
        ).astype("f")
    )
    s.body_velocity.fill(0)
    s.body_angular_velocity.fill(0)
    s._build_reference()
    s._gradients()
    s._forces(False)
    s._seed_populations()
    for _ in range(256):
        s._collide_momentum()
        s.hydrodynamic_force.fill(0)
        s.hydrodynamic_torque.fill(0)
        s._stream_home()
        s._recover_pressure()
        s._forces(False)
    target_lift = -volume * cfg.rho_ice / cfg.rho_water * s.gravity[1]
    surface_error = abs(float(s.hydrodynamic_force[0][1]) / target_lift - 1)
    stationary_speed = float(np.abs(s.velocity_lattice.to_numpy()[fluid]).sum(-1).max())
    return dict(
        cases=rows,
        net_ice_acceleration_relative_error=acceleration_error,
        thermal_force_relative_error=thermal_error,
        diffuse_free_surface_relative_error=surface_error,
        diffuse_surface_level_cells=level,
        stationary_column_speed_lattice=stationary_speed,
        passed=all(r["passed"] for r in rows)
        and acceleration_error < 0.003
        and thermal_error < 2e-5
        and surface_error < 0.01
        and stationary_speed < 1e-5,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("viscosity", "buoyancy", "all"), default="all")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--dx", type=float, default=1 / 3000)
    p.add_argument("--amplitude", type=float, default=1e-4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = dict(source_sha256=source_hashes(), backend="cuda")
    if args.mode in ("buoyancy", "all"):
        result["buoyancy"] = buoyancy()
    if args.mode in ("viscosity", "all"):
        result["viscosity"] = viscosity(args.steps, args.dx, args.amplitude)
    result["passed"] = all(
        result[k]["passed"] for k in ("buoyancy", "viscosity") if k in result
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASSED" if result["passed"] else "FAILED", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
