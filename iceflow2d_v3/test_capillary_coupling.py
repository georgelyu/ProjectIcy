"""Physical and discrete balances for surface tension at moving ice."""

import math
import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config, _prepare_test_cut_links


@ti.kernel
def _sample_capillary_force(
    s: ti.template(), out: ti.types.ndarray(dtype=ti.f64, ndim=3)
):
    for i, j in s.water_phase:
        force = ti.Vector.zero(ti.f32, 2)
        if s._is_fluid_cell(i, j):
            force = s._capillary_force_density(i, j)
        out[i, j, 0] = force.x
        out[i, j, 1] = force.y


def _forces(s):
    s._prepare_capillary_stress()
    out = np.empty((s.nx, s.ny, 2), dtype=np.float64)
    _sample_capillary_force(s, out)
    return out


def _isolated_config(**overrides):
    values = dict(
        resolution=(80, 96),
        dx=0.00025,
        water_width_fraction=78 / 80,
        water_height_fraction=40 / 96,
        ice_width_fraction=12 / 80,
        ice_height_fraction=12 / 96,
        ice_base_y_cells=70,
        gravity=(0.0, 0.0),
        phase_warmup_steps=0,
    )
    values.update(overrides)
    return _falling_config(**values)


class CapillaryCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_flat_interface_has_no_capillary_acceleration(self):
        s = IceFlow2D(_isolated_config())
        y = np.arange(s.ny) + 0.5
        # Test both an equilibrium-width and a deliberately thicker profile.
        for width in (5.0, 9.0):
            phase = (
                np.broadcast_to(
                    0.5 * (1.0 - np.tanh(2.0 * (y - 28.0) / width)),
                    (s.nx, s.ny),
                )
                .astype(np.float32)
                .copy()
            )
            s.water_phase.from_numpy(phase)
            force = _forces(s)
            self.assertLess(float(np.abs(force).max()), 1e-10)
            # Surface stress must remain present despite the zero divergence.
            self.assertGreater(float(s.capillary_stress_lattice.to_numpy().max()), 1e-6)

    def test_circular_interface_retains_laplace_pressure(self):
        s = IceFlow2D(_isolated_config())
        x, y = np.meshgrid(np.arange(s.nx), np.arange(s.ny), indexing="ij")
        r = np.hypot(x - 40, y - 32)
        for radius in (12.0, 20.0):
            phase = 0.5 * (1.0 - np.tanh(2.0 * (r - radius) / s.config.interface_width))
            s.water_phase.from_numpy(phase.astype(np.float32))
            force = _forces(s)
            # Integrate radial traction from drop centre into the gas:
            # Delta p = sigma/R in two dimensions (not 2*sigma/R).
            pressure_jump = -float(force[40:, 32, 0].sum())
            expected = s._surface_tension_lattice / radius
            self.assertAlmostEqual(pressure_jump / expected, 1.0, delta=0.06)
            np.testing.assert_allclose(force.sum(axis=(0, 1)), 0.0, atol=1e-9)

    def test_flat_interface_meeting_ice_stays_at_rest(self):
        # A 90-degree contact line has no curvature or imposed motion. The
        # original blocked-diagonal fallback generated centimetres/second here.
        s = IceFlow2D(
            _isolated_config(
                ice_width_fraction=16 / 80,
                ice_height_fraction=32 / 96,
                ice_base_y_cells=24,
                ice_initial_angle=0.0,
                viscosity_water=None,
                viscosity_air=None,
            )
        )
        y = np.arange(s.ny) + 0.5
        phase = (
            np.broadcast_to(
                0.5 * (1.0 - np.tanh(2.0 * (y - 40.0) / s.config.interface_width)),
                (s.nx, s.ny),
            )
            .astype(np.float32)
            .copy()
        )
        s.water_phase.from_numpy(phase)
        s._build_hydrostatic_reference()
        s._initialize_hydrostatic_equilibrium()
        s.thermal._initialize_water_state(s.water_phase, s.wall_mask, s.solid_mask)
        self.assertLess(float(np.abs(_forces(s)).max()), 1.0e-10)
        initial_volume = phase[
            (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
        ].sum(dtype=float)
        for _ in range(3200):
            s._advance_fluid()
        phase = s.water_phase.to_numpy()
        fluid = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
        water = fluid & (phase >= 0.5)
        velocity = s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s
        self.assertLess(float(np.linalg.norm(velocity[water], axis=1).max()), 1.0e-4)
        self.assertLess(abs(phase[fluid].sum(dtype=float) / initial_volume - 1), 1.0e-4)

    def test_ice_receives_opposite_capillary_force_and_torque(self):
        for angle in (-23.0, 0.0, 23.0):
            s = IceFlow2D(
                _isolated_config(
                    ice_base_y_cells=42.0,
                    ice_initial_angle=math.radians(angle),
                )
            )
            center = np.asarray(s.body_center[None], dtype=np.float64)
            x, y = np.meshgrid(
                np.arange(s.nx) + 0.5, np.arange(s.ny) + 0.5, indexing="ij"
            )
            # An off-centre curved interface intersects the tilted ice. Its
            # net fluid force and torque are nonzero and must reach the body.
            radius = np.hypot(x - center[0] + 9.0, y - center[1] + 3.0)
            phase = 0.5 * (
                1.0 - np.tanh(2.0 * (radius - 9.0) / s.config.interface_width)
            )
            s.water_phase.from_numpy(phase.astype(np.float32))
            force = _forces(s)
            fluid_force = force.sum(axis=(0, 1))
            fluid_torque = np.sum(
                (x - center[0]) * force[..., 1] - (y - center[1]) * force[..., 0]
            )
            # With zero gravity, pressure, populations and velocity, the actual
            # production cut-link path contributes only capillary traction.
            s.momentum_populations.fill(0.0)
            s.momentum_stream_buffer.fill(0.0)
            s.pressure_lattice.fill(0.0)
            s.momentum_velocity_lattice.fill(0.0)
            s.fluid_acceleration_lattice.fill(0.0)
            s.hydrodynamic_impulse.fill(0.0)
            s.hydrodynamic_torque.fill(0.0)
            _prepare_test_cut_links(s)
            body_force = np.asarray(s.hydrodynamic_impulse[None], dtype=float)
            body_torque = float(s.hydrodynamic_torque[None])
            self.assertGreater(float(np.linalg.norm(body_force)), 1e-5)
            scale = float(np.abs(force).sum())
            np.testing.assert_allclose(fluid_force + body_force, 0.0, atol=2e-6 * scale)
            self.assertLess(abs(fluid_torque + body_torque), 2e-6 * scale * s.nx)

    def test_stationary_drop_has_small_residual_flow_and_laplace_pressure(self):
        from iceflow2d_v3.benchmarks.check_capillary_equilibrium import (
            initialize_rest,
            make_case,
        )

        config, phase, radius = make_case("drop")
        s = IceFlow2D(config)
        initialize_rest(s, phase, 20.0)
        for _ in range(16000):
            s._advance_fluid()
        fluid = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
        water = fluid & (s.water_phase.to_numpy() >= 0.5)
        velocity = s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s
        # This coarse diffuse interface still has residual spurious currents;
        # keep them bounded without pretending the scheme is exactly balanced.
        self.assertLess(float(np.linalg.norm(velocity[water], axis=1).max()), 0.0025)
        pressure = (
            s.pressure_lattice.to_numpy()
            * config.rho_water
            * s.scales.velocity_scale_m_s**2
        )
        jump = (
            pressure[fluid & (radius < 12)].mean()
            - pressure[fluid & (radius > 30) & (radius < 38)].mean()
        )
        self.assertAlmostEqual(float(jump) / (config.sigma / 0.005), 1.0, delta=0.04)

    def test_horizontal_ice_impact_remains_stable(self):
        from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import (
            create_config,
            parse_args,
        )

        s = IceFlow2D(create_config(parse_args(["--ice-angle-degrees", "0"])))
        # Include the rebound as well as first contact: candidate capillary
        # laws that passed static tests became unstable between 0.1 and 0.3 s.
        for _ in range(3):
            s.step(round(0.1 / s.scales.dt_s))
            phase = s.water_phase.to_numpy()
            velocity = (
                s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s
            )
            self.assertTrue(np.isfinite(phase).all())
            self.assertTrue(np.isfinite(velocity).all())
            water = (
                (phase >= 0.5)
                & (s.solid_mask.to_numpy() == 0)
                & (s.wall_mask.to_numpy() == 0)
            )
            self.assertLess(float(np.linalg.norm(velocity[water], axis=1).max()), 0.5)
            self.assertLess(abs(float(s.body_center[None][0]) - s.nx / 2), 0.5)

    def test_zero_sigma_and_pure_phases_have_no_capillary_load(self):
        for sigma in (0.0, 0.072):
            s = IceFlow2D(_isolated_config(sigma=sigma))
            phases = [0.0, 1.0]
            if sigma == 0.0:
                phases += [
                    np.random.default_rng(10).random((s.nx, s.ny)).astype(np.float32)
                ]
            for phase in phases:
                if np.isscalar(phase):
                    s.water_phase.fill(phase)
                else:
                    s.water_phase.from_numpy(phase)
                np.testing.assert_array_equal(_forces(s), 0.0)
                np.testing.assert_array_equal(
                    s.capillary_stress_lattice.to_numpy(), 0.0
                )


if __name__ == "__main__":
    unittest.main()
