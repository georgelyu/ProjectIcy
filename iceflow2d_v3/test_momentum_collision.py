"""Pressure MRT contracts against an independent dense-matrix collision."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import (
    create_config,
    parse_args,
)
from iceflow2d_v3.lattice import lattice_direction
from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


_C = np.array(
    [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=np.float64,
)
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_COLLISION_CASES = (("rm_mrt", 1.0), ("cm_mrt", 1.0), ("cm_mrt", 0.25), ("cm_mrt", 0.6))


def _basis(frame_velocity):
    """Rows: 00, 10, 01, trace, deviatoric normal, shear, 21, 12, 22."""
    x, y = (_C - frame_velocity).T
    return np.array(
        [
            np.ones(9),
            x,
            y,
            x * x + y * y,
            x * x - y * y,
            x * y,
            x * x * y,
            x * y * y,
            x * x * y * y,
        ]
    )


def _equilibrium(density, pressure, velocity):
    # Construct the pressure and velocity contributions in population space;
    # never substitute the Maxwellian central moments of a mass distribution.
    cu = _C @ velocity
    result = _W * (
        3 * pressure + density * (3 * cu + 4.5 * cu**2 - 1.5 * velocity @ velocity)
    )
    result[0] -= 3 * pressure
    return result


def _reference_collision(
    populations,
    density,
    pressure,
    velocity,
    force,
    grad_phi,
    density_jump,
    nu_lattice,
    model,
    acoustic_rate=1.0,
):
    transform = _basis(velocity if model == "cm_mrt" else np.zeros(2))
    equilibrium = _equilibrium(density, pressure, velocity)
    source = (
        3
        * _W
        * (_C @ (density * force) + density_jump * (_C @ velocity) * (_C @ grad_phi))
    )
    omega = 1 / (0.5 + 3 * nu_lattice)
    lower_rate = omega if model == "rm_mrt" else 1.0
    rates = np.array(
        [
            lower_rate,
            lower_rate,
            lower_rate,
            acoustic_rate,
            omega,
            omega,
            1,
            1,
            acoustic_rate,
        ]
    )
    moments = transform @ populations
    post = (
        moments
        - rates * (moments - transform @ equilibrium)
        + (1 - 0.5 * rates) * (transform @ source)
    )
    return np.linalg.solve(transform, post)


@ti.data_oriented
class _CollisionProbe(IceFlow2D):
    def __init__(self, model, acoustic_rate=1.0):
        super().__init__(
            _falling_config(
                momentum_collision_model=model,
                cm_acoustic_relaxation_rate=acoustic_rate,
                phase_warmup_steps=0,
                gravity=(0.0, 0.0),
                sigma=0.0,
                viscosity_water=3.1417325796010753e-7,
                viscosity_air=1.4506451972115707e-5,
            )
        )
        self.probe_force_data = ti.Vector.field(4, ti.f32, shape=(self.nx, self.ny))

    @ti.func
    def _fluid_force_and_gradient(self, i, j):
        # Isolate collision with prescribed acceleration and density gradient.
        # The production source expression and its moment prefactors still run.
        return self.probe_force_data[i, j]

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        return raw_post


@ti.data_oriented
class _PeriodicShearProbe(IceFlow2D):
    @ti.func
    def _is_fluid_cell(self, i, j):
        return True

    @ti.func
    def _fluid_force_and_gradient(self, i, j):
        return ti.Vector.zero(ti.f32, 4)

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        return raw_post

    @ti.kernel
    def stream_periodic_momentum(self):
        for i, j, q in self.momentum_populations:
            c = lattice_direction(q)
            self.momentum_populations[i, j, q] = self.momentum_stream_buffer[
                (i - c.x + self.nx) % self.nx, (j - c.y + self.ny) % self.ny, q
            ]


class MomentumCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_both_frames_match_matrix_collision_with_full_source(self):
        # Include both pure phases, interface tails, nonzero pressure and
        # velocities, and a nonzero source zeroth moment u.grad(rho).
        phases = (0.0, 1.0, 0.001, 0.2, 0.8, 0.999)
        outputs = {}
        for model, acoustic_rate in _COLLISION_CASES:
            s = _CollisionProbe(model, acoustic_rate)
            s.reference_pressure_by_row.fill(0.002)
            probes = []
            for index, phase in enumerate(phases):
                cell = (4 + 4 * index, 6)
                velocity = np.array([0.03, -0.02]) * (-1 if index % 2 else 1)
                force = np.array([2e-4, -3e-4])
                gradient = np.array([0.017, -0.011])
                if phase in (0.0, 1.0):
                    gradient[:] = 0
                density = s._air_density_lattice + phase * (
                    s._water_density_lattice - s._air_density_lattice
                )
                pressure = (-1 if index % 2 else 1) * 3e-5
                # Preserve the half-force definition of the physical velocity;
                # perturb every other moment to expose projection mistakes.
                perturbation = np.array(
                    [
                        2e-5,
                        -0.5 * density * force[0],
                        -0.5 * density * force[1],
                        3e-4,
                        -2e-4,
                        1e-4,
                        4e-5,
                        -3e-5,
                        2e-5,
                    ]
                )
                populations = _equilibrium(density, pressure, velocity)
                populations += np.linalg.solve(_basis(np.zeros(2)), perturbation)
                s.water_phase[cell] = phase
                s.pressure_lattice[cell] = 0.002 + pressure
                s.momentum_velocity_lattice[cell] = velocity - 0.5 * force
                s.probe_force_data[cell] = np.r_[force, gradient]
                for q in range(9):
                    s.momentum_populations[cell[0], cell[1], q] = populations[q]
                probes.append(cell)

            before = s.momentum_populations.to_numpy().astype(np.float64)
            phase_values = s.water_phase.to_numpy()
            velocity_values = s.momentum_velocity_lattice.to_numpy()
            force_values = s.probe_force_data.to_numpy()
            pressure_values = (
                s.pressure_lattice.to_numpy()
                - s.hydrostatic_reference_pressure.to_numpy()
            )
            s._collide_momentum()
            after = s.momentum_stream_buffer.to_numpy().astype(np.float64)
            outputs[model, acoustic_rate] = np.array([after[cell] for cell in probes])
            for cell in probes:
                with self.subTest(model=model, acoustic_rate=acoustic_rate, cell=cell):
                    phi = float(phase_values[cell])
                    density_jump = s._water_density_lattice - s._air_density_lattice
                    density = s._air_density_lattice + phi * density_jump
                    force = force_values[cell][:2].astype(np.float64)
                    velocity = velocity_values[cell] + 0.5 * force_values[cell][:2]
                    nu = (
                        phi * s._water_viscosity_lattice
                        + (1 - phi) * s._air_viscosity_lattice
                    )
                    expected = _reference_collision(
                        before[cell],
                        density,
                        float(pressure_values[cell]),
                        velocity.astype(np.float64),
                        force,
                        force_values[cell][2:].astype(np.float64),
                        density_jump,
                        nu,
                        model,
                        acoustic_rate,
                    )
                    np.testing.assert_allclose(
                        after[cell], expected, rtol=2e-5, atol=8e-9
                    )
                    np.testing.assert_allclose(
                        _C.T @ (after[cell] - before[cell]),
                        density * force,
                        rtol=2e-4,
                        atol=8e-9,
                    )
        self.assertGreater(
            np.max(np.abs(outputs["cm_mrt", 1.0] - outputs["rm_mrt", 1.0])), 1e-6
        )
        self.assertGreater(
            np.max(np.abs(outputs["cm_mrt", 0.25] - outputs["cm_mrt", 1.0])), 1e-6
        )

    def test_moving_pressure_equilibrium_is_a_fixed_point(self):
        for model, acoustic_rate in _COLLISION_CASES:
            s = _CollisionProbe(model, acoustic_rate)
            probes = []
            for index, phi in enumerate((0.0, 0.01, 0.5, 1.0)):
                cell = (4 + 4 * index, 6)
                velocity = np.array([0.025, -0.037])
                density = s._air_density_lattice + phi * (
                    s._water_density_lattice - s._air_density_lattice
                )
                pressure = (-1 if index % 2 else 1) * 4e-5
                equilibrium = _equilibrium(density, pressure, velocity)
                self.assertAlmostEqual(equilibrium.sum(), 0, places=14)
                s.water_phase[cell] = phi
                s.pressure_lattice[cell] = pressure
                s.momentum_velocity_lattice[cell] = velocity
                for q in range(9):
                    s.momentum_populations[cell[0], cell[1], q] = equilibrium[q]
                probes.append(cell)
            before = s.momentum_populations.to_numpy()
            s._collide_momentum()
            after = s.momentum_stream_buffer.to_numpy()
            for cell in probes:
                with self.subTest(model=model, acoustic_rate=acoustic_rate, cell=cell):
                    np.testing.assert_allclose(
                        after[cell], before[cell], rtol=2e-5, atol=8e-9
                    )

    def test_default_high_density_cm_run_survives_water_entry(self):
        # Reusing the near-two shear rate for CM's zeroth and first moments
        # passed isolated collision tests but blew up at step 3268 in this case.
        for options in (
            [],
            [
                "--phase-normal-epsilon",
                "0",
                "--cm-acoustic-relaxation-rate",
                "1",
            ],
        ):
            with self.subTest(options=options):
                self._check_water_entry(options)

    def _check_water_entry(self, options):
        s = IceFlow2D(
            create_config(
                parse_args(["--momentum-collision-model", "cm_mrt", *options])
            )
        )
        initial_center = np.array(s.body_center[None])
        s.step(8192)
        self.assertEqual(s.steps, 8192)
        self.assertAlmostEqual(s.physical_time_s, s.thermal.time_s, places=13)
        self.assertLess(float(s.body_center[None][1]), initial_center[1])
        for field in (
            s.momentum_populations,
            s.phase_populations,
            s.water_phase,
            s.pressure_lattice,
            s.physical_velocity_lattice,
            s.thermal.water_temperature,
        ):
            self.assertTrue(np.isfinite(field.to_numpy()).all())
        self.assertLess(
            np.linalg.norm(s.physical_velocity_lattice.to_numpy(), axis=-1).max(), 0.2
        )

    def test_streamed_shear_wave_decays_at_the_prescribed_viscosity(self):
        # This checks the macroscopic transport coefficient after streaming,
        # independently of the single-collision matrix and stress tests.
        for model, acoustic_rate in _COLLISION_CASES[:3]:
            s = _PeriodicShearProbe(
                _falling_config(
                    momentum_collision_model=model,
                    cm_acoustic_relaxation_rate=acoustic_rate,
                    phase_warmup_steps=0,
                    gravity=(0.0, 0.0),
                    sigma=0.0,
                    viscosity_water=2e-5,
                )
            )
            wave_number = 2 * np.pi / s.ny
            wave = np.sin(wave_number * np.arange(s.ny))
            velocity = np.zeros((s.nx, s.ny, 2), dtype=np.float32)
            velocity[..., 0] = 0.02 + 0.01 * wave
            cu = np.einsum("ija,qa->ijq", velocity, _C)
            populations = _W * (
                3 * cu + 4.5 * cu**2 - 1.5 * np.sum(velocity**2, axis=-1)[..., None]
            )
            s.water_phase.fill(1.0)
            s.phase_populations.from_numpy(
                np.broadcast_to(_W, (s.nx, s.ny, 9)).astype(np.float32).copy()
            )
            s.momentum_velocity_lattice.from_numpy(velocity)
            s.momentum_populations.from_numpy(populations.astype(np.float32))
            s.pressure_lattice.fill(0.0)
            amplitudes = []
            times = []
            for step in range(1, 2049):
                s._collide_momentum()
                s.stream_periodic_momentum()
                s._update_streamed_macroscopic_fields()
                s._update_pressure()
                if step % 512 == 0:
                    ux = s.momentum_velocity_lattice.to_numpy()[..., 0]
                    amplitudes.append(2 * np.mean(ux * wave[None, :]))
                    times.append(step)
            measured_nu = -np.polyfit(times, np.log(amplitudes), 1)[0] / wave_number**2
            with self.subTest(model=model, acoustic_rate=acoustic_rate):
                self.assertAlmostEqual(
                    measured_nu / s._water_viscosity_lattice, 1.0, delta=0.02
                )

    def test_pressure_wave_damping_increases_without_changing_wave_speed(self):
        # Measure a travelling wave after collision AND streaming. A trace-only
        # modification can pass local stress tests while changing the sound speed.
        damping = {}
        for rate in (1.0, 0.25):
            s = _PeriodicShearProbe(
                _falling_config(
                    resolution=(32, 96),
                    momentum_collision_model="cm_mrt",
                    cm_acoustic_relaxation_rate=rate,
                    phase_warmup_steps=0,
                    gravity=(0.0, 0.0),
                    sigma=0.0,
                    viscosity_water=3.1417325796010753e-7,
                )
            )
            k = 2 * np.pi / s.ny
            mode = np.exp(-1j * k * np.arange(s.ny))
            pressure = np.broadcast_to(1e-5 * mode.real, (s.nx, s.ny)).copy()
            velocity = np.zeros((s.nx, s.ny, 2))
            velocity[..., 1] = pressure / np.sqrt(0.6)
            cu = velocity @ _C.T
            populations = _W * (
                3 * pressure[..., None]
                + 3 * cu
                + 4.5 * cu**2
                - 1.5 * np.sum(velocity**2, axis=-1)[..., None]
            )
            populations[..., 0] -= 3 * pressure
            s.water_phase.fill(1.0)
            s.phase_populations.from_numpy(
                np.broadcast_to(_W, (s.nx, s.ny, 9)).astype(np.float32).copy()
            )
            s.momentum_velocity_lattice.from_numpy(velocity.astype(np.float32))
            s.momentum_populations.from_numpy(populations.astype(np.float32))
            s.pressure_lattice.from_numpy(pressure.astype(np.float32))
            amplitudes, times = [], []
            for step in range(1, 1025):
                s._collide_momentum()
                s.stream_periodic_momentum()
                s._update_streamed_macroscopic_fields()
                s._update_pressure()
                if step >= 128 and step % 16 == 0:
                    amplitudes.append(
                        2 * np.mean(s.pressure_lattice.to_numpy() * mode[None, :])
                    )
                    times.append(step)
            amplitudes = np.asarray(amplitudes)
            speed = abs(np.polyfit(times, np.unwrap(np.angle(amplitudes)), 1)[0]) / k
            damping[rate] = -np.polyfit(times, np.log(abs(amplitudes)), 1)[0] / k**2
            with self.subTest(rate=rate):
                self.assertAlmostEqual(speed, np.sqrt(0.6), delta=0.005)
                self.assertGreater(damping[rate], 0)
        self.assertGreater(damping[0.25] / damping[1.0], 5.0)
        self.assertLess(damping[0.25] / damping[1.0], 9.0)


if __name__ == "__main__":
    unittest.main()
