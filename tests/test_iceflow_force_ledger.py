"""Force-ledger regressions for IceFlow2D pressure and momentum exchange.

The pure NumPy tests provide an analytic oracle for a future OBB-boundary
pressure integration.  The integration tests use CUDA by default.  A local
developer without a CUDA device may set ``ICEFLOW2D_TEST_CPU=1`` to exercise
the same Taichi kernels on CPU; that escape hatch does not change the
production simulator's CUDA-only policy.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np


_D2Q9_C = np.asarray(
    (
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ),
    dtype=np.int32,
)
_D2Q9_W = np.asarray(
    (
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    ),
    dtype=np.float64,
)


def _integrate_affine_pressure_on_box(
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    pressure_at_center: float,
    pressure_gradient: tuple[float, float],
    segments_per_edge: int = 32,
) -> tuple[np.ndarray, float]:
    """Midpoint-integrate ``-p n`` on all four edges of a rotated box.

    Midpoint quadrature is exact here because pressure is affine along every
    straight edge.  The helper is intentionally independent of Taichi so it
    can be reused as the analytic reference when pressure traction is added
    to the production simulator.
    """

    if segments_per_edge < 1:
        raise ValueError("segments_per_edge must be positive")
    width, height = map(float, size)
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    center_array = np.asarray(center, dtype=np.float64)
    gradient = np.asarray(pressure_gradient, dtype=np.float64)
    half_width = 0.5 * width
    half_height = 0.5 * height
    # Counter-clockwise edges with outward local normals.
    edges = (
        ((-half_width, -half_height), (half_width, -half_height), (0.0, -1.0)),
        ((half_width, -half_height), (half_width, half_height), (1.0, 0.0)),
        ((half_width, half_height), (-half_width, half_height), (0.0, 1.0)),
        ((-half_width, half_height), (-half_width, -half_height), (-1.0, 0.0)),
    )
    force = np.zeros(2, dtype=np.float64)
    torque = 0.0
    for start, end, local_normal in edges:
        start_array = np.asarray(start, dtype=np.float64)
        edge_vector = np.asarray(end, dtype=np.float64) - start_array
        segment_length = np.linalg.norm(edge_vector) / segments_per_edge
        normal = rotation @ np.asarray(local_normal, dtype=np.float64)
        for segment in range(segments_per_edge):
            fraction = (segment + 0.5) / segments_per_edge
            local_point = start_array + fraction * edge_vector
            relative = rotation @ local_point
            point = center_array + relative
            pressure = float(pressure_at_center) + float(
                gradient @ (point - center_array)
            )
            segment_force = -pressure * normal * segment_length
            force += segment_force
            torque += relative[0] * segment_force[1] - relative[1] * segment_force[0]
    return force, torque


class PressureTractionReferenceTests(unittest.TestCase):
    def test_constant_pressure_has_zero_force_and_torque_for_rotated_box(self):
        force, torque = _integrate_affine_pressure_on_box(
            center=(11.25, -3.5),
            size=(4.25, 2.5),
            angle=0.63,
            pressure_at_center=7.3,
            pressure_gradient=(0.0, 0.0),
            segments_per_edge=19,
        )
        np.testing.assert_allclose(force, (0.0, 0.0), atol=2.0e-13)
        self.assertAlmostEqual(torque, 0.0, delta=2.0e-13)

    def test_linear_pressure_force_is_minus_area_times_gradient(self):
        size = (4.25, 2.5)
        gradient = np.asarray((0.017, -0.011), dtype=np.float64)
        expected_force = -size[0] * size[1] * gradient
        for angle in (0.0, 0.37, 1.14):
            with self.subTest(angle=angle):
                force, torque = _integrate_affine_pressure_on_box(
                    center=(11.25, -3.5),
                    size=size,
                    angle=angle,
                    pressure_at_center=2.0,
                    pressure_gradient=tuple(gradient),
                    segments_per_edge=32,
                )
                np.testing.assert_allclose(force, expected_force, atol=3.0e-13)
                self.assertAlmostEqual(torque, 0.0, delta=3.0e-13)

    def test_constant_pressure_offset_does_not_change_linear_pressure_result(self):
        arguments = dict(
            center=(2.0, 3.0),
            size=(3.0, 5.0),
            angle=-0.41,
            pressure_gradient=(-0.02, 0.03),
            segments_per_edge=23,
        )
        force_zero, torque_zero = _integrate_affine_pressure_on_box(
            pressure_at_center=0.0, **arguments
        )
        force_offset, torque_offset = _integrate_affine_pressure_on_box(
            pressure_at_center=100.0, **arguments
        )
        np.testing.assert_allclose(force_offset, force_zero, atol=3.0e-12)
        self.assertAlmostEqual(torque_offset, torque_zero, delta=3.0e-12)


class IceFlowForceLedgerIntegrationTests(unittest.TestCase):
    """Kernel-level pressure/GME tests; CUDA is the normal execution path."""

    @classmethod
    def setUpClass(cls):
        try:
            import taichi as ti
        except ImportError as exc:
            raise unittest.SkipTest("Taichi is not installed") from exc

        import iceflow2d.simulator as simulator_module
        from iceflow2d import create_iceflow_config

        cls.ti = ti
        cls.simulator_module = simulator_module
        cls.create_iceflow_config = staticmethod(create_iceflow_config)
        cls._original_ensure_taichi_cuda = simulator_module.ensure_taichi_cuda
        cls._cpu_override = os.environ.get("ICEFLOW2D_TEST_CPU") == "1"
        if cls._cpu_override:
            runtime = ti.lang.impl.get_runtime()
            if runtime.prog is None:
                ti.init(
                    arch=ti.cpu,
                    default_fp=ti.f32,
                    default_ip=ti.i32,
                    offline_cache=False,
                )
            elif ti.lang.impl.current_cfg().arch != ti.cpu:
                raise unittest.SkipTest(
                    "ICEFLOW2D_TEST_CPU requires an uninitialized or CPU runtime"
                )
            # Test-only override: the production constructor remains CUDA-only.
            simulator_module.ensure_taichi_cuda = lambda: None
        else:
            try:
                simulator_module.ensure_taichi_cuda()
            except RuntimeError as exc:
                raise unittest.SkipTest(f"Taichi CUDA is unavailable: {exc}") from exc
        cls.IceFlow2D = simulator_module.IceFlow2D

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_cpu_override", False):
            cls.simulator_module.ensure_taichi_cuda = cls._original_ensure_taichi_cuda

    @staticmethod
    def _full_pool_fraction(size: int, boundary_cells: int) -> float:
        # Robustly makes int(size * fraction) == size - boundary_cells.
        return (size - boundary_cells + 0.5) / size

    def _make_fully_submerged_config(
        self,
        *,
        resolution=(48, 36),
        scheme="unified",
        gravity=(0.0, -9.8),
    ):
        nx, ny = resolution
        boundary = 3
        return self.create_iceflow_config(
            resolution=resolution,
            reference_length_cells=ny,
            phase_warmup_steps=0,
            water_width_fraction=self._full_pool_fraction(nx, boundary),
            water_height_fraction=self._full_pool_fraction(ny, boundary),
            ice_width_fraction=0.25,
            ice_height_fraction=0.25,
            ice_base_y_cells=14.0,
            gravity=gravity,
            hydrodynamic_force_scale=0.0,
            hydrostatic_buoyancy_scale=0.0,
            linear_damping=1.0,
            angular_damping=1.0,
            rigid_boundary_scheme=scheme,
        )

    @staticmethod
    def _manufacture_pressure_populations(
        simulation, pressure_offset, pressure_gradient
    ):
        nx, ny = simulation.nx, simulation.ny
        center = np.asarray(simulation.body_center[None], dtype=np.float64)
        x = np.arange(nx, dtype=np.float64)[:, None] + 0.5
        y = np.arange(ny, dtype=np.float64)[None, :] + 0.5
        pressure = (
            float(pressure_offset)
            + float(pressure_gradient[0]) * (x - center[0])
            + float(pressure_gradient[1]) * (y - center[1])
        )
        rho = simulation.rho.to_numpy().astype(np.float64)
        populations = _D2Q9_W[None, None, :] * (
            1.0 + 3.0 * pressure[:, :, None] / rho[:, :, None]
        )
        populations = populations.astype(np.float32)
        simulation.f.from_numpy(populations)
        simulation.f_post.from_numpy(populations)
        simulation.p_temp.fill(0.0)
        simulation._update_pressure()
        return pressure, populations

    @staticmethod
    def _set_affine_pressure_field(simulation, pressure_offset, pressure_gradient):
        center = np.asarray(simulation.body_center[None], dtype=np.float64)
        x = np.arange(simulation.nx, dtype=np.float64)[:, None] + 0.5
        y = np.arange(simulation.ny, dtype=np.float64)[None, :] + 0.5
        pressure = (
            float(pressure_offset)
            + float(pressure_gradient[0]) * (x - center[0])
            + float(pressure_gradient[1]) * (y - center[1])
        )
        active = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)
        pressure_field = np.zeros((simulation.nx, simulation.ny), dtype=np.float32)
        pressure_field[active] = pressure[active]
        simulation.p.from_numpy(pressure_field)
        simulation.p_temp.from_numpy(pressure_field)
        return pressure

    @staticmethod
    def _halfway_gme_oracle(simulation, populations):
        solid = simulation.solid.to_numpy()
        wall = simulation.wall.to_numpy()
        phase = np.clip(simulation.phi.to_numpy().astype(np.float64), 0.0, 1.0)
        rho = (
            simulation._rho_air_l
            + (simulation._rho_water_l - simulation._rho_air_l) * phase
        )
        expected = np.zeros(2, dtype=np.float64)
        nx, ny = solid.shape
        for i in range(nx):
            for j in range(ny):
                if wall[i, j] != 0 or solid[i, j] != 0:
                    continue
                for q, direction in enumerate(_D2Q9_C):
                    target_i = i + int(direction[0])
                    target_j = j + int(direction[1])
                    if (
                        0 <= target_i < nx
                        and 0 <= target_j < ny
                        and solid[target_i, target_j] == 1
                    ):
                        outgoing = float(populations[i, j, q])
                        # A stationary halfway bounce-back has reflected=outgoing.
                        expected += (
                            rho[i, j] * direction * (2.0 * outgoing - 2.0 * _D2Q9_W[q])
                        )
        return expected

    @staticmethod
    def _fixed_body_fluid_step(simulation, step_index):
        """Advance only the fluid/phase LBM, leaving rigid pose exactly fixed."""

        simulation._compute_fluid_force(step_index)
        simulation._collide_velocity()
        simulation._collide_phase()
        simulation._clear_stream_targets()
        simulation._stream_velocity()
        simulation._stream_phase()
        simulation._copy_lbm_next()
        simulation._update_fluid_macro()
        simulation._update_pressure()
        simulation._average_pressure()

    def test_manufactured_pressure_populations_drive_halfway_gme(self):
        config = self._make_fully_submerged_config(
            resolution=(96, 64), scheme="halfway", gravity=(0.0, 0.0)
        )
        simulation = self.IceFlow2D(config)

        constant_pressure, constant_populations = (
            self._manufacture_pressure_populations(
                simulation, pressure_offset=0.02, pressure_gradient=(0.0, 0.0)
            )
        )
        active = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)
        np.testing.assert_allclose(
            simulation.p.to_numpy()[active], constant_pressure[active], atol=2.0e-7
        )
        simulation._clear_stream_targets()
        simulation._stream_velocity()
        constant_gme = np.asarray(
            simulation.raw_hydrodynamic_impulse[None], dtype=np.float64
        )
        constant_pressure_mode = np.asarray(
            simulation.pressure_mode_hydrodynamic_impulse[None], dtype=np.float64
        )
        constant_oracle = self._halfway_gme_oracle(simulation, constant_populations)
        np.testing.assert_allclose(constant_gme, constant_oracle, atol=5.0e-6)
        np.testing.assert_allclose(constant_pressure_mode, constant_gme, atol=5.0e-6)
        np.testing.assert_allclose(constant_gme, (0.0, 0.0), atol=2.0e-5)

        gradient = np.asarray((8.0e-5, -1.0e-4), dtype=np.float64)
        linear_pressure, linear_populations = self._manufacture_pressure_populations(
            simulation, pressure_offset=0.02, pressure_gradient=tuple(gradient)
        )
        np.testing.assert_allclose(
            simulation.p.to_numpy()[active], linear_pressure[active], atol=3.0e-7
        )
        simulation._clear_stream_targets()
        simulation._stream_velocity()
        linear_gme = np.asarray(
            simulation.raw_hydrodynamic_impulse[None], dtype=np.float64
        )
        linear_pressure_mode = np.asarray(
            simulation.pressure_mode_hydrodynamic_impulse[None], dtype=np.float64
        )
        linear_oracle = self._halfway_gme_oracle(simulation, linear_populations)
        np.testing.assert_allclose(linear_gme, linear_oracle, rtol=3.0e-4, atol=6.0e-6)
        np.testing.assert_allclose(
            linear_pressure_mode, linear_gme, rtol=3.0e-4, atol=6.0e-6
        )
        self.assertGreater(np.linalg.norm(linear_gme), 2.0e-2)

        simulation._advance_rigid_ice()
        residual = np.asarray(
            simulation.residual_hydrodynamic_impulse[None], dtype=np.float64
        )
        np.testing.assert_allclose(residual, (0.0, 0.0), atol=8.0e-6)

        # The cut-link quadrature is only first-order geometric here, but its
        # response must have the pressure-traction sign and approximate scale.
        continuum_force = -config.ice_width * config.ice_height * gradient
        np.testing.assert_allclose(linear_gme, continuum_force, rtol=0.12, atol=2.0e-4)

    def test_affine_p_temp_forcing_is_visible_in_residual_gme(self):
        """Isolate pressure-gradient forcing after a normal k00=1 collision."""

        config = self._make_fully_submerged_config(
            resolution=(96, 64), scheme="unified", gravity=(0.0, 0.0)
        )
        simulation = self.IceFlow2D(config)
        gradient = np.asarray((8.0e-5, -1.0e-4), dtype=np.float64)
        self._set_affine_pressure_field(
            simulation,
            pressure_offset=0.02,
            pressure_gradient=tuple(gradient),
        )
        simulation._compute_pressure_traction()
        pressure_traction = np.asarray(
            simulation.pressure_traction_impulse[None], dtype=np.float64
        )
        expected_traction = -config.ice_width * config.ice_height * gradient
        np.testing.assert_allclose(
            pressure_traction, expected_traction, rtol=0.08, atol=2.0e-4
        )

        # This is the regular production sequence, not a manufactured f_post:
        # f starts at equilibrium, p_temp drives fluid_force, and collision
        # reconstructs populations with its normal k00=1 constraint.
        simulation._compute_fluid_force(0)
        simulation._collide_velocity()
        active = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)
        post_density_mode = simulation.f_post.to_numpy().sum(axis=2)
        np.testing.assert_allclose(post_density_mode[active], 1.0, atol=8.0e-6)
        simulation._clear_stream_targets()
        simulation._stream_velocity()
        raw_gme = np.asarray(
            simulation.raw_hydrodynamic_impulse[None], dtype=np.float64
        )
        pressure_mode_gme = np.asarray(
            simulation.pressure_mode_hydrodynamic_impulse[None], dtype=np.float64
        )
        simulation._advance_rigid_ice()
        residual_gme = np.asarray(
            simulation.residual_hydrodynamic_impulse[None], dtype=np.float64
        )
        np.testing.assert_allclose(
            residual_gme, raw_gme - pressure_mode_gme, atol=8.0e-6
        )

        traction_norm = float(np.linalg.norm(pressure_traction))
        pressure_mode_fraction = float(
            np.linalg.norm(pressure_mode_gme) / traction_norm
        )
        residual_projection_fraction = float(
            residual_gme @ pressure_traction / (pressure_traction @ pressure_traction)
        )
        transverse_residual_fraction = float(
            np.linalg.norm(
                residual_gme - residual_projection_fraction * pressure_traction
            )
            / traction_norm
        )
        self.assertLess(pressure_mode_fraction, 2.0e-3)
        self.assertGreater(
            residual_projection_fraction,
            0.02,
            msg=(
                "normal k00=1 collision hides affine p_temp from the zero-order "
                "pressure-mode ledger, but its gradient response must remain visible "
                "in residual GME"
            ),
        )
        self.assertLess(residual_projection_fraction, 0.15)
        self.assertLess(transverse_residual_fraction, 0.02)

    def test_pressure_traction_constant_field_is_gauge_invariant(self):
        config = self._make_fully_submerged_config(
            resolution=(96, 64), gravity=(0.0, 0.0)
        )
        simulation = self.IceFlow2D(config)
        expected_boundary_length = 2.0 * (config.ice_width + config.ice_height)
        measured = []
        for pressure_offset in (0.02, 3.02):
            self._set_affine_pressure_field(
                simulation,
                pressure_offset=pressure_offset,
                pressure_gradient=(0.0, 0.0),
            )
            simulation._compute_pressure_traction()
            self.assertEqual(int(simulation.pressure_sample_failures[None]), 0)
            self.assertAlmostEqual(
                float(simulation.pressure_boundary_length[None]),
                expected_boundary_length,
                delta=2.0e-5,
            )
            measured.append(
                (
                    np.asarray(
                        simulation.pressure_traction_impulse[None], dtype=np.float64
                    ),
                    float(simulation.pressure_traction_torque[None]),
                )
            )
        for force, torque in measured:
            np.testing.assert_allclose(force, (0.0, 0.0), atol=2.0e-4)
            self.assertAlmostEqual(torque, 0.0, delta=2.0e-4)
        np.testing.assert_allclose(measured[1][0], measured[0][0], atol=2.0e-4)
        self.assertAlmostEqual(measured[1][1], measured[0][1], delta=2.0e-4)

    def test_pressure_traction_recovers_affine_force_and_rotates_covariantly(self):
        config = self._make_fully_submerged_config(
            resolution=(96, 64), gravity=(0.0, 0.0)
        )
        simulation = self.IceFlow2D(config)
        area = config.ice_width * config.ice_height
        base_gradient = np.asarray((8.0e-5, -1.0e-4), dtype=np.float64)

        def measure(angle, gradient):
            simulation.body_angle[None] = angle
            simulation._rasterize_ice()
            self._set_affine_pressure_field(
                simulation,
                pressure_offset=0.02,
                pressure_gradient=tuple(gradient),
            )
            simulation._compute_pressure_traction()
            self.assertEqual(int(simulation.pressure_sample_failures[None]), 0)
            return (
                np.asarray(
                    simulation.pressure_traction_impulse[None], dtype=np.float64
                ),
                float(simulation.pressure_traction_torque[None]),
            )

        base_force, base_torque = measure(0.0, base_gradient)
        np.testing.assert_allclose(
            base_force, -area * base_gradient, rtol=0.08, atol=2.0e-4
        )
        self.assertLess(
            abs(base_torque), 0.03 * np.linalg.norm(base_force) * config.ice_width
        )

        angle = 0.47
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
        rotated_gradient = rotation @ base_gradient
        rotated_force, rotated_torque = measure(angle, rotated_gradient)
        np.testing.assert_allclose(
            rotated_force,
            -area * rotated_gradient,
            rtol=0.12,
            atol=3.0e-4,
        )
        np.testing.assert_allclose(
            rotated_force,
            rotation @ base_force,
            rtol=0.12,
            atol=3.0e-4,
        )
        self.assertLess(
            abs(rotated_torque),
            0.05 * np.linalg.norm(rotated_force) * config.ice_width,
        )

    def test_bottom_pressure_is_captured_when_horizontal_chord_sees_only_air(self):
        """Reproduce the vertical-entry cavity missed by horizontal chords."""

        config = self._make_fully_submerged_config(resolution=(96, 64))
        simulation = self.IceFlow2D(config)
        center = np.asarray(simulation.body_center[None], dtype=np.float64)
        bottom_y = center[1] - 0.5 * config.ice_height
        y = np.arange(config.ny, dtype=np.float64)[None, :] + 0.5
        active = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)

        # Water/high pressure exists immediately below the ice, while the
        # same-height samples beyond both side faces remain air/zero pressure.
        # This is the characteristic topology of a vertical-entry air cavity.
        water_below = active & (y < bottom_y)
        phase = np.zeros((config.nx, config.ny), dtype=np.float32)
        phase[water_below] = 1.0
        simulation.phi.from_numpy(phase)
        simulation.stored_phi.fill(0.0)
        density = (
            simulation._rho_air_l
            + (simulation._rho_water_l - simulation._rho_air_l) * phase
        )
        simulation.rho.from_numpy(density.astype(np.float32))

        theory_water_buoyancy = (
            -config.ice_width * config.ice_height * simulation._gravity_l[1]
        )
        theory_air_buoyancy = (
            theory_water_buoyancy * simulation._rho_air_l / simulation._rho_water_l
        )
        bottom_pressure = config.ice_height * abs(simulation._gravity_l[1])
        pressure = np.zeros((config.nx, config.ny), dtype=np.float32)
        pressure[water_below] = bottom_pressure
        simulation.p.from_numpy(pressure)
        simulation.p_temp.from_numpy(pressure)

        simulation._compute_hydrostatic_buoyancy()
        legacy_chord_force_y = float(simulation.buoyancy_impulse[None].y)
        displaced_water = float(simulation.displaced_water_volume[None])
        self.assertAlmostEqual(
            legacy_chord_force_y / theory_air_buoyancy,
            1.0,
            delta=0.02,
            msg="left/right air samples should reduce the legacy estimate to air buoyancy",
        )
        self.assertLess(legacy_chord_force_y, 0.01 * theory_water_buoyancy)
        self.assertLess(displaced_water, 1.0e-3 * config.ice_width * config.ice_height)

        simulation._compute_pressure_traction()
        pressure_force = np.asarray(
            simulation.pressure_traction_impulse[None], dtype=np.float64
        )
        self.assertEqual(int(simulation.pressure_sample_failures[None]), 0)
        self.assertGreater(
            pressure_force[1],
            0.80 * theory_water_buoyancy,
            msg="bottom-face pressure must produce a strong upward impulse",
        )
        self.assertAlmostEqual(
            pressure_force[1] / theory_water_buoyancy,
            1.0,
            delta=0.15,
        )
        self.assertLess(abs(pressure_force[0]), 0.03 * theory_water_buoyancy)
        self.assertLess(
            abs(float(simulation.pressure_traction_torque[None])),
            0.03 * theory_water_buoyancy * config.ice_width,
        )

    def test_hydrostatic_pressure_does_not_appear_as_complete_gme_buoyancy(self):
        config = self._make_fully_submerged_config()
        simulation = self.IceFlow2D(config)
        initial_center = np.asarray(simulation.body_center[None], dtype=np.float64)
        raw_gme_samples = []
        pressure_mode_samples = []
        settle_steps = 1300
        sample_steps = 300
        for step_index in range(settle_steps + sample_steps):
            self._fixed_body_fluid_step(simulation, step_index)
            if step_index >= settle_steps:
                raw_gme_samples.append(
                    np.asarray(
                        simulation.raw_hydrodynamic_impulse[None], dtype=np.float64
                    )
                )
                pressure_mode_samples.append(
                    np.asarray(
                        simulation.pressure_mode_hydrodynamic_impulse[None],
                        dtype=np.float64,
                    )
                )

        np.testing.assert_array_equal(
            np.asarray(simulation.body_center[None], dtype=np.float64), initial_center
        )
        np.testing.assert_array_equal(
            np.asarray(simulation.body_velocity[None], dtype=np.float64), (0.0, 0.0)
        )

        pressure = simulation.p_temp.to_numpy().astype(np.float64)
        solid = simulation.solid.to_numpy()
        wall = simulation.wall.to_numpy()
        boundary = config.boundary_cells
        center_x = float(initial_center[0])
        half_width = 0.5 * config.ice_width
        y_indices = np.arange(boundary, config.ny - boundary)
        slopes = []
        normalized_fit_errors = []
        hydrostatic_span = abs(simulation._gravity_l[1]) * len(y_indices)
        for i in range(boundary + 2, config.nx - boundary - 2):
            if abs(i + 0.5 - center_x) <= half_width + 2.0:
                continue
            usable = (wall[i, y_indices] == 0) & (solid[i, y_indices] == 0)
            heights = y_indices[usable].astype(np.float64) + 0.5
            values = pressure[i, y_indices[usable]]
            slope, intercept = np.polyfit(heights, values, 1)
            residual = values - (slope * heights + intercept)
            slopes.append(slope)
            normalized_fit_errors.append(
                float(np.sqrt(np.mean(residual * residual)) / hydrostatic_span)
            )
        mean_slope = float(np.mean(slopes))
        self.assertAlmostEqual(
            mean_slope / simulation._gravity_l[1],
            1.0,
            delta=0.20,
            msg="the fixed-body fluid must first establish a hydrostatic pressure gradient",
        )
        self.assertLess(max(normalized_fit_errors), 0.03)

        theory_buoyancy = (
            -config.ice_width * config.ice_height * simulation._gravity_l[1]
        )
        simulation._compute_hydrostatic_buoyancy()
        explicit_buoyancy = float(simulation.buoyancy_impulse[None].y)
        self.assertAlmostEqual(explicit_buoyancy / theory_buoyancy, 1.0, delta=0.01)

        simulation._compute_pressure_traction()
        self.assertEqual(int(simulation.pressure_sample_failures[None]), 0)
        pressure_traction = np.asarray(
            simulation.pressure_traction_impulse[None], dtype=np.float64
        )
        self.assertAlmostEqual(
            pressure_traction[1] / theory_buoyancy,
            1.0,
            delta=0.20,
            msg="the all-face pressure integral must recover Archimedes' force",
        )
        self.assertLess(abs(pressure_traction[0]), 0.05 * theory_buoyancy)
        self.assertLess(
            abs(float(simulation.pressure_traction_torque[None])),
            0.05 * theory_buoyancy * config.ice_width,
        )

        mean_gme = np.mean(np.asarray(raw_gme_samples), axis=0)
        mean_pressure_mode = np.mean(np.asarray(pressure_mode_samples), axis=0)
        gme_buoyancy_fraction = mean_gme[1] / theory_buoyancy
        # This is the key force-ledger result: although p(y) is hydrostatic,
        # current GME supplies neither the sign nor magnitude of Archimedes'
        # force.  It must not be counted as an already-complete static buoyancy.
        self.assertLess(abs(gme_buoyancy_fraction), 0.25)
        self.assertLess(abs(mean_pressure_mode[1] / theory_buoyancy), 0.25)
        self.assertLess(abs(mean_gme[0]), 0.05 * theory_buoyancy)


if __name__ == "__main__":
    unittest.main()
