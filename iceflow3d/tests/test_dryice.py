"""Open-tank geometry, CO2 properties and independent source/flux balances."""
# ruff: noqa: E741 -- lattice indices

from dataclasses import replace
import os
import unittest

import numpy as np
import taichi as ti

from iceflow3d.checkpoint import _fields
from iceflow3d.config import LatticeScales
from iceflow3d.dryice.config import DryIceConfig, DryIceThermalConfig
from iceflow3d.dryice.materials import co2_density, co2_dynamic_viscosity_pa_s
from iceflow3d.dryice.rigid import tank_boxes
from iceflow3d.dryice.simulator import DryIceFlow3D
from iceflow3d.examples.dambreak_four_ice_melting_3d import create_config as dambreak
from iceflow3d.examples.dry_ice_in_hot_water_3d import build_parser, create_config
from iceflow3d.lattice import DIRECTIONS, direction
from iceflow3d.home import HomeMomentum, reconstruct
from iceflow3d.viscosity import water_dynamic_viscosity_pa_s


@ti.kernel
def seed_uniform(s: ti.template(), phi: ti.f32, ux: ti.f32):
    for I in ti.grouped(s.water_phase):
        if s._fluid(I):
            s.water_phase[I] = phi
            s._set_equilibrium(I, phi, 0.0, ti.Vector([ux, 0.0, 0.0]))


@ti.kernel
def source_probe(s: ti.template(), out: ti.types.ndarray()):
    I = ti.Vector([3, 10, 3])
    for q in ti.static(range(27)):
        out[q] = reconstruct(q, s.home_post[I])


@ti.kernel
def viscosity_probe(s: ti.template(), out: ti.types.ndarray()):
    for i in range(2):
        out[i] = s._kinematic_viscosity(ti.Vector([3 + i, 10, 3]))


@ti.kernel
def viscosity_reference_probe(s: ti.template(), out: ti.types.ndarray()):
    for i in range(2):
        out[i] = s._mixture_kinematic_viscosity(ti.Vector([3 + i, 10, 3]))


@ti.kernel
def fixed_mirror_reference_probe(s: ti.template()) -> ti.i32:
    mismatches = 0
    for I in ti.grouped(s.water_phase):
        if s._fluid(I):
            for q in range(27):
                c = direction(q)
                K = I + c
                R = ti.Matrix.identity(ti.f32, 3)
                for a in ti.static(range(3)):
                    e = ti.Vector.zero(ti.i32, 3)
                    e[a] = c[a]
                    if c[a] != 0 and s._fixed_solid(I + e):
                        K[a], R[a, a] = I[a], -1.0
                if not s._fluid(K):
                    K = I
                actual_K, actual_R = s._fixed_mirror(I, c)
                if (K != actual_K).any() or (R != actual_R).any():
                    mismatches += 1
    return mismatches


@ti.kernel
def open_face_index_probe(
    s: ti.template(), inset: ti.template(), out: ti.types.ndarray()
):
    for k in range(out.shape[0]):
        I = s._open_face_index(k, inset)
        for a in ti.static(range(3)):
            out[k, a] = I[a]


class DryIceConfigurationTests(unittest.TestCase):
    def test_geometry_and_dambreak_scales(self):
        c = create_config()
        self.assertEqual(c, DryIceConfig())
        box = np.array(c.tank_upper_cells) - c.tank_lower_cells
        domain = np.array((c.nx, c.ny, c.nz)) - 2
        self.assertEqual(box[0] * box[2] / (domain[0] * domain[2]), 0.25)
        self.assertEqual(box[1] / domain[1], 1 / 3)
        self.assertEqual(len(tank_boxes(c)), 5)
        self.assertEqual(c.interface_width, 5)
        self.assertEqual(c.phase_normal_epsilon, 1e-6)
        self.assertEqual(dambreak().phase_normal_epsilon, 0.01)
        self.assertAlmostEqual(
            LatticeScales.from_iceflow_config(c).dt_s,
            LatticeScales.from_iceflow_config(dambreak()).dt_s,
        )
        self.assertEqual(c.thermal.properties.melting_temperature_c, -78.5)
        self.assertEqual(dambreak().thermal.properties.melting_temperature_c, 0)
        self.assertEqual(len(dambreak().bodies), 4)

    def test_reject_unsupported_parameters(self):
        c = create_config()
        for changes in (
            {"interface_width": 4},
            {"boundary_cells": 2},
            {"film_thickness_m": 0},
            {"water_surface_cells": 74},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(c, **changes)
        with self.assertRaises(ValueError):
            DryIceThermalConfig(initial_ice_temperature_c=-50)
        with self.assertRaises(ValueError):
            DryIceThermalConfig(initial_water_temperature_c=-5)

    def test_co2_viscosity_published_check_table(self):
        # Fenghour et al. (1998), Appendix IV; values in micro-Pa s.
        # Exclude the 304 K near-critical point: that entry additionally uses
        # the critical-enhancement model, irrelevant to atmospheric CO2.
        for t, rho, expected in (
            (220, 2.440, 11.06),
            (300, 1.773, 15.02),
            (800, 0.662, 35.09),
            (220, 1194.86, 269.37),
            (300, 1029.27, 132.55),
            (800, 407.828, 48.74),
        ):
            with self.subTest(t=t, rho=rho):
                self.assertAlmostEqual(
                    co2_dynamic_viscosity_pa_s(t - 273.15, rho) * 1e6,
                    expected,
                    delta=0.006,
                )
        self.assertAlmostEqual(co2_density(20), 1.82953, delta=1e-5)


@ti.data_oriented
class ReferenceDryIceFlow3D(DryIceFlow3D):
    # Taichi kernels must be accessed through a data-oriented instance so its
    # descriptor binds the receiver; calling Base.kernel(instance) is invalid.
    _reference_stream_home_bulk = HomeMomentum._stream_home_bulk
    _reference_stream_home_boundary = HomeMomentum._stream_home_boundary


class DryIceNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = build_parser().parse_args(
            [
                "--resolution",
                "24",
                "--phase-warmup-steps",
                "0",
                "--initial-position",
                "submerged",
                "--fixed-body",
                "--backend",
                os.environ.get("ICEFLOW3D_TEST_BACKEND", "cpu"),
            ]
        )
        cls.s = ReferenceDryIceFlow3D(create_config(args))
        cls.saved = {k: v.to_numpy() for k, v in _fields(cls.s).items()}
        cls.rigid_saved = {
            k: v.copy()
            for k, v in vars(cls.s.rigid).items()
            if isinstance(v, np.ndarray)
        }

    def setUp(self):
        for k, v in self.saved.items():
            getattr(self.s, k).from_numpy(v)
        for k, v in self.rigid_saved.items():
            setattr(self.s.rigid, k, v.copy())
        self.s.steps = self.s._thermal_pending = 0
        self.s.invalidate_gradient_cache()

    def test_central_slice_matches_full_snapshot_in_physical_units(self):
        from iceflow3d.dryice.slices import central_z_slice

        s = self.s
        seed_uniform(s, 0.75, 0.0125)
        # Retain both phases and a nonuniform thermal state to check masking.
        phase = s.water_phase.to_numpy()
        phase[: s.nx // 2] = 0.1
        s.water_phase.from_numpy(phase)
        energy = s.water_energy.to_numpy()
        volume = s.water_volume.to_numpy()
        energy[:] = -volume * s.rho_cp * 2.5
        s.water_energy.from_numpy(energy)
        fields, sliced = s.snapshot(), central_z_slice(s)
        z = s.nz // 2
        np.testing.assert_array_equal(
            sliced["water_phase"], fields["water_phase"][1:-1, 1:-1, z].T
        )
        np.testing.assert_array_equal(
            sliced["solid_body_id"], fields["solid_body_id"][1:-1, 1:-1, z].T
        )
        np.testing.assert_allclose(
            sliced["velocity_m_s"],
            fields["velocity_m_s"][1:-1, 1:-1, z].transpose(1, 0, 2),
            atol=1e-7,
        )
        visible = sliced["solid_body_id"] != -2
        np.testing.assert_allclose(
            sliced["temperature_c"][visible],
            fields["temperature_c"][1:-1, 1:-1, z].T[visible],
            atol=1e-5,
        )
        self.assertTrue(np.isnan(sliced["temperature_c"][~visible]).all())
        self.assertAlmostEqual(sliced["z_m"], (z - 0.5) * s.dx)
        self.assertAlmostEqual(sliced["x_m"][0], s.dx / 2)
        self.assertEqual(sliced["velocity_m_s"].shape, (s.ny - 2, s.nx - 2, 3))

    def test_adiabatic_hollow_tank_and_five_open_faces(self):
        s = self.s
        ids = s.solid_body_id.to_numpy()
        self.assertTrue((ids[:, 0, :] == -2).all())
        self.assertTrue((ids[0, 1:, :] == -4).all())
        self.assertTrue((ids[-1, 1:, :] == -4).all())
        self.assertTrue((ids[:, -1, :] == -4).all())
        self.assertTrue((ids[:, 1:, 0] == -4).all())
        self.assertTrue((ids[:, 1:, -1] == -4).all())
        lo, hi = map(
            lambda x: np.array(x, int),
            (s.config.tank_lower_cells, s.config.tank_upper_cells),
        )
        self.assertEqual(ids[lo[0], 3, 12], -2)
        self.assertEqual(ids[lo[0] + 2, 3, 12], -1)
        self.assertEqual(ids[12, hi[1], 12], -1)  # open rim, no lid
        # Even a temperature jump across the wall cannot conduct through it.
        volume = np.where(ids == -1, s.cell_volume, 0).astype(np.float32)
        energy = np.zeros(s.shape, np.float32)
        energy[lo[0] - 1, 3, 12] = -volume[lo[0] - 1, 3, 12] * s.rho_cp * 20
        energy[lo[0] + 1, 3, 12] = -volume[lo[0] + 1, 3, 12] * s.rho_cp * 40
        s.water_volume.from_numpy(volume)
        s.water_energy.from_numpy(energy)
        s._thermal_fluxes(False)
        flux = s.energy_flux.to_numpy()
        self.assertEqual(flux[lo[0] - 1, 3, 12, 0], 0)
        self.assertEqual(flux[lo[0], 3, 12, 0], 0)
        self.assertNotEqual(flux[lo[0] - 2, 3, 12, 0], 0)

    def test_physical_viscosities_reach_collision_without_a_floor(self):
        s = self.s
        phi = s.water_phase.to_numpy()
        phi[3, 10, 3], phi[4, 10, 3] = 0, 1
        s.water_phase.from_numpy(phi)
        V = s.water_volume.to_numpy()
        V[4, 10, 3] = s.cell_volume
        s.water_volume.from_numpy(V)
        E = s.water_energy.to_numpy()
        E[4, 10, 3] = s.cell_volume * s.rho_cp * (60 - s.water_reference_temperature)
        s.water_energy.from_numpy(E)
        out = np.empty(2, np.float32)
        viscosity_probe(s, out)
        expected = (
            np.array(
                [
                    co2_dynamic_viscosity_pa_s(20, s.config.rho_air) / s.config.rho_air,
                    water_dynamic_viscosity_pa_s(60) / s.config.rho_water,
                ]
            )
            * s.scales.dt_s
            / s.dx**2
        )
        np.testing.assert_allclose(out, expected, rtol=2e-5)

    def test_cached_fixed_mirrors_match_geometry_at_faces_edges_and_corners(self):
        self.assertEqual(fixed_mirror_reference_probe(self.s), 0)

    def test_five_face_enumeration_covers_every_edge_and_corner_once(self):
        s = self.s
        x, y, z = np.indices(s.shape)
        for inset in (0, 1):
            indices = np.empty((s._open_face_count(inset), 3), np.int32)
            open_face_index_probe(s, inset, indices)
            counts = np.zeros(s.shape, np.int32)
            np.add.at(counts, tuple(indices.T), 1)
            inside = (
                (x >= inset)
                & (x < s.nx - inset)
                & (y >= 1)
                & (y < s.ny - inset)
                & (z >= inset)
                & (z < s.nz - inset)
            )
            expected = inside & (
                (x == inset)
                | (x == s.nx - 1 - inset)
                | (y == s.ny - 1 - inset)
                | (z == inset)
                | (z == s.nz - 1 - inset)
            )
            np.testing.assert_array_equal(counts, expected.astype(np.int32))

    def test_moving_geometry_keeps_cached_force_classification_current(self):
        s = self.s
        s._gradients()
        s.rigid.origin += np.array([0.8, 0.35, 0.25])
        s.rigid.upload()
        s._refresh_geometry()
        previous, current = s.previous_body_id.to_numpy(), s.solid_body_id.to_numpy()
        self.assertTrue(np.any((previous == -1) & (current != -1)))
        self.assertTrue(np.any((previous != -1) & (current == -1)))
        cached = {
            name: getattr(s, name).to_numpy()
            for name in ("_home_boundary", "_home_moving_neighbor")
        }
        s._forces(True, moment_pressure=True, classified=True)
        expected = s.home_acceleration.to_numpy()
        s._forces(True, moment_pressure=True)
        for name, values in cached.items():
            np.testing.assert_array_equal(getattr(s, name).to_numpy(), values)
        np.testing.assert_array_equal(s.home_acceleration.to_numpy(), expected)

    def test_hydrostatic_head_below_submerged_body_matches_adjacent_water(self):
        p = self.s.pressure_lattice.to_numpy()
        self.assertGreater(p[12, 2, 12], 0)
        self.assertAlmostEqual(float(p[12, 2, 12]), float(p[9, 2, 9]), delta=1e-10)

    def test_viscosity_keeps_trace_liquid_in_the_dry_thermal_region(self):
        s = self.s
        actual, reference = np.empty(2, np.float32), np.empty(2, np.float32)
        for fractions in ((1e-9, 1e-7), (1e-5, 1e-3)):
            for i, phi in enumerate(fractions):
                s.water_phase[3 + i, 10, 3] = phi
                s.water_volume[3 + i, 10, 3] = 0
            viscosity_probe(s, actual)
            viscosity_reference_probe(s, reference)
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=0)

    def test_aa_open_boundary_water_balance_at_both_parities(self):
        s = self.s
        s.phase_gradient_lattice.fill(0)
        s.home_acceleration.fill(0)
        s.phase_parity[None] = 0
        seed_uniform(s, 0.4, 0.015)
        fluid = s.solid_body_id.to_numpy() == -1
        before = s.water_phase.to_numpy()[fluid].sum(dtype=float)
        s.water_outflow_lu[None] = 0
        s.phase_clip_volume[None] = 0
        for i in range(6):
            s._fill_open_halo()
            s._collide_phase(moving_walls=False)
            s._recover_phase()
            after = s.water_phase.to_numpy()[fluid].sum(dtype=float)
            residual = (
                after
                - before
                + float(s.water_outflow_lu[None])
                - float(s.phase_clip_volume[None])
            )
            self.assertAlmostEqual(residual, 0, delta=0.003, msg=f"parity {i % 2}")
        self.assertGreater(float(s.water_outflow_lu[None]), 0)

    def test_open_thermal_flux_removes_heat_without_heating_the_halo(self):
        s = self.s
        s.water_volume.fill(0)
        s.water_energy.fill(0)
        s.velocity_lattice.fill([0.02, 0, 0])
        I = (s.nx - 2, 12, 12)
        s.water_volume[I] = s.cell_volume
        s.water_energy[I] = 2.0
        s._thermal_fluxes(True)
        s._apply_thermal_fluxes(s.scales.dt_s, True)
        self.assertGreater(float(s.thermal_open_energy[None]), 0)
        self.assertAlmostEqual(
            float(s.water_energy.to_numpy().sum(dtype=float))
            + float(s.thermal_open_energy[None]),
            2,
            delta=2e-7,
        )
        self.assertEqual(float(s.water_energy[s.nx - 1, 12, 12]), 0)

    def test_thermal_update_matches_face_divergence_including_dry_cells(self):
        s = self.s
        rng = np.random.default_rng(27)
        ids = s.solid_body_id.to_numpy()
        fluid, opened = ids == -1, ids == -4
        volume = np.where(
            fluid & (rng.random(s.shape) < 0.2),
            s.cell_volume * rng.uniform(0.1, 1, s.shape),
            0,
        ).astype(np.float32)
        energy = (-volume * s.rho_cp * rng.uniform(0, 40, s.shape)).astype(np.float32)
        s.velocity_lattice.from_numpy(
            rng.uniform(-0.02, 0.02, (*s.shape, 3)).astype(np.float32)
        )
        for advect in (True, False):
            with self.subTest(advect=advect):
                s.water_volume.from_numpy(volume)
                s.water_energy.from_numpy(energy)
                s.thermal_open_energy[None] = 0
                s._thermal_fluxes(advect)
                ef, vf = s.energy_flux.to_numpy(), s.volume_flux.to_numpy()
                de, dv, boundary = (np.zeros(s.shape) for _ in range(3))
                for a in range(3):
                    incoming = np.roll(ef[..., a], 1, axis=a)
                    de += ef[..., a] - incoming
                    dv += vf[..., a] - np.roll(vf[..., a], 1, axis=a)
                    boundary += ef[..., a] * np.roll(
                        opened, -1, axis=a
                    ) - incoming * np.roll(opened, 1, axis=a)
                expected_e, expected_v = energy.copy(), volume.copy()
                expected_e[fluid] -= s.scales.dt_s * de[fluid]
                if advect:
                    expected_v[fluid] -= s.scales.dt_s * dv[fluid]
                s._apply_thermal_fluxes(s.scales.dt_s, advect)
                np.testing.assert_allclose(
                    s.water_energy.to_numpy(), expected_e, rtol=3e-6, atol=1e-9
                )
                np.testing.assert_allclose(
                    s.water_volume.to_numpy(), expected_v, rtol=2e-6, atol=1e-16
                )
                np.testing.assert_allclose(
                    float(s.thermal_open_energy[None]),
                    s.scales.dt_s * boundary[fluid].sum(),
                    rtol=5e-6,
                    atol=1e-8,
                )

    def test_specialized_stream_matches_generic_at_all_boundary_types(self):
        s = self.s
        rng = np.random.default_rng(216)
        home = rng.uniform(-0.01, 0.01, (*s.shape, 10)).astype(np.float32)
        post = rng.uniform(-0.01, 0.01, (*s.shape, 10)).astype(np.float32)
        velocity = rng.uniform(-0.01, 0.01, (*s.shape, 3)).astype(np.float32)
        s.home_post.from_numpy(post)
        s.old_velocity.from_numpy(velocity)
        s.home_acceleration.fill([1e-5, -2e-5, 3e-5])
        s.body_velocity[0] = [0.003, -0.002, 0.001]
        s.body_angular_velocity[0] = [0.0001, 0.0002, -0.0001]
        s._classify_home()
        flags = s._home_boundary.to_numpy()
        moving = s._home_moving_neighbor.to_numpy()
        self.assertTrue(np.any(flags == 0))
        self.assertTrue(np.any((flags == 2) & (moving == 0)))
        self.assertTrue(np.any((flags == 2) & (moving != 0)))

        def reset_inputs():
            s.home.from_numpy(home)
            s.velocity_lattice.from_numpy(velocity)
            s.open_volume_flux_cell.fill(0)
            s.hydrodynamic_force.fill(0)
            s.hydrodynamic_torque.fill(0)

        reset_inputs()
        s._reference_stream_home_bulk()
        s._reference_stream_home_boundary()
        names = (
            "home",
            "velocity_lattice",
            "open_volume_flux_cell",
            "hydrodynamic_force",
            "hydrodynamic_torque",
        )
        reference = {name: getattr(s, name).to_numpy() for name in names}
        reset_inputs()
        s._stream_home(classified=True)
        for name in names:
            with self.subTest(field=name):
                np.testing.assert_allclose(
                    getattr(s, name).to_numpy(), reference[name], rtol=2e-5, atol=2e-7
                )

    def test_sublimation_heat_mass_and_source_momentum_balance(self):
        s = self.s
        np.testing.assert_array_equal(s.material_energy.to_numpy(), 0)
        np.testing.assert_array_equal(s.water_energy.to_numpy(), 0)
        s.body_velocity[0] = [0.01, -0.02, 0.005]
        initial = s.material_mass.to_numpy().sum(dtype=float)
        energy = s.water_energy.to_numpy().sum(dtype=float)
        s._sublimate(s.scales.dt_s)
        emitted = float(s.sublimated_mass[None])
        heat = float(s.heat_from_water[None])
        self.assertGreater(emitted, 0)
        self.assertGreater(heat, 0)
        self.assertAlmostEqual(
            initial - s.material_mass.to_numpy().sum(dtype=float),
            emitted,
            delta=initial * 2e-7,
        )
        self.assertAlmostEqual(
            s.gas_mass_step.to_numpy().sum(dtype=float), emitted, delta=emitted * 2e-6
        )
        np.testing.assert_allclose(
            s.gas_momentum_step.to_numpy().sum(axis=(0, 1, 2), dtype=float),
            emitted * np.array([0.01, -0.02, 0.005]),
            rtol=3e-6,
        )
        self.assertAlmostEqual(
            heat + float(s.heat_from_reservoir[None]),
            emitted * s.config.effective_sublimation_enthalpy,
            delta=heat * 2e-6,
        )
        self.assertAlmostEqual(
            energy - s.water_energy.to_numpy().sum(dtype=float), heat, delta=0.001
        )

    def test_sub_ulp_sublimation_is_retained_and_runtime_setting_restored(self):
        s = self.s
        fast_math = ti.lang.impl.current_cfg().fast_math
        before = s.material_mass.to_numpy().astype(float)
        s._sublimate(s.scales.dt_s * 1e-5)
        self.assertEqual(ti.lang.impl.current_cfg().fast_math, fast_math)
        emitted = s.gas_mass_step.to_numpy().sum(dtype=float)
        self.assertGreater(emitted, 0)
        corrected_loss = (
            before - s.material_mass.to_numpy()
        ).sum() + s.sublimation_mass_remainder.to_numpy().sum(dtype=float)
        self.assertAlmostEqual(corrected_loss, emitted, delta=emitted * 3e-6)

    def test_pressure_source_has_required_volume_and_zero_momentum(self):
        s = self.s
        s.home.fill(0)
        s.home_acceleration.fill(0)
        s.velocity_lattice.fill(0)
        s.gas_mass_step.fill(0)
        dm = 1e-10
        s.gas_mass_step[3, 10, 3] = dm
        s._collide_momentum()
        out = np.empty(27, np.float32)
        source_probe(s, out)
        expected = dm / s.cell_volume * (1 / s.config.rho_air - 1 / s.config.rho_ice)
        self.assertAlmostEqual(out.sum(dtype=float), expected, delta=expected * 2e-7)
        np.testing.assert_allclose(np.array(DIRECTIONS).T @ out, 0, atol=1e-10)

    def test_prescribed_gas_source_closes_discrete_open_volume_budget(self):
        s = self.s
        s.home_acceleration.fill(0)
        s.water_volume.fill(0)
        s.water_energy.fill(0)
        s.phase_parity[None] = 0
        seed_uniform(s, 0, 0)
        source_lu = 1e-4
        dm = source_lu * s.cell_volume / (1 / s.config.rho_air - 1 / s.config.rho_ice)

        def prescribed_source(dt):
            s.gas_mass_step[3, 10, 3] = dm

        s._sublimate = prescribed_source
        try:
            s.step(200)
        finally:
            del s._sublimate
        fluid = s.solid_body_id.to_numpy() == -1
        storage = s.home.to_numpy()[..., 0][fluid].sum(dtype=float)
        outflow = s.open_volume_flux_cell.to_numpy().sum(dtype=float)
        self.assertGreater(outflow, 0)
        self.assertAlmostEqual(storage + outflow, 200 * source_lu, delta=2e-6)

    def test_cooling_is_resolved_and_unsupported_freezing_is_rejected(self):
        s = self.s
        wet = np.argwhere(s.water_phase.to_numpy() > 0.9)[0]
        I = tuple(wet)
        s.water_energy[I] = (
            s.rho_cp * float(s.water_volume[I]) * (-1 - s.water_reference_temperature)
        )
        with self.assertRaisesRegex(RuntimeError, "freezing"):
            s._advance_conduction(0)

    def test_tank_contact_stops_body_at_inner_floor(self):
        r, s = self.s.rigid, self.s
        floor = s.config.tank_lower_cells[1] + s.config.tank_wall_cells
        r.origin[0, 1] = floor - r.lower[0, 1] - 0.1
        r.velocity[0] = [0, -0.01, 0]
        r.resolve_contacts()
        self.assertAlmostEqual(r.origin[0, 1] + r.lower[0, 1], floor, places=7)
        self.assertGreaterEqual(r.velocity[0, 1], -1e-8)
        self.assertLess(np.linalg.norm(r.omega[0]), 1e-8)


if __name__ == "__main__":
    unittest.main()
