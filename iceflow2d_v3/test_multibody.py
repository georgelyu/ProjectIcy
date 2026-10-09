"""Multi-body geometry, load ownership, heat sharing, and contact regressions."""

from dataclasses import replace
import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceBodyConfig, IceFlow2D
from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as common
from iceflow2d_v3.examples import four_falling_ice_melting_2d as four
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config, _thermal_config


def _two_config(**overrides):
    values = dict(
        phase_warmup_steps=0,
        gravity=(0.0, 0.0),
        sigma=0.0,
        linear_damping=1.0,
        angular_damping=1.0,
        ice_bodies=(
            IceBodyConfig((4, 4), (9.0, 14.0)),
            IceBodyConfig((4, 4), (14.0, 14.0)),
        ),
    )
    values.update(overrides)
    return _falling_config(**values)


@ti.kernel
def _one_node_load(s: ti.template(), i: ti.i32, j: ti.i32):
    # Match the production grid loop: otherwise Taichi promotes the inlined
    # direction loop to a parallel kernel loop and shares its local accumulators.
    for _ in range(1):
        raw = ti.Vector([0.0, 0.13, 0.0, 0.07, 0.0, 0.03, 0.02, 0.01, 0.04])
        result = s._prepare_momentum_cut_links(i, j, raw)
        for q in range(9):
            s.momentum_stream_buffer[i, j, q] = result[q]


class MultiBodyConfigTests(unittest.TestCase):
    def test_four_block_physical_geometry_and_serialization(self):
        c = four.create_config()
        self.assertEqual(c.resolution, (450, 150))
        self.assertAlmostEqual(c.dx, 1.0 / 3000.0)
        self.assertAlmostEqual(c.nx * c.dx, 0.15)
        self.assertAlmostEqual(c.ny * c.dx, 0.05)
        self.assertAlmostEqual(c.water_height * c.dx, 0.025)
        self.assertEqual(len(c.bodies), 4)
        for i, body in enumerate(c.bodies):
            np.testing.assert_allclose(np.array(body.size_cells) * c.dx, (0.01, 0.01))
            np.testing.assert_allclose(
                np.array(body.center_cells) * c.dx, (0.03 * (i + 1), 0.04)
            )
            self.assertAlmostEqual(
                (body.center_cells[1] - body.size_cells[1] / 2) * c.dx - 0.025, 0.01
            )
            self.assertEqual(body.velocity_lattice, (0.0, 0.0))
            self.assertEqual(body.angle_rad, 0.0)
        self.assertEqual(len(c.to_dict()["ice_bodies"]), 4)

    def test_invalid_bodies_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            _two_config(
                ice_bodies=(
                    IceBodyConfig((6, 6), (10, 14), angle_rad=0.5),
                    IceBodyConfig((6, 6), (12, 14), angle_rad=-0.5),
                )
            )
        with self.assertRaisesRegex(ValueError, "wall"):
            _two_config(ice_bodies=(IceBodyConfig((6, 6), (1, 14)),))
        with self.assertRaisesRegex(ValueError, "temperature"):
            _two_config(ice_bodies=(IceBodyConfig((6, 6), (10, 14), temperature_c=1),))
        with self.assertRaisesRegex(ValueError, "ice_bodies"):
            _two_config(ice_bodies=())


class MultiBodyCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_four_block_default_survives_water_entry(self):
        # The original preset failed at 0.1004 s in the gas, while all four
        # bodies and the water-side velocity still looked healthy. Exercise
        # the actual requested grid past both entry and the initial rebound.
        s = IceFlow2D(four.create_config())
        s._thermal_stability_check_enabled = True
        initial_mass = s.body_mass_lattice.array().copy()
        previous_mass = initial_mass.copy()
        for _ in range(35):
            s.step(1200)
            velocity = s.physical_velocity_lattice.to_numpy()
            mass = s.body_mass_lattice.array()
            self.assertTrue(np.isfinite(velocity).all())
            self.assertLess(float(np.abs(velocity).sum(axis=-1).max()), 0.25)
            self.assertTrue(np.all(mass <= previous_mass + 1e-8))
            self.assertTrue(np.all(mass > 0.0))
            previous_mass = mass.copy()
        self.assertEqual(s.steps, 42000)
        self.assertAlmostEqual(s.physical_time_s, 0.35)
        self.assertTrue(np.all(mass < initial_mass * 0.95))
        self.assertTrue(np.all(s.body_center.to_numpy()[:, 1] < 75))

    def test_different_sizes_temperatures_and_independent_integration(self):
        s = IceFlow2D(
            _two_config(
                ice_bodies=(
                    IceBodyConfig((4, 6), (8, 34), temperature_c=-5),
                    IceBodyConfig((6, 8), (23, 34), temperature_c=-12),
                )
            )
        )
        np.testing.assert_allclose(
            s.body_mass_lattice.to_numpy(), 0.917 * np.array([24, 48]), rtol=1e-6
        )
        owners = s.thermal.material_body_id.to_numpy()
        temperature = s.thermal.body_temperature.to_numpy()
        np.testing.assert_allclose(temperature[owners == 0], -5)
        np.testing.assert_allclose(temperature[owners == 1], -12)
        self.assertTrue(np.all(s.thermal.body_solid_mass.to_numpy()[owners < 0] == 0))
        center = s.body_center.to_numpy().copy()
        s.hydrodynamic_impulse[0] = (s.body_mass_lattice[0] * 0.02, 0.0)
        s.hydrodynamic_torque[0] = s.body_inertia_lattice[0] * 0.003
        s._integrate_rigid_ice()
        np.testing.assert_allclose(
            s.body_velocity.to_numpy(), [[0.02, 0], [0, 0]], atol=1e-8
        )
        np.testing.assert_allclose(s.body_center.to_numpy()[1], center[1])
        np.testing.assert_allclose(
            s.body_angular_velocity.to_numpy(), [0.003, 0], atol=1e-9
        )

    def test_explicit_single_body_reproduces_legacy_configuration(self):
        config = _falling_config(phase_warmup_steps=0)
        simulations = [
            IceFlow2D(config),
            IceFlow2D(replace(config, ice_bodies=config.bodies)),
        ]
        for s in simulations:
            s.step(4)
        for name in (
            "water_phase",
            "momentum_velocity_lattice",
            "body_center",
            "body_mass_lattice",
        ):
            np.testing.assert_allclose(
                getattr(simulations[0], name).to_numpy(),
                getattr(simulations[1], name).to_numpy(),
                atol=1e-8,
            )

    def test_capillary_reactions_balance_about_a_common_origin(self):
        from iceflow2d_v3.test_capillary_coupling import _forces, _isolated_config
        from iceflow2d_v3.test_iceflow import _prepare_test_cut_links

        s = IceFlow2D(
            _isolated_config(
                ice_bodies=(
                    IceBodyConfig((8, 10), (32, 48), angle_rad=0.2),
                    IceBodyConfig((8, 10), (48, 52), angle_rad=-0.2),
                )
            )
        )
        x, y = np.meshgrid(np.arange(s.nx) + 0.5, np.arange(s.ny) + 0.5, indexing="ij")
        phase = 0.5 * (
            1 - np.tanh(2 * (np.hypot(x - 42, y - 47) - 11) / s.config.interface_width)
        )
        s.water_phase.from_numpy(phase.astype(np.float32))
        force = _forces(s)
        for name in (
            "momentum_populations",
            "momentum_stream_buffer",
            "pressure_lattice",
            "momentum_velocity_lattice",
            "fluid_acceleration_lattice",
            "hydrodynamic_impulse",
            "hydrodynamic_torque",
        ):
            getattr(s, name).fill(0)
        _prepare_test_cut_links(s)
        loads = s.hydrodynamic_impulse.to_numpy()
        centers = s.body_center.to_numpy()
        torque = s.hydrodynamic_torque.to_numpy().sum()
        torque += np.sum(centers[:, 0] * loads[:, 1] - centers[:, 1] * loads[:, 0])
        torque += np.sum(x * force[..., 1] - y * force[..., 0])
        scale = np.abs(force).sum()
        self.assertTrue(np.all(np.linalg.norm(loads, axis=1) > 1e-6))
        np.testing.assert_allclose(
            force.sum(axis=(0, 1)) + loads.sum(axis=0), 0, atol=2e-6 * scale
        )
        self.assertLess(abs(torque), 2e-6 * scale * s.nx)

    def test_one_fluid_node_routes_links_to_two_independent_bodies(self):
        s = IceFlow2D(_two_config())
        self.assertEqual(s.solid_body_id[10, 13], 0)
        self.assertEqual(s.solid_body_id[12, 13], 1)
        s.momentum_populations.fill(0)
        s.momentum_velocity_lattice.fill(0)
        s.fluid_acceleration_lattice.fill(0)
        s.pressure_lattice.fill(0)
        s.hydrodynamic_impulse.fill(0)
        s.hydrodynamic_torque.fill(0)
        _one_node_load(s, 11, 13)
        before = s.hydrodynamic_impulse.to_numpy().copy()
        self.assertTrue(np.all(np.linalg.norm(before, axis=1) > 0))
        s.body_velocity[0] = (0.01, 0.005)
        s.hydrodynamic_impulse.fill(0)
        s.hydrodynamic_torque.fill(0)
        _one_node_load(s, 11, 13)
        after = s.hydrodynamic_impulse.to_numpy()
        self.assertGreater(np.linalg.norm(after[0] - before[0]), 1e-5)
        np.testing.assert_allclose(after[1], before[1], atol=1e-12)

    def test_shared_water_heat_is_limited_once_and_split_symmetrically(self):
        s = IceFlow2D(_two_config())
        t = s.thermal
        t.water_sensible_energy.fill(0)
        t._body_heat_delta.fill(0)
        available = 1e-7
        t.water_sensible_energy[11, 13] = available
        t._exchange_interface_heat(1.0, s.water_phase, s.wall_mask, s.solid_mask)
        owners = t.material_body_id.to_numpy()
        heat = t._body_heat_delta.to_numpy()
        transferred = [heat[owners == b].sum() for b in range(2)]
        np.testing.assert_allclose(transferred, [available / 2] * 2, rtol=2e-7)
        self.assertGreaterEqual(t.water_sensible_energy.to_numpy().min(), -1e-18)
        self.assertAlmostEqual(
            heat.sum() + t.water_sensible_energy.to_numpy().sum(),
            available,
            delta=1e-18,
        )

    def test_one_body_disappears_without_erasing_the_other(self):
        s = IceFlow2D(_two_config())
        t = s.thermal
        owners = t.material_body_id.to_numpy()
        mass = t.body_solid_mass.to_numpy()
        surviving_mass = mass[owners == 1].sum()
        mass[owners == 0] = 0
        t.body_solid_mass.from_numpy(mass)
        s._refresh_eroded_body()
        np.testing.assert_array_equal(s.body_active.to_numpy(), [0, 1])
        self.assertFalse(np.any(s.solid_body_id.to_numpy() == 0))
        self.assertTrue(np.any(s.solid_body_id.to_numpy() == 1))
        self.assertAlmostEqual(
            t.body_solid_mass.to_numpy()[owners == 1].sum(), surviving_mass
        )
        s.step(2)
        self.assertTrue(np.isfinite(s.body_center.to_numpy()).all())

    def test_unresolved_remnant_freezes_only_its_own_motion(self):
        s = IceFlow2D(_two_config(gravity=(0, -9.8)))
        owners = s.thermal.material_body_id.to_numpy()
        mass = s.thermal.body_solid_mass.to_numpy()
        mass[owners == 0] *= 0.4
        s.thermal.body_solid_mass.from_numpy(mass)
        s._refresh_eroded_body()
        before = s.body_reference_origin.to_numpy().copy()
        s._integrate_rigid_ice()
        np.testing.assert_array_equal(s.body_reference_origin.to_numpy()[0], before[0])
        self.assertLess(s.body_velocity[1][1], 0)
        self.assertGreater(s.body_mass_lattice[0], 0)

    def test_contact_separates_eroded_bodies_and_balances_impulses(self):
        s = IceFlow2D(
            _two_config(
                ice_bodies=(
                    IceBodyConfig((4, 6), (9, 20)),
                    IceBodyConfig((6, 6), (20, 21)),
                )
            )
        )
        s.body_center[1] = (12.5, 21)
        s.body_velocity[0] = (0.1, 0)
        s.body_velocity[1] = (-0.05, 0)
        masses = s.body_mass_lattice.to_numpy()

        def momentum():
            velocities = s.body_velocity.to_numpy()
            centers = s.body_center.to_numpy()
            momenta = masses[:, None] * velocities
            angular = (
                s.body_inertia_lattice.to_numpy() * s.body_angular_velocity.to_numpy()
            ).sum()
            angular += (
                centers[:, 0] * momenta[:, 1] - centers[:, 1] * momenta[:, 0]
            ).sum()
            return momenta.sum(axis=0), angular

        linear, angular = momentum()
        self.assertTrue(s._solve_contact_and_rasterize())
        fractions = s._body_fractions.to_numpy()
        self.assertFalse(np.any((fractions[0] >= 0.5) & (fractions[1] >= 0.5)))
        linear_after, angular_after = momentum()
        np.testing.assert_allclose(linear_after, linear, atol=2e-7)
        self.assertAlmostEqual(angular_after, angular, delta=1e-5)
        self.assertGreater(np.linalg.norm(s.body_angular_velocity.to_numpy()), 1e-5)

    def test_contact_near_wall_and_melted_region_is_not_a_collider(self):
        s = IceFlow2D(
            _two_config(
                body_contact_iterations=1,
                ice_bodies=(
                    IceBodyConfig((4, 4), (4, 14)),
                    IceBodyConfig((4, 4), (12, 14)),
                ),
            )
        )
        s.body_center[1] = (7, 14)
        s._solve_contact_and_rasterize()
        self.assertGreaterEqual(s.body_center[0][0], 4 - 1e-5)
        self.assertEqual(s.contacts.detect(), 0)
        mass = s.thermal.body_solid_mass.to_numpy()
        mass[:2, :] = 0
        s.thermal.body_solid_mass.from_numpy(mass)
        s._refresh_eroded_body()
        # Contact support follows the eroded material, not the original rectangle.
        s._prepare_moving_body_contact_support()
        self.assertGreater(s._body_contact_support_extrema[0, 0], -1.0)

    def test_body_order_and_thermal_scheme_do_not_change_shared_bath(self):
        bodies = (
            IceBodyConfig((4, 4), (9, 14), temperature_c=-5),
            IceBodyConfig((4, 6), (15, 14), temperature_c=-10),
        )
        results = []
        for scheme, order in (
            ("enthalpy_fv", bodies),
            ("enthalpy_fv", bodies[::-1]),
            ("enthalpy_d2q5", bodies),
        ):
            c = _two_config(
                ice_bodies=order, thermal=replace(_thermal_config(), scheme=scheme)
            )
            s = IceFlow2D(c)
            s.step(4)
            results.append(
                (
                    s.water_phase.to_numpy(),
                    s.thermal.water_sensible_energy.to_numpy(),
                    s.body_mass_lattice.to_numpy(),
                )
            )
        np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-7)
        np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-9)
        np.testing.assert_allclose(results[0][2], results[1][2][::-1], atol=1e-8)
        np.testing.assert_allclose(results[0][1], results[2][1], atol=1e-9)
        np.testing.assert_allclose(results[0][2], results[2][2], atol=1e-8)

    def test_multibody_snapshot_csv_npz_and_metadata(self):
        s = IceFlow2D(_two_config())
        initial = common._capture_snapshot(s, initial_solid_volume_cells=32)
        s.step(2)
        final = common._capture_snapshot(s, initial_solid_volume_cells=32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            common.write_history_csv(path / "history.csv", [initial, final])
            common.write_fields_npz(path / "fields.npz", s.config, [initial, final])
            common.write_metadata(
                path / "metadata.json",
                args=common.parse_args([]),
                config=s.config,
                scales=s.scales,
                targets=[0, 2],
                initial=initial,
                final=final,
                snapshot_count=2,
                velocity_sequence=None,
                vorticity_sequence=None,
                temperature_sequence=None,
            )
            with (path / "history.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["body_id"] for row in rows], ["0", "1", "0", "1"])
            with np.load(path / "fields.npz") as fields:
                self.assertEqual(fields["body_center_m"].shape, (2, 2, 2))
                self.assertEqual(fields["body_mass_kg_m"].shape, (2, 2))
                self.assertEqual(fields["solid_body_id"].shape, (2, 48, 32))
                np.testing.assert_allclose(
                    fields["body_center_m"][0, :, 0], np.array([9, 14]) * s.config.dx
                )
            metadata = json.loads((path / "metadata.json").read_text())
            self.assertEqual(metadata["body_count"], 2)
            self.assertEqual(len(metadata["results"]["bodies"]), 2)


if __name__ == "__main__":
    unittest.main()
