from dataclasses import replace
import json
import tempfile
from pathlib import Path
import struct
import unittest
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from iceflow3d import IceFlowConfig, LatticeScales, ThermalConfig
from iceflow3d.examples.dambreak_four_ice_melting_3d import (
    build_parser,
    create_config,
    snapshot_targets,
)
from iceflow3d.rigid import (
    RigidBodies,
    advance_quaternion,
    apply_contact_impulse,
    apply_wall_contact_manifold,
    rotation_matrix,
    separating_axis,
)
from iceflow3d.reporting import write_vti, write_pvd


class ConfigurationTests(unittest.TestCase):
    def test_default_four_cubes_and_units(self):
        c = create_config()
        s = LatticeScales.from_iceflow_config(c)
        self.assertEqual((c.nx, c.ny, c.nz), (450, 150, 90))
        self.assertEqual(c.floor_water_depth_cells, 31)
        self.assertEqual((c.water_width, c.water_height), (198, 138))
        self.assertEqual(len(c.bodies), 4)
        np.testing.assert_allclose(
            [b.center_cells for b in c.bodies],
            [[255, 21, 45], [309, 21, 45], [363, 21, 45], [417, 21, 45]],
        )
        self.assertAlmostEqual(s.dx_m, 1 / 3000)
        self.assertAlmostEqual(s.dt_s, 1 / 120000)
        self.assertAlmostEqual(s.mass_kg, 1000 * (1 / 3000) ** 3)
        self.assertEqual(c.density_gradient_source_limit, 0.0)
        self.assertEqual(c.phase_flux_acceleration_limit, 0.0)
        self.assertFalse(c.gas_state_repair)

    def test_coarse_grid_is_cubic(self):
        args = build_parser().parse_args(["--resolution-x", "150"])
        c = create_config(args)
        self.assertEqual((c.nx, c.ny, c.nz), (150, 50, 30))
        self.assertEqual(c.bodies[0].size_cells, (10, 10, 10))
        self.assertEqual(c.boundary_cells, 1)

    def test_reject_non_cubic_or_invalid_configuration(self):
        c = create_config()
        for kwargs in (
            {"nz": 91},
            {"rho_water": float("nan")},
            {"phase_normal_epsilon": -1},
            {"density_gradient_source_limit": 0.1},
            {"cm_acoustic_relaxation_rate": 2},
            {"ice_bodies": (c.bodies[0], c.bodies[0])},
            {"gravity": (0, 0, -9.8)},
            {"boundary_cells": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(c, **kwargs)

    def test_reject_ice_outside_depth_and_hot_ice(self):
        c = create_config()
        with self.assertRaisesRegex(ValueError, "wall"):
            replace(c, ice_bodies=(replace(c.bodies[0], center_cells=(255, 21, 4)),))
        with self.assertRaises(ValueError):
            ThermalConfig(initial_ice_temperature_c=1)
        with self.assertRaises(ValueError):
            ThermalConfig(max_fourier_number=0.2)

    def test_rotation_checks_all_three_dimensions(self):
        args = build_parser().parse_args(["--ice-angle-degrees", "10", "15", "5"])
        c = create_config(args)
        r = rotation_matrix(c.bodies[0].quaternion_wxyz)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        self.assertGreater(c.bodies[0].center_cells[1], 21)

    def test_targets_include_zero_final_and_partial_interval(self):
        args = build_parser().parse_args(
            ["--steps", "17", "--output-interval-s", ".0001"]
        )
        self.assertEqual(snapshot_targets(create_config(args), args), [0, 12, 17])
        args.steps = 0
        self.assertEqual(snapshot_targets(create_config(args), args), [0])
        args.steps = -1
        with self.assertRaises(ValueError):
            snapshot_targets(create_config(args), args)


class RigidTests(unittest.TestCase):
    def test_fine_grid_coupled_contact_converges_with_default_budget(self):
        # Actual four-body/wall state at fine-grid step 15963. The old
        # 32-sweep cap leaves ~1e-6 cells of penetration; no tolerance changes.
        state = json.loads(
            (Path(__file__).parent / "fixtures/fine_grid_coupled_contact.json").read_text()
        )

        def make(budget):
            r = RigidBodies.__new__(RigidBodies)
            r.s = SimpleNamespace(
                shape=tuple(state["shape"]), body_count=state["body_count"],
                steps=state["source_step"] - 1,
                config=SimpleNamespace(
                    boundary_cells=state["boundary_cells"], body_contact_iterations=budget
                ),
            )
            for name, value in state["arrays"].items():
                setattr(r, name, np.array(value))
            return r

        def energy(r):
            result = 0.0
            for b in range(r.s.body_count):
                rotation = rotation_matrix(r.q[b])
                inertia = rotation @ r.inertia[b] @ rotation.T
                result += (r.mass[b] * r.velocity[b] @ r.velocity[b]
                           + r.omega[b] @ inertia @ r.omega[b]) / 2
            return result

        old = make(32)
        with self.assertRaisesRegex(RuntimeError, "contact projection did not converge"):
            old.resolve_contacts()
        repaired = make(IceFlowConfig().body_contact_iterations)
        before = energy(repaired)
        repaired.resolve_contacts()
        self.assertLess(repaired.contact_residuals()[0], 1e-7)
        self.assertLess(repaired.contact_residuals()[1], 1e-7)
        self.assertLessEqual(energy(repaired), before)
        larger = make(256)
        larger.resolve_contacts()
        for name in ("origin", "q", "velocity", "omega"):
            np.testing.assert_array_equal(getattr(repaired, name), getattr(larger, name))

    def late_floor_contact(self, iterations):
        # Captured before contact resolution in a restart of the 8 s case.
        state = json.loads(
            (Path(__file__).parent / "fixtures/late_floor_contact.json").read_text()
        )
        r = RigidBodies.__new__(RigidBodies)
        r.s = SimpleNamespace(
            shape=tuple(state.pop("shape")),
            body_count=1,
            steps=state.pop("source_step") - 1,
            config=SimpleNamespace(
                boundary_cells=1, body_contact_iterations=iterations
            ),
        )
        for name, value in state.items():
            setattr(r, name, np.array(value))
        return r

    def test_final_contact_sweep_uses_actual_constraint_residual(self):
        r = self.late_floor_contact(32)
        before_v, before_w = r.velocity.copy(), r.omega.copy()
        rotation = rotation_matrix(r.q[0])
        inertia = rotation @ r.inertia[0] @ rotation.T
        energy_before = (
            r.mass[0] * before_v[0] @ before_v[0] + before_w[0] @ inertia @ before_w[0]
        ) / 2
        r.resolve_contacts()
        penetration, closing = r.contact_residuals()
        self.assertLess(penetration, 1e-7)
        self.assertLess(closing, 1e-7)
        self.assertLess(
            (
                r.mass[0] * r.velocity[0] @ r.velocity[0]
                + r.omega[0] @ inertia @ r.omega[0]
            )
            / 2,
            energy_before,
        )
        np.testing.assert_array_equal(r.velocity[:, [0, 2]], before_v[:, [0, 2]])
        before = {k: v.copy() for k, v in vars(r).items() if isinstance(v, np.ndarray)}
        r.contact_residuals()
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(r, name), value)

    def test_contact_budget_still_rejects_unresolved_penetration(self):
        r = self.late_floor_contact(1)
        r.s.shape = (150, 8, 30)  # The remnant cannot fit between floor and lid.
        with self.assertRaisesRegex(RuntimeError, "closing_speed="):
            r.resolve_contacts()
        self.assertGreater(r.contact_residuals()[0], 1e-7)

    def test_low_inertia_remnant_uses_coupled_wall_impulses(self):
        r = self.late_floor_contact(32)
        r.velocity *= 10
        r.omega *= 10
        r.resolve_contacts()
        penetration, closing = r.contact_residuals()
        self.assertLess(penetration, 1e-7)
        self.assertLess(closing, 1e-12)

    def test_wall_manifold_is_passive_and_order_independent(self):
        normal = np.array([0.0, 1.0, 0.0])
        arms = np.array([[-5.5, -4.5, -5.0], [-5.5, -4.5, 5.0], [4.5, -4.5, -5.0]])
        inertia = np.diag([1.8, 2.6, 2.9])
        v0, w0 = np.array([0.02, -1e-4, -0.01]), np.array([1e-5, 2e-5, -3e-5])
        energy = (v0 @ v0 + w0 @ inertia @ w0) / 2
        results = []
        for points in (arms, arms[::-1]):
            v, w = v0.copy(), w0.copy()
            apply_wall_contact_manifold(
                v, w, 1.0, np.linalg.inv(inertia), points, normal
            )
            self.assertGreaterEqual(((v + np.cross(w, arms)) @ normal).min(), -1e-12)
            self.assertLessEqual((v @ v + w @ inertia @ w) / 2, energy)
            np.testing.assert_array_equal(v[[0, 2]], v0[[0, 2]])
            results.append(np.r_[v, w])
        np.testing.assert_allclose(*results, rtol=1e-12, atol=1e-14)

    def test_joint_floor_pair_and_corner_constraints(self):
        r = RigidBodies.__new__(RigidBodies)
        r.s = SimpleNamespace(
            shape=(20, 20, 20),
            body_count=2,
            steps=0,
            config=SimpleNamespace(boundary_cells=1, body_contact_iterations=32),
        )
        r.origin = np.array([[8.0, 1.9, 10.0], [9.8, 1.9, 10.0]])
        r.q = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        r.com = np.zeros((2, 3))
        r.mass = np.ones(2)
        r.inertia = np.array([np.eye(3) * 2 / 3] * 2)
        r.velocity = np.array([[0.1, -0.1, 0.0], [-0.1, -0.1, 0.0]])
        r.omega = np.array([[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]])
        r.lower, r.upper = -np.ones((2, 3)), np.ones((2, 3))
        r.resolve_contacts()
        self.assertGreaterEqual(np.min(r.origin[:, 1]), 2 - 1e-8)
        self.assertGreaterEqual(r.origin[1, 0] - r.origin[0, 0], 2 - 1e-8)
        for b in range(2):
            for x in (-1.0, 1.0):
                for z in (-1.0, 1.0):
                    speed = (r.velocity[b] + np.cross(r.omega[b], [x, -1.0, z]))[1]
                    self.assertGreaterEqual(speed, -1e-7)

    def test_quaternion_three_axis_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.03, -0.04, 0.02])
        for _ in range(100):
            q = advance_quaternion(q, omega)
        self.assertAlmostEqual(np.linalg.norm(q), 1)
        r = rotation_matrix(q)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-13)
        np.testing.assert_allclose(r @ omega, omega, atol=1e-13)

    def test_pair_collision_conserves_momentum_without_restitution(self):
        va, vb = np.array([1.0, 0.3, -0.1]), np.array([-1.0, 0.0, 0.2])
        wa, wb = np.array([0.0, 0.1, 0.2]), np.array([0.1, 0.0, 0.0])
        ra, rb, n = (
            np.array([1.0, 0.2, 0.3]),
            np.array([-1.0, 0.2, 0.3]),
            np.array([-1.0, 0.0, 0.0]),
        )
        before = 2 * va + 3 * vb
        angular_before = (
            np.cross(np.array([-1.0, 0.0, 0.0]), 2 * va)
            + np.cross(np.array([1.0, 0.0, 0.0]), 3 * vb)
            + wa
            + wb
        )
        kinetic_before = (2 * va @ va + 3 * vb @ vb + wa @ wa + wb @ wb) / 2
        apply_contact_impulse(va, wa, 2.0, np.eye(3), ra, n, vb, wb, 3.0, np.eye(3), rb)
        np.testing.assert_allclose(2 * va + 3 * vb, before, atol=1e-14)
        angular_after = (
            np.cross(np.array([-1.0, 0.0, 0.0]), 2 * va)
            + np.cross(np.array([1.0, 0.0, 0.0]), 3 * vb)
            + wa
            + wb
        )
        np.testing.assert_allclose(angular_after, angular_before, atol=1e-14)
        self.assertAlmostEqual((va + np.cross(wa, ra) - vb - np.cross(wb, rb)) @ n, 0.0)
        self.assertLess(
            (2 * va @ va + 3 * vb @ vb + wa @ wa + wb @ wb) / 2, kinetic_before
        )

    def test_wall_impulse_includes_rotation(self):
        v, w = np.array([0.2, -1.0, 0.3]), np.array([0.1, 0.0, 0.2])
        arm, n = np.array([1.0, -1.0, 0.5]), np.array([0.0, 1.0, 0.0])
        apply_contact_impulse(v, w, 1.0, np.eye(3), arm, n)
        self.assertAlmostEqual((v + np.cross(w, arm)) @ n, 0.0)
        np.testing.assert_allclose(v[[0, 2]], [0.2, 0.3])

    def test_sat_checks_depth_and_rotated_boxes(self):
        eye, half = np.eye(3), np.ones(3)
        self.assertIsNone(
            separating_axis(
                np.zeros(3), eye, half, np.array([0.0, 0.0, 3.0]), eye, half
            )
        )
        hit = separating_axis(
            np.zeros(3), eye, half, np.array([0.0, 0.0, 1.5]), eye, half
        )
        np.testing.assert_allclose(hit[0], [0, 0, -1])
        self.assertAlmostEqual(hit[1], 0.5)
        r = rotation_matrix(advance_quaternion([1, 0, 0, 0], np.array([0.3, 0.2, 0.1])))
        self.assertIsNotNone(
            separating_axis(np.zeros(3), eye, half, np.zeros(3), r, half)
        )


class OutputTests(unittest.TestCase):
    def test_vtk_order_units_and_vector_components(self):
        phi = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        velocity = np.stack([phi, phi + 100, phi + 200], axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.vti"
            write_vti(path, {"water_phase": phi, "velocity_m_s": velocity}, 0.001)
            raw = path.read_bytes()
            marker = b'<AppendedData encoding="raw">_'
            prefix, payload = raw.split(marker, 1)
            root = ET.fromstring(prefix + b"</VTKFile>")
            self.assertEqual(
                root.find("ImageData").attrib["Spacing"],
                ".001 .001 .001".replace(".001", "0.001"),
            )
            arrays = root.findall(".//DataArray")
            decoded = []
            for array in arrays:
                offset = int(array.attrib["offset"])
                (length,) = struct.unpack("<Q", payload[offset : offset + 8])
                decoded.append(
                    np.frombuffer(
                        payload[offset + 8 : offset + 8 + length], dtype="<f4"
                    )
                )
            np.testing.assert_array_equal(decoded[0], phi.ravel(order="F"))
            np.testing.assert_array_equal(
                decoded[1].reshape(-1, 3), velocity.transpose(2, 1, 0, 3).reshape(-1, 3)
            )
            write_pvd(Path(tmp) / "fields.pvd", [(0.0, "frame.vti"), (0.1, "next.vti")])
            self.assertEqual(
                len(ET.parse(Path(tmp) / "fields.pvd").findall(".//DataSet")), 2
            )


if __name__ == "__main__":
    unittest.main()
