"""Independent moment, forcing, AA parity and moving-mask regressions."""

# ruff: noqa: E741
import os
import unittest

import numpy as np
import taichi as ti

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.checkpoint import _fields
from iceflow3d.home import central_collision, reconstruct
from iceflow3d.lattice import DIRECTIONS, WEIGHTS
from iceflow3d.phase import PHASE_DIRECTIONS, PHASE_OPPOSITE, phase_collision
from iceflow3d.viscosity import water_dynamic_viscosity_pa_s


def small_config(**kwargs):
    values = dict(
        nx=20,
        ny=16,
        nz=12,
        domain_size_m=(0.02, 0.016, 0.012),
        boundary_cells=1,
        water_width=8,
        water_height=12,
        floor_water_depth_cells=5,
        ice_bodies=(IceBodyConfig((14, 5, 6), (4, 4, 4)),),
        phase_warmup_steps=0,
        backend=os.environ.get("ICEFLOW3D_TEST_BACKEND", "cpu"),
    )
    values.update(kwargs)
    return IceFlowConfig(**values)


@ti.kernel
def moment_probe(
    data: ti.types.ndarray(),
    populations: ti.types.ndarray(),
    collided: ti.types.ndarray(),
):
    for i in range(data.shape[0]):
        m = ti.Vector.zero(ti.f32, 10)
        for a in ti.static(range(10)):
            m[a] = data[i, a]
        a = ti.Vector([data[i, 10], data[i, 11], data[i, 12]])
        post = central_collision(m, a, data[i, 13], data[i, 14])
        for q in ti.static(range(27)):
            populations[i, q] = reconstruct(q, m)
        for k in ti.static(range(10)):
            collided[i, k] = post[k]


@ti.kernel
def phase_probe(data: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(data.shape[0]):
        h = ti.Vector.zero(ti.f32, 7)
        for q in ti.static(range(7)):
            h[q] = data[i, q]
        u = ti.Vector([data[i, 8], data[i, 9], data[i, 10]])
        n = ti.Vector([data[i, 11], data[i, 12], data[i, 13]])
        post, flux = phase_collision(h, data[i, 7], u, n, 0.1, 5.0)
        for q in ti.static(range(7)):
            out[i, q] = post[q]
        for a in ti.static(range(3)):
            out[i, a + 7] = flux[a]


@ti.kernel
def generic_force_probe(s: ti.template(), out: ti.types.ndarray()):
    """Exercise the general solid-link path without bulk/tank specialization."""
    for I in ti.grouped(s.water_phase):
        if s._fluid(I):
            force = s._force_at(I, True)
            acceleration = s._home_acceleration_at(I, force, False, True)
            for a in ti.static(range(3)):
                out[I.x, I.y, I.z, a] = force[a]
                out[I.x, I.y, I.z, a + 3] = acceleration[a]


@ti.kernel
def generic_gradient_probe(s: ti.template()):
    for I in ti.grouped(s.water_phase):
        s._write_gradient(I)


def phase_matrix_collision(h, phi, u, n, mobility=0.1, width=5):
    """Full 7x7 central-moment collision, independent of optimized code."""
    c = np.array(PHASE_DIRECTIONS, float)
    w = np.array([0.25] + [0.125] * 6)
    v = c - u
    T = np.array(
        [
            np.ones(7),
            v[:, 0],
            v[:, 1],
            v[:, 2],
            v[:, 0] ** 2 - v[:, 1] ** 2,
            v[:, 0] ** 2 - v[:, 2] ** 2,
            np.sum(v * v, axis=1),
        ]
    )
    rate = np.array([1, *([1 / (0.5 + 4 * mobility)] * 3), 1, 1, 1])
    eq = w * phi * (1 + 4 * c @ u)
    src = w * 4 * phi * (1 - phi) / width * (c @ n)
    return h + np.linalg.solve(T, -rate * (T @ (h - eq)) + (1 - rate / 2) * (T @ src))


class HomePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = IceFlow3D(small_config())

    def setUp(self):
        self.saved = {k: v.to_numpy() for k, v in _fields(self.s).items()}

    def tearDown(self):
        for k, v in self.saved.items():
            getattr(self.s, k).from_numpy(v)

    def test_storage_and_parameters(self):
        s = self.s
        self.assertFalse(hasattr(s, "f"))
        self.assertFalse(hasattr(s, "f_post"))
        self.assertFalse(hasattr(s, "h_post"))
        self.assertEqual(s.h.to_numpy().shape, (*s.shape, 7))
        self.assertEqual(s.home.to_numpy().shape, (*s.shape, 10))
        self.assertEqual(s.config.interface_width, 5)
        self.assertEqual(s.scales.dt_s, 2.5e-5)
        for name, field in _fields(s).items():
            self.assertNotEqual(str(field.dtype), "f64", name)

    def test_reconstruction_and_forced_central_collision(self):
        rng = np.random.default_rng(2428)
        data = rng.normal(0, 0.01, (128, 15)).astype(np.float32)
        data[:, 13] = rng.uniform(0.3, 1.99995, 128)
        data[:, 14] = rng.uniform(0.3, 1.9, 128)
        data[:, 10:13] *= 0.01
        populations = np.zeros((128, 27), np.float32)
        collided = np.zeros((128, 10), np.float32)
        moment_probe(data, populations, collided)
        c = np.array(DIRECTIONS)
        for row, g, actual in zip(data.astype(float), populations, collided):
            p, j = row[0], row[1:4]
            S = np.array(
                [
                    [row[4], row[7], row[8]],
                    [row[7], row[5], row[9]],
                    [row[8], row[9], row[6]],
                ]
            )
            np.testing.assert_allclose(g.sum(dtype=float), p, atol=3e-8)
            np.testing.assert_allclose(c.T @ g, j, atol=3e-8)
            np.testing.assert_allclose(
                np.einsum("q,qi,qj->ij", g, c, c) - np.eye(3) * p / 3, S, atol=3e-8
            )
            a, rate, bulk = row[10:13], row[13], row[14]
            u = j + a / 2
            # Direct central second moment. For velocity distributions the
            # equilibrium has the same p-dependent central offset as g_eq.
            v = c - u
            K = np.einsum("q,qi,qj->ij", g, v, v)
            eq2 = np.eye(3) * p / 3 + np.outer(u, u)
            Keq = eq2 - np.outer(u, u) * 2 + p * np.outer(u, u)
            residual = K - Keq
            # The central second source is zero; the incoming central
            # residual already contains the pre-collision half-force term.
            trace = np.trace(residual)
            Kpost = (
                Keq
                + (1 - rate) * (residual - np.eye(3) * trace / 3)
                + (1 - bulk) * np.eye(3) * trace / 3
            )
            jpost = j + a
            Snew = (
                Kpost
                + np.outer(u, jpost)
                + np.outer(jpost, u)
                - p * np.outer(u, u)
                - p / 3 * np.eye(3)
            )
            expected = np.r_[
                p, jpost, Snew.diagonal(), Snew[0, 1], Snew[0, 2], Snew[1, 2]
            ]
            np.testing.assert_allclose(actual, expected, atol=4e-8)

    def test_phase_collision_matches_full_central_matrix(self):
        rng = np.random.default_rng(247)
        data = np.zeros((256, 14), np.float32)
        data[:, 7] = rng.uniform(0.01, 0.99, 256)
        data[:, :7] = rng.normal(0, 0.02, (256, 7))
        data[:, 0] += data[:, 7] - data[:, :7].sum(axis=1)
        data[:, 8:11] = rng.uniform(-0.08, 0.08, (256, 3))
        data[:, 11:14] = rng.normal(size=(256, 3))
        data[:, 11:14] /= np.linalg.norm(data[:, 11:14], axis=1)[:, None]
        out = np.zeros((256, 10), np.float32)
        phase_probe(data, out)
        c = np.array(PHASE_DIRECTIONS)
        for row, actual in zip(data.astype(float), out):
            expected = phase_matrix_collision(row[:7], row[7], row[8:11], row[11:14])
            np.testing.assert_allclose(actual[:7], expected, atol=1.3e-7)
            np.testing.assert_allclose(
                actual[7:],
                0.5 * c.T @ (row[:7] + expected) - row[7] * row[8:11],
                atol=5e-8,
            )

    def test_reconstruction_recovers_off_diagonal_third_hermite_moments(self):
        # Independent quadrature identities catch the xyz multiplicity: its
        # six permutations require 1/cs^6, versus 1/(2*cs^6) for xxy, etc.
        rng = np.random.default_rng(27103)
        data = rng.uniform(-0.08, 0.08, (128, 15)).astype(np.float32)
        data[:, 13:15] = 1
        populations = np.zeros((128, 27), np.float32)
        collided = np.zeros((128, 10), np.float32)
        moment_probe(data, populations, collided)
        c = np.array(DIRECTIONS, dtype=float)
        u = data[:, 1:4].astype(float)
        S = np.empty((128, 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = data[:, 4:7].T
        S[:, 0, 1] = S[:, 1, 0] = data[:, 7]
        S[:, 0, 2] = S[:, 2, 0] = data[:, 8]
        S[:, 1, 2] = S[:, 2, 1] = data[:, 9]
        for a in range(3):
            for b in range(3):
                if a != b:
                    expected = (
                        S[:, a, a] * u[:, b]
                        + 2 * S[:, a, b] * u[:, a]
                        - 2 * u[:, a] ** 2 * u[:, b]
                    )
                    actual = populations.astype(float) @ (
                        (c[:, a] ** 2 - 1 / 3) * c[:, b]
                    )
                    np.testing.assert_allclose(actual, expected, atol=2e-8)
        expected = (
            S[:, 0, 2] * u[:, 1]
            + S[:, 1, 2] * u[:, 0]
            + S[:, 0, 1] * u[:, 2]
            - 2 * u.prod(axis=1)
        )
        np.testing.assert_allclose(
            populations.astype(float) @ c.prod(axis=1), expected, atol=2e-8
        )

    def test_pressure_product_rule_and_linear_viscous_force(self):
        s = self.s
        xyz = np.moveaxis(np.indices(s.shape), 0, -1)
        slope = np.array([0.01, -0.006, 0.004])
        phi = (0.4 + xyz @ slope).astype(np.float32)
        A = np.array([[0.001, 0.002, 0], [-0.001, 0.002, 0.001], [0, 0.001, -0.003]])
        s.water_phase.from_numpy(phi)
        s.velocity_lattice.from_numpy((xyz @ A.T).astype(np.float32))
        s.phase_gradient_lattice.fill(slope.tolist())
        s.capillary_stress.fill(0)
        s.phase_mass_flux.fill(0)
        volume = phi * s.cell_volume
        s.water_volume.from_numpy(volume)
        s.water_energy.from_numpy(volume * s.rho_cp * (20 - s.tm))
        I = (5, 6, 5)  # Full fluid stencil, away from the tank and the ice.
        rho = s.rho_air + (1 - s.rho_air) * float(phi[I])
        mu = (
            float(phi[I]) * water_dynamic_viscosity_pa_s(20)
            + (1 - float(phi[I])) * s.mu_air
        )
        nu = mu / (rho * s.config.rho_water) * s.scales.dt_s / s.scales.dx_m**2
        viscous = nu / rho * (A + A.T) @ ((1 - s.rho_air) * slope)
        for pressure in (0.0, 0.013):
            s.pressure_lattice.fill(pressure)
            s._forces(False)
            expected = s.force_density.to_numpy()[I].astype(float) / rho + viscous
            # Constant physical pressure: only the normalized-pressure
            # gradient must be cancelled. This tests all 26 links independently.
            for c, w in zip(np.array(DIRECTIONS), WEIGHTS):
                J = tuple(np.array(I) + c)
                rj = s.rho_air + (1 - s.rho_air) * float(phi[J])
                expected += 3 * w * c * (pressure / rj - pressure / rho)
            np.testing.assert_allclose(
                s.home_acceleration.to_numpy()[I], expected, rtol=3e-5, atol=3e-9
            )
            macro_acceleration = s.home_acceleration.to_numpy()
            moments = s.home.to_numpy()
            density = s.rho_air + (1 - s.rho_air) * phi
            moments[..., 0] = 3 * pressure / density
            s.home.from_numpy(moments)
            s._forces(False, moment_pressure=True)
            np.testing.assert_allclose(
                s.home_acceleration.to_numpy(), macro_acceleration, rtol=3e-5, atol=3e-9
            )

    def test_bulk_and_tank_forces_match_general_solid_links(self):
        s = self.s
        rng = np.random.default_rng(27314)
        phi = rng.uniform(0.05, 0.95, s.shape).astype(np.float32)
        s.water_phase.from_numpy(phi)
        s.velocity_lattice.from_numpy(
            rng.uniform(-0.01, 0.01, (*s.shape, 3)).astype(np.float32)
        )
        s.phase_mass_flux.from_numpy(
            rng.uniform(-1e-5, 1e-5, (*s.shape, 3)).astype(np.float32)
        )
        s.body_velocity.fill([0.001, -0.002, 0.003])
        s.body_angular_velocity.fill([0.0001, 0.0002, -0.0003])
        volume = phi * s.cell_volume
        s.water_volume.from_numpy(volume)
        s.water_energy.from_numpy(volume * s.rho_cp * (20 - s.tm))
        moments = s.home.to_numpy()
        moments[..., 0] = rng.uniform(-0.002, 0.002, s.shape)
        s.home.from_numpy(moments)
        s._gradients()
        reference = np.zeros((*s.shape, 6), np.float32)
        generic_force_probe(s, reference)
        s._forces(True, moment_pressure=True)
        actual = np.concatenate(
            (s.force_density.to_numpy(), s.home_acceleration.to_numpy()), axis=-1
        )
        flags = s._home_boundary.to_numpy()
        for flag in (0, 1, 2):
            self.assertGreater(np.count_nonzero(flags == flag), 0)
            np.testing.assert_allclose(
                actual[flags == flag], reference[flags == flag], rtol=1e-4, atol=2e-8
            )

    def test_regular_gradients_match_general_wall_mirrors(self):
        s = self.s
        rng = np.random.default_rng(27315)
        ids = s.solid_body_id.to_numpy()
        phi = rng.uniform(0.05, 0.95, s.shape).astype(np.float32)
        phi[ids != -1] = np.nan
        s.water_phase.from_numpy(phi)
        generic_gradient_probe(s)
        gradient, stress = (
            s.phase_gradient_lattice.to_numpy(),
            s.capillary_stress.to_numpy(),
        )
        s._gradients()
        np.testing.assert_allclose(
            s.phase_gradient_lattice.to_numpy(), gradient, rtol=2e-6, atol=8e-8
        )
        np.testing.assert_allclose(
            s.capillary_stress.to_numpy(), stress, rtol=4e-6, atol=1e-10
        )

    def test_fused_recovery_preserves_pressure_and_stability_guards(self):
        s = self.s
        I = (4, 6, 5)
        for invalid in ("none", "speed", "velocity_nan", "pressure_nan", "phase_nan"):
            for name, array in self.saved.items():
                getattr(s, name).from_numpy(array)
            if invalid == "speed":
                s.velocity_lattice[I] = [0.3, 0.3, 0]
            elif invalid == "velocity_nan":
                s.velocity_lattice[I] = [np.nan, 0, 0]
            elif invalid == "pressure_nan":
                m = s.home[I]
                m[0] = np.nan
                s.home[I] = m
            elif invalid == "phase_nan":
                s.h[(*I, 0)] = np.nan
            before = {k: v.to_numpy() for k, v in _fields(s).items()}
            s._recover_phase()
            s._recover_pressure()
            s._scan_stability()
            expected = {
                k: getattr(s, k).to_numpy()
                for k in (
                    "water_phase",
                    "pressure_lattice",
                    "h",
                    "maximum_speed",
                    "invalid_index",
                    "peak_speed",
                )
            }
            for name, array in before.items():
                getattr(s, name).from_numpy(array)
            s._recover_phase(recover_flow=True)
            for name, array in expected.items():
                np.testing.assert_allclose(
                    getattr(s, name).to_numpy(),
                    array,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                    err_msg=f"{invalid}: {name}",
                )
            if invalid == "none":
                s.check_stability(scanned=True)
            else:
                with self.assertRaises(RuntimeError):
                    s.check_stability(scanned=True)

    def test_aa_both_parities_match_two_buffer_push(self):
        s = self.s
        rng = np.random.default_rng(2471)
        ids = s.solid_body_id.to_numpy()
        phi = rng.uniform(0.3, 0.7, s.shape).astype(np.float32)
        u = rng.uniform(-0.015, 0.015, (*s.shape, 3)).astype(np.float32)
        grad = rng.uniform(-0.1, 0.1, (*s.shape, 3)).astype(np.float32)
        h = rng.uniform(0.01, 0.04, (*s.shape, 7)).astype(np.float32)
        h[..., 0] += phi - h.sum(-1)
        s.h.from_numpy(h)
        s.phase_parity[None] = 0
        s.velocity_lattice.from_numpy(u)
        s.phase_gradient_lattice.from_numpy(grad)
        out = np.empty_like(h)
        for step in range(4):
            wall_velocity = np.array(
                [0.002, -0.003, 0.004] if step >= 2 else [0.0, 0.0, 0.0], np.float32
            )
            s.body_velocity.fill(wall_velocity.tolist())
            s.water_phase.from_numpy(phi)
            expected = np.zeros_like(h)
            for I in zip(*np.where(ids == -1)):
                normal = grad[I].astype(float) / np.sqrt(
                    np.sum(grad[I].astype(float) ** 2)
                    + s.config.phase_normal_epsilon**2
                )
                post = phase_matrix_collision(
                    h[I].astype(float), float(phi[I]), u[I].astype(float), normal
                )
                for q, c in enumerate(PHASE_DIRECTIONS):
                    J = tuple(np.array(I) + c)
                    if ids[J] == -1:
                        expected[J + (q,)] = post[q]
                    else:
                        reflected = post[q]
                        if ids[J] >= 0:
                            reflected -= phi[I] * np.dot(c, wall_velocity)
                        expected[I + (PHASE_OPPOSITE[q],)] = reflected
            s._collide_phase()
            s.export_phase_populations(out)
            np.testing.assert_allclose(out[ids == -1], expected[ids == -1], atol=1.5e-7)
            self.assertEqual(s.phase_parity[None], (step + 1) % 2)
            h = expected
            phi = h.sum(-1)

    def test_odd_layout_survives_cover_and_exposure(self):
        s = self.s
        s._collide_phase()
        self.assertEqual(s.phase_parity[None], 1)
        before = np.empty((*s.shape, 7), np.float32)
        s.export_phase_populations(before)
        old = s.solid_body_id.to_numpy()
        new = old.copy()
        new[4:6, 5:8, 3:6] = 0
        new[(old == 0) & (np.indices(s.shape)[0] < 14)] = -1
        s.previous_body_id.from_numpy(old)
        s.solid_body_id.from_numpy(new)
        s._repair_phase_layout()
        after = np.empty_like(before)
        s.export_phase_populations(after)
        np.testing.assert_array_equal(
            after[(old == -1) & (new == -1)], before[(old == -1) & (new == -1)]
        )


if __name__ == "__main__":
    unittest.main()
