"""Run on CPU by default, or ICEFLOW3D_TEST_BACKEND=cuda for CUDA validation."""
# ruff: noqa: E741 -- Taichi grouped grid indices.

import os
import unittest
import numpy as np
import taichi as ti

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.tests.precision import exchange_rounding_budget
from iceflow3d.lattice import (
    DIRECTIONS,
    WEIGHTS,
    Q,
    momentum_equilibrium,
    phase_equilibrium,
    transform,
    weight,
    mixture_kinematic_viscosity,
)


@ti.kernel
def lattice_probe(out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    u = ti.Vector([0.03, -0.02, 0.04])
    raw = ti.Vector.zero(ti.f32, Q)
    for q in ti.static(range(Q)):
        raw[q] = ti.sin(q * 1.3) * 0.01
    restored = transform(transform(raw, u, False), u, True)
    for q in ti.static(range(Q)):
        out[q, 0] = momentum_equilibrium(q, 0.01, 0.8, u)
        out[q, 1] = phase_equilibrium(q, 0.6, u)
        out[q, 2] = restored[q]
        out[q, 3] = raw[q]


@ti.kernel
def seed_velocity(s: ti.template()):
    for I in ti.grouped(s.water_phase):
        if s._fluid(I):
            s._set_equilibrium(
                I,
                s.water_phase[I],
                s.pressure_lattice[I],
                ti.Vector([0.005, -0.003, 0.002]),
            )


@ti.kernel
def stress_probe(s: ti.template(), out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    from_gradient = ti.Matrix.zero(ti.f32, 3, 3)
    for q in range(Q):
        c = ti.Vector([q // 9 - 1, q // 3 % 3 - 1, q % 3 - 1])
        from_gradient += (
            3
            * weight(q)
            * s._stress_neighbor_velocity(ti.Vector([10, 10, 6]), c).outer_product(
                ti.cast(c, ti.f32)
            )
        )
    stress = -(from_gradient + from_gradient.transpose()) / 6
    for a, b in ti.static(ti.ndrange(3, 3)):
        out[a, b] = stress[a, b]


@ti.kernel
def constitutive_probe(
    out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    quadrature: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for q in range(Q):
        cy = q // 3 % 3 - 1
        quadrature[q, 0] = weight(q)
        quadrature[q, 1] = cy
    for i in range(101):
        phi = i / 100.0
        for j in range(3):
            out[i + 1, j] = mixture_kinematic_viscosity(
                phi, j * 45.0, 1000.0, 1.25, 1.8133064965144633e-5
            )


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = IceFlow3D(
            IceFlowConfig(
                nx=30,
                ny=20,
                nz=12,
                domain_size_m=(0.03, 0.02, 0.012),
                boundary_cells=1,
                water_width=12,
                water_height=16,
                floor_water_depth_cells=5,
                ice_bodies=(IceBodyConfig((20, 5, 6), (4, 4, 4)),),
                phase_warmup_steps=0,
                momentum_collision_model=os.environ.get(
                    "ICEFLOW3D_TEST_COLLISION", "cm_mrt"
                ),
                backend=os.environ.get("ICEFLOW3D_TEST_BACKEND", "cpu"),
            )
        )

    def test_01_lattice_and_equilibrium_moments(self):
        c, w = np.array(DIRECTIONS), np.array(WEIGHTS)
        self.assertAlmostEqual(w.sum(), 1.0)
        np.testing.assert_allclose(
            np.einsum("q,qi,qj->ij", w, c, c), np.eye(3) / 3, atol=1e-15
        )
        for a in range(3):
            for b in range(3):
                self.assertAlmostEqual(
                    np.sum(w * c[:, a] ** 2 * c[:, b] ** 2), 1 / 3 if a == b else 1 / 9
                )
        out = np.zeros((27, 4), np.float32)
        lattice_probe(out)
        u = np.array([0.03, -0.02, 0.04])
        self.assertAlmostEqual(float(out[:, 0].sum()), 0.0, places=7)
        np.testing.assert_allclose(c.T @ out[:, 0], 0.8 * u, atol=1e-8)
        np.testing.assert_allclose(
            np.einsum("q,qi,qj->ij", out[:, 0], c, c),
            0.8 * np.outer(u, u) + 0.01 * np.eye(3),
            atol=1e-8,
        )
        self.assertAlmostEqual(float(out[:, 1].sum()), 0.6, places=6)
        np.testing.assert_allclose(c.T @ out[:, 1], 0.6 * u, atol=1e-8)
        np.testing.assert_allclose(out[:, 2], out[:, 3], atol=2e-8)

    def test_01b_physical_mixture_viscosity_and_quadrature_precision(self):
        from iceflow3d.viscosity import water_dynamic_viscosity_pa_s

        out = np.zeros((102, 3), np.float32)
        quadrature = np.zeros((27, 2), np.float32)
        constitutive_probe(out, quadrature)
        # At hot-water tau-1/2 ~= 2.4e-5, a systematic ~1e-7 momentum
        # bias per step measurably corrupts physical shear-wave decay.
        w, cy = quadrature.astype(float).T
        np.testing.assert_allclose(
            [w.sum(), np.sum(3 * w * cy * cy)], 1, rtol=0, atol=1.5e-8
        )
        phi = np.linspace(0, 1, 101)
        rho = 1000 * phi + 1.25 * (1 - phi)
        air = 1.8133064965144633e-5
        for j, temperature in enumerate((0, 45, 90)):
            water = water_dynamic_viscosity_pa_s(temperature)
            mu = out[1:, j] * rho
            np.testing.assert_allclose(mu, phi * water + (1 - phi) * air, rtol=3e-6)
            self.assertLessEqual(float(mu.max()), water * (1 + 3e-6))
            self.assertGreaterEqual(float(mu.min()), air * (1 - 3e-6))

    def test_02_full_inertia_tensor(self):
        s = self.s
        mass = 64 * s.config.rho_ice / s.config.rho_water
        np.testing.assert_allclose(
            s.rigid.mass, [mass], rtol=8 * np.finfo(np.float32).eps
        )
        np.testing.assert_allclose(
            s.rigid.inertia[0],
            np.eye(3) * mass * 16 / 6,
            rtol=8 * np.finfo(np.float32).eps,
            atol=1e-5,
        )

    def test_03_linear_gradient_in_full_fluid_stencil(self):
        s = self.s
        saved = s.water_phase.to_numpy()
        i, j, k = np.indices(s.shape)
        linear = 0.2 + 0.008 * i + 0.006 * j + 0.009 * k
        s.water_phase.from_numpy(linear.astype(np.float32))
        s._gradients()
        gradients = s.phase_gradient_lattice.to_numpy()
        fluid = s.solid_body_id.to_numpy() == -1
        mask = fluid.copy()
        for c in DIRECTIONS:
            mask &= np.roll(fluid, tuple(c), axis=(0, 1, 2))
        np.testing.assert_allclose(
            gradients[mask], np.tile([0.008, 0.006, 0.009], (mask.sum(), 1)), atol=2e-7
        )
        s.water_phase.from_numpy(saved)
        s._gradients()

    def test_04_wall_mirror_preserves_tangential_slope(self):
        s = self.s
        saved = s.water_phase.to_numpy()
        i, j, k = np.indices(s.shape)
        try:
            s.water_phase.from_numpy((0.2 + 0.01 * j + 0.02 * k).astype(np.float32))
            s._gradients()
            np.testing.assert_allclose(
                s.phase_gradient_lattice.to_numpy()[1, 8:16, 3:9],
                np.broadcast_to([0.0, 0.01, 0.02], (8, 6, 3)),
                atol=2e-7,
            )
        finally:
            s.water_phase.from_numpy(saved)
            s._gradients()

    def test_05_heat_exchange_is_equal_and_opposite(self):
        s = self.s
        energy = s.water_energy.to_numpy()
        s._body_conduction(s.scales.dt_s)
        initial_delta = s.material_heat.to_numpy().sum(dtype=float)
        s._exchange_heat(s.scales.dt_s)
        received = s.material_heat.to_numpy().sum(dtype=float) - initial_delta
        removed = (energy.astype(float) - s.water_energy.to_numpy()).sum()
        self.assertGreater(received, 0.0)
        self.assertAlmostEqual(
            received,
            removed,
            delta=exchange_rounding_budget(s, energy)
            + 8 * np.finfo(np.float32).eps * received,
        )
        s.water_energy.from_numpy(energy)
        s.material_heat.fill(0)

    def test_06_three_direction_fv_transport_is_conservative(self):
        s = self.s
        old_u, old_e, old_v = (
            s.velocity_lattice.to_numpy(),
            s.water_energy.to_numpy(),
            s.water_volume.to_numpy(),
        )
        u = np.empty((*s.shape, 3), np.float32)
        u[:] = [0.01, 0.02, 0.03]
        s.velocity_lattice.from_numpy(u)
        s._thermal_fluxes(True)
        s._apply_thermal_fluxes(s.scales.dt_s, True)
        np.testing.assert_allclose(
            s.water_energy.to_numpy().sum(dtype=float),
            old_e.sum(dtype=float),
            rtol=2 * np.finfo(np.float32).eps,
        )
        np.testing.assert_allclose(
            s.water_volume.to_numpy().sum(dtype=float),
            old_v.sum(dtype=float),
            rtol=2 * np.finfo(np.float32).eps,
            atol=0,
        )
        self.assertGreaterEqual(s.water_volume.to_numpy().min(), 0)
        s.velocity_lattice.from_numpy(old_u)
        s.water_energy.from_numpy(old_e)
        s.water_volume.from_numpy(old_v)

    def test_07_melting_does_not_inject_water(self):
        s = self.s
        mass, energy = s.material_mass.to_numpy(), s.material_energy.to_numpy()
        phi, water_heat = s.water_phase.to_numpy(), s.water_energy.to_numpy()
        s.material_heat.from_numpy(mass * s.latent * 0.25)
        s._apply_melting()
        np.testing.assert_allclose(s.material_mass.to_numpy(), mass * 0.75, rtol=1e-7)
        np.testing.assert_array_equal(s.water_phase.to_numpy(), phi)
        np.testing.assert_array_equal(s.water_energy.to_numpy(), water_heat)
        s.material_mass.from_numpy(mass)
        s.material_energy.from_numpy(energy)
        s.material_heat.fill(0)

    def test_08_corner_mirror_has_zero_wall_normal_derivative(self):
        s = self.s
        saved = s.water_phase.to_numpy()
        # Even quadratic extension across all three lower wall planes. The
        # derivative is zero AT the walls, not at the first fluid centres.
        x = np.moveaxis(np.indices(s.shape), 0, -1) + 0.5 - s.config.boundary_cells
        coefficients = np.array([0.0002, 0.0003, 0.0004])
        try:
            s.water_phase.from_numpy(
                (0.2 + (x * x * coefficients).sum(-1)).astype(np.float32)
            )
            s._gradients()
            np.testing.assert_allclose(
                s.phase_gradient_lattice.to_numpy()[1:7, 1:7, 1:7],
                (2 * x * coefficients)[1:7, 1:7, 1:7],
                atol=2e-7,
            )
        finally:
            s.water_phase.from_numpy(saved)
            s._gradients()

    # HOME collision and D3Q7 AA/moving-mask checks are independently covered
    # by test_home_phase. The old D3Q27 population closure no longer applies.

    def test_10_cut_wall_stress_is_linear_exact_without_small_gap_amplification(self):
        s = self.s
        saved = {
            name: getattr(s, name).to_numpy()
            for name in (
                "solid_body_id",
                "coverage",
                "velocity_lattice",
                "body_velocity",
                "body_angular_velocity",
            )
        }
        ids = np.full(s.shape, -1, np.int32)
        ids[11:] = 0
        s.solid_body_id.from_numpy(ids)
        s.body_velocity.fill(0)
        s.body_angular_velocity.fill(0)
        out = np.zeros((3, 3), np.float32)
        expected = np.zeros((3, 3))
        expected[0, 1] = expected[1, 0] = -0.01 / 6
        try:
            for eta in (0.05, 0.1, 0.5, 0.95):
                coverage = np.full(s.shape, 0.5 - 0.5 * eta, np.float32)
                coverage[11:] += 0.5
                s.coverage.from_numpy(coverage)
                velocity = np.zeros((*s.shape, 3), np.float32)
                velocity[..., 1] = 0.01 * (np.arange(s.nx) - 10 - eta)[:, None, None]
                s.velocity_lattice.from_numpy(velocity)
                stress_probe(s, out)
                np.testing.assert_allclose(out, expected, atol=1e-8)
                # Like a centered interior gradient, a local velocity error
                # must not be multiplied by 1/eta in the boundary ghost.
                velocity[10, 10, 6] += [0.01, 0.02, 0.03]
                s.velocity_lattice.from_numpy(velocity)
                stress_probe(s, out)
                np.testing.assert_allclose(out, expected, atol=1e-8)
        finally:
            for name, values in saved.items():
                getattr(s, name).from_numpy(values)

    def test_11_momentum_initialization_preserves_relaxed_phase_flux(self):
        s = self.s
        saved = {
            name: getattr(s, name).to_numpy()
            for name in ("home", "h", "velocity_lattice")
        }
        try:
            # A zero-sum nonequilibrium mode represents phase flux left by
            # warmup. Reinitializing h from phi would silently discard it.
            relaxed = saved["h"].copy()
            relaxed[6, 8, 6, 0] += 0.01
            relaxed[6, 8, 6, 6] -= 0.01
            s.h.from_numpy(relaxed)
            s._seed_populations()
            np.testing.assert_array_equal(s.h.to_numpy(), relaxed)
        finally:
            for name, values in saved.items():
                getattr(s, name).from_numpy(values)

    def test_13_local_refill_matches_full_domain_stresses(self):
        from iceflow3d.checkpoint import _fields

        s = self.s
        fields = _fields(s)
        saved = {name: field.to_numpy() for name, field in fields.items()}
        origin = s.rigid.origin.copy()
        try:
            i, j, k = np.indices(s.shape)
            phi = (0.5 + 0.15 * np.sin(0.7 * i + 0.3 * j - 0.2 * k)).astype(np.float32)
            phi[s.solid_body_id.to_numpy() != -1] = 0
            s.water_phase.from_numpy(phi)
            s.rigid.origin[0] += [0.8, 0.35, 0.25]
            s.rigid.upload()
            s._refresh_geometry()
            new = (s.solid_body_id.to_numpy() == -1) & (
                s.previous_body_id.to_numpy() != -1
            )
            self.assertTrue(new.any())
            local_force = s.force_density.to_numpy()[new]
            local_f, local_h = s.home.to_numpy()[new], s.h.to_numpy()[new]
            # Refill populations precede thermal remapping. Use those same
            # thermal inputs in the full-domain reference, including the old
            # air-temperature fallback at newly exposed cells.
            s.water_volume.from_numpy(saved["water_volume"])
            s.water_energy.from_numpy(saved["water_energy"])
            s._gradients()
            s._forces(False)
            np.testing.assert_allclose(
                s.force_density.to_numpy()[new], local_force, atol=1e-12
            )
            s._project_refill()
            np.testing.assert_allclose(s.home.to_numpy()[new], local_f, atol=1e-10)
            np.testing.assert_array_equal(s.h.to_numpy()[new], local_h)
        finally:
            s.rigid.origin = origin
            for name, value in saved.items():
                fields[name].from_numpy(value)

    def test_14_bounded_cfl_advection_matches_exact_scan(self):
        from iceflow3d.checkpoint import _fields

        s = self.s
        fields = _fields(s)
        saved = {name: field.to_numpy() for name, field in fields.items()}
        rng = np.random.default_rng(1401)
        try:
            u = rng.uniform(-0.03, 0.03, (*s.shape, 3)).astype(np.float32)
            s.velocity_lattice.from_numpy(u)
            s.water_energy.from_numpy(
                saved["water_energy"] * rng.uniform(0.4, 1, s.shape)
            )
            start = {name: field.to_numpy() for name, field in fields.items()}
            s._advance_advection()
            expected = s.water_energy.to_numpy()
            for name, value in start.items():
                fields[name].from_numpy(value)
            s._advance_advection(float(np.abs(u).sum(axis=-1).max()))
            np.testing.assert_array_equal(s.water_energy.to_numpy(), expected)
            for name, value in start.items():
                fields[name].from_numpy(value)
            # An inconclusive bound must fall back to the exact stencil CFL.
            s._advance_advection(1.0)
            np.testing.assert_array_equal(s.water_energy.to_numpy(), expected)
        finally:
            for name, value in saved.items():
                fields[name].from_numpy(value)

    def test_15_column_mass_reduction_matches_nonuniform_material(self):
        s = self.s
        names = (
            "material_mass",
            "reduced_mass",
            "reduced_first",
            "reduced_second",
            "reduced_lower",
            "reduced_upper",
        )
        saved = {name: getattr(s, name).to_numpy() for name in names}
        rng = np.random.default_rng(1501)
        try:
            mass = saved["material_mass"] * rng.uniform(
                0.1, 1, saved["material_mass"].shape
            )
            mass[rng.random(mass.shape) < 0.3] = 0
            s.material_mass.from_numpy(mass)
            mass = s.material_mass.to_numpy().astype(float)
            s._reduce_mass()
            for b in range(s.body_count):
                points = (
                    np.moveaxis(np.indices(mass.shape[1:]), 0, -1)
                    + 0.5
                    - 0.5 * np.array(s.config.bodies[b].size_cells)
                ).reshape(-1, 3)
                # Taichi's default-f32 SI/LU literals are unchanged from the
                # original kernel; isolate the f64 reduction error here.
                m = mass[b].ravel() / float(np.float32(s.scales.mass_kg))
                expected_second = np.sum(
                    m * ((points * points).sum(axis=-1) + float(np.float32(1 / 6)))
                ) * np.eye(3) - np.einsum("n,ni,nj->ij", m, points, points)
                np.testing.assert_allclose(
                    s.reduced_mass.to_numpy()[b],
                    m.sum(),
                    rtol=8 * np.finfo(np.float32).eps,
                )
                np.testing.assert_allclose(
                    s.reduced_first.to_numpy()[b],
                    np.einsum("n,ni->i", m, points),
                    rtol=8 * np.finfo(np.float32).eps,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    s.reduced_second.to_numpy()[b],
                    expected_second,
                    rtol=8 * np.finfo(np.float32).eps,
                    atol=2e-6,
                )
                np.testing.assert_array_equal(
                    s.reduced_lower.to_numpy()[b], (points[m > 0] - 0.5).min(axis=0)
                )
                np.testing.assert_array_equal(
                    s.reduced_upper.to_numpy()[b], (points[m > 0] + 0.5).max(axis=0)
                )
        finally:
            for name, value in saved.items():
                getattr(s, name).from_numpy(value)

    def test_90_coupled_motion_melt_and_exposed_node_refill(self):
        s = self.s
        original_mass = s.rigid.mass.sum()
        original_center = s.body_center.to_numpy().copy()
        s.step(17)
        s.flush_thermal()
        self.assertLess(s.rigid.mass.sum(), original_mass)
        self.assertEqual(s.steps, 17)
        self.assertGreater(
            np.linalg.norm(s.body_center.to_numpy() - original_center), 0.0
        )
        snapshot = s.snapshot()
        for field in snapshot.values():
            self.assertTrue(np.isfinite(field).all())
        self.assertGreaterEqual(snapshot["water_phase"].min(), 0.0)
        self.assertLessEqual(snapshot["water_phase"].max(), 1.0)
        old_ice = s.solid_body_id.to_numpy() >= 0
        self.assertTrue(old_ice.any())
        # A disappearing block must receive its surrounding phase, not rho_i/rho_w.
        s.water_phase.fill(0.2)
        old_fluid = s.solid_body_id.to_numpy() == -1
        s.initial_phase_volume_m3 = float(old_fluid.sum() * 0.2 * s.cell_volume)
        s.material_mass.fill(0)
        s.rigid.update_mass()
        s._refresh_geometry()
        self.assertFalse((s.solid_body_id.to_numpy() >= 0).any())
        self.assertEqual(s.pending_cells, 0)
        phase = s.water_phase.to_numpy()
        # With global redistribution disabled, exposed nodes retain the donor
        # phase. No compensating reduction is applied to the existing fluid.
        np.testing.assert_allclose(phase[old_ice], 0.2, atol=1e-7)
        np.testing.assert_allclose(phase[old_fluid], 0.2, atol=1e-7)
        self.assertGreater(
            phase.sum(dtype=float) * s.cell_volume, s.initial_phase_volume_m3
        )
        self.assertEqual(float(s.phase_volume_projection_m3[None]), 0.0)
        np.testing.assert_array_equal(s.phase_projection_momentum_n_s[None], 0.0)
        self.assertTrue(np.isfinite(s.home.to_numpy()[old_ice]).all())


if __name__ == "__main__":
    unittest.main()
