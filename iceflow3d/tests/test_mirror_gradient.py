"""Neutral-wall extension, shared source gradient and moving-geometry cache."""

# ruff: noqa: E741
import os
import unittest

import numpy as np
import taichi as ti

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.checkpoint import _fields


@ti.kernel
def mirror_probe(
    s: ti.template(),
    I: ti.types.vector(3, ti.i32),
    c: ti.types.vector(3, ti.i32),
    out: ti.types.ndarray(),
):
    point = s._ice_mirror_point(I, I + c)
    for a in ti.static(range(3)):
        out[a] = point[a]
    out[3] = s._wall_phase_sample(I, c)


@ti.kernel
def seed_cached_source(s: ti.template()):
    for I in ti.grouped(s.water_phase):
        if s._fluid(I):
            s._set_momentum_equilibrium(
                I, s.water_phase[I], 0.01, s.velocity_lattice[I]
            )


class MirrorGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = IceFlow3D(
            IceFlowConfig(
                nx=36,
                ny=24,
                nz=20,
                domain_size_m=(0.036, 0.024, 0.020),
                boundary_cells=1,
                water_width=12,
                water_height=18,
                floor_water_depth_cells=8,
                phase_warmup_steps=0,
                ice_bodies=(IceBodyConfig((24.0, 10.0, 10.0), (8, 8, 8)),),
                backend=os.environ.get("ICEFLOW3D_TEST_BACKEND", "cpu"),
            )
        )

    def setUp(self):
        self.saved = {name: value.to_numpy() for name, value in _fields(self.s).items()}
        self.rigid = {
            k: v.copy()
            for k, v in vars(self.s.rigid).items()
            if isinstance(v, np.ndarray)
        }

    def tearDown(self):
        for name, value in self.rigid.items():
            setattr(self.s.rigid, name, value)
        for name, value in self.saved.items():
            getattr(self.s, name).from_numpy(value)
        self.s.invalidate_gradient_cache()

    def test_constant_ignores_poisoned_solid_phase(self):
        s = self.s
        ids = s.solid_body_id.to_numpy()
        phase = np.where(ids == -1, 0.73, np.nan).astype(np.float32)
        s.water_phase.from_numpy(phase)
        s._gradients()
        np.testing.assert_allclose(s.phase_gradient_lattice.to_numpy(), 0, atol=1e-7)
        self.assertTrue(np.isfinite(s.capillary_stress.to_numpy()).all())

    def test_tank_and_ice_face_keep_diagonal_tangential_offset(self):
        s = self.s
        i, j, k = np.indices(s.shape)
        phase = (0.2 + 0.012 * j + 0.018 * k).astype(np.float32)
        phase[s.solid_body_id.to_numpy() != -1] = np.nan
        s.water_phase.from_numpy(phase)
        s._gradients()
        g = s.phase_gradient_lattice.to_numpy()
        # Lower x tank wall and central part of left ice face x=19.5.
        for I in ((1, 10, 10), (19, 10, 10)):
            np.testing.assert_allclose(g[I], [0, 0.012, 0.018], atol=2e-7)
        out = np.zeros(4, np.float32)
        mirror_probe(s, (19, 10, 10), (1, 1, 0), out)
        np.testing.assert_allclose(out[:3], [19, 11, 10], atol=1e-6)
        self.assertAlmostEqual(float(out[3]), float(phase[19, 11, 10]), places=6)

    def test_oblique_cut_plane_reflection_and_fluid_only_interpolation(self):
        s = self.s
        coords = np.moveaxis(np.indices(s.shape), 0, -1)
        n = np.array([1.0, 2.0, 1.0]) / np.sqrt(6)
        anchor = np.array([10.0, 10.0, 10.0]) + 0.25 * np.array([1, 1, 0])
        level = (coords - anchor) @ n
        coverage = np.clip(0.5 + 0.1 * level, 0, 1).astype(np.float32)
        ids = np.where(level < 0, -1, 0).astype(np.int32)
        tank = self.saved["solid_body_id"] == -2
        ids[tank] = -2
        coverage[tank] = 0
        phase = (0.4 + 0.01 * coords[..., 0] - 0.005 * coords[..., 1]).astype(
            np.float32
        )
        phase[ids != -1] = np.nan
        s.coverage.from_numpy(coverage)
        s.solid_body_id.from_numpy(ids)
        s.water_phase.from_numpy(phase)
        out = np.zeros(4, np.float32)
        mirror_probe(s, (10, 10, 10), (1, 1, 0), out)
        J = np.array([11, 11, 10])
        mirror = J - 2 * np.dot(J - anchor, n) * n
        np.testing.assert_allclose(out[:3], mirror, atol=2e-6)
        # Independent positive-weight interpolation, excluding solid values.
        base = np.floor(mirror).astype(int)
        part = mirror - base
        samples, weights = [], []
        for offset in np.ndindex(2, 2, 2):
            index = tuple(base + offset)
            if ids[index] == -1:
                samples.append(phase[index])
                weights.append(np.prod(np.where(offset, part, 1 - part)))
        expected = np.average(samples, weights=weights)
        self.assertAlmostEqual(float(out[3]), expected, delta=1e-6)
        self.assertTrue(min(samples) <= out[3] <= max(samples))
        s._gradients()
        self.assertTrue(np.isfinite(s.phase_gradient_lattice.to_numpy()).all())

    def test_all_density_source_paths_read_the_cached_vector(self):
        s = self.s
        # Deliberately prescribe a gradient unlike the spatial phase. This
        # catches hidden stencil rescans in seeding, collision and recovery.
        s.water_phase.fill(0.6)
        g = np.array([0.002, -0.001, 0.003])
        u = np.array([0.005, 0.003, -0.002])
        s.phase_gradient_lattice.fill(g.tolist())
        s.velocity_lattice.fill(u.tolist())
        s.force_density.fill(0)
        seed_cached_source(s)
        I = (8, 10, 10)
        source = (1 - s.rho_air) * np.dot(u, g)
        f = s.f.to_numpy()[I]
        self.assertAlmostEqual(float(f.sum()), -0.5 * source, delta=2e-8)
        s._collide_momentum()
        self.assertAlmostEqual(
            float(s.f_post.to_numpy()[I].sum() - f.sum()), source, delta=2e-8
        )
        self.assertEqual(float(s.density_source_scale[I]), 1.0)
        # Both recovery entry points close the prescribed pressure with the
        # same cache, despite spatially constant phase (actual gradient=0).
        s._recover_momentum()
        self.assertAlmostEqual(float(s.pressure_lattice[I]), 0.01, delta=2e-8)
        moving = s.f.to_numpy().sum(-1, dtype=np.float64) - s.f.to_numpy()[..., 13]
        s.moving_population_sum.from_numpy(moving)
        s._recover_pressure()
        self.assertAlmostEqual(float(s.pressure_lattice[I]), 0.01, delta=2e-8)

    def check_geometry_refresh(self, melt=False):
        s = self.s
        i, j, k = np.indices(s.shape)
        phase = (0.5 + 0.2 * np.sin(0.3 * i + 0.2 * j - 0.1 * k)).astype(np.float32)
        phase[s.solid_body_id.to_numpy() != -1] = 0
        s.water_phase.from_numpy(phase)
        s._gradients()
        before = s.phase_gradient_lattice.to_numpy()
        if melt:
            s.material_mass.from_numpy(self.saved["material_mass"] * 0.96)
            s.rigid.update_mass()
        else:
            s.rigid.origin += [0.08, 0.04, 0.03]
            s.rigid.upload()
        s._refresh_geometry(phase_changed=False)
        np.testing.assert_array_equal(
            s.solid_body_id.to_numpy(), self.saved["solid_body_id"]
        )
        cached = s.phase_gradient_lattice.to_numpy()
        self.assertGreater(float(np.max(abs(cached - before))), 1e-6)
        stress = s.capillary_stress.to_numpy()
        s._gradients()
        np.testing.assert_array_equal(s.phase_gradient_lattice.to_numpy(), cached)
        if s.config.backend == "cpu":
            # Identical gradients; CPU loop lowering can contract the tensor
            # products differently (one f32 ULP), as for local heat exchange.
            np.testing.assert_allclose(
                s.capillary_stress.to_numpy(),
                stress,
                rtol=2 * np.finfo(np.float32).eps,
                atol=1e-20,
            )
        else:
            np.testing.assert_array_equal(s.capillary_stress.to_numpy(), stress)

    def test_subcell_motion_refreshes_cache_without_label_changes(self):
        self.check_geometry_refresh()

    def test_partial_melting_refreshes_cache_without_label_changes(self):
        self.check_geometry_refresh(melt=True)


if __name__ == "__main__":
    unittest.main()
