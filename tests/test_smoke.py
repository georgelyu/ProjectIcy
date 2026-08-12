from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixture2d import DamBreakConfig, Simulator2D, create_dambreak_config


class SmokeTests(unittest.TestCase):
    def test_ice_material_defaults(self):
        cfg = DamBreakConfig(mode="coupled", mpm_material="ice")
        self.assertFalse(cfg.mpm_plasticity)
        self.assertEqual(cfg.sand_density, 917.0)
        self.assertEqual(cfg.sand_youngs_modulus, 1.0e6)
        self.assertEqual(cfg.sand_poisson_ratio, 0.30)
        self.assertEqual(cfg.wall_friction, 0.10)
        self.assertEqual(cfg.coupled_fluid_start_step, -1)
        self.assertFalse(cfg.water_retention)

        custom = create_dambreak_config(mode="sand", mpm_material="ice", sand_youngs_modulus=8.0e5)
        self.assertEqual(custom.sand_youngs_modulus, 8.0e5)
        with self.assertRaisesRegex(ValueError, "water_retention"):
            create_dambreak_config(mode="coupled", mpm_material="ice", water_retention=True)
        with self.assertRaisesRegex(ValueError, "only applies"):
            create_dambreak_config(mode="fluid", mpm_material="ice")

    def test_ice_rejects_unstable_elastic_cfl(self):
        cfg = create_dambreak_config(
            mode="sand",
            mpm_material="ice",
            resolution=(64, 32),
            particles_per_cell=1,
        )
        with self.assertRaisesRegex(ValueError, "elastic CFL is too high"):
            Simulator2D(cfg)

        for name in ("mpm_dt", "sand_density", "sand_youngs_modulus"):
            with self.subTest(parameter=name):
                bad_cfg = create_dambreak_config(
                    mode="sand",
                    mpm_material="ice",
                    **{name: float("nan")},
                )
                with self.assertRaisesRegex(ValueError, f"{name} must be finite"):
                    Simulator2D(bad_cfg)

    def run_mode(self, mode: str, **overrides) -> Simulator2D:
        cfg = create_dambreak_config(
            mode=mode,
            resolution=(64, 32),
            phase_warmup_steps=10,
            particles_per_cell=1,
            output_dir=tempfile.mkdtemp(prefix=f"mixture2d_{mode}_"),
            **overrides,
        )
        sim = Simulator2D(cfg)
        sim.step(5)
        path = Path(cfg.output_dir) / "frame_00000.png"
        sim.save_frame(path)
        self.assertTrue(path.exists())
        return sim

    def test_fluid_smoke(self):
        sim = self.run_mode("fluid")
        diag = sim.diagnostics()
        self.assertTrue(diag["fluid_finite"])
        self.assertGreaterEqual(diag["phi_max"], 0.1)

    def test_sand_smoke(self):
        sim = self.run_mode("sand")
        diag = sim.diagnostics()
        self.assertTrue(diag["particles_finite"])
        self.assertTrue(diag["particles_inside"])
        self.assertLessEqual(abs(diag["phi_min"]), 1.0e-7)
        self.assertLessEqual(abs(diag["phi_max"]), 1.0e-7)
        self.assertLessEqual(abs(diag["bound_water_max"]), 1.0e-7)
        self.assertLessEqual(float(np.abs(sim.p_water_content.to_numpy()).max()), 1.0e-7)
        rel = abs(diag["grid_mass"] - diag["particle_mass"]) / max(diag["particle_mass"], 1.0e-6)
        self.assertLess(rel, 1.0e-5)

    def test_coupled_smoke(self):
        sim = self.run_mode("coupled")
        diag = sim.diagnostics()
        self.assertTrue(diag["fluid_finite"])
        self.assertTrue(diag["particles_finite"])
        self.assertGreaterEqual(diag["eps_min"], 1.0 - sim.cfg.delta_max - 1.0e-4)
        self.assertLessEqual(diag["eps_max"], 1.0 + 1.0e-4)
        self.assertTrue(np.isfinite(sim.fluid_force.to_numpy()).all())

        # Guard against diagnostics accidentally checking only the primary
        # velocity/position fields and missing NaNs in auxiliary state.
        sim.artificial_vis[0, 0] = float("nan")
        sim.epsinon_src[0, 0] = float("nan")
        corrupt_diag = sim.diagnostics()
        self.assertFalse(corrupt_diag["fluid_finite"])
        self.assertFalse(corrupt_diag["mpm_grid_finite"])

    def test_water_retention_smoke(self):
        sim = self.run_mode("coupled", water_retention=True)
        diag = sim.diagnostics()
        self.assertTrue(diag["bound_water_finite"])
        self.assertGreaterEqual(diag["bound_water_min"], -1.0e-7)
        self.assertLessEqual(
            diag["bound_water_max"],
            sim.cfg.ratio_max * sim._particle_vol() + 1.0e-6,
        )
        self.assertTrue(np.isfinite(sim.phi_absorbed.to_numpy()).all())

    def test_coupled_ice_stays_finite(self):
        cfg = create_dambreak_config(
            mode="coupled",
            mpm_material="ice",
            resolution=(64, 32),
            # Preserve the default run's lattice stiffness at this small test
            # resolution instead of changing the physical reference scale.
            reference_length_cells=300,
            phase_warmup_steps=10,
            particles_per_cell=1,
            # Start with the ice immersed so the short regression exercises
            # two-way fluid/solid coupling rather than only free settling.
            water_width_fraction=0.75,
            output_dir=tempfile.mkdtemp(prefix="mixture2d_coupled_ice_"),
        )
        sim = Simulator2D(cfg)
        self.assertLessEqual(sim._mpm_elastic_cfl, 0.4)

        for expected_step in range(10, 101, 10):
            sim.step(10)
            with self.subTest(step=expected_step):
                diag = sim.diagnostics()
                self.assertTrue(diag["fluid_finite"])
                self.assertTrue(diag["particles_finite"])
                self.assertTrue(diag["mpm_grid_finite"])
                self.assertTrue(diag["particles_inside"])
                self.assertGreater(diag["deformation_det_min"], 0.0)

                deformation = sim.p_F.to_numpy()
                singular_values = np.linalg.svd(deformation, compute_uv=False)
                self.assertGreater(float(singular_values.min()), 0.25)
                self.assertLess(float(singular_values.max()), 4.0)
                self.assertTrue((sim.p_state.to_numpy() == 0).all())
                self.assertTrue((sim.p_q.to_numpy() == 0.0).all())
                self.assertTrue((sim.p_vcs.to_numpy() == 0.0).all())


if __name__ == "__main__":
    unittest.main()
