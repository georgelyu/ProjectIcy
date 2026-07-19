from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixture2d import Simulator2D, create_dambreak_config


class SmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
