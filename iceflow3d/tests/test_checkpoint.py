import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.checkpoint import _fields, load_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_includes_pending_heat_and_host_body_state(self):
        sim = IceFlow3D(
            IceFlowConfig(
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
        )
        sim.step(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_checkpoint(sim, path)
            restored = load_checkpoint(path)
        self.assertEqual(restored.steps, 5)
        self.assertEqual(restored._thermal_pending, 5)
        self.assertEqual(restored.config, sim.config)
        for name, field in _fields(sim).items():
            np.testing.assert_array_equal(
                getattr(restored, name).to_numpy(), field.to_numpy(), err_msg=name
            )
        for name, value in vars(sim.rigid).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(restored.rigid, name), value)
        # Cross a thermal update, including melting and geometry remapping.
        sim.step(4)
        restored.step(4)
        for name in (
            "water_phase",
            "velocity_lattice",
            "water_energy",
            "material_mass",
            "body_center",
        ):
            np.testing.assert_allclose(
                getattr(restored, name).to_numpy(),
                getattr(sim, name).to_numpy(),
                rtol=2e-6,
                atol=1e-9,
                err_msg=name,
            )
        self.assertEqual(restored._thermal_pending, sim._thermal_pending)


if __name__ == "__main__":
    unittest.main()
