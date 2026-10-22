"""Cache invalidation and local heat exchange against full-domain references."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import taichi as ti

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.checkpoint import load_checkpoint, save_checkpoint


@ti.kernel
def full_heat(sim: ti.template(), dt: ti.f64):
    for I in ti.grouped(sim.water_volume):
        sim._exchange_heat_cell(I, dt)


class CacheAndHeatTests(unittest.TestCase):
    def make_sim(self):
        return IceFlow3D(IceFlowConfig(
            nx=60, ny=24, nz=20, domain_size_m=(.06, .024, .02),
            boundary_cells=1, water_width=12, water_height=19,
            floor_water_depth_cells=8, phase_warmup_steps=0,
            ice_bodies=tuple(IceBodyConfig((x, 7, 10), (4, 4, 4),
                (np.cos(.17), 0., np.sin(.17), 0.)) for x in (16, 28, 40, 52)),
            backend=os.environ.get('ICEFLOW3D_TEST_BACKEND', 'cpu'),
        ))

    def test_rotated_melting_boxes_match_full_heat(self):
        sim = self.make_sim()
        rng = np.random.default_rng(1603)
        mass = sim.material_mass.to_numpy()
        mass[0, :2] = 0
        sim.material_mass.from_numpy(mass)
        sim.rigid.origin += np.array([.3, .2, -.4])
        sim.rigid.update_mass()
        sim._rasterize()
        volume = np.where(sim.solid_body_id.to_numpy() == -1, .8*sim.cell_volume, 0.)
        energy = volume * sim.rho_cp * rng.uniform(50, 90, sim.shape)
        sim.water_volume.from_numpy(volume)
        sim.water_energy.from_numpy(energy)
        sim.material_heat.fill(0)
        full_heat(sim, sim.scales.dt_s)
        expected_energy = sim.water_energy.to_numpy()
        expected_heat = sim.material_heat.to_numpy()
        self.assertTrue(np.all(expected_heat.sum(axis=(1, 2, 3)) > 0))
        sim.water_energy.from_numpy(energy)
        sim.material_heat.fill(0)
        sim._exchange_heat(sim.scales.dt_s)
        actual_energy = sim.water_energy.to_numpy()
        actual_heat = sim.material_heat.to_numpy()
        if sim.config.backend == 'cpu':
            # CPU lowering of the two loop shapes rounds the f32 material
            # coordinates differently. Compare heat exchanged, not the much
            # larger stored energy, and still require identical heated cells.
            np.testing.assert_array_equal(energy != actual_energy, energy != expected_energy)
            np.testing.assert_allclose(energy-actual_energy, energy-expected_energy,
                                       rtol=8*np.finfo(np.float32).eps, atol=1e-15)
            np.testing.assert_allclose(actual_heat, expected_heat,
                                       rtol=8*np.finfo(np.float32).eps, atol=1e-15)
        else:
            np.testing.assert_array_equal(actual_energy, expected_energy)
            np.testing.assert_allclose(actual_heat, expected_heat, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(float((energy-actual_energy).sum()),
                               float(actual_heat.sum()), delta=1e-12)

    def test_covering_and_exposure_keep_gradient_cache_current(self):
        sim = self.make_sim()
        i, j, k = np.indices(sim.shape)
        phase = (.5 + .2*np.sin(.4*i+.3*j-.2*k)).astype(np.float32)
        phase[sim.solid_body_id.to_numpy() != -1] = 0
        sim.water_phase.from_numpy(phase)
        sim._gradients()
        sim.rigid.origin += np.array([.8, .35, .25])
        sim.rigid.upload()
        sim._refresh_geometry()
        ids, prev = sim.solid_body_id.to_numpy(), sim.previous_body_id.to_numpy()
        self.assertTrue(np.any((ids == -1) & (prev != -1)))
        self.assertTrue(np.any((ids != -1) & (prev == -1)))
        gradient, stress = sim.phase_gradient_lattice.to_numpy(), sim.capillary_stress.to_numpy()
        sim._gradients()
        np.testing.assert_array_equal(sim.phase_gradient_lattice.to_numpy(), gradient)
        if sim.config.backend == 'cpu':
            np.testing.assert_allclose(sim.capillary_stress.to_numpy(), stress,
                                       rtol=2*np.finfo(np.float32).eps, atol=1e-20)
        else:
            np.testing.assert_array_equal(sim.capillary_stress.to_numpy(), stress)

    def test_checkpoint_rebuilds_stale_cached_fields(self):
        sim = self.make_sim()
        sim.step(3)
        sim.phase_gradient_lattice.fill(1)
        sim.capillary_stress.fill(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            save_checkpoint(sim, path)
            restored = load_checkpoint(path)
        # The external-edit API and checkpoint loader must both rebuild before
        # the next collision, even when serialized cache values are stale.
        sim.invalidate_gradient_cache()
        sim.step(6)
        restored.step(6)
        for name in ('water_phase','velocity_lattice','water_energy','material_mass'):
            np.testing.assert_allclose(getattr(sim,name).to_numpy(), getattr(restored,name).to_numpy(), rtol=2e-6, atol=1e-10)


if __name__ == '__main__': unittest.main()
