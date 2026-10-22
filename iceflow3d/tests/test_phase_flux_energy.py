"""Conservative phase-mass momentum transport and its kinetic-energy budget."""
import os
import unittest
import numpy as np
from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig


class PhaseFluxEnergyTests(unittest.TestCase):
    def test_donor_flux_preserves_momentum_and_dissipates_velocity_jumps(self):
        sim = IceFlow3D(IceFlowConfig(
            nx=16, ny=12, nz=10, domain_size_m=(.016,.012,.01), boundary_cells=1,
            water_width=5, water_height=8, floor_water_depth_cells=3,
            phase_warmup_steps=0, gravity=(0,0,0), surface_tension_n_m=0,
            ice_bodies=(IceBodyConfig((9,4,5),(2,2,2)),),
            backend=os.environ.get('ICEFLOW3D_TEST_BACKEND','cpu')))
        rng = np.random.default_rng(5791)
        ids = np.full(sim.shape,-2,np.int32)
        ids[1:-1,1:-1,1:-1] = -1
        sim.solid_body_id.from_numpy(ids)
        sim.water_phase.from_numpy(rng.uniform(.01,.99,sim.shape).astype(np.float32))
        flux = rng.uniform(-1e-3,1e-3,(*sim.shape,3)).astype(np.float32)
        sim.phase_mass_flux.from_numpy(flux)
        for uniform in (False, True):
            with self.subTest(uniform_velocity=uniform):
                velocity = (np.broadcast_to(np.array([.03,-.02,.01],np.float32),
                            (*sim.shape,3)).copy() if uniform else
                            rng.uniform(-.1,.1,(*sim.shape,3)).astype(np.float32))
                sim.velocity_lattice.from_numpy(velocity)
                sim._forces(True)
                actual = sim.force_density.to_numpy().astype(float)
                central = np.zeros_like(actual)
                expected_dissipation = 0.
                # Sum each shared face once. This provides an independent
                # conservative reference and the analytical face-jump budget.
                for axis in range(3):
                    lo = [slice(None)]*3
                    hi = lo.copy()
                    lo[axis], hi[axis] = slice(None,-1), slice(1,None)
                    lo, hi = tuple(lo), tuple(hi)
                    fluid = (ids[lo]==-1) & (ids[hi]==-1)
                    mass = .5*(flux[lo][...,axis].astype(float)+flux[hi][...,axis])*fluid
                    u0, u1 = velocity[lo].astype(float), velocity[hi].astype(float)
                    centered_flux = mass[...,None]*.5*(u0+u1)
                    central[lo] -= centered_flux
                    central[hi] += centered_flux
                    expected_dissipation -= float((.5*abs(mass)*((u1-u0)**2).sum(-1)).sum())
                correction = actual-central
                work = float((velocity*correction).sum())
                self.assertLess(np.linalg.norm(actual.sum((0,1,2))),1e-8)
                self.assertAlmostEqual(work,expected_dissipation,delta=1e-8)
                if uniform:
                    self.assertLess(abs(correction).max(),1e-9)
                else:
                    self.assertLess(work,0)


if __name__ == '__main__':
    unittest.main()
