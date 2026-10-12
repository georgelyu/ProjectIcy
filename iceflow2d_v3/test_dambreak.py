"""CUDA checks for released-column pressure and conservative short-time flow."""

import unittest
import numpy as np
import taichi as ti

from .examples import dambreak_four_ice_melting_2d as example
from .simulator import IceFlow2D, ensure_taichi_cuda
from .config import IceBodyConfig
from .test_multibody import _two_config


@ti.kernel
def _sample_momentum_operators(s: ti.template(), out: ti.types.ndarray()):
    for i, j in s.water_phase:
        f = s._phase_flux_force_density(i, j)
        g = s._momentum_phase_gradient(i, j)
        out[i, j, 0], out[i, j, 1] = f.x, f.y
        out[i, j, 2], out[i, j, 3] = g.x, g.y
        out[i, j, 4] = s._fluid_kinematic_viscosity_lattice(i, j)


class DamBreakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_taichi_cuda()

    def test_column_pressure_is_hydrostatic_and_gate_jump_is_retained(self):
        args = example.parse_args(["--phase-warmup-steps", "0"])
        s = IceFlow2D(example.create_config(args))
        pressure = s.pressure_lattice.to_numpy()
        reference = s.reference_pressure_by_row.to_numpy()
        i, j = 30, 50
        expected = -s._gravity_lattice[1] * (1 - s._air_density_lattice) * (s.config.water_height - j - .5)
        self.assertAlmostEqual(float(pressure[i, j] - reference[j]), expected, delta=3e-11)
        self.assertAlmostEqual(float(pressure[300, j]), float(reference[j]), delta=1e-12)
        self.assertGreater(pressure[i, j], pressure[300, j])
        np.testing.assert_allclose(s.reference_density_by_row.to_numpy(), s._air_density_lattice)
        np.testing.assert_array_equal(s.physical_velocity_lattice.to_numpy(), 0.)
        self.assertAlmostEqual(s.phase_populations.to_numpy().sum(), 195 * 135, delta=1e-8)
        initial = s.phase_populations.to_numpy().sum() + s.body_mass_lattice.array().sum()
        # Check equilibrium population recovery without time-advancing a
        # deliberately sharp (unwarmed) 800:1 interface. Production uses W=5
        # with 2000 phase warm-up steps, tested by the complete trajectory.
        s._update_streamed_macroscopic_fields()
        s._update_pressure()
        current = s.phase_populations.to_numpy().sum() + s.body_mass_lattice.array().sum()
        self.assertAlmostEqual(current, initial, delta=1e-8)
        self.assertTrue(np.isfinite(s.momentum_populations.to_numpy()).all())
        self.assertTrue(np.isfinite(s.physical_velocity_lattice.to_numpy()).all())

    def test_one_cell_contact_gap_prevents_sealing_a_fluid_pore(self):
        s = IceFlow2D(_two_config(preserve_contact_fluid_gap=True))
        s.body_reference_origin[1] += np.array([-1., 0.], np.float32)
        s._refresh_fluid_domain(resolve_contact=True)
        masks = s._body_fractions.to_numpy() >= .5
        self.assertFalse(np.any(masks[0] & masks[1]))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.assertFalse(np.any(masks[0] & np.roll(masks[1], (di, dj), axis=(0, 1))))
        self.assertEqual(s.contacts.detect(), 0)

    def test_closed_gap_event_conserves_liquid_and_heat(self):
        s = IceFlow2D(_two_config(
            close_subgrid_wall_gaps=True, preserve_contact_fluid_gap=True,
            water_mass_conservation="conservative_lbm", phase_momentum_coupling="consistent_flux",
            ice_bodies=(IceBodyConfig((4,4),(9.,4.)), IceBodyConfig((4,4),(16.,4.))),
        ))
        water = float(s.phase_populations.to_numpy().sum())
        heat = float(s.thermal.water_sensible_energy.to_numpy().sum())
        ice = float(s.body_mass_lattice.array().sum())
        s.body_reference_origin[0] += np.array([0., .6], np.float32)
        s._refresh_fluid_domain(resolve_contact=True)
        self.assertTrue(np.any(s.subgrid_wall_gap.to_numpy()))
        # A known closed-aperture event budget must reappear in resolved fluid,
        # carrying the same heat. This exercises transfer, not just the mask.
        i, j = np.argwhere(s.subgrid_wall_gap.to_numpy())[0]
        amount = .25
        energy = amount*s.config.dx**2*s.config.rho_water*s.config.thermal.properties.specific_heat_water_j_kg_k*90.
        s.conservative_phase.event_volume[i, j] = amount
        s.conservative_phase.event_heat[i, j] = energy
        s.conservative_phase.remap()
        self.assertAlmostEqual(s.phase_populations.to_numpy().sum(), water+amount, delta=1e-8)
        self.assertAlmostEqual(s.thermal.water_sensible_energy.to_numpy().sum(), heat+energy, delta=1e-7)
        self.assertAlmostEqual(s.body_mass_lattice.array().sum(), ice, delta=1e-8)
        phi = s.water_phase.to_numpy()
        self.assertGreaterEqual(phi.min(), -1e-12)
        self.assertLessEqual(phi.max(), 1+1e-12)

    def test_momentum_tail_gate_is_symmetric_and_preserves_phase_and_heat(self):
        s = IceFlow2D(_two_config(
            phase_momentum_coupling="consistent_flux", phase_momentum_flux_model="accepted_links",
            water_mass_conservation="conservative_lbm",
            momentum_phase_gradient="open_links_air", phase_momentum_water_cutoff=.001,
            viscosity_water=4e-7, air_viscosity_floor_lattice=.1,
        ))
        x, y = np.meshgrid(np.arange(s.nx), np.arange(s.ny), indexing="ij")
        fluid = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
        phi = np.where(x < s.nx//2, .2, 5e-5)
        phi[~fluid] = 0
        u = np.stack([.01+.002*np.sin(y), .003*np.cos(x)], axis=-1).astype(np.float32)
        u[~fluid] = 0
        directions = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)])
        flux = np.zeros((s.nx, s.ny, 9), np.float64)
        # Independent shared-face fixture, including links that straddle the
        # cutoff. The two ends must apply the same eligibility decision.
        for i,j in np.argwhere(fluid):
            for q,opp in [(1,3),(2,4),(5,7),(8,6)]:
                ni,nj = np.array([i,j])+directions[q]
                if fluid[ni,nj]:
                    value = 1e-4*np.sin(i+.7*j+q)
                    flux[i,j,q], flux[ni,nj,opp] = value,-value
        s.water_phase.from_numpy(phi)
        s.stability_velocity.from_numpy(u)
        # raw_flux is the difference of the two outgoing phase populations.
        # The production donor limiter then turns it into the accepted flux.
        s.phase_stream_buffer.from_numpy(.5*flux)
        s.conservative_phase.prepare_transport()
        h = s.phase_populations.to_numpy()
        heat = s.thermal.water_sensible_energy.to_numpy()
        out = np.empty((s.nx,s.ny,5), np.float32)
        _sample_momentum_operators(s,out)
        np.testing.assert_allclose(out[fluid,:2].sum(axis=0,dtype=np.float64), 0, atol=1e-8)
        np.testing.assert_array_equal(out[(phi<.001)&fluid,:4],0)
        expected_nu = phi*s._water_viscosity_lattice+(1-phi)*.1
        np.testing.assert_allclose(out[fluid,4],expected_nu[fluid],rtol=1e-6)
        np.testing.assert_array_equal(s.phase_populations.to_numpy(),h)
        np.testing.assert_array_equal(s.thermal.water_sensible_energy.to_numpy(),heat)


if __name__ == "__main__":
    unittest.main()
