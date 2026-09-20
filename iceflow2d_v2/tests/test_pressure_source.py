"""Local manufactured dilatation: verify the CE remainder, not a sealed-box solution."""
import numpy as np
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.lattice import E, W


def test_manufactured_dilatation_pressure_half_source_and_ce_order():
    cfg = Config(nx=25, ny=50, gravity=0, sigma=0, smagorinsky=0, warmup_steps=0)
    geometry = Geometry(cfg)
    geometry.solid.fill(0); geometry.reconstruct(); geometry.rasterize()
    flow = Flow(cfg, geometry)
    flow.phi.fill(0); flow.h.fill(0); flow.reference_density.fill(cfg.rho_a_lu)
    e, w = np.asarray(E, float), np.asarray(W, float)
    ix = np.arange(cfg.nx)[:, None]+.5-cfg.nx/2
    errors = []
    region = (slice(5, -5), slice(5, -5))
    for source in (.002, .001, .0005):
        u = np.zeros((cfg.nx, cfg.ny, 2), np.float32)
        u[..., 0] = source*ix
        eu = np.einsum('qi,xyi->qxy', e, u)
        eq = w[:, None, None]*cfg.rho_a_lu*(3*eu+4.5*eu**2-1.5*np.sum(u*u, axis=-1))
        b = 1.8*w*cfg.rho_a_lu*source; b[0] = 0
        flow.f.from_numpy((eq-.5*b[:, None, None]).astype(np.float32))
        flow.raw_velocity.from_numpy(u); flow.velocity.from_numpy(u)
        flow.force.fill(0); flow.pressure.fill(0); flow.dilatation.fill(source)
        flow.predict()
        raw = flow.f.to_numpy().astype(float).sum(axis=0)
        remainder = (raw+.5*cfg.rho_a_lu*source)[region]/cfg.rho_a_lu
        # For u=(s*x,0), Taylor streaming gives rho*s^2 exactly at this
        # order. Omitting the new scalar source leaves an O(s) error instead.
        np.testing.assert_allclose(remainder, source**2, rtol=.015, atol=5e-9)
        pressure = flow.pressure.to_numpy()[region]/cfg.rho_a_lu
        assert np.max(np.abs(pressure)) < 2*source**2
        errors.append(float(np.mean(np.abs(remainder))))
    np.testing.assert_allclose(np.asarray(errors[:-1])/errors[1:], 4, rtol=.02)
