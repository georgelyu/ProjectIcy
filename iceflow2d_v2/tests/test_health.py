import numpy as np

from iceflow2d_v2.config import Config
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.health import Health
from iceflow2d_v2.state import State


def test_geometry_guard_measures_surviving_material_not_the_initial_box():
    cfg = Config(nx=25, ny=50, gravity=0, drop_height=-.012)
    geo = Geometry(cfg)
    solid = np.zeros((cfg.nb, cfg.nb), np.float32)
    a, b = cfg.nb-2, cfg.nb//2
    solid[a, b] = 1
    geo.solid.from_numpy(solid)
    geo.solid_comp.fill(0)
    geo.reconstruct(); geo.rasterize(); geo.remember_geometry()
    flow = Flow(cfg, geo)
    state = State(cfg, geo, flow)
    health = Health(cfg, geo, state, flow)
    center = np.asarray(geo.rotation[None])@np.array([a+.5-.5*cfg.nb, b+.5-.5*cfg.nb])
    for omega, expected_failure in [(.1, False), (.4, True)]:
        # The remote material origin moves rapidly while this cell rotates
        # about its own fixed centre. True maxima are omega/sqrt(2).
        geo.velocity[None] = [omega*center[1], -omega*center[0], omega]
        assert np.linalg.norm(np.asarray(geo.velocity[None])[:2])+abs(omega)*cfg.nb > .2
        health.reset(); health.observe()
        assert bool(health.flags[None] & 32) == expected_failure
