import numpy as np

from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.rigid import Rigid


def test_off_origin_remnant_coasts_without_spurious_translation_or_spin():
    """An asymmetric tiny remnant must coast at P/M and constant COM spin.

    Euler integration of the material origin and stale world first moments
    violate this exact torque-free solution, despite conserving global P/L.
    """
    cfg = Config(nx=25, ny=50, gravity=0, sigma=0, buoyancy=False)
    geo = Geometry(cfg)
    fraction = np.zeros((cfg.nb, cfg.nb), np.float32)
    fraction[-2, -2] = .01
    geo.solid.from_numpy(fraction)
    geo.reconstruct(); geo.rasterize()
    rigid = Rigid(cfg, geo)
    mass, inertia = float(geo.mass[None]), float(geo.inertia[None])
    center = np.asarray(geo.origin[None])+np.asarray(geo.first_moment[None])/mass
    velocity, omega = np.array([.002, -.003]), .1
    linear = mass*velocity
    angular = center[0]*linear[1]-center[1]*linear[0]+inertia*omega
    world = np.r_[linear, angular]
    high = world.astype(np.float32)
    rigid.momentum[None] = high
    rigid.momentum_comp[None] = (high.astype(float)-world).astype(np.float32)
    rigid.recover_velocity()
    initial_angle = float(geo.angle[None])
    for step in range(1, 201):
        rigid.advance_pose()
        geo.update_rotation(); geo.update_mass()
        rigid.recover_velocity()
        measured_center = np.asarray(geo.origin[None])+np.asarray(geo.first_moment[None])/float(geo.mass[None])
        np.testing.assert_allclose(measured_center, center+step*velocity, atol=3e-10, rtol=0)
        np.testing.assert_allclose(float(geo.angle[None]), initial_angle+step*omega, atol=3e-7, rtol=0)
        np.testing.assert_allclose(np.asarray(geo.velocity[None])[2], omega, atol=2e-8, rtol=0)
    np.testing.assert_array_equal(np.asarray(rigid.momentum[None]), high)
