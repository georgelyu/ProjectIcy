"""Independent NumPy moment oracle. Never import/run the legacy package."""
import numpy as np
import taichi as ti
from iceflow2d_v2.lattice import (E, W, OPP, ORDERS, pressure_equilibrium,
    phase_equilibrium, momentum_collision, phase_collision, reconstruct,
    volume_source_population, snibb, thermal_post, add_compensated)


@ti.kernel
def accumulate_cancellation(data: ti.types.ndarray(dtype=ti.f64, ndim=2),
                            out: ti.types.ndarray(dtype=ti.f64, ndim=1)):
    for i in range(data.shape[0]):
        value, correction = ti.cast(0, ti.f32), ti.cast(0, ti.f32)
        for j in range(data.shape[1]):
            value, correction = add_compensated(value, correction, data[i, j])
        out[i] = ti.cast(value, ti.f64)-ti.cast(correction, ti.f64)


def test_compensated_inventory_survives_near_complete_cancellation():
    import math
    data = np.array([[1., 1e-8, -1.], [1.+1e-8, -1., 0.],
                     [1e-8, 1., -1.], [1., -1., 1e-8]])
    expected = np.array([math.fsum(row) for row in data])
    out = np.zeros(len(data))
    accumulate_cancellation(data, out)
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-16)


@ti.kernel
def evaluate(out: ti.types.ndarray(dtype=ti.f32, ndim=2), data: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for sample in range(data.shape[0]):
        u = ti.Vector([data[sample, 0], data[sample, 1]])
        rho, p, phi = data[sample, 2], data[sample, 3], data[sample, 4]
        for q in range(9):
            out[sample, q] = pressure_equilibrium(q, p, rho, u)
            out[sample, 9+q] = phase_equilibrium(q, phi, u)
            out[sample, 18+q] = volume_source_population(q, rho, data[sample, 5])


def test_pressure_phase_and_volume_source_moments():
    rng = np.random.default_rng(331)
    inputs = rng.uniform(-0.05, 0.05, (50, 6)).astype(np.float32)
    inputs[:, 2] = np.geomspace(1/800, 1, 50)
    inputs[:, 3] *= 0.1
    inputs[:, 4] = np.linspace(0, 1, 50)
    output = np.zeros((50, 27), np.float32)
    evaluate(output, inputs)
    e, w = np.array(E, float), np.array(W, float)
    for x, y in zip(inputs.astype(float), output.astype(float)):
        u, rho, p, phi, source = x[:2], x[2], x[3], x[4], x[5]
        eu = e@u
        fe = w*(3*p+rho*(3*eu+4.5*eu**2-1.5*u@u)); fe[0] -= 3*p
        he = w*phi*(1+3*eu+4.5*eu**2-1.5*u@u)
        np.testing.assert_allclose(y[:9], fe, atol=6e-9, rtol=2e-5)
        np.testing.assert_allclose(y[9:18], he, atol=6e-8, rtol=2e-6)
        np.testing.assert_allclose(y[:9].sum(), 0, atol=6e-9)
        np.testing.assert_allclose(e.T@y[:9], rho*u, atol=6e-9)
        np.testing.assert_allclose(np.einsum('qi,qj,q->ij', e, e, y[:9]), rho*np.outer(u, u)+p*np.eye(2), atol=6e-9)
        np.testing.assert_allclose(y[18:].sum(), rho*source, atol=6e-9)
        np.testing.assert_allclose(e.T@y[18:], 0, atol=6e-9)
        assert y[18] == 0


@ti.kernel
def collisions(raw: ti.types.ndarray(dtype=ti.f32, ndim=2), eq: ti.types.ndarray(dtype=ti.f32, ndim=2),
               source: ti.types.ndarray(dtype=ti.f32, ndim=2), out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(raw.shape[0]):
        r, e, s = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
        for k in range(9): r[k], e[k], s[k] = raw[i, k], eq[i, k], source[i, k]
        fm = momentum_collision(r, e, s, 0.5003, 0.8)
        hm = phase_collision(r, e, s, 1.25)
        for k in range(9): out[i, k], out[i, k+9] = fm[k], hm[k]
        for q in range(9): out[i, q+18] = reconstruct(q, ti.Vector([0.023, -0.017]), hm)


def test_approved_collision_independent_mrt_oracle():
    rng = np.random.default_rng(82)
    r, eq, s = (rng.normal(0, 0.05, (20, 9)).astype(np.float32) for _ in range(3))
    out = np.empty((20, 27), np.float32); collisions(r, eq, s, out)
    transform = np.array([[(ex-.023)**a*(ey+.017)**b for ex, ey in E] for a, b in ORDERS])
    for i in range(20):
        raw, equilibrium, src = r[i].astype(float), eq[i].astype(float), s[i].astype(float)
        expected = equilibrium+.5*src
        expected[:3] = raw[:3]-(raw[:3]-equilibrium[:3])/.5003+(1-.5/.5003)*src[:3]
        trace = equilibrium[3]+equilibrium[4]+.5*(src[3]+src[4])
        dev = (raw[3]-raw[4])-(raw[3]-raw[4]-equilibrium[3]+equilibrium[4])/.8+(1-.5/.8)*(src[3]-src[4])
        expected[3:5] = [(trace+dev)/2, (trace-dev)/2]
        expected[5] = raw[5]-(raw[5]-equilibrium[5])/.8+(1-.5/.8)*src[5]
        np.testing.assert_allclose(out[i, :9], expected, rtol=2e-6, atol=6e-8)
        phase = equilibrium+.5*src
        phase[1:3] = raw[1:3]-1.25*(raw[1:3]-equilibrium[1:3])+.375*src[1:3]
        np.testing.assert_allclose(out[i, 9:18], phase, rtol=2e-6, atol=6e-8)
        np.testing.assert_allclose(transform@out[i, 18:], phase, rtol=2e-6, atol=6e-8)


@ti.kernel
def evaluate_boundary(out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for case in range(3):
        raw = ti.Vector([-.001, .02, -.01, .011, .03, -.01, .02, .003, -.002])
        u, vb = ti.Vector([.02, -.01]), ti.Vector([.005, .003])
        fraction = .05+.45*case
        for q in range(1, 9): out[case, q-1] = snibb(q, fraction, raw, .002*q, .003, .917, u, vb)


def test_snibb_uses_pre_collision_nonequilibrium():
    out = np.zeros((3, 8), np.float32); evaluate_boundary(out)
    raw = np.array([-.001, .02, -.01, .011, .03, -.01, .02, .003, -.002])
    def eq(q, u):
        e, w = np.array(E[q]), W[q]
        return w*(.009+.917*(3*e@u+4.5*(e@u)**2-1.5*u@u))-(.009 if q == 0 else 0)
    for case, fraction in enumerate((.05, .5, .95)):
        for q in range(1, 9):
            expected = (eq(OPP[q], np.array([.005, .003]))+.002*q-eq(q, np.array([.02, -.01]))+fraction*raw[OPP[q]])/(1+fraction)
            np.testing.assert_allclose(out[case, q-1], expected, rtol=2e-6, atol=2e-9)


@ti.kernel
def thermal_moments(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    old = ti.Vector([4.0, .03, .01, -.02, .08])
    result = thermal_post(7.0, .3, old, .4, .8)
    for k in range(5): out[k] = result[k]


def test_thermal_rest_stores_enthalpy_not_temperature():
    out = np.empty(5, np.float32); thermal_moments(out)
    np.testing.assert_allclose(out.sum(dtype=float), 7, atol=5e-7)
    np.testing.assert_allclose(out[1]+out[3], .4*.3/2, atol=1e-8)
    np.testing.assert_allclose(out[2]+out[4], .4*.3/2, atol=1e-8)
    np.testing.assert_allclose([out[1]-out[3], out[2]-out[4]], (1-1/.8)*np.array([.05, -.07]), atol=1e-8)
