"""D2Q9 primitives used by the IceFlow2D Taichi kernels."""

import taichi as ti

Q = 9


@ti.func
def _c(q):
    out = ti.Vector([0, 0], dt=ti.i32)
    if q == 1:
        out = ti.Vector([1, 0])
    elif q == 2:
        out = ti.Vector([0, 1])
    elif q == 3:
        out = ti.Vector([-1, 0])
    elif q == 4:
        out = ti.Vector([0, -1])
    elif q == 5:
        out = ti.Vector([1, 1])
    elif q == 6:
        out = ti.Vector([-1, 1])
    elif q == 7:
        out = ti.Vector([-1, -1])
    elif q == 8:
        out = ti.Vector([1, -1])
    return out


@ti.func
def _w(q):
    out = ti.cast(4.0 / 9.0, ti.f32)
    if 1 <= q <= 4:
        out = ti.cast(1.0 / 9.0, ti.f32)
    elif q >= 5:
        out = ti.cast(1.0 / 36.0, ti.f32)
    return out


@ti.func
def _opp(q):
    out = 0
    if q == 1:
        out = 3
    elif q == 2:
        out = 4
    elif q == 3:
        out = 1
    elif q == 4:
        out = 2
    elif q == 5:
        out = 7
    elif q == 6:
        out = 8
    elif q == 7:
        out = 5
    elif q == 8:
        out = 6
    return out


@ti.func
def _pressure_eq(q, pressure, rho, u):
    """Liang et al. pressure--momentum equilibrium for large density ratios.

    Its zeroth moment is zero, its first moment is ``rho * u``, and its
    second moment is ``rho * u u + pressure * I``.  Consequently material
    density remains an independent phase-field quantity rather than an
    isothermal equation-of-state pressure.
    """

    cq = ti.cast(_c(q), ti.f32)
    cu = cq.dot(u)
    uu = u.dot(u)
    velocity_part = _w(q) * rho * (3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
    pressure_part = 3.0 * _w(q) * pressure
    if q == 0:
        pressure_part -= 3.0 * pressure
    return pressure_part + velocity_part


@ti.func
def _heq(q, phi, u):
    cq = ti.cast(_c(q), ti.f32)
    cu = cq.dot(u)
    uu = u.dot(u)
    return _w(q) * phi * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)


@ti.func
def _inside(i, j, nx, ny):
    return 0 <= i < nx and 0 <= j < ny


@ti.func
def _rho_mix(phi, rho_water, rho_air):
    p = ti.min(1.0, ti.max(0.0, phi))
    return rho_water * p + rho_air * (1.0 - p)


@ti.func
def _inv1d(c, u, k):
    out = 0.0
    if c == 0:
        if k == 0:
            out = 1.0 - u * u
        elif k == 1:
            out = -2.0 * u
        else:
            out = -1.0
    elif c == 1:
        if k == 0:
            out = 0.5 * u * (u + 1.0)
        elif k == 1:
            out = u + 0.5
        else:
            out = 0.5
    else:
        if k == 0:
            out = 0.5 * u * (u - 1.0)
        elif k == 1:
            out = u - 0.5
        else:
            out = 0.5
    return out


@ti.func
def _reconstruct_central(cxi, cyi, ux, uy, k00, k10, k01, k20, k02, k11, k21, k12, k22):
    ix0 = _inv1d(cxi, ux, 0)
    ix1 = _inv1d(cxi, ux, 1)
    ix2 = _inv1d(cxi, ux, 2)
    iy0 = _inv1d(cyi, uy, 0)
    iy1 = _inv1d(cyi, uy, 1)
    iy2 = _inv1d(cyi, uy, 2)
    return (
        ix0 * iy0 * k00
        + ix1 * iy0 * k10
        + ix0 * iy1 * k01
        + ix2 * iy0 * k20
        + ix0 * iy2 * k02
        + ix1 * iy1 * k11
        + ix2 * iy1 * k21
        + ix1 * iy2 * k12
        + ix2 * iy2 * k22
    )


@ti.func
def _cross2(a, b):
    return a.x * b.y - a.y * b.x
