"""D2Q9 helpers shared by the IceFlow2D Taichi kernels.

These are the minimal velocity/phase-lattice primitives used by mixture2d's
pure-fluid path.  Coupling-specific rigid-boundary kernels live in
``simulator.py`` rather than changing these collision-basis helpers.
"""

from __future__ import annotations

import taichi as ti


Q = 9
C = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
OPPOSITE = (0, 3, 4, 1, 2, 7, 8, 5, 6)


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
def _feq(q, rho, u):
    cq = ti.cast(_c(q), ti.f32)
    cu = cq.dot(u)
    uu = u.dot(u)
    return _w(q) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)


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
def _viscosity_mix(phi, rho_water, rho_air, nu_water, nu_air):
    p = ti.min(1.0, ti.max(0.0, phi))
    rho = _rho_mix(p, rho_water, rho_air)
    return (rho_water * nu_water * p + rho_air * nu_air * (1.0 - p)) / rho


@ti.func
def _tau_mix(phi, artificial_vis, rho_water, rho_air, nu_water, nu_air):
    return 3.0 * (_viscosity_mix(phi, rho_water, rho_air, nu_water, nu_air) + artificial_vis) + 0.5


@ti.func
def _axis_c(index):
    out = ti.Vector([0, 0], dt=ti.i32)
    if index == 0:
        out = ti.Vector([1, 0])
    elif index == 1:
        out = ti.Vector([0, 1])
    elif index == 2:
        out = ti.Vector([-1, 0])
    else:
        out = ti.Vector([0, -1])
    return out


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


__all__ = [
    "Q",
    "C",
    "W",
    "OPPOSITE",
    "_c",
    "_w",
    "_opp",
    "_feq",
    "_heq",
    "_inside",
    "_rho_mix",
    "_viscosity_mix",
    "_tau_mix",
    "_axis_c",
    "_inv1d",
    "_reconstruct_central",
    "_cross2",
]

