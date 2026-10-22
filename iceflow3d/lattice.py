"""D3Q27 tensor-product pressure/momentum and Allen-Cahn primitives.

q = 9*(cx+1) + 3*(cy+1) + (cz+1), so the rest direction is 13.
Central transforms are separable 3x3 contractions, not dense 27x27 solves.
"""

import taichi as ti

from .viscosity import WATER_VISCOSITY_TERMS

Q = 27
REST = 13
DIRECTIONS = tuple(
    (i - 1, j - 1, k - 1) for i in range(3) for j in range(3) for k in range(3)
)
WEIGHTS = tuple(
    (2 / 3 if x == 0 else 1 / 6)
    * (2 / 3 if y == 0 else 1 / 6)
    * (2 / 3 if z == 0 else 1 / 6)
    for x, y, z in DIRECTIONS
)
W0 = WEIGHTS[REST]
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


@ti.func
def direction(q):
    return ti.Vector([q // 9 - 1, q // 3 % 3 - 1, q % 3 - 1])


@ti.func
def weight(q):
    c = direction(q)
    # Round each exact quadrature weight once. Multiplying three rounded
    # f32 factors biases the first equilibrium moment upward every step;
    # that bias is measurable at molecular water viscosities.
    order = ti.abs(c).sum()
    w = 1 / 216
    if order == 0:
        w = 8 / 27
    elif order == 1:
        w = 2 / 27
    elif order == 2:
        w = 1 / 54
    return w


@ti.func
def mixture_kinematic_viscosity(phi, temperature, rho_water, rho_air, mu_air):
    """Interpolate dynamic viscosity, then divide by mixture density (SI)."""
    fraction = ti.min(1.0, ti.max(0.0, phi))
    mu = fraction * water_viscosity(temperature) + (1 - fraction) * mu_air
    rho = fraction * rho_water + (1 - fraction) * rho_air
    return mu / rho


@ti.func
def density(phi, rho_air):
    return rho_air + (1 - rho_air) * ti.min(1.0, ti.max(0.0, phi))


@ti.func
def momentum_equilibrium(q, pressure, rho, u):
    c = ti.cast(direction(q), ti.f32)
    increment = 0.0
    for a in ti.static(range(3)):
        factor = 3 * c[a] * u[a] + 1.5 * (3 * c[a] ** 2 - 1) * u[a] ** 2
        increment += (1 + increment) * factor
    value = weight(q) * (3 * pressure + rho * increment)
    if q == REST:
        value -= 3 * pressure
    return value


@ti.func
def momentum_rest_velocity(rho, u):
    increment = 0.0
    for a in ti.static(range(3)):
        increment -= (1 + increment) * 1.5 * u[a] ** 2
    return W0 * rho * increment


@ti.func
def phase_equilibrium(q, phi, u):
    cu = ti.cast(direction(q), ti.f32).dot(u)
    return weight(q) * phi * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u.dot(u))


@ti.func
def inverse_basis(c, u, p):
    value = 0.0
    if c == 0:
        if p == 0:
            value = 1 - u * u
        elif p == 1:
            value = -2 * u
        else:
            value = -1.0
    else:
        if p == 0:
            value = 0.5 * u * (u + c)
        elif p == 1:
            value = u + 0.5 * c
        else:
            value = 0.5
    return value


@ti.func
def transform(values, u, inverse: ti.template()):
    current = values
    for axis in ti.static(range(3)):
        result = ti.Vector.zero(ti.f32, Q)
        for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
            for t in ti.static(range(3)):
                index = ti.static((a, b, c)[axis])
                source = ti.static(
                    (t * 9 + b * 3 + c, a * 9 + t * 3 + c, a * 9 + b * 3 + t)[axis]
                )
                basis = 0.0
                if ti.static(inverse):
                    basis = inverse_basis(index - 1, u[axis], t)
                else:
                    basis = (t - 1 - u[axis]) ** index
                result[a * 9 + b * 3 + c] += basis * current[source]
        current = result
    return current


@ti.func
def water_viscosity(temperature):
    t = (temperature + 273.15) / 300.0
    mu = 0.0
    for coefficient, exponent in ti.static(WATER_VISCOSITY_TERMS):
        mu += coefficient * t**exponent
    return mu * 1e-6
