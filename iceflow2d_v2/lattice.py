"""The three approved D2Q9 ingredients, expressed without importing iceflow2d.

Pressure equilibrium, phase equilibrium and MRT transforms follow the approved
lattice.py/collision formulas. Thermal moments are derived in derivation/.
"""

import taichi as ti

E = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
     (1, 1), (-1, 1), (-1, -1), (1, -1))
W = (4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)
ORDERS = ((0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (2,1), (1,2), (2,2))


@ti.func
def direction(q):
    ex, ey = 0, 0
    if q == 1: ex = 1
    elif q == 2: ey = 1
    elif q == 3: ex = -1
    elif q == 4: ey = -1
    elif q == 5: ex, ey = 1, 1
    elif q == 6: ex, ey = -1, 1
    elif q == 7: ex, ey = -1, -1
    elif q == 8: ex, ey = 1, -1
    return ti.Vector([ex, ey])


@ti.func
def weight(q):
    w = 4.0 / 9.0
    if q > 0: w = 1.0 / 9.0
    if q > 4: w = 1.0 / 36.0
    return w


@ti.func
def opposite(q):
    out = q + 2
    if q == 0: out = 0
    elif q == 3 or q == 4: out = q - 2
    elif q == 7 or q == 8: out = q - 2
    return out


@ti.func
def pressure_equilibrium(q, p, rho, u):
    eu = ti.cast(direction(q), ti.f32).dot(u)
    value = weight(q) * (3.0*p + rho*(3.0*eu + 4.5*eu*eu - 1.5*u.dot(u)))
    if q == 0: value -= 3.0*p
    return value


@ti.func
def phase_equilibrium(q, phi, u):
    eu = ti.cast(direction(q), ti.f32).dot(u)
    return weight(q)*phi*(1.0+3.0*eu+4.5*eu*eu-1.5*u.dot(u))


@ti.func
def inverse_basis(e, u, order: ti.template()):
    b = 0.0
    if e == 0:
        if ti.static(order == 0): b = 1.0-u*u
        elif ti.static(order == 1): b = -2.0*u
        else: b = -1.0
    elif e == 1:
        if ti.static(order == 0): b = 0.5*u*(u+1.0)
        elif ti.static(order == 1): b = u+0.5
        else: b = 0.5
    else:
        if ti.static(order == 0): b = 0.5*u*(u-1.0)
        elif ti.static(order == 1): b = u-0.5
        else: b = 0.5
    return b


@ti.func
def basis_moments(x, y):
    return ti.Vector([1.0, x, y, x*x, y*y, x*y, x*x*y, x*y*y, x*x*y*y])


@ti.func
def reconstruct(q, u, moments):
    e = direction(q)
    value = 0.0
    for k in ti.static(range(9)):
        value += (inverse_basis(e.x, u.x, ORDERS[k][0])
                  * inverse_basis(e.y, u.y, ORDERS[k][1]) * moments[k])
    return value


@ti.func
def momentum_collision(raw, eq, source, tau, tau_shear):
    result = eq + 0.5*source
    w, ws = 1.0/tau, 1.0/tau_shear
    for k in ti.static(range(3)):
        result[k] = raw[k]-w*(raw[k]-eq[k])+(1.0-0.5*w)*source[k]
    trace = eq[3]+eq[4]+0.5*(source[3]+source[4])
    dev = raw[3]-raw[4]-ws*(raw[3]-raw[4]-eq[3]+eq[4])
    dev += (1.0-0.5*ws)*(source[3]-source[4])
    result[3], result[4] = 0.5*(trace+dev), 0.5*(trace-dev)
    result[5] = raw[5]-ws*(raw[5]-eq[5])+(1.0-0.5*ws)*source[5]
    return result


@ti.func
def phase_collision(raw, eq, source, omega):
    result = eq + 0.5*source
    for k in ti.static((1,2)):
        result[k] = raw[k]-omega*(raw[k]-eq[k])+(1.0-0.5*omega)*source[k]
    return result


@ti.func
def volume_source_population(q, density, dilatation):
    """Zero-rest pressure source: sum B=rho*s, sum e B=0; not mass.

    See research/coupling_discretization_zh.md. The source is an additional
    coupling operator; none of the approved source-free collision rates change.
    """
    value = 0.0
    if q != 0:
        value = 1.8 * weight(q) * density * dilatation
    return value


@ti.func
def snibb(q, fraction, raw_post, pre, pressure, density, velocity, boundary_velocity):
    opposite_q = opposite(q)
    neq_pre = pre - pressure_equilibrium(q, pressure, density, velocity)
    return (pressure_equilibrium(opposite_q, pressure, density, boundary_velocity)
            + neq_pre + fraction * raw_post[opposite_q]) / (1.0 + fraction)


@ti.func
def thermal_post(enthalpy, temperature, populations, varpi, tau):
    """D2Q5 MRT, s0=0, s1=s2=1/tau, s3=s4=1."""
    mx = (1.0 - 1.0/tau) * (populations[1] - populations[3])
    my = (1.0 - 1.0/tau) * (populations[2] - populations[4])
    moving = 0.25 * varpi * temperature
    return ti.Vector([enthalpy-varpi*temperature, moving+0.5*mx,
                      moving+0.5*my, moving-0.5*mx, moving-0.5*my])


@ti.func
def add_compensated(value, correction, increment, precision: ti.template() = ti.f32):
    # TwoSum retains the low inventory even when a nearly complete removal
    # cancels the high part. Ordinary Kahan loses it in 1 + 1e-8 - 1 (f32).
    # Preserve a geometric f64 increment's low part as well.
    high = ti.cast(increment, precision)
    low = ti.cast(increment-high, precision)
    total = ti.cast(value+high, precision)
    back = ti.cast(total-value, precision)
    error = ti.cast(ti.cast(value-ti.cast(total-back, precision), precision)
                    +ti.cast(high-back, precision), precision)
    tail = ti.cast(ti.cast(error-correction, precision)+low, precision)
    updated = ti.cast(total+tail, precision)
    correction = ti.cast(ti.cast(updated-total, precision)-tail, precision)
    return updated, correction


@ti.func
def cross(a, b):
    return a.x*b.y-a.y*b.x
