"""Convex polygon integration in cell-local coordinates (new geometry code)."""

import taichi as ti


@ti.func
def square():
    p = ti.Matrix.zero(ti.f64, 12, 2)
    for k in ti.static(range(4)):
        x = ti.static((-0.5, 0.5, 0.5, -0.5)[k])
        y = ti.static((-0.5, -0.5, 0.5, 0.5)[k])
        p[k, 0], p[k, 1] = x, y
    return p


@ti.func
def clip(p, count, normal, offset):
    # A cut material square has <=5 vertices; intersecting a world square
    # can produce 9. Eight slots are insufficient for a corner-cut octagon.
    out = ti.Matrix.zero(ti.f64, 12, 2)
    nout = 0
    for k in range(count):
        previous = (k + count - 1) % count
        a = ti.Vector([p[previous, 0], p[previous, 1]])
        b = ti.Vector([p[k, 0], p[k, 1]])
        da, db = normal.dot(a)-offset, normal.dot(b)-offset
        if (da <= 0.0) != (db <= 0.0):
            x = a + da/(da-db)*(b-a)
            if da == 0.0: x = a
            elif db == 0.0: x = b
            append = True
            if nout > 0: append = out[nout-1, 0] != x.x or out[nout-1, 1] != x.y
            if append:
                out[nout, 0], out[nout, 1] = x.x, x.y
                nout += 1
        if db <= 0.0:
            append = True
            if nout > 0: append = out[nout-1, 0] != b.x or out[nout-1, 1] != b.y
            if append:
                out[nout, 0], out[nout, 1] = b.x, b.y
                nout += 1
    if nout > 1:
        if out[nout-1, 0] == out[0, 0] and out[nout-1, 1] == out[0, 1]: nout -= 1
    return out, nout


@ti.func
def moments(p, count):
    """Area, first moments, polar second moment about the cell centre."""
    result = ti.Vector.zero(ti.f64, 4)
    for k in range(count):
        other = (k+1) % count
        ax, ay = p[k, 0], p[k, 1]
        bx, by = p[other, 0], p[other, 1]
        c = ax*by - ay*bx
        result += ti.Vector([0.5*c, c*(ax+bx)/6, c*(ay+by)/6,
                             c*(ax*ax+ax*bx+bx*bx+ay*ay+ay*by+by*by)/12])
    return result


@ti.func
def plic(fraction, normal):
    """Invert true clipped area, never interpret the legacy SDF as distance."""
    p, count = square(), 4
    offset = ti.cast(1.0, ti.f64)
    if fraction <= 0.0:
        count = 0
    elif fraction < 1.0:
        radius = 0.5*(ti.abs(normal.x)+ti.abs(normal.y))
        lo, hi = -radius, radius
        for _ in range(48):
            mid = 0.5*(lo+hi)
            candidate, n = clip(square(), 4, normal, mid)
            if moments(candidate, n)[0] < fraction:
                lo = mid
            else:
                hi = mid
        offset = 0.5*(lo+hi)
        p, count = clip(square(), 4, normal, offset)
    return p, count, offset


@ti.func
def world_overlap(p, count, origin, material_center, rotation, world_cell):
    local = ti.Matrix.zero(ti.f64, 12, 2)
    for k in range(count):
        # Transform shared material vertices identically. Rounding a different
        # world centre for each cell first creates microscopic overlaps/gaps.
        v = origin-world_cell + rotation@(material_center+ti.Vector([p[k, 0], p[k, 1]]))
        local[k, 0], local[k, 1] = v.x, v.y
    n = count
    for side in ti.static(range(4)):
        normal = ti.Vector([0.0, 0.0])
        sign = ti.static(1.0 if side % 2 == 0 else -1.0)
        normal[ti.static(side//2)] = sign
        local, n = clip(local, n, normal, 0.5)
    return ti.max(0.0, moments(local, n)[0])


@ti.func
def segment_interval(p, count, start, end):
    """Parameter interval of a segment inside a CCW convex polygon."""
    lo, hi = ti.cast(0.0, ti.f64), ti.cast(1.0, ti.f64)
    if count < 3:
        hi = -1.0
    for k in range(count):
        kk = (k+1) % count
        a = ti.Vector([p[k, 0], p[k, 1]])
        b = ti.Vector([p[kk, 0], p[kk, 1]])
        normal = ti.Vector([b.y-a.y, a.x-b.x])
        v0, v1 = normal.dot(start-a), normal.dot(end-a)
        # The geometry is f64. A fixed f32-sized distance tolerance closes a
        # genuinely open face beside a thin cut volume, producing an artificial
        # almost-isolated pressure component. Use the same half-plane signs as
        # polygon clipping, including arbitrarily small nonzero slopes.
        # Test both endpoints directly. Computing the endpoint as v0+slope
        # can round an exactly covered endpoint outside, exposing a phantom
        # sliver that prematurely selects the next material cell's front.
        if (v0 <= 0.0) != (v1 <= 0.0):
            root = v0/(v0-v1)
            if v1 > 0.0: hi = ti.min(hi, root)
            else: lo = ti.max(lo, root)
        elif v0 > 0.0 and v1 > 0.0:
            hi = -1.0
    return ti.Vector([ti.max(0.0, lo), ti.min(1.0, hi)])
