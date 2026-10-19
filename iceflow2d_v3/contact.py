"""Raster contact validation and frictionless, inelastic pair impulses.

The LBM solid nodes represent cell-centred unit squares, as in wall contact.
This raster query detects actual overlapping solid cells. Equal/opposite
normal impulses include each body's rotational
effective mass. Normals and penetration have grid-scale accuracy; this does
not resolve a subcell lubrication film or continuous polygon contact.
Time stepping uses BodyMotion's continuous eroded-material envelope followed
by this raster check. The integer-depth position projection below is retained
only for initialization/standalone raster queries.
"""

import numpy as np
import taichi as ti

from .lattice import cross_product_2d


@ti.data_oriented
class BodyContacts:
    def __init__(self, simulation):
        self.s = simulation
        pairs = [
            (a, b)
            for a in range(simulation.body_count)
            for b in range(a + 1, simulation.body_count)
        ]
        self.count = len(pairs)
        self.pairs = ti.Vector.field(2, ti.i32, shape=self.count)
        self.pairs.from_numpy(np.asarray(pairs, dtype=np.int32))
        self.normal = ti.Vector.field(2, ti.f64, shape=self.count)
        self.point = ti.Vector.field(2, ti.f64, shape=self.count)
        self.depth = ti.field(ti.f64, shape=self.count)

    @ti.func
    def _occupied(self, b, i, j, candidate: ti.template()):
        solid = False
        if 0 <= i < self.s.nx and 0 <= j < self.s.ny:
            fraction = 0.0
            if ti.static(candidate):
                fraction = self.s.body_motion.fraction_at(b, i, j)
            else:
                fraction = self.s._body_fractions[b, i, j]
            solid = fraction >= ti.static(self.s.config.thermal.solid_liquid_threshold)
        return solid

    @ti.func
    def _run_length(self, b, i, j, di, dj, candidate: ti.template()):
        length = 0
        while self._occupied(b, i + length * di, j + length * dj, candidate):
            length += 1
        return length

    @ti.func
    def detect_pair(self, p, candidate: ti.template()):
        total = 0
        # Shared by the legacy raster query and the bounded motion predictor.
        for _ in range(1):
            a, b = self.pairs[p].x, self.pairs[p].y
            self.depth[p] = 0.0
            self.normal[p] = ti.Vector.zero(ti.f64, 2)
            self.point[p] = ti.Vector.zero(ti.f64, 2)
            if self.s.body_active[a] != 0 and self.s.body_active[b] != 0:
                oa, ob = (
                    self.s.body_reference_origin[a],
                    self.s.body_reference_origin[b],
                )
                lo_x = ti.max(
                    0,
                    ti.cast(
                        ti.round(
                            ti.max(
                                oa.x + self.s._body_contact_support_extrema[a, 0],
                                ob.x + self.s._body_contact_support_extrema[b, 0],
                            )
                        ),
                        ti.i32,
                    ),
                )
                hi_x = ti.min(
                    self.s.nx,
                    ti.cast(
                        ti.round(
                            ti.min(
                                oa.x + self.s._body_contact_support_extrema[a, 1],
                                ob.x + self.s._body_contact_support_extrema[b, 1],
                            )
                        ),
                        ti.i32,
                    ),
                )
                lo_y = ti.max(
                    0,
                    ti.cast(
                        ti.round(
                            ti.max(
                                oa.y + self.s._body_contact_support_extrema[a, 2],
                                ob.y + self.s._body_contact_support_extrema[b, 2],
                            )
                        ),
                        ti.i32,
                    ),
                )
                hi_y = ti.min(
                    self.s.ny,
                    ti.cast(
                        ti.round(
                            ti.min(
                                oa.y + self.s._body_contact_support_extrema[a, 3],
                                ob.y + self.s._body_contact_support_extrema[b, 3],
                            )
                        ),
                        ti.i32,
                    ),
                )
                if ti.static(candidate):
                    lo_x = ti.max(
                        0, self.s.body_motion.lower[a].x, self.s.body_motion.lower[b].x
                    )
                    hi_x = ti.min(
                        self.s.nx,
                        self.s.body_motion.upper[a].x,
                        self.s.body_motion.upper[b].x,
                    )
                    lo_y = ti.max(
                        0, self.s.body_motion.lower[a].y, self.s.body_motion.lower[b].y
                    )
                    hi_y = ti.min(
                        self.s.ny,
                        self.s.body_motion.upper[a].y,
                        self.s.body_motion.upper[b].y,
                    )
                depths = ti.Vector.zero(ti.i32, 4)
                point = ti.Vector.zero(ti.f64, 2)
                overlaps = 0
                for i, j in ti.ndrange((lo_x, hi_x), (lo_y, hi_y)):
                    if self._occupied(a, i, j, candidate) and self._occupied(
                        b, i, j, candidate
                    ):
                        if ti.static(candidate):
                            self.s.body_motion.limit_pair_step(a, b, i, j)
                        # Translation needed to separate the two contiguous
                        # occupied runs through this witness cell in each axis.
                        for d in ti.static(range(4)):
                            di, dj = ti.static(((-1, 0), (1, 0), (0, -1), (0, 1))[d])
                            distance = (
                                self._run_length(a, i, j, -di, -dj, candidate)
                                + self._run_length(b, i, j, di, dj, candidate)
                                - 1
                            )
                            depths[d] = ti.max(depths[d], distance)
                        point += ti.Vector(
                            [ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5]
                        )
                        overlaps += 1
                if overlaps > 0:
                    selected = 0
                    for d in ti.static(range(1, 4)):
                        if depths[d] < depths[selected]:
                            selected = d
                    n = ti.Vector.zero(ti.f64, 2)
                    if selected < 2:
                        n.x = -1.0 if selected == 0 else 1.0
                    else:
                        n.y = -1.0 if selected == 2 else 1.0
                    self.depth[p] = ti.cast(depths[selected], ti.f64)
                    self.normal[p] = n
                    self.point[p] = point / overlaps
                    total += 1
        return total

    @ti.kernel
    def detect(self) -> ti.i32:
        total = 0
        for p in range(self.count):
            total += self.detect_pair(p, False)
        return total

    @ti.func
    def apply_impulse(self, p):
        a, b = self.pairs[p].x, self.pairs[p].y
        ma, mb = self.s.body_mass_lattice[a], self.s.body_mass_lattice[b]
        ia, ib = self.s.body_inertia_lattice[a], self.s.body_inertia_lattice[b]
        n, point = self.normal[p], self.point[p]
        ra = point - ti.cast(self.s.body_center[a], ti.f64)
        rb = point - ti.cast(self.s.body_center[b], ti.f64)
        va = ti.cast(self.s.body_velocity[a], ti.f64)
        vb = ti.cast(self.s.body_velocity[b], ti.f64)
        wa = ti.cast(self.s.body_angular_velocity[a], ti.f64)
        wb = ti.cast(self.s.body_angular_velocity[b], ti.f64)
        relative = (
            va + wa * ti.Vector([-ra.y, ra.x]) - vb - wb * ti.Vector([-rb.y, rb.x])
        )
        closing = relative.dot(n)
        ca, cb = cross_product_2d(ra, n), cross_product_2d(rb, n)
        if closing < 0.0:
            impulse = -closing / (1.0 / ma + 1.0 / mb + ca**2 / ia + cb**2 / ib)
            self.s.body_velocity[a] = ti.cast(va + impulse * n / ma, ti.f32)
            self.s.body_velocity[b] = ti.cast(vb - impulse * n / mb, ti.f32)
            self.s.body_angular_velocity[a] = ti.cast(wa + impulse * ca / ia, ti.f32)
            self.s.body_angular_velocity[b] = ti.cast(wb - impulse * cb / ib, ti.f32)

    @ti.kernel
    def project(self):
        # Sequential pair impulses avoid races when one body contacts several
        # neighbors. The parent re-rasterizes and iterates the joint constraints.
        ti.loop_config(serialize=True)
        for p in range(self.count):
            if self.depth[p] > 0.0:
                a, b = self.pairs[p].x, self.pairs[p].y
                ma, mb = self.s.body_mass_lattice[a], self.s.body_mass_lattice[b]
                n = self.normal[p]
                self.apply_impulse(p)
                correction = (self.depth[p] + 1.0e-4) * n
                self.s.body_reference_origin[a] += ti.cast(
                    correction * mb / (ma + mb), ti.f32
                )
                self.s.body_reference_origin[b] -= ti.cast(
                    correction * ma / (ma + mb), ti.f32
                )
