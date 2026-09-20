"""Material PLIC polygons, exact cell overlaps and the approved SDF query.

Mass is the integrated polygon area, not the count of signed-distance pixels.
Only sample_fraction/query_sdf/cut_fraction follow the legacy whitelist.
"""
import math
import numpy as np
import taichi as ti
from .lattice import add_compensated
from .polygon import plic, moments, world_overlap, segment_interval, square
from .reduction import DeviceSum


@ti.data_oriented
class Geometry:
    def __init__(self, config):
        self.cfg = config
        self.nx, self.ny, self.nb = config.nx, config.ny, config.nb
        # A float64 oracle identified spurious O(1e-7) fluid pockets in fully
        # covered cells. Precision is raised only for geometry and its pose;
        # conserved phase/thermal/LB state remains compensated float32.
        self.origin = ti.Vector.field(2, ti.f64, ())
        self.origin_comp = ti.Vector.field(2, ti.f64, ())
        self.angle = ti.field(ti.f64, ())
        self.angle_comp = ti.field(ti.f64, ())
        self.rotation = ti.Matrix.field(2, 2, ti.f64, ())
        self.velocity = ti.Vector.field(3, ti.f32, ())
        self.solid = ti.field(ti.f32, (self.nb, self.nb))
        self.solid_comp = ti.field(ti.f32, (self.nb, self.nb))
        self.ice_heat = ti.field(ti.f32, (self.nb, self.nb))
        self.ice_heat_comp = ti.field(ti.f32, (self.nb, self.nb))
        self.ice_temperature = ti.field(ti.f32, (self.nb, self.nb))
        self.poly = ti.Matrix.field(12, 2, ti.f64, (self.nb, self.nb))
        self.count = ti.field(ti.i32, (self.nb, self.nb))
        self.normal = ti.Vector.field(2, ti.f64, (self.nb, self.nb))
        self.offset = ti.field(ti.f64, (self.nb, self.nb))
        self.integrals = ti.Vector.field(4, ti.f64, (self.nb, self.nb))
        self.sdf = ti.field(ti.f32, (self.nx, self.ny))
        self.mask = ti.field(ti.i32, (self.nx, self.ny))
        self.old_mask = ti.field(ti.i32, (self.nx, self.ny))
        self.occupancy = ti.field(ti.f64, (self.nx, self.ny))
        self.capacity = ti.field(ti.f64, (self.nx, self.ny))
        self.old_capacity = ti.field(ti.f64, (self.nx, self.ny))
        self.face_open = ti.Vector.field(2, ti.f64, (self.nx, self.ny))
        self.old_face_open = ti.Vector.field(2, ti.f64, (self.nx, self.ny))
        self.overlap = ti.field(ti.f64, (self.nb, self.nb, 9))
        self.full_overlap = ti.field(ti.f64, (self.nb, self.nb, 9))
        self.overlap_index = ti.Vector.field(2, ti.i32, (self.nb, self.nb, 9))
        self.mass_matrix = ti.Matrix.field(3, 3, ti.f64, ())
        self.mass = ti.field(ti.f64, ())
        self.first_moment = ti.Vector.field(2, ti.f64, ())
        self.inertia = ti.field(ti.f64, ())
        self.surface_point = ti.Vector.field(2, ti.f64, (self.nb, self.nb, 12))
        self.surface_local = ti.Vector.field(2, ti.f64, (self.nb, self.nb, 12))
        self.surface_normal = ti.Vector.field(2, ti.f64, (self.nb, self.nb, 12))
        self.surface_length = ti.field(ti.f64, (self.nb, self.nb, 12))
        self.surface_edge = ti.field(ti.i32, (self.nb, self.nb, 12))
        self.mass_sum = DeviceSum(self.nb*self.nb, 4, ti.f64)
        self.origin[None] = config.initial_origin
        self.angle[None] = math.radians(config.angle_deg)
        self.solid.fill(1.0)
        self.ice_heat.fill(config.ice_capacity_lu*config.ice_temperature/config.temperature_scale)
        self.ice_temperature.fill(config.ice_temperature/config.temperature_scale)
        self.reconstruct()
        self.rasterize()
        self.remember_geometry()

    @ti.func
    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    @ti.func
    def wall(self, i, j):
        b = ti.static(self.cfg.wall_cells)
        return i < b or j < b or i >= self.nx-b or j >= self.ny-b

    @ti.func
    def fluid(self, i, j):
        return self.inside(i, j) and not self.wall(i, j) and self.mask[i, j] == 0

    @ti.func
    def swept_aperture(self, i, j, ni, nj):
        value = 0.0
        if self.inside(ni, nj) and not self.wall(i, j) and not self.wall(ni, nj):
            left = ti.max(self.capacity[i, j], self.old_capacity[i, j])
            right = ti.max(self.capacity[ni, nj], self.old_capacity[ni, nj])
            if left > 0.0 and right > 0.0:
                x, y = ti.min(i, ni), ti.min(j, nj)
                if ni == i: value = ti.max(self.face_open[i, y].y, self.old_face_open[i, y].y)
                elif nj == j: value = ti.max(self.face_open[x, j].x, self.old_face_open[x, j].x)
                else:
                    first = ti.min(ti.max(self.face_open[x, j].x, self.old_face_open[x, j].x),
                                   ti.max(self.face_open[ni, y].y, self.old_face_open[ni, y].y))
                    second = ti.min(ti.max(self.face_open[i, y].y, self.old_face_open[i, y].y),
                                    ti.max(self.face_open[x, nj].x, self.old_face_open[x, nj].x))
                    value = 0.5*(first+second)
        return value

    @ti.func
    def material_fraction(self, a, b):
        value = ti.cast(0.0, ti.f64)
        if 0 <= a < self.nb and 0 <= b < self.nb:
            value = ti.cast(self.solid[a, b], ti.f64)-ti.cast(self.solid_comp[a, b], ti.f64)
        return value

    @ti.func
    def sample_fraction(self, local):
        grid = local+0.5*self.nb-0.5
        base = ti.cast(ti.floor(grid), ti.i32)
        fraction = 0.0
        for di, dj in ti.static(ti.ndrange(2, 2)):
            a, b = base.x+di, base.y+dj
            fraction += (ti.max(0.0, 1.0-ti.abs(grid.x-a))
                         *ti.max(0.0, 1.0-ti.abs(grid.y-b))*self.material_fraction(a, b))
        return ti.min(1.0, ti.max(0.0, fraction))

    @ti.func
    def query_sdf(self, point):
        local = self.rotation[None].transpose()@(point-self.origin[None])
        d = ti.abs(local)-0.5*self.nb
        distance = ti.max(d, 0.0).norm()+ti.min(ti.max(d.x, d.y), 0.0)
        nearest = ti.cast(ti.floor(local+0.5*self.nb), ti.i32)
        if 0 <= nearest.x < self.nb and 0 <= nearest.y < self.nb:
            distance = 0.5-self.sample_fraction(local)
        return distance

    @ti.func
    def body_velocity(self, point):
        r, v = point-self.origin[None], self.velocity[None]
        return ti.Vector([v.x-v.z*r.y, v.y+v.z*r.x])

    @ti.func
    def cut_fraction(self, i, j, ni, nj):
        d = ti.max(self.sdf[i, j], ti.static(1e-6/self.cfg.dx))
        return ti.min(0.95, ti.max(0.05, d/(d-self.sdf[ni, nj]+ti.static(1e-12/self.cfg.dx))))

    @ti.kernel
    def update_rotation(self):
        c, s = ti.cos(self.angle[None]), ti.sin(self.angle[None])
        self.rotation[None] = ti.Matrix([[c, -s], [s, c]])

    @ti.kernel
    def reconstruct(self):
        for a, b in self.solid:
            n = ti.Vector([self.material_fraction(a-1, b)-self.material_fraction(a+1, b),
                           self.material_fraction(a, b-1)-self.material_fraction(a, b+1)])
            # Melting must produce nested material polygons. Re-estimating a
            # partially cut cell's normal can add ice on one side while removing
            # more elsewhere, creating sealed liquid pockets with no GCL source.
            # Choose the normal when a full cell first becomes cut, then retreat
            # that plane until the cell is exhausted (first-order PLIC front).
            old_area = self.integrals[a, b][0]
            if 0.0 < old_area < 1.0 and self.normal[a, b].norm_sqr() > 0.0:
                n = self.normal[a, b]
            if n.norm_sqr() < 1e-14:
                n = self.normal[a, b]
                if n.norm_sqr() < 1e-14:
                    n = ti.Vector([1.0, 0.0])
            n = n/n.norm()
            p, count, d = plic(self.material_fraction(a, b), n)
            self.poly[a, b], self.count[a, b] = p, count
            self.normal[a, b], self.offset[a, b] = n, d
            self.integrals[a, b] = moments(p, count)

    @ti.kernel
    def build_overlaps(self):
        for a, b in self.solid:
            center = self.origin[None]+self.rotation[None]@ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb])
            base = ti.cast(ti.floor(center), ti.i32)-1
            for k in range(9):
                i, j = base.x+k//3, base.y+k%3
                self.overlap_index[a, b, k] = ti.Vector([i, j])
                self.overlap[a, b, k] = world_overlap(
                    self.poly[a, b], self.count[a, b], self.origin[None],
                    ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb]),
                    self.rotation[None], ti.Vector([i+0.5, j+0.5]))
                full = self.overlap[a, b, k]
                if self.material_fraction(a, b) < 1.0:
                    full = world_overlap(square(), 4, self.origin[None],
                        ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb]),
                        self.rotation[None], ti.Vector([i+0.5, j+0.5]))
                self.full_overlap[a, b, k] = full

    @ti.kernel
    def erode(self, accepted_mass: ti.template()):
        # A full cell starts retreat from its actually heated exposed face.
        # Keep this plane through the cell's lifetime: independently retreating
        # every partly exposed grid face opens artificial inter-cell channels.
        # The heat contact centroid selects the local first-order reconstruction;
        # all accepted face heat still enters the one conservative mass ledger.
        for a, b in self.solid:
            rates = ti.Vector.zero(ti.f64, 12)
            total = ti.cast(0.0, ti.f64)
            contact = ti.Vector.zero(ti.f64, 2)
            for k in range(12):
                dm = ti.cast(accepted_mass[a, b, k], ti.f64)
                if dm > 0.0:
                    rates[self.surface_edge[a, b, k]] += dm
                    total += dm
                    contact += dm*self.surface_local[a, b, k]
            if total > 0.0:
                normal = self.normal[a, b]
                if self.integrals[a, b][0] >= 1.0:
                    best, edge = ti.cast(-1.0, ti.f64), 0
                    for k in range(self.count[a, b]):
                        if rates[k] > best: best, edge = rates[k], k
                    other = (edge+1) % self.count[a, b]
                    poly = self.poly[a, b]
                    delta = ti.Vector([poly[other, 0]-poly[edge, 0], poly[other, 1]-poly[edge, 1]])
                    normal = ti.Vector([delta.y, -delta.x])/delta.norm()
                    # Heat on a small exposed end of a grid face melts its
                    # corner first. Retreating the entire face opens a deep
                    # artificial channel along otherwise covered neighbours.
                    # The heat centroid gives a connected corner PLIC; a
                    # uniformly heated planar face keeps its exact normal.
                    if contact.norm_sqr() > 0.0: normal = contact/contact.norm()
                poly, count, offset = plic(self.material_fraction(a, b), normal)
                self.poly[a, b], self.count[a, b] = poly, count
                self.normal[a, b], self.offset[a, b] = normal, offset
                self.integrals[a, b] = moments(poly, count)

    @ti.kernel
    def gather_capacity(self):
        for i, j in self.capacity:
            point = ti.Vector([i+0.5, j+0.5])
            local = self.rotation[None].transpose()@(point-self.origin[None])
            base = ti.cast(ti.floor(local+0.5*self.nb), ti.i32)
            occupied = ti.cast(0.0, ti.f64)
            completely_solid = True
            for ci, cj in ti.static(ti.ndrange(2, 2)):
                corner = self.rotation[None].transpose()@(ti.Vector([i+ci, j+cj])-self.origin[None])
                if ti.abs(corner.x) >= 0.5*self.nb or ti.abs(corner.y) >= 0.5*self.nb:
                    completely_solid = False
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):
                a, b = base.x+di, base.y+dj
                if 0 <= a < self.nb and 0 <= b < self.nb:
                    for k in range(9):
                        target = self.overlap_index[a, b, k]
                        if target.x == i and target.y == j:
                            occupied += self.overlap[a, b, k]
                            if self.full_overlap[a, b, k]-self.overlap[a, b, k] > 2e-14:
                                completely_solid = False
            # Exactly covered cells have zero fluid capacity. Resolving a
            # roundoff-sized interior "pocket" would make the GCL graph singular.
            if completely_solid: occupied = 1.0
            self.occupancy[i, j] = occupied
            self.sdf[i, j] = self.query_sdf(point)
            self.mask[i, j] = ti.cast(self.sdf[i, j] < 0.0 and occupied > 0.0, ti.i32)
            # Roundoff in adjacent polygon sums cannot create negative capacity.
            # Keep the unmodified occupancy as the geometry-error diagnostic.
            self.capacity[i, j] = ti.max(0.0, 1.0-occupied)
            # The approved smoothed SDF can be positive even when the actual
            # polygons completely cover this world cell. Such a cell has no
            # fluid state: colliding it with phi=0 would invent a gas interface
            # and a large capillary force. This exact-volume classification
            # changes neither the SDF/cut-link formula nor any positive volume.
            if self.capacity[i, j] == 0.0:
                self.mask[i, j] = 1
            if self.wall(i, j):
                self.capacity[i, j], self.mask[i, j] = 0.0, 0

    @ti.kernel
    def build_surfaces(self):
        for a, b in self.solid:
            for k in range(12):
                self.surface_length[a, b, k] = 0.0
            poly, count = self.poly[a, b], self.count[a, b]
            output = 0
            for k in range(count):
                kk = (k+1) % count
                start = ti.Vector([poly[k, 0], poly[k, 1]])
                end = ti.Vector([poly[kk, 0], poly[kk, 1]])
                delta, mid = end-start, 0.5*(start+end)
                length = delta.norm()
                if length > 0.0:
                    normal = ti.Vector([delta.y, -delta.x])/length
                    da, db = 0, 0
                    if ti.abs(mid.x) == 0.5 and delta.x == 0.0:
                        da = ti.cast(ti.math.sign(mid.x), ti.i32)
                    elif ti.abs(mid.y) == 0.5 and delta.y == 0.0:
                        db = ti.cast(ti.math.sign(mid.y), ti.i32)
                    # Exclude precisely the interval shared with a neighbouring polygon.
                    covered_lo, covered_hi = ti.cast(1.0, ti.f64), ti.cast(0.0, ti.f64)
                    na, nb = a+da, b+db
                    if (da != 0 or db != 0) and 0 <= na < self.nb and 0 <= nb < self.nb:
                        if self.material_fraction(na, nb) > 0.0:
                            shift = ti.Vector([da, db])
                            covered = segment_interval(self.poly[na, nb], self.count[na, nb], start-shift, end-shift)
                            covered_lo, covered_hi = covered.x, covered.y
                    for part in ti.static(range(2)):
                        lo, hi = ti.cast(0.0, ti.f64), ti.min(1.0, covered_lo)
                        if ti.static(part == 1):
                            lo, hi = ti.max(0.0, covered_hi), 1.0
                        if covered_hi < covered_lo:
                            lo, hi = 0.0, 1.0
                            if ti.static(part == 1): hi = 0.0
                        if hi > lo and output < 12:
                            local = ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb])+start+0.5*(lo+hi)*delta
                            self.surface_local[a, b, output] = start+0.5*(lo+hi)*delta
                            self.surface_point[a, b, output] = self.origin[None]+self.rotation[None]@local
                            self.surface_normal[a, b, output] = self.rotation[None]@normal
                            self.surface_length[a, b, output] = (hi-lo)*length
                            self.surface_edge[a, b, output] = k
                            output += 1

    @ti.kernel
    def integrate_open_faces(self):
        for i, j in self.face_open:
            opening = ti.Vector.zero(ti.f64, 2)
            for axis in ti.static(range(2)):
                ni, nj = i+1-axis, j+axis
                if self.inside(ni, nj) and not self.wall(i, j) and not self.wall(ni, nj):
                    start = ti.Vector([i+1.0, j+0.0])
                    end = ti.Vector([i+1.0, j+1.0])
                    if ti.static(axis == 1):
                        start = ti.Vector([i+0.0, j+1.0])
                    local_start = self.rotation[None].transpose()@(start-self.origin[None])
                    local_end = self.rotation[None].transpose()@(end-self.origin[None])
                    base = ti.cast(ti.floor(0.5*(local_start+local_end)+0.5*self.nb), ti.i32)
                    intervals = ti.Matrix.zero(ti.f64, 9, 2)
                    for k in range(9):
                        a, b = base.x+k//3-1, base.y+k % 3-1
                        interval = ti.Vector([2.0, -1.0], dt=ti.f64)
                        if 0 <= a < self.nb and 0 <= b < self.nb:
                            center = ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb])
                            interval = segment_interval(self.poly[a, b], self.count[a, b], local_start-center, local_end-center)
                            if interval.y <= interval.x: interval = ti.Vector([2.0, -1.0])
                        intervals[k, 0], intervals[k, 1] = interval.x, interval.y
                    # Union intervals, so a face coincident with two material
                    # cell edges is never counted twice (including zero angle).
                    for k in range(1, 9):
                        lo, hi = intervals[k, 0], intervals[k, 1]
                        index = k
                        while index > 0 and intervals[index-1, 0] > lo:
                            intervals[index, 0], intervals[index, 1] = intervals[index-1, 0], intervals[index-1, 1]
                            index -= 1
                        intervals[index, 0], intervals[index, 1] = lo, hi
                    covered, high = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
                    for k in range(9):
                        lo, hi = intervals[k, 0], intervals[k, 1]
                        if hi > ti.max(high, lo): covered += hi-ti.max(high, lo)
                        high = ti.max(high, hi)
                    opening[axis] = ti.max(0.0, 1.0-covered)
            self.face_open[i, j] = opening

    @ti.func
    def mass_value(self, index):
        a, b = index//self.nb, index % self.nb
        m = self.integrals[a, b]
        center = ti.Vector([a+0.5-0.5*self.nb, b+0.5-0.5*self.nb])
        first = ti.Vector([m[1], m[2]])
        r = self.rotation[None]@(m[0]*center+first)
        inertia = m[3]+2*center.dot(first)+m[0]*center.dot(center)
        return ti.static(self.cfg.rho_s_lu)*ti.Vector([m[0], r.x, r.y, inertia])

    @ti.func
    def mass_result(self, sums):
        m, sx, sy, ii = sums[0], sums[1], sums[2], sums[3]
        self.mass[None], self.first_moment[None] = m, ti.Vector([sx, sy])
        self.inertia[None] = ii-(sx*sx+sy*sy)/ti.max(m, 1e-30)
        self.mass_matrix[None] = ti.Matrix([[m, 0.0, -sy], [0.0, m, sx], [-sy, sx, ii]])

    def update_mass(self):
        self.mass_sum.run(self.mass_value, self.mass_result)

    @ti.kernel
    def remember_geometry(self):
        for i, j in self.capacity:
            self.old_capacity[i, j], self.old_mask[i, j] = self.capacity[i, j], self.mask[i, j]
            self.old_face_open[i, j] = self.face_open[i, j]

    def rasterize(self):
        self.update_rotation()
        self.build_overlaps()
        self.gather_capacity()
        self.integrate_open_faces()
        self.build_surfaces()
        self.update_mass()

    def totals(self):
        mass = self.solid.to_numpy().astype(np.float64)-self.solid_comp.to_numpy()
        heat = self.ice_heat.to_numpy().astype(np.float64)-self.ice_heat_comp.to_numpy()
        return float(mass.sum()*self.cfg.rho_s_lu), float(heat.sum())
