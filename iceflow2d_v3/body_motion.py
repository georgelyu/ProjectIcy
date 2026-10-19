"""Lightweight contact prediction on the geometry seen by the LBM.

Candidate coverage is sampled in small tiles; pair checks read only their
intersection boxes. Current eroded-material bounds provide conservative,
continuous wall/pair contacts with at most four small-body constraint sweeps.
The accepted tiles are committed after fluid streaming. A linear occupancy
crossing estimate and finally the old pose handle any remaining raster
overlap, including legacy checkpoints with interpenetrating envelopes.
"""

import math

import taichi as ti


@ti.data_oriented
class BodyMotion:
    def __init__(self, simulation):
        self.s = simulation
        n = simulation.body_count
        self.origin = ti.Vector.field(2, ti.f32, shape=n)
        self.angle = ti.field(ti.f32, shape=n)
        self.rotation = ti.Vector.field(2, ti.f32, shape=n)
        self.lower = ti.Vector.field(2, ti.i32, shape=n)
        self.upper = ti.Vector.field(2, ti.i32, shape=n)
        tile = max(
            math.ceil(math.hypot(b.size_cells[0] + 1, b.size_cells[1] + 1)) + 4
            for b in simulation.config.bodies
        )
        # Compact per-body tiles let CUDA evaluate bilinear coverage in
        # parallel, rather than resampling it in serial contact run scans.
        self.coverage = ti.field(ti.f32, shape=(n, tile, tile))
        self.affected = ti.field(ti.i32, shape=n)
        self.fraction = ti.field(ti.f32, shape=n)
        self.wall_sides = ti.field(ti.i32, shape=n)
        self.wall_range = ti.Vector.field(2, ti.f32, shape=(n, 4))
        self.wall_velocity = ti.Vector.field(2, ti.f32, shape=n)
        self.wall_omega = ti.field(ti.f32, shape=n)
        self.active = ti.field(ti.i32, shape=())
        self.retry_scale = ti.field(ti.f32, shape=n)
        self.changed = ti.field(ti.i32, shape=n)
        self.contact_links = ti.field(ti.i32, shape=max(1, n * (n - 1) // 2))
        self.pair_limit = ti.field(ti.f64, shape=max(1, n * (n - 1) // 2))
        self.material_lower = ti.Vector.field(2, ti.f32, shape=n)
        self.material_upper = ti.Vector.field(2, ti.f32, shape=n)
        self.wall_limit = ti.field(ti.f64, shape=(n, 4))
        self.wall_reference_ready = ti.field(ti.i32, shape=())
        # Sparse diagnostics, never used as a physical mass/momentum reservoir.
        self.contact_steps = ti.field(ti.i32, shape=n)
        self.reduced_steps = ti.field(ti.i32, shape=n)
        self.held_steps = ti.field(ti.i32, shape=n)

    @ti.kernel
    def refresh_material_bounds(self):
        """Enclose the current ice isocontour in material coordinates.

        Axis extrema of bilinear coverage occur on material grid edges.
        Four linear crossings per live material node suffice. The rotated
        enclosing box is a conservative, inexpensive wall-contact shape.
        """
        for b in range(self.s.body_count):
            self.material_lower[b] = ti.Vector([1e30, 1e30])
            self.material_upper[b] = ti.Vector([-1e30, -1e30])
        for i, j in self.s.thermal.body_solid_mass:
            b = self.s.thermal.material_body_id[i, j]
            if b >= 0:
                value = ti.cast(self.s.thermal.body_solid_fraction[i, j], ti.f32)
                threshold = ti.static(self.s.config.thermal.solid_liquid_threshold)
                if value >= threshold:
                    offset = self.s.thermal.material_offset[b]
                    size = self.s.thermal.material_size[b]
                    local = ti.Vector([i - offset + 0.5, j + 0.5]) - 0.5 * ti.cast(
                        size, ti.f32
                    )
                    low, high = local, local
                    for axis, sign in ti.static(ti.ndrange(2, 2)):
                        direction = 2 * sign - 1
                        ni, nj = i, j
                        if ti.static(axis == 0):
                            ni += direction
                        else:
                            nj += direction
                        neighbor = 0.0
                        if offset <= ni < offset + size.x and 0 <= nj < size.y:
                            neighbor = ti.cast(
                                self.s.thermal.body_solid_fraction[ni, nj], ti.f32
                            )
                        if neighbor < threshold:
                            crossing = local[axis] + direction * (value - threshold) / (
                                value - neighbor
                            )
                            if ti.static(sign == 0):
                                low[axis] = crossing
                            else:
                                high[axis] = crossing
                    for axis in ti.static(range(2)):
                        ti.atomic_min(self.material_lower[b][axis], low[axis])
                        ti.atomic_max(self.material_upper[b][axis], high[axis])

    @ti.func
    def _wall_support(self, b, side: ti.template(), candidate: ti.template()):
        origin = ti.cast(self.s.body_reference_origin[b], ti.f64)
        c, sn = ti.cos(self.s.body_angle[b]), ti.sin(self.s.body_angle[b])
        if ti.static(candidate):
            origin = ti.cast(self.origin[b], ti.f64)
            c, sn = self.rotation[b].x, self.rotation[b].y
        support, lower, upper = (
            ti.cast(1e30, ti.f64),
            ti.cast(1e30, ti.f64),
            ti.cast(-1e30, ti.f64),
        )
        for ix, iy in ti.static(ti.ndrange(2, 2)):
            local = ti.cast(self.material_lower[b], ti.f64)
            if ti.static(ix):
                local.x = self.material_upper[b].x
            if ti.static(iy):
                local.y = self.material_upper[b].y
            point = origin + ti.Vector(
                [c * local.x - sn * local.y, sn * local.x + c * local.y]
            )
            normal, tangent = point.y, point.x
            if ti.static(side == 1):
                normal = -point.y
            elif ti.static(side == 2):
                normal, tangent = point.x, point.y
            elif ti.static(side == 3):
                normal, tangent = -point.x, point.y
            if normal < support - 1e-4:
                lower, upper = tangent, tangent
            elif ti.abs(normal - support) <= 1e-4:
                lower, upper = ti.min(lower, tangent), ti.max(upper, tangent)
            support = ti.min(support, normal)
        return ti.Vector([support, lower, upper])

    @ti.func
    def _pair_support(self, a, b, candidate: ti.template()):
        oa, ob = self.s.body_reference_origin[a], self.s.body_reference_origin[b]
        ca, sa = ti.cos(self.s.body_angle[a]), ti.sin(self.s.body_angle[a])
        cb, sb = ti.cos(self.s.body_angle[b]), ti.sin(self.s.body_angle[b])
        if ti.static(candidate):
            oa, ob = self.origin[a], self.origin[b]
            ca, sa = self.rotation[a].x, self.rotation[a].y
            cb, sb = self.rotation[b].x, self.rotation[b].y
        ax, ay = ti.Vector([ca, sa]), ti.Vector([-sa, ca])
        bx, by = ti.Vector([cb, sb]), ti.Vector([-sb, cb])
        la = 0.5 * (self.material_lower[a] + self.material_upper[a])
        lb = 0.5 * (self.material_lower[b] + self.material_upper[b])
        ha = 0.5 * (self.material_upper[a] - self.material_lower[a])
        hb = 0.5 * (self.material_upper[b] - self.material_lower[b])
        pa = ti.cast(oa, ti.f64) + la.x * ax + la.y * ay
        pb = ti.cast(ob, ti.f64) + lb.x * bx + lb.y * by
        delta = pa - pb
        gap = ti.cast(-1e30, ti.f64)
        normal = ti.Vector([0.0, 0.0])
        for k in ti.static(range(4)):
            axis = ax
            if ti.static(k == 1):
                axis = ay
            elif ti.static(k == 2):
                axis = bx
            elif ti.static(k == 3):
                axis = by
            distance = delta.dot(axis)
            separation = ti.abs(distance) - (
                ha.x * ti.abs(ax.dot(axis))
                + ha.y * ti.abs(ay.dot(axis))
                + hb.x * ti.abs(bx.dot(axis))
                + hb.y * ti.abs(by.dot(axis))
            )
            if separation > gap:
                gap = separation
                normal = axis if distance >= 0 else -axis
        # Midpoints of opposing support faces; nearly parallel axes select
        # a face midpoint instead of alternating between its two corners.
        for k in ti.static(range(2)):
            aa, bb, ea, eb = ax, bx, ha.x, hb.x
            if ti.static(k == 1):
                aa, bb, ea, eb = ay, by, ha.y, hb.y
            da, db = normal.dot(aa), normal.dot(bb)
            if ti.abs(da) > 1e-4:
                pa -= (1.0 if da > 0 else -1.0) * ea * aa
            if ti.abs(db) > 1e-4:
                pb += (1.0 if db > 0 else -1.0) * eb * bb
        return gap, normal, 0.5 * (pa + pb)

    @ti.func
    def _initialize_wall_reference(self):
        if self.wall_reference_ready[None] == 0:
            for b in range(self.s.body_count):
                for side in ti.static(range(4)):
                    plane = ti.cast(ti.static(self.s.config.boundary_cells), ti.f64)
                    if ti.static(side == 1):
                        plane -= self.s.ny
                    elif ti.static(side == 3):
                        plane -= self.s.nx
                    # An old raster-based checkpoint may already penetrate
                    # this conservative envelope. Permit that initial depth,
                    # never a growing per-step penetration allowance.
                    if self.s._body_has_resolved_boundary[b] != 0:
                        plane = ti.min(plane, self._wall_support(b, side, False).x)
                    self.wall_limit[b, side] = plane
            if ti.static(self.s.body_count > 1):
                for p in range(self.s.contacts.count):
                    a, b = self.s.contacts.pairs[p]
                    if (
                        self.s._body_has_resolved_boundary[a] != 0
                        and self.s._body_has_resolved_boundary[b] != 0
                    ):
                        gap, _normal, _point = self._pair_support(a, b, False)
                        self.pair_limit[p] = ti.min(0.0, gap)
            self.wall_reference_ready[None] = 1

    @ti.func
    def _sample_fraction(self, b, i, j):
        relative = self.s._cell_point(i, j) - self.origin[b]
        c, sn = self.rotation[b].x, self.rotation[b].y
        local = ti.Vector(
            [c * relative.x + sn * relative.y, -sn * relative.x + c * relative.y]
        )
        return self.s._sample_body_fraction(local, b)

    @ti.func
    def fraction_at(self, b, i, j):
        local = ti.Vector([i, j]) - self.lower[b]
        value = 0.0
        if 0 <= local.x < ti.static(
            self.coverage.shape[1]
        ) and 0 <= local.y < ti.static(self.coverage.shape[2]):
            value = self.coverage[b, local.x, local.y]
        return value

    @ti.func
    def _raster_candidates(self, reset: ti.template()):
        for b, i, j in self.coverage:
            if self.changed[b] != 0:
                value = 0.0
                if self.s.body_active[b] != 0:
                    world = self.lower[b] + ti.Vector([i, j])
                    if world.x < self.upper[b].x and world.y < self.upper[b].y:
                        value = self._sample_fraction(b, world.x, world.y)
                self.coverage[b, i, j] = value

    @ti.func
    def _entry_fraction(self, old, new):
        threshold = ti.static(self.s.config.thermal.solid_liquid_threshold)
        fraction = 0.0
        if old < threshold:
            fraction = ti.min(1.0, (threshold - old) / ti.max(new - old, 1e-12))
        return fraction

    @ti.func
    def limit_pair_step(self, a, b, i, j):
        # Both bodies must occupy the witness before a pair overlap exists.
        # Estimate that last crossing from old/new *ice* coverage, not phi.
        ta = self._entry_fraction(
            self.s._body_fractions[a, i, j], self.fraction_at(a, i, j)
        )
        tb = self._entry_fraction(
            self.s._body_fractions[b, i, j], self.fraction_at(b, i, j)
        )
        fraction = 0.9 * ti.max(ta, tb)
        ti.atomic_min(self.retry_scale[a], fraction)
        ti.atomic_min(self.retry_scale[b], fraction)

    @ti.func
    def _update_tile_bounds(self, b):
        c, sn = self.rotation[b].x, self.rotation[b].y
        half = 0.5 * ti.cast(self.s.thermal.material_size[b], ti.f32) + 0.5
        extent = ti.Vector(
            [
                ti.abs(c) * half.x + ti.abs(sn) * half.y,
                ti.abs(sn) * half.x + ti.abs(c) * half.y,
            ]
        )
        self.lower[b] = ti.cast(ti.floor(self.origin[b] - extent - 0.5), ti.i32)
        self.upper[b] = ti.cast(ti.ceil(self.origin[b] + extent + 0.5), ti.i32)

    @ti.func
    def _set_candidate_pose(self, b, factor):
        old_angle = self.s.body_angle[b]
        angle = old_angle + factor * self.s.body_angular_velocity[b]
        old_c, old_s = ti.cos(old_angle), ti.sin(old_angle)
        c, sn = ti.cos(angle), ti.sin(angle)
        local = self.s.body_local_center_of_mass[b]
        offset = ti.Vector(
            [
                (old_c - c) * local.x - (old_s - sn) * local.y,
                (old_s - sn) * local.x + (old_c - c) * local.y,
            ]
        )
        self.origin[b] = ti.cast(
            ti.cast(self.s.body_reference_origin[b], ti.f64)
            + factor * ti.cast(self.s.body_velocity[b], ti.f64)
            + offset,
            ti.f32,
        )
        self.angle[b] = angle
        self.rotation[b] = ti.Vector([c, sn])
        self.fraction[b] = factor
        self._update_tile_bounds(b)

    @ti.func
    def _predict(self, reset: ti.template(), mode):
        for b in range(self.s.body_count):
            if ti.static(reset):
                self.affected[b] = 0
                if b == 0:
                    for p in range(self.contact_links.shape[0]):
                        self.contact_links[p] = 0
            self.changed[b] = ti.cast(
                ti.static(reset) or self.retry_scale[b] < 1.0, ti.i32
            )
            if self.changed[b] != 0:
                factor = 1.0
                if mode == 1:
                    factor = self.retry_scale[b]
                elif mode == 2:
                    factor = 0.0
                if self.s._body_has_resolved_boundary[b] == 0:
                    factor = 0.0
                    self.s.body_velocity[b] = ti.Vector([0.0, 0.0])
                    self.s.body_angular_velocity[b] = 0.0
                self._set_candidate_pose(b, factor)

    @ti.func
    def _constrain_walls(self, repredict: ti.template()):
        for b in range(self.s.body_count):
            if self.changed[b] != 0 and self.fraction[b] > 0.0:
                sides = 0
                for side in ti.static(range(4)):
                    support = self._wall_support(b, side, True)
                    if support.x < self.wall_limit[b, side]:
                        sides |= 1 << side
                        self.wall_range[b, side] = ti.cast(
                            ti.Vector([support.y, support.z]), ti.f32
                        )
                self.wall_sides[b] = sides
                if sides != 0:
                    self.affected[b] = 1
                    self._resolve_wall_impulse(b)
                    if ti.static(repredict):
                        self._set_candidate_pose(b, self.fraction[b])
                    # Continuous normal clipping preserves tangential motion.
                    # Its displacement is streamed this same step; there is
                    # no integer-cell push or post-stream position correction.
                    for side in ti.static(range(4)):
                        depth = ti.max(
                            0.0,
                            self.wall_limit[b, side]
                            - self._wall_support(b, side, True).x,
                        )
                        shift = ti.Vector([0.0, 1.0])
                        if ti.static(side == 1):
                            shift = ti.Vector([0.0, -1.0])
                        elif ti.static(side == 2):
                            shift = ti.Vector([1.0, 0.0])
                        elif ti.static(side == 3):
                            shift = ti.Vector([-1.0, 0.0])
                        if depth > 0:
                            # An inward float32 ulp prevents rounding a tiny
                            # wall correction back to the penetrating pose.
                            bias = 1.2e-7 * ti.max(
                                1.0, ti.abs(self.origin[b].dot(shift))
                            )
                            self.origin[b] = ti.cast(
                                ti.cast(self.origin[b], ti.f64)
                                + (depth + bias) * shift,
                                ti.f32,
                            )
                    self._update_tile_bounds(b)

    @ti.func
    def _constrain_pairs(self):
        # Four fixed sweeps over tiny body arrays. Use current eroded material
        # bounds, not the initial ice size. Continuous normal corrections
        # avoid raster locking of a pair translating together. Raster contact
        # remains the final admissibility test for old checkpoint allowances.
        if ti.static(self.s.body_count > 1):
            ti.loop_config(serialize=True)
            for _ in range(4):
                corrected = 0
                for p in range(self.s.contacts.count):
                    a, b = self.s.contacts.pairs[p]
                    if (
                        self.s._body_has_resolved_boundary[a] != 0
                        and self.s._body_has_resolved_boundary[b] != 0
                        and self.fraction[a] > 0.0
                        and self.fraction[b] > 0.0
                    ):
                        gap, normal, point = self._pair_support(a, b, True)
                        depth = self.pair_limit[p] - gap
                        if depth > 0:
                            corrected = 1
                            self.s.contacts.normal[p] = ti.cast(normal, ti.f64)
                            self.s.contacts.point[p] = point
                            self.s.contacts.apply_impulse(p)
                            ma, mb = (
                                self.s.body_mass_lattice[a],
                                self.s.body_mass_lattice[b],
                            )
                            correction = (depth + 1e-4) * normal
                            self.origin[a] += ti.cast(
                                correction * mb / (ma + mb), ti.f32
                            )
                            self.origin[b] -= ti.cast(
                                correction * ma / (ma + mb), ti.f32
                            )
                            self.affected[a], self.affected[b] = 1, 1
                            self.changed[a], self.changed[b] = 1, 1
                            self._update_tile_bounds(a)
                            self._update_tile_bounds(b)
                if corrected == 0:
                    break
                self._constrain_walls(False)

    @ti.func
    def _check(self):
        conflicts = 0
        for b in range(self.s.body_count):
            self.retry_scale[b] = 1.0
            sides = 0
            for side in ti.static(range(4)):
                if (
                    self.s.body_active[b] != 0
                    and self.s._body_has_resolved_boundary[b] != 0
                ):
                    support = self._wall_support(b, side, True)
                    limit = self.wall_limit[b, side]
                    if support.x < limit - 1e-5:
                        sides |= 1 << side
                        self.wall_range[b, side] = ti.cast(
                            ti.Vector([support.y, support.z]), ti.f32
                        )
                        old = self._wall_support(b, side, False).x
                        fraction = ti.max(
                            0.0,
                            ti.min(1.0, (old - limit) / ti.max(old - support.x, 1e-12)),
                        )
                        ti.atomic_min(
                            self.retry_scale[b], ti.cast(0.9 * fraction, ti.f32)
                        )
            self.wall_sides[b] = sides
            if sides != 0:
                self.affected[b] = 1
                conflicts += 1
        if ti.static(self.s.body_count > 1):
            for p in range(self.s.contacts.count):
                if self.s.contacts.detect_pair(p, True) != 0:
                    pair = self.s.contacts.pairs[p]
                    ti.atomic_max(self.affected[pair.x], 1)
                    ti.atomic_max(self.affected[pair.y], 1)
                    self.contact_links[p] = 1
                    conflicts += 1
            # At most four short passes for this case. Share a retry fraction
            # only within connected body contacts, never between independent
            # bodies that happen to touch the floor at the same time.
            ti.loop_config(serialize=True)
            for repeat in range(self.s.body_count):
                for p in range(self.s.contacts.count):
                    if self.contact_links[p] != 0:
                        pair = self.s.contacts.pairs[p]
                        factor = ti.min(
                            self.retry_scale[pair.x], self.retry_scale[pair.y]
                        )
                        self.retry_scale[pair.x] = factor
                        self.retry_scale[pair.y] = factor
        return conflicts

    @ti.func
    def _resolve_impulses(self):
        if ti.static(self.s.body_count > 1):
            ti.loop_config(serialize=True)
            for p in range(self.s.contacts.count):
                if self.s.contacts.depth[p] > 0:
                    self.s.contacts.apply_impulse(p)

    @ti.func
    def _resolve_wall_impulse(self, b):
        for _body in range(1):
            velocity = ti.cast(self.s.body_velocity[b], ti.f64)
            omega = ti.cast(self.s.body_angular_velocity[b], ti.f64)
            center = ti.cast(self.s.body_center[b], ti.f64)
            for _ in ti.static(range(4)):
                for side in ti.static(range(4)):
                    if (self.wall_sides[b] & (1 << side)) != 0:
                        normal = ti.Vector([0.0, 1.0])
                        tangent_com = center.x
                        if ti.static(side == 1):
                            normal = ti.Vector([0.0, -1.0])
                        elif ti.static(side == 2):
                            normal = ti.Vector([1.0, 0.0])
                            tangent_com = center.y
                        elif ti.static(side == 3):
                            normal = ti.Vector([-1.0, 0.0])
                            tangent_com = center.y
                        lower = self.wall_range[b, side].x - tangent_com
                        upper = self.wall_range[b, side].y - tangent_com
                        coefficient = normal.y - normal.x
                        arm = 0.5 * (lower + upper)
                        if coefficient * omega > 0:
                            arm = lower
                        elif coefficient * omega < 0:
                            arm = upper
                        lever = coefficient * arm
                        closing = velocity.dot(normal) + omega * lever
                        if closing < 0:
                            mass, inertia = (
                                self.s.body_mass_lattice[b],
                                self.s.body_inertia_lattice[b],
                            )
                            impulse = -closing / (1.0 / mass + lever * lever / inertia)
                            velocity += impulse * normal / mass
                            omega += impulse * lever / inertia
            self.s.body_velocity[b] = ti.cast(velocity, ti.f32)
            self.s.body_angular_velocity[b] = ti.cast(omega, ti.f32)

    @ti.func
    def _accept(self):
        for b in range(self.s.body_count):
            old_angle = self.s.body_angle[b]
            c0, s0 = ti.cos(old_angle), ti.sin(old_angle)
            c1, s1 = self.rotation[b].x, self.rotation[b].y
            local = self.s.body_local_center_of_mass[b]
            offset = ti.Vector(
                [
                    (c1 - c0) * local.x - (s1 - s0) * local.y,
                    (s1 - s0) * local.x + (c1 - c0) * local.y,
                ]
            )
            self.wall_velocity[b] = ti.cast(
                ti.cast(self.origin[b], ti.f64)
                - ti.cast(self.s.body_reference_origin[b], ti.f64)
                + offset,
                ti.f32,
            )
            self.wall_omega[b] = self.angle[b] - old_angle
            if self.affected[b] != 0:
                self.contact_steps[b] += 1
                if self.fraction[b] < 1.0:
                    self.reduced_steps[b] += 1
                    # This bounded fallback is deliberately dissipative. It
                    # is not an additional force or a conservative remap.
                    self.s.body_velocity[b] *= self.fraction[b]
                    self.s.body_angular_velocity[b] *= self.fraction[b]
                    if self.fraction[b] == 0.0:
                        self.held_steps[b] += 1
        self.active[None] = 1

    @ti.kernel
    def commit(self):
        for b in range(self.s.body_count):
            self.s.body_reference_origin[b] = self.origin[b]
            self.s.body_angle[b] = self.angle[b]
            self.s._kick_rigid_velocity(b)
        self.active[None] = 0

    @ti.kernel
    def commit_raster(self):
        """Reuse accepted tiles, including fractional subcell melt remnants."""
        for i, j in self.s._moving_body_fraction:
            coverage = 0.0
            for b in range(self.s.body_count):
                value = 0.0
                if self.s.wall_mask[i, j] == 0:
                    value = self.fraction_at(b, i, j)
                self.s._body_fractions[b, i, j] = value
                coverage = ti.max(coverage, value)
            self.s._moving_body_fraction[i, j] = coverage

    @ti.kernel
    def check(self) -> ti.i32:
        return self._check()

    @ti.kernel
    def solve(self, reset: ti.template(), mode: ti.i32) -> ti.i32:
        # One launch/host result per trial; material interpolation is a
        # parallel tiled pass. The four-body constraint work remains tiny.
        self._initialize_wall_reference()
        if ti.static(not reset):
            self._resolve_impulses()
        self._predict(reset, mode)
        self._constrain_walls(True)
        self._constrain_pairs()
        self._raster_candidates(reset)
        conflicts = self._check()
        if conflicts == 0:
            self._accept()
        return conflicts

    def prepare(self):
        conflicts = self.solve(True, 0)
        if conflicts:
            conflicts = self.solve(False, 0)
            if conflicts:
                conflicts = self.solve(False, 1)
        # Retrying at zero also absorbs any new contact with another moving
        # body into the affected set. All old poses are known admissible.
        for _ in range(self.s.body_count + 1):
            if not conflicts:
                break
            conflicts = self.solve(False, 2)
        if conflicts:
            raise RuntimeError("accepted old ice geometry is not contact-free")
