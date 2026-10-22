"""One D3Q27 phase gradient with neutral-wetting wall mirror samples.

Coordinates below are cell-centre indices (world lattice position minus 0.5).
Only phase values are extended into solids; density is affine in phase, so
its gradient is the same cached vector multiplied by the density contrast.
"""
# ruff: noqa: E741 -- Taichi grouped grid indices.

import taichi as ti

from .lattice import Q, REST, direction, weight


@ti.data_oriented
class MirrorGradient3D:
    # A link is at most sqrt(3) long. Reflection moves its endpoint by at
    # most 2*sqrt(3); trilinear interpolation adds <1 cell per coordinate.
    # Hence every phase dependency is within ceil(1+2*sqrt(3)+1) = 6 cells.
    # Coverage normals depend only on the two endpoints' one-cell halos.
    gradient_dependency_radius = 6

    @ti.func
    def _tank_mirror_index(self, J):
        """Exact reflection at the physical planes of the rectangular tank."""
        K = J
        b = ti.static(self.config.boundary_cells)
        shape = ti.Vector(self.shape)
        for a in ti.static(range(3)):
            if K[a] < b:
                K[a] = 2 * b - 1 - K[a]
            elif K[a] >= shape[a] - b:
                K[a] = 2 * (shape[a] - b) - 1 - K[a]
        return K

    @ti.func
    def _coverage_normal(self, I):
        """Geometry derivative, NOT a second phase/density gradient."""
        g = ti.Vector.zero(ti.f32, 3)
        for a in ti.static(range(3)):
            e = ti.Vector.zero(ti.i32, 3)
            e[a] = 1
            # Extend the geometry at tank walls without inventing an ice face.
            lo, hi = self._tank_mirror_index(I - e), self._tank_mirror_index(I + e)
            if self._inside(lo) and self._inside(hi):
                g[a] = 0.5 * (self.coverage[hi] - self.coverage[lo])
        return g

    @ti.func
    def _boundary_link_fraction(self, I, J):
        """Shared intersection for mirror sampling and population reflection."""
        return ti.min(
            0.95,
            ti.max(
                0.05,
                (self.config.thermal.solid_liquid_threshold - self.coverage[I])
                / ti.max(self.coverage[J] - self.coverage[I], 1e-6),
            ),
        )

    @ti.func
    def _ice_mirror_point(self, I, J):
        """Reflect J about the local coverage-isosurface tangent plane.

        The link intersection uses exactly the eta used by moving-wall
        population reflection. The normal is interpolated at that crossing.
        A degenerate/unresolved surface falls back to the link normal.
        """
        c = ti.cast(J - I, ti.f32)
        eta = self._boundary_link_fraction(I, J)
        n = (1 - eta) * self._coverage_normal(I) + eta * self._coverage_normal(J)
        if n.dot(c) <= 1e-6 * ti.sqrt(n.dot(n) * c.dot(c)) or n.dot(n) < 1e-12:
            n = c
        n = n / ti.sqrt(n.dot(n) + 1e-30)
        return ti.cast(J, ti.f32) - 2 * (1 - eta) * n.dot(c) * n

    @ti.func
    def _fluid_phase_interpolate(self, point, fallback):
        """Bounded trilinear sample using only fluid values.

        At cut cells, discard solid corners and renormalize the remaining
        positive weights. This is constant-exact and bounded, but need not
        be linear-exact when corners are missing. An unresolved sub-cell
        cavity with no fluid support uses its originating fluid value.
        Tank reflection also handles ice/tank contact near corners.
        """
        base = ti.cast(ti.floor(point), ti.i32)
        part = point - ti.cast(base, ti.f32)
        value, total = 0.0, 0.0
        for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
            J = self._tank_mirror_index(base + ti.Vector([a, b, c]))
            if self._fluid(J):
                w = (
                    (part.x if a else 1 - part.x)
                    * (part.y if b else 1 - part.y)
                    * (part.z if c else 1 - part.z)
                )
                value += w * self.water_phase[J]
                total += w
        result = fallback
        if total > 1e-7:
            result = value / total
        return result

    @ti.func
    def _wall_phase_sample(self, I, c):
        # A diagonal hitting a flat tank wall retains its tangential offset.
        # In particular (i+1,j-1,k) -> (i+1,j,k), NOT (i,j,k).
        J = self._tank_mirror_index(I + c)
        value = self.water_phase[I]
        if self._fluid(J):
            value = self.water_phase[J]
        elif self._inside(J):
            value = self._fluid_phase_interpolate(self._ice_mirror_point(I, J), value)
        return value

    @ti.func
    def _write_gradient(self, I):
        gradient = ti.Vector.zero(ti.f32, 3)
        if self._fluid(I):
            rhs = ti.Vector.zero(ti.f32, 3)
            wall = False
            # Keep the bulk path unrolled. Every fluid contribution is read
            # once; the boundary loop below adds ONLY missing solid links.
            for q in ti.static(range(Q)):
                if ti.static(q != REST):
                    c = direction(q)
                    if self._fluid(I + c):
                        rhs += (
                            weight(q)
                            * ti.cast(c, ti.f32)
                            * (self.water_phase[I + c] - self.water_phase[I])
                        )
                    else:
                        wall = True
            if wall:
                for q in range(Q):
                    if q != REST:
                        c = direction(q)
                        if not self._fluid(I + c):
                            rhs += (
                                weight(q)
                                * ti.cast(c, ti.f32)
                                * (self._wall_phase_sample(I, c) - self.water_phase[I])
                            )
            gradient = 3 * rhs
        self._store_gradient(I, gradient)

    @ti.func
    def _store_gradient(self, I, gradient):
        self.phase_gradient_lattice[I] = gradient
        self.capillary_stress[I] = (
            1.5
            * self.sigma
            * self.config.interface_width
            * (
                gradient.dot(gradient) * ti.Matrix.identity(ti.f32, 3)
                - gradient.outer_product(gradient)
            )
        )

    def _gradients(self, classified=False):
        if not classified:
            self._classify_home()
        self._gradients_regular()
        self._gradients_ice()

    @ti.func
    def _write_regular_gradient(self, I, flag):
        rhs = ti.Vector.zero(ti.f32, 3)
        if flag >= 0:
            for q in ti.static(range(Q)):
                if ti.static(q != REST):
                    c = direction(q)
                    J = I + c
                    if flag == 1:
                        J = self._tank_mirror_index(J)
                    rhs += (
                        weight(q)
                        * ti.cast(c, ti.f32)
                        * (self.water_phase[J] - self.water_phase[I])
                    )
        self._store_gradient(I, 3 * rhs)

    @ti.kernel
    def _gradients_regular(self):
        for I in ti.grouped(self.water_phase):
            flag = self._home_boundary[I]
            if flag < 2:
                self._write_regular_gradient(I, flag)

    @ti.kernel
    def _gradients_ice(self):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 2:
                self._write_gradient(I)

    @ti.func
    def _refill_gradient_bounds(self):
        # Coverage can change without any label flipping, e.g. sub-cell ice
        # movement or partial melting. Both change the mirror sampling point.
        lo = ti.min(self.refill_lower[None], self._gradient_geometry_lower[None])
        hi = ti.max(self.refill_upper[None], self._gradient_geometry_upper[None])
        changed = (hi >= lo).all()
        radius = ti.static(self.gradient_dependency_radius)
        lo = ti.max(lo - radius, 0)
        hi = ti.min(hi + radius + 1, ti.Vector(self.shape))
        if not changed:
            hi = lo
        return lo, hi

    def _refill_gradients(self):
        self._classify_refill_gradients()
        self._refill_gradients_regular()
        self._refill_gradients_ice()

    @ti.kernel
    def _classify_refill_gradients(self):
        lo, hi = self._refill_gradient_bounds()
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            # Geometry just changed, so the pre-stream classification may be
            # stale. Reclassify only this halo and use the same arithmetic as
            # the full gradient update, including planar tank corners.
            self._home_boundary[I] = ti.cast(self._home_boundary_at(I), ti.i8)

    @ti.kernel
    def _refill_gradients_regular(self):
        lo, hi = self._refill_gradient_bounds()
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            if self._home_boundary[I] < 2:
                self._write_regular_gradient(I, self._home_boundary[I])

    @ti.kernel
    def _refill_gradients_ice(self):
        lo, hi = self._refill_gradient_bounds()
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            if self._home_boundary[I] == 2:
                self._write_gradient(I)
