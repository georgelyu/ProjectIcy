"""One compact, connected 3x3 phase gradient for every flow operator.

A float32 2x2 linear least-squares solve replaces the quadratic QR fit.
Collinear donors retain their observed derivative and close the missing
normal derivative with zero. Solid values never enter the fit.
"""

from enum import IntEnum

import taichi as ti

from .lattice import (
    inside_grid,
    lattice_direction,
    lattice_weight,
    mixture_density,
    momentum_equilibrium,
    phase_equilibrium,
)
from .stability_options import pressure_preserving_stress_population


class PhaseGradientStatus(IntEnum):
    """Compact gradient mode; rank is the rank of a two-column linear fit."""

    INACTIVE = 0
    BULK = 1
    SINGLE_ROW_CLOSURE = 2
    LINEAR_FIT = 3
    ISOLATED_CONSTANT_CLOSURE = 5


@ti.data_oriented
class PhaseReconstruction:
    def _initialize_phase_reconstruction(self):
        shape = (self.nx, self.ny)
        self.phase_gradient_lattice = ti.Vector.field(2, ti.f32, shape=shape)
        self.phase_gradient_status = ti.field(ti.i32, shape=shape)
        self.phase_gradient_data_rank = ti.field(ti.i32, shape=shape)
        self.phase_gradient_spatial_rank = ti.field(ti.i32, shape=shape)
        self.phase_donor_mask = ti.field(ti.i32, shape=shape)
        self.refill_state = ti.Vector.field(4, ti.f32, shape=shape)
        self.refill_next_state = ti.Vector.field(4, ti.f32, shape=shape)
        self.refill_valid = ti.field(ti.i32, shape=shape)
        self.refill_next_valid = ti.field(ti.i32, shape=shape)
        self.refill_remaining = ti.field(ti.i32, shape=())
        self.refill_first_missing = ti.field(ti.i32, shape=())
        # A pending cell is encoded as solid_mask == 2 (1 is physical ice).
        # Keep its first step for sparse diagnostics, not another fluid state.
        self.refill_pending_since = ti.field(ti.i32, shape=shape)
        self.refill_pending_events = ti.field(ti.i32, shape=())
        self.refill_pending_activations = ti.field(ti.i32, shape=())
        self.refill_pending_max_age_steps = ti.field(ti.i32, shape=())
        self._pending_refill_count = 0
        self._pending_refill_peak = 0

    @property
    def source_gradient_lattice(self):
        """Historical diagnostic spelling; aliases the single gradient field."""
        return self.phase_gradient_lattice

    @ti.func
    def _phase_donor_allowed(self, i, j, di, dj):
        allowed = False
        if ti.abs(di) <= 1 and ti.abs(dj) <= 1:
            bit = (di + 1) * 3 + dj + 1
            allowed = ((self.phase_donor_mask[i, j] >> bit) & 1) != 0
        return allowed

    @ti.func
    def _unclipped_ice_fraction(self, b, i, j):
        return self._sample_body_fraction(
            self._body_local_coordinates(self._cell_point(i, j), b), b
        )

    @ti.func
    def _phase_visible_diagonal(self, i, j, ni, nj):
        """Test the exact piecewise-quadratic coverage along a D2Q9 link."""
        visible = True
        first, last = self._cell_point(i, j), self._cell_point(ni, nj)
        for b in range(self.body_count):
            if self.body_active[b] != 0:
                lo = self._body_local_coordinates(first, b)
                hi = self._body_local_coordinates(last, b)
                grid0 = lo + 0.5 * ti.cast(self.thermal.material_size[b], ti.f32) - 0.5
                grid1 = hi + 0.5 * ti.cast(self.thermal.material_size[b], ti.f32) - 0.5
                cuts = ti.Vector([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                count = 2
                for axis in ti.static(range(2)):
                    delta = grid1[axis] - grid0[axis]
                    if ti.abs(delta) > 1e-8:
                        base = ti.floor(ti.min(grid0[axis], grid1[axis]))
                        for crossing in ti.static(range(1, 3)):
                            t = (base + crossing - grid0[axis]) / delta
                            if t > 1e-6 and t < 1 - 1e-6 and count < 6:
                                cuts[count] = t
                                count += 1
                for repeat in ti.static(range(6)):
                    for k in ti.static(range(5)):
                        if cuts[k] > cuts[k + 1]:
                            temporary = cuts[k]
                            cuts[k] = cuts[k + 1]
                            cuts[k + 1] = temporary
                for k in range(5):
                    ta, tb = cuts[k], cuts[k + 1]
                    if tb - ta > 1e-6:
                        f0 = self._sample_body_fraction(lo + ta * (hi - lo), b)
                        f1 = self._sample_body_fraction(lo + tb * (hi - lo), b)
                        fm = self._sample_body_fraction(
                            lo + 0.5 * (ta + tb) * (hi - lo), b
                        )
                        aa = 2 * (f0 + f1 - 2 * fm)
                        bb = f1 - f0 - aa
                        maximum = ti.max(f0, f1)
                        if aa < -1e-8:
                            critical = -bb / (2 * aa)
                            if critical > 0 and critical < 1:
                                maximum = ti.max(
                                    maximum,
                                    f0 + bb * critical + aa * critical * critical,
                                )
                        if maximum >= self.config.thermal.solid_liquid_threshold - 1e-6:
                            visible = False
        return visible

    @ti.func
    def _phase_ice_geometry(self, i, j, ni, nj):
        b = self.solid_body_id[ni, nj]
        gradient = ti.Vector.zero(ti.f32, 2)
        for endpoint in ti.static(range(2)):
            x, y = i, j
            if ti.static(endpoint == 1):
                x, y = ni, nj
            gradient.x += self._unclipped_ice_fraction(
                b, x + 1, y
            ) - self._unclipped_ice_fraction(b, x - 1, y)
            gradient.y += self._unclipped_ice_fraction(
                b, x, y + 1
            ) - self._unclipped_ice_fraction(b, x, y - 1)
        direction = ti.Vector([ti.cast(ni - i, ti.f32), ti.cast(nj - j, ti.f32)])
        normal = direction / direction.norm()
        if gradient.norm() > 1e-10:
            normal = gradient / gradient.norm()
        fi = self._body_fractions[b, i, j]
        fs = self._body_fractions[b, ni, nj]
        fraction = 0.5
        if fs - fi > 1e-10:
            fraction = ti.min(
                1.0,
                ti.max(
                    0.0, (self.config.thermal.solid_liquid_threshold - fi) / (fs - fi)
                ),
            )
        # Geometry uses the threshold crossing with a fallback for a flat
        # coverage field; hydrodynamic cut links guard their level-set ratio.
        offset = fraction * direction
        return ti.Vector([normal.x, normal.y, offset.x, offset.y])

    @ti.kernel
    def _prepare_phase_geometry(self):
        """Only immediate, connected fluid candidates enter the nine-bit mask."""
        for i, j in self.water_phase:
            mask = 0
            if self._is_fluid_cell(i, j):
                mask = 1 << 4
                for q in range(1, 9):
                    c = lattice_direction(q)
                    ni, nj = i + c.x, j + c.y
                    if inside_grid(ni, nj, self.nx, self.ny):
                        if self._is_fluid_cell(ni, nj):
                            visible = True
                            if c.x != 0 and c.y != 0:
                                # A face-connected route suffices. If neither
                                # side is fluid, check the actual cut diagonal.
                                if not self._is_fluid_cell(
                                    i + c.x, j
                                ) and not self._is_fluid_cell(i, j + c.y):
                                    visible = self._phase_visible_diagonal(i, j, ni, nj)
                            if visible:
                                mask |= 1 << ((c.x + 1) * 3 + c.y + 1)
            self.phase_donor_mask[i, j] = mask

    @ti.kernel
    def _prepare_phase_gradients_kernel(self):
        """D2Q9-weighted linear fit; the full stencil is central differencing."""
        for i, j in self.water_phase:
            gradient = ti.Vector.zero(ti.f32, 2)
            rank, status = 0, ti.cast(PhaseGradientStatus.INACTIVE, ti.i32)
            if self._is_fluid_cell(i, j):
                phi = self.water_phase[i, j]
                xx, xy, yy = 0.0, 0.0, 0.0
                rhs = ti.Vector.zero(ti.f32, 2)
                for q in range(1, 9):
                    c = lattice_direction(q)
                    if self._phase_donor_allowed(i, j, c.x, c.y):
                        v = ti.cast(c, ti.f32)
                        w = lattice_weight(q)
                        xx += w * v.x * v.x
                        xy += w * v.x * v.y
                        yy += w * v.y * v.y
                        rhs += w * v * (self.water_phase[i + c.x, j + c.y] - phi)
                trace = xx + yy
                determinant = xx * yy - xy * xy
                status = PhaseGradientStatus.ISOLATED_CONSTANT_CLOSURE
                if trace > 1e-12:
                    if determinant > 1e-6 * trace * trace:
                        gradient = (
                            ti.Vector(
                                [
                                    yy * rhs.x - xy * rhs.y,
                                    xx * rhs.y - xy * rhs.x,
                                ]
                            )
                            / determinant
                        )
                        rank = 2
                        status = PhaseGradientStatus.LINEAR_FIT
                    else:
                        # Rank one: rhs lies along the observed line. This
                        # minimum-norm closure does not constrain its slope.
                        gradient = rhs / trace
                        rank = 1
                        status = PhaseGradientStatus.SINGLE_ROW_CLOSURE
                    if self.phase_donor_mask[i, j] == 511:
                        status = PhaseGradientStatus.BULK
            self.phase_gradient_lattice[i, j] = gradient
            self.phase_gradient_status[i, j] = status
            self.phase_gradient_spatial_rank[i, j] = rank
            self.phase_gradient_data_rank[i, j] = rank

    def _prepare_phase_gradients(self):
        self._prepare_phase_gradients_kernel()

    @ti.func
    def _connected_phase_ghost(self, i, j, sample_i, sample_j):
        offset = ti.Vector(
            [ti.cast(sample_i - i, ti.f32), ti.cast(sample_j - j, ti.f32)]
        )
        value = self.water_phase[i, j] + self.phase_gradient_lattice[i, j].dot(offset)
        return ti.min(1.0, ti.max(0.0, value))

    @ti.kernel
    def _seed_connected_refill(self):
        self.refill_remaining[None] = 0
        self.refill_first_missing[None] = self.nx * self.ny
        for i, j in self.water_phase:
            valid = 0
            state = ti.Vector.zero(ti.f32, 4)
            if self._is_fluid_cell(i, j):
                if self.previous_solid_mask[i, j] == 0:
                    velocity = self._physical_velocity_at(i, j)
                    state = ti.Vector(
                        [
                            ti.min(1.0, ti.max(0.0, self.water_phase[i, j])),
                            self.pressure_lattice[i, j]
                            - self.hydrostatic_reference_pressure[i, j],
                            velocity.x,
                            velocity.y,
                        ]
                    )
                    valid = 1
                else:
                    ti.atomic_add(self.refill_remaining[None], 1)
                    ti.atomic_min(self.refill_first_missing[None], i * self.ny + j)
            self.refill_state[i, j] = state
            self.refill_valid[i, j] = valid

    @ti.kernel
    def _propagate_connected_refill(self):
        self.refill_remaining[None] = 0
        self.refill_first_missing[None] = self.nx * self.ny
        for i, j in self.water_phase:
            valid = self.refill_valid[i, j]
            state = self.refill_state[i, j]
            if self._is_fluid_cell(i, j) and valid == 0:
                count = 0.0
                total = ti.Vector.zero(ti.f32, 4)
                for a, b in ti.ndrange(3, 3):
                    di, dj = a - 1, b - 1
                    if di != 0 or dj != 0:
                        if self._phase_donor_allowed(i, j, di, dj):
                            if self.refill_valid[i + di, j + dj] != 0:
                                total += self.refill_state[i + di, j + dj]
                                count += 1
                if count > 0:
                    state = total / count
                    valid = 1
                if valid == 0:
                    ti.atomic_add(self.refill_remaining[None], 1)
                    ti.atomic_min(self.refill_first_missing[None], i * self.ny + j)
            self.refill_next_state[i, j] = state
            self.refill_next_valid[i, j] = valid

    @ti.kernel
    def _defer_unresolved_refill(self, refill_step: ti.i32):
        """Keep donorless new components outside the active fluid domain.

        Propagation has reached a fixed point: no accepted donor link joins
        a resolved cell to an unresolved cell. Only the latter's own masks
        need clearing. Streaming sees these cells as stationary closed walls;
        no guessed phase, population or rigid-body mass is introduced.
        """
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j) and self.refill_valid[i, j] == 0:
                self.solid_mask[i, j] = ti.cast(2, ti.i8)
                self.phase_donor_mask[i, j] = 0
                self.momentum_velocity_lattice[i, j] = ti.Vector.zero(ti.f32, 2)
                self.fluid_acceleration_lattice[i, j] = ti.Vector.zero(ti.f32, 2)
                if self.refill_pending_since[i, j] == 0:
                    self.refill_pending_since[i, j] = refill_step
                    ti.atomic_add(self.refill_pending_events[None], 1)
                ti.atomic_max(
                    self.refill_pending_max_age_steps[None],
                    refill_step - self.refill_pending_since[i, j],
                )

    @ti.kernel
    def _commit_connected_refill(self, refill_step: ti.i32):
        for i, j in self.water_phase:
            if (
                self.wall_mask[i, j] == 0
                and (
                    self.previous_body_id[i, j] != self.solid_body_id[i, j]
                    or self.previous_solid_mask[i, j] == 2
                )
            ):
                if self.solid_mask[i, j] == 1:
                    self.momentum_velocity_lattice[i, j] = self._body_velocity_at(
                        self._cell_point(i, j), self.solid_body_id[i, j]
                    )
                elif self.solid_mask[i, j] == 0:
                    if self.refill_pending_since[i, j] != 0:
                        ti.atomic_add(self.refill_pending_activations[None], 1)
                        ti.atomic_max(
                            self.refill_pending_max_age_steps[None],
                            refill_step - self.refill_pending_since[i, j],
                        )
                        self.refill_pending_since[i, j] = 0
                    state = self.refill_state[i, j]
                    phi, pressure = state[0], state[1]
                    velocity = ti.Vector([state[2], state[3]])
                    density = mixture_density(
                        phi, self._water_density_lattice, self._air_density_lattice
                    )
                    self.water_phase[i, j] = phi
                    self.pressure_lattice[i, j] = (
                        pressure + self.hydrostatic_reference_pressure[i, j]
                    )
                    self.momentum_velocity_lattice[i, j] = velocity
                    stress = ti.Vector.zero(ti.f32, 3)
                    if ti.static(self.config.exposed_node_refill == "non_equilibrium"):
                        stress = self._extrapolate_exposed_stress(i, j, density)
                    for q in range(9):
                        population = momentum_equilibrium(
                            q, pressure, density, velocity
                        )
                        if ti.static(
                            self.config.exposed_node_refill == "non_equilibrium"
                        ):
                            population += pressure_preserving_stress_population(
                                q, stress, velocity
                            )
                        self.momentum_populations[i, j, q] = population
                        self.phase_populations[i, j, q] = phase_equilibrium(
                            q, phi, velocity
                        )
                self.fluid_acceleration_lattice[i, j] = ti.Vector.zero(ti.f32, 2)

    def _refill_connected_nodes(self):
        self._seed_connected_refill()
        remaining = int(self.refill_remaining[None])
        new_nodes = remaining
        self._pending_refill_count = 0
        while remaining > 0:
            self._propagate_connected_refill()
            new_remaining = int(self.refill_remaining[None])
            self.refill_state.copy_from(self.refill_next_state)
            self.refill_valid.copy_from(self.refill_next_valid)
            if new_remaining >= remaining:
                self._pending_refill_count = new_remaining
                self._pending_refill_peak = max(self._pending_refill_peak, new_remaining)
                self._defer_unresolved_refill(self.steps + 1)
                break
            remaining = new_remaining
        self._refill_had_openings = new_nodes > self._pending_refill_count
        self._commit_connected_refill(self.steps + 1)

    @ti.kernel
    def _project_refill_moments(self):
        """One fixed moment lift after the new phase gradient is available.

        Preserve sum(g), raw deviatoric stress and phi. Force is frozen for
        the next half-step recovery.
        """
        for i, j in self.water_phase:
            if (
                self._is_fluid_cell(i, j)
                and self.previous_solid_mask[i, j] != 0
                and self.refill_valid[i, j] == 1
            ):
                state = self.refill_state[i, j]
                velocity = ti.Vector([state[2], state[3]])
                rho = mixture_density(
                    state[0], self._water_density_lattice, self._air_density_lattice
                )
                force_data = self._fluid_force_and_gradient(i, j)
                force = ti.Vector([force_data[0], force_data[1]])
                # The old phase flux has the old geometry. Do not invent a new
                # flux during remapping; collision computes it from fresh h.
                grad = self._phase_central_gradient(i, j)
                if ti.static(self.config.density_gradient_source_limit > 0):
                    grad *= self._density_source_limiter(i, j, velocity, grad)
                sigma = (
                    self._water_density_lattice - self._air_density_lattice
                ) * velocity.dot(grad)
                momentum = ti.Vector.zero(ti.f32, 2)
                moving = 0.0
                for q in range(9):
                    value = self.momentum_populations[i, j, q]
                    momentum += ti.cast(lattice_direction(q), ti.f32) * value
                    if q != 0:
                        moving += value
                delta_j = rho * (velocity - 0.5 * force) - momentum
                s0 = -1.5 * lattice_weight(0) * velocity.dot(velocity)
                delta_p = (5.0 / 3.0) * state[1] - 0.5 * sigma - rho * s0 - moving
                for q in range(9):
                    correction = (
                        3.0
                        * lattice_weight(q)
                        * ti.cast(lattice_direction(q), ti.f32).dot(delta_j)
                    )
                    if q == 0:
                        correction -= delta_p
                    else:
                        correction += 1.8 * lattice_weight(q) * delta_p
                    self.momentum_populations[i, j, q] += correction
                self.momentum_velocity_lattice[i, j] = velocity - 0.5 * force
                self.fluid_acceleration_lattice[i, j] = force
