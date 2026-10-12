"""Conservative event accounting around a sharp-node phase LBM.

H carries liquid volume in units of dx**2, not an intensive phase fraction.
Bulk transport is the original phase collision plus limited, antisymmetric
D2Q9 link exchanges. Geometry transactions use connected-component recipient
lists and atomic capacity reservations; no water-total reduction is required.
The component construction is deliberately explicit: a disconnected pool must
never pay for another pool's geometry correction.
"""

import taichi as ti

from .lattice import lattice_direction, opposite_direction_index, phase_equilibrium


@ti.func
def square_overlap(cx, cy, cosine, sine, i, j):
    """Exact polygon intersection of a rotated unit material tile and a cell."""
    points = ti.Matrix.zero(ti.f64, 12, 2)
    scratch = ti.Matrix.zero(ti.f64, 12, 2)
    for k in ti.static(range(4)):
        x = ti.cast((-0.5, 0.5, 0.5, -0.5)[k], ti.f64)
        y = ti.cast((-0.5, -0.5, 0.5, 0.5)[k], ti.f64)
        points[k, 0] = cx + cosine * x - sine * y
        points[k, 1] = cy + sine * x + cosine * y
    count = 4
    for edge in ti.static(range(4)):
        axis = edge // 2
        lower = edge % 2 == 0
        value = ti.cast(i if axis == 0 else j, ti.f64) + (0.0 if lower else 1.0)
        out = 0
        for k in range(count):
            prev = (k + count - 1) % count
            a = ti.Vector([points[prev, 0], points[prev, 1]])
            b = ti.Vector([points[k, 0], points[k, 1]])
            ia = a[axis] >= value if lower else a[axis] <= value
            ib = b[axis] >= value if lower else b[axis] <= value
            if ia != ib:
                t = (value - a[axis]) / (b[axis] - a[axis])
                p = a + t * (b - a)
                scratch[out, 0], scratch[out, 1] = p.x, p.y
                out += 1
            if ib:
                scratch[out, 0], scratch[out, 1] = b.x, b.y
                out += 1
        count = out
        for k in range(count):
            points[k, 0], points[k, 1] = scratch[k, 0], scratch[k, 1]
    area = ti.cast(0.0, ti.f64)
    for k in range(count):
        other = (k + 1) % count
        # Translate before the cross product to avoid cancellation at large i,j.
        area += (points[k, 0] - cx) * (points[other, 1] - cy)
        area -= (points[k, 1] - cy) * (points[other, 0] - cx)
    return ti.abs(area) * 0.5


@ti.data_oriented
class ConservativePhaseLBM:
    """Conservative event transactions around the original sharp-node LBM.

    The phase collision is retained and push streaming is donor limited. Wall source,
    node creation/deletion, bounds and material melting enter the sign budget.
    No total-water statistic is a solver input.
    """
    def __init__(self, simulation):
        self.s = simulation
        self.nx, self.ny = simulation.nx, simulation.ny
        self.count = self.nx * self.ny
        shape = (self.nx, self.ny)
        for name in ("capacity", "previous_capacity", "ice_volume", "melt_volume", "melt_heat", "melt_rate", "event_volume", "event_heat", "energy_buffer", "momentum_phase", "pending", "pending_heat", "spare", "available", "specific_heat", "delta", "heat_delta", "outgoing_limit", "target_volume"):
            setattr(self, name, ti.field(ti.f64, shape=shape))
        self.parent = ti.field(ti.i32, shape=self.count)
        self.recipient_count = ti.field(ti.i32, shape=3)
        self.next = ti.field(ti.i32, shape=(3, self.count))
        self.pool = ti.Vector.field(8, ti.f64, shape=self.count)
        for name in ("interface", "correction_core", "correction_open"):
            setattr(self, name, ti.field(ti.i8, shape=shape))
        for name in ("changed", "unresolved", "events", "geometry_error", "bounds_needed", "topology_dirty"):
            setattr(self, name, ti.field(ti.i32, shape=()))
        self.components_ready = False
        self.material_cell_mass = ti.field(ti.f64, shape=())
        self.water_cell_mass = ti.field(ti.f64, shape=())
        self.cell_area = ti.field(ti.f64, shape=())
        self.material_cell_mass[None] = simulation.thermal._initial_body_cell_mass
        self.cell_area[None] = simulation.config.dx**2
        self.water_cell_mass[None] = simulation.config.rho_water * simulation.config.dx**2

    @ti.func
    def cell(self, i, j):
        result = False
        if 0 <= i < self.nx and 0 <= j < self.ny:
            result = self.capacity[i, j] > 0
        return result

    @ti.func
    def volume(self, i, j):
        value = ti.cast(0.0, ti.f64)
        for q in ti.static(range(9)):
            value += self.s.phase_populations[i, j, q]
        return value

    @ti.func
    def close_moment(self, i, j, value):
        moving = ti.cast(0.0, ti.f64)
        for q in ti.static(range(1, 9)):
            moving += self.s.phase_populations[i, j, q]
        self.s.phase_populations[i, j, 0] = value - moving

    @ti.func
    def geometry_open(self, i, j):
        result = False
        if 0 <= i < self.nx and 0 <= j < self.ny:
            result = self.capacity[i, j] > 0 or self.previous_capacity[i, j] > 0
            if ti.static(self.s.config.close_subgrid_wall_gaps):
                # Numerical gap closure excludes fluid dynamics, not a real
                # material barrier. Geometry events can traverse the original
                # subgrid channel to a resolved recipient without losing water.
                result = result or self.s.subgrid_wall_gap[i, j] != 0 or self.s.previous_subgrid_wall_gap[i, j] != 0
        return result

    @ti.func
    def root(self, index):
        value = index
        while self.parent[value] >= 0 and self.parent[value] != value:
            value = self.parent[value]
        return value

    @ti.kernel
    def update_capacity(self):
        self.geometry_error[None] = 0
        for i, j in self.capacity:
            self.previous_capacity[i, j] = self.capacity[i, j]
            self.capacity[i, j] = 1.0 if self.s._is_fluid_cell(i, j) else 0.0

    @ti.kernel
    def cache_momentum_phase(self):
        for i, j in self.capacity:
            self.momentum_phase[i, j] = self.s.water_phase[i, j]

    @ti.kernel
    def initialize(self):
        for i, j in self.capacity:
            phi = self.s.water_phase[i, j] if self.cell(i, j) else 0.0
            self.s.water_phase[i, j] = phi
            for q in ti.static(range(9)):
                self.s.phase_populations[i, j, q] = phase_equilibrium(q, phi, self.s._physical_velocity_at(i, j))
            self.close_moment(i, j, phi)

    @ti.func
    def raw_flux(self, i, j, q):
        d = lattice_direction(q)
        ni, nj = i + d.x, j + d.y
        value = ti.cast(0.0, ti.f64)
        if self.cell(i, j) and self.cell(ni, nj):
            value = self.s.phase_stream_buffer[i, j, q] - self.s.phase_stream_buffer[ni, nj, opposite_direction_index(q)]
        return value

    @ti.func
    def flux(self, i, j, q):
        d = lattice_direction(q)
        value = self.raw_flux(i, j, q)
        if value > 0:
            value *= self.outgoing_limit[i, j]
        elif value < 0:
            value *= self.outgoing_limit[i + d.x, j + d.y]
        return value

    @ti.kernel
    def prepare_transport(self):
        for i, j in self.capacity:
            out = ti.cast(0.0, ti.f64)
            for q in ti.static(range(1, 9)):
                out += ti.max(0.0, self.raw_flux(i, j, q))
            self.outgoing_limit[i, j] = ti.min(1.0, ti.max(self.s.water_phase[i, j], 0.0) / ti.max(out, 1e-30))

    @ti.kernel
    def limit_stream(self):
        for i, j in self.capacity:
            if self.cell(i, j):
                value = self.s.water_phase[i, j] - self.event_volume[i, j]
                for q in ti.static(range(1, 9)):
                    d = lattice_direction(q)
                    f = self.flux(i, j, q)
                    value -= f
                    if self.cell(i + d.x, j + d.y):
                        self.s.phase_populations[i, j, opposite_direction_index(q)] = self.s.phase_stream_buffer[i, j, q] - f
                # A moving-wall removal cannot overdraw the remaining water.
                if value < 0:
                    refund = -value
                    self.event_volume[i, j] -= refund
                    self.event_heat[i, j] -= refund * self.specific_heat[i, j]
                    value = 0.0
                self.close_moment(i, j, value)

    @ti.kernel
    def begin_fluid(self, advect_heat: ti.i32):
        for i, j in self.capacity:
            self.event_volume[i, j] = 0.0
            self.event_heat[i, j] = 0.0
            specific = ti.cast(0.0, ti.f64)
            if self.s.water_phase[i, j] > 1e-12:
                specific = self.s.thermal.water_sensible_energy[i, j] / self.s.water_phase[i, j]
            self.specific_heat[i, j] = specific
            energy = self.s.thermal.water_sensible_energy[i, j]
            if advect_heat != 0 and self.cell(i, j):
                for q in ti.static(range(1, 9)):
                    d = lattice_direction(q)
                    ni, nj = i + d.x, j + d.y
                    if self.cell(ni, nj):
                        f = self.flux(i, j, q)
                        donor = specific
                        if f < 0:
                            donor = 0.0
                            if self.s.water_phase[ni, nj] > 1e-12:
                                donor = self.s.thermal.water_sensible_energy[ni, nj] / self.s.water_phase[ni, nj]
                        energy -= f * donor
            self.energy_buffer[i, j] = energy

    @ti.func
    def record_wall(self, i, j, before, after):
        change = after - before
        self.event_volume[i, j] -= change
        if change < 0:
            self.event_heat[i, j] -= change * self.specific_heat[i, j]

    @ti.kernel
    def finish_energy(self):
        for i, j in self.capacity:
            self.s.thermal.water_sensible_energy[i, j] = self.energy_buffer[i, j] - self.event_heat[i, j]

    @ti.kernel
    def prepare_transaction(self):
        self.events[None] = 0
        for i, j in self.capacity:
            old = self.s.water_phase[i, j]
            target = ti.min(self.capacity[i, j], ti.max(old, 0.0))
            residual = self.event_volume[i, j] + old - target
            total_energy = self.s.thermal.water_sensible_energy[i, j] + self.event_heat[i, j]
            payload = ti.cast(0.0, ti.f64)
            if residual > 0:
                payload = total_energy * residual / ti.max(target + residual, 1e-30)
            self.s.thermal.water_sensible_energy[i, j] = total_energy - payload
            self.pending[i, j], self.pending_heat[i, j] = residual, payload
            self.available[i, j] = target
            self.target_volume[i, j] = target
            self.spare[i, j] = self.capacity[i, j] - target
            self.specific_heat[i, j] = (total_energy - payload) / ti.max(target, 1e-30)
            self.delta[i, j], self.heat_delta[i, j] = 0.0, 0.0
            if ti.abs(residual) > 1e-13:
                ti.atomic_or(self.events[None], 1)
            if old != target:
                velocity = self.s._physical_velocity_at(i, j)
                for q in ti.static(range(9)):
                    if target <= 1e-12 or target >= self.capacity[i, j] - 1e-12:
                        self.s.phase_populations[i, j, q] = phase_equilibrium(q, target, velocity)
                    else:
                        self.s.phase_populations[i, j, q] += phase_equilibrium(q, target - old, velocity)
                self.close_moment(i, j, target)

    @ti.kernel
    def classify_correction_core(self):
        for i, j in self.capacity:
            wet = False
            if self.geometry_open(i, j):
                wet = self.available[i, j] >= self.s.config.water_phase_cutoff or self.s.water_phase[i, j] >= self.s.config.water_phase_cutoff or ti.abs(self.pending[i, j]) > 1e-13
            self.correction_core[i, j] = ti.cast(wet, ti.i8)

    @ti.kernel
    def classify_correction_graph(self):
        self.topology_dirty[None] = 0
        for i, j in self.capacity:
            active = self.correction_core[i, j] != 0
            for q in ti.static(range(1, 9)):
                d = lattice_direction(q)
                ni, nj = i + d.x, j + d.y
                if 0 <= ni < self.nx and 0 <= nj < self.ny:
                    active = active or self.correction_core[ni, nj] != 0
            active = active and self.geometry_open(i, j)
            self.correction_open[i, j] = ti.cast(active, ti.i8)
            if active != (self.parent[i * self.ny + j] >= 0):
                ti.atomic_max(self.topology_dirty[None], 1)

    @ti.func
    def correction_cell(self, i, j):
        result = False
        if 0 <= i < self.nx and 0 <= j < self.ny:
            result = self.correction_open[i, j] != 0
        return result

    def repair_bounds(self):
        self.cache_momentum_phase()
        self.remap()
        self.s._synchronize_conservative_momentum()

    def recover(self, heat_enabled=0):
        """Explicit synchronization after diagnostic population initialization."""
        self.s._update_streamed_phase()
        self.s.thermal._synchronize_conservative_volume()

    @ti.kernel
    def project_material(self, thermal_steps: ti.f64, include_geometry: ti.i32):
        initial_mass = self.material_cell_mass[None]
        water_mass = self.water_cell_mass[None]
        for a, b in self.s.thermal.body_solid_mass:
            owner = self.s.thermal.material_body_id[a, b]
            if owner >= 0:
                angle = ti.cast(self.s.body_angle[owner], ti.f64)
                c, sn = ti.cos(angle), ti.sin(angle)
                local = ti.Vector([
                    a - self.s.thermal.material_offset[owner] + 0.5 - 0.5 * self.s.thermal.material_size[owner].x,
                    b + 0.5 - 0.5 * self.s.thermal.material_size[owner].y,
                ], dt=ti.f64)
                origin = ti.cast(self.s.body_reference_origin[owner], ti.f64)
                center = origin + ti.Vector([c * local.x - sn * local.y, sn * local.x + c * local.y])
                extent = 0.5 * (ti.abs(c) + ti.abs(sn))
                base_i, base_j = ti.cast(ti.floor(center.x - extent), ti.i32), ti.cast(ti.floor(center.y - extent), ti.i32)
                weights = ti.Vector.zero(ti.f64, 9)
                total = ti.cast(0.0, ti.f64)
                last = -1
                for k in ti.static(range(9)):
                    i, j = base_i + k // 3, base_j + k % 3
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        if self.s.wall_mask[i, j] == 0:
                            weights[k] = square_overlap(center.x, center.y, c, sn, i, j)
                            total += weights[k]
                            if weights[k] > 0:
                                last = k
                mass = self.s.thermal.body_solid_mass[a, b]
                melted = ti.cast(0.0, ti.f64)
                heat = ti.cast(0.0, ti.f64)
                if thermal_steps > 0:
                    melted = self.s.thermal.melted_mass[a, b] / water_mass
                    heat = self.s.thermal.melted_sensible_energy[a, b]
                if total > 1e-14:
                    remaining_ice, remaining_melt, remaining_heat = mass / initial_mass, melted, heat
                    for k in ti.static(range(9)):
                        if weights[k] > 0:
                            w = weights[k] / total
                            ice, melt, energy = mass / initial_mass * w, melted * w, heat * w
                            if k == last:
                                ice, melt, energy = remaining_ice, remaining_melt, remaining_heat
                            remaining_ice -= ice
                            remaining_melt -= melt
                            remaining_heat -= energy
                            i, j = base_i + k // 3, base_j + k % 3
                            if include_geometry != 0:
                                ti.atomic_add(self.ice_volume[i, j], ice)

                            if thermal_steps > 0 and melt > 0:
                                # Stefan normal speed uses the material's
                                # spatial loss, before sharp-node deposition
                                # redirects water to an exposed fluid node.
                                ti.atomic_add(self.melt_rate[i, j], melt / thermal_steps)
                                # Deposit into the sharp fluid domain. A material
                                # cell can melt before a lattice node is exposed.
                                if self.cell(i, j):
                                    ti.atomic_add(self.melt_volume[i, j], melt)
                                    ti.atomic_add(self.melt_heat[i, j], energy)
                                else:
                                    radius, count = 1, 0
                                    best = 2147483647
                                    while count == 0 and radius < ti.static(max(self.nx, self.ny)):
                                        for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                                            if ti.max(ti.abs(di), ti.abs(dj)) == radius and self.cell(i + di, j + dj):
                                                distance = di * di + dj * dj
                                                if distance < best:
                                                    best, count = distance, 1
                                                elif distance == best:
                                                    count += 1
                                        if count == 0:
                                            radius += 1
                                    if count > 0:
                                        # Share equal nearest distances: choosing
                                        # the first ring cell biases horizontal
                                        # melt/cooling toward the negative x axis.
                                        for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                                            if di * di + dj * dj == best and self.cell(i + di, j + dj):
                                                ti.atomic_add(self.melt_volume[i + di, j + dj], melt / count)
                                                ti.atomic_add(self.melt_heat[i + di, j + dj], energy / count)
                                    else:
                                        ti.atomic_max(self.geometry_error[None], 1)
                elif mass > 0 or melted > 0:
                    ti.atomic_max(self.geometry_error[None], 1)

    @ti.kernel
    def clear_material_projection(self):
        self.geometry_error[None] = 0
        for i, j in self.capacity:
            self.ice_volume[i, j] = 0.0
            self.melt_volume[i, j] = 0.0
            self.melt_heat[i, j] = 0.0
            self.melt_rate[i, j] = 0.0

    @ti.kernel
    def deposit_melt(self, thermal_steps: ti.f64):
        for i, j in self.capacity:
            amount = self.melt_volume[i, j]
            if amount != 0:
                before = self.volume(i, j)
                velocity = self.s._physical_velocity_at(i, j)
                for q in ti.static(range(9)):
                    self.s.phase_populations[i, j, q] += phase_equilibrium(q, amount, velocity)
                self.close_moment(i, j, before + amount)
                self.s.water_phase[i, j] = self.volume(i, j)
                self.s.thermal.water_sensible_energy[i, j] += self.melt_heat[i, j]

    def project_melting(self, thermal_steps):
        self.clear_material_projection()
        self.project_material(float(thermal_steps), 1)
        if self.geometry_error[None]:
            raise RuntimeError("melt mass has no fluid recipient in the container")
        if thermal_steps > 0:
            self.deposit_melt(float(thermal_steps))

    @ti.kernel
    def initialize_components(self):
        for priority in range(3):
            self.recipient_count[priority] = 0
        for index in self.parent:
            i, j = index // self.ny, index % self.ny
            self.parent[index] = index if self.correction_open[i, j] != 0 else -1
            for priority in ti.static(range(3)):
                self.next[priority, index] = -1

    @ti.kernel
    def hook_components(self):
        self.changed[None] = 0
        for index in self.parent:
            if self.parent[index] >= 0:
                i, j = index // self.ny, index % self.ny
                for di, dj in ti.static(((1, 0), (0, 1), (1, 1), (1, -1))):
                    ni, nj = i + di, j + dj
                    if self.correction_cell(ni, nj):
                        a, b = self.root(index), self.root(ni * self.ny + nj)
                        if a != b:
                            old = ti.atomic_min(self.parent[ti.max(a, b)], ti.min(a, b))
                            if old > ti.min(a, b):
                                ti.atomic_max(self.changed[None], 1)

    @ti.kernel
    def compress_components(self):
        for index in self.parent:
            if self.parent[index] >= 0:
                self.parent[index] = self.root(index)

    @ti.kernel
    def build_recipients(self):
        for priority in range(3):
            self.recipient_count[priority] = 0
        for index in self.parent:
            root = self.parent[index]
            if root >= 0:
                i, j = index // self.ny, index % self.ny
                interface = 0.05 < self.s.water_phase[i, j] < 0.95
                for priority in ti.static(range(3)):
                    selected = False
                    if ti.static(priority == 0):
                        selected = self.spare[i, j] > 1e-13
                    elif ti.static(priority == 1):
                        selected = interface and self.available[i, j] > 1e-13 and self.pending[i, j] >= 0.0
                    else:
                        selected = self.available[i, j] > 1e-13 and self.pending[i, j] >= 0.0
                    if selected:
                        slot = ti.atomic_add(self.recipient_count[priority], 1)
                        self.next[priority, slot] = index

    @ti.func
    def reserve(self, field: ti.template(), i, j, amount):
        old = ti.atomic_add(field[i, j], -amount)
        accepted = ti.min(amount, ti.max(old, 0.0))
        ti.atomic_add(field[i, j], amount - accepted)
        return accepted

    @ti.kernel
    def gather_pool(self):
        for index in self.parent:
            if self.parent[index] == index:
                self.pool[index] = ti.Vector.zero(ti.f64, 8)
                self.pool[index][4] = 1e30
                self.pool[index][6] = 1e30
        for index in self.parent:
            root = self.parent[index]
            if root >= 0:
                i, j = index // self.ny, index % self.ny
                c = self.capacity[i, j]
                phase = self.available[i, j] / c if c > 1e-14 else 0.0
                # Pool only the free liquid surface, never a partial ice cell.
                surface = c >= 1.0 - 1e-6 and 0.05 < phase < 0.95
                flags = 0
                if self.available[i, j] > 1e-30:
                    self.specific_heat[i, j] = (self.s.thermal.water_sensible_energy[i, j] + self.heat_delta[i, j]) / self.available[i, j]
                positive = ti.max(self.pending[i, j], 0.0)
                debt = ti.max(-self.pending[i, j], 0.0)
                ti.atomic_add(self.pool[root][0], positive)
                ti.atomic_add(self.pool[root][1], self.pending_heat[i, j])
                ti.atomic_add(self.pool[root][2], debt)
                # Counts and minimum capacities permit uniform allocation
                # without summing either full-domain or interface water mass.
                if surface and self.spare[i, j] > 1e-13:
                    flags |= 1
                    ti.atomic_add(self.pool[root][3], 1.0)
                    ti.atomic_min(self.pool[root][4], self.spare[i, j])
                if surface and self.pending[i, j] >= 0 and self.available[i, j] > 1e-13:
                    flags |= 2
                    ti.atomic_add(self.pool[root][5], 1.0)
                    ti.atomic_min(self.pool[root][6], self.available[i, j])
                    ti.atomic_add(self.pool[root][7], self.specific_heat[i, j])
                self.interface[i, j] = ti.cast(flags, ti.i8)

    @ti.kernel
    def settle_event_pool(self):
        # Settle opposite signs proportionally. Atomic recipient-list order
        # must not decide where cold meltwater and hot displaced water go.
        # Reductions contain event budgets only, never the water inventory.
        for index in self.parent:
            root = self.parent[index]
            if root >= 0:
                i, j = index // self.ny, index % self.ny
                p = self.pool[root]
                paired = ti.min(p[0], p[2])
                if self.pending[i, j] > 0 and p[0] > 0:
                    fraction = 1.0 - paired / p[0]
                    self.pending[i, j] *= fraction
                    self.pending_heat[i, j] *= fraction
                elif self.pending[i, j] < 0 and p[2] > 0:
                    accepted = -self.pending[i, j] * paired / p[2]
                    self.pending[i, j] += accepted
                    self.heat_delta[i, j] += accepted * p[1] / ti.max(p[0], 1e-30)

    @ti.kernel
    def distribute_interface_pool(self):
        for index in self.parent:
            root = self.parent[index]
            if root >= 0:
                i, j = index // self.ny, index % self.ny
                p = self.pool[root]
                receive_each, donate_each = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
                if p[3] > 0:
                    receive_each = ti.min(p[0] / p[3], p[4])
                if p[5] > 0:
                    donate_each = ti.min(p[2] / p[5], p[6])
                added, borrowed = receive_each * p[3], donate_each * p[5]
                if self.interface[i, j] & 1:
                    self.delta[i, j] += receive_each
                    self.heat_delta[i, j] += receive_each * p[1] / ti.max(p[0], 1e-30)
                    self.spare[i, j] -= receive_each
                if self.interface[i, j] & 2:
                    self.delta[i, j] -= donate_each
                    self.heat_delta[i, j] -= donate_each * self.specific_heat[i, j]
                    self.available[i, j] -= donate_each
                    self.spare[i, j] += donate_each
                if self.pending[i, j] > 0:
                    fraction = 1.0 - added / ti.max(p[0], 1e-30)
                    self.pending[i, j] *= fraction
                    self.pending_heat[i, j] *= fraction
                elif self.pending[i, j] < 0:
                    settled = -self.pending[i, j] * borrowed / ti.max(p[2], 1e-30)
                    self.pending[i, j] += settled
                    self.heat_delta[i, j] += settled * p[7] / ti.max(p[5], 1e-30)

    @ti.kernel
    def distribute_fallback_pool(self):
        # Exceptional insufficient-interface cases use individual capacity
        # reservations. No bulk-water sum is computed even on this fallback.
        for index in self.parent:
            if self.parent[index] >= 0:
                i, j = index // self.ny, index % self.ny
                remaining = self.pending[i, j]
                if remaining > 1e-15:
                    specific = self.pending_heat[i, j] / remaining
                    slot = 0
                    while slot < self.recipient_count[0] and remaining > 1e-15:
                        other = self.next[0, slot]
                        ni, nj = other // self.ny, other % self.ny
                        if other != index and self.parent[other] == self.parent[index]:
                            accepted = self.reserve(self.spare, ni, nj, remaining)
                            ti.atomic_add(self.delta[ni, nj], accepted)
                            ti.atomic_add(self.heat_delta[ni, nj], accepted * specific)
                            remaining -= accepted
                        slot += 1
                    self.pending[i, j] = remaining
                    self.pending_heat[i, j] = remaining * specific

    @ti.kernel
    def fill_requests(self):
        for index in self.parent:
            if self.parent[index] >= 0:
                i, j = index // self.ny, index % self.ny
                debt = ti.max(0.0, -self.pending[i, j])
                remaining = debt
                if remaining > 1e-15:
                    for priority in ti.static(range(1, 3)):
                        slot = 0
                        while slot < self.recipient_count[priority] and remaining > 1e-15:
                            other = self.next[priority, slot]
                            ni, nj = other // self.ny, other % self.ny
                            if other != index and self.parent[other] == self.parent[index]:
                                accepted = self.reserve(self.available, ni, nj, remaining)
                                energy = accepted * self.specific_heat[ni, nj]
                                ti.atomic_add(self.delta[ni, nj], -accepted)
                                ti.atomic_add(self.heat_delta[ni, nj], -energy)
                                repaid = ti.min(debt, accepted)
                                debt -= repaid
                                ti.atomic_add(self.delta[i, j], accepted - repaid)
                                ti.atomic_add(self.heat_delta[i, j], energy)
                                remaining -= accepted
                            slot += 1
                    if self.pending[i, j] < 0:
                        self.pending[i, j] = -debt

    @ti.func
    def open_l_path(self, i, j, ni, nj):
        horizontal_first = True
        for k in range(ti.abs(ni - i) + 1):
            x = i + k * (1 if ni >= i else -1)
            horizontal_first = horizontal_first and self.geometry_open(x, j)
        for k in range(ti.abs(nj - j) + 1):
            y = j + k * (1 if nj >= j else -1)
            horizontal_first = horizontal_first and self.geometry_open(ni, y)
        vertical_first = True
        if not horizontal_first:
            for k in range(ti.abs(nj - j) + 1):
                y = j + k * (1 if nj >= j else -1)
                vertical_first = vertical_first and self.geometry_open(i, y)
            for k in range(ti.abs(ni - i) + 1):
                x = i + k * (1 if ni >= i else -1)
                vertical_first = vertical_first and self.geometry_open(x, nj)
        return horizontal_first or vertical_first

    @ti.kernel
    def repair_negative_tails(self):
        # Negative values in the diffuse air tail have no local liquid seed.
        # Repair only debts that would otherwise leave negative physical water,
        # using the nearest reachable liquid donor. Positive droplets retain
        # their own labels and cannot compensate one another.
        for index in self.parent:
            i, j = index // self.ny, index % self.ny
            if self.pending[i, j] < 0:
                provisional = self.volume(i, j) + self.delta[i, j] + self.pending[i, j]
                if provisional < -1e-12:
                    remaining = -self.pending[i, j]
                    attempts = 0
                    while remaining > 1e-13 and attempts < 16:
                        nearest, distance = -1, ti.cast(1e30, ti.f64)
                        for slot in range(self.recipient_count[1]):
                            other = self.next[1, slot]
                            ni, nj = other // self.ny, other % self.ny
                            d = ti.cast((ni - i)**2 + (nj - j)**2, ti.f64)
                            if self.available[ni, nj] > 1e-13 and d < distance:
                                if self.open_l_path(i, j, ni, nj):
                                    nearest, distance = other, d
                        if nearest < 0:
                            break
                        ni, nj = nearest // self.ny, nearest % self.ny
                        accepted = self.reserve(self.available, ni, nj, remaining)
                        energy = accepted * self.specific_heat[ni, nj]
                        ti.atomic_add(self.delta[ni, nj], -accepted)
                        ti.atomic_add(self.heat_delta[ni, nj], -energy)
                        self.heat_delta[i, j] += energy
                        remaining -= accepted
                        attempts += 1
                    self.pending[i, j] = -remaining

    @ti.kernel
    def commit_transaction(self):
        self.unresolved[None] = 0
        for i, j in self.capacity:
            value = self.target_volume[i, j] + self.delta[i, j]
            if self.pending[i, j] < 0:
                # Unavailable water cannot be fabricated. Undo unpaid credits
                # (e.g. an isolated melting pore) before exposing the state.
                value += self.pending[i, j]
                self.delta[i, j] += self.pending[i, j]
                self.pending[i, j] = 0.0
            if ti.abs(self.pending[i, j]) > 1e-10 or value < -1e-10 or value > self.capacity[i, j] + 1e-10:
                ti.atomic_max(self.unresolved[None], 1)
            velocity = self.s._physical_velocity_at(i, j)
            if self.delta[i, j] != 0:
                for q in ti.static(range(9)):
                    self.s.phase_populations[i, j, q] += phase_equilibrium(q, self.delta[i, j], velocity)
                self.close_moment(i, j, value)
            self.s.water_phase[i, j] = value / self.capacity[i, j] if self.cell(i, j) else 0.0
            self.s.thermal.water_volume_m2[i, j] = value * self.cell_area[None]
            self.s.thermal.water_sensible_energy[i, j] += self.heat_delta[i, j]
            self.event_volume[i, j] = 0.0
            self.event_heat[i, j] = 0.0
            self.melt_volume[i, j] = 0.0
            self.melt_heat[i, j] = 0.0

    @ti.kernel
    def relocate_closed_gap_events(self):
        for i, j in self.pending:
            if self.capacity[i, j] == 0 and self.pending[i, j] > 1e-13:
                remaining = self.pending[i, j]
                specific = self.pending_heat[i, j] / remaining
                attempts = 0
                while remaining > 1e-13 and attempts < 16:
                    radius, nearest, distance = 1, -1, 2147483647
                    while nearest < 0 and radius < ti.static(max(self.nx, self.ny)):
                        for di, dj in ti.ndrange((-radius, radius+1), (-radius, radius+1)):
                            ni, nj = i+di, j+dj
                            if ti.max(ti.abs(di), ti.abs(dj)) == radius and self.cell(ni, nj):
                                squared = di*di + dj*dj
                                if self.spare[ni, nj] > 1e-13 and squared < distance:
                                    if self.open_l_path(i, j, ni, nj):
                                        nearest, distance = ni*self.ny+nj, squared
                        radius += 1
                    if nearest < 0:
                        break
                    ni, nj = nearest // self.ny, nearest % self.ny
                    accepted = self.reserve(self.spare, ni, nj, remaining)
                    ti.atomic_add(self.delta[ni, nj], accepted)
                    ti.atomic_add(self.heat_delta[ni, nj], accepted*specific)
                    remaining -= accepted
                    attempts += 1
                self.pending[i, j] = remaining
                self.pending_heat[i, j] = remaining*specific

    def remap(self, geometry=True):
        self.prepare_transaction()
        status = int(self.events[None])
        if status >> 4:
            raise RuntimeError(f"conservative phase geometry invalid ({status >> 4}): overlapping or out-of-domain material")
        if status & 1:
            self.classify_correction_core()
            self.classify_correction_graph()
            if not self.components_ready or self.topology_dirty[None]:
                self.initialize_components()
                for _ in range(32):
                    self.hook_components()
                    self.compress_components()
                    if not self.changed[None]:
                        break
                else:
                    raise RuntimeError("conservative phase connectivity did not converge")
                self.components_ready = True
                self.topology_dirty[None] = 0
            self.gather_pool()
            self.settle_event_pool()
            self.gather_pool()
            self.build_recipients()
            self.distribute_interface_pool()
            self.distribute_fallback_pool()
            self.fill_requests()
            self.repair_negative_tails()
            if self.s.config.close_subgrid_wall_gaps:
                self.relocate_closed_gap_events()
        self.commit_transaction()
        if self.unresolved[None]:
            raise RuntimeError("conservative phase transfer has insufficient capacity in a connected fluid region")

    @ti.func
    def boundary_offset(self, i, j, ni, nj, owner):
        gradient = ti.Vector.zero(ti.f64, 2)
        for axis in ti.static(range(2)):
            di, dj = (1, 0) if axis == 0 else (0, 1)
            plus, minus = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
            if 0 <= ni + di < self.nx and 0 <= nj + dj < self.ny:
                plus = ti.cast(self.s._body_fractions[owner, ni + di, nj + dj], ti.f64)
            if 0 <= ni - di < self.nx and 0 <= nj - dj < self.ny:
                minus = ti.cast(self.s._body_fractions[owner, ni - di, nj - dj], ti.f64)
            gradient[axis] = 0.5 * (plus - minus)
        length = gradient.norm()
        normal = -gradient / ti.max(length, 1e-12)
        rate = 0.5 * (self.melt_rate[i, j] + self.melt_rate[ni, nj])
        # j(1/rho_w-1/rho_i), with j obtained from the projected volume
        # production rate and the regularized interface measure |grad alpha_i|.
        return ti.cast(normal * (1.0 - self.s.config.rho_water / self.s.config.rho_ice) * rate / ti.max(length, 0.05), ti.f32)
