"""Single accepted Stefan exchange ledger and fixed-density numerical air."""
import taichi as ti
from .lattice import add_compensated, cross, direction
from .connectivity import Components
from .reduction import DeviceSum


@ti.data_oriented
class Coupling:
    def __init__(self, config, geometry, state, flow):
        self.cfg, self.geo, self.state, self.flow = config, geometry, state, flow
        body = (config.nb, config.nb, 12)
        world = (config.nx, config.ny)
        self.receiver = ti.Vector.field(2, ti.i32, body)
        self.point = ti.Vector.field(2, ti.f32, body)
        self.normal = ti.Vector.field(2, ti.f32, body)
        self.material_velocity = ti.Vector.field(2, ti.f32, body)
        self.liquid_velocity = ti.Vector.field(2, ti.f32, body)
        self.heat = ti.field(ti.f32, body)
        self.mass = ti.field(ti.f32, body)
        self.q_s = ti.field(ti.f32, body)
        self.exchange = ti.Vector.field(6, ti.f32, body)
        self.torque_extra = ti.field(ti.f64, body)
        self.heat_requested = ti.field(ti.f32, world)
        self.air_connected = ti.field(ti.i32, world)
        self.air_next = ti.field(ti.i32, world)
        self.air_components = Components(config.nx, config.ny)
        self.melt_step = ti.field(ti.f32, ())
        self.melt_total = ti.field(ti.f32, ())
        self.melt_comp = ti.field(ti.f32, ())
        self.air_volume = ti.field(ti.f32, ())
        self.air_source_total = ti.field(ti.f32, ())
        self.air_source_comp = ti.field(ti.f32, ())
        self.air_heat_total = ti.field(ti.f32, ())
        self.air_heat_comp = ti.field(ti.f32, ())
        self.air_heat_step = ti.field(ti.f32, world)
        self.torque_target = ti.Vector.field(2, ti.i32, world)
        self.torque_pair = ti.Vector.field(2, ti.f32, world)
        self.failure = ti.field(ti.i32, ())
        self.air_sum = DeviceSum(config.nx*config.ny, 1)
        self.melt_sum = DeviceSum(config.nb*config.nb*12, 1)
        self.heat_sum = DeviceSum(config.nx*config.ny, 1)

    @ti.kernel
    def seed_air(self):
        for i, j in self.air_connected:
            value = self.state.value(i, j)
            self.air_next[i, j] = ti.cast(value[1] > value[0] and value[1] > 1e-12 and not self.geo.wall(i, j), ti.i32)

    @ti.kernel
    def propagate_air(self):
        for i, j in self.air_connected:
            connected = self.air_connected[i, j]
            value = self.state.value(i, j)
            if value[1] > value[0] and value[1] > 1e-12 and not self.geo.wall(i, j):
                for axis in ti.static(range(2)):
                    for sign in ti.static((-1, 1)):
                        ni, nj = i+sign*(1-axis), j+sign*axis
                        if self.geo.inside(ni, nj): connected = ti.max(connected, self.air_connected[ni, nj])
            self.air_next[i, j] = connected

    @ti.kernel
    def accept_air(self):
        for i, j in self.air_connected: self.air_connected[i, j] = self.air_next[i, j]

    @ti.func
    def air_value(self, index):
        i, j = index//self.cfg.ny, index % self.cfg.ny
        value = 0.0
        if self.air_connected[i, j] != 0: value = self.state.value(i, j)[1]
        return ti.Vector([value])

    @ti.func
    def air_result(self, value):
        self.air_volume[None] = value[0]

    def reduce_air(self):
        self.air_sum.run(self.air_value, self.air_result)

    @ti.kernel
    def propose_heat(self):
        for a, b, k in self.heat:
            amount, receiver = 0.0, ti.Vector([-1, -1])
            point, normal = self.geo.surface_point[a, b, k], self.geo.surface_normal[a, b, k]
            length = self.geo.surface_length[a, b, k]
            if ti.static(self.cfg.melting):
                if length > 0.0 and self.geo.material_fraction(a, b) > 0.0:
                    # Heat belongs to the fluid touching this face. A nearest
                    # hot-cell search can borrow heat through an intervening
                    # solid strip and send melt into a disconnected receiver.
                    epsilon = ti.cast(32*2.220446049250313e-16, ti.f64)*ti.max(1.0, ti.abs(point).max())
                    base = ti.cast(ti.floor(point+epsilon*normal), ti.i32)
                    i, j = base.x, base.y
                    if self.geo.inside(i, j) and not self.geo.wall(i, j):
                        if self.geo.capacity[i, j] > 0.0 and self.state.value(i, j)[0] > 1e-12:
                            receiver = base
                    if receiver.x >= 0:
                        i, j = receiver.x, receiver.y
                        phase = self.state.value(i, j)[0]/self.geo.capacity[i, j]
                        distance = ti.max(0.25, ti.abs((self.state.heat_position[i, j]-point).dot(normal)))
                        conductance = ti.static(self.cfg.k_w*self.cfg.dt/(self.cfg.rho_w*self.cfg.cp_w*self.cfg.dx**2))*length*phase/distance
                        # An implicit water-side resistance prevents overcooling a thin film.
                        capacity = self.state.value(i, j)[0]
                        conductance /= 1+conductance/ti.max(capacity, 1e-30)
                        temperature = self.state.temperature_w[i, j]
                        amount = ti.max(0.0, conductance*temperature)
            self.receiver[a, b, k] = receiver
            self.point[a, b, k], self.normal[a, b, k] = point, normal
            self.heat[a, b, k] = amount

    @ti.kernel
    def gather_heat_requests(self):
        for i, j in self.heat_requested:
            point = ti.Vector([i+0.5, j+0.5])
            local = self.geo.rotation[None].transpose()@(point-self.geo.origin[None])
            base = ti.cast(ti.floor(local+0.5*self.cfg.nb), ti.i32)
            value = 0.0
            for di, dj in ti.ndrange((-3, 4), (-3, 4)):
                a, b = base.x+di, base.y+dj
                if 0 <= a < self.cfg.nb and 0 <= b < self.cfg.nb:
                    for k in range(12):
                        r = self.receiver[a, b, k]
                        if r.x == i and r.y == j: value += self.heat[a, b, k]
            self.heat_requested[i, j] = value

    @ti.kernel
    def accept_exchanges(self):
        for a, b in self.geo.solid:
            total = 0.0
            for k in range(12):
                r, q = self.receiver[a, b, k], self.heat[a, b, k]
                if r.x >= 0:
                    available = ti.max(0.0, self.state.value(r.x, r.y)[2])
                    q *= ti.min(1.0, 0.5*available/ti.max(self.heat_requested[r.x, r.y], 1e-30))
                self.heat[a, b, k] = q
                total += q
            fraction = self.geo.material_fraction(a, b)
            old_heat = self.geo.ice_heat[a, b]-self.geo.ice_heat_comp[a, b]
            sensible = ti.max(0.0, -old_heat)
            # Limit geometric motion, with exact exhaustion of a final small cell.
            max_melt = ti.min(fraction, 0.2)*ti.static(self.cfg.rho_s_lu)
            accepted = ti.min(total, sensible+max_melt*ti.static(self.cfg.latent_lu))
            if self.air_volume[None] <= 1e-20 and accepted > sensible:
                self.failure[None] = 1
                accepted = ti.min(accepted, sensible)
            qs = ti.min(accepted, sensible)
            mass = (accepted-qs)/ti.static(self.cfg.latent_lu)
            for k in range(12):
                self.torque_extra[a, b, k] = 0.0
                share = self.heat[a, b, k]/ti.max(total, 1e-30)
                dm, ql, qsolid = share*mass, share*accepted, share*qs
                self.heat[a, b, k], self.mass[a, b, k], self.q_s[a, b, k] = ql, dm, qsolid
                vs = self.geo.body_velocity(self.point[a, b, k])
                length = self.geo.surface_length[a, b, k]
                flux = dm/ti.max(length, 1e-30)
                vl = vs+flux*(1-1/ti.static(self.cfg.rho_s_lu))*self.normal[a, b, k]
                self.material_velocity[a, b, k], self.liquid_velocity[a, b, k] = vs, vl
                self.exchange[a, b, k] = ti.Vector([dm, 0.0, -ql, 0.0, dm*vl.x, dm*vl.y])
            if accepted > 0.0:
                self.geo.solid[a, b], self.geo.solid_comp[a, b] = add_compensated(self.geo.solid[a, b], self.geo.solid_comp[a, b], -mass/ti.static(self.cfg.rho_s_lu))
                self.geo.ice_heat[a, b], self.geo.ice_heat_comp[a, b] = add_compensated(self.geo.ice_heat[a, b], self.geo.ice_heat_comp[a, b], qs)

    @ti.func
    def melt_value(self, index):
        a, b, k = index//(self.cfg.nb*12), (index//12) % self.cfg.nb, index % 12
        return ti.Vector([self.mass[a, b, k]])

    @ti.func
    def melt_result(self, vector):
        value = vector[0]
        self.melt_step[None] = value
        self.melt_total[None], self.melt_comp[None] = add_compensated(self.melt_total[None], self.melt_comp[None], value)
        air = value*(1/ti.static(self.cfg.rho_s_lu)-1)
        self.air_source_total[None], self.air_source_comp[None] = add_compensated(self.air_source_total[None], self.air_source_comp[None], air)

    def reduce_melt(self):
        self.melt_sum.run(self.melt_value, self.melt_result)

    @ti.kernel
    def gather_sources(self):
        for i, j in self.state.source:
            point = ti.Vector([i+0.5, j+0.5])
            local = self.geo.rotation[None].transpose()@(point-self.geo.origin[None])
            base = ti.cast(ti.floor(local+0.5*self.cfg.nb), ti.i32)
            source = ti.Vector.zero(ti.f32, 6)
            torque = ti.cast(0.0, ti.f64)
            slip, contact = ti.Vector.zero(ti.f32, 2), 0.0
            for di, dj in ti.ndrange((-3, 4), (-3, 4)):
                a, b = base.x+di, base.y+dj
                if 0 <= a < self.cfg.nb and 0 <= b < self.cfg.nb:
                    for k in range(12):
                        r = self.receiver[a, b, k]
                        if r.x == i and r.y == j:
                            source += self.exchange[a, b, k]
                            p = ti.Vector([self.exchange[a, b, k][4], self.exchange[a, b, k][5]])
                            torque += cross(self.point[a, b, k]-point, p)
                            torque += self.torque_extra[a, b, k]
                            length = self.geo.surface_length[a, b, k]
                            slip += length*(self.liquid_velocity[a, b, k]-self.material_velocity[a, b, k])
                            contact += length
            air = 0.0
            if self.air_connected[i, j] != 0:
                air = self.melt_step[None]*(1/ti.static(self.cfg.rho_s_lu)-1)*self.state.value(i, j)[1]/ti.max(self.air_volume[None], 1e-30)
            heat = air*ti.static(self.cfg.air_capacity_lu)*self.state.temperature_a[i, j]
            source[1] += air
            source[3] += air*ti.static(self.cfg.air_capacity_lu)*(self.state.temperature_a[i, j]-ti.static(self.cfg.air_temperature/self.cfg.temperature_scale))
            # The numerical gas is created with the local gas velocity, its
            # artificial momentum is a separate term in the diagnostic budget.
            source[4] += air*ti.static(self.cfg.rho_a_lu)*self.flow.velocity[i, j].x
            source[5] += air*ti.static(self.cfg.rho_a_lu)*self.flow.velocity[i, j].y
            self.air_heat_step[i, j] = heat
            self.state.source[i, j] = source
            self.flow.dilatation[i, j] = air/ti.max(self.geo.capacity[i, j], 1e-20)
            self.flow.boundary_slip[i, j] = slip/ti.max(contact, 1e-30)
            target, pair, best = ti.Vector([-1, -1]), ti.Vector([0.0, 0.0]), 0.0
            if torque != 0.0:
                for q in range(1, 9):
                    e = direction(q)
                    ni, nj = i+e.x, j+e.y
                    # The receiver belongs to the conservative cut-volume graph;
                    # its centre may still be solid in the SDF/LB node mask.
                    if self.geo.inside(ni, nj) and not self.geo.wall(ni, nj) and self.geo.capacity[ni, nj] > 0.0 and self.geo.swept_aperture(i, j, ni, nj) > 0.0:
                        state = self.state.value(ni, nj)
                        mass = state[0]+ti.static(self.cfg.rho_a_lu)*state[1]
                        if mass > best:
                            best, target = mass, ti.Vector([ni, nj])
                            pair = torque*ti.Vector([-ti.cast(e.y, ti.f32), ti.cast(e.x, ti.f32)])/e.norm_sqr()
                if target.x < 0: self.failure[None] = 2
            self.torque_target[i, j], self.torque_pair[i, j] = target, pair

    @ti.kernel
    def transfer_interface_torque(self):
        # Relocate the interface impulse to cell centres by a connected local
        # force couple. Its net force is zero and moment is (x_Gamma-x_i) x P.
        # The rigid body still loses exactly the ledger impulse at x_Gamma.
        for i, j in self.state.source:
            delta = -self.torque_pair[i, j]
            for q in range(1, 9):
                e = direction(q)
                ni, nj = i+e.x, j+e.y
                if self.geo.inside(ni, nj):
                    if (self.torque_target[ni, nj] == ti.Vector([i, j])).all():
                        delta += self.torque_pair[ni, nj]
            self.state.source[i, j][4] += delta.x
            self.state.source[i, j][5] += delta.y

    @ti.func
    def air_heat_value(self, index):
        return ti.Vector([self.air_heat_step[index//self.cfg.ny, index % self.cfg.ny]])

    @ti.func
    def air_heat_result(self, vector):
        value = vector[0]
        self.air_heat_total[None], self.air_heat_comp[None] = add_compensated(self.air_heat_total[None], self.air_heat_comp[None], value)

    def reduce_air_heat(self):
        self.heat_sum.run(self.air_heat_value, self.air_heat_result)

    def prepare(self):
        self.seed_air()
        self.air_components.seed(self.air_next, 0.0)
        self.air_components.connect(self.geo.face_open, False)
        self.air_components.mark_top(self.cfg.ny-self.cfg.wall_cells-1)
        self.air_components.connected_to_top(self.air_connected)
        self.reduce_air()
        self.propose_heat(); self.gather_heat_requests(); self.accept_exchanges()
        self.reduce_melt()

    def finish(self):
        self.gather_sources(); self.transfer_interface_torque(); self.reduce_air_heat()

    def step(self):
        self.prepare(); self.finish()
