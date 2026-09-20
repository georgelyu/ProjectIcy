"""Extensive authority: volumes, water heat, reference-relative air heat, P.

AIR_HEAT stores E_air-C_air*V_air*T_air_initial. This exact change of reference
prevents cancellation of the entire heat inventory when a gas phase vanishes.
"""
import taichi as ti
from .lattice import add_compensated, direction, weight, phase_equilibrium, pressure_equilibrium

WATER, AIR, WATER_HEAT, AIR_HEAT, PX, PY = range(6)


@ti.data_oriented
class State:
    def __init__(self, config, geometry, flow):
        self.cfg, self.geo, self.flow = config, geometry, flow
        shape = (config.nx, config.ny)
        self.q = ti.Vector.field(6, ti.f32, shape)
        self.comp = ti.Vector.field(6, ti.f32, shape)
        self.source = ti.Vector.field(6, ti.f32, shape)
        self.temperature_w = ti.field(ti.f32, shape)
        self.temperature_a = ti.field(ti.f32, shape)
        # A merged temperature is a patch average. Its conduction resistance
        # must use that patch's sampling position, not every member's centre.
        self.heat_position = ti.Vector.field(2, ti.f32, shape)
        self.new_pressure = ti.field(ti.f32, shape)
        self.new_f_neq = ti.field(ti.f32, (9, *shape))
        self.new_h_neq = ti.field(ti.f32, (9, *shape))
        self.initialize()

    @ti.func
    def value(self, i, j):
        return self.q[i, j]-self.comp[i, j]

    @ti.func
    def volume(self, i, j):
        # The Neumann compatibility condition is a global volume identity.
        # Rounding q-comp to f32 here discards the very low parts we retain.
        return (ti.cast(self.q[i, j][0], ti.f64)-ti.cast(self.comp[i, j][0], ti.f64)
                +ti.cast(self.q[i, j][1], ti.f64)-ti.cast(self.comp[i, j][1], ti.f64))

    @ti.func
    def add(self, i, j, increment):
        self.q[i, j], self.comp[i, j] = add_compensated(self.q[i, j], self.comp[i, j], increment)

    @ti.kernel
    def initialize(self):
        for i, j in self.q:
            water = self.geo.capacity[i, j]*self.flow.phi[i, j]
            air = self.geo.capacity[i, j]-water
            tw, ta = ti.static(self.cfg.water_temperature/self.cfg.temperature_scale), ti.static(self.cfg.air_temperature/self.cfg.temperature_scale)
            exact = ti.Vector([water, air, water*tw, 0.0, 0.0, 0.0], dt=ti.f64)
            high = ti.cast(exact, ti.f32)
            self.q[i, j] = high
            self.comp[i, j] = ti.cast(ti.cast(high, ti.f64)-exact, ti.f32)
            self.temperature_w[i, j], self.temperature_a[i, j] = tw, ta
            self.heat_position[i, j] = ti.Vector([i+.5, j+.5])

    @ti.kernel
    def temperatures(self):
        for i, j in self.q:
            state = self.value(i, j)
            tw, ta = 0.0, ti.static(self.cfg.air_temperature/self.cfg.temperature_scale)
            if state[0] > 1e-20: tw = state[2]/state[0]
            if state[1] > 1e-20: ta += state[3]/(state[1]*ti.static(self.cfg.air_capacity_lu))
            self.temperature_w[i, j], self.temperature_a[i, j] = tw, ta
            self.flow.temperature[i, j] = tw

    @ti.kernel
    def prepare_new_nodes(self):
        # Separate gather pass: restore_populations writes macros, so reading
        # neighbours there would make extrapolation depend on thread ordering.
        for i, j in self.q:
            if self.geo.fluid(i, j) and self.geo.old_mask[i, j] != 0:
                fp, hp = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
                pressure, total = 0.0, 0.0
                for link in range(1, 9):
                    e = direction(link); ni, nj = i+e.x, j+e.y
                    if self.geo.fluid(ni, nj) and self.geo.old_mask[ni, nj] == 0 and self.geo.old_capacity[ni, nj] > 1e-6:
                        w = 1.0/ti.cast(e.dot(e), ti.f32)
                        p, phi = self.flow.pressure[ni, nj], self.flow.phi[ni, nj]
                        u, rho = self.flow.velocity[ni, nj], self.flow.rho(phi)
                        pressure += w*p; total += w
                        for q in range(9):
                            fp[q] += w*(self.flow.f[q, ni, nj]-pressure_equilibrium(q, p, rho, u))
                            hp[q] += w*(self.flow.h[q, ni, nj]-phase_equilibrium(q, phi, u))
                self.new_pressure[i, j] = pressure/ti.max(total, 1e-30)
                for q in range(9):
                    self.new_f_neq[q, i, j] = fp[q]/ti.max(total, 1e-30)
                    self.new_h_neq[q, i, j] = hp[q]/ti.max(total, 1e-30)

    @ti.kernel
    def restore_populations(self):
        for i, j in self.q:
            state, volume = self.value(i, j), self.geo.capacity[i, j]
            phi, u = 0.0, ti.Vector.zero(ti.f32, 2)
            if volume > 1e-20:
                phi = state[0]/volume
                mass = state[0]+ti.static(self.cfg.rho_a_lu)*state[1]
                if mass > 1e-14 and state[0]+state[1] > 1e-12:
                    u = ti.Vector([state[4], state[5]])/mass
                elif self.geo.mask[i, j] != 0:
                    # An empty, masked cell can retain f32 compensated residues.
                    # Extrapolate its macro velocity; never divide P by those
                    # residues, and never discard its extensive state.
                    u = self.geo.body_velocity(ti.Vector([i+0.5, j+0.5]))
            rho = self.flow.rho(phi)
            target = rho*(u-0.5*self.flow.force[i, j])
            if self.geo.fluid(i, j):
                old_phi, old_u = self.flow.phi[i, j], self.flow.velocity[i, j]
                old_rho = self.flow.rho(old_phi)
                if self.geo.old_mask[i, j] != 0:
                    self.flow.pressure[i, j] = self.new_pressure[i, j]
                    self.flow.force[i, j] = ti.Vector([0.0, 0.0])
                    target = rho*u
                    for k in range(9):
                        self.flow.f[k, i, j] = pressure_equilibrium(k, self.flow.pressure[i, j], rho, u)+self.new_f_neq[k, i, j]
                        self.flow.h[k, i, j] = phase_equilibrium(k, phi, u)+self.new_h_neq[k, i, j]
                else:
                    # Retain non-equilibrium moments about the accepted state.
                    # Correcting only the first moment leaves the old rho*u*u
                    # in the stress after a phase/volume/merge update.
                    for k in range(9):
                        self.flow.f[k, i, j] += (pressure_equilibrium(k, self.flow.pressure[i, j], rho, u)
                                                -pressure_equilibrium(k, self.flow.pressure[i, j], old_rho, old_u))
                        self.flow.h[k, i, j] += phase_equilibrium(k, phi, u)-phase_equilibrium(k, old_phi, old_u)
                hsum, raw = 0.0, ti.Vector.zero(ti.f32, 2)
                for k in range(9):
                    hsum += self.flow.h[k, i, j]
                    raw += ti.cast(direction(k), ti.f32)*self.flow.f[k, i, j]
                self.flow.h[0, i, j] += phi-hsum
                for k in range(9):
                    self.flow.f[k, i, j] += 3*weight(k)*ti.cast(direction(k), ti.f32).dot(target-raw)
            self.flow.phi[i, j] = phi
            self.flow.velocity[i, j] = u
            self.flow.raw_velocity[i, j] = u-0.5*self.flow.force[i, j]
