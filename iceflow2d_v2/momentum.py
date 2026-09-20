"""Conservative pressure/stress flux correction around the D2Q9 predictor.

The LB convective candidate is separated from traction. Accepted phase and
volume fluxes then carry momentum with the same donor weights as enthalpy.
Local SNIBB impulses are paired exactly with the rigid solver; they are not
inferred from a posteriori changes in velocity or global momentum corrections.
"""
import taichi as ti
from .lattice import direction, opposite, weight, cross
from .transport import LINK_Q


@ti.data_oriented
class Momentum:
    def __init__(self, cfg, geo, state, flow, transport):
        self.cfg, self.geo, self.state, self.flow, self.transport = cfg, geo, state, flow, transport
        self.traction = ti.Matrix.field(4, 2, ti.f32, (cfg.nx, cfg.ny))
        self.torque_target = ti.Vector.field(2, ti.i32, (cfg.nx, cfg.ny))
        self.torque_pair = ti.Vector.field(2, ti.f32, (cfg.nx, cfg.ny))

    @ti.kernel
    def build_tractions(self):
        for i, j in self.traction:
            value = ti.Matrix.zero(ti.f32, 4, 2)
            for k in ti.static(range(4)):
                q = ti.static(LINK_Q[k])
                e = direction(q)
                ni, nj = i+e.x, j+e.y
                if self.geo.fluid(i, j) and self.geo.fluid(ni, nj):
                    aperture = self.transport.aperture(i, j, ni, nj)
                    population = self.flow.fpost[q, i, j]+self.flow.fpost[opposite(q), ni, nj]
                    population += self.flow.capillary_population(q, i, j)+self.flow.capillary_population(opposite(q), ni, nj)
                    flux = ti.cast(e, ti.f32)*aperture*population
                    volume = self.transport.pred[i, j][k]
                    di, dj = i, j
                    if volume < 0.0: di, dj = ni, nj
                    donor = self.state.value(di, dj)
                    mass_density = (donor[0]+ti.static(self.cfg.rho_a_lu)*donor[1])/ti.max(self.geo.old_capacity[di, dj], 1e-20)
                    # Replace the baseline convective flux only once. The accepted
                    # remap/phase operators will supply its conservative counterpart.
                    flux -= volume*mass_density*self.flow.velocity[di, dj]
                    value[k, 0], value[k, 1] = flux.x, flux.y
            self.traction[i, j] = value
            impulse = self.flow.boundary_impulse[i, j]-self.flow.boundary_reference_impulse[i, j]
            point = ti.Vector([i+0.5, j+0.5])
            force = ti.Vector([impulse.x, impulse.y])
            missing_torque = -impulse.z-cross(self.geo.origin[None]-point, force)
            target, pair = ti.Vector([-1, -1]), ti.Vector([0.0, 0.0])
            for q in range(1, 5):
                e = direction(q)
                ni, nj = i+e.x, j+e.y
                if target.x < 0 and self.geo.fluid(ni, nj):
                    target = ti.Vector([ni, nj])
                    pair = missing_torque*ti.Vector([-ti.cast(e.y, ti.f32), ti.cast(e.x, ti.f32)])
            self.torque_target[i, j], self.torque_pair[i, j] = target, pair

    @ti.kernel
    def commit(self):
        for i, j in self.state.q:
            state = self.state.value(i, j)
            mass = state[0]+ti.static(self.cfg.rho_a_lu)*state[1]
            gravity = ti.Vector([0.0, -ti.static(self.cfg.gravity_lu)])
            reference_mass = self.flow.reference_density[i, j]*self.geo.old_capacity[i, j]
            force = (mass-reference_mass)*gravity
            if ti.static(self.cfg.buoyancy):
                # Conservative fluid in a cut cell exists even when its centre
                # is solid in the legacy SDF mask. Apply its physical body force
                # on that volume too, using the same Boussinesq law.
                phi = state[0]/ti.max(self.geo.old_capacity[i, j], 1e-20)
                force.y += ti.min(1.0, ti.max(0.0, 2*phi-1))*ti.static(self.cfg.beta*self.cfg.gravity_lu)*(self.state.temperature_w[i, j]*ti.static(self.cfg.temperature_scale)-ti.static(self.cfg.water_temperature))*self.geo.old_capacity[i, j]
            if self.geo.fluid(i, j):
                # Replace the predictor's pointwise chemical force by its
                # conservative stress flux, including body/wall reactions.
                force = mass*(self.flow.force[i, j]-self.flow.capillary_acceleration[i, j])
            # Preserve the frozen hydrostatic split consistently. Reintroducing
            # reference pressure through cut-face footprints while using SNIBB's
            # full link quadrature would create an unmatched hydrostatic force.
            impulse = (self.flow.boundary_impulse[i, j]-self.flow.boundary_reference_impulse[i, j]
                       +self.flow.wall_impulse[i, j]-self.flow.wall_reference_impulse[i, j])
            delta = force-ti.Vector([impulse.x, impulse.y])
            for k in range(4):
                e = self.transport.adjacent(i, j, k)
                ni, nj = i-e.x, j-e.y
                delta -= ti.Vector([self.traction[i, j][k, 0], self.traction[i, j][k, 1]])
                if self.geo.inside(ni, nj): delta += ti.Vector([self.traction[ni, nj][k, 0], self.traction[ni, nj][k, 1]])
            delta -= self.torque_pair[i, j]
            for q in range(1, 5):
                e = direction(q)
                ni, nj = i+e.x, j+e.y
                if self.geo.inside(ni, nj):
                    target = self.torque_target[ni, nj]
                    if target.x == i and target.y == j: delta += self.torque_pair[ni, nj]
            self.state.add(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0, delta.x, delta.y]))

    def step(self):
        self.build_tractions()
        self.commit()
