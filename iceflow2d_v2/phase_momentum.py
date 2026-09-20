"""Finite-volume, implicit momentum transfer for an eroding rigid body.

The removed material has the measured area, first moment and polar moment of
the actual old-minus-new polygons. Its integrated momentum is not a point mass
placed on the old boundary. A common implicit twist closes remaining-body and
outgoing-material momentum, including the prescribed Stefan relative flux.
"""
import taichi as ti
from .lattice import add_compensated, cross
from .reduction import DeviceSum


@ti.data_oriented
class PhaseMomentum:
    def __init__(self, cfg, geo, rigid, coupling):
        self.cfg, self.geo, self.rigid, self.coupling = cfg, geo, rigid, coupling
        self.old_integrals = ti.Vector.field(4, ti.f64, (cfg.nb, cfg.nb))
        self.old_mass = ti.field(ti.f64, ())
        self.old_inertia = ti.field(ti.f64, ())
        self.old_first = ti.Vector.field(2, ti.f64, ())
        self.old_world = ti.Vector.field(3, ti.f64, ())
        self.twist = ti.Vector.field(3, ti.f64, ())
        self.target_world = ti.Vector.field(3, ti.f64, ())
        self.roundoff_total = ti.Vector.field(3, ti.f64, ())
        self.slip_sum = DeviceSum(cfg.nb*cfg.nb*12, 3, ti.f64)
        self.exchange_sum = DeviceSum(cfg.nb*cfg.nb*12, 3, ti.f64)

    @ti.kernel
    def capture(self):
        for a, b in self.old_integrals:
            self.old_integrals[a, b] = self.geo.integrals[a, b]
        self.old_mass[None], self.old_inertia[None] = self.geo.mass[None], self.geo.inertia[None]
        self.old_first[None] = self.geo.first_moment[None]
        self.old_world[None] = (ti.cast(self.rigid.momentum[None], ti.f64)
                               -ti.cast(self.rigid.momentum_comp[None], ti.f64))

    @ti.func
    def relative_velocity(self, a, b, k):
        dm = ti.cast(self.coupling.mass[a, b, k], ti.f64)
        length = self.geo.surface_length[a, b, k]
        return (dm/ti.max(length, 1e-30)*(1-1/ti.static(self.cfg.rho_s_lu))
                *ti.cast(self.coupling.normal[a, b, k], ti.f64))

    @ti.func
    def slip_value(self, index):
        a, b, k = index//(self.cfg.nb*12), (index//12) % self.cfg.nb, index % 12
        impulse = ti.cast(self.coupling.mass[a, b, k], ti.f64)*self.relative_velocity(a, b, k)
        point = ti.cast(self.coupling.point[a, b, k], ti.f64)
        return ti.Vector([impulse.x, impulse.y, cross(point, impulse)])

    @ti.func
    def resolve_twist(self, slip):
        twist = ti.Vector.zero(ti.f64, 3)
        if ti.static(not self.cfg.fixed_body):
            mass, inertia = self.old_mass[None], self.old_inertia[None]
            if mass > 0.0 and inertia > 0.0:
                # G_old * v = P_old - J_relative. The matrix is evaluated in
                # COM coordinates, avoiding inversion around a remote origin.
                arm = self.old_first[None]/mass
                center = self.geo.origin[None]+arm
                world = self.old_world[None]-slip
                linear = ti.Vector([world.x, world.y])
                omega = (world.z-cross(center, linear))/inertia
                velocity = linear/mass-ti.Vector([-omega*arm.y, omega*arm.x])
                twist = ti.Vector([velocity.x, velocity.y, omega])
            elif self.coupling.melt_step[None] > 0.0:
                self.coupling.failure[None] = 3
        self.twist[None] = twist
        local = self.geo.mass_matrix[None]@twist
        linear = ti.Vector([local.x, local.y])
        self.target_world[None] = ti.Vector([local.x, local.y,
                                            local.z+cross(self.geo.origin[None], linear)])

    @ti.kernel
    def rewrite_exchanges(self):
        for a, b in self.geo.integrals:
            delta = self.old_integrals[a, b]-self.geo.integrals[a, b]
            base = ti.Vector([a+.5-.5*self.cfg.nb, b+.5-.5*self.cfg.nb])
            first_local = ti.Vector([delta[1], delta[2]])
            density = ti.static(self.cfg.rho_s_lu)
            first = density*self.geo.rotation[None]@(delta[0]*base+first_local)
            polar = density*(delta[3]+2*base.dot(first_local)+delta[0]*base.dot(base))
            twist = self.twist[None]
            velocity = ti.Vector([twist.x, twist.y])
            material_p = density*delta[0]*velocity+twist.z*ti.Vector([-first.y, first.x])
            material_l = cross(first, velocity)+twist.z*polar
            total = ti.cast(0.0, ti.f64)
            for k in range(12): total += ti.cast(self.coupling.mass[a, b, k], ti.f64)
            for k in range(12):
                dm = ti.cast(self.coupling.mass[a, b, k], ti.f64)
                point = ti.cast(self.coupling.point[a, b, k], ti.f64)
                arm = point-self.geo.origin[None]
                vs = velocity+twist.z*ti.Vector([-arm.y, arm.x])
                relative = self.relative_velocity(a, b, k)
                self.coupling.material_velocity[a, b, k] = vs
                self.coupling.liquid_velocity[a, b, k] = vs+relative
                impulse, angular = ti.Vector.zero(ti.f64, 2), ti.cast(0.0, ti.f64)
                if total > 0.0:
                    share = dm/total
                    material = share*material_p
                    slip = dm*relative
                    impulse = material+slip
                    angular = (cross(self.geo.origin[None], material)+share*material_l
                               +cross(point, slip))
                stored = ti.cast(impulse, ti.f32)
                self.coupling.exchange[a, b, k][4] = stored.x
                self.coupling.exchange[a, b, k][5] = stored.y
                # Includes the removed region's intrinsic angular momentum,
                # as well as relocating its resultant to the interface point.
                self.coupling.torque_extra[a, b, k] = angular-cross(point, ti.cast(stored, ti.f64))

    @ti.func
    def exchange_value(self, index):
        a, b, k = index//(self.cfg.nb*12), (index//12) % self.cfg.nb, index % 12
        record = self.coupling.exchange[a, b, k]
        impulse = ti.Vector([ti.cast(record[4], ti.f64), ti.cast(record[5], ti.f64)])
        point = ti.cast(self.coupling.point[a, b, k], ti.f64)
        angular = cross(point, impulse)+self.coupling.torque_extra[a, b, k]
        return ti.Vector([impulse.x, impulse.y, angular])

    @ti.func
    def commit(self, outgoing):
        if self.coupling.melt_step[None] > 0.0:
            self.rigid.phase_impulse[None], self.rigid.phase_impulse_comp[None] = add_compensated(
                self.rigid.phase_impulse[None], self.rigid.phase_impulse_comp[None], outgoing)
            if ti.static(self.cfg.fixed_body):
                self.rigid.support_impulse[None], self.rigid.support_impulse_comp[None] = add_compensated(
                    self.rigid.support_impulse[None], self.rigid.support_impulse_comp[None], outgoing)
            else:
                target = self.target_world[None]
                self.roundoff_total[None] += outgoing-(self.old_world[None]-target)
                high = ti.cast(target, ti.f32)
                self.rigid.momentum[None] = high
                self.rigid.momentum_comp[None] = ti.cast(ti.cast(high, ti.f64)-target, ti.f32)

    def finish(self):
        self.slip_sum.run(self.slip_value, self.resolve_twist)
        self.rewrite_exchanges()
        self.exchange_sum.run(self.exchange_value, self.commit)
