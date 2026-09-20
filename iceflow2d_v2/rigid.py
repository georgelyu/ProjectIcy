"""New variable-mass rigid solver; fixed material origin, world angular momentum."""
import taichi as ti
from .lattice import add_compensated, cross
from .reduction import DeviceSum


@ti.data_oriented
class Rigid:
    def __init__(self, config, geometry):
        self.cfg, self.geo = config, geometry
        self.momentum = ti.Vector.field(3, ti.f32, ())
        self.momentum_comp = ti.Vector.field(3, ti.f32, ())
        self.start = ti.Vector.field(3, ti.f32, ())
        self.gravity_impulse = ti.Vector.field(3, ti.f32, ())
        self.gravity_impulse_comp = ti.Vector.field(3, ti.f32, ())
        self.contact_impulse = ti.Vector.field(3, ti.f32, ())
        self.contact_impulse_comp = ti.Vector.field(3, ti.f32, ())
        self.support_impulse = ti.Vector.field(3, ti.f32, ())
        self.support_impulse_comp = ti.Vector.field(3, ti.f32, ())
        self.phase_impulse = ti.Vector.field(3, ti.f32, ())
        self.phase_impulse_comp = ti.Vector.field(3, ti.f32, ())
        self.iteration_residual = ti.field(ti.f32, ())
        self.phase_sum = DeviceSum(config.nb*config.nb*12, 3)

    @ti.func
    def local_momentum(self, world):
        return ti.Vector([world.x, world.y, world.z-cross(self.geo.origin[None], ti.Vector([world.x, world.y]))])

    @ti.func
    def world_impulse(self, local):
        return ti.Vector([local.x, local.y, local.z+cross(self.geo.origin[None], ti.Vector([local.x, local.y]))])

    @ti.func
    def gravity(self):
        force = -self.geo.mass[None]*ti.static(self.cfg.gravity_lu)
        return ti.Vector([0.0, force, -self.geo.first_moment[None].x*ti.static(self.cfg.gravity_lu)])

    @ti.kernel
    def begin(self):
        # The compensation does not change during the candidate iterations.
        # Keep the high part here and combine both parts in f64 below.
        self.start[None] = self.momentum[None]

    @ti.kernel
    def iterate(self, flow: ti.template()):
        if ti.static(not self.cfg.fixed_body):
            if self.geo.mass[None] > 0.0 and self.geo.inertia[None] > 0.0:
                matrix = self.geo.mass_matrix[None]
                start = ti.cast(self.start[None], ti.f64)-ti.cast(self.momentum_comp[None], ti.f64)
                residual = self.local_momentum(start)+flow.body_force[None]+self.gravity()-matrix@self.geo.velocity[None]
                self.geo.velocity[None] += (matrix+flow.body_drag[None]).inverse()@residual
                self.iteration_residual[None] = residual.norm()

    @ti.kernel
    def commit_force(self, flow: ti.template()):
        gravity = self.world_impulse(self.gravity())
        delta = self.world_impulse(flow.body_force[None])+gravity
        self.gravity_impulse[None], self.gravity_impulse_comp[None] = add_compensated(self.gravity_impulse[None], self.gravity_impulse_comp[None], gravity)
        if ti.static(self.cfg.fixed_body):
            self.support_impulse[None], self.support_impulse_comp[None] = add_compensated(self.support_impulse[None], self.support_impulse_comp[None], -delta)
        else:
            self.momentum[None], self.momentum_comp[None] = add_compensated(self.momentum[None], self.momentum_comp[None], delta)

    @ti.func
    def phase_value(self, index):
        a, b, k = index//(self.cfg.nb*12), (index//12) % self.cfg.nb, index % 12
        p = ti.Vector([self.phase_records[a, b, k][4], self.phase_records[a, b, k][5]])
        return ti.Vector([p.x, p.y, cross(self.phase_points[a, b, k], p)])

    @ti.func
    def phase_result(self, value):
        self.phase_impulse[None], self.phase_impulse_comp[None] = add_compensated(self.phase_impulse[None], self.phase_impulse_comp[None], value)
        if ti.static(not self.cfg.fixed_body):
            self.momentum[None], self.momentum_comp[None] = add_compensated(self.momentum[None], self.momentum_comp[None], -value)
        else:
            self.support_impulse[None], self.support_impulse_comp[None] = add_compensated(self.support_impulse[None], self.support_impulse_comp[None], value)

    def remove_phase_momentum(self, records, points):
        self.phase_records, self.phase_points = records, points
        self.phase_sum.run(self.phase_value, self.phase_result)

    @ti.kernel
    def recover_velocity(self):
        if ti.static(not self.cfg.fixed_body):
            m, inertia = self.geo.mass[None], self.geo.inertia[None]
            if m > 0.0 and inertia > 0.0:
                world = ti.cast(self.momentum[None], ti.f64)-ti.cast(self.momentum_comp[None], ti.f64)
                p = ti.Vector([world.x, world.y])
                r = self.geo.first_moment[None]/m
                com = self.geo.origin[None]+r
                omega = (world.z-cross(com, p))/inertia
                v = p/m-ti.Vector([-omega*r.y, omega*r.x])
                self.geo.velocity[None] = ti.Vector([v.x, v.y, omega])
            else:
                self.geo.velocity[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def advance_pose(self):
        if ti.static(not self.cfg.fixed_body):
            mass, inertia = self.geo.mass[None], self.geo.inertia[None]
            if mass > 0.0 and inertia > 0.0:
                world = ti.cast(self.momentum[None], ti.f64)-ti.cast(self.momentum_comp[None], ti.f64)
                linear = ti.Vector([world.x, world.y])
                arm = self.geo.first_moment[None]/mass
                center = self.geo.origin[None]+arm
                angle = (world.z-cross(center, linear))/inertia
                cosine, sine = ti.cos(angle), ti.sin(angle)
                # Translate the centre of mass, then rotate the fixed material
                # origin around it. Euler-stepping the origin changes the COM
                # by O(omega^2*arm), which creates large spurious spin when the
                # surviving remnant has a tiny moment of inertia.
                rotated = ti.Vector([cosine*arm.x-sine*arm.y, sine*arm.x+cosine*arm.y])
                delta = linear/mass+arm-rotated
                self.geo.origin[None], self.geo.origin_comp[None] = add_compensated(self.geo.origin[None], self.geo.origin_comp[None], delta, ti.f64)
                self.geo.angle[None], self.geo.angle_comp[None] = add_compensated(self.geo.angle[None], self.geo.angle_comp[None], angle, ti.f64)

    @ti.kernel
    def contact_side(self, side: ti.template()):
        # Frictionless perfectly inelastic normal impulse at each extreme vertex.
        # The position correction and impulse are explicit diagnostics.
        if ti.static(not self.cfg.fixed_body):
            origin = self.geo.origin[None]
            radius = 0.707107*self.cfg.nb
            near = (origin.x-radius < self.cfg.wall_cells or origin.y-radius < self.cfg.wall_cells
                    or origin.x+radius > self.cfg.nx-self.cfg.wall_cells
                    or origin.y+radius > self.cfg.ny-self.cfg.wall_cells)
            axis, sign = ti.static(side//2), 1.0
            if ti.static(side % 2 == 1): sign = -1.0
            extent = ti.cast(-1e20, ti.f32)
            for a, b in self.geo.count:
                if near:
                    for k in range(self.geo.count[a, b]):
                        local = ti.Vector([a+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 0], b+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 1]])
                        p = self.geo.origin[None]+self.geo.rotation[None]@local
                        ti.atomic_max(extent, ti.cast(sign*p[axis], ti.f32))
            # Reproduce the former serial f64 > f32 comparison, including its
            # rounding ties: the last p above rounded extent wins; otherwise
            # the first p rounding to that extent wins. This keeps the contact
            # point/impulse unchanged, not just the mathematical maximum.
            winner = -1
            for a, b in self.geo.count:
                if near:
                    for k in range(self.geo.count[a, b]):
                        local = ti.Vector([a+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 0], b+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 1]])
                        p = self.geo.origin[None]+self.geo.rotation[None]@local
                        if ti.cast(sign*p[axis], ti.f32) == extent:
                            index = (a*self.cfg.nb+b)*12+k
                            key = self.cfg.nb*self.cfg.nb*12-1-index
                            if sign*p[axis] > extent: key = self.cfg.nb*self.cfg.nb*12+index
                            ti.atomic_max(winner, key)
            limit = -ti.cast(self.cfg.wall_cells, ti.f32)
            if ti.static(side % 2 == 0):
                limit = ti.cast(self.cfg.nx-self.cfg.wall_cells, ti.f32)
                if ti.static(side//2 == 1): limit = ti.cast(self.cfg.ny-self.cfg.wall_cells, ti.f32)
            if winner >= 0 and extent > limit and self.geo.mass[None] > 0.0:
                index = self.cfg.nb*self.cfg.nb*12-1-winner
                if winner >= self.cfg.nb*self.cfg.nb*12:
                    index = winner-self.cfg.nb*self.cfg.nb*12
                a, b, k = index//(self.cfg.nb*12), (index//12) % self.cfg.nb, index % 12
                local = ti.Vector([a+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 0], b+0.5-0.5*self.cfg.nb+self.geo.poly[a, b][k, 1]])
                point = ti.cast(self.geo.origin[None]+self.geo.rotation[None]@local, ti.f32)
                normal = ti.Vector([0.0, 0.0]); normal[axis] = -sign
                self.geo.origin[None] += normal*(extent-limit)
                speed = self.geo.body_velocity(point).dot(normal)
                if speed < 0.0:
                    com = self.geo.origin[None]+self.geo.first_moment[None]/self.geo.mass[None]
                    arm = cross(point-com, normal)
                    effective = 1/self.geo.mass[None]+arm*arm/ti.max(self.geo.inertia[None], 1e-30)
                    impulse = -speed/effective*normal
                    delta = ti.Vector([impulse.x, impulse.y, cross(point, impulse)])
                    self.momentum[None], self.momentum_comp[None] = add_compensated(self.momentum[None], self.momentum_comp[None], delta)
                    self.contact_impulse[None], self.contact_impulse_comp[None] = add_compensated(self.contact_impulse[None], self.contact_impulse_comp[None], delta)

    def contact(self):
        # Preserve the sequential side order and original impulse law. Each
        # side's extremum uses the pose resulting from the previous side.
        for side in range(4):
            self.contact_side(side)
