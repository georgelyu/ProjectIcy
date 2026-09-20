"""Per-step device guards, retained across a host sampling batch."""
import taichi as ti


@ti.data_oriented
class Health:
    def __init__(self, cfg, geo, state, flow):
        self.cfg, self.geo, self.state, self.flow = cfg, geo, state, flow
        self.flags = ti.field(ti.i32, ())
        self.capacity_peak = ti.field(ti.f32, ())
        self.mach_peak = ti.field(ti.f32, ())

    @ti.kernel
    def reset(self):
        self.flags[None], self.capacity_peak[None], self.mach_peak[None] = 0, 0.0, 0.0

    @ti.kernel
    def observe(self):
        for i, j in self.state.q:
            q = self.state.value(i, j)
            capacity = self.geo.capacity[i, j]
            error = ti.abs(q[0]+q[1]-capacity)
            ti.atomic_max(self.capacity_peak[None], ti.cast(error, ti.f32))
            flags = 0
            if error > ti.static(self.cfg.capacity_tolerance): flags |= 1
            if q[0] < -1e-7 or q[1] < -1e-7: flags |= 2
            if (ti.math.isnan(q)).any() or (ti.math.isinf(q)).any(): flags |= 4
            if (ti.math.isnan(self.flow.velocity[i, j])).any() or (ti.math.isinf(self.flow.velocity[i, j])).any(): flags |= 4
            if capacity > 1e-8:
                phi = q[0]/capacity
                if phi < -1e-6 or phi > 1+1e-5: flags |= 8
                speed = self.flow.velocity[i, j].norm()*1.7320508075688772
                ti.atomic_max(self.mach_peak[None], speed)
                if speed > ti.static(self.cfg.max_mach): flags |= 16
            if flags: ti.atomic_or(self.flags[None], flags)
        # The fixed material origin can lie far outside a melted remnant.
        # Its translational speed and omega*INITIAL_size are then a very loose
        # bound: translation/rotation cancel at the surviving material. An
        # affine rigid velocity attains its norm maximum at a polygon vertex.
        for a, b in self.geo.count:
            for k in range(self.geo.count[a, b]):
                local = ti.Vector([a+.5-.5*self.cfg.nb+self.geo.poly[a, b][k, 0],
                                   b+.5-.5*self.cfg.nb+self.geo.poly[a, b][k, 1]])
                point = self.geo.origin[None]+self.geo.rotation[None]@local
                if self.geo.body_velocity(point).norm() > .2:
                    ti.atomic_or(self.flags[None], 32)
