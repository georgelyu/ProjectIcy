"""Device-side external impulse accounting, separate from thermal enthalpy."""
import taichi as ti
from .lattice import add_compensated, cross
from .reduction import DeviceSum


@ti.data_oriented
class Budgets:
    def __init__(self, cfg, geo, state, flow, rigid, coupling):
        self.cfg, self.geo, self.state = cfg, geo, state
        self.flow, self.rigid, self.coupling = flow, rigid, coupling
        self.external = ti.Vector.field(3, ti.f32, ())
        self.external_comp = ti.Vector.field(3, ti.f32, ())
        self.gravity = ti.Vector.field(3, ti.f32, ())
        self.gravity_comp = ti.Vector.field(3, ti.f32, ())
        self.wall = ti.Vector.field(3, ti.f32, ())
        self.wall_comp = ti.Vector.field(3, ti.f32, ())
        self.buoyancy = ti.Vector.field(3, ti.f32, ())
        self.buoyancy_comp = ti.Vector.field(3, ti.f32, ())
        self.numerical_air = ti.Vector.field(3, ti.f32, ())
        self.numerical_air_comp = ti.Vector.field(3, ti.f32, ())
        self.forces = ti.Vector.field(9, ti.f32, ())
        self.force_sum = DeviceSum(cfg.nx*cfg.ny, 9)
        self.gravity_work = ti.field(ti.f32, ())
        self.gravity_work_comp = ti.field(ti.f32, ())
        self.air_sum = DeviceSum(cfg.nx*cfg.ny, 3)

    @ti.func
    def force_value(self, index):
        i, j = index//self.cfg.ny, index % self.cfg.ny
        q = self.state.value(i, j)
        gravity = -(q[0]+ti.static(self.cfg.rho_a_lu)*q[1])*ti.static(self.cfg.gravity_lu)
        phi = q[0]/ti.max(self.geo.capacity[i, j], 1e-20)
        anomaly = 0.0
        if ti.static(self.cfg.buoyancy):
            anomaly = ti.min(1.0, ti.max(0.0, 2*phi-1))*ti.static(self.cfg.beta)*(self.state.temperature_w[i, j]*ti.static(self.cfg.temperature_scale)-ti.static(self.cfg.water_temperature))*ti.static(self.cfg.gravity_lu)*self.geo.capacity[i, j]
        wall = self.flow.wall_impulse[i, j]
        return ti.Vector([0.0, gravity, (i+.5)*gravity, 0.0, anomaly, (i+.5)*anomaly, -wall.x, -wall.y, -wall.z])

    @ti.func
    def force_result(self, value):
        self.forces[None] = value

    def gather_forces(self):
        self.force_sum.run(self.force_value, self.force_result)

    @ti.kernel
    def accumulate(self):
        total = self.forces[None]
        gravity = ti.Vector([total[0], total[1], total[2]])
        gravity.y -= self.geo.mass[None]*ti.static(self.cfg.gravity_lu)
        gravity.z -= (self.geo.origin[None].x*self.geo.mass[None]+self.geo.first_moment[None].x)*ti.static(self.cfg.gravity_lu)
        buoyancy = ti.Vector([total[3], total[4], total[5]])
        wall = ti.Vector([total[6], total[7], total[8]])
        self.gravity[None], self.gravity_comp[None] = add_compensated(self.gravity[None], self.gravity_comp[None], gravity)
        self.buoyancy[None], self.buoyancy_comp[None] = add_compensated(self.buoyancy[None], self.buoyancy_comp[None], buoyancy)
        self.wall[None], self.wall_comp[None] = add_compensated(self.wall[None], self.wall_comp[None], wall)
        self.external[None], self.external_comp[None] = add_compensated(self.external[None], self.external_comp[None], gravity+buoyancy+wall)

    @ti.func
    def air_value(self, index):
        i, j = index//self.cfg.ny, index % self.cfg.ny
        mass = self.state.source[i, j][1]*ti.static(self.cfg.rho_a_lu)
        momentum = mass*self.flow.velocity[i, j]
        return ti.Vector([momentum.x, momentum.y, cross(ti.Vector([i+0.5, j+0.5]), momentum)])

    @ti.func
    def air_result(self, value):
        self.numerical_air[None], self.numerical_air_comp[None] = add_compensated(self.numerical_air[None], self.numerical_air_comp[None], value)
        self.external[None], self.external_comp[None] = add_compensated(self.external[None], self.external_comp[None], value)

    def accumulate_numerical_air(self):
        self.air_sum.run(self.air_value, self.air_result)
