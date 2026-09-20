"""Implicit, angular-conservative viscosity when no SNIBB body links remain.

An unresolved remnant can lie wholly between lattice nodes. Its real boundary
then needs a viscous coupling even though the nodal bounce-back set is empty.
Integrate mu*ds/d over the polygon surfaces and solve the body together with
the rigid modes of the existing connected fluid patches. Interpolation and
spreading use the same mass inner product, so this operator dissipates kinetic
energy and conserves total linear/angular momentum without a mass cutoff.
"""
import taichi as ti
from .lattice import add_compensated, cross
from .reduction import DeviceSum


@ti.data_oriented
class SubgridViscous:
    def __init__(self, cfg, geo, state, flow, rigid, patches):
        self.cfg, self.geo, self.state, self.flow, self.rigid = cfg, geo, state, flow, rigid
        # Keep only field aliases: snapshot traversal must not rename the
        # existing agglomeration fields when this component is added.
        self.patch_root = patches.root
        shape = (cfg.nx, cfg.ny)
        self.drag = ti.Matrix.field(3, 3, ti.f64, shape)
        self.moments = ti.Vector.field(7, ti.f64, shape)
        self.mobility = ti.Matrix.field(3, 3, ti.f64, shape)
        self.effective = ti.Matrix.field(3, 3, ti.f64, shape)
        self.patch_velocity = ti.Vector.field(3, ti.f64, shape)
        self.patch_change = ti.Vector.field(3, ti.f64, shape)
        self.dissipation = ti.field(ti.f64, shape)
        self.enabled = ti.field(ti.i32, ())
        self.failure = ti.field(ti.i32, ())
        self.center = ti.Vector.field(2, ti.f64, ())
        self.body_velocity = ti.Vector.field(3, ti.f64, ())
        self.initial_velocity = ti.Vector.field(3, ti.f64, ())
        self.body_momentum = ti.Vector.field(3, ti.f64, ())
        self.support_change = ti.Vector.field(3, ti.f64, ())
        self.body_metric = ti.Matrix.field(3, 3, ti.f64, ())
        self.active_steps = ti.field(ti.i64, ())
        self.dissipated_total = ti.field(ti.f64, ())
        self.impulse_total = ti.Vector.field(3, ti.f64, ())
        self.effective_sum = DeviceSum(cfg.nx*cfg.ny, 12, ti.f64)
        self.energy_sum = DeviceSum(cfg.nx*cfg.ny, 1, ti.f64)

    @ti.kernel
    def reset(self):
        mass, inertia = self.geo.mass[None], self.geo.inertia[None]
        self.enabled[None] = ti.cast(mass > 0.0 and inertia > 0.0 and self.flow.body_drag[None].trace() == 0.0, ti.i32)
        self.failure[None] = 0
        if self.enabled[None]:
            center = self.geo.origin[None]+self.geo.first_moment[None]/mass
            p = ti.cast(self.rigid.momentum[None], ti.f64)-ti.cast(self.rigid.momentum_comp[None], ti.f64)
            p.z -= cross(center, ti.Vector([p.x, p.y]))
            metric = ti.Matrix([[mass, 0.0, 0.0], [0.0, mass, 0.0], [0.0, 0.0, inertia]])
            self.center[None], self.body_metric[None], self.body_momentum[None] = center, metric, p
            v = ti.Vector([p.x/mass, p.y/mass, p.z/inertia])
            if ti.static(self.cfg.fixed_body): v = ti.Vector.zero(ti.f64, 3)
            self.initial_velocity[None], self.body_velocity[None] = v, v
            self.active_steps[None] += 1
        for i, j in self.drag:
            self.drag[i, j] = ti.Matrix.zero(ti.f64, 3, 3)
            self.moments[i, j] = ti.Vector.zero(ti.f64, 7)
            self.effective[i, j] = ti.Matrix.zero(ti.f64, 3, 3)
            self.patch_change[i, j] = ti.Vector.zero(ti.f64, 3)
            self.patch_velocity[i, j] = ti.Vector.zero(ti.f64, 3)
            self.dissipation[i, j] = 0.0

    @ti.kernel
    def integrate_drag(self):
        for a, b, k in self.geo.surface_length:
            length = self.geo.surface_length[a, b, k]
            if self.enabled[None] and length > 0.0:
                point = self.geo.surface_point[a, b, k]
                normal = self.geo.surface_normal[a, b, k]
                epsilon = ti.cast(32*2.220446049250313e-16, ti.f64)*ti.max(1.0, ti.abs(point).max())
                cell = ti.cast(ti.floor(point+epsilon*normal), ti.i32)
                if self.geo.inside(cell.x, cell.y) and not self.geo.wall(cell.x, cell.y):
                    root = self.patch_root[cell.x, cell.y]
                    if root >= 0 and self.geo.capacity[cell.x, cell.y] > 0.0:
                        q = self.state.value(cell.x, cell.y)
                        rho = (q[0]+ti.static(self.cfg.rho_a_lu)*q[1])/self.geo.capacity[cell.x, cell.y]
                        nu = (self.flow.tau_shear[cell.x, cell.y]-.5)/3
                        distance = ti.max(.25, ti.abs((self.state.heat_position[cell.x, cell.y]-point).dot(normal)))
                        coefficient = ti.max(0.0, rho*nu)*length/distance
                        r = point-self.center[None]
                        # Exact line integral of J^T J, including each edge's
                        # intrinsic second moment, not a point-mass quadrature.
                        d = coefficient*ti.Matrix([[1.0, 0.0, -r.y], [0.0, 1.0, r.x],
                            [-r.y, r.x, r.norm_sqr()+length*length/12]])
                        x, y = root//self.cfg.ny, root%self.cfg.ny
                        for u, v in ti.static(ti.ndrange(3, 3)):
                            ti.atomic_add(self.drag[x, y][u, v], d[u, v])
                    else: ti.atomic_or(self.failure[None], 1)

    @ti.kernel
    def gather_fluid(self):
        for i, j in self.patch_root:
            root = self.patch_root[i, j]
            if self.enabled[None] and root >= 0:
                x, y = root//self.cfg.ny, root%self.cfg.ny
                if self.drag[x, y].trace() > 0.0:
                    q = ti.cast(self.state.q[i, j], ti.f64)-ti.cast(self.state.comp[i, j], ti.f64)
                    mass = q[0]+ti.static(self.cfg.rho_a_lu)*q[1]
                    r = ti.Vector([i+.5, j+.5])-self.center[None]
                    value = ti.Vector([mass, mass*r.x, mass*r.y, mass*r.norm_sqr(),
                                       q[4], q[5], r.x*q[5]-r.y*q[4]])
                    for k in ti.static(range(7)): ti.atomic_add(self.moments[x, y][k], value[k])

    @ti.kernel
    def eliminate_fluid(self):
        for i, j in self.drag:
            d, m = self.drag[i, j], self.moments[i, j]
            if self.enabled[None] and d.trace() > 0.0:
                metric = ti.Matrix([[m[0], 0.0, -m[2]], [0.0, m[0], m[1]], [-m[2], m[1], m[3]]])
                if m[0] > 0.0 and metric.determinant() > 0.0:
                    inverse = metric.inverse()
                    effective = (ti.Matrix.identity(ti.f64, 3)+d@inverse).inverse()@d
                    self.mobility[i, j] = inverse
                    self.effective[i, j] = .5*(effective+effective.transpose())
                    self.patch_velocity[i, j] = inverse@ti.Vector([m[4], m[5], m[6]])
                else: ti.atomic_or(self.failure[None], 2)

    @ti.func
    def effective_value(self, index):
        i, j = index//self.cfg.ny, index%self.cfg.ny
        value = ti.Vector.zero(ti.f64, 12)
        if self.enabled[None]:
            d = self.effective[i, j]
            rhs = d@self.patch_velocity[i, j]
            for k in ti.static(range(9)): value[k] = d[k//3, k%3]
            for k in ti.static(range(3)): value[k+9] = rhs[k]
        return value

    @ti.func
    def solve_body(self, value):
        if self.enabled[None] and self.failure[None] == 0:
            d = ti.Matrix.zero(ti.f64, 3, 3)
            for k in ti.static(range(9)): d[k//3, k%3] = value[k]
            rhs = ti.Vector([value[9], value[10], value[11]])
            self.support_change[None] = -rhs
            if ti.static(not self.cfg.fixed_body):
                self.body_velocity[None] = (self.body_metric[None]+d).inverse()@(self.body_momentum[None]+rhs)
            else: self.body_velocity[None] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def patch_impulses(self):
        for i, j in self.drag:
            if self.enabled[None] and self.failure[None] == 0 and self.drag[i, j].trace() > 0.0:
                before = self.patch_velocity[i, j]
                impulse = self.effective[i, j]@(before-self.body_velocity[None])
                change = -self.mobility[i, j]@impulse
                self.patch_change[i, j] = change
                # Fluid kinetic change; body contribution is added once below.
                self.dissipation[i, j] = impulse.dot(before+.5*change)

    @ti.kernel
    def commit_fluid(self):
        for i, j in self.patch_root:
            root = self.patch_root[i, j]
            if self.enabled[None] and self.failure[None] == 0 and root >= 0:
                x, y = root//self.cfg.ny, root%self.cfg.ny
                if self.drag[x, y].trace() > 0.0:
                    q = ti.cast(self.state.q[i, j], ti.f64)-ti.cast(self.state.comp[i, j], ti.f64)
                    mass = q[0]+ti.static(self.cfg.rho_a_lu)*q[1]
                    r, v = ti.Vector([i+.5, j+.5])-self.center[None], self.patch_change[x, y]
                    delta = mass*ti.Vector([v.x-v.z*r.y, v.y+v.z*r.x])
                    self.state.add(i, j, ti.Vector([0.0, 0.0, 0.0, 0.0, delta.x, delta.y]))

    @ti.func
    def energy_value(self, index):
        return ti.Vector([self.dissipation[index//self.cfg.ny, index%self.cfg.ny]])

    @ti.func
    def commit_body(self, energy):
        if self.enabled[None] and self.failure[None] == 0:
            before, after = self.initial_velocity[None], self.body_velocity[None]
            local = self.body_metric[None]@(after-before)
            if ti.static(self.cfg.fixed_body):
                # Sum fluid reaction for the fixed support, using the same
                # eliminated system; the free-body update is its counterpart.
                local = self.support_change[None]
            delta = ti.Vector([local.x, local.y, local.z+cross(self.center[None], ti.Vector([local.x, local.y]))])
            if ti.static(self.cfg.fixed_body):
                self.rigid.support_impulse[None], self.rigid.support_impulse_comp[None] = add_compensated(
                    self.rigid.support_impulse[None], self.rigid.support_impulse_comp[None], delta)
            else:
                self.rigid.momentum[None], self.rigid.momentum_comp[None] = add_compensated(
                    self.rigid.momentum[None], self.rigid.momentum_comp[None], delta)
            self.impulse_total[None] += delta
            self.dissipated_total[None] += energy[0]-.5*local.dot(after+before)

    @ti.kernel
    def observe(self, health: ti.template()):
        if self.failure[None]: ti.atomic_or(health.flags[None], 128)

    def step(self):
        self.reset(); self.integrate_drag(); self.gather_fluid(); self.eliminate_fluid()
        self.effective_sum.run(self.effective_value, self.solve_body)
        self.patch_impulses(); self.commit_fluid()
        self.energy_sum.run(self.energy_value, self.commit_body)
        self.rigid.recover_velocity()
