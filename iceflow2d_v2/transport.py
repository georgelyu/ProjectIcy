"""Connected graph volume projection and conservative implicit remapping.

A single oriented edge owns every transfer. GPU PCG and implicit upwind use
only device state; sampling is the only host synchronization point.
"""
import taichi as ti
from .lattice import direction, weight, opposite, add_compensated
from .connectivity import Components
from .reduction import DeviceSum

LINK_Q = (1, 2, 5, 6)


@ti.data_oriented
class Transport:
    def __init__(self, config, geometry, state, flow):
        self.cfg, self.geo, self.state, self.flow = config, geometry, state, flow
        shape = (config.nx, config.ny)
        self.k = ti.Vector.field(4, ti.f32, shape)
        self.pred = ti.Vector.field(4, ti.f32, shape)
        self.flux = ti.Vector.field(4, ti.f32, shape)
        self.phase = ti.Vector.field(4, ti.f32, shape)
        self.phase_flux = ti.Matrix.field(4, 6, ti.f32, shape)
        self.limits = ti.Vector.field(2, ti.f32, shape)
        self.limited = ti.field(ti.f32, shape)
        # The cut-aperture linear solve needs f64 after reproduced f32 loss of
        # conjugacy and recursive/actual residual disagreement. This precision
        # change is confined to projection; coefficients, transported state,
        # multigrid, LB and thermal fields remain f32.
        self.diag = ti.field(ti.f64, shape)
        self.x = ti.field(ti.f64, shape)
        self.r = ti.field(ti.f64, shape)
        self.z = ti.field(ti.f64, shape)
        self.d = ti.field(ti.f64, shape)
        self.ad = ti.field(ti.f64, shape)
        self.rhs = ti.Vector.field(6, ti.f32, shape)
        self.c = ti.Vector.field(6, ti.f32, shape)
        self.next_c = ti.Vector.field(6, ti.f32, shape)
        self.transfer = ti.Matrix.field(4, 6, ti.f32, shape)
        self.gcl = ti.field(ti.f32, shape)
        self.capacity_residual = ti.field(ti.f32, shape)
        self.dot_parts = ti.Vector.field(2, ti.f32, (config.nx,))
        self.rr = ti.field(ti.f64, ())
        self.old_rr = ti.field(ti.f64, ())
        self.dad = ti.field(ti.f64, ())
        self.initial_rr = ti.field(ti.f64, ())
        self.iterations_used = ti.field(ti.i32, ())
        self.compatibility = ti.field(ti.f64, ())
        self.component = ti.field(ti.i32, shape)
        self.component_next = ti.field(ti.i32, shape)
        self.component_sum = ti.field(ti.f64, (config.nx*config.ny+1,))
        self.component_count = ti.field(ti.i32, (config.nx*config.ny+1,))
        self.components = Components(config.nx, config.ny)
        self.null_sum = DeviceSum(config.nx*config.ny, 2, ti.f64)
        self.dot_sum = DeviceSum(config.nx*config.ny, 2, ti.f64)
        self.multigrid = None
        self.polish_override = None
        if config.projection_solver == 'multigrid':
            from .multigrid import Multigrid
            self.multigrid = Multigrid(config.nx, config.ny, config.multigrid_cycles)

    @ti.func
    def adjacent(self, i, j, k):
        q = 1
        if k == 1: q = 2
        elif k == 2: q = 5
        elif k == 3: q = 6
        return direction(q)

    @ti.func
    def open_face(self, i, j, axis):
        return ti.max(self.geo.face_open[i, j][axis], self.geo.old_face_open[i, j][axis])

    @ti.func
    def aperture(self, i, j, ni, nj):
        return self.geo.swept_aperture(i, j, ni, nj)

    @ti.kernel
    def build_predictor(self):
        for i, j in self.k:
            for k in ti.static(range(4)):
                q = ti.static(LINK_Q[k])
                e = direction(q)
                ni, nj = i+e.x, j+e.y
                aperture = self.aperture(i, j, ni, nj)
                total, phase = 0.0, 0.0
                if aperture > 0.0:
                    if ti.static(self.cfg.advection):
                        if self.geo.fluid(i, j) and self.geo.fluid(ni, nj):
                            total = 3*weight(q)*aperture*ti.cast(e, ti.f32).dot(self.flow.velocity[i, j]+self.flow.velocity[ni, nj])
                            left = self.state.value(i, j)[0]/ti.max(self.geo.old_capacity[i, j], 1e-30)
                            right = self.state.value(ni, nj)[0]/ti.max(self.geo.old_capacity[ni, nj], 1e-30)
                            upwind = left
                            if total < 0: upwind = right
                            phase = aperture*(self.flow.hpost[q, i, j]-self.flow.hpost[opposite(q), ni, nj])-total*upwind
                self.k[i, j][k] = 3*weight(q)*aperture
                self.pred[i, j][k], self.phase[i, j][k] = total, phase

    @ti.kernel
    def phase_limits(self):
        for i, j in self.limits:
            out_water, out_air = 0.0, 0.0
            for k in range(4):
                outgoing = self.phase[i, j][k]
                out_water += ti.max(0.0, outgoing)
                out_air += ti.max(0.0, -outgoing)
                e = self.adjacent(i, j, k)
                ni, nj = i-e.x, j-e.y
                if self.geo.inside(ni, nj):
                    incoming = self.phase[ni, nj][k]
                    out_water += ti.max(0.0, -incoming)
                    out_air += ti.max(0.0, incoming)
            value = self.state.value(i, j)
            self.limits[i, j] = ti.Vector([ti.min(1.0, ti.max(0.0, value[0])/ti.max(out_water, 1e-30)),
                                          ti.min(1.0, ti.max(0.0, value[1])/ti.max(out_air, 1e-30))])

    @ti.kernel
    def phase_transfers(self):
        for i, j in self.phase_flux:
            result, limited = ti.Matrix.zero(ti.f32, 4, 6), 0.0
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                raw = self.phase[i, j][k]
                if self.geo.inside(ni, nj) and raw != 0.0:
                    factor = ti.min(self.limits[i, j].x, self.limits[ni, nj].y)
                    di, dj, ai, aj = i, j, ni, nj
                    if raw < 0:
                        factor = ti.min(self.limits[ni, nj].x, self.limits[i, j].y)
                        di, dj, ai, aj = ni, nj, i, j
                    amount = raw*factor
                    limited += ti.abs(raw-amount)
                    uw, ua = self.flow.velocity[di, dj], self.flow.velocity[ai, aj]
                    p = amount*(uw-ti.static(self.cfg.rho_a_lu)*ua)
                    vector = ti.Vector([amount, -amount, amount*self.state.temperature_w[di, dj],
                                        -amount*ti.static(self.cfg.air_capacity_lu)*(self.state.temperature_a[ai, aj]-ti.static(self.cfg.air_temperature/self.cfg.temperature_scale)), p.x, p.y])
                    for m in ti.static(range(6)): result[k, m] = vector[m]
            self.phase_flux[i, j], self.limited[i, j] = result, limited

    @ti.kernel
    def commit_phase(self):
        for i, j in self.state.q:
            delta = ti.Vector.zero(ti.f32, 6)
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i-e.x, j-e.y
                for m in ti.static(range(6)):
                    delta[m] -= self.phase_flux[i, j][k, m]
                    if self.geo.inside(ni, nj): delta[m] += self.phase_flux[ni, nj][k, m]
            self.state.add(i, j, delta)

    @ti.kernel
    def initialize_projection(self):
        for i, j in self.r:
            volume_source = ti.cast(self.state.source[i, j][0], ti.f64)+ti.cast(self.state.source[i, j][1], ti.f64)
            residual = self.state.volume(i, j)+volume_source-self.geo.capacity[i, j]
            diagonal = ti.cast(0.0, ti.f64)
            for k in range(4):
                residual -= self.pred[i, j][k]
                diagonal += self.k[i, j][k]
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                if self.geo.inside(ni, nj):
                    residual -= self.k[i, j][k]*(self.x[i, j]-self.x[ni, nj])
                ni, nj = i-e.x, j-e.y
                if self.geo.inside(ni, nj):
                    residual += self.pred[ni, nj][k]
                    diagonal += self.k[ni, nj][k]
                    residual -= self.k[ni, nj][k]*(self.x[i, j]-self.x[ni, nj])
            # Keep the previous potential as a warm start. This is especially
            # useful for the slowly moving geometry, without changing sources.
            self.diag[i, j], self.r[i, j] = diagonal, residual
            self.z[i, j] = 0.0
            if diagonal > 1e-15: self.z[i, j] = residual/diagonal
            self.d[i, j] = self.z[i, j]

    @ti.kernel
    def seed_components(self):
        for i, j in self.component:
            label = -1
            if self.diag[i, j] > 1e-15:
                label = 1+i*self.cfg.ny+j
                # For this single body (smaller than the box), the exterior of
                # its enclosing circle is one connected fluid component.
                relative = ti.Vector([i+.5, j+.5])-self.geo.origin[None]
                if relative.norm() > 0.707107*self.cfg.nb+2: label = 0
            self.component[i, j] = label
        for k in self.component_sum:
            self.component_sum[k], self.component_count[k] = 0.0, 0

    @ti.kernel
    def propagate_components(self):
        for i, j in self.component:
            label = self.component[i, j]
            if label > 0:
                for k in range(4):
                    e = self.adjacent(i, j, k)
                    ni, nj = i+e.x, j+e.y
                    if self.geo.inside(ni, nj) and self.k[i, j][k] > 0.0 and self.component[ni, nj] >= 0:
                        label = ti.min(label, self.component[ni, nj])
                    ni, nj = i-e.x, j-e.y
                    if self.geo.inside(ni, nj) and self.k[ni, nj][k] > 0.0 and self.component[ni, nj] >= 0:
                        label = ti.min(label, self.component[ni, nj])
            self.component_next[i, j] = label

    @ti.kernel
    def accept_components(self):
        for i, j in self.component: self.component[i, j] = self.component_next[i, j]

    @ti.kernel
    def reduce_small_components(self):
        for i, j in self.component:
            label = self.component[i, j]
            if label > 0:
                ti.atomic_add(self.component_sum[label], self.r[i, j])
                ti.atomic_add(self.component_count[label], 1)

    @ti.func
    def null_value(self, index):
        i, j = index//self.cfg.ny, index % self.cfg.ny
        value = ti.Vector.zero(ti.f64, 2)
        if self.component[i, j] == 0: value = ti.Vector([self.r[i, j], 1.0])
        return value

    @ti.func
    def null_result(self, value):
        self.compatibility[None] = value[0]/ti.max(value[1], 1.0)

    def nullspace_component(self):
        self.null_sum.run(self.null_value, self.null_result)

    @ti.kernel
    def project_nullspace(self):
        # Project only the linear solve onto the Neumann range. The actual
        # source/state is untouched; the discarded component remains in GCL.
        for i, j in self.r:
            if self.diag[i, j] > 1e-15:
                mean = self.compatibility[None]
                label = self.component[i, j]
                if label > 0:
                    mean = self.component_sum[label]/ti.max(self.component_count[label], 1)
                self.r[i, j] -= mean
                self.z[i, j] = self.r[i, j]/self.diag[i, j]
                self.d[i, j] = self.z[i, j]

    @ti.kernel
    def matrix_product(self):
        for i, j in self.ad:
            result = ti.cast(0.0, ti.f64)
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                if self.geo.inside(ni, nj): result += self.k[i, j][k]*(self.d[i, j]-self.d[ni, nj])
                ni, nj = i-e.x, j-e.y
                if self.geo.inside(ni, nj): result += self.k[ni, nj][k]*(self.d[i, j]-self.d[ni, nj])
            self.ad[i, j] = result

    @ti.func
    def dot_value(self, index):
        i, j = index//self.cfg.ny, index % self.cfg.ny
        return ti.Vector([self.r[i, j]*self.z[i, j], self.d[i, j]*self.ad[i, j]])

    @ti.func
    def dot_result(self, value):
        self.rr[None], self.dad[None] = value.x, value.y

    @ti.kernel
    def collect_cg_products(self):
        # Both terms are nonnegative. Parallel f64 reduction avoids the long
        # serial compensated tail without reducing the precision of the solve.
        # d^T A d = sum_edges k_ij (d_i-d_j)^2 also avoids subtracting large
        # signed products when the potential has an arbitrary Neumann gauge.
        for part in self.dot_sum.first:
            value = ti.Vector.zero(ti.f64, 2)
            for offset in range(32):
                index = part+offset*self.dot_sum.first.shape[0]
                if index < self.cfg.nx*self.cfg.ny:
                    i, j = index//self.cfg.ny, index % self.cfg.ny
                    value.x += self.r[i, j]*self.z[i, j]
                    for k in range(4):
                        e = self.adjacent(i, j, k)
                        ni, nj = i+e.x, j+e.y
                        if self.geo.inside(ni, nj):
                            difference = self.d[i, j]-self.d[ni, nj]
                            value.y += self.k[i, j][k]*difference*difference
            self.dot_sum.first[part] = value

    @ti.kernel
    def group_cg_products(self):
        for part in self.dot_sum.second:
            value = ti.Vector.zero(ti.f64, 2)
            for offset in range(32):
                index = part+offset*self.dot_sum.second.shape[0]
                if index < self.dot_sum.first.shape[0]: value += self.dot_sum.first[index]
            self.dot_sum.second[part] = value

    @ti.kernel
    def finish_cg_products(self):
        self.rr[None], self.dad[None] = 0.0, 0.0
        for part in self.dot_sum.second:
            self.rr[None] += self.dot_sum.second[part].x
            self.dad[None] += self.dot_sum.second[part].y

    def dot_columns(self):
        self.collect_cg_products(); self.group_cg_products()

    def dot_finish(self):
        self.finish_cg_products()

    @ti.kernel
    def start_cg(self):
        self.initial_rr[None] = self.rr[None]
        self.iterations_used[None] = 0

    @ti.kernel
    def accept_multigrid(self, fine: ti.template()):
        for i, j in self.x:
            self.x[i, j] += fine.x[i, j]

    @ti.kernel
    def cg_update(self):
        for i, j in self.x:
            # Capacity is an absolute geometric constraint. Tiny apertures can
            # make the initial preconditioned norm enormous; a relative stop
            # then accepts visibly unclosed volume (reproduced in Stefan).
            if self.rr[None] > 1e-14 and self.dad[None] > 1e-25:
                if i == 0 and j == 0: self.iterations_used[None] += 1
                alpha = self.rr[None]/self.dad[None]
                self.x[i, j] += alpha*self.d[i, j]
                self.r[i, j] -= alpha*self.ad[i, j]
                self.z[i, j] = 0.0
                if self.diag[i, j] > 1e-15: self.z[i, j] = self.r[i, j]/self.diag[i, j]

    @ti.kernel
    def remember_rr(self):
        self.old_rr[None] = self.rr[None]

    @ti.kernel
    def cg_direction(self):
        for i, j in self.d:
            value = ti.cast(0.0, ti.f64)
            if self.rr[None] > 1e-14:
                beta = self.rr[None]/ti.max(self.old_rr[None], 1e-30)
                value = self.z[i, j]+beta*self.d[i, j]
            self.d[i, j] = value

    @ti.kernel
    def volume_fluxes(self):
        for i, j in self.flux:
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                value = self.pred[i, j][k]
                if self.geo.inside(ni, nj): value += self.k[i, j][k]*(self.x[i, j]-self.x[ni, nj])
                self.flux[i, j][k] = value

    @ti.kernel
    def initialize_transport(self):
        for i, j in self.c:
            rhs = self.state.value(i, j)+self.state.source[i, j]
            out, net = 0.0, 0.0
            for k in range(4):
                value = self.flux[i, j][k]
                out += ti.max(0.0, value); net += value
                e = self.adjacent(i, j, k)
                ni, nj = i-e.x, j-e.y
                if self.geo.inside(ni, nj):
                    value = self.flux[ni, nj][k]
                    out += ti.max(0.0, -value); net -= value
            volume = self.geo.capacity[i, j]
            self.diag[i, j] = volume+out
            self.rhs[i, j] = rhs
            self.c[i, j] = rhs/ti.max(rhs[0]+rhs[1], 1e-20)
            self.gcl[i, j] = self.geo.old_capacity[i, j]+self.state.source[i, j][0]+self.state.source[i, j][1]-volume-net

    @ti.kernel
    def mix(self):
        for i, j in self.c:
            rhs = self.rhs[i, j]
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                if self.geo.inside(ni, nj): rhs += ti.max(0.0, -self.flux[i, j][k])*self.c[ni, nj]
                ni, nj = i-e.x, j-e.y
                if self.geo.inside(ni, nj): rhs += ti.max(0.0, self.flux[ni, nj][k])*self.c[ni, nj]
            self.next_c[i, j] = rhs/ti.max(self.diag[i, j], 1e-20)

    @ti.kernel
    def accept_mix(self):
        for i, j in self.c: self.c[i, j] = self.next_c[i, j]

    @ti.kernel
    def form_transfers(self):
        for i, j in self.transfer:
            result = ti.Matrix.zero(ti.f32, 4, 6)
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i+e.x, j+e.y
                value = self.flux[i, j][k]
                donor = self.c[i, j]
                if value < 0.0 and self.geo.inside(ni, nj): donor = self.c[ni, nj]
                for m in ti.static(range(6)): result[k, m] = value*donor[m]
            self.transfer[i, j] = result

    @ti.kernel
    def commit_transport(self):
        for i, j in self.state.q:
            delta = self.state.source[i, j]
            for k in range(4):
                e = self.adjacent(i, j, k)
                ni, nj = i-e.x, j-e.y
                for m in ti.static(range(6)):
                    delta[m] -= self.transfer[i, j][k, m]
                    if self.geo.inside(ni, nj): delta[m] += self.transfer[ni, nj][k, m]
            self.state.add(i, j, delta)
            value = self.state.value(i, j)
            self.capacity_residual[i, j] = value[0]+value[1]-self.geo.capacity[i, j]

    def phase_step(self):
        self.build_predictor()
        self.phase_limits()
        self.phase_transfers()
        self.commit_phase()

    def remap(self):
        self.build_predictor()
        self.initialize_projection()
        self.seed_components()
        self.components.seed(self.diag, 1e-15)
        self.components.connect(self.k, True)
        self.components.labels(self.component)
        self.reduce_small_components()
        self.nullspace_component(); self.project_nullspace()
        iterations = self.cfg.pressure_iterations
        if self.multigrid is not None:
            self.multigrid.solve(self.k, self.r)
            self.accept_multigrid(self.multigrid.levels[0])
            # Recompute the true fine residual in projection precision; the
            # f32 multigrid residual is not the residual of the accepted x.
            self.initialize_projection()
            self.seed_components(); self.components.labels(self.component)
            self.reduce_small_components()
            self.nullspace_component(); self.project_nullspace()
            iterations = self.cfg.pressure_polish_iterations
        if self.polish_override is not None: iterations = self.polish_override
        self.dot_columns(); self.dot_finish(); self.start_cg()
        for _ in range(iterations):
            self.matrix_product()
            self.dot_columns(); self.dot_finish()
            self.cg_update(); self.remember_rr()
            self.dot_columns(); self.dot_finish(); self.cg_direction()
        self.volume_fluxes()
        self.initialize_transport()
        for _ in range(self.cfg.transport_iterations):
            self.mix(); self.accept_mix()
        self.form_transfers()
        self.commit_transport()
