"""Whitelist D2Q9 predictor; additional conservative coupling is external.

No imports from iceflow2d. The unsourced raw/central moment collisions, force
stencils, push streaming and local SNIBB formula retain the approved rules.
"""
import taichi as ti
from .reduction import DeviceSum
from .lattice import (direction, weight, opposite, pressure_equilibrium,
                      phase_equilibrium, basis_moments, reconstruct,
                      momentum_collision, phase_collision, snibb,
                      volume_source_population, add_compensated, cross)


@ti.data_oriented
class Flow:
    def __init__(self, config, geometry):
        self.cfg, self.geo = config, geometry
        self.nx, self.ny = config.nx, config.ny
        shape = (self.nx, self.ny)
        # q-major SoA; the post-collision buffers never alias streaming targets.
        self.f = ti.field(ti.f32, (9, *shape))
        self.h = ti.field(ti.f32, (9, *shape))
        self.fpost = ti.field(ti.f32, (9, *shape))
        self.hpost = ti.field(ti.f32, (9, *shape))
        self.prepared = ti.field(ti.f32, (9, *shape))
        self.phi = ti.field(ti.f32, shape)
        self.raw_velocity = ti.Vector.field(2, ti.f32, shape)
        self.velocity = ti.Vector.field(2, ti.f32, shape)
        self.force = ti.Vector.field(2, ti.f32, shape)
        self.capillary_acceleration = ti.Vector.field(2, ti.f32, shape)
        self.capillary_stress = ti.Vector.field(3, ti.f32, shape)
        self.temperature = ti.field(ti.f32, shape)
        self.pressure = ti.field(ti.f32, shape)
        self.reference_pressure = ti.field(ti.f32, shape)
        self.reference_density = ti.field(ti.f32, shape)
        self.dilatation = ti.field(ti.f32, shape)
        self.tau_shear = ti.field(ti.f32, shape)
        self.tau_physical = ti.field(ti.f32, shape)
        self.boundary_phase = ti.field(ti.f32, shape)
        self.boundary_slip = ti.Vector.field(2, ti.f32, shape)
        self.boundary_impulse = ti.Vector.field(3, ti.f32, shape)
        self.boundary_reference_impulse = ti.Vector.field(3, ti.f32, shape)
        self.boundary_drag = ti.Matrix.field(3, 3, ti.f32, shape)
        self.wall_impulse = ti.Vector.field(3, ti.f32, shape)
        self.wall_reference_impulse = ti.Vector.field(3, ti.f32, shape)
        self.body_force = ti.Vector.field(3, ti.f32, ())
        self.body_drag = ti.Matrix.field(3, 3, ti.f32, ())
        self.boundary_sum = DeviceSum(self.nx*self.ny, 12)
        self.initialize()

    @ti.func
    def rho(self, phi):
        bounded = ti.min(1.0, ti.max(0.0, phi))
        return ti.static(self.cfg.rho_a_lu)+(1.0-ti.static(self.cfg.rho_a_lu))*bounded

    @ti.func
    def capillary_population(self, q, i, j):
        # Hermite second-moment representation of minus the Korteweg stress.
        # Pair forces are parallel to their links, preserving angular momentum.
        e, s = ti.cast(direction(q), ti.f32), self.capillary_stress[i, j]
        return 4.5*weight(q)*((e.x*e.x-1/3)*s.x+2*e.x*e.y*s.y+(e.y*e.y-1/3)*s.z)

    @ti.func
    def neighbor(self, i, j, ex, ey, distance):
        value = self.phi[i, j]
        if self.geo.fluid(i+ex, j+ey):
            value = self.phi[i+ex, j+ey]
            if distance == 2:
                value = self.phi[i, j]
                if self.geo.fluid(i+2*ex, j+2*ey):
                    value = self.phi[i+2*ex, j+2*ey]
        return value

    @ti.kernel
    def initialize(self):
        for i, j in self.phi:
            phi = 0.5*(1.0-ti.tanh(2.0*(j+0.5-ti.static(self.cfg.water_level/self.cfg.dx))/ti.static(self.cfg.interface_width)))
            if self.geo.wall(i, j): phi = 0.0
            self.phi[i, j] = phi
            temperature = ti.static(self.cfg.water_temperature/self.cfg.temperature_scale)
            self.temperature[i, j] = temperature
            # Analytic, piecewise-linear hydrostatic reference. Its derivative
            # is cellwise constant since the initial level is a grid face.
            reference_phi = ti.min(1.0, ti.max(0.0, ti.static(self.cfg.water_level/self.cfg.dx)-j))
            self.reference_density[i, j] = self.rho(reference_phi)
            depth = ti.max(0.0, ti.static(self.ny-self.cfg.wall_cells)-j-0.5)
            water_depth = ti.max(0.0, ti.static(self.cfg.water_level/self.cfg.dx)-j-0.5)
            self.reference_pressure[i, j] = ti.static(self.cfg.gravity_lu)*(ti.static(self.cfg.rho_a_lu)*depth+(1.0-ti.static(self.cfg.rho_a_lu))*water_depth)
            self.pressure[i, j] = 0.0  # relative to the frozen hydrostatic reference
            for q in range(9):
                self.f[q, i, j] = 0.0
                self.h[q, i, j] = phase_equilibrium(q, phi, ti.Vector([0.0, 0.0]))

    @ti.kernel
    def freeze_reference_density(self):
        # Initialize the dynamic part of hydrostatic pressure from the smooth
        # initial phase profile. The sharp reference itself stays unchanged.
        for i, j in self.reference_density:
            level = ti.static(self.cfg.water_level/self.cfg.dx)
            width = ti.static(self.cfg.interface_width)
            y = j+0.5
            # Softplus remainder of the tanh integral relative to the sharp
            # profile, evaluated without cancellation of two large pressures.
            relative = (ti.static(self.cfg.gravity_lu*(1-self.cfg.rho_a_lu))*0.25*width
                        *ti.log(1+ti.exp(-ti.abs(4*(y-level)/width))))
            rho = self.rho(self.phi[i, j])
            force = ti.Vector([0.0, -(rho-self.reference_density[i, j])*ti.static(self.cfg.gravity_lu)/rho])
            self.pressure[i, j], self.force[i, j] = relative, force
            self.raw_velocity[i, j] = -0.5*force
            for q in range(9):
                self.f[q, i, j] = pressure_equilibrium(q, relative, rho, ti.Vector([0.0, 0.0]))-1.5*weight(q)*rho*ti.cast(direction(q), ti.f32).dot(force)

    @ti.kernel
    def warmup_collision(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                phi, grad = self.phi[i, j], ti.Vector.zero(ti.f32, 2)
                for q in range(9):
                    e = direction(q)
                    p1, p2 = self.neighbor(i, j, e.x, e.y, 1), self.neighbor(i, j, e.x, e.y, 2)
                    grad += weight(q)*ti.cast(e, ti.f32)*(2*(p1-phi)+.5*(4*p1-p2-3*phi))
                normal = grad/(grad.norm()+1e-14)
                compression = 4*phi*(1-phi)/ti.static(self.cfg.interface_width)
                raw, eq, source = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
                for q in range(9):
                    e = ti.cast(direction(q), ti.f32)
                    basis = basis_moments(e.x, e.y)
                    raw += basis*self.h[q, i, j]
                    eq += basis*weight(q)*phi
                    source += basis*weight(q)*compression*e.dot(normal)
                post = phase_collision(raw, eq, source, ti.static(1/(3*self.cfg.mobility+.5)))
                for q in range(9): self.hpost[q, i, j] = reconstruct(q, ti.Vector([0.0, 0.0]), post)

    @ti.kernel
    def warmup_stream(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                for q in range(9):
                    e = direction(q)
                    if self.geo.fluid(i+e.x, j+e.y): self.h[q, i+e.x, j+e.y] = self.hpost[q, i, j]
                    else: self.h[opposite(q), i, j] = self.hpost[q, i, j]

    @ti.kernel
    def collide(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                phi = self.phi[i, j]
                bounded = ti.min(1.0, ti.max(0.0, phi))
                rho = self.rho(phi)
                grad, source_grad = ti.Vector.zero(ti.f32, 2), ti.Vector.zero(ti.f32, 2)
                laplace, near_solid = 0.0, 0.0
                du = ti.Matrix.zero(ti.f32, 2, 2)
                for q in range(9):
                    e = direction(q)
                    ef = ti.cast(e, ti.f32)
                    p1, p2 = self.neighbor(i, j, e.x, e.y, 1), self.neighbor(i, j, e.x, e.y, 2)
                    source_grad += 3.0*weight(q)*ef*(p1-phi)
                    grad += weight(q)*ef*(2.0*(p1-phi)+0.5*(4*p1-p2-3*phi))
                    laplace += 6.0*weight(q)*(p1-phi)
                    neighbor_u = ti.Vector([0.0, 0.0])
                    ni, nj = i+e.x, j+e.y
                    if self.geo.fluid(ni, nj):
                        neighbor_u = self.raw_velocity[ni, nj]
                    elif self.geo.inside(ni, nj) and self.geo.mask[ni, nj] == 1:
                        neighbor_u = self.geo.body_velocity(ti.Vector([ni+0.5, nj+0.5]))
                        if q > 0: near_solid = 1.0
                    du += 3.0*weight(q)*(neighbor_u-self.raw_velocity[i, j]).outer_product(ef)
                nu_art = ti.static(self.cfg.smagorinsky**2)*ti.sqrt(2*(du[0, 0]**2+du[1, 1]**2+0.5*(du[0, 1]+du[1, 0])**2))
                nu = ti.static(self.cfg.dt/self.cfg.dx**2)*(bounded*ti.static(self.cfg.nu_w)+(1-bounded)*ti.static(self.cfg.nu_a))+nu_art
                tau = 0.5+3*nu
                shear_tau = ti.max(tau, 0.5+ti.static(self.cfg.shear_tau_floor-0.5)*ti.max(1-bounded, near_solid))
                self.tau_physical[i, j], self.tau_shear[i, j] = tau, shear_tau
                gravity = ti.Vector([0.0, -ti.static(self.cfg.gravity_lu)])
                force_density = (rho-self.reference_density[i, j])*gravity
                if ti.static(self.cfg.buoyancy):
                    anomaly = -ti.static(self.cfg.beta)*(self.temperature[i, j]*ti.static(self.cfg.temperature_scale)-ti.static(self.cfg.water_temperature))
                    force_density += ti.min(1.0, ti.max(0.0, 2*bounded-1))*anomaly*gravity
                sigma = ti.static(self.cfg.sigma*self.cfg.dt**2/(self.cfg.rho_w*self.cfg.dx**3))
                chemical = 48*sigma/ti.static(self.cfg.interface_width)*bounded*(bounded-1)*(bounded-0.5)-1.5*sigma*ti.static(self.cfg.interface_width)*laplace
                self.capillary_acceleration[i, j] = chemical*grad/rho
                kappa = 1.5*sigma*ti.static(self.cfg.interface_width)
                potential = 12*sigma/ti.static(self.cfg.interface_width)*bounded**2*(1-bounded)**2
                isotropic = potential+.5*kappa*grad.norm_sqr()
                self.capillary_stress[i, j] = ti.Vector([kappa*grad.x**2-isotropic,
                                                        kappa*grad.x*grad.y, kappa*grad.y**2-isotropic])
                force_density += chemical*grad
                force = force_density/rho
                self.force[i, j] = force
                u = self.raw_velocity[i, j]+0.5*force
                self.velocity[i, j] = u
                raw, eq, src = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
                hr, he, hs = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
                normal = grad/(grad.norm()+1e-14)
                compression = 4*phi*(1-phi)/ti.static(self.cfg.interface_width)
                for q in range(9):
                    e = ti.cast(direction(q), ti.f32)
                    basis = basis_moments(e.x, e.y)
                    central = basis_moments(e.x-u.x, e.y-u.y)
                    source = 3*weight(q)*(e.dot(force_density)+(1-ti.static(self.cfg.rho_a_lu))*e.dot(u)*e.dot(source_grad))
                    source += volume_source_population(q, rho, self.dilatation[i, j])
                    raw += basis*self.f[q, i, j]
                    eq += basis*pressure_equilibrium(q, self.pressure[i, j], rho, u)
                    src += basis*source
                    hr += central*self.h[q, i, j]
                    he += central*phase_equilibrium(q, phi, u)
                    hs += central*weight(q)*compression*e.dot(normal)
                fm = momentum_collision(raw, eq, src, tau, shear_tau)
                hm = phase_collision(hr, he, hs, ti.static(1/(3*self.cfg.mobility+0.5)))
                for q in range(9):
                    self.fpost[q, i, j] = reconstruct(q, ti.Vector([0.0, 0.0]), fm)
                    self.hpost[q, i, j] = reconstruct(q, u, hm)

    @ti.kernel
    def prepare_links(self):
        for i, j in self.phi:
            impulse, wall_impulse = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
            reference_impulse, wall_reference = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
            drag = ti.Matrix.zero(ti.f32, 3, 3)
            phase_delta = 0.0
            if self.geo.fluid(i, j):
                raw = ti.Vector.zero(ti.f32, 9)
                for q in range(9): raw[q] = self.fpost[q, i, j]
                rho, u = self.rho(self.phi[i, j]), self.velocity[i, j]
                for q in range(9):
                    value = raw[q]
                    e = direction(q)
                    ef = ti.cast(e, ti.f32)
                    ni, nj = i+e.x, j+e.y
                    if q > 0 and self.geo.inside(ni, nj) and self.geo.mask[ni, nj] == 1:
                        fraction = self.geo.cut_fraction(i, j, ni, nj)
                        point = ti.Vector([i+0.5, j+0.5])+fraction*ef
                        vb = self.geo.body_velocity(point)+self.boundary_slip[i, j]
                        value = snibb(q, fraction, raw, self.f[q, i, j], self.pressure[i, j], rho, u, vb)
                        exchange = (ef-vb)*raw[q]-(-ef-vb)*value
                        exchange += 2*self.capillary_population(q, i, j)*ef
                        relative = point-self.geo.origin[None]
                        impulse += ti.Vector([exchange.x, exchange.y, cross(relative, exchange)])
                        jac = ti.Vector([ef.x, ef.y, cross(relative, ef)])
                        drag += 3*weight(q)*rho/(1+fraction)*jac.outer_product(jac)
                        phase_delta -= 6*weight(q)*self.phi[i, j]*ef.dot(vb)
                    elif q > 0 and not self.geo.fluid(ni, nj):
                        exchange = 2*ef*(raw[q]+self.capillary_population(q, i, j))
                        point = ti.Vector([i+0.5, j+0.5])+0.5*ef
                        wall_impulse += ti.Vector([exchange.x, exchange.y, cross(point, exchange)])
                    self.prepared[q, i, j] = value
            if not self.geo.wall(i, j):
                # Divergence theorem for the fixed reference pressure, using
                # the very same solid/fluid cell volumes as the gravity split.
                buoyancy = self.reference_density[i, j]*self.geo.occupancy[i, j]*ti.static(self.cfg.gravity_lu)
                reference_impulse = ti.Vector([0.0, buoyancy, (i+0.5-self.geo.origin[None].x)*buoyancy])
                impulse += reference_impulse
                # Cartesian wall-face integrals: p at the boundary, not p at
                # the neighbouring cell centre. Corner faces are distinct.
                pref, grad = self.reference_pressure[i, j], self.reference_density[i, j]*ti.static(self.cfg.gravity_lu)
                if i == self.cfg.wall_cells:
                    wall_reference += ti.Vector([-pref, 0.0, (j+.5)*pref])
                if i == self.nx-self.cfg.wall_cells-1:
                    wall_reference += ti.Vector([pref, 0.0, -(j+.5)*pref])
                if j == self.cfg.wall_cells:
                    wall_reference += ti.Vector([0.0, -(pref+.5*grad), -(i+.5)*(pref+.5*grad)])
                if j == self.ny-self.cfg.wall_cells-1:
                    wall_reference += ti.Vector([0.0, pref-.5*grad, (i+.5)*(pref-.5*grad)])
                wall_impulse += wall_reference
            self.boundary_impulse[i, j], self.boundary_drag[i, j] = impulse, drag
            self.boundary_reference_impulse[i, j] = reference_impulse
            self.wall_impulse[i, j], self.boundary_phase[i, j] = wall_impulse, phase_delta
            self.wall_reference_impulse[i, j] = wall_reference

    @ti.kernel
    def correct_boundary_pressure(self):
        # Coupling correction, not a change to the SNIBB population formula.
        # Internal stress uses open-link areas. The boundary pressure traction
        # must close those same areas, or a constant pressure accelerates a cut
        # cell and creates an acoustic mode independent of the time step.
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                normal_defect = ti.Vector([0.0, 0.0])
                capillary_defect = ti.Vector([0.0, 0.0])
                center = ti.Vector([i+.5, j+.5])
                for q in range(1, 9):
                    e = direction(q); ni, nj = i+e.x, j+e.y
                    ef = ti.cast(e, ti.f32)
                    if self.geo.fluid(ni, nj):
                        missing = 1-self.geo.swept_aperture(i, j, ni, nj)
                        normal_defect += 6*weight(q)*missing*ef
                        capillary_defect += 2*self.capillary_population(q, i, j)*missing*ef
                force = self.pressure[i, j]*normal_defect+capillary_defect
                # These normal tractions act at link midpoints; e x e = 0,
                # hence their exact moment is also obtained at the cell centre.
                self.boundary_impulse[i, j] += ti.Vector([force.x, force.y, cross(center-self.geo.origin[None], force)])

    def prepare_boundary(self):
        self.prepare_links()
        self.correct_boundary_pressure()

    @ti.func
    def boundary_value(self, index):
        i, j = index//self.ny, index % self.ny
        result = ti.Vector.zero(ti.f32, 12)
        for k in ti.static(range(3)): result[k] = self.boundary_impulse[i, j][k]
        for k in ti.static(range(9)): result[k+3] = self.boundary_drag[i, j][k//3, k%3]
        return result

    @ti.func
    def boundary_result(self, value):
        for k in ti.static(range(3)): self.body_force[None][k] = value[k]
        for k in ti.static(range(9)): self.body_drag[None][k//3, k%3] = value[k+3]

    def reduce_boundary(self):
        self.boundary_sum.run(self.boundary_value, self.boundary_result)

    @ti.kernel
    def stream(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                for q in range(9):
                    e = direction(q)
                    ni, nj = i+e.x, j+e.y
                    outgoing_h = self.hpost[q, i, j]
                    if self.geo.fluid(ni, nj):
                        self.f[q, ni, nj], self.h[q, ni, nj] = self.prepared[q, i, j], outgoing_h
                    else:
                        if self.geo.inside(ni, nj) and self.geo.mask[ni, nj] == 1:
                            fraction = self.geo.cut_fraction(i, j, ni, nj)
                            vb = self.geo.body_velocity(ti.Vector([i+0.5, j+0.5])+fraction*ti.cast(e, ti.f32))+self.boundary_slip[i, j]
                            outgoing_h -= 6*weight(q)*self.phi[i, j]*ti.cast(e, ti.f32).dot(vb)
                        self.f[opposite(q), i, j], self.h[opposite(q), i, j] = self.prepared[q, i, j], outgoing_h

    @ti.kernel
    def recover_moments(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                phi, momentum = 0.0, ti.Vector.zero(ti.f32, 2)
                for q in range(9):
                    phi += self.h[q, i, j]
                    momentum += ti.cast(direction(q), ti.f32)*self.f[q, i, j]
                self.phi[i, j] = phi
                self.raw_velocity[i, j] = momentum/self.rho(phi)
                self.velocity[i, j] = self.raw_velocity[i, j]+0.5*self.force[i, j]

    @ti.kernel
    def recover_pressure(self):
        for i, j in self.phi:
            if self.geo.fluid(i, j):
                moving, grad = 0.0, ti.Vector.zero(ti.f32, 2)
                for q in range(9):
                    e = direction(q)
                    if q > 0: moving += self.f[q, i, j]
                    grad += 3*weight(q)*ti.cast(e, ti.f32)*(self.neighbor(i, j, e.x, e.y, 1)-self.phi[i, j])
                rho, u = self.rho(self.phi[i, j]), self.velocity[i, j]
                self.pressure[i, j] = 0.6*(moving+0.5*(1-ti.static(self.cfg.rho_a_lu))*u.dot(grad)
                                         +0.5*rho*self.dilatation[i, j]-(2/3)*rho*u.dot(u))

    def predict(self):
        self.collide()
        self.prepare_boundary()
        self.reduce_boundary()
        self.stream()
        self.recover_moments()
        self.recover_pressure()
