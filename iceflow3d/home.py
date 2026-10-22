"""Velocity-based HOME (Li, Wu & Desbrun 2024, equations 25--31).

Two ten-scalar moment buffers replace the two D3Q27 momentum distributions.
The D3Q27 stencil is retained for its isotropy; populations exist only in
registers. S stores the symmetric second Hermite moment, NOT a velocity
gradient. Collision relaxes its central deviatoric and trace parts (CM-MRT).
There is no finite-difference stress replacement or HRR collision.
"""

# ruff: noqa: E741
import taichi as ti

from .lattice import Q, REST, density, direction, mixture_kinematic_viscosity, weight


@ti.func
def pack(p, u, S):
    return ti.Vector(
        [p, u.x, u.y, u.z, S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[0, 2], S[1, 2]]
    )


@ti.func
def unpack(m):
    u = ti.Vector([m[1], m[2], m[3]])
    S = ti.Matrix([[m[4], m[7], m[8]], [m[7], m[5], m[9]], [m[8], m[9], m[6]]])
    return m[0], u, S


@ti.func
def reconstruct(q, m):
    """Third-order filtered reconstruction, paper (28), cs^2 = 1/3."""
    p, u, S = unpack(m)
    c = ti.cast(direction(q), ti.f32)
    value = p + 3 * c.dot(u) + 4.5 * (c.dot(S @ c) - S.trace() / 3)
    third = 0.0
    for a, b in ti.static(ti.ndrange(3, 3)):
        if ti.static(a != b):
            third += (
                (c[a] * c[a] - 1 / 3)
                * c[b]
                * (S[a, a] * u[b] + 2 * S[a, b] * u[a] - 2 * u[a] ** 2 * u[b])
            )
    # xyz has six distinct permutations in H3:A3 (xxy has three).
    # Its coefficient is 27, matching Home_ThreePhase's Axyz_t27.
    third += (
        2
        * c.x
        * c.y
        * c.z
        * (S[0, 2] * u.y + S[1, 2] * u.x + S[0, 1] * u.z - 2 * u.x * u.y * u.z)
    )
    return weight(q) * (value + 13.5 * third)


@ti.func
def equilibrium(q, p, u):
    return reconstruct(q, pack(p, u, u.outer_product(u)))


@ti.func
def central_collision(m, acceleration, rate, bulk):
    p, j, S = unpack(m)
    # The stored pre-collision first moment excludes the half force.
    u = j + 0.5 * acceleration
    uu = u.outer_product(u)
    residual = S - uu
    forcing = acceleration.outer_product(u) + u.outer_product(acceleration)
    eye = ti.Matrix.identity(ti.f32, 3)
    post = (
        uu
        + (1 - rate) * (residual - residual.trace() / 3 * eye)
        + (1 - bulk) * residual.trace() / 3 * eye
        + (1 - 0.5 * rate) * (forcing - forcing.trace() / 3 * eye)
        + (1 - 0.5 * bulk) * forcing.trace() / 3 * eye
    )
    # bulk=1 gives exactly paper (29)--(31). Retain the configured bulk rate.
    return pack(p, j + acceleration, post)


@ti.data_oriented
class HomeMomentum:
    @ti.func
    def _home_neighbor(self, I, c):
        return I + c

    @ti.func
    def _set_momentum_equilibrium(self, I, phi, p, u):
        a = self.home_acceleration[I]
        self.home[I] = pack(
            3 * p / density(phi, self.rho_air),
            u - 0.5 * a,
            u.outer_product(u) - 0.5 * (a.outer_product(u) + u.outer_product(a)),
        )
        self.velocity_lattice[I] = u

    @ti.func
    def _set_equilibrium(self, I, phi, p, u):
        self._set_momentum_equilibrium(I, phi, p, u)
        self._set_phase_equilibrium(I, phi, u)

    @ti.func
    def _kinematic_viscosity(self, I):
        return (
            mixture_kinematic_viscosity(
                self.water_phase[I],
                self._water_temperature(I),
                self.config.rho_water,
                self.config.rho_air,
                self.mu_air,
            )
            * self.scales.dt_s
            / self.scales.dx_m**2
        )

    @ti.func
    def _force_link_fluid(self, J, interior: ti.template(), tank: ti.template()):
        valid = True
        if ti.static(not interior):
            if ti.static(tank):
                b = ti.static(self.config.boundary_cells)
                valid = (J >= b).all() and (J < ti.Vector(self.shape) - b).all()
            else:
                valid = self._fluid(J)
        return valid

    @ti.func
    def _home_acceleration_at(
        self,
        I,
        force,
        interior: ti.template() = False,
        moment_pressure: ti.template() = False,
        tank: ti.template() = False,
    ):
        rho = density(self.water_phase[I], self.rho_air)
        a = force / rho
        # Discrete product-rule correction: grad(p/rho) - grad(p)/rho.
        # This includes the cs^2 factor implied by p*=p/(rho*cs^2), and
        # cancels the normalized-pressure gradient for constant physical p.
        # Using only -p*grad(rho)/rho^2 leaves a large truncation residual
        # across a water/air interface (density ratio 800).
        for q in ti.static(range(Q)):
            if ti.static(q != REST):
                c = direction(q)
                cf = ti.cast(c, ti.f32)
                J = I + c
                if self._force_link_fluid(J, interior, tank):
                    rj = density(self.water_phase[J], self.rho_air)
                    if ti.static(moment_pressure):
                        # p_j = rho_j * HOME_pstar_j / 3. Reuse the stored
                        # normalized pressure instead of dividing 26 times.
                        # rho-rho_j makes the pure-phase correction exactly 0.
                        a += weight(q) * cf * self.home[J][0] * ((rho - rj) / rho)
                    else:
                        # Explicit macro fallback for initialization/refill
                        # and diagnostic callers that edit physical pressure.
                        pj = self.pressure_lattice[J]
                        a += 3 * weight(q) * cf * pj * (1 / rj - 1 / rho)
        # Variable-density viscosity force, paper (23). This derivative is
        # ONLY a body force; the collision shear remains the kinetic CM mode.
        grad_rho = (1 - self.rho_air) * self.phase_gradient_lattice[I]
        if grad_rho.dot(grad_rho) > 0:
            # Home_ThreePhase uses a six-face centered velocity gradient in
            # this force. Preserve our bounded moving-wall extrapolation.
            # This is second order and linear-exact, and does not replace
            # the central-moment collision stress (no HRR).
            gradient_u = ti.Matrix.zero(ti.f32, 3, 3)
            for axis in ti.static(range(3)):
                c = ti.Vector.zero(ti.i32, 3)
                c[axis] = 1
                derivative = ti.Vector.zero(ti.f32, 3)
                if ti.static(interior):
                    derivative = 0.5 * (
                        self.velocity_lattice[I + c] - self.velocity_lattice[I - c]
                    )
                elif ti.static(tank):
                    left, right = -self.velocity_lattice[I], -self.velocity_lattice[I]
                    if self._force_link_fluid(I - c, False, True):
                        left = self.velocity_lattice[I - c]
                    if self._force_link_fluid(I + c, False, True):
                        right = self.velocity_lattice[I + c]
                    derivative = 0.5 * (right - left)
                else:
                    derivative = 0.5 * (
                        self._stress_neighbor_velocity(I, c)
                        - self._stress_neighbor_velocity(I, -c)
                    )
                for component in ti.static(range(3)):
                    gradient_u[component, axis] = derivative[component]
            a += (
                self._kinematic_viscosity(I)
                / rho
                * (gradient_u + gradient_u.transpose())
                @ grad_rho
            )
        return a

    @ti.kernel
    def _collide_momentum(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                self.old_velocity[I] = self.velocity_lattice[I]
                self.old_pressure[I] = self.pressure_lattice[I]
                rate = 1 / (0.5 + 3 * self._kinematic_viscosity(I))
                self.home_post[I] = central_collision(
                    self.home[I],
                    self.home_acceleration[I],
                    rate,
                    self.config.cm_acoustic_relaxation_rate,
                )

    @ti.func
    def _home_reflection(self, I, outgoing):
        c = direction(outgoing)
        cf = ti.cast(c, ti.f32)
        J = I + c
        raw = reconstruct(outgoing, self.home_post[I])
        reflected = raw  # stationary tank: halfway bounce-back
        if self._inside(J) and self.solid_body_id[J] >= 0:
            b = self.solid_body_id[J]
            eta = self._boundary_link_fraction(I, J)
            point = ti.cast(I, ti.f32) + 0.5 + eta * cf
            ub = self._body_velocity_at(point, b)
            p = self.home[I][0]
            neq = reconstruct(outgoing, self.home[I]) - equilibrium(
                outgoing, p, self.old_velocity[I]
            )
            reflected = (
                equilibrium(26 - outgoing, p, ub)
                + neq
                + eta * reconstruct(26 - outgoing, self.home_post[I])
            ) / (1 + eta)
            # g carries velocity, so convert its exchange back to momentum.
            # No rho*cs^2 background is present in this pressure formulation.
            rho = density(self.water_phase[I], self.rho_air)
            impulse = rho * ((cf - ub) * raw - (-cf - ub) * reflected)
            pref = (1 - eta) * self.reference_pressure[
                I.y
            ] + eta * self.reference_pressure[J.y]
            impulse += 6 * weight(outgoing) * pref * cf
            capillary = (
                -3
                * weight(outgoing)
                * (self.capillary_stress[I] + self._capillary_neighbor(I, c))
                @ cf
            )
            ti.atomic_add(self.hydrodynamic_force[b], impulse + capillary)
            torque = (point - self.body_center[b]).cross(impulse)
            torque += (ti.cast(I, ti.f32) + 0.5 + 0.5 * cf - self.body_center[b]).cross(
                capillary
            )
            ti.atomic_add(self.hydrodynamic_torque[b], torque)
        return reflected

    def _stream_home(self, classified=False):
        if not classified:
            self._classify_home()
        self._stream_home_bulk()
        self._stream_home_boundary()

    @ti.func
    def _home_boundary_at(self, I):
        flag = -1
        if self._fluid(I):
            flag = 0
            for q in ti.static(range(Q)):
                if ti.static(q != REST):
                    J = self._home_neighbor(I, -direction(q))
                    if not self._fluid(J):
                        flag = ti.max(flag, 1)
                        if self._inside(J):
                            if (
                                self.solid_body_id[J] >= 0
                                or self.solid_body_id[J] == -3
                            ):
                                flag = 2
        return flag

    @ti.kernel
    def _classify_home(self):
        for I in ti.grouped(self.water_phase):
            flag = self._home_boundary_at(I)
            self._home_boundary[I] = ti.cast(flag, ti.i8)

    @ti.kernel
    def _stream_home_bulk(self):
        ti.loop_config(block_dim=128)
        for I in ti.grouped(self.water_phase):
            flag = self._home_boundary[I]
            if flag >= 0:
                if flag < 2:
                    p = 0.0
                    j = ti.Vector.zero(ti.f32, 3)
                    S = ti.Matrix.zero(ti.f32, 3, 3)
                    for q in ti.static(range(Q)):
                        c = direction(q)
                        cf = ti.cast(c, ti.f32)
                        J = self._home_neighbor(I, -c)
                        m = self.home_post[I]
                        if self._fluid(J):
                            m = self.home_post[J]
                        else:
                            # For stationary bounce-back, R_-q(p,u,S) is
                            # exactly R_q(p,-u,S), including third order.
                            for a in ti.static(range(1, 4)):
                                m[a] = -m[a]
                        value = reconstruct(q, m)
                        p += value
                        j += cf * value
                        S += (
                            cf.outer_product(cf) - ti.Matrix.identity(ti.f32, 3) / 3
                        ) * value
                    self.home[I] = pack(p, j, S)
                    self.velocity_lattice[I] = j + 0.5 * self.home_acceleration[I]
            else:
                self.velocity_lattice[I] = ti.Vector.zero(ti.f32, 3)
                if self.solid_body_id[I] >= 0:
                    self.velocity_lattice[I] = self._body_velocity_at(
                        ti.cast(I, ti.f32) + 0.5, self.solid_body_id[I]
                    )

    @ti.kernel
    def _stream_home_boundary(self):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 2:
                p = 0.0
                j = ti.Vector.zero(ti.f32, 3)
                S = ti.Matrix.zero(ti.f32, 3, 3)
                for q in range(Q):
                    c = direction(q)
                    cf = ti.cast(c, ti.f32)
                    J = self._home_neighbor(I, -c)
                    value = 0.0
                    if self._fluid(J):
                        value = reconstruct(q, self.home_post[J])
                    else:
                        value = self._home_reflection(I, 26 - q)
                    p += value
                    j += cf * value
                    S += (
                        cf.outer_product(cf) - ti.Matrix.identity(ti.f32, 3) / 3
                    ) * value
                self.home[I] = pack(p, j, S)
                self.velocity_lattice[I] = j + 0.5 * self.home_acceleration[I]

    @ti.kernel
    def _recover_pressure(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                self.pressure_lattice[I] = (
                    self.home[I][0] * density(self.water_phase[I], self.rho_air) / 3
                )
