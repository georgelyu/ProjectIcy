"""Experimental stability operators; all are disabled by the default config.

The pressure populations have sum(g_eq)=0. Reconstruction below therefore
preserves their pressure moment, rather than using mass-Maxwellian moments.
See benchmarks/independent_stability_options_20260921.md for derivation and
the separate (including unsuccessful) validation results.
"""

import taichi as ti

from .lattice import (
    inside_grid,
    lattice_direction,
    lattice_weight,
    mixture_density,
    momentum_equilibrium,
    reconstruct_central_moment_population,
)


@ti.data_oriented
class StabilityOptions:
    @ti.func
    def _stability_neighbor(self, i, j, di, dj):
        # Periodic verification fixtures override this one addressing primitive.
        return ti.Vector([i + di, j + dj])

    @ti.kernel
    def _cache_stability_velocity(self):
        for i, j in self.water_phase:
            self.stability_velocity[i, j] = self._physical_velocity_at(i, j)

    @ti.func
    def _regularization_velocity_neighbor(self, i, j, di, dj):
        index = self._stability_neighbor(i, j, di, dj)
        center = self.stability_velocity[i, j]
        # Half-way stationary wall reflection. At a cut body link use the
        # actual wall fraction, so the linear normal derivative is reproduced.
        value = -center
        if inside_grid(index.x, index.y, self.nx, self.ny):
            if self._is_fluid_cell(index.x, index.y):
                value = self.stability_velocity[index.x, index.y]
            elif self.solid_mask[index.x, index.y] != 0:
                fraction = self._cut_link_fraction(i, j, index.x, index.y)
                boundary = self._body_velocity_at(
                    self._cell_point(i, j) + fraction * ti.Vector([di, dj]),
                    self.solid_body_id[index.x, index.y],
                )
                value = center + (boundary - center) / ti.max(fraction, 1.0e-3)
        return value

    @ti.func
    def _regularized_deviatoric_stress(self, i, j, density, tau):
        gradient = ti.Matrix.zero(ti.f32, 2, 2)
        for q in range(1, 9):
            c = lattice_direction(q)
            neighbor = self._regularization_velocity_neighbor(i, j, c.x, c.y)
            gradient += (
                3.0 * lattice_weight(q) * neighbor.outer_product(ti.cast(c, ti.f32))
            )
        # Pi^neq + source/2 = -rho*tau*cs^2*(grad u + grad u^T).
        factor = -2.0 * density * tau / 3.0
        return factor * ti.Vector(
            [gradient[0, 0] - gradient[1, 1], 0.5 * (gradient[0, 1] + gradient[1, 0])]
        )

    @ti.kernel
    def _measure_phase_mass_flux(self):
        """Time-centered nonadvective flux from the actual phase collision.

        J = delta_rho * (sum_q c_q (h_q+h_q*)/2 - phi*u).
        For the implemented collision this reduces, at CE order, to
        -delta_rho*M*(grad(phi)-lambda*n). No second normal or gradient is
        substituted here. Phase and this flux use the same explicit velocity.
        """
        for i, j in self.water_phase:
            flux = ti.Vector.zero(ti.f32, 2)
            if self._is_fluid_cell(i, j):
                for q in range(9):
                    flux += (
                        ti.cast(lattice_direction(q), ti.f32)
                        * 0.5
                        * (
                            self.phase_populations[i, j, q]
                            + self.phase_stream_buffer[i, j, q]
                        )
                    )
                flux -= self.water_phase[i, j] * self.stability_velocity[i, j]
                flux *= ti.static(
                    self._water_density_lattice - self._air_density_lattice
                )
            self.phase_mass_flux[i, j] = flux

    @ti.func
    def _phase_flux_force_density(self, i, j):
        """Full -div(u tensor J), using antisymmetric shared face fluxes.

        This includes J.grad(u), as well as u*div(J). Closed solid/wall faces
        carry no nonadvective mass flux. The correction is passed through the
        existing force projection AND half-force velocity/pressure recovery.
        """
        force = ti.Vector.zero(ti.f32, 2)
        for axis, sign in ti.static(ti.ndrange(2, 2)):
            direction = 2 * sign - 1
            di, dj = 0, 0
            if ti.static(axis == 0):
                di = direction
            else:
                dj = direction
            index = self._stability_neighbor(i, j, di, dj)
            if inside_grid(index.x, index.y, self.nx, self.ny):
                if self._is_fluid_cell(index.x, index.y):
                    face_mass = 0.5 * (
                        self.phase_mass_flux[i, j][axis]
                        + self.phase_mass_flux[index.x, index.y][axis]
                    )
                    face_velocity = 0.5 * (
                        self.stability_velocity[i, j]
                        + self.stability_velocity[index.x, index.y]
                    )
                    force -= direction * face_mass * face_velocity
        return force

    @ti.func
    def _extrapolate_exposed_stress(self, i, j, density):
        """Extrapolate density-normalized stress from persistent fluid only.

        Never read another concurrently refilled node. Extrapolate kinematic
        stress, rather than copying a water-sized stress into air. Molecular
        relaxation time can still vary across the jump. Geometry's phase/volume
        reconstruction is unchanged.
        """
        result = ti.Vector.zero(ti.f32, 3)
        radius, count = 1, 0.0
        while count == 0.0 and radius < ti.static(max(self.nx, self.ny)):
            total = ti.Vector.zero(ti.f32, 3)
            for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                ni, nj = i + di, j + dj
                if ti.max(ti.abs(di), ti.abs(dj)) == radius and inside_grid(
                    ni, nj, self.nx, self.ny
                ):
                    if (
                        self._is_fluid_cell(ni, nj)
                        and self.previous_solid_mask[ni, nj] == 0
                    ):
                        rho = mixture_density(
                            self.water_phase[ni, nj],
                            self._water_density_lattice,
                            self._air_density_lattice,
                        )
                        velocity = self.momentum_velocity_lattice[ni, nj]
                        pressure = (
                            self.pressure_lattice[ni, nj]
                            - self.hydrostatic_reference_pressure[ni, nj]
                        )
                        stress = ti.Vector.zero(ti.f32, 3)
                        for q in range(9):
                            c = ti.cast(lattice_direction(q), ti.f32)
                            residual = self.momentum_populations[
                                ni, nj, q
                            ] - momentum_equilibrium(q, pressure, rho, velocity)
                            stress += residual * ti.Vector(
                                [c.x * c.x, c.x * c.y, c.y * c.y]
                            )
                        total += stress / rho
                        count += 1.0
            if count > 0.0:
                result = density * total / count
            radius += 1
        return result


@ti.func
def pressure_preserving_stress_population(q, stress, velocity):
    """Lift stress without altering sum(g), momentum or recovered pressure.

    Raw 21/12 are recursively reconstructed from the stress. Raw 22 equals
    raw trace: sum_{q != 0} delta_g = delta_trace - delta_22 = 0. This is the
    essential distinction from standard mass-population regularization.
    """
    xx, xy, yy = stress[0], stress[1], stress[2]
    return reconstruct_central_moment_population(
        lattice_direction(q).x,
        lattice_direction(q).y,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        xx,
        yy,
        xy,
        velocity.y * xx + 2.0 * velocity.x * xy,
        velocity.x * yy + 2.0 * velocity.y * xy,
        xx + yy,
    )
