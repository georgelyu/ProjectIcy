"""Three-dimensional finite-volume heat transport and disappearing ice.

Water stores volume [m^3] and sensible energy [J] on the world grid. Ice
stores mass [kg] and sensible energy [J] on an independent material grid.
Walls and water/air interfaces are adiabatic. Aperture reconstruction follows
v3's local temperature extension; its energy change is explicitly reported.
"""
# ruff: noqa: E741 -- I and J are Taichi grouped grid indices.

import math
import numpy as np
import taichi as ti

from .lattice import AXES, FACES


@ti.data_oriented
class Thermal3D:
    def _allocate_thermal(self):
        cfg, p = self.config.thermal, self.config.thermal.properties
        self.dx = self.scales.dx_m
        self.cell_volume = self.dx**3
        self.initial_cell_mass = self.config.rho_ice * self.cell_volume
        self.rho_cp = self.config.rho_water * p.specific_heat_water_j_kg_k
        self.tm = p.melting_temperature_c
        self.cp_ice = p.specific_heat_ice_j_kg_k
        self.k_water, self.k_ice = p.conductivity_water_w_m_k, p.conductivity_ice_w_m_k
        self.latent = p.latent_heat_j_kg
        sizes = np.array([b.size_cells for b in self.config.bodies], dtype=np.int32)
        self.material_size = ti.Vector.field(3, ti.i32, shape=self.body_count)
        self.material_size.from_numpy(sizes)
        temperatures = [
            cfg.initial_ice_temperature_c
            if b.temperature_c is None
            else b.temperature_c
            for b in self.config.bodies
        ]
        self.initial_body_temperature = ti.field(ti.f32, shape=self.body_count)
        self.initial_body_temperature.from_numpy(np.array(temperatures, np.float32))
        material_shape = (self.body_count, *map(int, sizes.max(axis=0)))
        for name in ("material_mass", "material_energy", "material_heat"):
            setattr(self, name, ti.field(ti.f32, shape=material_shape))
        for name in ("water_volume", "water_energy", "target_energy"):
            setattr(self, name, ti.field(ti.f32, shape=self.shape))
        self.sampled_ice_temperature = ti.field(ti.f32, shape=self.shape)
        self.energy_flux = ti.Vector.field(3, ti.f32, shape=self.shape)
        self.volume_flux = ti.Vector.field(3, ti.f32, shape=self.shape)
        self.remap_energy = ti.field(ti.f32, shape=())
        self.discarded_heat = ti.field(ti.f32, shape=())
        self.max_outflow = ti.field(ti.f32, shape=())
        self.thermal_invalid = ti.field(ti.i32, shape=())

    @ti.kernel
    def _initialize_material(self):
        for b, i, j, k in self.material_mass:
            if self._material_inside(ti.Vector([i, j, k]), b):
                self.material_mass[b, i, j, k] = self.initial_cell_mass
                self.material_energy[b, i, j, k] = (
                    self.initial_cell_mass
                    * self.cp_ice
                    * (self.initial_body_temperature[b] - self.tm)
                )

    @ti.func
    def _water_target(self, I):
        value = ti.cast(0.0, ti.f32)
        if self._fluid(I) and self.water_phase[I] > 1e-3:
            value = ti.cast(self.water_phase[I], ti.f32) * self.cell_volume
        return value

    @ti.func
    def _water_temperature(self, I):
        temperature = ti.cast(self.config.thermal.initial_air_temperature_c, ti.f32)
        if self.water_volume[I] > 1e-30:
            temperature = self.tm + self.water_energy[I] / (
                self.rho_cp * self.water_volume[I]
            )
        return ti.cast(temperature, ti.f32)

    @ti.func
    def _ice_temperature(self, b, J):
        mass = self.material_mass[b, J.x, J.y, J.z]
        temp = ti.cast(self.tm, ti.f32)
        if mass > 1e-30:
            temp += self.material_energy[b, J.x, J.y, J.z] / (self.cp_ice * mass)
        return temp

    @ti.kernel
    def _initialize_water_heat(self):
        for I in ti.grouped(self.water_volume):
            self.water_volume[I] = self._water_target(I)
            self.water_energy[I] = (
                self.water_volume[I]
                * self.rho_cp
                * (self.config.thermal.initial_water_temperature_c - self.tm)
            )

    @ti.kernel
    def _build_water_targets(self):
        for I in ti.grouped(self.water_volume):
            target = self._water_target(I)
            specific = ti.cast(0.0, ti.f32)
            if target > 0:
                if self.water_volume[I] > 1e-30:
                    specific = self.water_energy[I] / self.water_volume[I]
                else:
                    total = ti.cast(0.0, ti.f32)
                    # Immediate wet donors, including cells just covered by ice,
                    # are read from the immutable old thermal state.
                    for a, b, c in ti.ndrange(3, 3, 3):
                        J = I + ti.Vector([a - 1, b - 1, c - 1])
                        if self._inside(J):
                            if self.water_volume[J] > 1e-30:
                                specific += self.water_energy[J]
                                total += self.water_volume[J]
                    if total > 1e-30:
                        specific /= total
            self.target_energy[I] = target * specific

    @ti.kernel
    def _commit_water_targets(self):
        for I in ti.grouped(self.water_volume):
            ti.atomic_add(
                self.remap_energy[None], self.target_energy[I] - self.water_energy[I]
            )
            self.water_volume[I] = self._water_target(I)
            self.water_energy[I] = self.target_energy[I]

    def _synchronize_water(self):
        self._build_water_targets()
        self._commit_water_targets()

    @ti.kernel
    def _thermal_fluxes(self, advect: ti.template()):
        for I in ti.grouped(self.water_volume):
            ef, vf = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
            for a in ti.static(range(3)):
                J = I + ti.Vector(ti.static(AXES[a]))
                if self._fluid(I) and self._fluid(J):
                    vi, vj = self.water_volume[I], self.water_volume[J]
                    if vi > 1e-30 and vj > 1e-30:
                        if ti.static(advect):
                            speed = (
                                0.5
                                * (
                                    self.velocity_lattice[I][a]
                                    + self.velocity_lattice[J][a]
                                )
                                * self.scales.velocity_m_s
                            )
                            V, E = vi, self.water_energy[I]
                            if speed < 0:
                                V, E = vj, self.water_energy[J]
                            vf[a] = (
                                speed * ti.min(1.0, V / self.cell_volume) * self.dx**2
                            )
                            ef[a] = vf[a] * E / V
                        else:
                            aperture = ti.min(vi, vj) / self.cell_volume
                            ef[a] = (
                                self.k_water
                                * self.dx
                                * aperture
                                * (
                                    self._water_temperature(I)
                                    - self._water_temperature(J)
                                )
                            )
            self.energy_flux[I], self.volume_flux[I] = ef, vf

    @ti.kernel
    def _apply_thermal_fluxes(self, dt: ti.f32, advect: ti.template()):
        for I in ti.grouped(self.water_volume):
            de, dv = ti.cast(0.0, ti.f32), ti.cast(0.0, ti.f32)
            for a in ti.static(range(3)):
                J = I - ti.Vector(ti.static(AXES[a]))
                de += self.energy_flux[I][a]
                dv += self.volume_flux[I][a]
                if self._inside(J):
                    de -= self.energy_flux[J][a]
                    dv -= self.volume_flux[J][a]
            self.water_energy[I] -= dt * de
            if ti.static(advect):
                self.water_volume[I] -= dt * dv

    @ti.kernel
    def _measure_outflow(self):
        self.max_outflow[None] = 0.0
        for I in ti.grouped(self.water_volume):
            if self._fluid(I) and self.water_volume[I] > 1e-30:
                rate = 0.0
                for a, sign in ti.static(ti.ndrange(3, 2)):
                    c = ti.Vector.zero(ti.i32, 3)
                    c[a] = 2 * sign - 1
                    J = I + c
                    if self._fluid(J) and self.water_volume[J] > 1e-30:
                        rate += ti.max(
                            0.0,
                            (2 * sign - 1)
                            * 0.5
                            * (
                                self.velocity_lattice[I][a]
                                + self.velocity_lattice[J][a]
                            ),
                        )
                ti.atomic_max(self.max_outflow[None], rate)

    def _advance_advection(self, speed_bound=None):
        if not self.config.thermal.advection_enabled:
            return
        # Each velocity component is in [-S, S], so the two outgoing faces
        # on an axis sum to at most S. Convex refill does not increase the
        # fluid L1 speed bound from the preceding stability scan. If 3*S
        # already guarantees one substep, the six-neighbor CFL scan is idle.
        rate = 3 * speed_bound if speed_bound is not None else math.inf
        if rate > self.config.thermal.max_courant_number:
            self._measure_outflow()
            rate = float(self.max_outflow[None])
        if not math.isfinite(rate):
            raise RuntimeError(f"nonfinite thermal velocity at step {self.steps + 1}")
        substeps = max(1, math.ceil(rate / self.config.thermal.max_courant_number))
        self._check_substeps(substeps)
        for _ in range(substeps):
            self._thermal_fluxes(True)
            self._apply_thermal_fluxes(self.scales.dt_s / substeps, True)
            self._synchronize_water()

    def _check_substeps(self, count):
        if count > self.config.thermal.max_substeps_per_update:
            raise RuntimeError(
                f"thermal update needs {count} substeps at step {self.steps + 1}"
            )

    def _advance_conduction(self, dt):
        alpha = max(
            self.k_water / self.rho_cp, self.k_ice / (self.config.rho_ice * self.cp_ice)
        )
        substeps = max(
            1,
            math.ceil(
                alpha * dt / (self.dx**2 * self.config.thermal.max_fourier_number)
            ),
        )
        self._check_substeps(substeps)
        for _ in range(substeps):
            self._body_conduction(dt / substeps)
            self._thermal_fluxes(False)
            self._apply_thermal_fluxes(dt / substeps, False)
            self._exchange_heat(dt / substeps)
            self._apply_melting()
            self.rigid.update_mass(upload=False)
            self.rigid.resolve_contacts()
            self.rigid.upload()
            self._refresh_geometry(phase_changed=False)
        if self.config.check_stability:
            self._check_heat()
            if int(self.thermal_invalid[None]):
                raise RuntimeError(f"invalid thermal state at step {self.steps + 1}")

    @ti.kernel
    def _body_conduction(self, dt: ti.f32):
        for b, i, j, k in self.material_mass:
            I = ti.Vector([i, j, k])
            mass = self.material_mass[b, i, j, k]
            heat = ti.cast(0.0, ti.f32)
            if mass > 0:
                temperature = self._ice_temperature(b, I)
                for face in ti.static(FACES):
                    J = I + ti.Vector(face)
                    if self._material_inside(J, b):
                        other = self.material_mass[b, J.x, J.y, J.z]
                        if other > 0:
                            aperture = ti.min(mass, other) / self.initial_cell_mass
                            heat += (
                                dt
                                * self.k_ice
                                * self.dx
                                * aperture
                                * (self._ice_temperature(b, J) - temperature)
                            )
            self.material_heat[b, i, j, k] = heat

    @ti.func
    def _touches_thermal_ice(self, I):
        near = False
        for f in ti.static(range(7)):
            J = I
            if ti.static(f < 6):
                J = I + ti.Vector(ti.static(FACES[f]))
            if self._inside(J):
                if self.coverage_body_id[J] >= 0 and self.coverage[J] > 1e-8:
                    near = True
        return near

    @ti.func
    def _thermal_ice_bounds(self):
        # Include the rotated material box, trilinear support and heat stencil.
        # Static body iteration avoids a parallel outer-loop bounds reduction.
        lo, hi = ti.Vector(self.shape), ti.Vector([0, 0, 0])
        for b in ti.static(range(self.body_count)):
            half = 0.5 * ti.cast(self.material_size[b], ti.f32) + 1.0
            extent = ti.abs(self.body_rotation[b]) @ half
            lower = ti.cast(ti.floor(self.body_origin[b] - extent), ti.i32) - 3
            upper = ti.cast(ti.ceil(self.body_origin[b] + extent), ti.i32) + 3
            lo, hi = ti.min(lo, lower), ti.max(hi, upper)
        return ti.max(lo, 0), ti.min(hi, ti.Vector(self.shape))

    @ti.func
    def _exchange_heat_cell(self, I, dt):
        if (
            self.water_volume[I] > 1e-30
            and self._fluid(I)
            and self._touches_thermal_ice(I)
        ):
            # Seven world samples x eight trilinear material donors.
            heat = ti.Vector.zero(ti.f32, 56)
            body = ti.Vector.zero(ti.i32, 7)
            base_x = ti.Vector.zero(ti.i32, 7)
            base_y = ti.Vector.zero(ti.i32, 7)
            base_z = ti.Vector.zero(ti.i32, 7)
            requested = ti.cast(0.0, ti.f32)
            tw = self._water_temperature(I)
            aperture = self.water_volume[I] / self.cell_volume
            for f in ti.static(range(7)):
                J = I
                if ti.static(f < 6):
                    J = I + ti.Vector(ti.static(FACES[f]))
                if self._inside(J):
                    b = self.coverage_body_id[J]
                    fraction = self.coverage[J]
                    if b >= 0 and fraction > 1e-8:
                        grid = self._material_grid(ti.cast(J, ti.f32) + 0.5, b)
                        base = ti.cast(ti.floor(grid), ti.i32)
                        part = grid - ti.cast(base, ti.f32)
                        body[f] = b
                        base_x[f], base_y[f], base_z[f] = base.x, base.y, base.z
                        contact = ti.min(
                            aperture,
                            ti.min(
                                1.0,
                                fraction
                                / self.config.thermal.solid_liquid_threshold,
                            ),
                        )
                        for a, c, d in ti.static(ti.ndrange(2, 2, 2)):
                            K = base + ti.Vector([a, c, d])
                            if self._material_inside(K, b):
                                mass = self.material_mass[b, K.x, K.y, K.z]
                                if mass > 0:
                                    w = (
                                        (part.x if a else 1 - part.x)
                                        * (part.y if c else 1 - part.y)
                                        * (part.z if d else 1 - part.z)
                                    )
                                    share = (
                                        w
                                        * mass
                                        / (self.initial_cell_mass * fraction)
                                    )
                                    value = (
                                        dt
                                        * (
                                            2
                                            * self.k_water
                                            * self.k_ice
                                            / (self.k_water + self.k_ice)
                                        )
                                        * self.dx
                                        * contact
                                        * share
                                        * ti.max(
                                            0.0, tw - self._ice_temperature(b, K)
                                        )
                                    )
                                    slot = ti.static(8 * f + 4 * a + 2 * c + d)
                                    heat[slot] = value
                                    requested += value
            scale = ti.min(
                1.0, ti.max(0.0, self.water_energy[I]) / ti.max(requested, 1e-30)
            )
            for f in ti.static(range(7)):
                for a, c, d in ti.static(ti.ndrange(2, 2, 2)):
                    slot = ti.static(8 * f + 4 * a + 2 * c + d)
                    value = scale * heat[slot]
                    if value > 0:
                        ti.atomic_add(
                            self.material_heat[
                                body[f], base_x[f] + a, base_y[f] + c, base_z[f] + d
                            ],
                            value,
                        )
            self.water_energy[I] -= scale * requested

    @ti.kernel
    def _exchange_heat(self, dt: ti.f32):
        lo, hi = self._thermal_ice_bounds()
        # The union box visits overlapping body neighborhoods exactly once.
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            self._exchange_heat_cell(ti.Vector([i, j, k]), dt)

    @ti.kernel
    def _apply_melting(self):
        for b, i, j, k in self.material_mass:
            mass = self.material_mass[b, i, j, k]
            energy = self.material_energy[b, i, j, k] + self.material_heat[b, i, j, k]
            if energy > 0:
                melted = ti.min(mass, energy / self.latent)
                ti.atomic_add(self.discarded_heat[None], energy - melted * self.latent)
                mass -= melted
                energy = 0.0
            if mass <= 1e-6 * self.initial_cell_mass:
                ti.atomic_add(self.discarded_heat[None], energy - mass * self.latent)
                mass, energy = 0.0, 0.0
            self.material_mass[b, i, j, k] = mass
            self.material_energy[b, i, j, k] = energy

    @ti.kernel
    def _sample_ice_temperature(self):
        for I in ti.grouped(self.water_volume):
            t = ti.cast(self.tm, ti.f32)
            b = self.solid_body_id[I]
            if b >= 0:
                grid = self._material_grid(ti.cast(I, ti.f32) + 0.5, b)
                J = ti.cast(ti.round(grid), ti.i32)
                for a in ti.static(range(3)):
                    J[a] = ti.min(self.material_size[b][a] - 1, ti.max(0, J[a]))
                t = self._ice_temperature(b, J)
            self.sampled_ice_temperature[I] = ti.cast(t, ti.f32)

    @ti.kernel
    def _check_heat(self):
        self.thermal_invalid[None] = 0
        for I in ti.grouped(self.water_volume):
            e, v = self.water_energy[I], self.water_volume[I]
            if ti.math.isnan(e) or ti.math.isinf(e) or e < -1e-10 or v < -1e-20:
                ti.atomic_add(self.thermal_invalid[None], 1)
        for I in ti.grouped(self.material_mass):
            m, e = self.material_mass[I], self.material_energy[I]
            if ti.math.isnan(m) or ti.math.isnan(e) or ti.math.isinf(e) or m < 0:
                ti.atomic_add(self.thermal_invalid[None], 1)
