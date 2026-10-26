"""Heat-limited sublimation with an explicit unresolved CO2-film resistance.

The gas is isothermal at the reservoir state. L_eff includes warming newly
formed gas to that state. A ray through resolved gas adds its thermal
resistance to the specified subgrid film. Fixed tank walls stop the ray and
have zero heat flux. This is a closure, not a resolved prediction of film
thickness, freezing, dissolution, or microscopic nucleation.
"""
# ruff: noqa: E741 -- lattice indices

import math
import numpy as np
import taichi as ti
from ..thermal import Thermal3D
from ..lattice import AXES
from .materials import co2_dynamic_viscosity_pa_s


@ti.data_oriented
class SublimationThermal:
    def _allocate_thermal(self):
        Thermal3D._allocate_thermal(self)
        self._home_moving_neighbor = ti.field(ti.i8, shape=self.shape)
        self._fixed_mirror_faces = ti.field(ti.u8, shape=self.shape)
        self._thermal_flux_active = ti.field(ti.i8, shape=self.shape)
        self._transient_field_names = self._transient_field_names | {
            "_home_moving_neighbor",
            "_fixed_mirror_faces",
            "_thermal_flux_active",
        }
        self.water_reference_temperature = (
            self.config.thermal.initial_water_temperature_c
        )
        # Evaluate T-Ts before uploading: fast-math reassociation of m*cp*(T-Ts)
        # can otherwise create positive initial enthalpy even when T == Ts.
        self.initial_solid_specific_energy = ti.field(ti.f32, shape=self.body_count)
        self.initial_solid_specific_energy.from_numpy(
            np.array(
                [
                    self.cp_ice
                    * (
                        (
                            self.config.thermal.initial_ice_temperature_c
                            if b.temperature_c is None
                            else b.temperature_c
                        )
                        - self.tm
                    )
                    for b in self.config.bodies
                ],
                np.float32,
            )
        )
        self.mu_air = co2_dynamic_viscosity_pa_s(
            self.config.thermal.initial_air_temperature_c, self.config.rho_air
        )
        self.nu_air = (
            self.mu_air / self.config.rho_air * self.scales.dt_s / self.scales.dx_m**2
        )
        self.heat_search_cells = max(
            2, math.ceil(self.config.heat_search_distance_m / self.dx)
        )
        self.gas_mass_step = ti.field(ti.f32, shape=self.shape)
        self.gas_momentum_step = ti.Vector.field(3, ti.f32, shape=self.shape)
        self.sublimation_mass_before = ti.field(ti.f32, shape=self.material_mass.shape)
        self.sublimation_mass_remainder = ti.field(
            ti.f32, shape=self.material_mass.shape
        )
        self.source_integrals = ti.Vector.field(
            3, ti.f32, shape=self.material_mass.shape
        )
        self.source_integral_remainders = ti.Vector.field(
            3, ti.f32, shape=self.material_mass.shape
        )
        self.open_volume_flux_cell = ti.field(ti.f32, shape=self.shape)
        self.face_gas_mass = ti.field(ti.f32, shape=(*self.material_mass.shape, 6))
        self.face_gas_cell = ti.Vector.field(
            3, ti.i32, shape=(*self.material_mass.shape, 6)
        )
        self.sublimation_temperature_before = ti.field(ti.f32, shape=self.shape)
        for name in (
            "sublimated_mass",
            "heat_from_water",
            "heat_from_reservoir",
            "water_outflow_lu",
            "thermal_open_energy",
        ):
            setattr(self, name, ti.field(ti.f32, shape=()))
        self.minimum_water_temperature = ti.field(ti.f32, shape=())

    @ti.kernel
    def _initialize_material(self):
        for b, i, j, k in self.material_mass:
            if self._material_inside(ti.Vector([i, j, k]), b):
                self.material_mass[b, i, j, k] = self.initial_cell_mass
                self.material_energy[b, i, j, k] = (
                    self.initial_cell_mass * self.initial_solid_specific_energy[b]
                )

    @ti.kernel
    def _initialize_water_heat(self):
        # Store cooling relative to initial water temperature so that small
        # heat withdrawals do not round away against a 138.5 K baseline.
        for I in ti.grouped(self.water_volume):
            self.water_volume[I] = self._water_target(I)
            self.water_energy[I] = 0.0

    @ti.func
    def _water_temperature(self, I):
        value = ti.cast(self.config.thermal.initial_air_temperature_c, ti.f32)
        if self.water_volume[I] > 1e-30:
            value = self.water_reference_temperature + self.water_energy[I] / (
                self.rho_cp * self.water_volume[I]
            )
        return value

    @ti.func
    def _deposit_gas(self, I, dm, ub):
        total = 0.0
        for a, b, c in ti.ndrange(3, 3, 3):
            J = I + ti.Vector([a - 1, b - 1, c - 1])
            if self._fluid(J) and self._connected(I, J - I):
                total += ti.cast(
                    (2 if a == 1 else 1) * (2 if b == 1 else 1) * (2 if c == 1 else 1),
                    ti.f32,
                )
        for a, b, c in ti.ndrange(3, 3, 3):
            J = I + ti.Vector([a - 1, b - 1, c - 1])
            if self._fluid(J) and self._connected(I, J - I):
                weight = (
                    ti.cast(
                        (2 if a == 1 else 1)
                        * (2 if b == 1 else 1)
                        * (2 if c == 1 else 1),
                        ti.f32,
                    )
                    / total
                )
                ti.atomic_add(self.gas_mass_step[J], dm * weight)
                ti.atomic_add(self.gas_momentum_step[J], dm * weight * ub)

    def _sublimate(self, dt):
        # Exposure and donor temperatures must be read from immutable state:
        # several faces can cool the same cell and retire in the same step.
        # Fast-math reassociation destroys the compensated mass subtraction
        # and mixes terms separated by many orders of magnitude. Compile only
        # these NEW kernels with strict FP32 arithmetic. Restore the shared
        # runtime configuration so legacy flow kernels retain their settings.
        cfg = ti.lang.impl.current_cfg()
        fast_math = cfg.fast_math
        try:
            cfg.fast_math = False
            self.sublimation_mass_before.copy_from(self.material_mass)
            self._freeze_sublimation_temperature()
            self.face_gas_mass.fill(0)
            self._sublimation_heat_and_mass_strict(dt)
            self._spread_sublimation_gas()
        finally:
            cfg.fast_math = fast_math

    @ti.kernel
    def _freeze_sublimation_temperature(self):
        for I in ti.grouped(self.water_phase):
            self.sublimation_temperature_before[I] = self._water_temperature(I)

    @ti.kernel
    def _sublimation_heat_and_mass_strict(self, dt: ti.f32):
        for b, i, j, k in self.material_mass:
            stored_mass = self.sublimation_mass_before[b, i, j, k]
            mass = ti.max(
                0.0, stored_mass - self.sublimation_mass_remainder[b, i, j, k]
            )
            if mass > 0:
                local = ti.Vector([i, j, k])
                point = self.body_origin[b] + self.body_rotation[b] @ (
                    ti.cast(local, ti.f32)
                    + 0.5
                    - 0.5 * ti.cast(self.material_size[b], ti.f32)
                )
                heat = ti.Vector.zero(ti.f32, 6)
                wet = ti.Vector.zero(ti.i32, 6)
                donors = ti.Matrix.zero(ti.i32, 6, 3)
                injection = ti.Matrix.zero(ti.i32, 6, 3)
                total = 0.0
                for f in range(6):
                    face = ti.Vector.zero(ti.i32, 3)
                    face[f // 2] = 2 * (f % 2) - 1
                    neighbor = local + face
                    exposed = True
                    if self._material_inside(neighbor, b):
                        exposed = (
                            self.sublimation_mass_before[
                                b, neighbor.x, neighbor.y, neighbor.z
                            ]
                            <= 0
                        )
                    if exposed:
                        normal = self.body_rotation[b] @ ti.cast(face, ti.f32)
                        found_gas, found_water, blocked = False, False, False
                        gas_cell, water_cell = (
                            ti.Vector([-1, -1, -1]),
                            ti.Vector([-1, -1, -1]),
                        )
                        gap = 0.0
                        for r in range(1, self.heat_search_cells + 1):
                            J = ti.cast(ti.floor(point + r * normal), ti.i32)
                            if not blocked and not found_water and self._inside(J):
                                if self.solid_body_id[J] == -2:
                                    blocked = True
                                elif self._fluid(J):
                                    if not found_gas:
                                        gas_cell, found_gas = J, True
                                    if (
                                        self.water_phase[J] > 0.5
                                        and self.water_volume[J] > 0
                                    ):
                                        water_cell, found_water = J, True
                                        gap = ti.max(0.0, (r - 1.0) * self.dx)
                        if found_gas:
                            area = self.dx**2 * ti.pow(
                                mass / self.initial_cell_mass, 2.0 / 3.0
                            )
                            temperature = self.config.thermal.initial_air_temperature_c
                            resistance = (
                                0.5
                                * ti.cast(self.material_size[b].min(), ti.f32)
                                * self.dx
                                / self.config.co2_conductivity_w_m_k
                            )
                            if found_water:
                                temperature = self.sublimation_temperature_before[
                                    water_cell
                                ]
                                resistance = (
                                    (self.config.film_thickness_m + gap)
                                    / self.config.co2_conductivity_w_m_k
                                    + 0.5 * self.dx / self.k_water
                                )
                                wet[f] = 1
                                for a in ti.static(range(3)):
                                    donors[f, a] = water_cell[a]
                            heat[f] = (
                                dt
                                * area
                                * ti.max(
                                    0.0, temperature - self._ice_temperature(b, local)
                                )
                                / resistance
                            )
                            total += heat[f]
                            for a in ti.static(range(3)):
                                injection[f, a] = gas_cell[a]
                old_energy = self.material_energy[b, i, j, k]
                used = ti.min(
                    total,
                    ti.max(
                        0.0,
                        -old_energy + mass * self.config.effective_sublimation_enthalpy,
                    ),
                )
                energy = old_energy + used
                dm = ti.min(
                    mass,
                    ti.max(0.0, energy) / self.config.effective_sublimation_enthalpy,
                )
                # Compensated subtraction retains losses below one material
                # cell's FP32 ulp during the weak gas-side heating stage.
                loss = dm + self.sublimation_mass_remainder[b, i, j, k]
                remaining = ti.max(0.0, stored_mass - loss)
                self.material_mass[b, i, j, k] = remaining
                self.sublimation_mass_remainder[b, i, j, k] = loss - (
                    stored_mass - remaining
                )
                self.material_energy[b, i, j, k] = ti.min(0.0, energy)
                ti.atomic_add(self.sublimated_mass[None], dm)
                integrals = ti.Vector([dm, 0.0, 0.0])
                for f in range(6):
                    if heat[f] > 0 and total > 0:
                        portion = heat[f] / total
                        q = used * portion
                        if wet[f] == 1:
                            D = ti.Vector([donors[f, 0], donors[f, 1], donors[f, 2]])
                            ti.atomic_add(self.water_energy[D], -q)
                            ti.atomic_add(self.heat_from_water[None], q)
                            integrals[1] += q
                        else:
                            ti.atomic_add(self.heat_from_reservoir[None], q)
                            integrals[2] += q
                        I = ti.Vector(
                            [injection[f, 0], injection[f, 1], injection[f, 2]]
                        )
                        self.face_gas_mass[b, i, j, k, f] = dm * portion
                        self.face_gas_cell[b, i, j, k, f] = I
                # Compensated per-cell totals avoid millions of tiny additions
                # to an ever larger global scalar over a long simulation.
                old = self.source_integrals[b, i, j, k]
                increment = integrals - self.source_integral_remainders[b, i, j, k]
                updated = old + increment
                self.source_integral_remainders[b, i, j, k] = (
                    updated - old
                ) - increment
                self.source_integrals[b, i, j, k] = updated

    @ti.kernel
    def _spread_sublimation_gas(self):
        for b, i, j, k, f in self.face_gas_mass:
            dm = self.face_gas_mass[b, i, j, k, f]
            if dm > 0:
                I = self.face_gas_cell[b, i, j, k, f]
                self._deposit_gas(
                    I, dm, self._body_velocity_at(ti.cast(I, ti.f32) + 0.5, b)
                )

    @ti.kernel
    def _thermal_fluxes(self, advect: ti.template()):
        for I in ti.grouped(self.water_volume):
            ef, vf = ti.Vector.zero(ti.f32, 3), ti.Vector.zero(ti.f32, 3)
            for a in ti.static(range(3)):
                J = I + ti.Vector(ti.static(AXES[a]))
                if self._inside(J):
                    fi, fj = self._fluid(I), self._fluid(J)
                    if (fi and fj) or (fi and self._open(J)) or (self._open(I) and fj):
                        vi, vj = self.water_volume[I], self.water_volume[J]
                        if ti.static(advect):
                            # With no liquid donor, the original upwind flux
                            # is exactly zero, regardless of gas velocity.
                            if vi > 1e-30 or vj > 1e-30:
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
                                # The reservoir supplies CO2 only; its water V=0.
                                if V > 1e-30:
                                    vf[a] = (
                                        speed
                                        * ti.min(1.0, V / self.cell_volume)
                                        * self.dx**2
                                    )
                                    ef[a] = vf[a] * E / V
                        else:
                            if fi and fj and vi > 1e-30 and vj > 1e-30:
                                ef[a] = (
                                    self.k_water
                                    * self.dx
                                    * ti.min(vi, vj)
                                    / self.cell_volume
                                    * (
                                        self._water_temperature(I)
                                        - self._water_temperature(J)
                                    )
                                )
            self.energy_flux[I], self.volume_flux[I] = ef, vf
            self._thermal_flux_active[I] = ti.cast(
                (ef != 0.0).any() or (vf != 0.0).any(), ti.i8
            )

    @ti.kernel
    def _apply_thermal_fluxes(self, dt: ti.f32, advect: ti.template()):
        for I in ti.grouped(self.water_volume):
            if self._fluid(I):
                # Read immutable face activity, not neighboring water_volume:
                # the latter is updated in place by this kernel.
                active = self._thermal_flux_active[I] != 0
                for a in ti.static(range(3)):
                    c = ti.Vector(ti.static(AXES[a]))
                    active = active or self._thermal_flux_active[I - c] != 0
                if active:
                    de, dv, boundary_heat = 0.0, 0.0, 0.0
                    for a in ti.static(range(3)):
                        c = ti.Vector(ti.static(AXES[a]))
                        de += self.energy_flux[I][a] - self.energy_flux[I - c][a]
                        dv += self.volume_flux[I][a] - self.volume_flux[I - c][a]
                        if self._open(I + c):
                            boundary_heat += self.energy_flux[I][a]
                        if self._open(I - c):
                            boundary_heat -= self.energy_flux[I - c][a]
                    self.water_energy[I] -= dt * de
                    # Avoid global atomic contention on zero contributions.
                    if boundary_heat != 0.0:
                        ti.atomic_add(
                            self.thermal_open_energy[None], dt * boundary_heat
                        )
                    if ti.static(advect):
                        self.water_volume[I] -= dt * dv

    @ti.kernel
    def _apply_sensible_heat(self):
        for I in ti.grouped(self.material_mass):
            self.material_energy[I] += self.material_heat[I]

    def _advance_conduction(self, dt):
        alpha = max(
            self.k_water / self.rho_cp, self.k_ice / (self.config.rho_ice * self.cp_ice)
        )
        count = max(
            1,
            math.ceil(
                alpha * dt / (self.dx**2 * self.config.thermal.max_fourier_number)
            ),
        )
        self._check_substeps(count)
        for _ in range(count):
            self._body_conduction(dt / count)
            self._apply_sensible_heat()
            self._thermal_fluxes(False)
            self._apply_thermal_fluxes(dt / count, False)
        self._check_heat()
        self._check_water_freezing()
        if int(self.thermal_invalid[None]):
            raise RuntimeError("invalid dry-ice thermal state")
        if float(self.minimum_water_temperature[None]) < 0:
            raise RuntimeError(
                "water reached freezing; the sublimation example requires a water-freezing model for this state"
            )

    @ti.kernel
    def _check_water_freezing(self):
        self.minimum_water_temperature[None] = (
            self.config.thermal.initial_water_temperature_c
        )
        for I in ti.grouped(self.water_phase):
            if self._fluid(I) and self.water_phase[I] > 0.5:
                ti.atomic_min(
                    self.minimum_water_temperature[None], self._water_temperature(I)
                )

    @ti.kernel
    def _check_heat(self):
        self.thermal_invalid[None] = 0
        for I in ti.grouped(self.water_volume):
            e, v = self.water_energy[I], self.water_volume[I]
            if ti.math.isnan(e) or ti.math.isinf(e) or ti.math.isnan(v) or v < -1e-20:
                ti.atomic_add(self.thermal_invalid[None], 1)
        for I in ti.grouped(self.material_mass):
            m, e = self.material_mass[I], self.material_energy[I]
            if ti.math.isnan(m) or ti.math.isnan(e) or ti.math.isinf(e) or m < 0:
                ti.atomic_add(self.thermal_invalid[None], 1)
