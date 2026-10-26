"""Independent open-tank sublimation extension of the existing HOME solver."""
# ruff: noqa: E741 -- lattice indices

import numpy as np
import taichi as ti
from ..simulator import IceFlow3D
from ..config import integer
from ..home import central_collision
from ..lattice import density
from .config import DryIceConfig
from .boundaries import OpenTankBoundaries
from .thermal import SublimationThermal
from .rigid import OpenTankBodies
from .flow import OpenTankFlow


@ti.data_oriented
class DryIceFlow3D(OpenTankFlow, OpenTankBoundaries, SublimationThermal, IceFlow3D):
    def __init__(self, config):
        if not isinstance(config, DryIceConfig):
            raise TypeError("DryIceFlow3D requires DryIceConfig")
        super().__init__(config)
        self.rigid = OpenTankBodies(self)
        self.rigid.update_mass()
        self._fill_open_halo()
        self.initial_gas_mass_kg = self.gas_mass_kg()
        self.initial_solid_mass_kg = float(
            self.material_mass.to_numpy().sum(dtype=float)
        )
        fluid = self.solid_body_id.to_numpy() == -1
        self.initial_water_phase_volume_m3 = float(
            self.water_phase.to_numpy()[fluid].sum(dtype=float) * self.cell_volume
        )
        self.initial_pressure_storage_kg = float(
            self.home.to_numpy()[..., 0][fluid].sum(dtype=float)
            * self.cell_volume
            * config.rho_air
        )
        self.initial_heat_j = float(
            self.water_energy.to_numpy().sum(dtype=float)
            + self.material_energy.to_numpy().sum(dtype=float)
        )

    @ti.kernel
    def _collide_momentum(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                # The old velocity is needed only by moving-wall reflection
                # and is captured there before that cell is streamed. The
                # shared solver's old-pressure buffer has no reader here.
                rho = density(self.water_phase[I], self.rho_air)
                dm = self.gas_mass_step[I]
                acceleration = self.home_acceleration[I]
                if dm != 0.0:
                    # Sublimation deposits mass and its accompanying momentum
                    # together. Almost all cells have neither source.
                    acceleration += (
                        self.gas_momentum_step[I] - dm * self.velocity_lattice[I]
                    ) / (rho * self.scales.mass_kg)
                    self.home_acceleration[I] = acceleration
                post = central_collision(
                    self.home[I],
                    acceleration,
                    1 / (0.5 + 3 * self._kinematic_viscosity(I)),
                    self.config.cm_acoustic_relaxation_rate,
                )
                # dp*/dt + div(u) = S_V. The source has zeroth moment S_V,
                # zero first and zero second Hermite moments. The solid's
                # volume loss is already represented by retreating geometry.
                post[0] += (
                    dm
                    / self.cell_volume
                    * (1 / self.config.rho_air - 1 / self.config.rho_ice)
                )
                self.home_post[I] = post

    def step(self, num_steps=1):
        for _ in range(integer("num_steps", num_steps, 0)):
            self.gas_mass_step.fill(0)
            self.gas_momentum_step.fill(0)
            if self.config.sublimation_enabled:
                self._sublimate(self.scales.dt_s)
            self._fill_open_halo()
            if not self._gradient_cache_valid:
                self._gradients()
            self._collide_phase()
            # Initialization / invalidation classifies the full grid above;
            # moving geometry refreshes its affected halo at the end of step.
            self._forces(True, moment_pressure=True, classified=True)
            self._collide_momentum()
            self.hydrodynamic_force.fill(0)
            self.hydrodynamic_torque.fill(0)
            self._stream_home(classified=True)
            self._recover_phase(recover_flow=True)
            self._fill_open_halo()
            self._gradients(classified=True)
            speed = (
                self.check_stability(scanned=True)
                if self.config.check_stability
                else None
            )
            self.rigid.update_mass()
            self.rigid.advance()
            self._refresh_geometry()
            self._advance_advection(speed)
            self._thermal_pending += 1
            if self._thermal_pending >= self.config.thermal.update_interval_lbm_steps:
                self._advance_conduction(self._thermal_pending * self.scales.dt_s)
                self._thermal_pending = 0
            self._gradient_cache_valid = True
            self.steps += 1

    def gas_mass_kg(self):
        ids, phi = self.solid_body_id.to_numpy(), self.water_phase.to_numpy()
        return float(
            np.where(ids == -1, 1 - phi, 0).sum(dtype=float)
            * self.cell_volume
            * self.config.rho_air
        )

    def diagnostics(self):
        result = super().diagnostics()
        mass = float(
            self.material_mass.to_numpy().sum(dtype=float)
            - self.sublimation_mass_remainder.to_numpy().sum(dtype=float)
        )
        emitted, heat_water, heat_gas = (
            self.source_integrals.to_numpy().astype(float)
            - self.source_integral_remainders.to_numpy().astype(float)
        ).sum(axis=(0, 1, 2, 3))
        water_out = float(self.water_outflow_lu[None]) * self.cell_volume
        volume_out = (
            float(self.open_volume_flux_cell.to_numpy().sum(dtype=float))
            * self.cell_volume
        )
        gas_out = (volume_out - water_out) * self.config.rho_air
        gas_mass = self.gas_mass_kg()
        fluid = self.solid_body_id.to_numpy() == -1
        phase_volume = float(
            self.water_phase.to_numpy()[fluid].sum(dtype=float) * self.cell_volume
        )
        pressure_storage = (
            float(
                self.home.to_numpy()[..., 0][fluid].sum(dtype=float)
                * self.cell_volume
                * self.config.rho_air
            )
            - self.initial_pressure_storage_kg
        )
        result.update(
            {
                "solid_co2_mass_kg": mass,
                "sublimated_co2_mass_kg": emitted,
                "solid_mass_source_residual_kg": self.initial_solid_mass_kg
                - mass
                - emitted,
                "gas_co2_mass_kg": gas_mass,
                "gas_net_outflow_kg": gas_out,
                "co2_mass_residual_kg": gas_mass
                - self.initial_gas_mass_kg
                + gas_out
                - emitted,
                "co2_pressure_storage_kg": pressure_storage,
                "co2_continuity_residual_kg": gas_mass
                - self.initial_gas_mass_kg
                + gas_out
                - emitted
                + pressure_storage,
                "water_net_outflow_m3": water_out,
                "water_phase_volume_m3": phase_volume,
                "phase_volume_change_ratio": phase_volume
                / self.initial_water_phase_volume_m3
                - 1,
                "water_mass_residual_kg": (
                    phase_volume - self.initial_water_phase_volume_m3 + water_out
                )
                * self.config.rho_water,
                "heat_from_water_j": heat_water,
                "heat_from_reservoir_j": heat_gas,
                "sublimation_enthalpy_j": emitted
                * self.config.effective_sublimation_enthalpy,
                "water_heat_net_outflow_j": float(self.thermal_open_energy[None]),
                "thermal_budget_residual_j": result["water_energy_j"]
                + result["ice_sensible_energy_j"]
                + emitted * self.config.effective_sublimation_enthalpy
                + float(self.thermal_open_energy[None])
                - self.initial_heat_j
                - heat_gas
                - result["thermal_remap_energy_j"],
                "co2_dynamic_viscosity_pa_s": self.mu_air,
                "dt_s": self.scales.dt_s,
                "interface_width_cells": self.config.interface_width,
            }
        )
        return result

    def snapshot(self):
        fields = super().snapshot()
        volume, energy = self.water_volume.to_numpy(), self.water_energy.to_numpy()
        water_temp = np.full(self.shape, self.water_reference_temperature, np.float32)
        wet = volume > 1e-30
        water_temp[wet] += energy[wet] / (self.rho_cp * volume[wet])
        fields["water_temperature_c"] = water_temp
        # A trace of diffuse liquid in a gas-dominated cell must not colour the
        # isothermal gas as hot water. Retain separate liquid-component data.
        fluid = fields["solid_body_id"] == -1
        fields["temperature_c"][fluid] = np.where(
            fields["water_phase"][fluid] >= 0.5,
            water_temp[fluid],
            self.config.thermal.initial_air_temperature_c,
        )
        fields["gas_mass_source_kg_m3_s"] = self.gas_mass_step.to_numpy() / (
            self.cell_volume * self.scales.dt_s
        )
        return fields
