"""Configuration for a separate, fixed-wall, open CO2 reservoir example."""

from dataclasses import dataclass, field, replace
import math

from ..config import (
    IceBodyConfig,
    IceFlowConfig,
    ThermalConfig,
    PhaseChangeProperties,
    finite,
)
from ..viscosity import water_dynamic_viscosity_pa_s
from .materials import co2_density, co2_dynamic_viscosity_pa_s


@dataclass(frozen=True)
class DryIceThermalConfig(ThermalConfig):
    properties: PhaseChangeProperties = field(
        default_factory=lambda: PhaseChangeProperties(
            melting_temperature_c=-78.5,
            specific_heat_ice_j_kg_k=850.0,
            conductivity_ice_w_m_k=0.20,
            latent_heat_j_kg=573000.0,
        )
    )
    initial_water_temperature_c: float = 60.0
    initial_ice_temperature_c: float = -78.5

    def __post_init__(self):
        # Run the inherited numerical validation with WATER's freezing point,
        # without evaluating a liquid-water viscosity at CO2's sublimation point.
        ThermalConfig(
            **{
                k: getattr(self, k)
                for k in ThermalConfig.__dataclass_fields__
                if k not in ("properties", "initial_ice_temperature_c")
            },
            properties=replace(self.properties, melting_temperature_c=0.0),
            initial_ice_temperature_c=0.0,
        )
        finite("initial_ice_temperature_c", self.initial_ice_temperature_c)
        if self.initial_ice_temperature_c > self.properties.melting_temperature_c:
            raise ValueError("dry ice starts at or below its sublimation temperature")
        if self.initial_water_temperature_c < 0:
            raise ValueError("this example does not include freezing of water")


@dataclass(frozen=True)
class DryIceConfig(IceFlowConfig):
    # Array includes one halo on each side; physical domain is [1, n-1]^3.
    nx: int = 218
    ny: int = 218
    nz: int = 218
    domain_size_m: tuple[float, float, float] = (218 / 3000,) * 3
    boundary_cells: int = 1
    water_width: int = 216
    water_height: int = 52
    floor_water_depth_cells: int = 0
    ice_bodies: tuple[IceBodyConfig, ...] = field(
        default_factory=lambda: (IceBodyConfig((109.0, 79.0, 109.0), (24, 24, 24)),)
    )
    rho_ice: float = 1560.0
    rho_air: float = co2_density(20.0)
    phase_warmup_steps: int = 200
    # For |grad(phi)| << epsilon the compression normal vanishes. The legacy
    # epsilon=0.01 cannot confine tails below epsilon*W/4=0.0125, which produces
    # a heavy diffuse cloud outside an open tank. Keep W=5 and resolve the
    # normal down to a negligible liquid fraction instead.
    phase_normal_epsilon: float = 1e-6
    thermal: DryIceThermalConfig = field(default_factory=DryIceThermalConfig)
    tank_lower_cells: tuple[float, float, float] = (55.0, 1.0, 55.0)
    tank_upper_cells: tuple[float, float, float] = (163.0, 73.0, 163.0)
    tank_wall_cells: int = 3
    water_surface_cells: float = 52.0
    atmospheric_pressure_pa: float = 101325.0
    film_thickness_m: float = 0.0001
    co2_conductivity_w_m_k: float = 0.0146
    co2_specific_heat_j_kg_k: float = 850.0
    heat_search_distance_m: float = 0.008
    sublimation_enabled: bool = True
    body_motion_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.thermal, DryIceThermalConfig):
            raise TypeError("dry ice requires DryIceThermalConfig")
        if self.interface_width != 5.0:
            raise ValueError("the dry-ice example requires W=5")
        if self.boundary_cells != 1:
            raise ValueError("open-tank storage uses exactly one halo cell")
        if (
            self.tank_wall_cells < 1
            or int(self.tank_wall_cells) != self.tank_wall_cells
        ):
            raise ValueError("tank walls must contain an integer number of cells")
        for name in (
            "film_thickness_m",
            "co2_conductivity_w_m_k",
            "co2_specific_heat_j_kg_k",
            "heat_search_distance_m",
            "atmospheric_pressure_pa",
        ):
            finite(name, getattr(self, name), positive=True)
        lo, hi, t = self.tank_lower_cells, self.tank_upper_cells, self.tank_wall_cells
        for a, n in enumerate((self.nx, self.ny, self.nz)):
            if not (1 <= lo[a] < hi[a] <= n - 1) or hi[a] - lo[a] <= 2 * t:
                raise ValueError("invalid open-tank geometry")
            if not math.isclose(lo[a], round(lo[a])) or not math.isclose(
                hi[a], round(hi[a])
            ):
                raise ValueError("fixed tank faces must be grid aligned")
        if lo[1] != 1:
            raise ValueError("the tank must rest on the ground")
        if not lo[1] + t < self.water_surface_cells < hi[1]:
            raise ValueError("water must leave freeboard below the rim")
        co2_dynamic_viscosity_pa_s(self.thermal.initial_air_temperature_c, self.rho_air)
        water_dynamic_viscosity_pa_s(self.thermal.initial_water_temperature_c)
        if len(self.bodies) != 1:
            raise ValueError("the dry-ice example supports one sublimating body")
        if self.effective_sublimation_enthalpy <= 0:
            raise ValueError("effective sublimation enthalpy must be positive")
        if not isinstance(self.sublimation_enabled, bool) or not isinstance(
            self.body_motion_enabled, bool
        ):
            raise ValueError("motion and sublimation switches must be boolean")

    @property
    def physical_size_m(self):
        dx = self.domain_size_m[0] / self.nx
        return tuple((n - 2) * dx for n in (self.nx, self.ny, self.nz))

    @property
    def effective_sublimation_enthalpy(self):
        # Gas is injected at the fixed reservoir temperature. Its initial
        # warming is charged to the same heat donor, not created for free.
        return (
            self.thermal.properties.latent_heat_j_kg
            + self.co2_specific_heat_j_kg_k
            * (
                self.thermal.initial_air_temperature_c
                - self.thermal.properties.melting_temperature_c
            )
        )
