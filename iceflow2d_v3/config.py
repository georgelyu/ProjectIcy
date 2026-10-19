"""Configuration for coupled falling-ice melting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from .viscosity import air_dynamic_viscosity_pa_s, water_dynamic_viscosity_pa_s

DEFAULT_REFERENCE_VELOCITY_M_S = 4.0


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


ThermalBoundaryKind = Literal["adiabatic", "dirichlet"]
WaterBuoyancyModel = Literal["linear"]
MovingBodyThermalScheme = Literal["body_ale"]
ThermalScheme = Literal["enthalpy_fv", "enthalpy_d2q5"]

_ADIABATIC = 0
_DIRICHLET = 1
_BOUNDARY_CODE = {
    "adiabatic": _ADIABATIC,
    "dirichlet": _DIRICHLET,
}


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True, slots=True)
class PhaseChangeProperties:
    """Specific heat, conductivity, and latent heat for water and ice.

    Densities are intentionally not duplicated here.  A coupled solver gets
    them from the hydrodynamic configuration so that momentum and energy
    cannot silently use different material densities.
    """

    melting_temperature_c: float = 0.0
    specific_heat_water_j_kg_k: float = 4186.0
    specific_heat_ice_j_kg_k: float = 2100.0
    conductivity_water_w_m_k: float = 0.60
    conductivity_ice_w_m_k: float = 2.20
    latent_heat_j_kg: float = 334000.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "melting_temperature_c",
            _finite("melting_temperature_c", self.melting_temperature_c),
        )
        for name in (
            "specific_heat_water_j_kg_k",
            "specific_heat_ice_j_kg_k",
            "conductivity_water_w_m_k",
            "conductivity_ice_w_m_k",
            "latent_heat_j_kg",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))


@dataclass(frozen=True, slots=True)
class ThermalBoundary:
    """Thermal condition on one side of the active container.

    ``value`` is a temperature in degrees Celsius for ``dirichlet``;
    ``adiabatic`` requires a zero value.
    """

    kind: ThermalBoundaryKind = "adiabatic"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _BOUNDARY_CODE:
            raise ValueError("thermal boundary kind must be 'adiabatic' or 'dirichlet'")
        value = _finite("thermal boundary value", self.value)
        if self.kind == "adiabatic" and value != 0.0:
            raise ValueError("adiabatic thermal boundary value must be zero")
        object.__setattr__(self, "value", value)

    @classmethod
    def adiabatic(cls) -> "ThermalBoundary":
        return cls("adiabatic", 0.0)

    @classmethod
    def dirichlet(cls, temperature_c: float) -> "ThermalBoundary":
        return cls("dirichlet", temperature_c)


@dataclass(frozen=True, slots=True)
class ThermalBoundarySet:
    """Named thermal conditions for the four container sides."""

    left: ThermalBoundary = field(default_factory=ThermalBoundary.adiabatic)
    right: ThermalBoundary = field(default_factory=ThermalBoundary.adiabatic)
    bottom: ThermalBoundary = field(default_factory=ThermalBoundary.adiabatic)
    top: ThermalBoundary = field(default_factory=ThermalBoundary.adiabatic)

    def __post_init__(self) -> None:
        for side in ("left", "right", "bottom", "top"):
            if not isinstance(getattr(self, side), ThermalBoundary):
                raise TypeError(f"{side} must be a ThermalBoundary")


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    """Physical and numerical controls for coupled heat and phase change.

    The ambient water is required to start at or above the melting
    temperature and the initial ice at or below it.  Only original material
    ice melts; melted material disappears without adding liquid.

    ``scheme`` selects finite-volume or tau=1 D2Q5 bulk conduction. Both
    schemes use the same split advection, moving geometry and latent heat.
    """

    properties: PhaseChangeProperties = field(default_factory=PhaseChangeProperties)
    boundaries: ThermalBoundarySet = field(default_factory=ThermalBoundarySet)
    initial_water_temperature_c: float = 20.0
    initial_ice_temperature_c: float = 0.0
    initial_air_temperature_c: float = 20.0
    advection_enabled: bool = True
    water_air_interface_adiabatic: bool = True
    update_interval_lbm_steps: int = 1
    solid_liquid_threshold: float = 0.5
    max_fourier_number: float = 0.15
    max_courant_number: float = 0.50
    max_substeps_per_update: int = 64
    water_buoyancy_model: WaterBuoyancyModel = "linear"
    thermal_expansion_water_1_k: float = 2.1e-4
    buoyancy_reference_temperature_c: float | None = None
    scheme: ThermalScheme = "enthalpy_fv"
    moving_body_scheme: MovingBodyThermalScheme = "body_ale"

    def __post_init__(self) -> None:
        if not isinstance(self.properties, PhaseChangeProperties):
            raise TypeError("properties must be a PhaseChangeProperties")
        if not isinstance(self.boundaries, ThermalBoundarySet):
            raise TypeError("boundaries must be a ThermalBoundarySet")
        if self.scheme not in ("enthalpy_fv", "enthalpy_d2q5"):
            raise ValueError("scheme must be 'enthalpy_fv' or 'enthalpy_d2q5'")
        if self.moving_body_scheme != "body_ale":
            raise ValueError("moving_body_scheme must be 'body_ale'")
        if self.water_buoyancy_model != "linear":
            raise ValueError("water_buoyancy_model must be 'linear'")
        if not isinstance(self.advection_enabled, bool):
            raise ValueError("advection_enabled must be a boolean")
        if not isinstance(self.water_air_interface_adiabatic, bool):
            raise ValueError("water_air_interface_adiabatic must be a boolean")
        if (
            isinstance(self.update_interval_lbm_steps, bool)
            or int(self.update_interval_lbm_steps) != self.update_interval_lbm_steps
            or int(self.update_interval_lbm_steps) < 1
        ):
            raise ValueError("update_interval_lbm_steps must be a positive integer")
        object.__setattr__(
            self, "update_interval_lbm_steps", int(self.update_interval_lbm_steps)
        )
        if (
            self.moving_body_scheme == "body_ale"
            and not self.water_air_interface_adiabatic
        ):
            raise ValueError(
                "body_ale currently requires an adiabatic water/air thermal interface"
            )
        threshold = _finite("solid_liquid_threshold", self.solid_liquid_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("solid_liquid_threshold must be in (0, 1)")
        object.__setattr__(self, "solid_liquid_threshold", threshold)

        if self.buoyancy_reference_temperature_c is None:
            object.__setattr__(
                self,
                "buoyancy_reference_temperature_c",
                _finite(
                    "initial_water_temperature_c",
                    self.initial_water_temperature_c,
                ),
            )

        for name in (
            "initial_water_temperature_c",
            "initial_ice_temperature_c",
            "initial_air_temperature_c",
            "buoyancy_reference_temperature_c",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

        melting = self.properties.melting_temperature_c
        if self.initial_water_temperature_c < melting:
            raise ValueError(
                "initial_water_temperature_c must not be below melting_temperature_c"
            )
        if self.initial_ice_temperature_c > melting:
            raise ValueError(
                "initial_ice_temperature_c must not exceed melting_temperature_c"
            )

        fourier = _positive("max_fourier_number", self.max_fourier_number)
        # A half-cell Dirichlet boundary doubles that face coefficient.  At a
        # square-grid corner this produces six diffusion contributions.
        if fourier > 1.0 / 6.0:
            raise ValueError("max_fourier_number must not exceed 1/6")
        object.__setattr__(self, "max_fourier_number", fourier)

        courant = _positive("max_courant_number", self.max_courant_number)
        if courant > 1.0:
            raise ValueError("max_courant_number must not exceed one")
        object.__setattr__(self, "max_courant_number", courant)
        if (
            isinstance(self.max_substeps_per_update, bool)
            or int(self.max_substeps_per_update) != self.max_substeps_per_update
            or int(self.max_substeps_per_update) < 1
        ):
            raise ValueError("max_substeps_per_update must be a positive integer")
        object.__setattr__(
            self, "max_substeps_per_update", int(self.max_substeps_per_update)
        )
        object.__setattr__(
            self,
            "thermal_expansion_water_1_k",
            _non_negative(
                "thermal_expansion_water_1_k",
                self.thermal_expansion_water_1_k,
            ),
        )


@dataclass(frozen=True, slots=True)
class LatticeScales:
    """Physical units represented by one lattice cell and one LBM step."""

    dx_m: float
    dt_s: float
    reference_density_kg_m3: float
    reference_lattice_velocity: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "dx_m",
            "dt_s",
            "reference_density_kg_m3",
            "reference_lattice_velocity",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))

    @property
    def velocity_scale_m_s(self) -> float:
        """Physical m/s represented by one lattice cell per step."""

        return self.dx_m / self.dt_s

    @property
    def reference_velocity_m_s(self) -> float:
        return self.reference_lattice_velocity * self.velocity_scale_m_s

    @classmethod
    def from_reference_velocity(
        cls,
        *,
        dx_m: float,
        reference_velocity_m_s: float,
        reference_density_kg_m3: float,
        reference_lattice_velocity: float = 0.1,
    ) -> "LatticeScales":
        """Map a fixed physical reference speed to a lattice speed."""

        spacing = _positive("dx_m", dx_m)
        reference_velocity = _positive("reference_velocity_m_s", reference_velocity_m_s)
        lattice_velocity = _positive(
            "reference_lattice_velocity", reference_lattice_velocity
        )
        dt_s = lattice_velocity * spacing / reference_velocity
        return cls(
            dx_m=spacing,
            dt_s=dt_s,
            reference_density_kg_m3=reference_density_kg_m3,
            reference_lattice_velocity=lattice_velocity,
        )

    @classmethod
    def from_iceflow_config(cls, config: Any) -> "LatticeScales":
        """Build scales from an IceFlowConfig-like object without importing it."""

        try:
            return cls.from_reference_velocity(
                dx_m=config.dx,
                reference_velocity_m_s=config.reference_velocity,
                reference_density_kg_m3=config.rho_water,
            )
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "config must provide dx, reference_velocity, and rho_water"
            ) from exc


def _default_thermal_config() -> ThermalConfig:
    """Build the insulated-bath defaults without importing Taichi eagerly."""

    return ThermalConfig(
        boundaries=ThermalBoundarySet(),
        initial_water_temperature_c=90.0,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=20.0,
        update_interval_lbm_steps=8,
        buoyancy_reference_temperature_c=90.0,
    )


@dataclass(frozen=True, slots=True)
class IceBodyConfig:
    """One ice rectangle, in lattice cells, radians and cells per LBM step.

    IDs are stable indices in ``IceFlowConfig.ice_bodies``. Temperature defaults
    to the common thermal configuration. Bodies melt independently; no fracture
    or welding is performed.
    """

    size_cells: tuple[int, int]
    center_cells: tuple[float, float]
    angle_rad: float = 0.0
    velocity_lattice: tuple[float, float] = (0.0, 0.0)
    angular_velocity_lattice: float = 0.0
    temperature_c: float | None = None

    def __post_init__(self):
        if len(self.size_cells) != 2 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2
            for n in self.size_cells
        ):
            raise ValueError("size_cells must contain two integers of at least two")
        object.__setattr__(self, "size_cells", tuple(self.size_cells))
        for name in ("center_cells", "velocity_lattice"):
            values = getattr(self, name)
            if len(values) != 2:
                raise ValueError(f"{name} must contain two values")
            object.__setattr__(self, name, tuple(_finite(name, v) for v in values))
        for name in ("angle_rad", "angular_velocity_lattice"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.temperature_c is not None:
            object.__setattr__(
                self, "temperature_c", _finite("temperature_c", self.temperature_c)
            )


def _rectangles_overlap(a: IceBodyConfig, b: IceBodyConfig) -> bool:
    """Separating-axis check on the two initially oriented rectangles."""
    axes_a = (
        (math.cos(a.angle_rad), math.sin(a.angle_rad)),
        (-math.sin(a.angle_rad), math.cos(a.angle_rad)),
    )
    axes_b = (
        (math.cos(b.angle_rad), math.sin(b.angle_rad)),
        (-math.sin(b.angle_rad), math.cos(b.angle_rad)),
    )
    delta = tuple(b.center_cells[d] - a.center_cells[d] for d in range(2))
    for axis in (*axes_a, *axes_b):
        radius = sum(
            0.5 * body.size_cells[d] * abs(sum(axis[k] * axes[d][k] for k in range(2)))
            for body, axes in ((a, axes_a), (b, axes_b))
            for d in range(2)
        )
        if abs(sum(delta[d] * axis[d] for d in range(2))) >= radius - 1e-9:
            return False
    return True


@dataclass(slots=True)
class IceFlowConfig:
    """Validated inputs for one or more independently melting body-ALE ice blocks."""

    resolution: tuple[int, int] = (200, 400)
    dx: float = 1.25e-4
    # Fixed physical speed represented by 0.1 lattice cells per LBM step.
    reference_velocity: float = DEFAULT_REFERENCE_VELOCITY_M_S

    # Physical two-phase parameters.
    rho_water: float = 1000.0
    rho_air: float = 1.25
    rho_ice: float = 917.0
    # Optional constant kinematic viscosities [m^2/s]. None uses molecular
    # mu(T)/rho: local water temperature, prescribed (untransported) air temperature.
    viscosity_water: float | None = None
    viscosity_air: float | None = None
    sigma: float = 0.0
    gravity: tuple[float, float] = (0.0, -9.8)
    interface_width: float = 5.0
    mobility: float = 0.1
    # Gradient in lattice units used to regularize the phase compression
    # normal. Zero restores the original normalization.
    phase_normal_epsilon: float = 1.0e-3
    phase_gradient_scheme: Literal["legacy", "connected_ls"] = "legacy"
    phase_warmup_steps: int = 500
    thermal: ThermalConfig = field(default_factory=_default_thermal_config)
    # Build a frozen hydrostatic reference (pool or air atmosphere), initialize
    # the fluid at rest with any water-column pressure residual, and
    # complete dynamic momentum exchange with the matching equilibrium
    # traction.  This is one well-balanced formulation, not an independently
    # added Archimedes force.  No sub-grid quadrature is used.
    well_balanced_hydrostatics: bool = True
    # Liang et al.'s pressure--momentum distribution is used for the
    # large-density-ratio phase-field flow.  Material density remains rho(phi);
    # the populations carry rho*u and pressure rather than imposing the
    # isothermal relation p=cs^2*rho across the air--water jump.
    momentum_collision_model: Literal["rm_mrt", "cm_mrt", "cm_hrr"] = "cm_mrt"
    # CM trace and 22 relax together to damp pressure waves without changing
    # shear viscosity. Resolve the omitted value to 0.25 for CM, or 1 for RM
    # so selecting RM alone keeps its original collision. One disables damping.
    cm_acoustic_relaxation_rate: float | None = None
    # Experimental, independently selectable stability candidates. HRR blends
    # kinetic deviatoric stress (1) with a finite-difference reconstruction (0).
    # The specified molecular viscosity and interface width are unchanged;
    # HRR's finite-wavelength truncation damping is measured in the benchmark.
    hrr_kinetic_blend: float = 0.0
    phase_momentum_coupling: Literal["legacy", "consistent_flux"] = "legacy"
    exposed_node_refill: Literal["equilibrium", "non_equilibrium"] = "equilibrium"
    # Local numerical-source guards, independent of viscosity and phase width.
    density_gradient_source_limit: float = 0.0
    phase_flux_acceleration_limit: float = 0.0
    refill_moment_projection: bool = False
    gas_state_repair: bool = False

    # Falling-ice geometry in lattice-cell coordinates.
    water_width_fraction: float = 0.9875
    water_height_fraction: float = 0.60125
    # Additional full-width water layer, measured above the inner floor.
    # Zero preserves the single rectangular water region.
    floor_water_depth_cells: int = 0
    ice_width_fraction: float = 0.3225
    ice_height_fraction: float = 0.16125
    ice_base_y_cells: float = 254.6672141068609
    boundary_cells: int = 3

    # Initial state of the freely moving rigid ice rectangle.
    ice_initial_velocity: tuple[float, float] = (0.0, 0.0)
    ice_initial_angle: float = math.radians(5.0)
    ice_initial_angular_velocity: float = 0.0
    # None preserves the legacy single-rectangle inputs. Explicit bodies take
    # precedence over those inputs, and may have different sizes and temperatures.
    ice_bodies: tuple[IceBodyConfig, ...] | None = None
    body_contact_iterations: int = 32
    # Kept in serialized metadata for compatibility; alternate values are no
    # longer accepted by the specialized solver.
    ice_fixed: bool = False
    rigid_boundary_scheme: Literal["unified"] = "unified"

    # Hydrodynamic drag is resolved by cut-link momentum exchange.  Damping
    # remains an optional, explicitly configured numerical model for the
    # rigid-body state; both defaults disable this additional damping.
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    # Wet-cell threshold for thermal transport and its CFL checks; this does
    # not clip the fluid phase or its distributions.
    water_phase_cutoff: float = 1.0e-3

    output_dir: str = "iceflow2d_v3/outputs/iceflow2d_v3/coupled_falling_ice_melting_2d"
    show_gui: bool = False
    save_npz: bool = False

    def __post_init__(self) -> None:
        if self.ice_bodies is not None:
            self.ice_bodies = tuple(self.ice_bodies)
            if not self.ice_bodies or not all(
                isinstance(b, IceBodyConfig) for b in self.ice_bodies
            ):
                raise ValueError("ice_bodies must contain at least one IceBodyConfig")
        # Store validated numeric values as numbers: accepting float-like inputs
        # but retaining strings used to fail later in geometry and CUDA setup.
        for descriptor in fields(self):
            if descriptor.type == "float":
                setattr(
                    self,
                    descriptor.name,
                    _finite(descriptor.name, getattr(self, descriptor.name)),
                )
        if self.rigid_boundary_scheme != "unified":
            raise ValueError("rigid_boundary_scheme must be 'unified'")
        if self.momentum_collision_model not in ("rm_mrt", "cm_mrt", "cm_hrr"):
            raise ValueError(
                "momentum_collision_model must be 'rm_mrt', 'cm_mrt' or 'cm_hrr'"
            )
        if self.cm_acoustic_relaxation_rate is None:
            self.cm_acoustic_relaxation_rate = (
                1.0 if self.momentum_collision_model == "rm_mrt" else 0.25
            )
        self.cm_acoustic_relaxation_rate = _finite(
            "cm_acoustic_relaxation_rate", self.cm_acoustic_relaxation_rate
        )
        if not 0.0 < self.cm_acoustic_relaxation_rate < 2.0:
            raise ValueError("cm_acoustic_relaxation_rate must be in (0, 2)")
        if (
            self.momentum_collision_model == "rm_mrt"
            and self.cm_acoustic_relaxation_rate != 1.0
        ):
            raise ValueError(
                "cm_acoustic_relaxation_rate != 1 requires momentum_collision_model='cm_mrt' or 'cm_hrr'"
            )
        if not 0.0 <= self.hrr_kinetic_blend <= 1.0:
            raise ValueError("hrr_kinetic_blend must be in [0, 1]")
        for name in ("density_gradient_source_limit", "phase_flux_acceleration_limit"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.refill_moment_projection, bool):
            raise ValueError("refill_moment_projection must be a boolean")
        if not isinstance(self.gas_state_repair, bool):
            raise ValueError("gas_state_repair must be a boolean")
        if self.gas_state_repair and (
            self.phase_momentum_coupling != "consistent_flux"
            or self.phase_gradient_scheme != "connected_ls"
        ):
            raise ValueError("gas_state_repair requires consistent_flux and connected_ls")
        if self.phase_momentum_coupling not in ("legacy", "consistent_flux"):
            raise ValueError(
                "phase_momentum_coupling must be 'legacy' or 'consistent_flux'"
            )
        if self.exposed_node_refill not in ("equilibrium", "non_equilibrium"):
            raise ValueError(
                "exposed_node_refill must be 'equilibrium' or 'non_equilibrium'"
            )
        if self.phase_gradient_scheme not in ("legacy", "connected_ls"):
            raise ValueError("phase_gradient_scheme must be 'legacy' or 'connected_ls'")
        if (
            self.refill_moment_projection
            and self.phase_gradient_scheme != "connected_ls"
        ):
            raise ValueError(
                "refill_moment_projection requires phase_gradient_scheme='connected_ls'"
            )

        try:
            nx, ny = self.resolution
        except (TypeError, ValueError) as exc:
            raise ValueError("resolution must contain exactly two integers") from exc
        if (
            isinstance(nx, bool)
            or isinstance(ny, bool)
            or int(nx) != nx
            or int(ny) != ny
        ):
            raise ValueError("resolution must contain exactly two integers")
        nx, ny = int(nx), int(ny)
        self.resolution = (nx, ny)
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution components must be positive")
        if (
            isinstance(self.boundary_cells, bool)
            or int(self.boundary_cells) != self.boundary_cells
        ):
            raise ValueError("boundary_cells must be an integer")
        self.boundary_cells = int(self.boundary_cells)
        if self.boundary_cells < 1:
            raise ValueError("boundary_cells must be at least one")
        if nx <= 2 * self.boundary_cells + 4 or ny <= 2 * self.boundary_cells + 4:
            raise ValueError(
                "resolution is too small for the requested boundary thickness"
            )
        if (
            isinstance(self.phase_warmup_steps, bool)
            or int(self.phase_warmup_steps) != self.phase_warmup_steps
        ):
            raise ValueError("phase_warmup_steps must be an integer")
        self.phase_warmup_steps = int(self.phase_warmup_steps)
        if self.phase_warmup_steps < 0:
            raise ValueError("phase_warmup_steps must be non-negative")
        if self.well_balanced_hydrostatics is not True:
            raise ValueError("well_balanced_hydrostatics must be True")
        if not isinstance(self.thermal, ThermalConfig):
            raise TypeError("thermal must be a ThermalConfig")

        _positive("dx", self.dx)
        self.reference_velocity = _positive(
            "reference_velocity", self.reference_velocity
        )
        _positive("rho_water", self.rho_water)
        _positive("rho_air", self.rho_air)
        _positive("rho_ice", self.rho_ice)
        for name in ("viscosity_water", "viscosity_air"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _positive(name, value))
        if _finite("sigma", self.sigma) < 0.0:
            raise ValueError("sigma must be non-negative")
        _positive("interface_width", self.interface_width)
        _positive("mobility", self.mobility)
        _non_negative("phase_normal_epsilon", self.phase_normal_epsilon)
        if len(self.gravity) != 2:
            raise ValueError("gravity must contain exactly two components")
        self.gravity = tuple(
            _finite(f"gravity[{index}]", component)
            for index, component in enumerate(self.gravity)
        )

        for name in (
            "water_width_fraction",
            "water_height_fraction",
            "ice_width_fraction",
            "ice_height_fraction",
        ):
            value = _finite(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        base_y = _finite("ice_base_y_cells", self.ice_base_y_cells)
        if base_y < self.boundary_cells:
            raise ValueError("ice_base_y_cells must not overlap the bottom wall")
        floor_depth = _non_negative(
            "floor_water_depth_cells", self.floor_water_depth_cells
        )
        if int(floor_depth) != floor_depth:
            raise ValueError("floor_water_depth_cells must be an integer")
        self.floor_water_depth_cells = int(floor_depth)
        if self.floor_water_height > self.water_height:
            raise ValueError("floor water must not exceed the water-column height")

        if len(self.ice_initial_velocity) != 2:
            raise ValueError("ice_initial_velocity must contain exactly two components")
        initial_velocity = tuple(
            _finite(f"ice_initial_velocity[{index}]", component)
            for index, component in enumerate(self.ice_initial_velocity)
        )
        self.ice_initial_velocity = initial_velocity
        _finite("ice_initial_angle", self.ice_initial_angle)
        _finite("ice_initial_angular_velocity", self.ice_initial_angular_velocity)
        if self.ice_fixed is not False:
            raise ValueError("ice_fixed must be False")

        for name in ("linear_damping", "angular_damping"):
            value = _finite(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        cutoff = _finite("water_phase_cutoff", self.water_phase_cutoff)
        if not 0.0 < cutoff < 0.5:
            raise ValueError("water_phase_cutoff must be in (0, 0.5)")

        if (
            self.water_width <= self.boundary_cells
            or self.water_height <= self.boundary_cells
        ):
            raise ValueError("water block must contain at least one interior cell")
        if (
            self.water_width > nx - self.boundary_cells
            or self.water_height > ny - self.boundary_cells
        ):
            raise ValueError("water block must fit inside the container walls")
        if self.ice_width <= 1 or self.ice_height <= 1:
            raise ValueError(
                "ice rectangle must be at least two cells in each direction"
            )
        if self.well_balanced_hydrostatics:
            if abs(float(self.gravity[0])) > 1.0e-14:
                raise ValueError(
                    "the row-wise hydrostatic reference requires vertical gravity"
                )
        if (
            isinstance(self.body_contact_iterations, bool)
            or not isinstance(self.body_contact_iterations, int)
            or self.body_contact_iterations < 1
        ):
            raise ValueError("body_contact_iterations must be a positive integer")
        lower_contact = float(self.boundary_cells)
        upper_x_contact = float(nx - self.boundary_cells)
        upper_y_contact = float(ny - self.boundary_cells)
        for index, body in enumerate(self.bodies):
            c, s = abs(math.cos(body.angle_rad)), abs(math.sin(body.angle_rad))
            width, height = body.size_cells
            extent_x, extent_y = (
                (c * width + s * height) * 0.5,
                (s * width + c * height) * 0.5,
            )
            cx, cy = body.center_cells
            if (
                cx - extent_x < lower_contact - 1e-9
                or cx + extent_x > upper_x_contact + 1e-9
            ):
                raise ValueError("initial ice rectangle overlaps a side wall")
            if (
                cy - extent_y < lower_contact - 1e-9
                or cy + extent_y > upper_y_contact + 1e-9
            ):
                raise ValueError(
                    "initial ice rectangle overlaps the bottom or top wall"
                )
            if (
                body.temperature_c is not None
                and body.temperature_c > self.thermal.properties.melting_temperature_c
            ):
                raise ValueError(
                    "ice body temperature must not exceed melting temperature"
                )
            if any(_rectangles_overlap(body, other) for other in self.bodies[:index]):
                raise ValueError("initial ice bodies overlap")

        if self.thermal.moving_body_scheme != "body_ale":
            raise ValueError("thermal moving_body_scheme must be 'body_ale'")
        if not self.thermal.water_air_interface_adiabatic:
            raise ValueError("thermal water/air interface must be adiabatic")
        # MRT collision requires finite tau > 0.5.  Large physical
        # viscosities can legitimately map to tau > 2 when a deliberately low
        # fixed reference velocity is used, so no arbitrary upper bound on
        # tau is imposed here.
        water_temperatures = [
            self.thermal.initial_water_temperature_c,
            self.thermal.properties.melting_temperature_c,
        ]
        water_temperatures.extend(
            boundary.value
            for side in ("left", "right", "bottom", "top")
            if (boundary := getattr(self.thermal.boundaries, side)).kind == "dirichlet"
        )
        viscosities = [
            ("viscosity_water", self.water_viscosity_at(temperature))
            for temperature in water_temperatures
        ]
        viscosities.append(
            (
                "viscosity_air",
                self.air_viscosity_at(self.thermal.initial_air_temperature_c),
            )
        )
        for name, viscosity in viscosities:
            nu_lattice = (
                viscosity * 0.1 / (float(self.dx) * float(self.reference_velocity))
            )
            tau = 0.5 + 3.0 * nu_lattice
            if not math.isfinite(tau) or tau <= 0.5:
                raise ValueError(
                    f"{name} gives an unstable lattice relaxation time ({tau})"
                )
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

    @property
    def nx(self) -> int:
        return int(self.resolution[0])

    @property
    def ny(self) -> int:
        return int(self.resolution[1])

    @property
    def water_width(self) -> int:
        return max(self.boundary_cells + 2, int(self.nx * self.water_width_fraction))

    @property
    def water_height(self) -> int:
        return max(self.boundary_cells + 2, int(self.ny * self.water_height_fraction))

    @property
    def floor_water_height(self) -> int:
        return self.boundary_cells + self.floor_water_depth_cells

    @property
    def ice_width(self) -> int:
        if self.ice_bodies:
            return self.ice_bodies[0].size_cells[0]
        return max(2, int(self.nx * self.ice_width_fraction))

    @property
    def ice_height(self) -> int:
        if self.ice_bodies:
            return self.ice_bodies[0].size_cells[1]
        return max(2, int(self.ny * self.ice_height_fraction))

    @property
    def ice_initial_center(self) -> tuple[float, float]:
        if self.ice_bodies:
            return self.ice_bodies[0].center_cells
        # Align the analytic rectangle with cell centers and avoid an extra
        # raster column when a custom resolution produces an odd ice width.
        left_cell = (self.nx - self.ice_width) // 2
        return (
            left_cell + 0.5 * self.ice_width,
            float(self.ice_base_y_cells) + 0.5 * self.ice_height,
        )

    @property
    def ice_mass_lattice(self) -> float:
        return float(self.rho_ice / self.rho_water * self.ice_width * self.ice_height)

    @property
    def ice_inertia_lattice(self) -> float:
        return self.ice_mass_lattice * (self.ice_width**2 + self.ice_height**2) / 12.0

    @property
    def bodies(self) -> tuple[IceBodyConfig, ...]:
        if self.ice_bodies is not None:
            return self.ice_bodies
        return (
            IceBodyConfig(
                size_cells=(self.ice_width, self.ice_height),
                center_cells=self.ice_initial_center,
                angle_rad=self.ice_initial_angle,
                velocity_lattice=self.ice_initial_velocity,
                angular_velocity_lattice=self.ice_initial_angular_velocity,
                temperature_c=self.thermal.initial_ice_temperature_c,
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def water_viscosity_at(self, temperature_c: float) -> float:
        """Physical kinematic viscosity [m^2/s], or the explicit constant override."""

        if self.viscosity_water is not None:
            return self.viscosity_water
        return water_dynamic_viscosity_pa_s(temperature_c) / self.rho_water

    def air_viscosity_at(self, temperature_c: float) -> float:
        """Physical kinematic viscosity [m^2/s], or the explicit constant override."""

        if self.viscosity_air is not None:
            return self.viscosity_air
        return air_dynamic_viscosity_pa_s(temperature_c) / self.rho_air


def create_iceflow_config(**overrides) -> IceFlowConfig:
    valid_names = {item.name for item in fields(IceFlowConfig)}
    for key in overrides:
        if key not in valid_names:
            raise TypeError(f"Unknown IceFlowConfig option: {key}")
    return IceFlowConfig(**overrides)
