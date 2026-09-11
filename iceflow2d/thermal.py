"""Thermal configuration and body-ALE phase-change kernels.

:class:`MovingBodyThermal2D` stores eroding ice in a material grid and water
  volume/sensible energy in the world grid.  Paired volume--energy advection,
  pose remapping, interface heat, and melt sources preserve the relevant
  extensive sums before the parent solver updates rigid mass and momentum.
  Importing this module does not initialize Taichi and remains valid when
  Taichi is absent.

The device discretization stores physical volumetric enthalpy in J/m^3 on the
same ``(nx, ny)`` cell layout as the LBM solver.  One heat/enthalpy flux is
computed for every grid face and is then differenced by its two neighbouring
cells.  Internal-face contributions therefore cancel exactly at the discrete
level.  Diffusion uses harmonic face conductivity and advection uses a
first-order upwind sensible-enthalpy flux.  The sensible form is essential
when ice and water use their physical, unequal densities: their latent-energy
reference offsets differ even when both sides of a face are at the same
temperature, and those offsets must not create a spurious advective flux.

Container thermal boundary conditions are applied where a non-wall cell is
adjacent to a ``wall`` cell, rather than at the outermost array index.  This is
important because IceFlow2D normally represents a container wall with several
layers of lattice cells.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

try:  # Configuration objects remain importable without the GPU dependency.
    import taichi as ti
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    ti = None  # type: ignore[assignment]


ThermalBoundaryKind = Literal["adiabatic", "dirichlet"]
WaterBuoyancyModel = Literal["linear"]
MovingBodyThermalScheme = Literal["body_ale"]

_ADIABATIC = 0
_DIRICHLET = 1
_BOUNDARY_CODE = {
    "adiabatic": _ADIABATIC,
    "dirichlet": _DIRICHLET,
}


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(name: str, value: float) -> float:
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = _finite(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True, slots=True)
class PhaseChangeProperties:
    """Specific heat, conductivity, and latent heat for air/water/ice.

    Densities are intentionally not duplicated here.  A coupled solver gets
    them from the hydrodynamic configuration so that momentum and energy
    cannot silently use different material densities.
    """

    melting_temperature_c: float = 0.0
    specific_heat_water_j_kg_k: float = 4186.0
    specific_heat_ice_j_kg_k: float = 2100.0
    specific_heat_air_j_kg_k: float = 1005.0
    conductivity_water_w_m_k: float = 0.60
    conductivity_ice_w_m_k: float = 2.20
    conductivity_air_w_m_k: float = 0.026
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
            "specific_heat_air_j_kg_k",
            "conductivity_water_w_m_k",
            "conductivity_ice_w_m_k",
            "conductivity_air_w_m_k",
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
    ice melts; melt water detaches and cannot refreeze onto the single rigid
    remnant.
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
    freshwater_density_max_temperature_c: float = 4.0
    freshwater_density_quadratic_coefficient_1_k2: float = 8.0e-6
    scheme: Literal["enthalpy_fv"] = "enthalpy_fv"
    moving_body_scheme: MovingBodyThermalScheme = "body_ale"

    def __post_init__(self) -> None:
        if not isinstance(self.properties, PhaseChangeProperties):
            raise TypeError("properties must be a PhaseChangeProperties")
        if not isinstance(self.boundaries, ThermalBoundarySet):
            raise TypeError("boundaries must be a ThermalBoundarySet")
        if self.scheme != "enthalpy_fv":
            raise ValueError("scheme must be 'enthalpy_fv'")
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
            "freshwater_density_max_temperature_c",
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
        object.__setattr__(
            self,
            "freshwater_density_quadratic_coefficient_1_k2",
            _non_negative(
                "freshwater_density_quadratic_coefficient_1_k2",
                self.freshwater_density_quadratic_coefficient_1_k2,
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


@dataclass(frozen=True, slots=True)
class MovingBodyThermalTotals:
    """Mass and reference-energy reductions for moving-body phase change."""

    initial_body_mass_kg_m: float
    solid_body_mass_kg_m: float
    melted_mass_kg_m: float
    water_mass_kg_m: float
    total_mass_kg_m: float
    body_sensible_energy_j_m: float
    water_sensible_energy_j_m: float
    latent_energy_j_m: float
    total_energy_j_m: float


if ti is not None:

    @ti.data_oriented
    class MovingBodyThermal2D:
        """Body-frame ice enthalpy coupled to world-frame water sensible heat.

        The material grid is immutable under rigid translation and rotation.
        Only ``body_solid_mass`` and ``body_sensible_energy`` evolve there;
        world rasterization is a read-only pose transform owned by the parent
        solver.  Water stores cell volume and sensible energy as extensive
        quantities.  Every ice--water face heat transfer is removed from the
        water cell and atomically added to exactly one material cell before
        phase change is applied.

        This first moving-body implementation is intentionally one-way: melt
        water detaches from the rigid body and cannot subsequently refreeze
        onto it.  Water/air heat transfer is adiabatic, as required by
        ``ThermalConfig(moving_body_scheme="body_ale")``.
        """

        def __init__(
            self,
            nx: int,
            ny: int,
            body_nx: int,
            body_ny: int,
            config: ThermalConfig,
            scales: LatticeScales,
            *,
            density_water_kg_m3: float,
            density_ice_kg_m3: float,
            water_phase_cutoff: float = 0.0,
        ):
            for name, value, minimum in (
                ("nx", nx, 2),
                ("ny", ny, 2),
                ("body_nx", body_nx, 1),
                ("body_ny", body_ny, 1),
            ):
                if (
                    isinstance(value, bool)
                    or int(value) != value
                    or int(value) < minimum
                ):
                    raise ValueError(f"{name} must be an integer of at least {minimum}")
            if not isinstance(config, ThermalConfig):
                raise TypeError("config must be a ThermalConfig")
            if not isinstance(scales, LatticeScales):
                raise TypeError("scales must be a LatticeScales")
            if config.moving_body_scheme != "body_ale":
                raise ValueError(
                    "MovingBodyThermal2D requires moving_body_scheme='body_ale'"
                )

            self.nx = int(nx)
            self.ny = int(ny)
            self.body_nx = int(body_nx)
            self.body_ny = int(body_ny)
            self.config = config
            self.scales = scales
            self.steps = 0
            self.time_s = 0.0
            # Filled by ``IceFlow2D`` after construction.  Keeping the
            # reference optional lets this component remain importable and
            # usable for its material kernels without owning world geometry.
            self._rasterize_world_callback = None
            self._rho_water = _positive("density_water_kg_m3", density_water_kg_m3)
            self._rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
            self._water_phase_cutoff = _non_negative(
                "water_phase_cutoff", water_phase_cutoff
            )
            if self._water_phase_cutoff >= 0.5:
                raise ValueError("water_phase_cutoff must be below 0.5")
            props = config.properties
            self._melting = float(props.melting_temperature_c)
            self._cp_water = float(props.specific_heat_water_j_kg_k)
            self._cp_ice = float(props.specific_heat_ice_j_kg_k)
            self._k_water = float(props.conductivity_water_w_m_k)
            self._k_ice = float(props.conductivity_ice_w_m_k)
            self._latent = float(props.latent_heat_j_kg)
            self._dx = float(scales.dx_m)
            self._cell_area = self._dx * self._dx
            self._velocity_scale = float(scales.velocity_scale_m_s)
            self._threshold = float(config.solid_liquid_threshold)
            self._advection_enabled = bool(config.advection_enabled)

            boundaries = config.boundaries
            self._left_kind = _BOUNDARY_CODE[boundaries.left.kind]
            self._right_kind = _BOUNDARY_CODE[boundaries.right.kind]
            self._bottom_kind = _BOUNDARY_CODE[boundaries.bottom.kind]
            self._top_kind = _BOUNDARY_CODE[boundaries.top.kind]
            self._left_value = float(boundaries.left.value)
            self._right_value = float(boundaries.right.value)
            self._bottom_value = float(boundaries.bottom.value)
            self._top_value = float(boundaries.top.value)

            body_shape = (self.body_nx, self.body_ny)
            world_shape = (self.nx, self.ny)
            self.body_initial_mass = ti.field(ti.f64, shape=body_shape)
            self.body_solid_mass = ti.field(ti.f64, shape=body_shape)
            self.body_sensible_energy = ti.field(ti.f64, shape=body_shape)
            self.body_temperature = ti.field(ti.f64, shape=body_shape)
            self.body_solid_fraction = ti.field(ti.f32, shape=body_shape)
            self._body_heat_delta = ti.field(ti.f64, shape=body_shape)
            self._body_flux_x = ti.field(ti.f64, shape=(self.body_nx + 1, self.body_ny))
            self._body_flux_y = ti.field(ti.f64, shape=(self.body_nx, self.body_ny + 1))
            self._body_melt_target = ti.field(ti.i32, shape=body_shape)
            self._body_melt_mass_step = ti.field(ti.f64, shape=body_shape)
            self._body_melt_volume_step = ti.field(ti.f64, shape=body_shape)
            self._body_melt_sensible_step = ti.field(ti.f64, shape=body_shape)

            self.water_volume_m2 = ti.field(ti.f64, shape=world_shape)
            self.water_sensible_energy = ti.field(ti.f64, shape=world_shape)
            self.water_temperature = ti.field(ti.f64, shape=world_shape)
            self._water_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
            self._water_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
            self._water_volume_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
            self._water_volume_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
            # Conservative projection between the extensive thermal water
            # state and the current LBM water aperture.  The target is built
            # from ``phi`` but is capped by one physical cell area, so newly
            # wetted cells receive a thermal state and no cell can hold more
            # water than its geometric capacity.
            self._phase_target_volume = ti.field(ti.f64, shape=world_shape)
            self._phase_volume_before = ti.field(ti.f64, shape=())
            self._phase_energy_before = ti.field(ti.f64, shape=())
            self._phase_base_capacity = ti.field(ti.f64, shape=())
            self._phase_full_capacity = ti.field(ti.f64, shape=())
            self._phase_donor_volume = ti.field(ti.f64, shape=())
            self._phase_donor_energy = ti.field(ti.f64, shape=())
            self._phase_receiver_volume = ti.field(ti.f64, shape=())
            self._phase_volume_after = ti.field(ti.f64, shape=())
            self._phase_energy_after = ti.field(ti.f64, shape=())
            self._interface_heat_requested = ti.field(ti.f64, shape=world_shape)
            self.water_melt_mass_source = ti.field(ti.f64, shape=world_shape)
            # Read-only candidate set used while body-cell threads scatter
            # newly melted water.  It is frozen in a separate kernel so one
            # thread cannot make a previously dry cell eligible while another
            # thread is still selecting its fallback target.
            self._melt_injection_eligible = ti.field(ti.i8, shape=world_shape)

            self.world_body_indicator = ti.field(ti.f32, shape=world_shape)
            self.world_body_indicator_prev = ti.field(ti.f32, shape=world_shape)
            # Rasterization produces one coverage value.  Keep the material
            # coverage and its interpolation data here; the parent solver
            # owns the single world SDF used by both rasterization and LBM.
            self.world_body_solid_fraction = self.world_body_indicator
            self.world_body_local_i = ti.field(ti.i32, shape=world_shape)
            self.world_body_local_j = ti.field(ti.i32, shape=world_shape)
            # Bilinear material contributors used by the thermal contact
            # operator.  Keeping these interpolation coordinates avoids
            # assigning heat to a zero-mass nearest cell after nonuniform
            # erosion.
            self.world_body_interp_base_i = ti.field(ti.i32, shape=world_shape)
            self.world_body_interp_base_j = ti.field(ti.i32, shape=world_shape)
            self.world_body_interp_fraction_x = ti.field(ti.f32, shape=world_shape)
            self.world_body_interp_fraction_y = ti.field(ti.f32, shape=world_shape)
            self.world_temperature_c = ti.field(ti.f64, shape=world_shape)
            self.enthalpy_j_m3 = ti.field(ti.f64, shape=world_shape)
            self.liquid_fraction = ti.field(ti.f32, shape=world_shape)
            self.ice_material = ti.field(ti.i8, shape=world_shape)
            # Stable public aliases used by the coupled example output layer.
            self.temperature_c = self.world_temperature_c

            self.boundary_power = ti.field(ti.f64, shape=())
            self.boundary_heat_input = ti.field(ti.f64, shape=())
            self.boundary_heat_input_j_m = self.boundary_heat_input
            self.ale_water_volume_residual_m2 = ti.field(ti.f64, shape=())
            self.ale_water_energy_residual_j_m = ti.field(ti.f64, shape=())
            self.phase_aperture_volume_residual_m2 = ti.field(ti.f64, shape=())
            self.phase_aperture_energy_residual_j_m = ti.field(ti.f64, shape=())
            self.phase_aperture_capacity_margin_m2 = ti.field(ti.f64, shape=())
            self._ale_removed_volume = ti.field(ti.f64, shape=())
            self._ale_removed_energy = ti.field(ti.f64, shape=())
            self._ale_release_weight = ti.field(ti.f64, shape=())
            self._ale_fallback_weight = ti.field(ti.f64, shape=())
            self._ale_assigned_volume = ti.field(ti.f64, shape=())
            self._ale_assigned_energy = ti.field(ti.f64, shape=())
            # A material cell normally injects melt into its paired interface
            # water cell.  If that local topology disappears within a thermal
            # substep, all four extensive sources enter this conservative
            # fallback pool instead of being silently discarded.
            self._unassigned_melt_mass = ti.field(ti.f64, shape=())
            self._unassigned_melt_volume = ti.field(ti.f64, shape=())
            self._unassigned_melt_energy = ti.field(ti.f64, shape=())
            self._melt_fallback_free_weight = ti.field(ti.f64, shape=())
            self._melt_fallback_wet_weight = ti.field(ti.f64, shape=())
            self._interval_body_melt_mass = ti.field(ti.f64, shape=())
            self._interval_water_melt_mass = ti.field(ti.f64, shape=())
            self.melt_injection_mass_residual_kg_m = ti.field(ti.f64, shape=())
            self._initial_body_mass_sum = ti.field(ti.f64, shape=())
            self._solid_body_mass_sum = ti.field(ti.f64, shape=())
            self._water_mass_sum = ti.field(ti.f64, shape=())
            self._body_sensible_sum = ti.field(ti.f64, shape=())
            self._water_sensible_sum = ti.field(ti.f64, shape=())
            alpha_water = self._k_water / (self._rho_water * self._cp_water)
            alpha_ice = self._k_ice / (self._rho_ice * self._cp_ice)
            self.maximum_diffusivity_m2_s = max(alpha_water, alpha_ice)
            self.maximum_diffusion_time_step_s = (
                config.max_fourier_number
                * self._dx
                * self._dx
                / self.maximum_diffusivity_m2_s
            )

        def maximum_advection_time_step_s(
            self, max_velocity_lattice_l1: float
        ) -> float:
            speed = _non_negative("max_velocity_lattice_l1", max_velocity_lattice_l1)
            if not self._advection_enabled or speed == 0.0:
                return math.inf
            return self.config.max_courant_number * self.scales.dt_s / speed

        def _validate_substep_count(
            self,
            substeps: int,
            *,
            process: str,
            time_step_s: float,
            max_velocity_lattice_l1: float | None = None,
        ) -> int:
            if substeps > self.config.max_substeps_per_update:
                detail = ""
                if max_velocity_lattice_l1 is not None:
                    detail = f", max_velocity_lattice_l1={max_velocity_lattice_l1!r}"
                raise RuntimeError(
                    "thermal stability requires "
                    f"{substeps} {process} substeps for one coupling update, "
                    "exceeding max_substeps_per_update="
                    f"{self.config.max_substeps_per_update}; "
                    f"time_step_s={time_step_s:.9g}{detail}. Reduce the "
                    "thermal update interval or the lattice velocity, and "
                    "inspect the LBM state before raising this safety limit."
                )
            return substeps

        def required_advection_substeps(
            self,
            time_step_s: float,
            *,
            max_velocity_lattice_l1: float | None = None,
        ) -> int:
            """Return the FAST upwind substeps required by the Courant bound."""

            dt = _positive("time_step_s", time_step_s)
            load = 0.0
            if max_velocity_lattice_l1 is not None:
                advection_limit = self.maximum_advection_time_step_s(
                    max_velocity_lattice_l1
                )
                if math.isfinite(advection_limit):
                    load = dt / advection_limit
            substeps = max(1, int(math.ceil(load - 1.0e-14)))
            return self._validate_substep_count(
                substeps,
                process="advection",
                time_step_s=dt,
                max_velocity_lattice_l1=max_velocity_lattice_l1,
            )

        def required_diffusion_substeps(self, time_step_s: float) -> int:
            """Return the SLOW conduction substeps required by the Fourier bound."""

            dt = _positive("time_step_s", time_step_s)
            load = dt / self.maximum_diffusion_time_step_s
            substeps = max(1, int(math.ceil(load - 1.0e-14)))
            return self._validate_substep_count(
                substeps,
                process="diffusion",
                time_step_s=dt,
            )

        def initialize(
            self,
            water_phase: Any,
            wall: Any,
            solid: Any,
            *,
            body_center: Any | None = None,
            body_angle: Any | None = None,
        ) -> None:
            """Initialize the material body and extensive world-water state."""

            # World rasterization is owned by ``IceFlow2D``.  Keep the pose
            # arguments for source compatibility with callers of the former
            # thermal-owned implementation, but deliberately do not use
            # them here: the parent solver performs the initial geometry
            # transaction after this state has been initialized.
            del body_center, body_angle

            self._initialize_body_state()
            self._initialize_water_state(water_phase, wall, solid)
            self._reset_diagnostics()
            self._recover_body_state()
            self._recover_water_state()
            self.steps = 0
            self.time_s = 0.0

        def rasterize_world(
            self,
            body_center: Any | None = None,
            body_angle: Any | None = None,
            wall: Any | None = None,
            *,
            resolve_contact: bool = False,
        ) -> None:
            """Delegate world rasterization to the owning simulator.

            The implementation used to live in this thermal component.  A
            narrow forwarding shim keeps older code that calls
            ``simulation.thermal.rasterize_world(...)`` working while the
            simulator owns fraction sampling, the shared SDF, and contact
            projection.
            The legacy pose and wall arguments are accepted only for source
            compatibility and are ignored; the simulator reads its own
            authoritative fields.
            ``resolve_contact`` is accepted for symmetry with the simulator
            entry point.  It defaults to ``False`` to preserve the former
            thermal-only rasterization semantics; the simulator entry point
            defaults to the complete geometry/contact transaction.
            """

            callback = getattr(self, "_rasterize_world_callback", None)
            if callback is None:
                raise RuntimeError(
                    "world rasterization is owned by IceFlow2D; no "
                    "rasterize-world callback has been registered"
            )
            if body_center is None and body_angle is None and wall is None:
                callback(resolve_contact=bool(resolve_contact))
                return None
            if body_center is None or body_angle is None or wall is None:
                raise TypeError(
                    "body_center, body_angle, and wall must be supplied together"
                )
            # The simulator now reads its own authoritative pose and wall
            # fields.  The legacy values are validated above for callers that
            # still pass them, then deliberately discarded.
            callback(resolve_contact=bool(resolve_contact))
            return None

        def advance_fast(
            self,
            time_step_s: float,
            velocity: Any,
            water_phase: Any,
            wall: Any,
            solid: Any,
            *,
            max_velocity_lattice_l1: float | None = None,
        ) -> int:
            """Advance one pose remap and water advection at the LBM rate."""

            dt = _positive("time_step_s", time_step_s)
            substeps = self.required_advection_substeps(
                dt, max_velocity_lattice_l1=max_velocity_lattice_l1
            )
            self._conservative_remap_water(water_phase, wall)
            # The conservative remap closes global V/S sums, but its release
            # weights may temporarily place that volume in a partial or still-
            # sharp cell.  Restore the current LBM aperture before selecting
            # the first upwind donor.
            self.synchronize_water_aperture(
                water_phase, wall, solid, refresh_derived=False
            )
            sub_dt = dt / substeps
            for substep_index in range(substeps):
                self._compute_water_advection_flux_x(velocity, water_phase, wall, solid)
                self._compute_water_advection_flux_y(velocity, water_phase, wall, solid)
                self._update_water_advection(sub_dt, wall)
                # The upwind volume flux is conservative globally, while an
                # individual partial cell can temporarily outrun its current
                # phase aperture.  Reconcile after every Courant substep so
                # the next face flux always sees an admissible capacity.
                if substep_index + 1 < substeps:
                    self.synchronize_water_aperture(
                        water_phase, wall, solid, refresh_derived=False
                    )
            return substeps

        def advance_slow(
            self,
            time_step_s: float,
            water_phase: Any,
            wall: Any,
            solid: Any,
            body_reference_origin: Any,
            body_angle: Any,
            *,
            rasterize_world_callback: Any | None = None,
        ) -> int:
            """Advance conduction, wall heat, phase change, and melt injection."""

            # ``IceFlow2D`` supplies the callback so that each thermal
            # substep sees the current eroded material raster.  Falling back
            # to the callback registered by the simulator keeps direct
            # callers source-compatible while leaving this component free of
            # rasterization kernels.
            if rasterize_world_callback is None:
                rasterize_world_callback = getattr(
                    self, "_rasterize_world_callback", None
                )

            dt = _positive("time_step_s", time_step_s)
            substeps = self.required_diffusion_substeps(dt)
            sub_dt = dt / substeps
            self._reset_interval_sources()
            for substep_index in range(substeps):
                self._recover_body_state()
                self._recover_water_state()
                self._reset_substep_sources()
                self._compute_body_flux_x()
                self._compute_body_flux_y()
                self._accumulate_body_conduction(sub_dt)
                self._reset_boundary_power()
                self._compute_water_conduction_flux_x(water_phase, wall, solid)
                self._compute_water_conduction_flux_y(water_phase, wall, solid)
                self._update_water_conduction(sub_dt, wall)
                self._accumulate_boundary_heat(sub_dt)
                self._recover_water_state()
                self._compute_interface_heat_requests(sub_dt, water_phase, wall, solid)
                self._apply_interface_heat(sub_dt, water_phase, wall, solid)
                self._apply_body_heat_and_phase_change()
                self._freeze_melt_injection_eligibility(water_phase, wall, solid)
                self._inject_melt_water(
                    body_reference_origin,
                    body_angle,
                )
                self._distribute_unassigned_melt(water_phase, wall, solid)
                # Restore the per-cell aperture bound before the next heat
                # substep; this keeps V and S admissible even when one
                # coupling interval requires N_sub > 1.
                self.synchronize_water_aperture(
                    water_phase, wall, solid, refresh_derived=False
                )
                if substep_index + 1 < substeps:
                    # A diffusion interval can melt enough material to alter
                    # the continuous contact aperture before its next
                    # substep.  Refresh the material-to-world weights at the
                    # fixed SLOW pose; the parent solver rebuilds the sharp
                    # LBM mask once after the complete interval.
                    self._recover_body_state()
                    if rasterize_world_callback is not None:
                        rasterize_world_callback(resolve_contact=False)
            self._reduce_melt_injection_residual()
            self._recover_body_state()
            self._recover_water_state()
            if rasterize_world_callback is not None:
                rasterize_world_callback(resolve_contact=False)
            self.steps += substeps
            self.time_s += dt
            return substeps

        def synchronize_water_aperture(
            self,
            water_phase: Any,
            wall: Any,
            solid: Any,
            *,
            refresh_derived: bool = True,
        ) -> None:
            """Conservatively align extensive water state with the LBM aperture.

            For legal water-side cells, the phase-weighted target is
            ``dx**2 * phi``.  If the thermal and phase totals differ between
            coupling updates, the target is scaled down or its remaining
            geometric capacity is filled proportionally.  Donor water and
            its sensible energy are pooled with one common weight, preserving
            both global extensive sums and a spatially uniform temperature.
            """

            self._measure_phase_aperture(water_phase, wall, solid)
            total_volume = float(self._phase_volume_before[None])
            full_capacity = float(self._phase_full_capacity[None])
            tolerance = 1.0e-12 * max(
                self._cell_area, abs(total_volume), abs(full_capacity)
            )
            if total_volume < -tolerance:
                raise RuntimeError(
                    "thermal water advection produced a negative total volume: "
                    f"{total_volume:.17g} m^2"
                )
            if total_volume > full_capacity + tolerance:
                raise RuntimeError(
                    "thermal water volume exceeds the current fluid aperture "
                    "capacity: "
                    f"volume={total_volume:.17g} m^2, "
                    f"capacity={full_capacity:.17g} m^2"
                )
            self._build_phase_aperture_targets(water_phase, wall, solid)
            self._measure_phase_aperture_transfer()
            self._apply_phase_aperture_transfer()
            self._finish_phase_aperture_transfer()
            if refresh_derived:
                # Public callers observe a self-consistent temperature and
                # display state immediately after changing V and S.  This is
                # deliberately not a pose rasterization: ALE history and the
                # pending-pose flag must remain untouched.
                self._recover_water_state()
                self._compose_world_temperature(wall)

        def mass_energy_totals(self) -> MovingBodyThermalTotals:
            """Synchronously reduce all extensive moving-body thermal fields."""

            self._reduce_totals()
            initial_body = float(self._initial_body_mass_sum[None])
            solid_body = float(self._solid_body_mass_sum[None])
            melted = initial_body - solid_body
            water_mass = float(self._water_mass_sum[None])
            body_sensible = float(self._body_sensible_sum[None])
            water_sensible = float(self._water_sensible_sum[None])
            latent = self._latent * melted
            return MovingBodyThermalTotals(
                initial_body_mass_kg_m=initial_body,
                solid_body_mass_kg_m=solid_body,
                melted_mass_kg_m=melted,
                water_mass_kg_m=water_mass,
                total_mass_kg_m=solid_body + water_mass,
                body_sensible_energy_j_m=body_sensible,
                water_sensible_energy_j_m=water_sensible,
                latent_energy_j_m=latent,
                total_energy_j_m=body_sensible + water_sensible + latent,
            )

        def total_enthalpy_j_m(self, wall: Any | None = None) -> float:
            """Return conserved body/water energy per unit out-of-plane depth."""

            del wall
            return self.mass_energy_totals().total_energy_j_m

        @ti.kernel
        def _initialize_body_state(self):
            cell_mass = ti.cast(ti.static(self._rho_ice * self._cell_area), ti.f64)
            cp_ice = ti.cast(ti.static(self._cp_ice), ti.f64)
            melting = ti.cast(ti.static(self._melting), ti.f64)
            initial_temperature = ti.cast(
                ti.static(float(self.config.initial_ice_temperature_c)), ti.f64
            )
            for i, j in self.body_initial_mass:
                mass = cell_mass
                self.body_initial_mass[i, j] = mass
                self.body_solid_mass[i, j] = mass
                self.body_sensible_energy[i, j] = (
                    mass * cp_ice * (initial_temperature - melting)
                )
                self.body_temperature[i, j] = initial_temperature
                self.body_solid_fraction[i, j] = 1.0

        @ti.kernel
        def _initialize_water_state(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            rho_cp = ti.cast(ti.static(self._rho_water * self._cp_water), ti.f64)
            melting = ti.cast(ti.static(self._melting), ti.f64)
            initial_temperature = ti.cast(
                ti.static(float(self.config.initial_water_temperature_c)), ti.f64
            )
            for i, j in self.water_volume_m2:
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                volume = ti.cast(0.0, ti.f64)
                if (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                ):
                    volume = area * ti.cast(phase, ti.f64)
                self.water_volume_m2[i, j] = volume
                self.water_sensible_energy[i, j] = (
                    rho_cp * volume * (initial_temperature - melting)
                )
                self.water_temperature[i, j] = initial_temperature

        @ti.kernel
        def _reset_diagnostics(self):
            self.boundary_power[None] = 0.0
            self.boundary_heat_input[None] = 0.0
            self.ale_water_volume_residual_m2[None] = 0.0
            self.ale_water_energy_residual_j_m[None] = 0.0
            self.phase_aperture_volume_residual_m2[None] = 0.0
            self.phase_aperture_energy_residual_j_m[None] = 0.0
            self.phase_aperture_capacity_margin_m2[None] = 0.0
            self.melt_injection_mass_residual_kg_m[None] = 0.0

        @ti.kernel
        def _reset_interval_sources(self):
            self._interval_body_melt_mass[None] = 0.0
            self._interval_water_melt_mass[None] = 0.0
            self.melt_injection_mass_residual_kg_m[None] = 0.0
            for i, j in self.water_melt_mass_source:
                self.water_melt_mass_source[i, j] = 0.0

        @ti.kernel
        def _reset_substep_sources(self):
            self._unassigned_melt_mass[None] = 0.0
            self._unassigned_melt_volume[None] = 0.0
            self._unassigned_melt_energy[None] = 0.0
            self._melt_fallback_free_weight[None] = 0.0
            self._melt_fallback_wet_weight[None] = 0.0
            for i, j in self._body_heat_delta:
                self._body_heat_delta[i, j] = 0.0
                self._body_melt_target[i, j] = -1
                self._body_melt_mass_step[i, j] = 0.0
                self._body_melt_volume_step[i, j] = 0.0
                self._body_melt_sensible_step[i, j] = 0.0
            for i, j in self._interface_heat_requested:
                self._interface_heat_requested[i, j] = 0.0

        @ti.kernel
        def _reset_boundary_power(self):
            self.boundary_power[None] = 0.0

        @ti.kernel
        def _recover_body_state(self):
            melting = ti.cast(ti.static(self._melting), ti.f64)
            cp_ice = ti.cast(ti.static(self._cp_ice), ti.f64)
            for i, j in self.body_initial_mass:
                initial_mass = self.body_initial_mass[i, j]
                solid_mass = ti.min(
                    initial_mass, ti.max(0.0, self.body_solid_mass[i, j])
                )
                self.body_solid_mass[i, j] = solid_mass
                fraction = ti.cast(0.0, ti.f64)
                temperature = ti.cast(melting, ti.f64)
                if initial_mass > 1.0e-30:
                    fraction = solid_mass / initial_mass
                if solid_mass > 0.0:
                    # The contact aperture is stored/rasterized in f32.  Keep
                    # a positive representational aperture after the physical
                    # fraction falls below the reliable interpolation range;
                    # the associated heat is still debited from water and
                    # pays the exact latent energy before mass reaches zero.
                    fraction = ti.max(fraction, 1.0e-30)
                if solid_mass > 1.0e-30:
                    temperature += self.body_sensible_energy[i, j] / (
                        solid_mass * cp_ice
                    )
                    temperature = ti.min(melting, temperature)
                self.body_solid_fraction[i, j] = ti.cast(fraction, ti.f32)
                self.body_temperature[i, j] = temperature

        @ti.kernel
        def _recover_water_state(self):
            melting = ti.cast(ti.static(self._melting), ti.f64)
            rho_cp = ti.cast(ti.static(self._rho_water * self._cp_water), ti.f64)
            air_temperature = ti.cast(
                ti.static(float(self.config.initial_air_temperature_c)), ti.f64
            )
            for i, j in self.water_volume_m2:
                volume = self.water_volume_m2[i, j]
                temperature = ti.cast(air_temperature, ti.f64)
                if volume > 1.0e-30:
                    temperature = melting + self.water_sensible_energy[i, j] / (
                        rho_cp * volume
                    )
                self.water_temperature[i, j] = temperature

        @ti.kernel
        def _measure_phase_aperture(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            """Reduce thermal totals and the legal water-side capacities."""

            self._phase_volume_before[None] = 0.0
            self._phase_energy_before[None] = 0.0
            self._phase_base_capacity[None] = 0.0
            self._phase_full_capacity[None] = 0.0
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            for i, j in self.water_volume_m2:
                volume = self.water_volume_m2[i, j]
                ti.atomic_add(self._phase_volume_before[None], volume)
                ti.atomic_add(
                    self._phase_energy_before[None],
                    self.water_sensible_energy[i, j],
                )
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                if (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                ):
                    ti.atomic_add(
                        self._phase_base_capacity[None],
                        area * ti.cast(phase, ti.f64),
                    )
                    ti.atomic_add(self._phase_full_capacity[None], area)

        @ti.kernel
        def _build_phase_aperture_targets(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            """Build a bounded target from phase volume and spare aperture."""

            area = ti.cast(ti.static(self._cell_area), ti.f64)
            total = ti.max(0.0, self._phase_volume_before[None])
            base = self._phase_base_capacity[None]
            full = self._phase_full_capacity[None]
            base_scale = ti.cast(0.0, ti.f64)
            spare_scale = ti.cast(0.0, ti.f64)
            if total <= base and base > 1.0e-30:
                base_scale = total / base
            elif total > base:
                base_scale = 1.0
                if full > base + 1.0e-30:
                    spare_scale = ti.min(1.0, (total - base) / (full - base))
            for i, j in self._phase_target_volume:
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                target = ti.cast(0.0, ti.f64)
                if (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                ):
                    phase64 = ti.cast(phase, ti.f64)
                    target = area * (
                        base_scale * phase64 + spare_scale * (1.0 - phase64)
                    )
                self._phase_target_volume[i, j] = ti.min(area, ti.max(0.0, target))

        @ti.kernel
        def _measure_phase_aperture_transfer(self):
            self._phase_donor_volume[None] = 0.0
            self._phase_donor_energy[None] = 0.0
            self._phase_receiver_volume[None] = 0.0
            for i, j in self.water_volume_m2:
                volume = self.water_volume_m2[i, j]
                target = self._phase_target_volume[i, j]
                if volume > target:
                    excess = volume - target
                    removed_energy = ti.cast(0.0, ti.f64)
                    if volume > 1.0e-30:
                        removed_energy = (
                            self.water_sensible_energy[i, j] * excess / volume
                        )
                    ti.atomic_add(self._phase_donor_volume[None], excess)
                    ti.atomic_add(self._phase_donor_energy[None], removed_energy)
                elif target > volume:
                    ti.atomic_add(self._phase_receiver_volume[None], target - volume)

        @ti.kernel
        def _apply_phase_aperture_transfer(self):
            donor_volume = self._phase_donor_volume[None]
            donor_energy = self._phase_donor_energy[None]
            receiver_volume = self._phase_receiver_volume[None]
            for i, j in self.water_volume_m2:
                volume = self.water_volume_m2[i, j]
                target = self._phase_target_volume[i, j]
                if volume > target:
                    ratio = ti.cast(0.0, ti.f64)
                    if volume > 1.0e-30:
                        ratio = target / volume
                    self.water_volume_m2[i, j] = target
                    self.water_sensible_energy[i, j] *= ratio
                elif target > volume and receiver_volume > 1.0e-30:
                    fraction = (target - volume) / receiver_volume
                    self.water_volume_m2[i, j] += donor_volume * fraction
                    self.water_sensible_energy[i, j] += donor_energy * fraction

        @ti.kernel
        def _finish_phase_aperture_transfer(self):
            self._phase_volume_after[None] = 0.0
            self._phase_energy_after[None] = 0.0
            for i, j in self.water_volume_m2:
                ti.atomic_add(
                    self._phase_volume_after[None], self.water_volume_m2[i, j]
                )
                ti.atomic_add(
                    self._phase_energy_after[None],
                    self.water_sensible_energy[i, j],
                )
            volume_residual = (
                self._phase_volume_after[None] - self._phase_volume_before[None]
            )
            energy_residual = (
                self._phase_energy_after[None] - self._phase_energy_before[None]
            )
            self.phase_aperture_volume_residual_m2[None] = volume_residual
            self.phase_aperture_energy_residual_j_m[None] = energy_residual
            self.phase_aperture_capacity_margin_m2[None] = (
                self._phase_full_capacity[None] - self._phase_volume_after[None]
            )

        @ti.kernel
        def _prepare_ale_water_remap(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            self._ale_removed_volume[None] = 0.0
            self._ale_removed_energy[None] = 0.0
            self._ale_release_weight[None] = 0.0
            self._ale_fallback_weight[None] = 0.0
            self._ale_assigned_volume[None] = 0.0
            self._ale_assigned_energy[None] = 0.0
            for i, j in self.water_volume_m2:
                old_cover = ti.min(
                    1.0, ti.max(0.0, self.world_body_indicator_prev[i, j])
                )
                new_cover = ti.min(1.0, ti.max(0.0, self.world_body_indicator[i, j]))
                if new_cover > old_cover and self.water_volume_m2[i, j] > 0.0:
                    old_open = ti.max(ti.cast(1.0 - old_cover, ti.f64), 1.0e-12)
                    swept_ratio = ti.min(
                        1.0,
                        ti.cast(new_cover - old_cover, ti.f64) / old_open,
                    )
                    removed_volume = self.water_volume_m2[i, j] * swept_ratio
                    removed_energy = self.water_sensible_energy[i, j] * swept_ratio
                    self.water_volume_m2[i, j] -= removed_volume
                    self.water_sensible_energy[i, j] -= removed_energy
                    ti.atomic_add(self._ale_removed_volume[None], removed_volume)
                    ti.atomic_add(self._ale_removed_energy[None], removed_energy)
                release = ti.cast(ti.max(old_cover - new_cover, 0.0), ti.f64) * area
                if (
                    release > 0.0
                    and wall[i, j] == 0
                    and (
                        water_phase[i, j] > ti.static(self._water_phase_cutoff)
                        or self.water_volume_m2[i, j] > 1.0e-30
                    )
                ):
                    ti.atomic_add(self._ale_release_weight[None], release)
                if (
                    wall[i, j] == 0
                    and new_cover < 0.5
                    and water_phase[i, j] > ti.static(self._water_phase_cutoff)
                ):
                    ti.atomic_add(
                        self._ale_fallback_weight[None],
                        area * ti.min(1.0, ti.max(0.0, water_phase[i, j])),
                    )

        @ti.kernel
        def _distribute_ale_water_remap(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            removed_volume = self._ale_removed_volume[None]
            removed_energy = self._ale_removed_energy[None]
            release_total = self._ale_release_weight[None]
            fallback_total = self._ale_fallback_weight[None]
            use_release = release_total > 1.0e-30
            denominator = release_total if use_release else fallback_total
            for i, j in self.water_volume_m2:
                old_cover = ti.min(
                    1.0, ti.max(0.0, self.world_body_indicator_prev[i, j])
                )
                new_cover = ti.min(1.0, ti.max(0.0, self.world_body_indicator[i, j]))
                weight = ti.cast(0.0, ti.f64)
                if use_release:
                    if wall[i, j] == 0 and (
                        water_phase[i, j] > ti.static(self._water_phase_cutoff)
                        or self.water_volume_m2[i, j] > 1.0e-30
                    ):
                        weight = (
                            ti.cast(ti.max(old_cover - new_cover, 0.0), ti.f64) * area
                        )
                elif (
                    wall[i, j] == 0
                    and new_cover < 0.5
                    and water_phase[i, j] > ti.static(self._water_phase_cutoff)
                ):
                    weight = area * ti.cast(
                        ti.min(1.0, ti.max(0.0, water_phase[i, j])), ti.f64
                    )
                if weight > 0.0 and denominator > 1.0e-30:
                    fraction = weight / denominator
                    volume = fraction * removed_volume
                    energy = fraction * removed_energy
                    self.water_volume_m2[i, j] += volume
                    self.water_sensible_energy[i, j] += energy
                    ti.atomic_add(self._ale_assigned_volume[None], volume)
                    ti.atomic_add(self._ale_assigned_energy[None], energy)

        @ti.kernel
        def _finish_ale_water_remap(self):
            volume_residual = (
                self._ale_assigned_volume[None] - self._ale_removed_volume[None]
            )
            energy_residual = (
                self._ale_assigned_energy[None] - self._ale_removed_energy[None]
            )
            self.ale_water_volume_residual_m2[None] = volume_residual
            self.ale_water_energy_residual_j_m[None] = energy_residual

        def _conservative_remap_water(self, water_phase: Any, wall: Any) -> None:
            """Apply a discrete geometric-conservation remap for pose changes.

            Water volume and sensible energy use the same swept-coverage
            weights.  Thus a uniform specific sensible energy remains uniform,
            while the two extensive global sums change only by the recorded
            roundoff residuals.
            """

            self._prepare_ale_water_remap(water_phase, wall)
            self._distribute_ale_water_remap(water_phase, wall)
            self._finish_ale_water_remap()

        @ti.kernel
        def _compose_world_temperature(self, wall: ti.template()):
            air_temperature = ti.cast(
                ti.static(float(self.config.initial_air_temperature_c)), ti.f64
            )
            melting = ti.cast(ti.static(self._melting), ti.f64)
            threshold = ti.static(self._threshold)
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            latent_water_volume = ti.cast(
                ti.static(self._rho_water * self._latent), ti.f64
            )
            for i, j in self.world_temperature_c:
                temperature = ti.cast(air_temperature, ti.f64)
                enthalpy = ti.cast(0.0, ti.f64)
                liquid = ti.cast(1.0, ti.f64)
                material = False
                if self.water_volume_m2[i, j] > 1.0e-30:
                    temperature = self.water_temperature[i, j]
                    enthalpy = (
                        latent_water_volume * self.water_volume_m2[i, j]
                        + self.water_sensible_energy[i, j]
                    ) / area
                    material = True
                body_i = self.world_body_local_i[i, j]
                body_j = self.world_body_local_j[i, j]
                if (
                    self.world_body_solid_fraction[i, j] >= threshold
                    and body_i >= 0
                    and body_j >= 0
                ):
                    temperature = self.body_temperature[body_i, body_j]
                    initial_mass = self.body_initial_mass[body_i, body_j]
                    solid_mass = self.body_solid_mass[body_i, body_j]
                    melted_mass = ti.max(0.0, initial_mass - solid_mass)
                    enthalpy = (
                        self.body_sensible_energy[body_i, body_j]
                        + melted_mass * ti.static(self._latent)
                    ) / area
                    liquid = 1.0 - self.body_solid_fraction[body_i, body_j]
                    material = True
                if wall[i, j] != 0:
                    temperature = melting
                    enthalpy = 0.0
                    liquid = 0.0
                    material = False
                self.world_temperature_c[i, j] = temperature
                self.enthalpy_j_m3[i, j] = enthalpy
                self.liquid_fraction[i, j] = ti.cast(liquid, ti.f32)
                self.ice_material[i, j] = ti.cast(1 if material else 0, ti.i8)

        @ti.kernel
        def _compute_body_flux_x(self):
            k_ice = ti.cast(ti.static(self._k_ice), ti.f64)
            for face_i, j in self._body_flux_x:
                flux = ti.cast(0.0, ti.f64)
                if 0 < face_i < ti.static(self.body_nx):
                    left_i = face_i - 1
                    right_i = face_i
                    left_fraction = self.body_solid_fraction[left_i, j]
                    right_fraction = self.body_solid_fraction[right_i, j]
                    if left_fraction > 0.0 and right_fraction > 0.0:
                        fraction = ti.cast(
                            ti.min(left_fraction, right_fraction), ti.f64
                        )
                        flux = (
                            -k_ice
                            * fraction
                            * (
                                self.body_temperature[right_i, j]
                                - self.body_temperature[left_i, j]
                            )
                        )
                self._body_flux_x[face_i, j] = flux

        @ti.kernel
        def _compute_body_flux_y(self):
            k_ice = ti.cast(ti.static(self._k_ice), ti.f64)
            for i, face_j in self._body_flux_y:
                flux = ti.cast(0.0, ti.f64)
                if 0 < face_j < ti.static(self.body_ny):
                    bottom_j = face_j - 1
                    top_j = face_j
                    bottom_fraction = self.body_solid_fraction[i, bottom_j]
                    top_fraction = self.body_solid_fraction[i, top_j]
                    if bottom_fraction > 0.0 and top_fraction > 0.0:
                        fraction = ti.cast(
                            ti.min(bottom_fraction, top_fraction), ti.f64
                        )
                        flux = (
                            -k_ice
                            * fraction
                            * (
                                self.body_temperature[i, top_j]
                                - self.body_temperature[i, bottom_j]
                            )
                        )
                self._body_flux_y[i, face_j] = flux

        @ti.kernel
        def _accumulate_body_conduction(self, time_step_s: ti.f64):
            for i, j in self._body_heat_delta:
                divergence = (
                    self._body_flux_x[i + 1, j]
                    - self._body_flux_x[i, j]
                    + self._body_flux_y[i, j + 1]
                    - self._body_flux_y[i, j]
                )
                self._body_heat_delta[i, j] -= time_step_s * divergence

        @ti.kernel
        def _compute_water_conduction_flux_x(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            k_water = ti.cast(ti.static(self._k_water), ti.f64)
            for face_i, j in self._water_flux_x:
                flux = ti.cast(0.0, ti.f64)
                boundary_inward = ti.cast(0.0, ti.f64)
                if 0 < face_i < ti.static(self.nx):
                    left_i = face_i - 1
                    right_i = face_i
                    left_active = (
                        wall[left_i, j] == 0
                        and solid[left_i, j] == 0
                        and water_phase[left_i, j] > ti.static(self._water_phase_cutoff)
                    )
                    right_active = (
                        wall[right_i, j] == 0
                        and solid[right_i, j] == 0
                        and water_phase[right_i, j]
                        > ti.static(self._water_phase_cutoff)
                    )
                    if left_active and right_active:
                        if (
                            self.water_volume_m2[left_i, j] > 1.0e-30
                            and self.water_volume_m2[right_i, j] > 1.0e-30
                        ):
                            left_aperture = ti.min(
                                1.0,
                                ti.max(
                                    0.0,
                                    self.water_volume_m2[left_i, j] / area,
                                ),
                            )
                            right_aperture = ti.min(
                                1.0,
                                ti.max(
                                    0.0,
                                    self.water_volume_m2[right_i, j] / area,
                                ),
                            )
                            face_aperture = ti.min(left_aperture, right_aperture)
                            flux = (
                                -k_water
                                * face_aperture
                                * (
                                    self.water_temperature[right_i, j]
                                    - self.water_temperature[left_i, j]
                                )
                            )
                    elif (
                        wall[left_i, j] != 0
                        and right_active
                        and self.water_volume_m2[right_i, j] > 1.0e-30
                    ):
                        local_aperture = ti.min(
                            1.0,
                            ti.max(
                                0.0,
                                self.water_volume_m2[right_i, j] / area,
                            ),
                        )
                        if ti.static(self._left_kind == _DIRICHLET):
                            flux = (
                                2.0
                                * k_water
                                * local_aperture
                                * (
                                    ti.static(self._left_value)
                                    - self.water_temperature[right_i, j]
                                )
                            )
                        boundary_inward = flux
                    elif (
                        left_active
                        and self.water_volume_m2[left_i, j] > 1.0e-30
                        and wall[right_i, j] != 0
                    ):
                        local_aperture = ti.min(
                            1.0,
                            ti.max(
                                0.0,
                                self.water_volume_m2[left_i, j] / area,
                            ),
                        )
                        if ti.static(self._right_kind == _DIRICHLET):
                            flux = (
                                2.0
                                * k_water
                                * local_aperture
                                * (
                                    self.water_temperature[left_i, j]
                                    - ti.static(self._right_value)
                                )
                            )
                        boundary_inward = -flux
                self._water_flux_x[face_i, j] = flux
                if boundary_inward != 0.0:
                    ti.atomic_add(self.boundary_power[None], boundary_inward)

        @ti.kernel
        def _compute_water_advection_flux_x(
            self,
            velocity: ti.template(),
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            dx = ti.cast(ti.static(self._dx), ti.f64)
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            velocity_scale = ti.cast(ti.static(self._velocity_scale), ti.f64)
            for face_i, j in self._water_flux_x:
                energy_flux = ti.cast(0.0, ti.f64)
                volume_flux = ti.cast(0.0, ti.f64)
                if ti.static(self._advection_enabled):
                    if 0 < face_i < ti.static(self.nx):
                        left_i = face_i - 1
                        right_i = face_i
                        left_active = (
                            wall[left_i, j] == 0
                            and solid[left_i, j] == 0
                            and water_phase[left_i, j]
                            > ti.static(self._water_phase_cutoff)
                        )
                        right_active = (
                            wall[right_i, j] == 0
                            and solid[right_i, j] == 0
                            and water_phase[right_i, j]
                            > ti.static(self._water_phase_cutoff)
                        )
                        if left_active and right_active:
                            speed = (
                                0.5
                                * ti.cast(
                                    velocity[left_i, j].x + velocity[right_i, j].x,
                                    ti.f64,
                                )
                                * velocity_scale
                            )
                            upwind_i = right_i
                            if speed >= 0.0:
                                upwind_i = left_i
                            upwind_volume = self.water_volume_m2[upwind_i, j]
                            upwind_fraction = ti.min(
                                1.0, ti.max(0.0, upwind_volume / area)
                            )
                            volume_flux = speed * upwind_fraction * dx
                            energy_density = self.water_sensible_energy[
                                upwind_i, j
                            ] / ti.max(upwind_volume, 1.0e-30)
                            energy_flux = volume_flux * energy_density
                self._water_flux_x[face_i, j] = energy_flux
                self._water_volume_flux_x[face_i, j] = volume_flux

        @ti.kernel
        def _compute_water_conduction_flux_y(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            k_water = ti.cast(ti.static(self._k_water), ti.f64)
            for i, face_j in self._water_flux_y:
                flux = ti.cast(0.0, ti.f64)
                boundary_inward = ti.cast(0.0, ti.f64)
                if 0 < face_j < ti.static(self.ny):
                    bottom_j = face_j - 1
                    top_j = face_j
                    bottom_active = (
                        wall[i, bottom_j] == 0
                        and solid[i, bottom_j] == 0
                        and water_phase[i, bottom_j]
                        > ti.static(self._water_phase_cutoff)
                    )
                    top_active = (
                        wall[i, top_j] == 0
                        and solid[i, top_j] == 0
                        and water_phase[i, top_j] > ti.static(self._water_phase_cutoff)
                    )
                    if bottom_active and top_active:
                        if (
                            self.water_volume_m2[i, bottom_j] > 1.0e-30
                            and self.water_volume_m2[i, top_j] > 1.0e-30
                        ):
                            bottom_aperture = ti.min(
                                1.0,
                                ti.max(
                                    0.0,
                                    self.water_volume_m2[i, bottom_j] / area,
                                ),
                            )
                            top_aperture = ti.min(
                                1.0,
                                ti.max(
                                    0.0,
                                    self.water_volume_m2[i, top_j] / area,
                                ),
                            )
                            face_aperture = ti.min(bottom_aperture, top_aperture)
                            flux = (
                                -k_water
                                * face_aperture
                                * (
                                    self.water_temperature[i, top_j]
                                    - self.water_temperature[i, bottom_j]
                                )
                            )
                    elif (
                        wall[i, bottom_j] != 0
                        and top_active
                        and self.water_volume_m2[i, top_j] > 1.0e-30
                    ):
                        local_aperture = ti.min(
                            1.0,
                            ti.max(
                                0.0,
                                self.water_volume_m2[i, top_j] / area,
                            ),
                        )
                        if ti.static(self._bottom_kind == _DIRICHLET):
                            flux = (
                                2.0
                                * k_water
                                * local_aperture
                                * (
                                    ti.static(self._bottom_value)
                                    - self.water_temperature[i, top_j]
                                )
                            )
                        boundary_inward = flux
                    elif (
                        bottom_active
                        and self.water_volume_m2[i, bottom_j] > 1.0e-30
                        and wall[i, top_j] != 0
                    ):
                        local_aperture = ti.min(
                            1.0,
                            ti.max(
                                0.0,
                                self.water_volume_m2[i, bottom_j] / area,
                            ),
                        )
                        if ti.static(self._top_kind == _DIRICHLET):
                            flux = (
                                2.0
                                * k_water
                                * local_aperture
                                * (
                                    self.water_temperature[i, bottom_j]
                                    - ti.static(self._top_value)
                                )
                            )
                        boundary_inward = -flux
                self._water_flux_y[i, face_j] = flux
                if boundary_inward != 0.0:
                    ti.atomic_add(self.boundary_power[None], boundary_inward)

        @ti.kernel
        def _compute_water_advection_flux_y(
            self,
            velocity: ti.template(),
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            dx = ti.cast(ti.static(self._dx), ti.f64)
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            velocity_scale = ti.cast(ti.static(self._velocity_scale), ti.f64)
            for i, face_j in self._water_flux_y:
                energy_flux = ti.cast(0.0, ti.f64)
                volume_flux = ti.cast(0.0, ti.f64)
                if ti.static(self._advection_enabled):
                    if 0 < face_j < ti.static(self.ny):
                        bottom_j = face_j - 1
                        top_j = face_j
                        bottom_active = (
                            wall[i, bottom_j] == 0
                            and solid[i, bottom_j] == 0
                            and water_phase[i, bottom_j]
                            > ti.static(self._water_phase_cutoff)
                        )
                        top_active = (
                            wall[i, top_j] == 0
                            and solid[i, top_j] == 0
                            and water_phase[i, top_j]
                            > ti.static(self._water_phase_cutoff)
                        )
                        if bottom_active and top_active:
                            speed = (
                                0.5
                                * ti.cast(
                                    velocity[i, bottom_j].y + velocity[i, top_j].y,
                                    ti.f64,
                                )
                                * velocity_scale
                            )
                            upwind_j = top_j
                            if speed >= 0.0:
                                upwind_j = bottom_j
                            upwind_volume = self.water_volume_m2[i, upwind_j]
                            upwind_fraction = ti.min(
                                1.0, ti.max(0.0, upwind_volume / area)
                            )
                            volume_flux = speed * upwind_fraction * dx
                            energy_density = self.water_sensible_energy[
                                i, upwind_j
                            ] / ti.max(upwind_volume, 1.0e-30)
                            energy_flux = volume_flux * energy_density
                self._water_flux_y[i, face_j] = energy_flux
                self._water_volume_flux_y[i, face_j] = volume_flux

        @ti.kernel
        def _update_water_advection(self, time_step_s: ti.f64, wall: ti.template()):
            for i, j in self.water_sensible_energy:
                energy_divergence = (
                    self._water_flux_x[i + 1, j]
                    - self._water_flux_x[i, j]
                    + self._water_flux_y[i, j + 1]
                    - self._water_flux_y[i, j]
                )
                volume_divergence = (
                    self._water_volume_flux_x[i + 1, j]
                    - self._water_volume_flux_x[i, j]
                    + self._water_volume_flux_y[i, j + 1]
                    - self._water_volume_flux_y[i, j]
                )
                if wall[i, j] == 0:
                    self.water_sensible_energy[i, j] -= time_step_s * energy_divergence
                    self.water_volume_m2[i, j] -= time_step_s * volume_divergence

        @ti.kernel
        def _update_water_conduction(self, time_step_s: ti.f64, wall: ti.template()):
            for i, j in self.water_sensible_energy:
                energy_divergence = (
                    self._water_flux_x[i + 1, j]
                    - self._water_flux_x[i, j]
                    + self._water_flux_y[i, j + 1]
                    - self._water_flux_y[i, j]
                )
                if wall[i, j] == 0:
                    self.water_sensible_energy[i, j] -= time_step_s * energy_divergence

        @ti.kernel
        def _accumulate_boundary_heat(self, time_step_s: ti.f64):
            self.boundary_heat_input[None] += time_step_s * self.boundary_power[None]

        @ti.kernel
        def _compute_interface_heat_requests(
            self,
            time_step_s: ti.f64,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            k_face = ti.cast(
                ti.static(
                    2.0 * self._k_water * self._k_ice / (self._k_water + self._k_ice)
                ),
                ti.f64,
            )
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            threshold = ti.cast(ti.static(self._threshold), ti.f64)
            for i, j in self._interface_heat_requested:
                requested = ti.cast(0.0, ti.f64)
                active_water = (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and water_phase[i, j] > ti.static(self._water_phase_cutoff)
                    and self.water_volume_m2[i, j] > 1.0e-30
                )
                if active_water:
                    water_aperture = ti.min(
                        1.0,
                        ti.max(0.0, self.water_volume_m2[i, j] / area),
                    )
                    # The zero offset is the mixed-cell contact needed after
                    # a remnant drops below the sharp LBM threshold.  The
                    # four axial offsets retain the resolved surface faces.
                    for di, dj in ti.static(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < ti.static(self.nx) and 0 <= nj < ti.static(
                            self.ny
                        ):
                            world_fraction = ti.cast(
                                self.world_body_solid_fraction[ni, nj], ti.f64
                            )
                            if world_fraction > 0.0:
                                body_aperture = ti.min(1.0, world_fraction / threshold)
                                contact_aperture = ti.min(water_aperture, body_aperture)
                                base_i = self.world_body_interp_base_i[ni, nj]
                                base_j = self.world_body_interp_base_j[ni, nj]
                                fraction_x = ti.cast(
                                    self.world_body_interp_fraction_x[ni, nj],
                                    ti.f64,
                                )
                                fraction_y = ti.cast(
                                    self.world_body_interp_fraction_y[ni, nj],
                                    ti.f64,
                                )
                                for ci, cj in ti.static(ti.ndrange(2, 2)):
                                    body_i = base_i + ci
                                    body_j = base_j + cj
                                    if (
                                        0 <= body_i < ti.static(self.body_nx)
                                        and 0 <= body_j < ti.static(self.body_ny)
                                        and self.body_solid_mass[body_i, body_j] > 0.0
                                    ):
                                        weight_x = fraction_x
                                        weight_y = fraction_y
                                        if ti.static(ci == 0):
                                            weight_x = 1.0 - fraction_x
                                        if ti.static(cj == 0):
                                            weight_y = 1.0 - fraction_y
                                        contribution = (
                                            weight_x
                                            * weight_y
                                            * ti.cast(
                                                self.body_solid_fraction[
                                                    body_i, body_j
                                                ],
                                                ti.f64,
                                            )
                                        )
                                        if contribution > 0.0:
                                            share = contribution / world_fraction
                                            delta_temperature = ti.max(
                                                0.0,
                                                self.water_temperature[i, j]
                                                - self.body_temperature[body_i, body_j],
                                            )
                                            requested += (
                                                time_step_s
                                                * k_face
                                                * contact_aperture
                                                * share
                                                * delta_temperature
                                            )
                self._interface_heat_requested[i, j] = requested

        @ti.kernel
        def _apply_interface_heat(
            self,
            time_step_s: ti.f64,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            k_face = ti.cast(
                ti.static(
                    2.0 * self._k_water * self._k_ice / (self._k_water + self._k_ice)
                ),
                ti.f64,
            )
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            threshold = ti.cast(ti.static(self._threshold), ti.f64)
            ny = ti.static(self.ny)
            for i, j in self._interface_heat_requested:
                requested = self._interface_heat_requested[i, j]
                scale = ti.cast(0.0, ti.f64)
                if requested > 0.0:
                    scale = ti.min(
                        1.0,
                        ti.max(0.0, self.water_sensible_energy[i, j]) / requested,
                    )
                transferred = ti.cast(0.0, ti.f64)
                if scale > 0.0:
                    water_aperture = ti.min(
                        1.0,
                        ti.max(0.0, self.water_volume_m2[i, j] / area),
                    )
                    for di, dj in ti.static(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < ti.static(self.nx) and 0 <= nj < ti.static(
                            self.ny
                        ):
                            world_fraction = ti.cast(
                                self.world_body_solid_fraction[ni, nj], ti.f64
                            )
                            if world_fraction > 0.0:
                                body_aperture = ti.min(1.0, world_fraction / threshold)
                                contact_aperture = ti.min(water_aperture, body_aperture)
                                base_i = self.world_body_interp_base_i[ni, nj]
                                base_j = self.world_body_interp_base_j[ni, nj]
                                fraction_x = ti.cast(
                                    self.world_body_interp_fraction_x[ni, nj],
                                    ti.f64,
                                )
                                fraction_y = ti.cast(
                                    self.world_body_interp_fraction_y[ni, nj],
                                    ti.f64,
                                )
                                for ci, cj in ti.static(ti.ndrange(2, 2)):
                                    body_i = base_i + ci
                                    body_j = base_j + cj
                                    if (
                                        0 <= body_i < ti.static(self.body_nx)
                                        and 0 <= body_j < ti.static(self.body_ny)
                                        and self.body_solid_mass[body_i, body_j] > 0.0
                                    ):
                                        weight_x = fraction_x
                                        weight_y = fraction_y
                                        if ti.static(ci == 0):
                                            weight_x = 1.0 - fraction_x
                                        if ti.static(cj == 0):
                                            weight_y = 1.0 - fraction_y
                                        contribution = (
                                            weight_x
                                            * weight_y
                                            * ti.cast(
                                                self.body_solid_fraction[
                                                    body_i, body_j
                                                ],
                                                ti.f64,
                                            )
                                        )
                                        if contribution > 0.0:
                                            share = contribution / world_fraction
                                            delta_temperature = ti.max(
                                                0.0,
                                                self.water_temperature[i, j]
                                                - self.body_temperature[body_i, body_j],
                                            )
                                            heat = (
                                                scale
                                                * time_step_s
                                                * k_face
                                                * contact_aperture
                                                * share
                                                * delta_temperature
                                            )
                                            transferred += heat
                                            ti.atomic_add(
                                                self._body_heat_delta[body_i, body_j],
                                                heat,
                                            )
                                            ti.atomic_max(
                                                self._body_melt_target[body_i, body_j],
                                                i * ny + j,
                                            )
                self.water_sensible_energy[i, j] -= transferred

        @ti.kernel
        def _apply_body_heat_and_phase_change(self):
            latent = ti.cast(ti.static(self._latent), ti.f64)
            rho_water = ti.cast(ti.static(self._rho_water), ti.f64)
            for i, j in self.body_solid_mass:
                heat = self._body_heat_delta[i, j]
                solid_mass = self.body_solid_mass[i, j]
                sensible = self.body_sensible_energy[i, j]
                melt_mass = ti.cast(0.0, ti.f64)
                melt_sensible = ti.cast(0.0, ti.f64)
                if heat <= 0.0:
                    if solid_mass > 0.0:
                        sensible += heat
                else:
                    sensible_deficit = ti.max(-sensible, 0.0)
                    sensible_gain = ti.min(heat, sensible_deficit)
                    sensible += sensible_gain
                    remaining = heat - sensible_gain
                    melt_mass = ti.min(solid_mass, remaining / latent)
                    solid_mass -= melt_mass
                    melt_sensible = ti.max(0.0, remaining - melt_mass * latent)
                # Do not silently discard a positive terminal residue: it is
                # melted only when ``remaining`` supplies the corresponding
                # latent heat, so mass, volume, and energy sources stay paired.
                if solid_mass <= 0.0:
                    solid_mass = 0.0
                    sensible = 0.0
                melt_volume = melt_mass / rho_water
                self.body_solid_mass[i, j] = solid_mass
                self.body_sensible_energy[i, j] = sensible
                self._body_melt_mass_step[i, j] = melt_mass
                self._body_melt_volume_step[i, j] = melt_volume
                self._body_melt_sensible_step[i, j] = melt_sensible
                if melt_mass > 0.0:
                    ti.atomic_add(self._interval_body_melt_mass[None], melt_mass)

        @ti.kernel
        def _freeze_melt_injection_eligibility(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            for i, j in self._melt_injection_eligible:
                eligible = (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and water_phase[i, j] > ti.static(self._water_phase_cutoff)
                    and self.water_volume_m2[i, j] > 1.0e-30
                )
                self._melt_injection_eligible[i, j] = ti.cast(eligible, ti.i8)

        @ti.kernel
        def _inject_melt_water(
            self,
            body_reference_origin: ti.template(),
            body_angle: ti.template(),
        ):
            origin = body_reference_origin[None]
            angle = body_angle[None]
            cosine = ti.cos(angle)
            sine = ti.sin(angle)
            ny = ti.static(self.ny)
            for body_i, body_j in self._body_melt_mass_step:
                melt_mass = self._body_melt_mass_step[body_i, body_j]
                if melt_mass > 0.0:
                    local_x = (
                        ti.cast(body_i, ti.f64) + 0.5 - ti.static(0.5 * self.body_nx)
                    )
                    local_y = (
                        ti.cast(body_j, ti.f64) + 0.5 - ti.static(0.5 * self.body_ny)
                    )
                    world_x = origin.x + cosine * local_x - sine * local_y
                    world_y = origin.y + sine * local_x + cosine * local_y
                    encoded = self._body_melt_target[body_i, body_j]
                    target_i = -1
                    target_j = -1
                    if encoded >= 0:
                        target_i = encoded // ny
                        target_j = encoded - target_i * ny
                    else:
                        base_i = ti.cast(ti.floor(world_x), ti.i32)
                        base_j = ti.cast(ti.floor(world_y), ti.i32)
                        best_distance = 1.0e30
                        for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
                            wi = base_i + di
                            wj = base_j + dj
                            if (
                                0 <= wi < ti.static(self.nx)
                                and 0 <= wj < ti.static(self.ny)
                                and self._melt_injection_eligible[wi, wj] != 0
                            ):
                                distance = ti.cast(di * di + dj * dj, ti.f32)
                                if distance < best_distance:
                                    best_distance = distance
                                    target_i = wi
                                    target_j = wj
                    volume = self._body_melt_volume_step[body_i, body_j]
                    energy = self._body_melt_sensible_step[body_i, body_j]
                    if target_i >= 0 and target_j >= 0:
                        ti.atomic_add(self.water_volume_m2[target_i, target_j], volume)
                        ti.atomic_add(
                            self.water_sensible_energy[target_i, target_j], energy
                        )
                        ti.atomic_add(
                            self.water_melt_mass_source[target_i, target_j], melt_mass
                        )
                        ti.atomic_add(self._interval_water_melt_mass[None], melt_mass)
                    else:
                        ti.atomic_add(self._unassigned_melt_mass[None], melt_mass)
                        ti.atomic_add(self._unassigned_melt_volume[None], volume)
                        ti.atomic_add(self._unassigned_melt_energy[None], energy)

        @ti.kernel
        def _measure_melt_fallback_weights(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            self._melt_fallback_free_weight[None] = 0.0
            self._melt_fallback_wet_weight[None] = 0.0
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            for i, j in self.water_volume_m2:
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                if (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                    and self.water_volume_m2[i, j] > 1.0e-30
                ):
                    ti.atomic_add(
                        self._melt_fallback_free_weight[None],
                        ti.max(area - self.water_volume_m2[i, j], 0.0),
                    )
                    ti.atomic_add(
                        self._melt_fallback_wet_weight[None],
                        self.water_volume_m2[i, j],
                    )

        @ti.kernel
        def _apply_melt_fallback(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
            use_free_capacity: ti.i32,
        ):
            area = ti.cast(ti.static(self._cell_area), ti.f64)
            denominator = self._melt_fallback_wet_weight[None]
            if use_free_capacity != 0:
                denominator = self._melt_fallback_free_weight[None]
            mass = self._unassigned_melt_mass[None]
            volume = self._unassigned_melt_volume[None]
            energy = self._unassigned_melt_energy[None]
            if denominator > 1.0e-30 and mass > 0.0:
                self._interval_water_melt_mass[None] += mass
            for i, j in self.water_volume_m2:
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                eligible = (
                    wall[i, j] == 0
                    and solid[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                    and self.water_volume_m2[i, j] > 1.0e-30
                )
                weight = ti.cast(0.0, ti.f64)
                if eligible:
                    if use_free_capacity != 0:
                        weight = ti.max(area - self.water_volume_m2[i, j], 0.0)
                    else:
                        weight = self.water_volume_m2[i, j]
                if weight > 0.0 and denominator > 1.0e-30:
                    fraction = weight / denominator
                    added_mass = fraction * mass
                    added_volume = fraction * volume
                    added_energy = fraction * energy
                    self.water_volume_m2[i, j] += added_volume
                    self.water_sensible_energy[i, j] += added_energy
                    self.water_melt_mass_source[i, j] += added_mass

        def _distribute_unassigned_melt(
            self, water_phase: Any, wall: Any, solid: Any
        ) -> None:
            """Conservatively inject melt whose local interface target vanished."""

            unassigned_mass = float(self._unassigned_melt_mass[None])
            if unassigned_mass <= 0.0:
                return
            self._measure_melt_fallback_weights(water_phase, wall, solid)
            free_weight = float(self._melt_fallback_free_weight[None])
            wet_weight = float(self._melt_fallback_wet_weight[None])
            if free_weight > 1.0e-30:
                self._apply_melt_fallback(water_phase, wall, solid, 1)
            elif wet_weight > 1.0e-30:
                # The following aperture projection will either recover spare
                # capacity from other thermal donors or report infeasibility.
                self._apply_melt_fallback(water_phase, wall, solid, 0)
            else:
                raise RuntimeError(
                    "melt water has no legal LBM water cell for conservative "
                    "mass and energy injection"
                )

        @ti.kernel
        def _reduce_melt_injection_residual(self):
            self.melt_injection_mass_residual_kg_m[None] = (
                self._interval_water_melt_mass[None]
                - self._interval_body_melt_mass[None]
            )

        @ti.kernel
        def _reduce_totals(self):
            self._initial_body_mass_sum[None] = 0.0
            self._solid_body_mass_sum[None] = 0.0
            self._water_mass_sum[None] = 0.0
            self._body_sensible_sum[None] = 0.0
            self._water_sensible_sum[None] = 0.0
            rho_water = ti.static(self._rho_water)
            for i, j in self.body_initial_mass:
                ti.atomic_add(
                    self._initial_body_mass_sum[None], self.body_initial_mass[i, j]
                )
                ti.atomic_add(
                    self._solid_body_mass_sum[None], self.body_solid_mass[i, j]
                )
                ti.atomic_add(
                    self._body_sensible_sum[None], self.body_sensible_energy[i, j]
                )
            for i, j in self.water_volume_m2:
                ti.atomic_add(
                    self._water_mass_sum[None], rho_water * self.water_volume_m2[i, j]
                )
                ti.atomic_add(
                    self._water_sensible_sum[None], self.water_sensible_energy[i, j]
                )

else:

    class MovingBodyThermal2D:
        """Unavailable Taichi moving-body component placeholder."""

        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError(
                "MovingBodyThermal2D requires the optional 'taichi' package and "
                "an initialized Taichi runtime"
            )


__all__ = [
    "LatticeScales",
    "MovingBodyThermal2D",
    "MovingBodyThermalScheme",
    "MovingBodyThermalTotals",
    "PhaseChangeProperties",
    "ThermalBoundary",
    "ThermalBoundaryKind",
    "ThermalBoundarySet",
    "ThermalConfig",
]
