"""Reusable thermal configuration and enthalpy finite-volume kernels.

The module has two deliberately separated layers:

* The dataclasses, lattice-unit conversion, and NumPy enthalpy inversion are
  usable without Taichi.  They form the CPU reference surface used by tests
  and by the standalone Stefan benchmark.
* :class:`EnthalpyFV2D` owns Taichi fields and kernels intended to be held by
  ``IceFlow2D`` after its CUDA runtime has been initialized.  Importing this
  module does not initialize Taichi and remains valid when Taichi is absent.

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

import numpy as np

try:  # Configuration and NumPy helpers must work without the GPU dependency.
    import taichi as ti
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    ti = None  # type: ignore[assignment]


ThermalBoundaryKind = Literal["adiabatic", "dirichlet", "neumann"]
WaterBuoyancyModel = Literal["linear", "freshwater_quadratic"]

_ADIABATIC = 0
_DIRICHLET = 1
_NEUMANN = 2
_BOUNDARY_CODE = {
    "adiabatic": _ADIABATIC,
    "dirichlet": _DIRICHLET,
    "neumann": _NEUMANN,
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

    ``value`` is a temperature in degrees Celsius for ``dirichlet`` and a heat
    flux in W/m^2 for ``neumann``.  Neumann heat flux is positive *into* the
    active domain on every side.  ``adiabatic`` requires a zero value.
    """

    kind: ThermalBoundaryKind = "adiabatic"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _BOUNDARY_CODE:
            raise ValueError(
                "thermal boundary kind must be 'adiabatic', 'dirichlet', or 'neumann'"
            )
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

    @classmethod
    def neumann(cls, inward_heat_flux_w_m2: float) -> "ThermalBoundary":
        return cls("neumann", inward_heat_flux_w_m2)


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
    temperature and the initial ice at or below it.  During the run, every
    initially resolved water cell uses the same ice/water enthalpy law and can
    therefore freeze when a cold boundary removes enough energy.
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
    water_buoyancy_model: WaterBuoyancyModel = "linear"
    thermal_expansion_water_1_k: float = 2.1e-4
    buoyancy_reference_temperature_c: float | None = None
    freshwater_density_max_temperature_c: float = 4.0
    freshwater_density_quadratic_coefficient_1_k2: float = 8.0e-6
    scheme: Literal["enthalpy_fv"] = "enthalpy_fv"

    def __post_init__(self) -> None:
        if not isinstance(self.properties, PhaseChangeProperties):
            raise TypeError("properties must be a PhaseChangeProperties")
        if not isinstance(self.boundaries, ThermalBoundarySet):
            raise TypeError("boundaries must be a ThermalBoundarySet")
        if self.scheme != "enthalpy_fv":
            raise ValueError("scheme must be 'enthalpy_fv'")
        if self.water_buoyancy_model not in ("linear", "freshwater_quadratic"):
            raise ValueError(
                "water_buoyancy_model must be 'linear' or "
                "'freshwater_quadratic'"
            )
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
        if self.water_buoyancy_model == "freshwater_quadratic":
            reference_factor = 1.0 - (
                self.freshwater_density_quadratic_coefficient_1_k2
                * (
                    self.buoyancy_reference_temperature_c
                    - self.freshwater_density_max_temperature_c
                )
                ** 2
            )
            if reference_factor <= 0.0:
                raise ValueError(
                    "freshwater quadratic density must remain positive at the "
                    "buoyancy reference temperature"
                )


def water_density_ratio_to_reference(
    temperature_c: Any, config: ThermalConfig
) -> float | np.ndarray:
    """Return liquid-water density divided by its buoyancy reference density.

    This ratio is used only in the Boussinesq gravity source.  It does not
    replace the material density used by the air--water LBM, the enthalpy law,
    or the ice/water mass-conversion model.  The quadratic branch implements
    ``rho(T)=rho_star*(1-beta*(T-T_star)**2)`` and normalizes it by the same
    expression evaluated at the configured far-field reference temperature.
    """

    if not isinstance(config, ThermalConfig):
        raise TypeError("config must be a ThermalConfig")
    anomaly = water_density_anomaly_ratio_to_reference(temperature_c, config)
    return 1.0 + anomaly


def water_density_anomaly_ratio_to_reference(
    temperature_c: Any, config: ThermalConfig
) -> float | np.ndarray:
    """Return ``(rho_water(T) - rho_water(T_inf)) / rho_water(T_inf)``.

    The quadratic expression is evaluated directly instead of subtracting two
    nearly equal density ratios.  ``T_inf`` is the configured far-field
    buoyancy reference and defaults to the initial bath temperature.
    """

    if not isinstance(config, ThermalConfig):
        raise TypeError("config must be a ThermalConfig")
    temperature = np.asarray(temperature_c, dtype=np.float64)
    reference = float(config.buoyancy_reference_temperature_c)
    if config.water_buoyancy_model == "linear":
        anomaly = -float(config.thermal_expansion_water_1_k) * (
            temperature - reference
        )
    else:
        beta = float(config.freshwater_density_quadratic_coefficient_1_k2)
        maximum_temperature = float(config.freshwater_density_max_temperature_c)
        reference_factor = 1.0 - beta * (reference - maximum_temperature) ** 2
        anomaly = beta * (
            (reference - maximum_temperature) ** 2
            - (temperature - maximum_temperature) ** 2
        ) / reference_factor
    if anomaly.ndim == 0:
        return float(anomaly)
    return anomaly


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

    def diffusivity_to_lattice(self, diffusivity_m2_s: float) -> float:
        diffusivity = _non_negative("diffusivity_m2_s", diffusivity_m2_s)
        return diffusivity * self.dt_s / (self.dx_m * self.dx_m)

    def velocity_to_lattice(self, velocity_m_s: float) -> float:
        return _finite("velocity_m_s", velocity_m_s) / self.velocity_scale_m_s

    def velocity_to_physical(self, velocity_lattice: float) -> float:
        return _finite("velocity_lattice", velocity_lattice) * self.velocity_scale_m_s

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
        reference_velocity = _positive(
            "reference_velocity_m_s", reference_velocity_m_s
        )
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


def recover_temperature_and_liquid_fraction_numpy(
    enthalpy_j_m3: np.ndarray,
    properties: PhaseChangeProperties,
    *,
    density_ice_kg_m3: float,
    density_water_kg_m3: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert physical ice/water volumetric enthalpy into ``(T, liquid)``.

    The reference state is solid ice at the melting temperature, where
    ``H=0``.  The latent interval ends at ``rho_ice * L``.  A distinct water
    density may be supplied for liquid sensible heat; omitting it selects the
    equal-density Stefan approximation.
    """

    if not isinstance(properties, PhaseChangeProperties):
        raise TypeError("properties must be a PhaseChangeProperties")
    rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
    rho_water = (
        rho_ice
        if density_water_kg_m3 is None
        else _positive("density_water_kg_m3", density_water_kg_m3)
    )
    enthalpy = np.asarray(enthalpy_j_m3, dtype=np.float64)
    temperature = np.empty_like(enthalpy)
    liquid_fraction = np.empty_like(enthalpy)

    latent_volume = rho_ice * properties.latent_heat_j_kg
    melting = properties.melting_temperature_c
    solid = enthalpy < 0.0
    mushy = (enthalpy >= 0.0) & (enthalpy <= latent_volume)
    liquid = enthalpy > latent_volume

    temperature[solid] = melting + enthalpy[solid] / (
        rho_ice * properties.specific_heat_ice_j_kg_k
    )
    liquid_fraction[solid] = 0.0
    temperature[mushy] = melting
    liquid_fraction[mushy] = enthalpy[mushy] / latent_volume
    temperature[liquid] = melting + (enthalpy[liquid] - latent_volume) / (
        rho_water * properties.specific_heat_water_j_kg_k
    )
    liquid_fraction[liquid] = 1.0
    return temperature, liquid_fraction


def phase_change_enthalpy_numpy(
    temperature_c: np.ndarray,
    liquid_fraction: np.ndarray,
    properties: PhaseChangeProperties,
    *,
    density_ice_kg_m3: float,
    density_water_kg_m3: float | None = None,
) -> np.ndarray:
    """Return physical enthalpy for a thermodynamically consistent state.

    A temperature below the melting point requires zero liquid fraction; a
    temperature above it requires unit liquid fraction.  At the melting point
    any liquid fraction in ``[0, 1]`` is valid.
    """

    if not isinstance(properties, PhaseChangeProperties):
        raise TypeError("properties must be a PhaseChangeProperties")
    rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
    rho_water = (
        rho_ice
        if density_water_kg_m3 is None
        else _positive("density_water_kg_m3", density_water_kg_m3)
    )
    temperature, fraction = np.broadcast_arrays(
        np.asarray(temperature_c, dtype=np.float64),
        np.asarray(liquid_fraction, dtype=np.float64),
    )
    if not np.isfinite(temperature).all() or not np.isfinite(fraction).all():
        raise ValueError("temperature and liquid_fraction must be finite")
    if (
        float(np.min(fraction, initial=0.0)) < 0.0
        or float(np.max(fraction, initial=1.0)) > 1.0
    ):
        raise ValueError("liquid_fraction must lie in [0, 1]")

    melting = properties.melting_temperature_c
    below = temperature < melting
    above = temperature > melting
    if np.any(below & (fraction != 0.0)):
        raise ValueError("sub-melting states must have zero liquid fraction")
    if np.any(above & (fraction != 1.0)):
        raise ValueError("super-melting states must have unit liquid fraction")

    latent_volume = rho_ice * properties.latent_heat_j_kg
    enthalpy = fraction * latent_volume
    enthalpy = np.where(
        below,
        rho_ice * properties.specific_heat_ice_j_kg_k * (temperature - melting),
        enthalpy,
    )
    enthalpy = np.where(
        above,
        latent_volume
        + rho_water * properties.specific_heat_water_j_kg_k * (temperature - melting),
        enthalpy,
    )
    return np.asarray(enthalpy, dtype=np.float64)


def phase_change_water_target_cells(
    initial_water_volume_cells: float,
    initial_solid_volume_cells: float,
    current_solid_volume_cells: float,
    *,
    density_ice_kg_m3: float,
    density_water_kg_m3: float,
) -> float:
    """Return total physical liquid-water volume after melting or freezing.

    Volumes are expressed in lattice-cell areas for the two-dimensional unit
    depth model.  A decrease of one solid cell generates
    ``rho_ice / rho_water`` water cells; the missing volume for real ice is
    taken up by motion of the water/air free surface.  Use
    :func:`phase_change_active_water_target_cells` when the quantity being
    projected excludes a separate sharp solid mask.
    """

    initial_water = _non_negative(
        "initial_water_volume_cells", initial_water_volume_cells
    )
    initial_solid = _non_negative(
        "initial_solid_volume_cells", initial_solid_volume_cells
    )
    current_solid = _non_negative(
        "current_solid_volume_cells", current_solid_volume_cells
    )
    rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
    rho_water = _positive("density_water_kg_m3", density_water_kg_m3)
    target = initial_water + rho_ice / rho_water * (initial_solid - current_solid)
    if target < -1.0e-12:
        raise ValueError("phase change would require a negative water volume")
    return max(0.0, target)


def phase_change_active_water_target_cells(
    initial_water_volume_cells: float,
    initial_solid_volume_cells: float,
    current_solid_volume_cells: float,
    initial_sharp_geometry_volume_cells: float,
    current_sharp_geometry_volume_cells: float,
    *,
    density_ice_kg_m3: float,
    density_water_kg_m3: float,
) -> float:
    """Return the water target measured only over active LBM cells.

    ``phase_change_water_target_cells`` gives the total physical liquid-water
    volume.  A sharp LBM mask, however, adds or removes whole active cells when
    the continuous liquid fraction crosses its geometry threshold.  This
    target adds the physical ice/water expansion and subtracts that discrete
    geometry change, so a node transition cannot create a projection spike.
    When the sharp geometry volume equals the continuous solid volume, the
    result reduces exactly to the total-water formula.
    """

    initial_water = _non_negative(
        "initial_water_volume_cells", initial_water_volume_cells
    )
    initial_solid = _non_negative(
        "initial_solid_volume_cells", initial_solid_volume_cells
    )
    current_solid = _non_negative(
        "current_solid_volume_cells", current_solid_volume_cells
    )
    initial_geometry = _non_negative(
        "initial_sharp_geometry_volume_cells",
        initial_sharp_geometry_volume_cells,
    )
    current_geometry = _non_negative(
        "current_sharp_geometry_volume_cells",
        current_sharp_geometry_volume_cells,
    )
    rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
    rho_water = _positive("density_water_kg_m3", density_water_kg_m3)
    density_ratio = rho_ice / rho_water
    target = (
        initial_water
        + (1.0 - density_ratio) * (current_solid - initial_solid)
        - (current_geometry - initial_geometry)
    )
    if target < -1.0e-12:
        raise ValueError("phase change would require a negative active water volume")
    return max(0.0, target)


def taichi_available() -> bool:
    """Return whether the optional Taichi package was importable."""

    return ti is not None


if ti is not None:

    @ti.data_oriented
    class EnthalpyFV2D:
        """Taichi finite-volume enthalpy component for an IceFlow2D lattice.

        ``water_phase``, ``wall``, ``solid``, and ``velocity`` remain owned by
        the parent flow solver and are supplied to :meth:`initialize` and
        :meth:`advance`.  Velocity is in lattice cells per LBM step; it is
        converted to m/s before forming the physical advective energy flux.

        ``ice_material`` marks every initial ice or water cell governed by the
        ice/water enthalpy law; initial air remains outside the thermal domain
        when the water/air interface is adiabatic.  It deliberately does not
        move by itself, so a moving free surface or moving ice body must
        conservatively remap this field in a later coupling stage.
        """

        def __init__(
            self,
            nx: int,
            ny: int,
            config: ThermalConfig,
            scales: LatticeScales,
            *,
            density_water_kg_m3: float,
            density_air_kg_m3: float,
            density_ice_kg_m3: float,
        ):
            if isinstance(nx, bool) or int(nx) != nx or int(nx) < 2:
                raise ValueError("nx must be an integer of at least two")
            if isinstance(ny, bool) or int(ny) != ny or int(ny) < 2:
                raise ValueError("ny must be an integer of at least two")
            if not isinstance(config, ThermalConfig):
                raise TypeError("config must be a ThermalConfig")
            if not isinstance(scales, LatticeScales):
                raise TypeError("scales must be a LatticeScales")

            self.nx = int(nx)
            self.ny = int(ny)
            self.config = config
            self.scales = scales
            self.steps = 0
            self.time_s = 0.0

            self._rho_water = _positive("density_water_kg_m3", density_water_kg_m3)
            self._rho_air = _positive("density_air_kg_m3", density_air_kg_m3)
            self._rho_ice = _positive("density_ice_kg_m3", density_ice_kg_m3)
            props = config.properties
            self._melting = float(props.melting_temperature_c)
            self._cp_water = float(props.specific_heat_water_j_kg_k)
            self._cp_ice = float(props.specific_heat_ice_j_kg_k)
            self._cp_air = float(props.specific_heat_air_j_kg_k)
            self._k_water = float(props.conductivity_water_w_m_k)
            self._k_ice = float(props.conductivity_ice_w_m_k)
            self._k_air = float(props.conductivity_air_w_m_k)
            self._latent = float(props.latent_heat_j_kg)
            self._latent_ice_volume = self._rho_ice * self._latent
            self._latent_water_volume = self._rho_water * self._latent
            self._velocity_scale = float(scales.velocity_scale_m_s)
            self._dx = float(scales.dx_m)
            self._advection_enabled = bool(config.advection_enabled)
            self._water_air_adiabatic = bool(config.water_air_interface_adiabatic)

            boundaries = config.boundaries
            self._left_kind = _BOUNDARY_CODE[boundaries.left.kind]
            self._right_kind = _BOUNDARY_CODE[boundaries.right.kind]
            self._bottom_kind = _BOUNDARY_CODE[boundaries.bottom.kind]
            self._top_kind = _BOUNDARY_CODE[boundaries.top.kind]
            self._left_value = float(boundaries.left.value)
            self._right_value = float(boundaries.right.value)
            self._bottom_value = float(boundaries.bottom.value)
            self._top_value = float(boundaries.top.value)

            shape = (self.nx, self.ny)
            # f64 is intentional: for physical diffusivities the enthalpy
            # increment of one LBM step can be below f32 resolution.
            self.enthalpy_j_m3 = ti.field(ti.f64, shape=shape)
            self.temperature_c = ti.field(ti.f64, shape=shape)
            self.liquid_fraction = ti.field(ti.f32, shape=shape)
            self.conductivity_w_m_k = ti.field(ti.f64, shape=shape)
            self.ice_material = ti.field(ti.i8, shape=shape)
            self.flux_x_w_m2 = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
            self.flux_y_w_m2 = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
            self.boundary_power_w_m = ti.field(ti.f64, shape=())
            self.boundary_heat_input_j_m = ti.field(ti.f64, shape=())
            self._enthalpy_sum_j_m = ti.field(ti.f64, shape=())
            self._solid_volume_cells = ti.field(ti.f64, shape=())

            alpha_water = self._k_water / (self._rho_water * self._cp_water)
            alpha_air = self._k_air / (self._rho_air * self._cp_air)
            alpha_ice = self._k_ice / (self._rho_ice * self._cp_ice)
            self.maximum_diffusivity_m2_s = max(alpha_water, alpha_ice)
            if not self._water_air_adiabatic:
                self.maximum_diffusivity_m2_s = max(
                    self.maximum_diffusivity_m2_s, alpha_air
                )
            self.maximum_diffusion_time_step_s = (
                config.max_fourier_number
                * self._dx
                * self._dx
                / self.maximum_diffusivity_m2_s
            )

        def maximum_advection_time_step_s(
            self, max_velocity_lattice_l1: float
        ) -> float:
            """Return the upwind CFL limit for a lattice L1 speed bound."""

            speed = _non_negative("max_velocity_lattice_l1", max_velocity_lattice_l1)
            if not self._advection_enabled or speed == 0.0:
                return math.inf
            return self.config.max_courant_number * self.scales.dt_s / speed

        def required_substeps(
            self,
            time_step_s: float,
            *,
            max_velocity_lattice_l1: float | None = None,
        ) -> int:
            """Return a conservative explicit conduction/advection substep count.

            Supplying a velocity bound accounts for the combined diffusion and
            advection monotonicity budget.  If it is omitted, only the
            diffusion requirement can be checked without synchronously reading
            the parent solver's velocity field back to Python.
            """

            dt = _positive("time_step_s", time_step_s)
            normalized_load = dt / self.maximum_diffusion_time_step_s
            if max_velocity_lattice_l1 is not None:
                advection_limit = self.maximum_advection_time_step_s(
                    max_velocity_lattice_l1
                )
                if math.isfinite(advection_limit):
                    normalized_load += dt / advection_limit
            return max(1, int(math.ceil(normalized_load - 1.0e-14)))

        def initialize(self, water_phase: Any, wall: Any, solid: Any) -> None:
            """Initialize physical enthalpy from the parent flow fields."""

            self._initialize_fields(water_phase, wall, solid)
            self._reset_diagnostics()
            self.steps = 0
            self.time_s = 0.0

        def advance(
            self,
            time_step_s: float,
            velocity: Any,
            water_phase: Any,
            wall: Any,
            solid: Any,
            *,
            max_velocity_lattice_l1: float | None = None,
        ) -> int:
            """Advance by a physical interval and return the substep count.

            ``max_velocity_lattice_l1`` should be supplied by the parent from a
            known low-Mach cap when strict combined CFL/Fourier enforcement is
            desired.  Omitting it avoids a device-to-host velocity reduction.
            """

            dt = _positive("time_step_s", time_step_s)
            substeps = self.required_substeps(
                dt, max_velocity_lattice_l1=max_velocity_lattice_l1
            )
            sub_dt = dt / substeps
            # The Allen--Cahn water/air phase can change between calls.  Sync
            # temperature and conductivity with the current phase before the
            # first face flux is formed; subsequent substeps recover below.
            self._recover_state(water_phase, wall)
            for _ in range(substeps):
                self._reset_boundary_power()
                self._compute_flux_x(velocity, water_phase, wall, solid)
                self._compute_flux_y(velocity, water_phase, wall, solid)
                self._update_enthalpy(sub_dt, wall)
                self._accumulate_boundary_heat(sub_dt)
                self._recover_state(water_phase, wall)
            self.steps += substeps
            self.time_s += dt
            return substeps

        def total_enthalpy_j_m(self, wall: Any) -> float:
            """Return energy per unit out-of-plane depth over non-wall cells."""

            self._reduce_enthalpy(wall)
            return float(self._enthalpy_sum_j_m[None])

        def solid_volume_cells(self) -> float:
            """Return ``sum(1-lambda)`` over phase-change material cells."""

            self._reduce_solid_volume()
            return float(self._solid_volume_cells[None])

        @ti.kernel
        def _initialize_fields(
            self,
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            melting = ti.static(self._melting)
            rho_water = ti.static(self._rho_water)
            rho_air = ti.static(self._rho_air)
            rho_ice = ti.static(self._rho_ice)
            cp_water = ti.static(self._cp_water)
            cp_air = ti.static(self._cp_air)
            cp_ice = ti.static(self._cp_ice)
            k_water = ti.static(self._k_water)
            k_ice = ti.static(self._k_ice)
            k_air = ti.static(self._k_air)
            latent_ice = ti.static(self._latent_ice_volume)
            latent_water = ti.static(self._latent_water_volume)
            initial_water = ti.static(float(self.config.initial_water_temperature_c))
            initial_air = ti.static(float(self.config.initial_air_temperature_c))
            initial_ice = ti.static(float(self.config.initial_ice_temperature_c))
            for i, j in self.enthalpy_j_m3:
                if wall[i, j] != 0:
                    self.enthalpy_j_m3[i, j] = 0.0
                    self.temperature_c[i, j] = melting
                    self.liquid_fraction[i, j] = 0.0
                    self.conductivity_w_m_k[i, j] = 0.0
                    self.ice_material[i, j] = ti.cast(0, ti.i8)
                elif solid[i, j] != 0:
                    self.enthalpy_j_m3[i, j] = (
                        rho_ice * cp_ice * (initial_ice - melting)
                    )
                    self.temperature_c[i, j] = initial_ice
                    self.liquid_fraction[i, j] = 0.0
                    self.conductivity_w_m_k[i, j] = k_ice
                    self.ice_material[i, j] = ti.cast(1, ti.i8)
                else:
                    phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                    if phase >= 0.5:
                        self.enthalpy_j_m3[i, j] = latent_ice + (
                            rho_water * cp_water * (initial_water - melting)
                        )
                        self.temperature_c[i, j] = initial_water
                        self.liquid_fraction[i, j] = 1.0
                        self.conductivity_w_m_k[i, j] = k_water
                        self.ice_material[i, j] = ti.cast(1, ti.i8)
                    else:
                        air_energy = rho_air * cp_air * (initial_air - melting)
                        if ti.static(self._water_air_adiabatic):
                            # The first coupled model treats phi<0.5 as a
                            # thermally inactive air reservoir.  Ignoring later
                            # diffuse-interface jitter here prevents a changing
                            # phi from inventing latent heat in that reservoir.
                            self.enthalpy_j_m3[i, j] = air_energy
                            self.temperature_c[i, j] = initial_air
                            self.conductivity_w_m_k[i, j] = k_air
                        else:
                            # For a conductive air domain, retain the resolved
                            # sub-threshold water fraction in the mixture
                            # enthalpy and use the same reference in recovery.
                            water_energy = latent_water + rho_water * cp_water * (
                                initial_water - melting
                            )
                            capacity = (
                                phase * rho_water * cp_water
                                + (1.0 - phase) * rho_air * cp_air
                            )
                            mixture_enthalpy = (
                                phase * water_energy + (1.0 - phase) * air_energy
                            )
                            self.enthalpy_j_m3[i, j] = mixture_enthalpy
                            self.temperature_c[i, j] = melting + (
                                mixture_enthalpy - phase * latent_water
                            ) / ti.max(capacity, 1.0e-30)
                            self.conductivity_w_m_k[i, j] = (
                                phase * k_water + (1.0 - phase) * k_air
                            )
                        self.liquid_fraction[i, j] = 1.0
                        self.ice_material[i, j] = ti.cast(0, ti.i8)

        @ti.kernel
        def _reset_diagnostics(self):
            self.boundary_power_w_m[None] = 0.0
            self.boundary_heat_input_j_m[None] = 0.0
            self._enthalpy_sum_j_m[None] = 0.0

        @ti.kernel
        def _reset_boundary_power(self):
            self.boundary_power_w_m[None] = 0.0

        @ti.kernel
        def _compute_flux_x(
            self,
            velocity: ti.template(),
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            nx = ti.static(self.nx)
            dx = ti.static(self._dx)
            velocity_scale = ti.static(self._velocity_scale)
            melting = ti.static(self._melting)
            rho_water_cp = ti.static(self._rho_water * self._cp_water)
            left_value = ti.static(self._left_value)
            right_value = ti.static(self._right_value)
            for face_i, j in self.flux_x_w_m2:
                flux = ti.cast(0.0, ti.f64)
                boundary_inward = ti.cast(0.0, ti.f64)
                if 0 < face_i < nx:
                    left_i = face_i - 1
                    right_i = face_i
                    left_wall = wall[left_i, j] != 0
                    right_wall = wall[right_i, j] != 0
                    if not left_wall and not right_wall:
                        left_material = self.ice_material[left_i, j] != 0
                        right_material = self.ice_material[right_i, j] != 0
                        conduct_face = True
                        if ti.static(self._water_air_adiabatic):
                            conduct_face = left_material and right_material
                        if conduct_face:
                            k_left = self.conductivity_w_m_k[left_i, j]
                            k_right = self.conductivity_w_m_k[right_i, j]
                            k_face = (
                                2.0
                                * k_left
                                * k_right
                                / ti.max(k_left + k_right, 1.0e-30)
                            )
                            flux = (
                                -k_face
                                * (
                                    self.temperature_c[right_i, j]
                                    - self.temperature_c[left_i, j]
                                )
                                / dx
                            )
                            if ti.static(self._advection_enabled):
                                if solid[left_i, j] == 0 and solid[right_i, j] == 0:
                                    speed = (
                                        0.5
                                        * (
                                            velocity[left_i, j].x
                                            + velocity[right_i, j].x
                                        )
                                        * velocity_scale
                                    )
                                    upwind_temperature = self.temperature_c[right_i, j]
                                    if speed >= 0.0:
                                        upwind_temperature = self.temperature_c[
                                            left_i, j
                                        ]
                                    upwind_capacity = rho_water_cp
                                    if speed >= 0.0:
                                        if self.ice_material[left_i, j] == 0:
                                            upwind_phase = ti.min(
                                                1.0,
                                                ti.max(0.0, water_phase[left_i, j]),
                                            )
                                            upwind_capacity = (
                                                upwind_phase * rho_water_cp
                                                + (1.0 - upwind_phase)
                                                * ti.static(
                                                    self._rho_air * self._cp_air
                                                )
                                            )
                                    else:
                                        if self.ice_material[right_i, j] == 0:
                                            upwind_phase = ti.min(
                                                1.0,
                                                ti.max(0.0, water_phase[right_i, j]),
                                            )
                                            upwind_capacity = (
                                                upwind_phase * rho_water_cp
                                                + (1.0 - upwind_phase)
                                                * ti.static(
                                                    self._rho_air * self._cp_air
                                                )
                                            )
                                    sensible_enthalpy = upwind_capacity * (
                                        upwind_temperature - melting
                                    )
                                    flux += speed * sensible_enthalpy
                    elif left_wall and not right_wall:
                        apply_boundary = True
                        if ti.static(self._water_air_adiabatic):
                            apply_boundary = self.ice_material[right_i, j] != 0
                        if apply_boundary:
                            k_cell = self.conductivity_w_m_k[right_i, j]
                            if ti.static(self._left_kind == _DIRICHLET):
                                flux = (
                                    2.0
                                    * k_cell
                                    * (left_value - self.temperature_c[right_i, j])
                                    / dx
                                )
                            elif ti.static(self._left_kind == _NEUMANN):
                                flux = left_value
                        boundary_inward = flux
                    elif not left_wall and right_wall:
                        apply_boundary = True
                        if ti.static(self._water_air_adiabatic):
                            apply_boundary = self.ice_material[left_i, j] != 0
                        if apply_boundary:
                            k_cell = self.conductivity_w_m_k[left_i, j]
                            if ti.static(self._right_kind == _DIRICHLET):
                                flux = (
                                    2.0
                                    * k_cell
                                    * (self.temperature_c[left_i, j] - right_value)
                                    / dx
                                )
                            elif ti.static(self._right_kind == _NEUMANN):
                                flux = -right_value
                        boundary_inward = -flux
                self.flux_x_w_m2[face_i, j] = flux
                if boundary_inward != 0.0:
                    ti.atomic_add(self.boundary_power_w_m[None], boundary_inward * dx)

        @ti.kernel
        def _compute_flux_y(
            self,
            velocity: ti.template(),
            water_phase: ti.template(),
            wall: ti.template(),
            solid: ti.template(),
        ):
            ny = ti.static(self.ny)
            dx = ti.static(self._dx)
            velocity_scale = ti.static(self._velocity_scale)
            melting = ti.static(self._melting)
            rho_water_cp = ti.static(self._rho_water * self._cp_water)
            bottom_value = ti.static(self._bottom_value)
            top_value = ti.static(self._top_value)
            for i, face_j in self.flux_y_w_m2:
                flux = ti.cast(0.0, ti.f64)
                boundary_inward = ti.cast(0.0, ti.f64)
                if 0 < face_j < ny:
                    bottom_j = face_j - 1
                    top_j = face_j
                    bottom_wall = wall[i, bottom_j] != 0
                    top_wall = wall[i, top_j] != 0
                    if not bottom_wall and not top_wall:
                        bottom_material = self.ice_material[i, bottom_j] != 0
                        top_material = self.ice_material[i, top_j] != 0
                        conduct_face = True
                        if ti.static(self._water_air_adiabatic):
                            conduct_face = bottom_material and top_material
                        if conduct_face:
                            k_bottom = self.conductivity_w_m_k[i, bottom_j]
                            k_top = self.conductivity_w_m_k[i, top_j]
                            k_face = (
                                2.0
                                * k_bottom
                                * k_top
                                / ti.max(k_bottom + k_top, 1.0e-30)
                            )
                            flux = (
                                -k_face
                                * (
                                    self.temperature_c[i, top_j]
                                    - self.temperature_c[i, bottom_j]
                                )
                                / dx
                            )
                            if ti.static(self._advection_enabled):
                                if solid[i, bottom_j] == 0 and solid[i, top_j] == 0:
                                    speed = (
                                        0.5
                                        * (
                                            velocity[i, bottom_j].y
                                            + velocity[i, top_j].y
                                        )
                                        * velocity_scale
                                    )
                                    upwind_temperature = self.temperature_c[i, top_j]
                                    if speed >= 0.0:
                                        upwind_temperature = self.temperature_c[
                                            i, bottom_j
                                        ]
                                    upwind_capacity = rho_water_cp
                                    if speed >= 0.0:
                                        if self.ice_material[i, bottom_j] == 0:
                                            upwind_phase = ti.min(
                                                1.0,
                                                ti.max(0.0, water_phase[i, bottom_j]),
                                            )
                                            upwind_capacity = (
                                                upwind_phase * rho_water_cp
                                                + (1.0 - upwind_phase)
                                                * ti.static(
                                                    self._rho_air * self._cp_air
                                                )
                                            )
                                    else:
                                        if self.ice_material[i, top_j] == 0:
                                            upwind_phase = ti.min(
                                                1.0,
                                                ti.max(0.0, water_phase[i, top_j]),
                                            )
                                            upwind_capacity = (
                                                upwind_phase * rho_water_cp
                                                + (1.0 - upwind_phase)
                                                * ti.static(
                                                    self._rho_air * self._cp_air
                                                )
                                            )
                                    sensible_enthalpy = upwind_capacity * (
                                        upwind_temperature - melting
                                    )
                                    flux += speed * sensible_enthalpy
                    elif bottom_wall and not top_wall:
                        apply_boundary = True
                        if ti.static(self._water_air_adiabatic):
                            apply_boundary = self.ice_material[i, top_j] != 0
                        if apply_boundary:
                            k_cell = self.conductivity_w_m_k[i, top_j]
                            if ti.static(self._bottom_kind == _DIRICHLET):
                                flux = (
                                    2.0
                                    * k_cell
                                    * (bottom_value - self.temperature_c[i, top_j])
                                    / dx
                                )
                            elif ti.static(self._bottom_kind == _NEUMANN):
                                flux = bottom_value
                        boundary_inward = flux
                    elif not bottom_wall and top_wall:
                        apply_boundary = True
                        if ti.static(self._water_air_adiabatic):
                            apply_boundary = self.ice_material[i, bottom_j] != 0
                        if apply_boundary:
                            k_cell = self.conductivity_w_m_k[i, bottom_j]
                            if ti.static(self._top_kind == _DIRICHLET):
                                flux = (
                                    2.0
                                    * k_cell
                                    * (self.temperature_c[i, bottom_j] - top_value)
                                    / dx
                                )
                            elif ti.static(self._top_kind == _NEUMANN):
                                flux = -top_value
                        boundary_inward = -flux
                self.flux_y_w_m2[i, face_j] = flux
                if boundary_inward != 0.0:
                    ti.atomic_add(self.boundary_power_w_m[None], boundary_inward * dx)

        @ti.kernel
        def _update_enthalpy(self, time_step_s: ti.f64, wall: ti.template()):
            inverse_dx = ti.static(1.0 / self._dx)
            for i, j in self.enthalpy_j_m3:
                if wall[i, j] == 0:
                    divergence = (
                        self.flux_x_w_m2[i + 1, j]
                        - self.flux_x_w_m2[i, j]
                        + self.flux_y_w_m2[i, j + 1]
                        - self.flux_y_w_m2[i, j]
                    ) * inverse_dx
                    self.enthalpy_j_m3[i, j] -= time_step_s * divergence

        @ti.kernel
        def _accumulate_boundary_heat(self, time_step_s: ti.f64):
            self.boundary_heat_input_j_m[None] += (
                time_step_s * self.boundary_power_w_m[None]
            )

        @ti.kernel
        def _recover_state(self, water_phase: ti.template(), wall: ti.template()):
            melting = ti.static(self._melting)
            latent_ice = ti.static(self._latent_ice_volume)
            latent_water = ti.static(self._latent_water_volume)
            rho_ice_cp = ti.static(self._rho_ice * self._cp_ice)
            rho_water_cp = ti.static(self._rho_water * self._cp_water)
            rho_air_cp = ti.static(self._rho_air * self._cp_air)
            k_ice = ti.static(self._k_ice)
            k_water = ti.static(self._k_water)
            k_air = ti.static(self._k_air)
            for i, j in self.enthalpy_j_m3:
                if wall[i, j] == 0:
                    enthalpy = self.enthalpy_j_m3[i, j]
                    if self.ice_material[i, j] != 0:
                        temperature = ti.cast(melting, ti.f64)
                        fraction = ti.cast(0.0, ti.f64)
                        if enthalpy < 0.0:
                            temperature = melting + enthalpy / rho_ice_cp
                        elif enthalpy <= latent_ice:
                            fraction = enthalpy / latent_ice
                        else:
                            temperature = (
                                melting + (enthalpy - latent_ice) / rho_water_cp
                            )
                            fraction = 1.0
                        self.temperature_c[i, j] = temperature
                        self.liquid_fraction[i, j] = ti.cast(fraction, ti.f32)
                        self.conductivity_w_m_k[i, j] = (
                            1.0 - fraction
                        ) * k_ice + fraction * k_water
                    else:
                        if ti.static(self._water_air_adiabatic):
                            self.temperature_c[i, j] = melting + enthalpy / rho_air_cp
                            self.conductivity_w_m_k[i, j] = k_air
                        else:
                            phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                            capacity = phase * rho_water_cp + (1.0 - phase) * rho_air_cp
                            self.temperature_c[i, j] = melting + (
                                enthalpy - phase * latent_water
                            ) / ti.max(capacity, 1.0e-30)
                            self.conductivity_w_m_k[i, j] = (
                                phase * k_water + (1.0 - phase) * k_air
                            )
                        self.liquid_fraction[i, j] = 1.0
                else:
                    self.temperature_c[i, j] = melting
                    self.liquid_fraction[i, j] = 0.0
                    self.conductivity_w_m_k[i, j] = 0.0

        @ti.kernel
        def _reduce_enthalpy(self, wall: ti.template()):
            self._enthalpy_sum_j_m[None] = 0.0
            cell_area = ti.static(self._dx * self._dx)
            for i, j in self.enthalpy_j_m3:
                if wall[i, j] == 0:
                    ti.atomic_add(
                        self._enthalpy_sum_j_m[None],
                        self.enthalpy_j_m3[i, j] * cell_area,
                    )

        @ti.kernel
        def _reduce_solid_volume(self):
            self._solid_volume_cells[None] = 0.0
            for i, j in self.liquid_fraction:
                if self.ice_material[i, j] != 0:
                    fraction = ti.min(1.0, ti.max(0.0, self.liquid_fraction[i, j]))
                    ti.atomic_add(
                        self._solid_volume_cells[None], 1.0 - ti.cast(fraction, ti.f64)
                    )

else:

    class EnthalpyFV2D:
        """Unavailable Taichi component placeholder.

        Keeping the symbol importable lets CPU-only users construct and test
        all thermal configuration and NumPy reference objects.
        """

        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError(
                "EnthalpyFV2D requires the optional 'taichi' package and an "
                "initialized Taichi runtime"
            )


__all__ = [
    "EnthalpyFV2D",
    "LatticeScales",
    "PhaseChangeProperties",
    "ThermalBoundary",
    "ThermalBoundaryKind",
    "ThermalBoundarySet",
    "ThermalConfig",
    "phase_change_active_water_target_cells",
    "phase_change_enthalpy_numpy",
    "phase_change_water_target_cells",
    "recover_temperature_and_liquid_fraction_numpy",
    "taichi_available",
]
