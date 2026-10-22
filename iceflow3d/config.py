"""Validated SI/LU configuration. y is vertical; z is the tank depth."""

from dataclasses import asdict, dataclass, field
import math

from .rigid import rotation_matrix, separating_axis
from .viscosity import air_dynamic_viscosity_pa_s, water_dynamic_viscosity_pa_s


def finite(name, value, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return value


def integer(name, value, minimum=1):
    number = finite(name, value)
    if int(number) != number or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def vector(name, value, length=3):
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} components")
    return tuple(finite(name, v) for v in value)


@dataclass(frozen=True)
class PhaseChangeProperties:
    melting_temperature_c: float = 0.0
    specific_heat_water_j_kg_k: float = 4186.0
    specific_heat_ice_j_kg_k: float = 2100.0
    conductivity_water_w_m_k: float = 0.60
    conductivity_ice_w_m_k: float = 2.20
    latent_heat_j_kg: float = 334000.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                finite(
                    name, getattr(self, name), positive=name != "melting_temperature_c"
                ),
            )


@dataclass(frozen=True)
class ThermalConfig:
    properties: PhaseChangeProperties = field(default_factory=PhaseChangeProperties)
    initial_water_temperature_c: float = 90.0
    initial_ice_temperature_c: float = 0.0
    initial_air_temperature_c: float = 20.0
    update_interval_lbm_steps: int = 8
    max_fourier_number: float = 0.10
    max_courant_number: float = 0.5
    max_substeps_per_update: int = 128
    solid_liquid_threshold: float = 0.5
    advection_enabled: bool = True
    thermal_expansion_water_1_k: float = 2.1e-4

    def __post_init__(self):
        if not isinstance(self.properties, PhaseChangeProperties):
            raise TypeError("properties must be PhaseChangeProperties")
        for name in (
            "initial_water_temperature_c",
            "initial_ice_temperature_c",
            "initial_air_temperature_c",
        ):
            object.__setattr__(self, name, finite(name, getattr(self, name)))
        tm = self.properties.melting_temperature_c
        if self.initial_water_temperature_c < tm or self.initial_ice_temperature_c > tm:
            raise ValueError(
                "initial ice must be <= melting point and water >= melting point"
            )
        water_dynamic_viscosity_pa_s(self.initial_water_temperature_c)
        water_dynamic_viscosity_pa_s(tm)
        air_dynamic_viscosity_pa_s(self.initial_air_temperature_c)
        for name in ("update_interval_lbm_steps", "max_substeps_per_update"):
            object.__setattr__(self, name, integer(name, getattr(self, name)))
        for name, limit in (
            ("max_fourier_number", 1 / 6),
            ("max_courant_number", 1.0),
            ("solid_liquid_threshold", 1.0),
        ):
            value = finite(name, getattr(self, name), positive=True)
            if value > limit or name == "solid_liquid_threshold" and value == 1:
                raise ValueError(f"{name} is outside its stable/physical range")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "thermal_expansion_water_1_k",
            finite(
                "thermal_expansion_water_1_k",
                self.thermal_expansion_water_1_k,
                nonnegative=True,
            ),
        )
        if not isinstance(self.advection_enabled, bool):
            raise ValueError("advection_enabled must be boolean")


@dataclass(frozen=True)
class IceBodyConfig:
    center_cells: tuple[float, float, float]
    size_cells: tuple[int, int, int] = (30, 30, 30)
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity_lattice: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_lattice: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temperature_c: float | None = None

    def __post_init__(self):
        for name in ("center_cells", "velocity_lattice", "angular_velocity_lattice"):
            object.__setattr__(self, name, vector(name, getattr(self, name)))
        if len(self.size_cells) != 3:
            raise ValueError("size_cells must contain three sizes")
        object.__setattr__(
            self,
            "size_cells",
            tuple(integer("size_cells", n, 2) for n in self.size_cells),
        )
        q = vector("quaternion_wxyz", self.quaternion_wxyz, 4)
        norm = math.sqrt(sum(v * v for v in q))
        if norm < 1e-12:
            raise ValueError("quaternion must be nonzero")
        object.__setattr__(self, "quaternion_wxyz", tuple(v / norm for v in q))
        if self.temperature_c is not None:
            object.__setattr__(
                self, "temperature_c", finite("temperature_c", self.temperature_c)
            )


@dataclass(frozen=True)
class LatticeScales:
    dx_m: float
    dt_s: float
    reference_density_kg_m3: float

    @classmethod
    def from_iceflow_config(cls, config):
        dx = config.domain_size_m[0] / config.nx
        return cls(
            dx,
            dx * config.reference_lattice_velocity / config.reference_velocity_m_s,
            config.rho_water,
        )

    @property
    def velocity_m_s(self):
        return self.dx_m / self.dt_s

    @property
    def mass_kg(self):
        return self.reference_density_kg_m3 * self.dx_m**3

    @property
    def pressure_pa(self):
        return self.reference_density_kg_m3 * self.velocity_m_s**2

    @property
    def force_n(self):
        return self.mass_kg * self.dx_m / self.dt_s**2


@dataclass(frozen=True)
class IceFlowConfig:
    nx: int = 450
    ny: int = 150
    nz: int = 90
    domain_size_m: tuple[float, float, float] = (0.15, 0.05, 0.03)
    boundary_cells: int = 3
    water_width: int = 198
    water_height: int = 138
    floor_water_depth_cells: int = 31
    ice_bodies: tuple[IceBodyConfig, ...] = field(
        default_factory=lambda: tuple(
            IceBodyConfig((x, 21.0, 45.0)) for x in (255.0, 309.0, 363.0, 417.0)
        )
    )
    rho_water: float = 1000.0
    rho_air: float = 1.25
    rho_ice: float = 917.0
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    reference_velocity_m_s: float = 4.0
    reference_lattice_velocity: float = 0.1
    surface_tension_n_m: float = 0.072
    interface_width: float = 5.0
    mobility: float = 0.1
    momentum_collision_model: str = "cm_mrt"
    hrr_kinetic_blend: float = 0.0
    hrr_cubic_correction: bool = True
    # Regularize weak bulk gradients in the phase-compression normal (LU).
    # Match the 2D four-ice example; larger values also weaken interface tails.
    phase_normal_epsilon: float = 0.01
    phase_warmup_steps: int = 2000
    cm_acoustic_relaxation_rate: float = 0.6
    # Kept for saved configuration compatibility; the unified gradient has
    # no density-only limiter. A nonzero historical setting is rejected.
    density_gradient_source_limit: float = 0.0
    # Historical diagnostic: limiting J in momentum alone breaks consistency.
    phase_flux_acceleration_limit: float = 0.0
    gas_state_repair: bool = False
    # Kept in saved configuration schemas; the global correction is disabled.
    conserve_phase_volume: bool = False
    check_stability: bool = True
    maximum_velocity_lattice: float = 0.5
    # Coupled body/wall contacts can need more than 32 sweeps on the fine grid.
    # Converged contacts still return early with the original residual limits.
    body_contact_iterations: int = 128
    linear_damping: float = 1.0
    angular_damping: float = 1.0
    backend: str = "cuda"
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    def __post_init__(self):
        import numpy as np

        for name in ("nx", "ny", "nz", "boundary_cells", "body_contact_iterations"):
            object.__setattr__(self, name, integer(name, getattr(self, name)))
        object.__setattr__(
            self,
            "phase_warmup_steps",
            integer("phase_warmup_steps", self.phase_warmup_steps, 0),
        )
        object.__setattr__(
            self, "domain_size_m", vector("domain_size_m", self.domain_size_m)
        )
        if min(self.domain_size_m) <= 0:
            raise ValueError("domain sizes must be positive")
        sizes = (self.nx, self.ny, self.nz)
        dx = self.domain_size_m[0] / self.nx
        if any(
            not math.isclose(length / n, dx, rel_tol=1e-10)
            for n, length in zip(sizes, self.domain_size_m)
        ):
            raise ValueError("D3Q27 requires cubic cells (dx = dy = dz)")
        if min(sizes) <= 2 * self.boundary_cells + 3:
            raise ValueError("grid is too small for the walls")
        for name in (
            "rho_water",
            "rho_air",
            "rho_ice",
            "reference_velocity_m_s",
            "reference_lattice_velocity",
            "interface_width",
            "mobility",
            "maximum_velocity_lattice",
        ):
            object.__setattr__(
                self, name, finite(name, getattr(self, name), positive=True)
            )
        if self.rho_air >= self.rho_water:
            raise ValueError("rho_air must be smaller than rho_water")
        for name in (
            "surface_tension_n_m",
            "phase_normal_epsilon",
            "density_gradient_source_limit",
            "phase_flux_acceleration_limit",
        ):
            object.__setattr__(
                self, name, finite(name, getattr(self, name), nonnegative=True)
            )
        if self.density_gradient_source_limit != 0:
            raise ValueError(
                "unified wall-mirror gradient requires density_gradient_source_limit=0"
            )
        for name in ("linear_damping", "angular_damping"):
            value = finite(name, getattr(self, name), nonnegative=True)
            if value > 1:
                raise ValueError(f"{name} must be <= 1")
            object.__setattr__(self, name, value)
        rate = finite("cm_acoustic_relaxation_rate", self.cm_acoustic_relaxation_rate)
        if not 0 < rate < 2:
            raise ValueError("cm_acoustic_relaxation_rate must be in (0, 2)")
        object.__setattr__(self, "cm_acoustic_relaxation_rate", rate)
        object.__setattr__(self, "gravity", vector("gravity", self.gravity))
        if self.gravity[0] != 0 or self.gravity[2] != 0 or self.gravity[1] > 0:
            raise ValueError(
                "hydrostatic reference requires gravity in the -y direction"
            )
        for name in ("water_width", "water_height", "floor_water_depth_cells"):
            object.__setattr__(self, name, integer(name, getattr(self, name), 0))
        b = self.boundary_cells
        if not b < self.water_width <= self.nx - b:
            raise ValueError("water_width is outside the inner walls")
        if not b < self.water_height < self.ny - b:
            raise ValueError("water_height must leave an air cap")
        if not 0 <= self.floor_water_depth_cells < self.water_height - b:
            raise ValueError("floor water must be lower than the column")
        if self.backend not in ("cuda", "cpu"):
            raise ValueError("backend must be cuda or cpu")
        if self.momentum_collision_model not in ("cm_mrt", "cm_hrr"):
            raise ValueError("momentum_collision_model must be cm_mrt or cm_hrr")
        blend = finite("hrr_kinetic_blend", self.hrr_kinetic_blend)
        if not 0 <= blend <= 1:
            raise ValueError("hrr_kinetic_blend must be in [0, 1]")
        object.__setattr__(self, "hrr_kinetic_blend", blend)
        for name in (
            "check_stability",
            "gas_state_repair",
            "hrr_cubic_correction",
            "conserve_phase_volume",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.thermal, ThermalConfig):
            raise TypeError("thermal must be ThermalConfig")
        bodies = tuple(self.ice_bodies)
        if not bodies or not all(isinstance(body, IceBodyConfig) for body in bodies):
            raise ValueError("ice_bodies must contain IceBodyConfig objects")
        object.__setattr__(self, "ice_bodies", bodies)
        for i, body in enumerate(bodies):
            rotation = rotation_matrix(body.quaternion_wxyz)
            half = np.asarray(body.size_cells) / 2
            extent = abs(rotation) @ half
            center = np.asarray(body.center_cells)
            if np.any(center - extent < b - 1e-9) or np.any(
                center + extent > np.asarray(sizes) - b + 1e-9
            ):
                raise ValueError("initial ice overlaps a wall")
            if (
                body.temperature_c is not None
                and body.temperature_c > self.thermal.properties.melting_temperature_c
            ):
                raise ValueError("initial ice temperature exceeds melting point")
            for other in bodies[:i]:
                hit = separating_axis(
                    center,
                    rotation,
                    half,
                    np.asarray(other.center_cells),
                    rotation_matrix(other.quaternion_wxyz),
                    np.asarray(other.size_cells) / 2,
                )
                if hit is not None and hit[1] > 1e-8:
                    raise ValueError("initial ice bodies overlap")

    @property
    def bodies(self):
        return self.ice_bodies

    def to_dict(self):
        return asdict(self)


def create_iceflow_config(**overrides):
    return IceFlowConfig(**overrides)
