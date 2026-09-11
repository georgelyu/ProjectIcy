"""Configuration for coupled falling-ice melting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .thermal import ThermalConfig

DEFAULT_REFERENCE_VELOCITY_M_S = 4.0


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


def _default_thermal_config() -> ThermalConfig:
    """Build the insulated-bath defaults without importing Taichi eagerly."""

    from .thermal import ThermalBoundarySet, ThermalConfig

    return ThermalConfig(
        boundaries=ThermalBoundarySet(),
        initial_water_temperature_c=90.0,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=20.0,
        update_interval_lbm_steps=8,
        buoyancy_reference_temperature_c=90.0,
    )


@dataclass(slots=True)
class IceFlowConfig:
    """Validated inputs for one body-ALE falling-and-melting ice run."""

    resolution: tuple[int, int] = (200, 400)
    dx: float = 1.25e-4
    # Fixed physical speed represented by 0.1 lattice cells per LBM step.
    reference_velocity: float = DEFAULT_REFERENCE_VELOCITY_M_S

    # Physical two-phase parameters.
    rho_water: float = 1000.0
    rho_air: float = 1.25
    rho_ice: float = 917.0
    viscosity_water: float = 1.0e-6
    viscosity_air: float = 1.5e-5
    sigma: float = 0.072
    gravity: tuple[float, float] = (0.0, -9.8)
    cd: float = 0.1
    interface_width: float = 5.0
    mobility: float = 0.1
    phase_warmup_steps: int = 500
    thermal: ThermalConfig = field(default_factory=_default_thermal_config)
    # Build a frozen hydrostatic reference, initialize the fluid at rest, and
    # complete dynamic momentum exchange with the matching equilibrium
    # traction.  This is one well-balanced formulation, not an independently
    # added Archimedes force.  No sub-grid quadrature is used.
    well_balanced_hydrostatics: bool = True
    # Liang et al.'s pressure--momentum distribution is used for the
    # large-density-ratio phase-field flow.  Material density remains rho(phi);
    # the populations carry rho*u and pressure rather than imposing the
    # isothermal relation p=cs^2*rho across the air--water jump.

    # Falling-ice geometry in lattice-cell coordinates.
    water_width_fraction: float = 0.9875
    water_height_fraction: float = 0.60125
    ice_width_fraction: float = 0.3225
    ice_height_fraction: float = 0.16125
    ice_base_y_cells: float = 254.6672141068609
    boundary_cells: int = 3

    # Initial state of the freely moving rigid ice rectangle.
    ice_initial_velocity: tuple[float, float] = (0.0, 0.0)
    ice_initial_angle: float = math.radians(5.0)
    ice_initial_angular_velocity: float = 0.0
    # Kept in serialized metadata for compatibility; alternate values are no
    # longer accepted by the specialized solver.
    ice_fixed: bool = False
    rigid_boundary_scheme: Literal["unified"] = "unified"

    # Hydrodynamic drag is resolved by cut-link momentum exchange.  Damping
    # remains an optional, explicitly configured numerical model for the
    # rigid-body state.
    linear_damping: float = 1.0
    angular_damping: float = 0.9990
    # The conservative phase projection is an all-or-nothing constraint, not
    # a relaxation.  The tolerance is relative to max(1, target volume).  A
    # correction that would move the diffuse interface farther than the
    # configured distance is rejected instead of contaminating either bulk
    # phase with an additive source.
    # f64 is used for all global reductions and root arithmetic, while phi
    # itself remains f32; this relative tolerance covers that final storage
    # quantization (about 1e-6 cells in the small regression lattice).
    volume_projection_tolerance: float = 1.0e-8
    # Outside this resolved diffuse-interface range, values are numerical
    # bulk tails and are canonicalized to the exact pure phases before the
    # global constraint is solved.
    volume_projection_interface_cutoff: float = 1.0e-3
    volume_projection_max_shift: float = 0.50
    volume_projection_max_iterations: int = 64
    # Optional endpoint of a conservative shear-relaxation envelope.  The
    # effective floor is interpolated linearly from tau=1/2 in pure water to
    # this value in pure air, and reaches the same endpoint in the one-cell
    # neighborhood of sharp ice cut links.  Only deviatoric stress modes are
    # affected; ``None`` disables the floor.
    air_interface_relaxation_time: float | None = 0.8

    output_dir: str = "outputs/iceflow2d/coupled_falling_ice_melting_2d"
    show_gui: bool = False
    save_npz: bool = False

    def __post_init__(self) -> None:
        if self.rigid_boundary_scheme != "unified":
            raise ValueError("rigid_boundary_scheme must be 'unified'")

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
        from .thermal import ThermalConfig

        if not isinstance(self.thermal, ThermalConfig):
            raise TypeError("thermal must be a ThermalConfig")

        _positive("dx", self.dx)
        self.reference_velocity = _positive(
            "reference_velocity", self.reference_velocity
        )
        _positive("rho_water", self.rho_water)
        _positive("rho_air", self.rho_air)
        _positive("rho_ice", self.rho_ice)
        _positive("viscosity_water", self.viscosity_water)
        _positive("viscosity_air", self.viscosity_air)
        if _finite("sigma", self.sigma) < 0.0:
            raise ValueError("sigma must be non-negative")
        if _finite("cd", self.cd) < 0.0:
            raise ValueError("cd must be non-negative")
        _positive("interface_width", self.interface_width)
        _positive("mobility", self.mobility)
        if len(self.gravity) != 2:
            raise ValueError("gravity must contain exactly two components")
        for index, component in enumerate(self.gravity):
            _finite(f"gravity[{index}]", component)

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
        _positive("volume_projection_tolerance", self.volume_projection_tolerance)
        interface_cutoff = _finite(
            "volume_projection_interface_cutoff",
            self.volume_projection_interface_cutoff,
        )
        if not 0.0 < interface_cutoff < 0.5:
            raise ValueError("volume_projection_interface_cutoff must be in (0, 0.5)")
        _positive("volume_projection_max_shift", self.volume_projection_max_shift)
        if (
            isinstance(self.volume_projection_max_iterations, bool)
            or int(self.volume_projection_max_iterations)
            != self.volume_projection_max_iterations
        ):
            raise ValueError("volume_projection_max_iterations must be an integer")
        self.volume_projection_max_iterations = int(
            self.volume_projection_max_iterations
        )
        if self.volume_projection_max_iterations < 1:
            raise ValueError("volume_projection_max_iterations must be positive")
        if self.air_interface_relaxation_time is not None:
            if isinstance(self.air_interface_relaxation_time, bool):
                raise ValueError("air_interface_relaxation_time must be a real number")
            air_interface_tau = _finite(
                "air_interface_relaxation_time",
                self.air_interface_relaxation_time,
            )
            if not 0.5 < air_interface_tau <= 2.0:
                raise ValueError("air_interface_relaxation_time must be in (0.5, 2]")
            self.air_interface_relaxation_time = air_interface_tau

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
            if self.water_width != nx - self.boundary_cells:
                raise ValueError(
                    "hydrostatic reference requires a full-width horizontal pool"
                )
            full_height = self.water_height == ny - self.boundary_cells
            if not full_height and abs(float(self.gravity[0])) > 1.0e-14:
                raise ValueError(
                    "a two-phase hydrostatic reference requires vertical gravity"
                )
        # Check the initially rotated rectangle against the physical walls.
        c = abs(math.cos(float(self.ice_initial_angle)))
        s = abs(math.sin(float(self.ice_initial_angle)))
        extent_x = c * self.ice_width * 0.5 + s * self.ice_height * 0.5
        extent_y = s * self.ice_width * 0.5 + c * self.ice_height * 0.5
        cx, cy = self.ice_initial_center
        lower_contact = float(self.boundary_cells)
        upper_x_contact = float(nx - self.boundary_cells)
        upper_y_contact = float(ny - self.boundary_cells)
        if cx - extent_x < lower_contact or cx + extent_x > upper_x_contact:
            raise ValueError("initial ice rectangle overlaps a side wall")
        if cy - extent_y < lower_contact - 1.0e-9 or cy + extent_y > upper_y_contact:
            raise ValueError("initial ice rectangle overlaps the bottom or top wall")

        if self.thermal.moving_body_scheme != "body_ale":
            raise ValueError("thermal moving_body_scheme must be 'body_ale'")
        if not self.thermal.water_air_interface_adiabatic:
            raise ValueError("thermal water/air interface must be adiabatic")
        if self.water_height >= ny - self.boundary_cells:
            raise ValueError(
                "thermal phase change requires a water/air free surface for "
                "continuous phase-volume projection"
            )

        # Central-moment collision requires finite tau > 0.5.  Large physical
        # viscosities can legitimately map to tau > 2 when a deliberately low
        # fixed reference velocity is used, so no arbitrary upper bound on
        # tau is imposed here.
        for name, viscosity in (
            ("viscosity_water", float(self.viscosity_water)),
            ("viscosity_air", float(self.viscosity_air)),
        ):
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
    def ice_width(self) -> int:
        return max(2, int(self.nx * self.ice_width_fraction))

    @property
    def ice_height(self) -> int:
        return max(2, int(self.ny * self.ice_height_fraction))

    @property
    def ice_initial_center(self) -> tuple[float, float]:
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

    def to_dict(self) -> dict:
        return asdict(self)


def create_iceflow_config(**overrides) -> IceFlowConfig:
    valid_names = {item.name for item in fields(IceFlowConfig)}
    for key in overrides:
        if key not in valid_names:
            raise TypeError(f"Unknown IceFlowConfig option: {key}")
    return IceFlowConfig(**overrides)
