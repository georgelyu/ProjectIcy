"""Configuration for the 2D ice--two-phase-flow and optional thermal model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .thermal import ThermalConfig

RigidBoundaryScheme = Literal["unified", "halfway"]


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


@dataclass(slots=True)
class IceFlowConfig:
    """Validated inputs for a sharp-boundary rigid-ice dam-break run.

    The model contains no MPM or porosity/epsilon.  Thermal phase change is an
    optional fixed-body coupling; leaving ``thermal=None`` preserves the
    original non-thermal solver path.
    """

    resolution: tuple[int, int] = (600, 300)
    dx: float = 0.01
    reference_length_cells: int = 300

    # Physical two-phase parameters.
    rho_water: float = 1000.0
    rho_air: float = 1.25
    rho_ice: float = 917.0
    viscosity_water: float = 1.0e-4
    viscosity_air: float = 1.0e-2
    sigma: float = 0.072
    gravity: tuple[float, float] = (0.0, -9.8)
    cd: float = 0.1
    interface_width: float = 5.0
    mobility: float = 0.1
    phase_warmup_steps: int = 500
    thermal: ThermalConfig | None = None
    # Build a frozen hydrostatic reference, initialize the fluid at rest, and
    # complete dynamic momentum exchange with the matching equilibrium
    # traction.  This is one well-balanced formulation, not an independently
    # added Archimedes force.  No sub-grid quadrature is used.
    well_balanced_hydrostatics: bool = False
    # Liang et al.'s pressure--momentum distribution is used for the
    # large-density-ratio phase-field flow.  Material density remains rho(phi);
    # the populations carry rho*u and pressure rather than imposing the
    # isothermal relation p=cs^2*rho across the air--water jump.

    # Dam-break and rigid-body geometry, in lattice-cell coordinates.
    water_width_fraction: float = 0.25
    water_height_fraction: float = 2.0 / 3.0
    ice_width_fraction: float = 0.15
    ice_height_fraction: float = 0.30
    ice_base_y_cells: float = 3.0
    boundary_cells: int = 3

    # One rigid ice rectangle, either freely moving or held at its initial pose.
    ice_initial_velocity: tuple[float, float] = (0.0, 0.0)
    ice_initial_angle: float = 0.0
    ice_initial_angular_velocity: float = 0.0
    ice_fixed: bool = False
    rigid_boundary_scheme: RigidBoundaryScheme = "unified"

    # Rigid-wall collision controls.  Speeds are lattice speeds per LBM step;
    # the caps preserve the low-Mach operating range.
    # Hydrodynamic drag is already resolved by cut-link momentum exchange.
    # Avoid artificial per-step loss of rigid translational momentum by
    # default; values below one remain available only as an explicitly
    # requested numerical stabilizer.
    linear_damping: float = 1.0
    angular_damping: float = 0.9990
    max_ice_speed: float = 0.08
    max_ice_angular_speed: float = 1.0e-3
    wall_restitution: float = 0.10
    wall_friction: float = 0.10
    # Coulomb coefficient for the smooth/wet floor.  This deliberately small
    # baseline adds measurable contact friction while allowing dam-break flow
    # to make the ice slide.  It is a scenario parameter, not a universal
    # ice--substrate material constant.
    bottom_wall_friction: float = 0.03

    # The conservative phase projection is an all-or-nothing constraint, not
    # a relaxation.  The tolerance is relative to max(1, target volume).  A
    # correction that would move the diffuse interface farther than the
    # configured distance is rejected instead of contaminating either bulk
    # phase with an additive source.
    # f64 is used for all global reductions and root arithmetic, while phi
    # itself remains f32; this relative tolerance covers that final storage
    # quantization (about 1e-6 cells in the small regression lattice).
    volume_projection_tolerance: float = 1.0e-9
    # Outside this resolved diffuse-interface range, values are numerical
    # bulk tails and are canonicalized to the exact pure phases before the
    # global constraint is solved.
    volume_projection_interface_cutoff: float = 1.0e-3
    volume_projection_max_shift: float = 0.50
    volume_projection_max_iterations: int = 64

    output_dir: str = "outputs/iceflow2d/coupled_ice"
    show_gui: bool = False
    save_npz: bool = False

    def __post_init__(self) -> None:
        if self.rigid_boundary_scheme not in ("unified", "halfway"):
            raise ValueError("rigid_boundary_scheme must be 'unified' or 'halfway'")

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
            isinstance(self.reference_length_cells, bool)
            or int(self.reference_length_cells) != self.reference_length_cells
        ):
            raise ValueError("reference_length_cells must be an integer")
        self.reference_length_cells = int(self.reference_length_cells)
        if self.reference_length_cells <= 0:
            raise ValueError("reference_length_cells must be positive")
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
        if not isinstance(self.well_balanced_hydrostatics, bool):
            raise ValueError("well_balanced_hydrostatics must be a boolean")
        if self.thermal is not None:
            # Keep config.py loadable as a standalone file for the existing
            # geometry-only examples; the package-relative thermal module is
            # needed only when thermal coupling is actually requested.
            from .thermal import ThermalConfig

            if not isinstance(self.thermal, ThermalConfig):
                raise TypeError("thermal must be a ThermalConfig or None")

        _positive("dx", self.dx)
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
        initial_omega = _finite(
            "ice_initial_angular_velocity", self.ice_initial_angular_velocity
        )
        if not isinstance(self.ice_fixed, bool):
            raise ValueError("ice_fixed must be a boolean")
        if self.ice_fixed and (
            math.hypot(*initial_velocity) > 0.0 or abs(initial_omega) > 0.0
        ):
            raise ValueError(
                "fixed ice must have zero initial linear and angular velocity"
            )

        for name in ("linear_damping", "angular_damping"):
            value = _finite(name, getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        max_speed = _positive("max_ice_speed", self.max_ice_speed)
        max_omega = _positive("max_ice_angular_speed", self.max_ice_angular_speed)
        if max_speed > 0.20:
            raise ValueError("max_ice_speed must not exceed 0.20 lattice cells/step")
        if math.hypot(*initial_velocity) > max_speed:
            raise ValueError("ice_initial_velocity exceeds max_ice_speed")
        if abs(initial_omega) > max_omega:
            raise ValueError(
                "ice_initial_angular_velocity exceeds max_ice_angular_speed"
            )
        restitution = _finite("wall_restitution", self.wall_restitution)
        friction = _finite("wall_friction", self.wall_friction)
        bottom_friction = _finite("bottom_wall_friction", self.bottom_wall_friction)
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("wall_restitution must be in [0, 1]")
        if not 0.0 <= friction <= 1.0:
            raise ValueError("wall_friction must be in [0, 1]")
        if not 0.0 <= bottom_friction <= 1.0:
            raise ValueError("bottom_wall_friction must be in [0, 1]")
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
        # Check the initially rotated rectangle against the container.  The
        # default bottom is exactly y=3, matching the standard scenario.
        c = abs(math.cos(float(self.ice_initial_angle)))
        s = abs(math.sin(float(self.ice_initial_angle)))
        extent_x = c * self.ice_width * 0.5 + s * self.ice_height * 0.5
        extent_y = s * self.ice_width * 0.5 + c * self.ice_height * 0.5
        cx, cy = self.ice_initial_center
        if (
            cx - extent_x < self.boundary_cells
            or cx + extent_x > nx - self.boundary_cells
        ):
            raise ValueError("initial ice rectangle overlaps a side wall")
        if (
            cy - extent_y < self.boundary_cells - 1.0e-9
            or cy + extent_y > ny - self.boundary_cells
        ):
            raise ValueError("initial ice rectangle overlaps the bottom or top wall")

        if self.thermal is not None:
            if not self.ice_fixed:
                raise ValueError(
                    "the first thermal coupling requires ice_fixed=True; "
                    "moving melting ice needs conservative thermal remapping"
                )
            if self.well_balanced_hydrostatics:
                raise ValueError(
                    "thermal phase change is not yet compatible with a frozen "
                    "well-balanced hydrostatic reference"
                )
            if not self.thermal.water_air_interface_adiabatic:
                raise ValueError(
                    "the first LBM thermal coupling requires an adiabatic "
                    "water/air thermal interface"
                )
            if self.water_height >= ny - self.boundary_cells:
                raise ValueError(
                    "thermal phase change requires a water/air free surface for "
                    "continuous phase-volume projection"
                )

        # Central-moment collision requires tau > 0.5.  The upper bound and
        # body-surface Mach cap reject clearly unstable custom configurations.
        g_ref = abs(float(self.gravity[1])) or 9.8
        u_ref = math.sqrt(g_ref * self.reference_length_cells * float(self.dx) * 4.0)
        for name, viscosity in (
            ("viscosity_water", float(self.viscosity_water)),
            ("viscosity_air", float(self.viscosity_air)),
        ):
            nu_lattice = viscosity * 0.1 / (float(self.dx) * u_ref)
            tau = 0.5 + 3.0 * nu_lattice
            if not math.isfinite(tau) or not 0.5 < tau <= 2.0:
                raise ValueError(
                    f"{name} gives an unstable lattice relaxation time ({tau})"
                )
        radius = math.hypot(self.ice_width * 0.5, self.ice_height * 0.5)
        if max_speed + max_omega * radius > 0.25:
            raise ValueError(
                "rigid-body caps permit a boundary speed above 0.25 lattice cells/step"
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
        # Align the analytic rectangle with cell centers.  This equals
        # (300, 48) for the default 90x90 body, and avoids an extra raster
        # column when a custom resolution produces an odd ice width.
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
    if "resolution" in overrides and "reference_length_cells" not in overrides:
        resolution = overrides["resolution"]
        if not hasattr(resolution, "__len__") or len(resolution) != 2:
            raise ValueError("resolution must contain exactly two integers")
        overrides["reference_length_cells"] = int(resolution[1])
    return IceFlowConfig(**overrides)
