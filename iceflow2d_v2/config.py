"""Physical inputs and explicit SI <-> lattice conversions."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    nx: int = 100
    ny: int = 200
    width: float = 0.025
    height: float = 0.050
    wall_cells: int = 3
    water_level: float = 0.030
    ice_size: float = 0.008
    angle_deg: float = 5.0
    drop_height: float = 0.0015
    reference_velocity: float = 4.0
    gravity: float = 9.8
    rho_w: float = 1000.0
    rho_s: float = 917.0
    rho_a: float = 1.25
    nu_w: float = 1.0e-6
    nu_a: float = 1.5e-5
    sigma: float = 0.072
    cp_w: float = 4186.0
    cp_s: float = 2100.0
    cp_a: float = 1005.0
    k_w: float = 0.60
    k_s: float = 2.20
    k_a: float = 0.026
    latent: float = 334000.0
    water_temperature: float = 90.0
    ice_temperature: float = 0.0
    air_temperature: float = 20.0
    temperature_scale: float = 90.0
    beta: float = 2.1e-4
    interface_width: float = 5.0
    mobility: float = 0.1
    smagorinsky: float = 0.1
    shear_tau_floor: float = 0.8
    coupling_steps: int = 1
    pressure_iterations: int = 240
    projection_solver: str = 'cg'
    multigrid_cycles: int = 6
    pressure_polish_iterations: int = 24
    transport_iterations: int = 40
    warmup_steps: int = 500
    ice_air_heat_exchange: bool = False
    thermal_tau: float = 1.0
    max_mach: float = 0.25
    capacity_tolerance: float = 2.0e-5
    mass_tolerance: float = 1.0e-5
    energy_tolerance: float = 1.0e-6
    rigid_iterations: int = 3
    fixed_body: bool = False
    melting: bool = True
    advection: bool = True
    buoyancy: bool = True

    def __post_init__(self):
        if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration values must be finite.")
        if min(self.nx, self.ny) < 16 or self.wall_cells < 1:
            raise ValueError("Grid or wall thickness is too small.")
        if 2*self.wall_cells >= min(self.nx, self.ny) or min(self.width, self.height, self.ice_size) <= 0:
            raise ValueError("The domain must have a positive fluid interior and ice size.")
        if not math.isclose(self.width / self.nx, self.height / self.ny):
            raise ValueError("Square lattice cells are required.")
        if self.coupling_steps < 1 or self.reference_velocity <= 0:
            raise ValueError("Invalid coupling interval or velocity scale.")
        if min(self.rho_w, self.rho_s, self.rho_a, self.latent,
               self.cp_w, self.cp_s, self.cp_a, self.k_w, self.k_s,
               self.k_a, self.temperature_scale) <= 0:
            raise ValueError("Densities, heat capacities and latent heat must be positive.")
        if abs(self.ice_size / self.dx - round(self.ice_size / self.dx)) > 1e-7:
            raise ValueError("Ice size must be aligned with the material grid.")
        if self.coupling_steps not in (1, 2, 4, 8):
            raise ValueError("Conduction blocks support 1, 2, 4 or 8 fluid steps; interface exchange stays synchronous.")
        if self.warmup_steps < 0:
            raise ValueError("Phase warm-up steps must be nonnegative.")
        if self.ice_air_heat_exchange:
            raise ValueError("Ice/air thermal contact is not validated yet; use the adiabatic baseline.")
        if self.thermal_tau != 1.0:
            raise ValueError("Cut-cell thermal flux presently requires thermal_tau=1.")
        if self.ice_temperature > 0:
            raise ValueError("Initially solid ice must be at or below the melting point (0 C).")
        if self.water_temperature < 0:
            raise ValueError("This first interface closure supports melting, not refreezing.")
        if self.rho_s >= self.rho_w:
            raise ValueError("This air closure requires rho_s < rho_w.")
        if min(self.pressure_iterations, self.transport_iterations, self.rigid_iterations) < 1:
            raise ValueError("Iteration counts must be positive.")
        if self.projection_solver not in ('cg', 'multigrid') or min(self.multigrid_cycles, self.pressure_polish_iterations) < 1:
            raise ValueError('Invalid projection solver or iteration count.')
        if not 0 < self.interface_width or self.mobility <= 0 or self.shear_tau_floor <= 0.5:
            raise ValueError("Invalid phase or shear parameters.")
        if min(self.mass_tolerance, self.energy_tolerance, self.capacity_tolerance, self.max_mach) <= 0:
            raise ValueError("Diagnostic tolerances must be positive.")
        if any(not 0 < self.thermal_weight(m) < 1 for m in ('water', 'ice', 'air')):
            raise ValueError("Thermal time step requires 0 < varpi < 1 in each material.")

    @property
    def dx(self):
        return self.width / self.nx

    @property
    def dt(self):
        return 0.1 * self.dx / self.reference_velocity

    @property
    def nb(self):
        return round(self.ice_size / self.dx)

    @property
    def rho_s_lu(self):
        return self.rho_s / self.rho_w

    @property
    def rho_a_lu(self):
        return self.rho_a / self.rho_w

    @property
    def gravity_lu(self):
        return self.gravity * self.dt**2 / self.dx

    @property
    def mass_unit(self):
        return self.rho_w * self.dx**2  # kg per metre out-of-plane thickness

    @property
    def energy_unit(self):
        return self.mass_unit * self.cp_w * self.temperature_scale  # J/m

    @property
    def latent_lu(self):
        return self.latent / (self.cp_w * self.temperature_scale)

    @property
    def ice_capacity_lu(self):
        return self.rho_s_lu * self.cp_s / self.cp_w

    @property
    def air_capacity_lu(self):
        return self.rho_a_lu * self.cp_a / self.cp_w

    def thermal_weight(self, material):
        rho, cp, conductivity = {
            "water": (self.rho_w, self.cp_w, self.k_w),
            "ice": (self.rho_s, self.cp_s, self.k_s),
            "air": (self.rho_a, self.cp_a, self.k_a),
        }[material]
        return 2 * conductivity * self.dt * self.coupling_steps / (rho * cp * self.dx**2 * (self.thermal_tau - 0.5))

    @property
    def initial_origin(self):
        theta = math.radians(self.angle_deg)
        extent = self.nb * 0.5 * (abs(math.cos(theta)) + abs(math.sin(theta)))
        return (self.nx * 0.5, (self.water_level + self.drop_height) / self.dx + extent)

    def metadata(self):
        return {**asdict(self), "dx_m": self.dx, "dt_s": self.dt,
                "mass_unit_kg_m": self.mass_unit, "energy_unit_J_m": self.energy_unit,
                "air_model": "fixed-density numerical air; air mass error reported",
                "air_heat_exchange": "water/air and ice/air adiabatic; independent air enthalpy",
                "precision": "LB/thermal/conserved state f32 with compensation; geometry, pose and capacity Krylov solve f64 after reproduced f32 failures; multigrid f32"}
