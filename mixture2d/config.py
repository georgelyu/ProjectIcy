from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Mode = Literal["fluid", "sand", "coupled"]


@dataclass(slots=True)
class DamBreakConfig:
    mode: Mode = "fluid"
    resolution: tuple[int, int] = (600, 300)
    dx: float = 0.01
    reference_length_cells: int = 300

    # Paper Sec. 3-4 physical parameters. The solver converts these to lattice
    # units internally following the dimensionless procedure described in
    # paper Sec. 5.1.
    rho_water: float = 1000.0
    rho_air: float = 1.25
    viscosity_water: float = 1.0e-4
    viscosity_air: float = 1.0e-2
    sigma: float = 0.072
    gravity: tuple[float, float] = (0.0, -9.8)
    cd: float = 0.1
    interface_width: float = 5.0
    mobility: float = 0.1

    # Coupling/retention parameters: delta_max is the maximum sediment packing
    # fraction from paper Eq. 25, ci/chi appear in the drag coefficient in
    # Sec. 4.3, and ratio_max is r_max in the retention capacity of Eq. 55.
    delta_max: float = 0.64
    ci: float = 0.39
    chi: float = 3.7
    ratio_max: float = 0.293
    water_retention: bool = False

    # Reference initialization performs 500 phase warm-start steps. In coupled
    # dam-break runs, the reference code lets sand settle before starting the
    # LBM fluid step; this is exposed here for reproducibility.
    phase_warmup_steps: int = 500
    coupled_fluid_start_step: int = 600 * 25

    # Dam-break geometry in grid-cell coordinates.
    water_width_fraction: float = 0.25
    water_height_fraction: float = 2.0 / 3.0
    sand_width_fraction: float = 0.15
    sand_height_fraction: float = 0.30
    sand_base_y_cells: float = 3.0
    boundary_cells: int = 3

    # Paper Sec. 4.2 MPM and Drucker-Prager controls.
    particles_per_cell: int = 2
    particle_volume_fraction: float = 1.0
    sand_density: float = 850.0
    sand_youngs_modulus: float = 3.537e5
    sand_poisson_ratio: float = 0.3
    # With q=0 and h0/h1/h2/h3 = 35/9/0.2/10, the hardening law
    # gives an initial friction angle of 25 degrees and alpha=0.267765.
    sand_initial_alpha: float = 0.267765
    sand_cohesion_factor: float = 1.0e-2
    # Piecewise retention-cohesion curve for partially saturated sand.
    # phi[1] = ratio_max * delta_max, so cohesion peaks when the packed sand
    # has reached its bound-water capacity and decreases as free water appears.
    retention_cohesion_c: tuple[float, float, float] = (3.82e-3, 8.82e-3, 1.0e-2)
    retention_cohesion_phi: tuple[float, float, float] = (0.108, 0.18752, 0.3)
    mpm_dt: float = 1.0
    wall_friction: float = 0.75

    # Output controls.
    output_dir: str = "outputs/dam_break_2d"
    show_gui: bool = False
    save_npz: bool = False

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
    def sand_width(self) -> int:
        return max(self.boundary_cells + 2, int(self.nx * self.sand_width_fraction))

    @property
    def sand_height(self) -> int:
        return max(self.boundary_cells + 2, int(self.ny * self.sand_height_fraction))

    @property
    def particle_count(self) -> int:
        cells_x = max(1, self.sand_width)
        cells_y = max(1, self.sand_height)
        ppc = max(1, int(self.particles_per_cell))
        return cells_x * cells_y * ppc * ppc

    def to_dict(self) -> dict:
        return asdict(self)


def create_dambreak_config(**overrides) -> DamBreakConfig:
    cfg = DamBreakConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown DamBreakConfig option: {key}")
        setattr(cfg, key, value)
    if "resolution" in overrides and "reference_length_cells" not in overrides:
        cfg.reference_length_cells = int(cfg.resolution[1])
    return cfg
