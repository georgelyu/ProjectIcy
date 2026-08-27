"""CPU reference solver for fixed two-dimensional ice melting.

The benchmark places a mechanically fixed rectangular ice body at the centre
of a warm-water bath.  Heat transfer and phase change are advanced with a
conservative cell-centred finite-volume enthalpy discretization.  Fluid motion
and the ice/water density change are intentionally omitted so this module can
validate the two-dimensional thermal core before it is coupled to the LBM
solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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


def _positive_integer(name: str, value: int, *, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


@dataclass(slots=True)
class Stefan2DConfig:
    """Physical and numerical controls for the fixed-ice bath benchmark."""

    domain_width_m: float = 0.030
    domain_height_m: float = 0.030
    cells_x: int = 120
    cells_y: int = 120
    ice_width_m: float = 0.012
    ice_height_m: float = 0.012
    end_time_s: float = 180.0
    output_interval_s: float = 30.0

    melting_temperature_c: float = 0.0
    bath_temperature_c: float = 20.0

    # Equal density is intentional for this first two-dimensional benchmark.
    density_kg_m3: float = 1000.0
    specific_heat_water_j_kg_k: float = 4186.0
    specific_heat_ice_j_kg_k: float = 2100.0
    conductivity_water_w_m_k: float = 0.60
    conductivity_ice_w_m_k: float = 2.20
    latent_heat_j_kg: float = 334000.0

    # With half-cell Dirichlet faces, a square-grid corner has six explicit
    # diffusion contributions.  Fo <= 1/6 is therefore the monotone bound.
    fourier_number: float = 0.15
    time_step_s: float | None = None

    def __post_init__(self) -> None:
        self.cells_x = _positive_integer("cells_x", self.cells_x, minimum=4)
        self.cells_y = _positive_integer("cells_y", self.cells_y, minimum=4)
        _positive("domain_width_m", self.domain_width_m)
        _positive("domain_height_m", self.domain_height_m)
        _positive("ice_width_m", self.ice_width_m)
        _positive("ice_height_m", self.ice_height_m)
        _positive("end_time_s", self.end_time_s)
        _positive("output_interval_s", self.output_interval_s)
        _positive("density_kg_m3", self.density_kg_m3)
        _positive("specific_heat_water_j_kg_k", self.specific_heat_water_j_kg_k)
        _positive("specific_heat_ice_j_kg_k", self.specific_heat_ice_j_kg_k)
        _positive("conductivity_water_w_m_k", self.conductivity_water_w_m_k)
        _positive("conductivity_ice_w_m_k", self.conductivity_ice_w_m_k)
        _positive("latent_heat_j_kg", self.latent_heat_j_kg)

        melting = _finite("melting_temperature_c", self.melting_temperature_c)
        bath = _finite("bath_temperature_c", self.bath_temperature_c)
        if bath <= melting:
            raise ValueError("bath_temperature_c must exceed melting_temperature_c")

        if self.ice_width_m >= self.domain_width_m:
            raise ValueError("ice_width_m must be smaller than domain_width_m")
        if self.ice_height_m >= self.domain_height_m:
            raise ValueError("ice_height_m must be smaller than domain_height_m")
        self._validate_aligned_geometry("x", self.ice_width_m / self.dx_m, self.cells_x)
        self._validate_aligned_geometry(
            "y", self.ice_height_m / self.dy_m, self.cells_y
        )

        fourier = _positive("fourier_number", self.fourier_number)
        if fourier > 1.0 / 6.0:
            raise ValueError("fourier_number must not exceed 1/6")
        if self.default_time_step_s > self.maximum_time_step_s * (1.0 + 1.0e-12):
            raise ValueError("fourier_number exceeds the rectangular-grid limit")
        if self.time_step_s is not None:
            requested = _positive("time_step_s", self.time_step_s)
            if requested > self.maximum_time_step_s * (1.0 + 1.0e-12):
                raise ValueError(
                    "time_step_s exceeds the explicit diffusion stability limit "
                    f"({self.maximum_time_step_s:.9g} s)"
                )

    @staticmethod
    def _validate_aligned_geometry(axis: str, ice_cells: float, cells: int) -> None:
        rounded_ice = round(ice_cells)
        if not math.isclose(ice_cells, rounded_ice, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f"ice geometry on {axis} must align with cell faces")
        if rounded_ice < 2:
            raise ValueError(f"ice geometry on {axis} must span at least two cells")
        margin_cells = 0.5 * (cells - rounded_ice)
        if not math.isclose(
            margin_cells, round(margin_cells), rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"centred ice geometry on {axis} must align with cell faces"
            )
        if margin_cells < 1.0:
            raise ValueError(
                f"ice geometry on {axis} needs at least one bath cell per side"
            )

    @property
    def dx_m(self) -> float:
        return float(self.domain_width_m) / self.cells_x

    @property
    def dy_m(self) -> float:
        return float(self.domain_height_m) / self.cells_y

    @property
    def maximum_diffusivity_m2_s(self) -> float:
        # This bound also covers interpolated mushy-cell properties.
        return max(self.conductivity_water_w_m_k, self.conductivity_ice_w_m_k) / (
            self.density_kg_m3
            * min(
                self.specific_heat_water_j_kg_k,
                self.specific_heat_ice_j_kg_k,
            )
        )

    @property
    def maximum_time_step_s(self) -> float:
        inverse_spacing_sum = 1.0 / self.dx_m**2 + 1.0 / self.dy_m**2
        return 1.0 / (3.0 * self.maximum_diffusivity_m2_s * inverse_spacing_sum)

    @property
    def default_time_step_s(self) -> float:
        spacing = min(self.dx_m, self.dy_m)
        return self.fourier_number * spacing * spacing / self.maximum_diffusivity_m2_s

    @property
    def actual_time_step_s(self) -> float:
        if self.time_step_s is not None:
            return float(self.time_step_s)
        return self.default_time_step_s


def recover_temperature_and_liquid_fraction(
    enthalpy_j_m3: np.ndarray, config: Stefan2DConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Invert volumetric enthalpy into temperature and liquid fraction."""

    enthalpy = np.asarray(enthalpy_j_m3, dtype=np.float64)
    temperature = np.empty_like(enthalpy)
    liquid_fraction = np.empty_like(enthalpy)
    melting = float(config.melting_temperature_c)
    latent_volume = config.density_kg_m3 * config.latent_heat_j_kg

    solid = enthalpy < 0.0
    phase_change = (enthalpy >= 0.0) & (enthalpy <= latent_volume)
    liquid = enthalpy > latent_volume

    temperature[solid] = melting + enthalpy[solid] / (
        config.density_kg_m3 * config.specific_heat_ice_j_kg_k
    )
    liquid_fraction[solid] = 0.0

    temperature[phase_change] = melting
    liquid_fraction[phase_change] = enthalpy[phase_change] / latent_volume

    temperature[liquid] = melting + (enthalpy[liquid] - latent_volume) / (
        config.density_kg_m3 * config.specific_heat_water_j_kg_k
    )
    liquid_fraction[liquid] = 1.0
    return temperature, liquid_fraction


def d4_symmetry_error(field: np.ndarray) -> float:
    """Return the maximum error over the eight symmetries of a square."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("D4 symmetry requires a square two-dimensional field")
    transforms = (
        np.rot90(values, 1),
        np.rot90(values, 2),
        np.rot90(values, 3),
        np.flip(values, axis=0),
        np.flip(values, axis=1),
        values.T,
        np.flip(values.T, axis=0),
    )
    return max(
        float(np.max(np.abs(values - transformed))) for transformed in transforms
    )


@dataclass(slots=True)
class Stefan2DSnapshot:
    time_s: float
    temperature_c: np.ndarray
    liquid_fraction: np.ndarray
    enthalpy_j_m3: np.ndarray
    ice_area_m2: float
    equivalent_square_side_m: float
    equivalent_square_half_width_m: float
    equivalent_uniform_melt_depth_m: float
    horizontal_midline_solid_width_m: float
    vertical_midline_solid_height_m: float
    melted_fraction: float
    total_enthalpy_j_m: float
    boundary_heat_input_j_m: float
    energy_residual_j_m: float
    liquid_fraction_symmetry_error: float
    temperature_symmetry_error_c: float
    enthalpy_symmetry_error_j_m3: float
    normalized_symmetry_error: float


class Stefan2DSolver:
    """Conservative explicit finite-volume enthalpy solver on a 2D grid."""

    def __init__(self, config: Stefan2DConfig):
        if not isinstance(config, Stefan2DConfig):
            raise TypeError("config must be a Stefan2DConfig")
        self.config = config
        self.x_m = (np.arange(config.cells_x, dtype=np.float64) + 0.5) * config.dx_m
        self.y_m = (np.arange(config.cells_y, dtype=np.float64) + 0.5) * config.dy_m
        xx, yy = np.meshgrid(self.x_m, self.y_m, indexing="xy")
        ice_mask = (
            np.abs(xx - 0.5 * config.domain_width_m) < 0.5 * config.ice_width_m
        ) & (np.abs(yy - 0.5 * config.domain_height_m) < 0.5 * config.ice_height_m)

        liquid_enthalpy = config.density_kg_m3 * (
            config.latent_heat_j_kg
            + config.specific_heat_water_j_kg_k
            * (config.bath_temperature_c - config.melting_temperature_c)
        )
        self.enthalpy_j_m3 = np.full(
            (config.cells_y, config.cells_x), liquid_enthalpy, dtype=np.float64
        )
        self.enthalpy_j_m3[ice_mask] = 0.0
        self.temperature_c, self.liquid_fraction = (
            recover_temperature_and_liquid_fraction(self.enthalpy_j_m3, config)
        )
        self._flux_x_w_m2 = np.zeros(
            (config.cells_y, config.cells_x + 1), dtype=np.float64
        )
        self._flux_y_w_m2 = np.zeros(
            (config.cells_y + 1, config.cells_x), dtype=np.float64
        )
        self.time_s = 0.0
        self.steps = 0
        self.boundary_heat_input_j_m = 0.0
        self.initial_ice_area_m2 = self.ice_area_m2
        expected_area = config.ice_width_m * config.ice_height_m
        if not math.isclose(
            self.initial_ice_area_m2,
            expected_area,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "rasterized initial ice area disagrees with configuration"
            )
        self._initial_total_enthalpy_j_m = self.total_enthalpy_j_m

    @property
    def total_enthalpy_j_m(self) -> float:
        cell_area = self.config.dx_m * self.config.dy_m
        return float(np.sum(self.enthalpy_j_m3, dtype=np.float64) * cell_area)

    @property
    def ice_area_m2(self) -> float:
        """Return the phase-fraction integral interpreted as solid area."""

        cell_area = self.config.dx_m * self.config.dy_m
        return float(np.sum(1.0 - self.liquid_fraction, dtype=np.float64) * cell_area)

    @property
    def equivalent_square_side_m(self) -> float:
        """Return the side length of a square with the current solid area."""

        return math.sqrt(max(0.0, self.ice_area_m2))

    @property
    def equivalent_square_half_width_m(self) -> float:
        """Return the half-width of a square with the current solid area."""

        return 0.5 * self.equivalent_square_side_m

    @property
    def equivalent_uniform_melt_depth_m(self) -> float:
        """Return the uniform inward erosion depth matching the solid area.

        For an initial rectangle of width ``w`` and height ``h``, this is the
        smaller root of ``(w - 2 d) (h - 2 d) = A_solid``.  It reduces to half
        the loss of equivalent square side for the default square ice body.
        """

        width = self.config.ice_width_m
        height = self.config.ice_height_m
        discriminant = (width - height) ** 2 + 4.0 * max(0.0, self.ice_area_m2)
        return 0.25 * (width + height - math.sqrt(discriminant))

    @property
    def melted_fraction(self) -> float:
        return 1.0 - self.ice_area_m2 / self.initial_ice_area_m2

    @property
    def horizontal_midline_solid_width_m(self) -> float:
        """Return the solid-fraction integral along the horizontal centreline."""

        midpoint = self.config.cells_y // 2
        if self.config.cells_y % 2:
            profile = 1.0 - self.liquid_fraction[midpoint, :]
        else:
            profile = 1.0 - np.mean(
                self.liquid_fraction[midpoint - 1 : midpoint + 1, :], axis=0
            )
        return float(np.sum(profile, dtype=np.float64) * self.config.dx_m)

    @property
    def vertical_midline_solid_height_m(self) -> float:
        """Return the solid-fraction integral along the vertical centreline."""

        midpoint = self.config.cells_x // 2
        if self.config.cells_x % 2:
            profile = 1.0 - self.liquid_fraction[:, midpoint]
        else:
            profile = 1.0 - np.mean(
                self.liquid_fraction[:, midpoint - 1 : midpoint + 1], axis=1
            )
        return float(np.sum(profile, dtype=np.float64) * self.config.dy_m)

    @property
    def energy_residual_j_m(self) -> float:
        stored_change = self.total_enthalpy_j_m - self._initial_total_enthalpy_j_m
        return stored_change - self.boundary_heat_input_j_m

    def _cell_conductivity(self) -> np.ndarray:
        fraction = self.liquid_fraction
        return (
            (1.0 - fraction) * self.config.conductivity_ice_w_m_k
            + fraction * self.config.conductivity_water_w_m_k
        )

    def step(self, time_step_s: float | None = None) -> None:
        dt = (
            self.config.actual_time_step_s
            if time_step_s is None
            else _positive("time_step_s", time_step_s)
        )
        if dt > self.config.maximum_time_step_s * (1.0 + 1.0e-12):
            raise ValueError("time step exceeds the explicit diffusion stability limit")

        config = self.config
        conductivity = self._cell_conductivity()
        flux_x = self._flux_x_w_m2
        flux_y = self._flux_y_w_m2

        flux_x[:, 0] = (
            2.0
            * config.conductivity_water_w_m_k
            * (config.bath_temperature_c - self.temperature_c[:, 0])
            / config.dx_m
        )
        flux_x[:, -1] = (
            2.0
            * config.conductivity_water_w_m_k
            * (self.temperature_c[:, -1] - config.bath_temperature_c)
            / config.dx_m
        )
        left = conductivity[:, :-1]
        right = conductivity[:, 1:]
        face_conductivity_x = 2.0 * left * right / (left + right)
        flux_x[:, 1:-1] = (
            -face_conductivity_x
            * (self.temperature_c[:, 1:] - self.temperature_c[:, :-1])
            / config.dx_m
        )

        flux_y[0, :] = (
            2.0
            * config.conductivity_water_w_m_k
            * (config.bath_temperature_c - self.temperature_c[0, :])
            / config.dy_m
        )
        flux_y[-1, :] = (
            2.0
            * config.conductivity_water_w_m_k
            * (self.temperature_c[-1, :] - config.bath_temperature_c)
            / config.dy_m
        )
        bottom = conductivity[:-1, :]
        top = conductivity[1:, :]
        face_conductivity_y = 2.0 * bottom * top / (bottom + top)
        flux_y[1:-1, :] = (
            -face_conductivity_y
            * (self.temperature_c[1:, :] - self.temperature_c[:-1, :])
            / config.dy_m
        )

        divergence = (flux_x[:, :-1] - flux_x[:, 1:]) / config.dx_m + (
            flux_y[:-1, :] - flux_y[1:, :]
        ) / config.dy_m
        self.enthalpy_j_m3 += dt * divergence
        boundary_power_w_m = config.dy_m * float(
            np.sum(flux_x[:, 0] - flux_x[:, -1], dtype=np.float64)
        ) + config.dx_m * float(np.sum(flux_y[0, :] - flux_y[-1, :], dtype=np.float64))
        self.boundary_heat_input_j_m += dt * boundary_power_w_m
        self.time_s += dt
        self.steps += 1
        self.temperature_c, self.liquid_fraction = (
            recover_temperature_and_liquid_fraction(self.enthalpy_j_m3, self.config)
        )

        if not (
            np.isfinite(self.enthalpy_j_m3).all()
            and np.isfinite(self.temperature_c).all()
            and np.isfinite(self.liquid_fraction).all()
        ):
            raise RuntimeError("non-finite thermal state")
        if (
            float(np.min(self.liquid_fraction)) < -1.0e-12
            or float(np.max(self.liquid_fraction)) > 1.0 + 1.0e-12
        ):
            raise RuntimeError("liquid fraction left [0, 1]")

    def _field_symmetry_error(self, values: np.ndarray) -> float:
        error = max(
            float(np.max(np.abs(values - np.flip(values, axis=0)))),
            float(np.max(np.abs(values - np.flip(values, axis=1)))),
        )
        square_case = (
            values.shape[0] == values.shape[1]
            and math.isclose(
                self.config.domain_width_m,
                self.config.domain_height_m,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                self.config.ice_width_m,
                self.config.ice_height_m,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
        if square_case:
            error = max(error, d4_symmetry_error(values))
        return error

    def _symmetry_errors(self) -> tuple[float, float, float, float]:
        liquid_error = self._field_symmetry_error(self.liquid_fraction)
        temperature_error = self._field_symmetry_error(self.temperature_c)
        enthalpy_error = self._field_symmetry_error(self.enthalpy_j_m3)
        temperature_scale = max(
            1.0,
            abs(self.config.bath_temperature_c - self.config.melting_temperature_c),
        )
        enthalpy_scale = self.config.density_kg_m3 * (
            self.config.latent_heat_j_kg
            + self.config.specific_heat_water_j_kg_k
            * (self.config.bath_temperature_c - self.config.melting_temperature_c)
        )
        normalized_error = max(
            liquid_error,
            temperature_error / temperature_scale,
            enthalpy_error / enthalpy_scale,
        )
        return liquid_error, temperature_error, enthalpy_error, normalized_error

    def snapshot(self) -> Stefan2DSnapshot:
        symmetry_errors = self._symmetry_errors()
        return Stefan2DSnapshot(
            time_s=float(self.time_s),
            temperature_c=self.temperature_c.copy(),
            liquid_fraction=self.liquid_fraction.copy(),
            enthalpy_j_m3=self.enthalpy_j_m3.copy(),
            ice_area_m2=self.ice_area_m2,
            equivalent_square_side_m=self.equivalent_square_side_m,
            equivalent_square_half_width_m=self.equivalent_square_half_width_m,
            equivalent_uniform_melt_depth_m=self.equivalent_uniform_melt_depth_m,
            horizontal_midline_solid_width_m=(self.horizontal_midline_solid_width_m),
            vertical_midline_solid_height_m=(self.vertical_midline_solid_height_m),
            melted_fraction=self.melted_fraction,
            total_enthalpy_j_m=self.total_enthalpy_j_m,
            boundary_heat_input_j_m=float(self.boundary_heat_input_j_m),
            energy_residual_j_m=self.energy_residual_j_m,
            liquid_fraction_symmetry_error=symmetry_errors[0],
            temperature_symmetry_error_c=symmetry_errors[1],
            enthalpy_symmetry_error_j_m3=symmetry_errors[2],
            normalized_symmetry_error=symmetry_errors[3],
        )

    def run(self) -> list[Stefan2DSnapshot]:
        """Advance to all configured output times, including zero and the end."""

        output_times = [0.0]
        next_time = float(self.config.output_interval_s)
        while next_time < self.config.end_time_s - 1.0e-12:
            output_times.append(next_time)
            next_time += self.config.output_interval_s
        output_times.append(float(self.config.end_time_s))

        snapshots = [self.snapshot()]
        for target_time in output_times[1:]:
            while self.time_s < target_time - 1.0e-14:
                remaining = target_time - self.time_s
                self.step(min(self.config.actual_time_step_s, remaining))
            self.time_s = target_time
            snapshots.append(self.snapshot())
        return snapshots
