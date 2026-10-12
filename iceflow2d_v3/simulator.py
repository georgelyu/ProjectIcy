"""2D air--water LBM coupled to independently moving, melting ice bodies.

The pressure--momentum distribution uses selectable raw- or central-moment MRT;
the conservative phase distribution uses central-moment collision. An
eroded material coverage excludes ice nodes from the fluid domain, and cut
lattice links enforce each body's impermeable boundary. Link momentum exchange
and inelastic body contact drive the freely moving ice bodies.

A material-frame thermal solver evolves the remaining ice and water heat.
Melted material disappears; only the surviving ice contributes to geometry
and rigid-body mass properties.
"""

import math
from typing import Any

import numpy as np
import taichi as ti

from .config import (
    IceBodyConfig,
    IceFlowConfig,
    LatticeScales,
    ThermalConfig,
    _positive,
    _non_negative,
    _BOUNDARY_CODE,
    _DIRICHLET,
)
from .lattice import (
    D2Q9_DIRECTION_COUNT,
    lattice_direction,
    cross_product_2d,
    phase_equilibrium,
    inside_grid,
    opposite_direction_index,
    momentum_equilibrium,
    reconstruct_central_moment_population,
    mixture_density,
    lattice_weight,
    water_dynamic_viscosity_pa_s,
)
from .thermal_d2q5 import D2Q5Thermal2D
from .body_state import BodyField
from .contact import BodyContacts
from .stability_options import StabilityOptions, pressure_preserving_stress_population
from .conservative_phase import ConservativePhaseLBM

_MAX_D2Q9_LATTICE_VELOCITY_L1 = 16.0


def ensure_taichi_cuda():
    """Initialize the one supported backend without resetting an active runtime."""

    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        try:
            ti.init(
                arch=ti.cuda,
                default_fp=ti.f32,
                default_ip=ti.i32,
                enable_fallback=False,
            )
        except Exception as exc:  # pragma: no cover - depends on host GPU
            raise RuntimeError(
                "IceFlow2D requires Taichi CUDA; no CPU fallback is provided"
            ) from exc
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("IceFlow2D requires an existing or new Taichi CUDA runtime")


class _DerivedField:
    """Read-only, allocation-free field view; device reads inline arithmetic.

    Host reads materialize NumPy values on demand. This is deliberately not a
    Taichi field: no persistent device allocation, upload, or refresh kernel.
    """

    def __init__(
        self, device_reader, host_reader, *, host_writer=None, single_body=False
    ):
        self._device_reader = device_reader
        self._host_reader = host_reader
        self._host_writer = host_writer
        self._single_body = single_body

    def __getitem__(self, index):
        if ti.lang.impl.inside_kernel():
            indices = (
                index if isinstance(index, tuple) else (0 if index is None else index,)
            )
            return self._device_reader(*indices)
        values = self._host_reader()
        if self._single_body:
            if index not in (None, 0):
                raise IndexError("body_id outside the one-body state")
            return values
        return values if index is None else values[index]

    def __setitem__(self, index, value):
        if self._host_writer is None or ti.lang.impl.inside_kernel():
            raise TypeError("derived values are read-only; update the underlying state")
        self._host_writer(index, value)

    def to_numpy(self):
        return np.asarray(self._host_reader()).copy()


@ti.data_oriented
class MovingBodyThermal2D:
    """Body-frame ice enthalpy coupled to world-frame water sensible heat.

    The material grid is immutable under rigid translation and rotation.
    ``body_solid_mass`` and ``body_sensible_energy`` define its state;
    world rasterization is a read-only pose transform owned by the parent
    solver.  Water stores cell volume and sensible energy as extensive
    quantities.  Every ice--water face heat transfer is removed from the
    water cell and atomically added to exactly one material cell before
    phase change is applied.

    Bulk conduction uses finite volumes by default, or optional D2Q5
    populations on both grids when ``config.scheme == "enthalpy_d2q5"``.

    Melted material is removed from the body without a liquid source.
    Water/air heat transfer is adiabatic, as required by
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
        bodies: tuple[IceBodyConfig, ...] | None = None,
    ):
        for name, value, minimum in (
            ("nx", nx, 2),
            ("ny", ny, 2),
            ("body_nx", body_nx, 1),
            ("body_ny", body_ny, 1),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
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
        self.body_count = len(bodies) if bodies else 1
        # Material rectangles occupy disjoint regions of a 2D atlas. One empty
        # column separates them, so both FV and D2Q5 conduction retain their
        # existing face stencils without exchanging heat between bodies.
        sizes = np.asarray(
            [b.size_cells for b in bodies] if bodies else [(body_nx, body_ny)],
            dtype=np.int32,
        )
        offsets = np.zeros(self.body_count, dtype=np.int32)
        if self.body_count > 1:
            offsets[1:] = np.cumsum(sizes[:-1, 0] + 1)
        self.body_nx = int(offsets[-1] + sizes[-1, 0])
        self.body_ny = int(sizes[:, 1].max())
        self.material_offset = ti.field(ti.i32, shape=self.body_count)
        self.material_size = ti.Vector.field(2, ti.i32, shape=self.body_count)
        self.material_body_id = ti.field(ti.i32, shape=(self.body_nx, self.body_ny))
        self.initial_body_temperature = ti.field(ti.f64, shape=self.body_count)
        owners = np.full((self.body_nx, self.body_ny), -1, dtype=np.int32)
        for b, (width, height) in enumerate(sizes):
            owners[offsets[b] : offsets[b] + width, :height] = b
        self.material_offset.from_numpy(offsets)
        self.material_size.from_numpy(sizes)
        self.material_body_id.from_numpy(owners)
        self.initial_body_temperature.from_numpy(
            np.asarray(
                [
                    config.initial_ice_temperature_c
                    if b.temperature_c is None
                    else b.temperature_c
                    for b in bodies
                ]
                if bodies
                else [config.initial_ice_temperature_c],
                dtype=np.float64,
            )
        )
        self.config = config
        self.scales = scales
        self.steps = 0
        self.time_s = 0.0
        self._water_density_kg_m3 = _positive(
            "density_water_kg_m3", density_water_kg_m3
        )
        self._ice_density_kg_m3 = _positive("density_ice_kg_m3", density_ice_kg_m3)
        self._water_phase_cutoff = _non_negative(
            "water_phase_cutoff", water_phase_cutoff
        )
        if self._water_phase_cutoff >= 0.5:
            raise ValueError("water_phase_cutoff must be below 0.5")
        props = config.properties
        self._melting_temperature_c = float(props.melting_temperature_c)
        self._water_specific_heat = float(props.specific_heat_water_j_kg_k)
        self._ice_specific_heat = float(props.specific_heat_ice_j_kg_k)
        self._water_conductivity = float(props.conductivity_water_w_m_k)
        self._ice_conductivity = float(props.conductivity_ice_w_m_k)
        self._latent_heat_j_kg = float(props.latent_heat_j_kg)
        self._cell_size_m = float(scales.dx_m)
        self._cell_area = self._cell_size_m * self._cell_size_m
        self._velocity_scale = float(scales.velocity_scale_m_s)
        self._initial_body_cell_mass = self._ice_density_kg_m3 * self._cell_area
        self._solid_fraction_threshold = float(config.solid_liquid_threshold)
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
        self.body_solid_mass = ti.field(ti.f64, shape=body_shape)
        self.body_sensible_energy = ti.field(ti.f64, shape=body_shape)
        self._body_heat_delta = ti.field(ti.f64, shape=body_shape)
        self.melted_mass = ti.field(ti.f64, shape=body_shape)
        self.melted_sensible_energy = ti.field(ti.f64, shape=body_shape)
        self.conservative_phase = None
        self._conservative_phase_enabled = False
        self.last_thermal_substep_s = 0.0

        self.water_volume_m2 = ti.field(ti.f64, shape=world_shape)
        self.water_sensible_energy = ti.field(ti.f64, shape=world_shape)
        self._water_energy_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
        self._water_energy_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
        self._water_volume_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
        self._water_volume_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
        # Freeze local temperature reconstruction before committing the new aperture.
        self._phase_target_energy = ti.field(ti.f64, shape=world_shape)

        self.world_body_solid_fraction = ti.field(ti.f32, shape=world_shape)
        self.body_world_fraction = ti.field(
            ti.f32, shape=(self.body_count, *world_shape)
        )
        # One shared coverage raster; interpolation coordinates and signed
        # distance are computed from the authoritative pose when needed.
        self.body_solid_fraction = _DerivedField(
            self._body_fraction_at, self._body_fraction_numpy
        )
        self.body_temperature = _DerivedField(
            self._body_temperature_at, self._body_temperature_numpy
        )
        self.water_temperature = _DerivedField(
            self._water_temperature_at, self._water_temperature_numpy
        )

        alpha_water = self._water_conductivity / (
            self._water_density_kg_m3 * self._water_specific_heat
        )
        alpha_ice = self._ice_conductivity / (
            self._ice_density_kg_m3 * self._ice_specific_heat
        )
        self.maximum_diffusivity_m2_s = max(alpha_water, alpha_ice)
        self.maximum_diffusion_time_step_s = (
            config.max_fourier_number
            * self._cell_size_m
            * self._cell_size_m
            / self.maximum_diffusivity_m2_s
        )
        self._d2q5 = D2Q5Thermal2D(self) if config.scheme == "enthalpy_d2q5" else None

    def maximum_advection_time_step_s(
        self, maximum_outflow_rate_lattice: float
    ) -> float:
        speed = _non_negative(
            "maximum_outflow_rate_lattice", maximum_outflow_rate_lattice
        )
        if not self._advection_enabled or speed == 0.0:
            return math.inf
        return self.config.max_courant_number * self.scales.dt_s / speed

    def _validate_substep_count(
        self,
        substeps: int,
        *,
        process: str,
        time_step_s: float,
        maximum_outflow_rate_lattice: float | None = None,
    ) -> int:
        if substeps > self.config.max_substeps_per_update:
            detail = ""
            if maximum_outflow_rate_lattice is not None:
                detail = (
                    f", maximum_outflow_rate_lattice={maximum_outflow_rate_lattice!r}"
                )
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
        maximum_outflow_rate_lattice: float | None = None,
    ) -> int:
        """Return the FAST upwind substeps required by the Courant bound."""

        dt = _positive("time_step_s", time_step_s)
        load = 0.0
        if maximum_outflow_rate_lattice is not None:
            advection_limit = self.maximum_advection_time_step_s(
                maximum_outflow_rate_lattice
            )
            if math.isfinite(advection_limit):
                load = dt / advection_limit
        substeps = max(1, int(math.ceil(load - 1.0e-14)))
        return self._validate_substep_count(
            substeps,
            process="advection",
            time_step_s=dt,
            maximum_outflow_rate_lattice=maximum_outflow_rate_lattice,
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
        wall_mask: Any,
        solid_mask: Any,
    ) -> None:
        """Initialize the material body and extensive world-water state."""

        self._initialize_body_state(self._initial_body_cell_mass)
        self._initialize_water_state(water_phase, wall_mask, solid_mask)
        if self._d2q5 is not None:
            self._d2q5.initialize()

        self.steps = 0
        self.time_s = 0.0

    def advance_fast(
        self,
        time_step_s: float,
        velocity: Any,
        water_phase: Any,
        wall_mask: Any,
        solid_mask: Any,
        *,
        maximum_outflow_rate_lattice: float | None = None,
    ) -> int:
        """Advance one pose remap and water advection at the LBM rate."""

        dt = _positive("time_step_s", time_step_s)
        substeps = self.required_advection_substeps(
            dt, maximum_outflow_rate_lattice=maximum_outflow_rate_lattice
        )
        self.synchronize_water_aperture(water_phase, wall_mask, solid_mask)
        sub_dt = dt / substeps
        self.last_thermal_substep_s = sub_dt
        for _ in range(substeps):
            self._compute_water_advection_flux_x(
                velocity, water_phase, wall_mask, solid_mask
            )
            self._compute_water_advection_flux_y(
                velocity, water_phase, wall_mask, solid_mask
            )
            self._update_water_advection(sub_dt, wall_mask)
            self.synchronize_water_aperture(water_phase, wall_mask, solid_mask)
        return substeps

    def advance_slow(
        self,
        time_step_s: float,
        water_phase: Any,
        wall_mask: Any,
        solid_mask: Any,
        body_reference_origin: Any,
        body_angle: Any,
        *,
        on_body_eroded: Any | None = None,
    ) -> int:
        """Advance conduction and remove melted material from the ice body."""

        self.body_reference_origin = body_reference_origin
        self.body_angle = body_angle
        dt = _positive("time_step_s", time_step_s)
        substeps = self.required_diffusion_substeps(dt)
        sub_dt = dt / substeps
        self.last_thermal_substep_s = sub_dt
        for _ in range(substeps):
            if self._d2q5 is None:
                self._accumulate_body_conduction(sub_dt)
                self._compute_water_conduction_flux_x(
                    water_phase, wall_mask, solid_mask
                )
                self._compute_water_conduction_flux_y(
                    water_phase, wall_mask, solid_mask
                )
                self._update_water_conduction(sub_dt, wall_mask)
            else:
                self._d2q5.collide(sub_dt)
                self._d2q5.stream(sub_dt, water_phase, wall_mask, solid_mask)
            self._exchange_interface_heat(sub_dt, water_phase, wall_mask, solid_mask)
            self._apply_body_heat_and_phase_change()
            if self._d2q5 is not None:
                self._d2q5.synchronize()
            if on_body_eroded is not None:
                # The next diffusion substep must see the new ice boundary,
                # fluid cells and water temperature together.
                on_body_eroded()

        self.steps += substeps
        self.time_s += dt
        return substeps

    def synchronize_water_aperture(
        self,
        water_phase: Any,
        wall_mask: Any,
        solid_mask: Any,
    ) -> None:
        """Match water volume to the current phase and reconstruct heat locally.

        Existing water keeps its temperature. Newly wet cells use the nearest
        wet neighborhood; a disconnected cell starts at the melting point.
        No global volume or energy correction is applied.
        """
        if self.conservative_phase is not None:
            self._synchronize_conservative_volume()
            if self._d2q5 is not None:
                self._d2q5.synchronize()
            return
        self._build_phase_aperture_targets(water_phase, wall_mask, solid_mask)
        self._apply_phase_aperture_transfer(water_phase, wall_mask, solid_mask)
        if self._d2q5 is not None:
            self._d2q5.synchronize()

    @ti.kernel
    def _synchronize_conservative_volume(self):
        for i, j in self.water_volume_m2:
            self.water_volume_m2[i, j] = self.conservative_phase.cell_area[None] * self.conservative_phase.volume(i, j)

    @ti.kernel
    def _initialize_body_state(self, cell_mass: ti.f64):
        for i, j in self.body_solid_mass:
            b = self.material_body_id[i, j]
            self.body_solid_mass[i, j] = 0.0
            self.body_sensible_energy[i, j] = 0.0
            if b >= 0:
                self.body_solid_mass[i, j] = cell_mass
                self.body_sensible_energy[i, j] = (
                    cell_mass
                    * self._ice_specific_heat
                    * (self.initial_body_temperature[b] - self._melting_temperature_c)
                )

    @ti.kernel
    def _initialize_water_state(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        rho_cp = ti.cast(
            ti.static(self._water_density_kg_m3 * self._water_specific_heat), ti.f64
        )
        melting = ti.cast(ti.static(self._melting_temperature_c), ti.f64)
        initial_temperature = ti.cast(
            ti.static(float(self.config.initial_water_temperature_c)), ti.f64
        )
        for i, j in self.water_volume_m2:
            phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
            volume = ti.cast(0.0, ti.f64)
            if (
                wall_mask[i, j] == 0
                and solid_mask[i, j] == 0
                and phase > ti.static(self._water_phase_cutoff)
            ):
                volume = area * ti.cast(phase, ti.f64)
            if ti.static(self._conservative_phase_enabled):
                volume = self.conservative_phase.cell_area[None] * self.conservative_phase.volume(i, j)
            self.water_volume_m2[i, j] = volume
            self.water_sensible_energy[i, j] = (
                rho_cp * volume * (initial_temperature - melting)
            )

    @ti.kernel
    def _build_phase_aperture_targets(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        for i, j in self.water_volume_m2:
            target = self._water_aperture_target(
                i, j, water_phase, wall_mask, solid_mask
            )
            specific = ti.cast(0.0, ti.f64)
            if target > 0.0:
                volume = self.water_volume_m2[i, j]
                if volume > 1.0e-30:
                    specific = self.water_sensible_energy[i, j] / volume
                else:
                    # Read the old heat state, including water just covered by ice.
                    # Whole-body removal can expose more than three cell layers.
                    weight = ti.cast(0.0, ti.f64)
                    radius = 1
                    while weight <= 1.0e-30 and radius < ti.static(
                        max(self.nx, self.ny)
                    ):
                        for di, dj in ti.ndrange(
                            (-radius, radius + 1), (-radius, radius + 1)
                        ):
                            ni, nj = i + di, j + dj
                            if (
                                ti.max(ti.abs(di), ti.abs(dj)) == radius
                                and 0 <= ni < self.nx
                                and 0 <= nj < self.ny
                                and wall_mask[ni, nj] == 0
                            ):
                                neighbor_volume = self.water_volume_m2[ni, nj]
                                if neighbor_volume > 1.0e-30:
                                    distance2 = ti.cast(di * di + dj * dj, ti.f64)
                                    specific += (
                                        self.water_sensible_energy[ni, nj] / distance2
                                    )
                                    weight += neighbor_volume / distance2
                        radius += 1
                    if weight > 1.0e-30:
                        specific /= weight
            self._phase_target_energy[i, j] = target * specific

    @ti.kernel
    def _apply_phase_aperture_transfer(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        for i, j in self.water_volume_m2:
            self.water_volume_m2[i, j] = self._water_aperture_target(
                i, j, water_phase, wall_mask, solid_mask
            )
            self.water_sensible_energy[i, j] = self._phase_target_energy[i, j]

    @ti.kernel
    def _accumulate_body_conduction(self, time_step_s: ti.f64):
        # Mass and sensible energy are immutable in this launch. Each face can
        # therefore be evaluated at both endpoints without a flux buffer or race.
        for i, j in self.body_solid_mass:
            outward_heat_rate = ti.cast(0.0, ti.f64)
            fraction = self._body_fraction_at(i, j)
            temperature = self._body_temperature_at(i, j)
            for offset_i, offset_j in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                neighbor_i, neighbor_j = i + offset_i, j + offset_j
                if 0 <= neighbor_i < self.body_nx and 0 <= neighbor_j < self.body_ny:
                    face_fraction = ti.cast(
                        ti.min(
                            fraction, self._body_fraction_at(neighbor_i, neighbor_j)
                        ),
                        ti.f64,
                    )
                    outward_heat_rate += (
                        self._ice_conductivity
                        * face_fraction
                        * (
                            temperature
                            - self._body_temperature_at(neighbor_i, neighbor_j)
                        )
                    )
            self._body_heat_delta[i, j] = -time_step_s * outward_heat_rate

    @ti.kernel
    def _compute_water_conduction_flux_x(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        k_water = ti.cast(ti.static(self._water_conductivity), ti.f64)
        for face_i, j in self._water_energy_flux_x:
            flux = ti.cast(0.0, ti.f64)
            if 0 < face_i < ti.static(self.nx):
                left_i = face_i - 1
                right_i = face_i
                left_active = (
                    wall_mask[left_i, j] == 0
                    and solid_mask[left_i, j] == 0
                    and water_phase[left_i, j] > ti.static(self._water_phase_cutoff)
                )
                right_active = (
                    wall_mask[right_i, j] == 0
                    and solid_mask[right_i, j] == 0
                    and water_phase[right_i, j] > ti.static(self._water_phase_cutoff)
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
                    wall_mask[left_i, j] != 0
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
                elif (
                    left_active
                    and self.water_volume_m2[left_i, j] > 1.0e-30
                    and wall_mask[right_i, j] != 0
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
            self._water_energy_flux_x[face_i, j] = flux

    @ti.kernel
    def _compute_water_advection_flux_x(
        self,
        velocity: ti.template(),
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        dx = ti.cast(ti.static(self._cell_size_m), ti.f64)
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        velocity_scale = ti.cast(ti.static(self._velocity_scale), ti.f64)
        for face_i, j in self._water_energy_flux_x:
            energy_flux = ti.cast(0.0, ti.f64)
            volume_flux = ti.cast(0.0, ti.f64)
            if ti.static(self._advection_enabled):
                if 0 < face_i < ti.static(self.nx):
                    left_i = face_i - 1
                    right_i = face_i
                    left_active = (
                        wall_mask[left_i, j] == 0
                        and solid_mask[left_i, j] == 0
                        and water_phase[left_i, j] > ti.static(self._water_phase_cutoff)
                    )
                    right_active = (
                        wall_mask[right_i, j] == 0
                        and solid_mask[right_i, j] == 0
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
                        upwind_fraction = ti.min(1.0, ti.max(0.0, upwind_volume / area))
                        volume_flux = speed * upwind_fraction * dx
                        energy_density = self.water_sensible_energy[
                            upwind_i, j
                        ] / ti.max(upwind_volume, 1.0e-30)
                        energy_flux = volume_flux * energy_density
            self._water_energy_flux_x[face_i, j] = energy_flux
            self._water_volume_flux_x[face_i, j] = volume_flux

    @ti.kernel
    def _compute_water_conduction_flux_y(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        k_water = ti.cast(ti.static(self._water_conductivity), ti.f64)
        for i, face_j in self._water_energy_flux_y:
            flux = ti.cast(0.0, ti.f64)
            if 0 < face_j < ti.static(self.ny):
                bottom_j = face_j - 1
                top_j = face_j
                bottom_active = (
                    wall_mask[i, bottom_j] == 0
                    and solid_mask[i, bottom_j] == 0
                    and water_phase[i, bottom_j] > ti.static(self._water_phase_cutoff)
                )
                top_active = (
                    wall_mask[i, top_j] == 0
                    and solid_mask[i, top_j] == 0
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
                    wall_mask[i, bottom_j] != 0
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
                elif (
                    bottom_active
                    and self.water_volume_m2[i, bottom_j] > 1.0e-30
                    and wall_mask[i, top_j] != 0
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
            self._water_energy_flux_y[i, face_j] = flux

    @ti.kernel
    def _compute_water_advection_flux_y(
        self,
        velocity: ti.template(),
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        dx = ti.cast(ti.static(self._cell_size_m), ti.f64)
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        velocity_scale = ti.cast(ti.static(self._velocity_scale), ti.f64)
        for i, face_j in self._water_energy_flux_y:
            energy_flux = ti.cast(0.0, ti.f64)
            volume_flux = ti.cast(0.0, ti.f64)
            if ti.static(self._advection_enabled):
                if 0 < face_j < ti.static(self.ny):
                    bottom_j = face_j - 1
                    top_j = face_j
                    bottom_active = (
                        wall_mask[i, bottom_j] == 0
                        and solid_mask[i, bottom_j] == 0
                        and water_phase[i, bottom_j]
                        > ti.static(self._water_phase_cutoff)
                    )
                    top_active = (
                        wall_mask[i, top_j] == 0
                        and solid_mask[i, top_j] == 0
                        and water_phase[i, top_j] > ti.static(self._water_phase_cutoff)
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
                        upwind_fraction = ti.min(1.0, ti.max(0.0, upwind_volume / area))
                        volume_flux = speed * upwind_fraction * dx
                        energy_density = self.water_sensible_energy[
                            i, upwind_j
                        ] / ti.max(upwind_volume, 1.0e-30)
                        energy_flux = volume_flux * energy_density
            self._water_energy_flux_y[i, face_j] = energy_flux
            self._water_volume_flux_y[i, face_j] = volume_flux

    @ti.kernel
    def _update_water_advection(self, time_step_s: ti.f64, wall_mask: ti.template()):
        for i, j in self.water_sensible_energy:
            energy_divergence = (
                self._water_energy_flux_x[i + 1, j]
                - self._water_energy_flux_x[i, j]
                + self._water_energy_flux_y[i, j + 1]
                - self._water_energy_flux_y[i, j]
            )
            volume_divergence = (
                self._water_volume_flux_x[i + 1, j]
                - self._water_volume_flux_x[i, j]
                + self._water_volume_flux_y[i, j + 1]
                - self._water_volume_flux_y[i, j]
            )
            if wall_mask[i, j] == 0:
                self.water_sensible_energy[i, j] -= time_step_s * energy_divergence
                self.water_volume_m2[i, j] -= time_step_s * volume_divergence

    @ti.kernel
    def _update_water_conduction(self, time_step_s: ti.f64, wall_mask: ti.template()):
        for i, j in self.water_sensible_energy:
            energy_divergence = (
                self._water_energy_flux_x[i + 1, j]
                - self._water_energy_flux_x[i, j]
                + self._water_energy_flux_y[i, j + 1]
                - self._water_energy_flux_y[i, j]
            )
            if wall_mask[i, j] == 0:
                self.water_sensible_energy[i, j] -= time_step_s * energy_divergence

    @ti.kernel
    def _exchange_interface_heat(
        self,
        time_step_s: ti.f64,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        """Compute each contact once, cap by available water heat, then scatter.

        Requests and material indices live in thread-local registers, not world
        fields. Water threads only modify their own energy; material heat deltas
        use atomics and are consumed by the next launch.
        """
        conductivity = ti.cast(
            ti.static(
                2.0
                * self._water_conductivity
                * self._ice_conductivity
                / (self._water_conductivity + self._ice_conductivity)
            ),
            ti.f64,
        )
        for i, j in self.water_volume_m2:
            if (
                wall_mask[i, j] == 0
                and solid_mask[i, j] == 0
                and water_phase[i, j] > self._water_phase_cutoff
                and self.water_volume_m2[i, j] > 1.0e-30
            ):
                requests = ti.Vector.zero(ti.f64, 20 * self.body_count)
                targets = ti.Vector.zero(ti.i32, 20 * self.body_count)
                requested_heat = ti.cast(0.0, ti.f64)
                water_aperture = ti.min(
                    1.0, self.water_volume_m2[i, j] / self._cell_area
                )
                water_temperature = self._water_temperature_at(i, j)
                for b in range(self.body_count):
                    for face in ti.static(range(5)):
                        offset = ti.static(
                            ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))[face]
                        )
                        world_i, world_j = i + offset[0], j + offset[1]
                        if 0 <= world_i < self.nx and 0 <= world_j < self.ny:
                            coverage = ti.cast(
                                self.body_world_fraction[b, world_i, world_j], ti.f64
                            )
                            if ti.static(self.body_count == 1):
                                coverage = ti.cast(
                                    self.world_body_solid_fraction[world_i, world_j],
                                    ti.f64,
                                )
                            if coverage > 0.0:
                                contact = ti.min(
                                    water_aperture,
                                    ti.min(
                                        1.0, coverage / self._solid_fraction_threshold
                                    ),
                                )
                                grid = self._body_grid_coordinates(world_i, world_j, b)
                                base_i, base_j = (
                                    ti.cast(ti.floor(grid.x), ti.i32),
                                    ti.cast(ti.floor(grid.y), ti.i32),
                                )
                                fraction_x, fraction_y = (
                                    ti.cast(grid.x - base_i, ti.f64),
                                    ti.cast(grid.y - base_j, ti.f64),
                                )
                                for offset_i, offset_j in ti.static(ti.ndrange(2, 2)):
                                    body_i, body_j = (
                                        base_i + offset_i,
                                        base_j + offset_j,
                                    )
                                    if (
                                        self.material_offset[b]
                                        <= body_i
                                        < self.material_offset[b]
                                        + self.material_size[b].x
                                        and 0 <= body_j < self.material_size[b].y
                                    ):
                                        if self.body_solid_mass[body_i, body_j] > 0.0:
                                            weight_x = (
                                                fraction_x
                                                if ti.static(offset_i == 1)
                                                else 1.0 - fraction_x
                                            )
                                            weight_y = (
                                                fraction_y
                                                if ti.static(offset_j == 1)
                                                else 1.0 - fraction_y
                                            )
                                            share = (
                                                weight_x
                                                * weight_y
                                                * ti.cast(
                                                    self._body_fraction_at(
                                                        body_i, body_j
                                                    ),
                                                    ti.f64,
                                                )
                                                / coverage
                                            )
                                            heat = (
                                                time_step_s
                                                * conductivity
                                                * contact
                                                * share
                                                * ti.max(
                                                    0.0,
                                                    water_temperature
                                                    - self._body_temperature_at(
                                                        body_i, body_j
                                                    ),
                                                )
                                            )
                                            slot = b * 20 + ti.static(
                                                face * 4 + offset_i * 2 + offset_j
                                            )
                                            requests[slot] = heat
                                            targets[slot] = (
                                                body_i * self.body_ny + body_j
                                            )
                                            requested_heat += heat
                scale = ti.cast(0.0, ti.f64)
                if requested_heat > 0.0:
                    scale = ti.min(
                        1.0,
                        ti.max(0.0, self.water_sensible_energy[i, j]) / requested_heat,
                    )
                transferred_heat = ti.cast(0.0, ti.f64)
                for slot in range(20 * self.body_count):
                    heat = scale * requests[slot]
                    if heat > 0.0:
                        body_i, body_j = (
                            targets[slot] // self.body_ny,
                            targets[slot] % self.body_ny,
                        )
                        ti.atomic_add(self._body_heat_delta[body_i, body_j], heat)
                        transferred_heat += heat
                self.water_sensible_energy[i, j] -= transferred_heat

    @ti.kernel
    def _apply_body_heat_and_phase_change(self):
        """Warm the remaining ice to the melting point, then erase melted mass."""
        latent = ti.cast(ti.static(self._latent_heat_j_kg), ti.f64)
        for i, j in self.body_solid_mass:
            mass = self.body_solid_mass[i, j]
            old_mass = mass
            sensible = self.body_sensible_energy[i, j] + self._body_heat_delta[i, j]
            extra_heat = ti.max(0.0, sensible - mass * latent)
            if sensible > 0.0:
                mass = ti.max(0.0, mass - sensible / latent)
                sensible = 0.0
            # Discard sub-grid residue so an eroded cell can disappear completely.
            if ti.static(not self._conservative_phase_enabled):
                if mass <= ti.static(1.0e-6 * self._initial_body_cell_mass):
                    mass = 0.0
                    sensible = 0.0
            self.body_solid_mass[i, j] = mass
            self.body_sensible_energy[i, j] = sensible
            self.melted_mass[i, j] = old_mass - mass
            self.melted_sensible_energy[i, j] = extra_heat

    @ti.func
    def _body_fraction_at(self, i, j):
        mass = ti.min(
            self._initial_body_cell_mass, ti.max(0.0, self.body_solid_mass[i, j])
        )
        fraction = mass / self._initial_body_cell_mass
        if mass > 0.0:
            fraction = ti.max(fraction, 1.0e-30)
        return ti.cast(fraction, ti.f32)

    @ti.func
    def _body_temperature_at(self, i, j):
        temperature = ti.cast(ti.static(self._melting_temperature_c), ti.f64)
        mass = self.body_solid_mass[i, j]
        if mass > 1.0e-30:
            temperature += self.body_sensible_energy[i, j] / (
                mass * self._ice_specific_heat
            )
        return ti.min(ti.cast(self._melting_temperature_c, ti.f64), temperature)

    @ti.func
    def _water_temperature_at(self, i, j):
        temperature = ti.cast(ti.static(self.config.initial_air_temperature_c), ti.f64)
        volume = self.water_volume_m2[i, j]
        minimum = ti.cast(1e-30, ti.f64)
        if ti.static(self._conservative_phase_enabled):
            minimum = self.conservative_phase.cell_area[None] * 1e-12
        if volume > minimum:
            temperature = self._melting_temperature_c + self.water_sensible_energy[
                i, j
            ] / (self._water_density_kg_m3 * self._water_specific_heat * volume)
        return temperature

    @ti.func
    def _body_grid_coordinates(self, i, j, b=0):
        origin = ti.Vector.zero(ti.f32, 2)
        angle = ti.cast(0.0, ti.f32)
        if ti.static(self.body_count == 1):
            origin = self.body_reference_origin[None]
            angle = self.body_angle[None]
        else:
            origin = self.body_reference_origin[b]
            angle = self.body_angle[b]
        cosine, sine = ti.cos(angle), ti.sin(angle)
        relative_x, relative_y = (
            ti.cast(i, ti.f32) + 0.5 - origin.x,
            ti.cast(j, ti.f32) + 0.5 - origin.y,
        )
        return ti.Vector(
            [
                cosine * relative_x
                + sine * relative_y
                + 0.5 * self.material_size[b].x
                - 0.5
                + self.material_offset[b],
                -sine * relative_x
                + cosine * relative_y
                + 0.5 * self.material_size[b].y
                - 0.5,
            ]
        )

    @ti.func
    def _water_aperture_target(
        self,
        i,
        j,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        volume = ti.cast(0.0, ti.f64)
        phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
        if (
            wall_mask[i, j] == 0
            and solid_mask[i, j] == 0
            and phase > self._water_phase_cutoff
        ):
            volume = self._cell_area * ti.cast(phase, ti.f64)
        if ti.static(self._conservative_phase_enabled):
            volume = self.conservative_phase.cell_area[None] * self.conservative_phase.volume(i, j)
        return volume

    def _body_fraction_numpy(self):
        mass = np.clip(
            self.body_solid_mass.to_numpy(), 0.0, self._initial_body_cell_mass
        )
        fraction = mass / self._initial_body_cell_mass
        return np.where(mass > 0.0, np.maximum(fraction, 1.0e-30), 0.0).astype(
            np.float32
        )

    def _body_temperature_numpy(self):
        mass = self.body_solid_mass.to_numpy()
        sensible = self.body_sensible_energy.to_numpy()
        temperature = np.full(mass.shape, self._melting_temperature_c, dtype=np.float64)
        np.divide(
            sensible,
            mass * self._ice_specific_heat,
            out=temperature,
            where=mass > 1.0e-30,
        )
        temperature[mass > 1.0e-30] += self._melting_temperature_c
        return np.minimum(self._melting_temperature_c, temperature)

    def _water_temperature_numpy(self):
        volume = self.water_volume_m2.to_numpy()
        sensible = self.water_sensible_energy.to_numpy()
        temperature = np.full(
            volume.shape, self.config.initial_air_temperature_c, dtype=np.float64
        )
        wet = volume > (self._cell_area * 1e-12 if self._conservative_phase_enabled else 1e-30)
        temperature[wet] = self._melting_temperature_c + sensible[wet] / (
            self._water_density_kg_m3 * self._water_specific_heat * volume[wet]
        )
        return temperature

    def sample_world_fields(self, wall_mask):
        """Build output-only arrays on the host at a snapshot boundary."""
        volume = self.water_volume_m2.to_numpy()
        water_energy = self.water_sensible_energy.to_numpy()
        mass = self.body_solid_mass.to_numpy()
        body_energy = self.body_sensible_energy.to_numpy()
        wet = volume > (self._cell_area * 1e-12 if self._conservative_phase_enabled else 1e-30)
        temperature = np.full(
            volume.shape, self.config.initial_air_temperature_c, dtype=np.float64
        )
        temperature[wet] = self._melting_temperature_c + water_energy[wet] / (
            self._water_density_kg_m3 * self._water_specific_heat * volume[wet]
        )
        enthalpy = np.where(
            wet,
            (self._water_density_kg_m3 * self._latent_heat_j_kg * volume + water_energy)
            / self._cell_area,
            0.0,
        )
        liquid = np.clip(volume / self._cell_area, 0.0, 1.0).astype(np.float32)
        material = wet.copy()
        sizes = self.material_size.to_numpy()
        offsets = self.material_offset.to_numpy()
        coverage = self.body_world_fraction.to_numpy()
        if self.body_count == 1:
            coverage[0] = self.world_body_solid_fraction.to_numpy()
        for b in range(self.body_count):
            origin = np.asarray(
                self.body_reference_origin[None if self.body_count == 1 else b],
                dtype=np.float32,
            )
            angle = np.float32(self.body_angle[None if self.body_count == 1 else b])
            cosine, sine = np.cos(angle), np.sin(angle)
            x, y = np.meshgrid(
                np.arange(self.nx, dtype=np.float32) + 0.5 - origin[0],
                np.arange(self.ny, dtype=np.float32) + 0.5 - origin[1],
                indexing="ij",
            )
            body_i = (
                np.floor(cosine * x + sine * y + 0.5 * sizes[b, 0]).astype(np.int32)
                + offsets[b]
            )
            body_j = np.floor(-sine * x + cosine * y + 0.5 * sizes[b, 1]).astype(
                np.int32
            )
            ice = (
                (coverage[b] >= self._solid_fraction_threshold)
                & (body_i >= offsets[b])
                & (body_i < offsets[b] + sizes[b, 0])
                & (body_j >= 0)
                & (body_j < sizes[b, 1])
            )
            selected_mass = mass[body_i[ice], body_j[ice]]
            selected_energy = body_energy[body_i[ice], body_j[ice]]
            body_temperature = np.full(
                selected_mass.shape, self._melting_temperature_c, dtype=np.float64
            )
            live = selected_mass > 1.0e-30
            body_temperature[live] += selected_energy[live] / (
                selected_mass[live] * self._ice_specific_heat
            )
            temperature[ice] = np.minimum(self._melting_temperature_c, body_temperature)
            enthalpy[ice] = selected_energy / self._cell_area
            liquid[ice] = 0.0
            material[ice] = True
        wall_mask = wall_mask != 0
        (
            temperature[wall_mask],
            enthalpy[wall_mask],
            liquid[wall_mask],
            material[wall_mask],
        ) = (self._melting_temperature_c, 0.0, 0.0, False)
        return {
            "temperature_c": temperature,
            "enthalpy_j_m3": enthalpy,
            "liquid_fraction": liquid,
            "phase_change_material": material.astype(np.int8),
        }


@ti.data_oriented
class IceFlow2D(StabilityOptions):
    """CUDA solver for the coupled falling-ice melting scenario."""

    def __init__(self, config):
        if not isinstance(config, IceFlowConfig):
            raise TypeError("config must be an IceFlowConfig")
        if config.thermal is None:
            raise ValueError("IceFlow2D requires thermal body-ALE coupling")
        if config.ice_fixed or config.thermal.moving_body_scheme != "body_ale":
            raise ValueError("IceFlow2D supports only freely moving body-ALE ice")
        if config.rigid_boundary_scheme != "unified":
            raise ValueError("IceFlow2D supports only unified cut-link boundaries")
        if not config.well_balanced_hydrostatics:
            raise ValueError("IceFlow2D requires well-balanced hydrostatics")
        if config.thermal.water_buoyancy_model != "linear":
            raise ValueError("IceFlow2D supports only the linear buoyancy model")
        ensure_taichi_cuda()
        self.config = config
        self._conservative_phase_enabled = config.water_mass_conservation == "conservative_lbm"
        self._phase_dtype = ti.f64 if self._conservative_phase_enabled else ti.f32
        self.body_count = len(config.bodies)
        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.steps = 0
        self.scales = LatticeScales.from_iceflow_config(config)

        # Dimensional-to-lattice conversion.
        u_ref_lattice = self.scales.reference_lattice_velocity
        u_ref = self.scales.reference_velocity_m_s
        c_u = self.scales.velocity_scale_m_s
        self._time_step_s = self.scales.dt_s
        self._water_density_lattice = 1.0
        self._air_density_lattice = float(config.rho_air / config.rho_water)
        self._viscosity_to_lattice = self.scales.dt_s / config.dx**2
        self._temperature_dependent_water_viscosity = config.viscosity_water is None
        self._water_viscosity_lattice = (
            config.water_viscosity_at(config.thermal.initial_water_temperature_c)
            * self._viscosity_to_lattice
        )
        self._air_viscosity_lattice = (
            config.air_viscosity_at(config.thermal.initial_air_temperature_c)
            * self._viscosity_to_lattice
        )
        self._gravity_lattice = (
            float(config.gravity[0] * config.dx / (c_u * c_u)),
            float(config.gravity[1] * config.dx / (c_u * c_u)),
        )
        self._surface_tension_lattice = float(
            config.sigma
            * u_ref_lattice
            * u_ref_lattice
            / (u_ref * u_ref)
            / config.dx
            / config.rho_water
        )
        masses = np.asarray(
            [
                config.rho_ice / config.rho_water * b.size_cells[0] * b.size_cells[1]
                for b in config.bodies
            ]
        )
        inertias = np.asarray(
            [
                m * (b.size_cells[0] ** 2 + b.size_cells[1] ** 2) / 12.0
                for m, b in zip(masses, config.bodies)
            ]
        )
        self._initial_body_mass_lattice = masses[0] if self.body_count == 1 else masses
        self._initial_body_inertia_lattice = (
            inertias[0] if self.body_count == 1 else inertias
        )
        self._initial_masses = ti.field(ti.f64, shape=self.body_count)
        self._initial_inertias = ti.field(ti.f64, shape=self.body_count)
        self._initial_masses.from_numpy(masses)
        self._initial_inertias.from_numpy(inertias)
        shape_xy = (self.nx, self.ny)

        # The momentum distribution carries pressure and momentum:
        # sum(c*f)=rho(phi)*u and its second moment contains pressure, while
        # sum(f) is deliberately not material density.  h is the conservative
        # Allen--Cahn phase distribution.
        self.momentum_populations = self._allocate_populations()
        self.momentum_stream_buffer = self._allocate_populations()
        self.phase_populations = self._allocate_populations(self._phase_dtype)
        self.phase_stream_buffer = self._allocate_populations(self._phase_dtype)
        self.water_phase = ti.field(self._phase_dtype, shape=shape_xy)
        self.momentum_velocity_lattice = ti.Vector.field(2, ti.f32, shape=shape_xy)
        # Retained across streaming because pressure reconstruction uses the
        # half-force velocity at the next time level.
        self.fluid_acceleration_lattice = ti.Vector.field(2, ti.f32, shape=shape_xy)
        if (
            config.momentum_collision_model == "cm_hrr"
            or config.phase_momentum_coupling == "consistent_flux"
        ):
            self.stability_velocity = ti.Vector.field(2, ti.f32, shape=shape_xy)
        if config.phase_momentum_coupling == "consistent_flux":
            self.phase_mass_flux = ti.Vector.field(2, ti.f32, shape=shape_xy)
        # Symmetric capillary stress (xx, xy, yy), refreshed before collision.
        # Fluid forcing and contact-line traction use this same time level.
        self.capillary_stress_lattice = ti.Vector.field(3, ti.f32, shape=shape_xy)
        self.pressure_lattice = ti.field(ti.f32, shape=shape_xy)
        self.reference_pressure_by_row = ti.field(ti.f32, shape=self.ny)
        self.reference_density_by_row = ti.field(ti.f32, shape=self.ny)
        self.hydrostatic_reference_pressure = _DerivedField(
            self._reference_pressure_at,
            lambda: np.broadcast_to(
                self.reference_pressure_by_row.to_numpy(), (self.nx, self.ny)
            ),
        )
        self.hydrostatic_reference_density = _DerivedField(
            self._reference_density_at,
            lambda: np.broadcast_to(
                self.reference_density_by_row.to_numpy(), (self.nx, self.ny)
            ),
        )

        # Wall and distance are read-only computed views. The two masks hold
        # committed time levels while contact/thermal kernels update coverage.
        self.wall_mask = _DerivedField(self._wall_at, self._wall_numpy)
        self.solid_mask = ti.field(ti.i8, shape=shape_xy)
        self.body_signed_distance_m = _DerivedField(
            self._signed_distance_at, self._signed_distance_numpy
        )
        self.previous_solid_mask = ti.field(ti.i8, shape=shape_xy)
        self.solid_body_id = ti.field(ti.i32, shape=shape_xy)
        self.previous_body_id = ti.field(ti.i32, shape=shape_xy)
        self.solid_body_id.fill(-1)
        self.previous_body_id.fill(-1)
        self._body_has_resolved_boundary = BodyField(ti.i32, self.body_count)

        # Rigid state and the only load accumulator consumed by integration.
        self.body_center = _DerivedField(
            self._body_center_at,
            self._body_center_numpy,
            host_writer=self._set_body_center_host,
            single_body=self.body_count == 1,
        )
        self.body_velocity = BodyField(ti.f32, self.body_count, components=2)
        self.body_angle = BodyField(ti.f32, self.body_count)
        self.body_angular_velocity = BodyField(ti.f32, self.body_count)
        # Dynamic mass properties let the material-frame thermal path erode
        # the rigid remnant.
        self.body_initial_mass_lattice = _DerivedField(
            self._read_body_initial_mass_lattice,
            lambda: self._initial_body_mass_lattice,
            single_body=self.body_count == 1,
        )
        self.body_mass_lattice = BodyField(ti.f64, self.body_count)
        self.body_inertia_lattice = BodyField(ti.f64, self.body_count)
        self.body_local_center_of_mass = BodyField(
            ti.f64, self.body_count, components=2
        )
        self.body_reference_origin = BodyField(ti.f32, self.body_count, components=2)
        self.body_active = _DerivedField(
            self._read_body_active,
            lambda: (self.body_inertia_lattice.to_numpy() > 0.0).astype(np.int8),
            single_body=self.body_count == 1,
        )
        self.cumulative_melted_mass_lattice = _DerivedField(
            self._read_cumulative_melted_mass_lattice,
            lambda: self._initial_body_mass_lattice - self.body_mass_lattice.to_numpy(),
            single_body=self.body_count == 1,
        )
        self._body_reduced_mass = BodyField(ti.f64, self.body_count)
        self._body_reduced_first_moment = BodyField(
            ti.f64, self.body_count, components=2
        )
        self._body_reduced_inertia_origin = BodyField(ti.f64, self.body_count)
        # For an eroding body, container contact is reconstructed from the
        # current unclipped sharp world raster rather than the initial rectangle.
        # The four entries are min-x, max-x, min-y and max-y in the world
        # orientation but relative to ``body_reference_origin``.
        self._body_contact_support_extrema = BodyField(
            ti.f32, self.body_count, tail=(4,)
        )
        self._body_contact_geometry_active = _DerivedField(
            self._read__body_contact_geometry_active,
            lambda: (
                self._body_contact_support_extrema.to_numpy()[..., 1]
                > self._body_contact_support_extrema.to_numpy()[..., 0]
            ).astype(np.int8),
        )
        # The continuous body coverage is first evaluated without applying
        # the container mask.  Contact reduction consumes this scratch field
        # so the same fraction evaluation can be reused by the world
        # rasterizer; a second evaluation is needed only when projection
        # changes the accepted pose.
        self._body_contact_projection_changed = BodyField(ti.i8, self.body_count)
        # Cut-link loads are signed sums with strong cancellation.  CUDA is
        # free to order global atomics differently as block scheduling
        # changes, so reduce in f64 and round only at rigid integration.
        self.hydrodynamic_impulse = BodyField(ti.f64, self.body_count, components=2)
        self.hydrodynamic_torque = BodyField(ti.f64, self.body_count)
        self._thermal_lbm_steps_pending = 0
        # The full thermal/LBM stability scan is useful while debugging, but
        # it can require a device-to-host field copy on the error path.  Keep
        # it opt-in and do not expose it as a per-call argument; the fast
        # coupling still prepares its velocity field and CFL reduction every
        # step when this is disabled.
        self._thermal_stability_check_enabled = False
        self.thermal = MovingBodyThermal2D(
            self.nx,
            self.ny,
            config.ice_width,
            config.ice_height,
            config.thermal,
            self.scales,
            density_water_kg_m3=config.rho_water,
            density_ice_kg_m3=config.rho_ice,
            water_phase_cutoff=config.water_phase_cutoff,
            bodies=config.bodies,
        )
        # Share references to the authoritative pose and the one coverage
        # allocation; the thermal component owns no second pose or raster.
        self.thermal.body_reference_origin = self.body_reference_origin
        self.thermal.body_angle = self.body_angle
        self._moving_body_fraction = self.thermal.world_body_solid_fraction
        self._body_fractions = self.thermal.body_world_fraction
        if config.close_subgrid_wall_gaps:
            self.subgrid_wall_gap = ti.field(ti.i8, shape=shape_xy)
            self.previous_subgrid_wall_gap = ti.field(ti.i8, shape=shape_xy)
        if self._conservative_phase_enabled:
            self.conservative_phase = ConservativePhaseLBM(self)
            self.thermal.conservative_phase = self.conservative_phase
            self.thermal._conservative_phase_enabled = True
        self.temperature = _DerivedField(
            None,
            lambda: self.sample_thermal_fields()["temperature_c"],
        )
        self.thermal_enthalpy = _DerivedField(
            None, lambda: self.sample_thermal_fields()["enthalpy_j_m3"]
        )
        self.liquid_fraction = _DerivedField(
            None, lambda: self.sample_thermal_fields()["liquid_fraction"]
        )
        self.phase_change_material = _DerivedField(
            None, lambda: self.sample_thermal_fields()["phase_change_material"]
        )
        self.physical_velocity_lattice = _DerivedField(
            self._physical_velocity_at, self._physical_velocity_numpy
        )
        self.maximum_water_outflow_rate_lattice = ti.field(ti.f32, shape=())
        self.maximum_fluid_speed_l1 = ti.field(ti.f32, shape=())
        self.thermal_state_invalid = ti.field(ti.i32, shape=())
        self.phase_change_initial_solid_volume = _DerivedField(
            self._read_phase_change_initial_solid_volume,
            lambda: float(
                sum(b.size_cells[0] * b.size_cells[1] for b in self.config.bodies)
            ),
        )
        self.phase_change_current_solid_volume = _DerivedField(
            self._read_phase_change_current_solid_volume,
            lambda: float(self.body_mass_lattice.array().sum())
            * (self.config.rho_water / self.config.rho_ice),
        )

        self.contacts = BodyContacts(self) if self.body_count > 1 else None
        self._initialize_body()
        self._initialize_geometry()
        # Seed the material fractions before the first world raster so the
        # canonical SDF and the initial sharp mask are derived from the same
        # geometry.  The full thermal state is initialized later, after the
        # diffuse phase warm-up has reached its final composition.
        self.thermal._initialize_body_state(self.thermal._initial_body_cell_mass)
        self._solve_contact_and_rasterize()
        self._initialize_moving_body_geometry()
        self._initialize_fluid()
        if self._conservative_phase_enabled:
            self.conservative_phase.update_capacity()
            self.conservative_phase.project_melting(0.0)
            self.conservative_phase.initialize()
        if config.phase_warmup_steps > 0:
            self._warm_start_phase(config.phase_warmup_steps)
        self._update_streamed_macroscopic_fields()
        self._bound_water_phase()
        # Initialize heat after the water phase warm-up has finished.
        self.thermal.initialize(self.water_phase, self.wall_mask, self.solid_mask)

        self._update_moving_body_mass_properties(initialize=True)
        self._build_hydrostatic_reference()
        self._initialize_hydrostatic_equilibrium()
        if config.initial_water_geometry == "dam_break":
            self._initialize_dam_break_pressure()

    def step(self, num_steps=1):
        """Advance flow, rigid motion, and heat in their coupling order."""
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, (int, np.integer))
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a non-negative integer")
        for _ in range(num_steps):
            self._advance_fluid()
            self._advance_body()
            self._advance_thermal()
            self.steps += 1

    def _advance_fluid(self):
        self._collide_momentum()
        if self.config.phase_momentum_coupling == "legacy":
            self._collide_phase()
            if self._conservative_phase_enabled:
                self.conservative_phase.prepare_transport()
        if self._conservative_phase_enabled:
            self.conservative_phase.begin_fluid(1 if self.config.thermal.advection_enabled else 0)
        self._stream()
        if self._conservative_phase_enabled:
            self.conservative_phase.limit_stream()
            self.conservative_phase.finish_energy()
        self._update_streamed_macroscopic_fields()
        self._update_pressure()
        if self._conservative_phase_enabled:
            self.conservative_phase.repair_bounds()
        else:
            self._bound_water_phase()

    def _advance_body(self):
        self._integrate_rigid_ice()
        self._refresh_fluid_domain(resolve_contact=True)

    def _refresh_fluid_domain(self, *, resolve_contact):
        """Commit geometry and reconstruct exposed ambient fluid in one place."""
        self._solve_contact_and_rasterize(resolve_contact=resolve_contact)
        self._update_solid_mask()
        if self._conservative_phase_enabled:
            self.conservative_phase.cache_momentum_phase()
        self._refill_changed_nodes()
        if self._conservative_phase_enabled:
            thermal_steps = 0.0 if resolve_contact else self.thermal.last_thermal_substep_s / self._time_step_s
            self.conservative_phase.update_capacity()
            if thermal_steps > 0:
                self.conservative_phase.project_melting(thermal_steps)
            self.conservative_phase.remap()
            self._synchronize_conservative_momentum()
            if self.thermal._d2q5 is not None:
                self.thermal._d2q5.synchronize()

    def _advance_thermal(self):
        self._advance_moving_thermal_fast(target_step=self.steps + 1)
        self._thermal_lbm_steps_pending += 1
        if (
            self._thermal_lbm_steps_pending
            >= self.config.thermal.update_interval_lbm_steps
        ):
            self._advance_moving_thermal_slow(self._thermal_lbm_steps_pending)
            self._thermal_lbm_steps_pending = 0

    @property
    def physical_time_s(self):
        return self.steps * self._time_step_s

    def _advance_moving_thermal_fast(self, *, target_step):
        """Remap the moving aperture and advect water/energy at every LBM step."""

        if getattr(self, "_conservative_phase_enabled", False):
            if self._thermal_stability_check_enabled:
                self._check_thermal_advection_stability(target_step=target_step)
            return
        self._measure_water_outflow_rate()
        maximum_velocity_l1 = float(self.maximum_water_outflow_rate_lattice[None])
        if self._thermal_stability_check_enabled:
            self._check_thermal_advection_stability(target_step=target_step)
        self.thermal.advance_fast(
            self._time_step_s,
            self.physical_velocity_lattice,
            self.water_phase,
            self.wall_mask,
            self.solid_mask,
            maximum_outflow_rate_lattice=maximum_velocity_l1,
        )

    def _advance_moving_thermal_slow(self, lbm_steps):
        """Conduct heat, erase melted ice, and refresh the remaining geometry."""
        self.thermal.advance_slow(
            lbm_steps * self._time_step_s,
            self.water_phase,
            self.wall_mask,
            self.solid_mask,
            self.body_reference_origin,
            self.body_angle,
            on_body_eroded=self._refresh_eroded_body,
        )

    def _refresh_eroded_body(self):
        """Finish each erosion substep before any further heat exchange."""
        if self._conservative_phase_enabled:
            self.conservative_phase.cache_momentum_phase()
        self._update_moving_body_mass_properties()
        self._refresh_fluid_domain(resolve_contact=False)
        self.thermal.synchronize_water_aperture(
            self.water_phase, self.wall_mask, self.solid_mask
        )

    @ti.kernel
    def _synchronize_conservative_momentum(self):
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                before = ti.cast(self.conservative_phase.momentum_phase[i, j], ti.f32)
                after = ti.cast(self.water_phase[i, j], ti.f32)
                if before != after:
                    old_density = mixture_density(before, self._water_density_lattice, self._air_density_lattice)
                    new_density = mixture_density(after, self._water_density_lattice, self._air_density_lattice)
                    pressure = self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
                    velocity = self.momentum_velocity_lattice[i, j]
                    for q in ti.static(range(9)):
                        self.momentum_populations[i, j, q] += momentum_equilibrium(q, pressure, new_density, velocity) - momentum_equilibrium(q, pressure, old_density, velocity)

    # Geometry and initialization

    @ti.func
    def _is_fluid_cell(self, i, j):
        return self.wall_mask[i, j] == 0 and self.solid_mask[i, j] == 0

    @ti.func
    def _is_phase_cell(self, i, j):
        active = self._is_fluid_cell(i, j)
        if ti.static(self._conservative_phase_enabled):
            active = self.conservative_phase.cell(i, j)
        return active

    @ti.func
    def _phase_velocity_at(self, i, j):
        velocity = self.momentum_velocity_lattice[i, j] + 0.5 * self.fluid_acceleration_lattice[i, j]
        if self.solid_mask[i, j] != 0:
            velocity = self._body_velocity_at(self._cell_point(i, j), self.solid_body_id[i, j])
        return velocity

    @ti.kernel
    def _measure_water_outflow_rate(self):
        """Bound the sum of outward face speeds for upwind volume positivity."""
        self.maximum_water_outflow_rate_lattice[None] = 0.0
        for i, j in self.water_phase:
            if (
                self._is_fluid_cell(i, j)
                and self.water_phase[i, j] > self.config.water_phase_cutoff
            ):
                velocity = self._physical_velocity_at(i, j)
                outflow = 0.0
                for offset_i, offset_j in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    neighbor_i, neighbor_j = i + offset_i, j + offset_j
                    if inside_grid(neighbor_i, neighbor_j, self.nx, self.ny):
                        if (
                            self._is_fluid_cell(neighbor_i, neighbor_j)
                            and self.water_phase[neighbor_i, neighbor_j]
                            > self.config.water_phase_cutoff
                        ):
                            neighbor_velocity = self._physical_velocity_at(
                                neighbor_i, neighbor_j
                            )
                            face_velocity = 0.5 * (velocity + neighbor_velocity)
                            outflow += ti.max(
                                0.0,
                                offset_i * face_velocity.x + offset_j * face_velocity.y,
                            )
                if (
                    ti.math.isnan(velocity.x)
                    or ti.math.isnan(velocity.y)
                    or ti.math.isinf(velocity.x)
                    or ti.math.isinf(velocity.y)
                ):
                    outflow = float("inf")
                ti.atomic_max(self.maximum_water_outflow_rate_lattice[None], outflow)

    @ti.kernel
    def _collect_thermal_stability_metrics(self):
        """Collect optional finite-state and full-fluid speed diagnostics."""

        self.maximum_fluid_speed_l1[None] = 0.0
        self.thermal_state_invalid[None] = 0
        for i, j in self.momentum_velocity_lattice:
            if self._is_fluid_cell(i, j):
                velocity = self.physical_velocity_lattice[i, j]
                state_invalid = (
                    ti.math.isnan(velocity.x)
                    or ti.math.isinf(velocity.x)
                    or ti.math.isnan(velocity.y)
                    or ti.math.isinf(velocity.y)
                    or ti.math.isnan(self.water_phase[i, j])
                    or ti.math.isinf(self.water_phase[i, j])
                )
                for q in range(D2Q9_DIRECTION_COUNT):
                    state_invalid = (
                        state_invalid
                        or ti.math.isnan(self.phase_populations[i, j, q])
                        or ti.math.isinf(self.phase_populations[i, j, q])
                    )
                if state_invalid:
                    ti.atomic_max(self.thermal_state_invalid[None], 1)
                ti.atomic_max(
                    self.maximum_fluid_speed_l1[None],
                    ti.abs(velocity.x) + ti.abs(velocity.y),
                )

    def _check_thermal_advection_stability(self, *, target_step):
        """Run the opt-in LBM sanity scan before thermal subcycling."""

        self._collect_thermal_stability_metrics()
        maximum_water_velocity_l1 = float(self.maximum_water_outflow_rate_lattice[None])
        maximum_fluid_velocity_l1 = float(self.maximum_fluid_speed_l1[None])
        self._validate_thermal_advection_velocity(
            maximum_water_velocity_l1,
            maximum_fluid_velocity_l1,
            target_step=target_step,
        )

    def _validate_thermal_advection_velocity(
        self,
        maximum_water_velocity_l1,
        maximum_fluid_velocity_l1,
        *,
        target_step,
    ):
        """Reject a failed LBM state after the optional metrics scan."""

        maximum_water = float(maximum_water_velocity_l1)
        maximum_fluid = float(maximum_fluid_velocity_l1)
        invalid = int(self.thermal_state_invalid[None]) != 0
        if (
            not invalid
            and math.isfinite(maximum_fluid)
            and maximum_fluid <= _MAX_D2Q9_LATTICE_VELOCITY_L1
        ):
            return

        velocity = self.physical_velocity_lattice.to_numpy()
        phase = self.water_phase.to_numpy()
        populations = self.phase_populations.to_numpy()
        active = (self.wall_mask.to_numpy() == 0) & (self.solid_mask.to_numpy() == 0)
        speed_l1 = np.abs(velocity[..., 0]) + np.abs(velocity[..., 1])
        invalid_velocity = ~np.isfinite(speed_l1)
        invalid_phase = ~np.isfinite(phase)
        invalid_populations = ~np.all(np.isfinite(populations), axis=-1)
        invalid_state = active & (
            invalid_velocity | invalid_phase | invalid_populations
        )
        step_number = int(target_step)
        if np.any(invalid_state):
            flat_index = int(np.flatnonzero(invalid_state)[0])
            i, j = np.unravel_index(flat_index, invalid_state.shape)
            invalid_fields = []
            if invalid_velocity[i, j]:
                invalid_fields.append("velocity")
            if invalid_phase[i, j]:
                invalid_fields.append("phase")
            if invalid_populations[i, j]:
                invalid_fields.append("phase populations")
            raise RuntimeError(
                "LBM state became non-finite before thermal coupling at step "
                f"{step_number}: invalid_fields={','.join(invalid_fields)}, "
                f"cell=({i},{j}), cell_speed={float(speed_l1[i, j]):.9g}, "
                f"phi={float(phase[i, j]):.9g}. This is a hydrodynamic "
                "instability, not a thermal workload; rescale the lattice "
                "units and check the Mach number, relaxation times, and "
                "forcing."
            )
        ranking = np.where(active & np.isfinite(speed_l1), speed_l1, -np.inf)
        flat_index = int(np.argmax(ranking))
        i, j = np.unravel_index(flat_index, ranking.shape)
        local_speed = float(speed_l1[i, j])
        local_phase = float(phase[i, j])
        raise RuntimeError(
            "LBM velocity left the D2Q9 stability envelope before thermal "
            f"coupling at step {step_number}: full-domain max "
            f"|u_x|+|u_y|={maximum_fluid:.9g} (cell=({i},{j}), "
            f"cell_speed={local_speed:.9g}, phi={local_phase:.9g}), "
            f"water-side max={maximum_water:.9g}, permitted full-domain "
            f"maximum={_MAX_D2Q9_LATTICE_VELOCITY_L1:.9g}. This is a "
            "hydrodynamic instability, not a thermal workload; rescale the "
            "lattice units and check the Mach number, relaxation times, "
            "forcing, and flow discretization."
        )

    @ti.func
    def _body_velocity_at(self, point, b=0):
        relative = point - self.body_center[b]
        omega = self.body_angular_velocity[b]
        return self.body_velocity[b] + omega * ti.Vector([-relative.y, relative.x])

    @ti.func
    def _cell_point(self, i, j):
        return ti.Vector([ti.cast(i, ti.f32) + 0.5, ti.cast(j, ti.f32) + 0.5])

    @ti.func
    def _phase_sample_inside_walls(self, i, j):
        """Mirror a phase stencil across the container's half-cell walls.

        Zero normal derivative keeps the tangential variation. Replacing a
        blocked diagonal with the centre value would erase that variation
        and manufacture capillary stress at an otherwise flat interface.
        """
        lower = ti.static(self.config.boundary_cells)
        upper_x = ti.static(self.nx - self.config.boundary_cells)
        upper_y = ti.static(self.ny - self.config.boundary_cells)
        if i < lower:
            i = 2 * lower - 1 - i
        elif i >= upper_x:
            i = 2 * upper_x - 1 - i
        if j < lower:
            j = 2 * lower - 1 - j
        elif j >= upper_y:
            j = 2 * upper_y - 1 - j
        return ti.Vector([i, j])

    @ti.func
    def _ice_mirror_sample(self, fi, fj, si, sj, sample_i, sample_j):
        """Reflect a stencil point across the local cut-link ice surface.

        A neutral wetting boundary removes only the normal derivative. In
        particular, a blocked diagonal must retain its tangential offset.
        The same geometry is used for phase and capillary-stress ghosts.
        """
        b = ti.max(self.solid_body_id[si, sj], 0)
        gradient = ti.Vector.zero(ti.f32, 2)
        for endpoint in ti.static(range(2)):
            i, j = fi, fj
            if ti.static(endpoint == 1):
                i, j = si, sj
            gradient.x += (
                self._body_fractions[b, ti.min(i + 1, self.nx - 1), j]
                - self._body_fractions[b, ti.max(i - 1, 0), j]
            )
            gradient.y += (
                self._body_fractions[b, i, ti.min(j + 1, self.ny - 1)]
                - self._body_fractions[b, i, ti.max(j - 1, 0)]
            )
        direction = ti.Vector([ti.cast(si - fi, ti.f32), ti.cast(sj - fj, ti.f32)])
        normal = gradient / (gradient.norm() + 1.0e-14)
        if gradient.norm() < 1.0e-10:
            normal = direction / (direction.norm() + 1.0e-14)
        crossing = ti.Vector([ti.cast(fi, ti.f32), ti.cast(fj, ti.f32)])
        crossing += self._cut_link_fraction(fi, fj, si, sj) * direction
        point = ti.Vector([ti.cast(sample_i, ti.f32), ti.cast(sample_j, ti.f32)])
        reflected = point - 2.0 * (point - crossing).dot(normal) * normal
        return ti.Vector([reflected.x, reflected.y, normal.x, normal.y])

    @ti.func
    def _ice_phase_ghost(self, fi, fj, si, sj, sample_i, sample_j):
        mirror = self._ice_mirror_sample(fi, fj, si, sj, sample_i, sample_j)
        base = ti.cast(ti.floor(ti.Vector([mirror.x, mirror.y])), ti.i32)
        center = ti.cast(self.water_phase[fi, fj], ti.f32)
        difference, weight = 0.0, 0.0
        for di, dj in ti.static(ti.ndrange(2, 2)):
            i, j = base.x + di, base.y + dj
            w = ti.max(0.0, 1.0 - ti.abs(mirror.x - i))
            w *= ti.max(0.0, 1.0 - ti.abs(mirror.y - j))
            index = self._phase_sample_inside_walls(i, j)
            if inside_grid(index.x, index.y, self.nx, self.ny):
                if self._is_fluid_cell(index.x, index.y):
                    # Interpolate differences to preserve pure phases exactly.
                    difference += w * (ti.cast(self.water_phase[index.x, index.y], ti.f32) - center)
                    weight += w
        value = center
        if weight > 1.0e-10:
            value += difference / weight
        return value

    @ti.func
    def _phase_neighbor(self, i, j, step_x, step_y, distance):
        value = ti.cast(self.water_phase[i, j], ti.f32)
        first = self._phase_sample_inside_walls(i + step_x, j + step_y)
        ni1, nj1 = first.x, first.y
        if inside_grid(ni1, nj1, self.nx, self.ny):
            if self._is_fluid_cell(ni1, nj1):
                value = ti.cast(self.water_phase[ni1, nj1], ti.f32)
                if distance == 2:
                    second = self._phase_sample_inside_walls(
                        i + 2 * step_x, j + 2 * step_y
                    )
                    ni2, nj2 = second.x, second.y
                    if inside_grid(ni2, nj2, self.nx, self.ny):
                        if self._is_fluid_cell(ni2, nj2):
                            value = ti.cast(self.water_phase[ni2, nj2], ti.f32)
                        elif self.solid_mask[ni2, nj2] != 0:
                            value = self._ice_phase_ghost(ni1, nj1, ni2, nj2, ni2, nj2)
            elif self.solid_mask[ni1, nj1] != 0:
                target = self._phase_sample_inside_walls(
                    i + distance * step_x, j + distance * step_y
                )
                value = self._ice_phase_ghost(i, j, ni1, nj1, target.x, target.y)
        return value

    # ------------------------------------------------------------------
    # Moving-body world rasterization

    @ti.func
    def _body_local_coordinates(self, point, b=0):
        """Transform a world cell centre into material-frame coordinates."""

        center = self.body_reference_origin[b]
        angle = self.body_angle[b]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        relative = point - center
        return ti.Vector(
            [
                cosine * relative.x + sine * relative.y,
                -sine * relative.x + cosine * relative.y,
            ]
        )

    @ti.func
    def _nearest_body_index(self, local, b=0):
        body_i = ti.cast(
            ti.floor(
                local.x
                + (
                    0.5 * self.thermal.material_size[b].x
                    + self.thermal.material_offset[b]
                )
            ),
            ti.i32,
        )
        body_j = ti.cast(
            ti.floor(local.y + (0.5 * self.thermal.material_size[b].y)), ti.i32
        )
        return ti.Vector([body_i, body_j])

    @ti.func
    def _sample_body_fraction(self, local, b=0):
        """Bilinearly sample the eroded material grid without wall clipping."""

        grid_x = (
            local.x
            + 0.5 * self.thermal.material_size[b].x
            - 0.5
            + self.thermal.material_offset[b]
        )
        grid_y = local.y + 0.5 * self.thermal.material_size[b].y - 0.5
        base_i = ti.cast(ti.floor(grid_x), ti.i32)
        base_j = ti.cast(ti.floor(grid_y), ti.i32)
        fraction = ti.cast(0.0, ti.f32)
        for di, dj in ti.static(ti.ndrange(2, 2)):
            body_i = base_i + di
            body_j = base_j + dj
            if (
                self.thermal.material_offset[b]
                <= body_i
                < self.thermal.material_offset[b] + self.thermal.material_size[b].x
                and 0 <= body_j < self.thermal.material_size[b].y
            ):
                weight_x = 1.0 - ti.abs(grid_x - ti.cast(body_i, ti.f32))
                weight_y = 1.0 - ti.abs(grid_y - ti.cast(body_j, ti.f32))
                fraction += (
                    ti.max(0.0, weight_x)
                    * ti.max(0.0, weight_y)
                    * ti.cast(self.thermal.body_solid_fraction[body_i, body_j], ti.f32)
                )
        return ti.min(1.0, ti.max(0.0, fraction))

    @ti.kernel
    def _calculate_moving_body_fraction(self):
        """Keep per-body coverage for links/heat and its union for fluid masking."""
        for i, j in self._moving_body_fraction:
            coverage = ti.cast(0.0, ti.f32)
            for b in range(self.body_count):
                local = self._body_local_coordinates(self._cell_point(i, j), b)
                fraction = self._sample_body_fraction(local, b)
                self._body_fractions[b, i, j] = fraction
                coverage = ti.max(coverage, fraction)
            self._moving_body_fraction[i, j] = coverage

    @ti.kernel
    def _commit_moving_body_raster(self):
        """Clip the accepted coverage in place; contact has finished reading it."""
        for i, j in self._moving_body_fraction:
            if ti.static(self.config.close_subgrid_wall_gaps):
                self.previous_subgrid_wall_gap[i, j] = self.subgrid_wall_gap[i, j]
                self.subgrid_wall_gap[i, j] = ti.cast(0, ti.i8)
            if self.wall_mask[i, j] != 0:
                self._moving_body_fraction[i, j] = 0.0
                for b in range(self.body_count):
                    self._body_fractions[b, i, j] = 0.0
            elif ti.static(self.config.close_subgrid_wall_gaps):
                # Wall gaps narrower than the configured resolution are
                # below the diffuse interface width. Close their aperture;
                # conservative geometry remapping transfers all displaced
                # liquid/heat. Material ice mass is never changed here.
                # Probe analytic material coverage to avoid in-place races.
                radius = ti.static(self.config.subgrid_wall_gap_cells)
                lower = ti.static(self.config.boundary_cells)
                distances = ti.Vector([i-lower+1, j-lower+1, self.nx-lower-i, self.ny-lower-j])
                near_wall = False
                for q in ti.static(range(4)):
                    near_wall = near_wall or 1 <= distances[q] <= radius
                if near_wall:
                    for b in range(self.body_count):
                        fraction = self._body_fractions[b, i, j]
                        if fraction < self.config.thermal.solid_liquid_threshold:
                            close_gap = False
                            for q in ti.static(range(1, 5)):
                                distance = distances[q-1]
                                if 1 <= distance <= radius:
                                    d = lattice_direction(q)
                                    for steps in ti.static(range(1, radius+1)):
                                        if steps <= radius+1-distance and not close_gap:
                                            neighbor = self._cell_point(i+steps*d.x, j+steps*d.y)
                                            covered = self._sample_body_fraction(self._body_local_coordinates(neighbor, b), b)
                                            close_gap = close_gap or covered >= self.config.thermal.solid_liquid_threshold
                            if close_gap:
                                self.subgrid_wall_gap[i, j] = ti.cast(1, ti.i8)
                                effective = self.config.thermal.solid_liquid_threshold + 1e-4
                                self._body_fractions[b, i, j] = effective
                                self._moving_body_fraction[i, j] = ti.max(self._moving_body_fraction[i, j], effective)

    def _initialize_body(self):
        specs = self.config.bodies
        self.body_velocity.from_numpy(
            np.asarray([b.velocity_lattice for b in specs], dtype=np.float32)
        )
        self.body_angle.from_numpy(
            np.asarray([b.angle_rad for b in specs], dtype=np.float32)
        )
        self.body_angular_velocity.from_numpy(
            np.asarray([b.angular_velocity_lattice for b in specs], dtype=np.float32)
        )
        self.body_mass_lattice.from_numpy(self._initial_masses.to_numpy())
        self.body_inertia_lattice.from_numpy(self._initial_inertias.to_numpy())
        self.body_local_center_of_mass.fill(0.0)
        self.body_reference_origin.from_numpy(
            np.asarray([b.center_cells for b in specs], dtype=np.float32)
        )
        self.hydrodynamic_impulse.fill(0.0)
        self.hydrodynamic_torque.fill(0.0)
        self._body_contact_projection_changed.fill(0)

    @ti.kernel
    def _initialize_geometry(self):
        # The static wall mask is computed directly from the container bounds.
        for i, j in self.solid_mask:
            self.solid_mask[i, j] = ti.cast(0, ti.i8)
            self.previous_solid_mask[i, j] = ti.cast(0, ti.i8)

    @ti.kernel
    def _initialize_fluid(self):
        water_width = ti.static(self.config.water_width)
        water_height = ti.static(self.config.water_height)
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            water_indicator = 1.0 if i < water_width and j < water_height else 0.0
            active = self._is_fluid_cell(i, j)
            phi0 = water_indicator if active else 0.0
            velocity = ti.Vector([0.0, 0.0])
            if self.solid_mask[i, j] == 1:
                velocity = self._body_velocity_at(
                    self._cell_point(i, j), ti.max(self.solid_body_id[i, j], 0)
                )
            # In moving-body runs, inactive solid cells retain the covered
            # phase directly in phi until that cell is exposed again.
            reservoir_phase = phi0
            if self.solid_mask[i, j] == 1:
                reservoir_phase = water_indicator
            self.water_phase[i, j] = reservoir_phase
            self.momentum_velocity_lattice[i, j] = velocity
            self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])
            self.pressure_lattice[i, j] = 0.0
            material_density = mixture_density(phi0, rho_water, rho_air)
            for q in range(D2Q9_DIRECTION_COUNT):
                self.momentum_populations[i, j, q] = momentum_equilibrium(
                    q, 0.0, material_density, velocity
                )
                self.phase_populations[i, j, q] = (
                    phase_equilibrium(q, phi0, velocity) if active else 0.0
                )

    def _build_hydrostatic_reference(self):
        """Freeze row density and its hydrostatic prefix integral at initialization."""
        phase = np.clip(self.water_phase.to_numpy().astype(np.float64), 0.0, 1.0)
        active = (self.wall_mask.to_numpy() == 0) & (self.solid_mask.to_numpy() == 0)
        row_count = np.count_nonzero(active, axis=0)
        phase_profile = (np.arange(self.ny) < self.config.water_height).astype(
            np.float64
        )
        np.divide(
            np.where(active, phase, 0.0).sum(axis=0),
            row_count,
            out=phase_profile,
            where=row_count > 0,
        )
        if self.config.initial_water_geometry == "dam_break":
            # A row-averaged water column would impose a water-sized reference
            # force on dry air. Use the ambient-air hydrostatic gauge instead;
            # the water excess pressure is initialized independently below.
            phase_profile.fill(0.0)
        density = (
            self._air_density_lattice
            + (self._water_density_lattice - self._air_density_lattice) * phase_profile
        )
        pressure = np.zeros(self.ny, dtype=np.float64)
        surface = self.config.water_height
        gravity_y = self._gravity_lattice[1]
        pressure[surface - 1] = -0.5 * gravity_y * density[surface - 1]
        for row in range(surface - 2, -1, -1):
            pressure[row] = pressure[row + 1] - 0.5 * gravity_y * (
                density[row] + density[row + 1]
            )
        pressure[surface] = 0.5 * gravity_y * density[surface]
        for row in range(surface + 1, self.ny):
            pressure[row] = pressure[row - 1] + 0.5 * gravity_y * (
                density[row - 1] + density[row]
            )
        if not (np.isfinite(density).all() and np.isfinite(pressure).all()):
            raise RuntimeError(
                "hydrostatic reference construction produced non-finite values"
            )
        self.reference_density_by_row.from_numpy(density.astype(np.float32))
        self.reference_pressure_by_row.from_numpy(pressure.astype(np.float32))

    def _initialize_dam_break_pressure(self):
        """Integrate the actual initial column density above each cell centre.

        p = p_air + p_excess, with d(p_excess)/dy = (rho-rho_air)*g_y.
        Each column uses its warmed-up phase, so the vertical free face retains
        the pressure jump released by removing the gate. No horizontal-pool
        equilibrium is imposed on the downstream air.
        """
        phase = np.clip(self.water_phase.to_numpy().astype(np.float64), 0.0, 1.0)
        excess = (self._water_density_lattice - self._air_density_lattice) * phase
        dynamic = -self._gravity_lattice[1] * (
            np.cumsum(excess[:, ::-1], axis=1)[:, ::-1] - 0.5 * excess
        )
        reference = self.reference_pressure_by_row.to_numpy()[None, :]
        self.pressure_lattice.from_numpy((reference + dynamic).astype(np.float32))
        self._initialize_dam_break_populations()

    @ti.kernel
    def _initialize_dam_break_populations(self):
        rest = ti.Vector([0.0, 0.0])
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                density = mixture_density(
                    ti.cast(self.water_phase[i, j], ti.f32),
                    ti.static(self._water_density_lattice),
                    ti.static(self._air_density_lattice),
                )
                dynamic = self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
                self.momentum_velocity_lattice[i, j] = rest
                self.fluid_acceleration_lattice[i, j] = rest
                for q in ti.static(range(9)):
                    self.momentum_populations[i, j, q] = momentum_equilibrium(
                        q, dynamic, density, rest
                    )

    @ti.kernel
    def _initialize_hydrostatic_equilibrium(self):
        """Set the two-phase fluid to the frozen-reference rest state."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        rest = ti.Vector([0.0, 0.0])
        for i, j in self.water_phase:
            point = self._cell_point(i, j)
            hydrostatic_pressure = self.hydrostatic_reference_pressure[i, j]
            # In hydrostatic mode this reservoir stores only the dynamic
            # pressure residual.  The spatial reference is always added back
            # at the newly exposed cell, so rigid motion cannot advect a
            # hydrostatic gauge value with the body.
            if self._is_fluid_cell(i, j):
                water_phase = ti.cast(self.water_phase[i, j], ti.f32)
                material_density = mixture_density(water_phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] = rest
                self.pressure_lattice[i, j] = hydrostatic_pressure
                for q in range(D2Q9_DIRECTION_COUNT):
                    # Only p_dyn is carried by the populations.  The frozen
                    # p_H traction remains in the Guo completion.
                    self.momentum_populations[i, j, q] = momentum_equilibrium(
                        q, 0.0, material_density, rest
                    )
                    if ti.static(not self._conservative_phase_enabled):
                        self.phase_populations[i, j, q] = phase_equilibrium(q, water_phase, rest)
            else:
                boundary_velocity = ti.Vector([0.0, 0.0])
                if self.solid_mask[i, j] == 1:
                    boundary_velocity = self._body_velocity_at(point)
                self.momentum_velocity_lattice[i, j] = boundary_velocity
                self.pressure_lattice[i, j] = self.hydrostatic_reference_pressure[i, j]
            if ti.static(self._conservative_phase_enabled):
                phi = self.water_phase[i, j]
                capacity = self.conservative_phase.capacity[i, j]
                for q in ti.static(range(9)):
                    self.phase_populations[i, j, q] = capacity * phase_equilibrium(q, phi, rest)
                self.conservative_phase.close_moment(i, j, capacity * phi)
                self.water_phase[i, j] = capacity * phi

    def _initialize_moving_body_geometry(self):
        self._update_solid_mask()
        self.previous_solid_mask.copy_from(self.solid_mask)
        self.previous_body_id.copy_from(self.solid_body_id)

    @ti.kernel
    def _update_solid_mask(self):
        for b in range(self.body_count):
            self._body_has_resolved_boundary[b] = 0
        for i, j in self.solid_mask:
            self.previous_solid_mask[i, j] = self.solid_mask[i, j]
            self.previous_body_id[i, j] = self.solid_body_id[i, j]
            owner = -1
            if self.wall_mask[i, j] == 0:
                for b in range(self.body_count):
                    if self.body_active[b] != 0 and self._body_fractions[
                        b, i, j
                    ] >= ti.static(self.config.thermal.solid_liquid_threshold):
                        owner = b
                        ti.atomic_max(self._body_has_resolved_boundary[b], 1)
            self.solid_body_id[i, j] = owner
            self.solid_mask[i, j] = ti.cast(owner >= 0, ti.i8)

    def _update_moving_body_mass_properties(self, *, initialize: bool = False):
        """Reduce the eroding material grid and update the rigid COM state."""

        self._reduce_moving_body_mass_properties()
        self._apply_moving_body_mass_properties(1 if initialize else 0)

    @ti.kernel
    def _reduce_moving_body_mass_properties(self):
        for b in range(self.body_count):
            self._body_reduced_mass[b] = 0.0
            self._body_reduced_first_moment[b] = ti.Vector([0.0, 0.0])
            self._body_reduced_inertia_origin[b] = 0.0
        for i, j in self.thermal.body_solid_mass:
            b = self.thermal.material_body_id[i, j]
            if b >= 0:
                mass = self.thermal.body_solid_mass[i, j] / ti.cast(
                    ti.static(self.config.rho_water * self.config.dx**2), ti.f64
                )
                if ti.static(self._conservative_phase_enabled):
                    mass = self.thermal.body_solid_mass[i, j] / self.conservative_phase.water_cell_mass[None]
                fraction = ti.cast(self.thermal.body_solid_fraction[i, j], ti.f64)
                local = ti.Vector(
                    [
                        ti.cast(i - self.thermal.material_offset[b], ti.f64)
                        + 0.5
                        - 0.5 * self.thermal.material_size[b].x,
                        ti.cast(j, ti.f64)
                        + 0.5
                        - 0.5 * self.thermal.material_size[b].y,
                    ]
                )
                ti.atomic_add(self._body_reduced_mass[b], mass)
                ti.atomic_add(self._body_reduced_first_moment[b], mass * local)
                ti.atomic_add(
                    self._body_reduced_inertia_origin[b],
                    mass * (local.dot(local) + fraction / 6.0),
                )

    @ti.kernel
    def _apply_moving_body_mass_properties(self, initialize: ti.i32):
        for b in range(self.body_count):
            mass = self._body_reduced_mass[b]
            old_com = self.body_local_center_of_mass[b]
            new_com = old_com
            inertia = ti.cast(0.0, ti.f64)
            if mass > 1.0e-12:
                new_com = self._body_reduced_first_moment[b] / mass
                inertia = ti.max(
                    0.0,
                    self._body_reduced_inertia_origin[b] - mass * new_com.dot(new_com),
                )
            active = inertia > 1.0e-14 * ti.max(self._initial_inertias[b], 1.0)
            if initialize == 0 and active:
                # The surviving material keeps its rigid velocity at the new COM.
                shift = new_com - old_com
                cosine, sine = ti.cos(self.body_angle[b]), ti.sin(self.body_angle[b])
                world_shift = ti.Vector(
                    [
                        cosine * shift.x - sine * shift.y,
                        sine * shift.x + cosine * shift.y,
                    ]
                )
                velocity = ti.cast(self.body_velocity[b], ti.f64)
                velocity += self.body_angular_velocity[b] * ti.Vector(
                    [-world_shift.y, world_shift.x]
                )
                self.body_velocity[b] = ti.cast(velocity, ti.f32)
            self.body_mass_lattice[b] = mass
            self.body_local_center_of_mass[b] = new_com
            self.body_inertia_lattice[b] = inertia if active else 0.0
            if not active:
                self.body_velocity[b] = ti.Vector([0.0, 0.0])
                self.body_angular_velocity[b] = 0.0
                for index in ti.static(range(4)):
                    self._body_contact_support_extrema[b, index] = 0.0

    @ti.func
    def _extrapolate_exposed_fluid(self, i, j):
        """Read only persistent fluid, never other cells being filled by this launch."""
        state = ti.Vector(
            [
                ti.min(1.0, ti.max(0.0, ti.cast(self.water_phase[i, j], ti.f32))),
                self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j],
                self._body_velocity_at(
                    self._cell_point(i, j), ti.max(self.previous_body_id[i, j], 0)
                ).x,
                self._body_velocity_at(
                    self._cell_point(i, j), ti.max(self.previous_body_id[i, j], 0)
                ).y,
            ]
        )
        radius = 1
        count = 0.0
        # Usually the first ring suffices. A complete erosion update can open
        # several layers at once; search further instead of seeding an air pocket.
        while count == 0.0 and radius < ti.static(max(self.nx, self.ny)):
            total = ti.Vector.zero(ti.f32, 4)
            for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                ni, nj = i + di, j + dj
                if ti.max(ti.abs(di), ti.abs(dj)) == radius and inside_grid(
                    ni, nj, self.nx, self.ny
                ):
                    if (
                        self._is_fluid_cell(ni, nj)
                        and self.previous_solid_mask[ni, nj] == 0
                    ):
                        velocity = self.momentum_velocity_lattice[ni, nj]
                        total += ti.Vector(
                            [
                                ti.min(1.0, ti.max(0.0, ti.cast(self.water_phase[ni, nj], ti.f32))),
                                self.pressure_lattice[ni, nj]
                                - self.hydrostatic_reference_pressure[ni, nj],
                                velocity.x,
                                velocity.y,
                            ]
                        )
                        count += 1.0
            if count > 0.0:
                state = total / count
            radius += 1
        return state

    @ti.kernel
    def _refill_changed_nodes(self):
        """Extend ambient fluid into cells exposed by either motion or erosion.

        This is a boundary reconstruction, independent of removed ice mass.
        In particular, erosion underwater must not create an air/water interface.
        Persistent fluid cells are untouched; pressure always stores total p.
        """
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            if (
                self.wall_mask[i, j] == 0
                and self.previous_body_id[i, j] != self.solid_body_id[i, j]
            ):
                old_volume = ti.cast(0.0, ti.f64)
                old_energy = ti.cast(0.0, ti.f64)
                if ti.static(self._conservative_phase_enabled):
                    if self.previous_solid_mask[i, j] == 0:
                        old_volume = self.conservative_phase.volume(i, j)
                        old_energy = self.thermal.water_sensible_energy[i, j]
                new_volume = ti.cast(0.0, ti.f64)
                if self.solid_mask[i, j] == 1:
                    self.momentum_velocity_lattice[i, j] = self._body_velocity_at(
                        self._cell_point(i, j), self.solid_body_id[i, j]
                    )
                    if ti.static(self._conservative_phase_enabled):
                        self.water_phase[i, j] = 0.0
                        for q in ti.static(range(9)):
                            self.phase_populations[i, j, q] = 0.0
                else:
                    state = self._extrapolate_exposed_fluid(i, j)
                    phase, dynamic_pressure = state[0], state[1]
                    velocity = ti.Vector([state[2], state[3]])
                    density = mixture_density(phase, rho_water, rho_air)
                    self.water_phase[i, j] = phase
                    new_volume = ti.cast(phase, ti.f64)
                    if ti.static(self._conservative_phase_enabled):
                        self.conservative_phase.momentum_phase[i, j] = phase
                    self.momentum_velocity_lattice[i, j] = velocity
                    self.pressure_lattice[i, j] = (
                        dynamic_pressure + self.hydrostatic_reference_pressure[i, j]
                    )
                    stress = ti.Vector.zero(ti.f32, 3)
                    if ti.static(self.config.exposed_node_refill == "non_equilibrium"):
                        stress = self._extrapolate_exposed_stress(i, j, density)
                    for q in range(D2Q9_DIRECTION_COUNT):
                        self.momentum_populations[i, j, q] = momentum_equilibrium(
                            q, dynamic_pressure, density, velocity
                        )
                        if ti.static(
                            self.config.exposed_node_refill == "non_equilibrium"
                        ):
                            self.momentum_populations[i, j, q] += (
                                pressure_preserving_stress_population(
                                    q, stress, velocity
                                )
                            )
                        self.phase_populations[i, j, q] = phase_equilibrium(q, phase, velocity)
                if ti.static(self._conservative_phase_enabled):
                    self.conservative_phase.close_moment(i, j, new_volume)
                    self.conservative_phase.event_volume[i, j] += old_volume - new_volume
                    self.conservative_phase.event_heat[i, j] += old_energy
                    self.thermal.water_sensible_energy[i, j] = 0.0
                self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])

    # ------------------------------------------------------------------
    # Two-phase pressure--momentum LBM

    @ti.kernel
    def _prepare_capillary_stress(self):
        """Build the tangential diffuse-interface stress from the current phase.

        K = kappa (|grad(phi)|^2 I - grad(phi) grad(phi)^T).
        Its interface integral is sigma for the equilibrium tanh profile.
        This is the chemical-potential formulation with the isotropic term
        kappa*|grad(phi)|^2/2 - beta*phi^2*(1-phi)^2 absorbed into pressure.
        In particular, an axis-aligned flat interface has no normal stress
        or force, even when its discrete profile is not an exact tanh.
        """
        for i, j in self.capillary_stress_lattice:
            stress = ti.Vector.zero(ti.f32, 3)
            if ti.static(self._surface_tension_lattice > 0.0):
                if self._is_fluid_cell(i, j):
                    phi = ti.cast(self.water_phase[i, j], ti.f32)
                    gradient = ti.Vector.zero(ti.f32, 2)
                    for q in range(1, D2Q9_DIRECTION_COUNT):
                        direction = lattice_direction(q)
                        phi1 = self._phase_neighbor(i, j, direction.x, direction.y, 1)
                        phi2 = self._phase_neighbor(i, j, direction.x, direction.y, 2)
                        gradient += (
                            lattice_weight(q)
                            * ti.cast(direction, ti.f32)
                            * (
                                2.0 * (phi1 - phi)
                                + 0.5 * (4.0 * phi1 - phi2 - 3.0 * phi)
                            )
                        )
                    kappa = ti.static(
                        1.5
                        * self._surface_tension_lattice
                        * self.config.interface_width
                    )
                    stress = kappa * ti.Vector(
                        [gradient.y**2, -gradient.x * gradient.y, gradient.x**2]
                    )
            self.capillary_stress_lattice[i, j] = stress

    @ti.func
    def _phase_central_gradient(self, i, j):
        gradient = ti.Vector.zero(ti.f32, 2)
        phi = ti.cast(self.water_phase[i, j], ti.f32)
        for q in range(1, D2Q9_DIRECTION_COUNT):
            direction = lattice_direction(q)
            neighbor = self._phase_neighbor(i, j, direction.x, direction.y, 1)
            gradient += (
                3.0 * lattice_weight(q) * ti.cast(direction, ti.f32) * (neighbor - phi)
            )
        return gradient

    @ti.func
    def _momentum_phase_gradient(self, i, j):
        gradient = ti.Vector.zero(ti.f32, 2)
        center = ti.cast(self.water_phase[i, j], ti.f32)
        if ti.static(self.config.momentum_phase_gradient == "open_links_air"):
            if center < 0.05:
                for q in ti.static(range(1, 9)):
                    d = lattice_direction(q)
                    ni, nj = i+d.x, j+d.y
                    if inside_grid(ni, nj, self.nx, self.ny) and self._is_fluid_cell(ni, nj):
                        gradient += 3.0*lattice_weight(q)*ti.cast(d, ti.f32)*(ti.cast(self.water_phase[ni, nj], ti.f32)-center)
            else:
                gradient = self._phase_central_gradient(i, j)
        else:
            gradient = self._phase_central_gradient(i, j)
        if ti.static(self.config.phase_momentum_water_cutoff > 0):
            if center < self.config.phase_momentum_water_cutoff:
                gradient = ti.Vector.zero(ti.f32, 2)
        return gradient

    @ti.func
    def _capillary_traction(self, stress, direction):
        return ti.Vector(
            [
                stress[0] * direction.x + stress[1] * direction.y,
                stress[1] * direction.x + stress[2] * direction.y,
            ]
        )

    @ti.func
    def _ice_capillary_stress(self, i, j, ni, nj):
        """Neutral-wetting tensor extension, K_ghost = R K_mirror R^T."""
        mirror = self._ice_mirror_sample(i, j, ni, nj, ni, nj)
        base = ti.cast(ti.floor(ti.Vector([mirror.x, mirror.y])), ti.i32)
        stress = ti.Vector.zero(ti.f32, 3)
        weight = 0.0
        for di, dj in ti.static(ti.ndrange(2, 2)):
            x, y = base.x + di, base.y + dj
            w = ti.max(0.0, 1.0 - ti.abs(mirror.x - x))
            w *= ti.max(0.0, 1.0 - ti.abs(mirror.y - y))
            index = self._phase_sample_inside_walls(x, y)
            if inside_grid(index.x, index.y, self.nx, self.ny):
                if self._is_fluid_cell(index.x, index.y):
                    stress += w * self.capillary_stress_lattice[index.x, index.y]
                    weight += w
        if weight > 1.0e-10:
            stress /= weight
        else:
            stress = self.capillary_stress_lattice[i, j]
        rxx = 1.0 - 2.0 * mirror.z**2
        ryy = 1.0 - 2.0 * mirror.w**2
        rxy = -2.0 * mirror.z * mirror.w
        return ti.Vector(
            [
                rxx**2 * stress[0] + 2.0 * rxx * rxy * stress[1] + rxy**2 * stress[2],
                rxx * rxy * stress[0]
                + (rxx * ryy + rxy**2) * stress[1]
                + rxy * ryy * stress[2],
                rxy**2 * stress[0] + 2.0 * rxy * ryy * stress[1] + ryy**2 * stress[2],
            ]
        )

    @ti.func
    def _capillary_force_density(self, i, j):
        """Divergence of shared link tractions, in lattice units.

        On a fluid-fluid link the averaged stress gives exactly opposite
        forces to its two cells. On a blocked link the fluid-side stress
        supplies the boundary traction; its reaction goes to ice (below)
        or to the fixed container. No global force subtraction is applied.
        """
        force = ti.Vector.zero(ti.f32, 2)
        if ti.static(self._surface_tension_lattice > 0.0):
            local_stress = self.capillary_stress_lattice[i, j]
            for q in range(1, D2Q9_DIRECTION_COUNT):
                direction = lattice_direction(q)
                ni, nj = i + direction.x, j + direction.y
                neighbor_stress = local_stress
                if inside_grid(ni, nj, self.nx, self.ny) and self._is_fluid_cell(
                    ni, nj
                ):
                    neighbor_stress = self.capillary_stress_lattice[ni, nj]
                elif (
                    inside_grid(ni, nj, self.nx, self.ny)
                    and self.wall_mask[ni, nj] != 0
                ):
                    # Reflect the tensor as well as phi at a container wall.
                    # A blocked diagonal still samples a different tangential
                    # location; replacing it by local_stress creates a false
                    # normal force even for a horizontal planar interface.
                    reflected = self._phase_sample_inside_walls(ni, nj)
                    neighbor_stress = self.capillary_stress_lattice[
                        reflected.x, reflected.y
                    ]
                    if (reflected.x != ni) != (reflected.y != nj):
                        neighbor_stress[1] = -neighbor_stress[1]
                elif (
                    inside_grid(ni, nj, self.nx, self.ny)
                    and self.solid_mask[ni, nj] != 0
                ):
                    neighbor_stress = self._ice_capillary_stress(i, j, ni, nj)
                force += (
                    3.0
                    * lattice_weight(q)
                    * self._capillary_traction(
                        local_stress + neighbor_stress, direction
                    )
                )
        return force

    @ti.func
    def _fluid_kinematic_viscosity_lattice(self, i, j):
        """Molecular water viscosity and the explicitly configured gas floor."""

        phi = ti.min(1.0, ti.max(0.0, ti.cast(self.water_phase[i, j], ti.f32)))
        nu_water = ti.cast(ti.static(self._water_viscosity_lattice), ti.f32)
        if ti.static(self._temperature_dependent_water_viscosity):
            # Dry diffuse tails have no transported water temperature. Retain
            # the initial water value there, rather than using the air temperature.
            temperature_resolved = phi > 0.0
            if ti.static(self._conservative_phase_enabled):
                # Mass in diffuse tails is retained, while thermophysical
                # reconstruction uses the same resolved wet region as heat.
                temperature_resolved = phi > ti.static(self.config.water_phase_cutoff)
            if temperature_resolved and self.thermal.water_volume_m2[i, j] > 1.0e-30:
                temperature = ti.cast(self.thermal._water_temperature_at(i, j), ti.f32)
                nu_water = water_dynamic_viscosity_pa_s(temperature) * ti.static(
                    self._viscosity_to_lattice / self.config.rho_water
                )
        nu_air = ti.static(max(self._air_viscosity_lattice, self.config.air_viscosity_floor_lattice))
        return phi * nu_water + (1.0 - phi) * nu_air

    @ti.func
    def _fluid_force_and_gradient(self, i, j):
        gravity = ti.Vector(
            [ti.static(self._gravity_lattice[0]), ti.static(self._gravity_lattice[1])]
        )
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        water_phase = ti.cast(self.water_phase[i, j], ti.f32)
        bounded_phi = ti.min(1.0, ti.max(0.0, water_phase))
        density = mixture_density(bounded_phi, rho_water, rho_air)

        source_gradient = self._momentum_phase_gradient(i, j)

        # Assemble gravity sources as force densities.  The frozen
        # hydrostatic reference removes only the base far-field load; the
        # temperature-dependent density anomaly must remain additive.
        reference_density = self.hydrostatic_reference_density[i, j]
        gravity_force_density = (density - reference_density) * gravity
        reference_temperature = ti.static(
            float(self.config.thermal.buoyancy_reference_temperature_c)
        )
        # The thermal field deliberately treats the water/air interface as
        # adiabatic and uses the air temperature for dry thermal cells. A smooth
        # water-side gate prevents that unrelated air value from forcing the
        # diffuse layer.
        water_weight = ti.min(1.0, ti.max(0.0, 2.0 * bounded_phi - 1.0))
        if water_weight > 0:
            temperature = ti.cast(self.thermal._water_temperature_at(i, j), ti.f32)
            expansion = ti.static(float(self.config.thermal.thermal_expansion_water_1_k))
            density_anomaly_ratio = -expansion * (temperature - reference_temperature)
            gravity_force_density += water_weight * rho_water * density_anomaly_ratio * gravity
        force = (gravity_force_density + self._capillary_force_density(i, j)) / density

        return ti.Vector(
            [
                force.x,
                force.y,
                source_gradient.x,
                source_gradient.y,
            ]
        )

    def _collide_momentum(self):
        self._prepare_capillary_stress()
        if (
            self.config.momentum_collision_model == "cm_hrr"
            or self.config.phase_momentum_coupling == "consistent_flux"
        ):
            self._cache_stability_velocity()
        if self.config.phase_momentum_coupling == "consistent_flux":
            # Explicit coupling: collide phase at the current physical velocity,
            # measure its actual time-centered mass flux, then apply the full
            # conservative momentum-flux divergence through the force operator.
            self._collide_phase()
            if self._conservative_phase_enabled:
                self.conservative_phase.prepare_transport()
            self._measure_phase_mass_flux()
        self._collide_momentum_kernel()

    @ti.kernel
    def _collide_momentum_kernel(self):
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                force_data = self._fluid_force_and_gradient(i, j)
                force = ti.Vector([force_data[0], force_data[1]])
                grad_phi = ti.Vector([force_data[2], force_data[3]])
                if ti.static(self.config.phase_momentum_coupling == "consistent_flux"):
                    rho_for_flux = mixture_density(
                        self.water_phase[i, j], rho_water, rho_air
                    )
                    force += self._phase_flux_force_density(i, j) / rho_for_flux
                self.fluid_acceleration_lattice[i, j] = force
                water_phase = ti.cast(self.water_phase[i, j], ti.f32)
                velocity = self.momentum_velocity_lattice[i, j] + 0.5 * force
                moment_frame_velocity = ti.Vector.zero(ti.f32, 2)
                if ti.static(self.config.momentum_collision_model != "rm_mrt"):
                    moment_frame_velocity = velocity
                nu_local = self._fluid_kinematic_viscosity_lattice(i, j)
                relaxation_time = 0.5 + 3.0 * nu_local
                # Liang et al. PRE 97, 033309 (2018), Eqs. (15)--(23).
                material_density = mixture_density(water_phase, rho_water, rho_air)
                dynamic_pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                force_density = material_density * force
                delta_rho = rho_water - rho_air
                moment_00 = 0.0
                moment_10 = 0.0
                moment_01 = 0.0
                moment_20 = 0.0
                moment_02 = 0.0
                moment_11 = 0.0
                moment_22 = 0.0
                equilibrium_moment_00 = 0.0
                equilibrium_moment_10 = 0.0
                equilibrium_moment_01 = 0.0
                equilibrium_moment_20 = 0.0
                equilibrium_moment_02 = 0.0
                equilibrium_moment_11 = 0.0
                equilibrium_moment_21 = 0.0
                equilibrium_moment_12 = 0.0
                equilibrium_moment_22 = 0.0
                force_moment_00 = 0.0
                force_moment_10 = 0.0
                force_moment_01 = 0.0
                force_moment_20 = 0.0
                force_moment_02 = 0.0
                force_moment_11 = 0.0
                force_moment_21 = 0.0
                force_moment_12 = 0.0
                force_moment_22 = 0.0
                pressure_moment = 0.0
                equilibrium_pressure_moment = 0.0
                force_pressure_moment = 0.0
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = ti.cast(lattice_direction(q), ti.f32)
                    cx = direction.x
                    cy = direction.y
                    if ti.static(self.config.momentum_collision_model != "rm_mrt"):
                        cx -= moment_frame_velocity.x
                        cy -= moment_frame_velocity.y
                    value = self.momentum_populations[i, j, q]
                    equilibrium = momentum_equilibrium(
                        q,
                        dynamic_pressure,
                        material_density,
                        velocity,
                    )
                    base_source = (
                        lattice_weight(q)
                        * 3.0
                        * (
                            direction.dot(force_density)
                            + delta_rho
                            * direction.dot(velocity)
                            * direction.dot(grad_phi)
                        )
                    )
                    if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                        # Pressure is carried by raw (22 - trace), not the
                        # Maxwellian zeroth moment. Relax this mode explicitly.
                        pressure_basis = (
                            direction.x** 2 * direction.y** 2 - direction.dot(direction)
                        )
                        pressure_moment += pressure_basis * value
                        equilibrium_pressure_moment += pressure_basis * equilibrium
                        force_pressure_moment += pressure_basis * base_source
                    moment_00 += value
                    moment_10 += cx * value
                    moment_01 += cy * value
                    moment_20 += cx * cx * value
                    moment_02 += cy * cy * value
                    moment_11 += cx * cy * value
                    if ti.static(
                        self.config.momentum_collision_model != "rm_mrt"
                        and self.config.cm_acoustic_relaxation_rate != 1.0
                    ):
                        moment_22 += cx * cx * cy * cy * value
                    equilibrium_moment_00 += equilibrium
                    equilibrium_moment_10 += cx * equilibrium
                    equilibrium_moment_01 += cy * equilibrium
                    equilibrium_moment_20 += cx * cx * equilibrium
                    equilibrium_moment_02 += cy * cy * equilibrium
                    equilibrium_moment_11 += cx * cy * equilibrium
                    equilibrium_moment_21 += cx * cx * cy * equilibrium
                    equilibrium_moment_12 += cx * cy * cy * equilibrium
                    equilibrium_moment_22 += cx * cx * cy * cy * equilibrium
                    force_moment_00 += base_source
                    force_moment_10 += cx * base_source
                    force_moment_01 += cy * base_source
                    force_moment_20 += cx * cx * base_source
                    force_moment_02 += cy * cy * base_source
                    force_moment_11 += cx * cy * base_source
                    force_moment_21 += cx * cx * cy * base_source
                    force_moment_12 += cx * cy * cy * base_source
                    force_moment_22 += cx * cx * cy * cy * base_source

                # Both frames project the same Liang equilibrium and source.
                # In particular, sum(g_eq)=0: standard mass-distribution
                # Maxwellian central moments would change this pressure model.
                # Deviatoric stress retains the molecular rate. CM trace/22 use
                # the acoustic rate; RM and the other ghost modes retain rate 1.
                # CM damps the zeroth moment
                # in one step, independently of viscosity. Its first moments
                # must use that same rate: the inverse shift mixes them with
                # moment_00, including the nonzero density-gradient source.
                # This preserves the raw momentum increment rho*force with the
                # half-force velocity. RM retains all its original rates.
                omega = 1.0 / relaxation_time
                source_prefactor = 1.0 - 0.5 * omega
                lower_omega = omega
                if ti.static(self.config.momentum_collision_model != "rm_mrt"):
                    lower_omega = 1.0
                lower_source_prefactor = 1.0 - 0.5 * lower_omega
                post_moment_00 = (
                    moment_00
                    - lower_omega * (moment_00 - equilibrium_moment_00)
                    + lower_source_prefactor * force_moment_00
                )
                post_moment_10 = (
                    moment_10
                    - lower_omega * (moment_10 - equilibrium_moment_10)
                    + lower_source_prefactor * force_moment_10
                )
                post_moment_01 = (
                    moment_01
                    - lower_omega * (moment_01 - equilibrium_moment_01)
                    + lower_source_prefactor * force_moment_01
                )
                trace_eq = equilibrium_moment_20 + equilibrium_moment_02
                trace_src = force_moment_20 + force_moment_02
                post_trace = trace_eq + 0.5 * trace_src
                if ti.static(
                    self.config.momentum_collision_model != "rm_mrt"
                    and self.config.cm_acoustic_relaxation_rate != 1.0
                ):
                    # At rest p = 3/5 * (raw_trace - raw_22). Relax these
                    # together: changing only trace alters the pressure mode.
                    acoustic_rate = ti.static(self.config.cm_acoustic_relaxation_rate)
                    trace = moment_20 + moment_02
                    post_trace = (
                        trace
                        - acoustic_rate * (trace - trace_eq)
                        + (1.0 - 0.5 * acoustic_rate) * trace_src
                    )
                difference = moment_20 - moment_02
                difference_eq = equilibrium_moment_20 - equilibrium_moment_02
                difference_src = force_moment_20 - force_moment_02
                if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                    reconstructed = self._regularized_deviatoric_stress(
                        i, j, material_density, relaxation_time
                    )
                    blend = ti.static(self.config.hrr_kinetic_blend)
                    difference = (
                        difference_eq
                        + blend * (difference - difference_eq)
                        + (1.0 - blend) * (reconstructed[0] - 0.5 * difference_src)
                    )
                    moment_11 = (
                        equilibrium_moment_11
                        + blend * (moment_11 - equilibrium_moment_11)
                        + (1.0 - blend) * (reconstructed[1] - 0.5 * force_moment_11)
                    )
                post_difference = (
                    difference
                    - omega * (difference - difference_eq)
                    + source_prefactor * difference_src
                )
                post_moment_20 = 0.5 * (post_trace + post_difference)
                post_moment_02 = 0.5 * (post_trace - post_difference)
                post_moment_11 = (
                    moment_11
                    - omega * (moment_11 - equilibrium_moment_11)
                    + source_prefactor * force_moment_11
                )
                post_moment_21 = equilibrium_moment_21 + 0.5 * force_moment_21
                post_moment_12 = equilibrium_moment_12 + 0.5 * force_moment_12
                post_moment_22 = equilibrium_moment_22 + 0.5 * force_moment_22
                if ti.static(
                    self.config.momentum_collision_model != "rm_mrt"
                    and self.config.cm_acoustic_relaxation_rate != 1.0
                ):
                    acoustic_rate = ti.static(self.config.cm_acoustic_relaxation_rate)
                    post_moment_22 = (
                        moment_22
                        - acoustic_rate * (moment_22 - equilibrium_moment_22)
                        + (1.0 - 0.5 * acoustic_rate) * force_moment_22
                    )
                raw_post = ti.Vector.zero(ti.f32, D2Q9_DIRECTION_COUNT)
                if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                    a = ti.static(self.config.cm_acoustic_relaxation_rate)
                    post_pressure_moment = (
                        pressure_moment
                        - a * (pressure_moment - equilibrium_pressure_moment)
                        + (1.0 - 0.5 * a) * force_pressure_moment
                    )
                    ux, uy = moment_frame_velocity.x, moment_frame_velocity.y
                    raw_trace = (
                        post_moment_20
                        + post_moment_02
                        + 2.0 * (ux * post_moment_10 + uy * post_moment_01)
                        + (ux * ux + uy * uy) * post_moment_00
                    )
                    # Third central nonequilibrium moments have been removed
                    # above, i.e. raw third moments follow the stress recursion.
                    # Close fourth order with the independent pressure mode.
                    post_moment_22 = (
                        raw_trace
                        + post_pressure_moment
                        - (
                            2.0 * uy * post_moment_21
                            + uy * uy * post_moment_20
                            + 2.0 * ux * post_moment_12
                            + 4.0 * ux * uy * post_moment_11
                            + 2.0 * ux * uy * uy * post_moment_10
                            + ux * ux * post_moment_02
                            + 2.0 * ux * ux * uy * post_moment_01
                            + ux * ux * uy * uy * post_moment_00
                        )
                    )
                for q in range(D2Q9_DIRECTION_COUNT):
                    raw_post[q] = reconstruct_central_moment_population(
                        lattice_direction(q).x,
                        lattice_direction(q).y,
                        moment_frame_velocity.x,
                        moment_frame_velocity.y,
                        post_moment_00,
                        post_moment_10,
                        post_moment_01,
                        post_moment_20,
                        post_moment_02,
                        post_moment_11,
                        post_moment_21,
                        post_moment_12,
                        post_moment_22,
                    )
                prepared = self._prepare_momentum_cut_links(i, j, raw_post)
                for q in range(D2Q9_DIRECTION_COUNT):
                    self.momentum_stream_buffer[i, j, q] = prepared[q]

    @ti.kernel
    def _collide_phase(self):
        interface_width = ti.static(self.config.interface_width)
        phase_relaxation_rate = ti.static(1.0 / (3.0 * self.config.mobility + 0.5))
        for i, j in self.water_phase:
            if self._is_phase_cell(i, j):
                capacity = ti.cast(1.0, self._phase_dtype)
                if ti.static(self._conservative_phase_enabled):
                    capacity = self.conservative_phase.capacity[i, j]
                water_phase = ti.cast(self.water_phase[i, j], ti.f32)
                velocity = (
                    self.momentum_velocity_lattice[i, j]
                    + 0.5 * self.fluid_acceleration_lattice[i, j]
                )
                if ti.static(self._conservative_phase_enabled):
                    velocity = self._phase_velocity_at(i, j)
                grad_phi = ti.Vector([0.0, 0.0])
                alpha = ti.static(1.0 / 3.0)
                for direction_index in range(D2Q9_DIRECTION_COUNT):
                    integer_direction = lattice_direction(direction_index)
                    direction = ti.cast(integer_direction, ti.f32)
                    phi1 = self._phase_neighbor(
                        i, j, integer_direction.x, integer_direction.y, 1
                    )
                    phi2 = self._phase_neighbor(
                        i, j, integer_direction.x, integer_direction.y, 2
                    )
                    grad_phi += (
                        (1.0 - alpha)
                        * 3.0
                        * lattice_weight(direction_index)
                        * direction
                        * (phi1 - water_phase)
                    )
                    grad_phi += (
                        alpha
                        * 1.5
                        * lattice_weight(direction_index)
                        * direction
                        * (4.0 * phi1 - phi2 - 3.0 * water_phase)
                    )
                normal = grad_phi / (grad_phi.norm() + 1.0e-14)
                if ti.static(self.config.phase_normal_epsilon > 0.0):
                    # Weak bulk gradients must not receive a unit compression
                    # direction. Keep the small guard for subnormal epsilon.
                    normal = grad_phi / (
                        ti.sqrt(
                            grad_phi.dot(grad_phi)
                            + ti.static(self.config.phase_normal_epsilon**2)
                        )
                        + 1.0e-14
                    )
                compression = 4.0 * water_phase * (1.0 - water_phase) / interface_width

                phase_volume = ti.cast(0.0, ti.f64)
                phase_raw_moment_10 = ti.cast(0.0, ti.f32)
                phase_raw_moment_01 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_10 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_01 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_20 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_02 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_11 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_21 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_12 = ti.cast(0.0, ti.f32)
                phase_equilibrium_moment_22 = ti.cast(0.0, ti.f32)
                phase_force_moment_10 = ti.cast(0.0, ti.f32)
                phase_force_moment_01 = ti.cast(0.0, ti.f32)
                phase_force_moment_20 = ti.cast(0.0, ti.f32)
                phase_force_moment_02 = ti.cast(0.0, ti.f32)
                phase_force_moment_11 = ti.cast(0.0, ti.f32)
                phase_force_moment_21 = ti.cast(0.0, ti.f32)
                phase_force_moment_12 = ti.cast(0.0, ti.f32)
                phase_force_moment_22 = ti.cast(0.0, ti.f32)
                for q in range(D2Q9_DIRECTION_COUNT):
                    cx = ti.cast(lattice_direction(q).x, ti.f32)
                    cy = ti.cast(lattice_direction(q).y, ti.f32)
                    rx = cx - velocity.x
                    ry = cy - velocity.y
                    rx2 = rx * rx
                    ry2 = ry * ry
                    if ti.static(self._conservative_phase_enabled):
                        phase_volume += self.phase_populations[i, j, q]
                    h_value = ti.cast(self.phase_populations[i, j, q] / capacity, ti.f32)
                    h_equilibrium = phase_equilibrium(q, water_phase, velocity)
                    phase_force = (
                        lattice_weight(q)
                        * compression
                        * ti.Vector([cx, cy]).dot(normal)
                    )
                    phase_raw_moment_10 += h_value * rx
                    phase_raw_moment_01 += h_value * ry
                    phase_equilibrium_moment_10 += h_equilibrium * rx
                    phase_equilibrium_moment_01 += h_equilibrium * ry
                    phase_equilibrium_moment_20 += h_equilibrium * rx2
                    phase_equilibrium_moment_02 += h_equilibrium * ry2
                    phase_equilibrium_moment_11 += h_equilibrium * rx * ry
                    phase_equilibrium_moment_21 += h_equilibrium * rx2 * ry
                    phase_equilibrium_moment_12 += h_equilibrium * rx * ry2
                    phase_equilibrium_moment_22 += h_equilibrium * rx2 * ry2
                    phase_force_moment_10 += phase_force * rx
                    phase_force_moment_01 += phase_force * ry
                    phase_force_moment_20 += phase_force * rx2
                    phase_force_moment_02 += phase_force * ry2
                    phase_force_moment_11 += phase_force * rx * ry
                    phase_force_moment_21 += phase_force * rx2 * ry
                    phase_force_moment_12 += phase_force * rx * ry2
                    phase_force_moment_22 += phase_force * rx2 * ry2
                # The phase collision conserves phi and its source has exactly
                # zero zeroth moment. Re-summing equilibrium/source floats here
                # introduces a small biased mass source on every time step.
                phase_post_moment_00 = water_phase
                phase_post_moment_10 = (
                    phase_raw_moment_10
                    - phase_relaxation_rate
                    * (phase_raw_moment_10 - phase_equilibrium_moment_10)
                    + (1.0 - 0.5 * phase_relaxation_rate) * phase_force_moment_10
                )
                phase_post_moment_01 = (
                    phase_raw_moment_01
                    - phase_relaxation_rate
                    * (phase_raw_moment_01 - phase_equilibrium_moment_01)
                    + (1.0 - 0.5 * phase_relaxation_rate) * phase_force_moment_01
                )
                phase_post_moment_20 = (
                    phase_equilibrium_moment_20 + 0.5 * phase_force_moment_20
                )
                phase_post_moment_02 = (
                    phase_equilibrium_moment_02 + 0.5 * phase_force_moment_02
                )
                phase_post_moment_11 = (
                    phase_equilibrium_moment_11 + 0.5 * phase_force_moment_11
                )
                phase_post_moment_21 = (
                    phase_equilibrium_moment_21 + 0.5 * phase_force_moment_21
                )
                phase_post_moment_12 = (
                    phase_equilibrium_moment_12 + 0.5 * phase_force_moment_12
                )
                phase_post_moment_22 = (
                    phase_equilibrium_moment_22 + 0.5 * phase_force_moment_22
                )
                for q in range(D2Q9_DIRECTION_COUNT):
                    self.phase_stream_buffer[i, j, q] = (
                        capacity * reconstruct_central_moment_population(
                            lattice_direction(q).x,
                            lattice_direction(q).y,
                            velocity.x,
                            velocity.y,
                            phase_post_moment_00,
                            phase_post_moment_10,
                            phase_post_moment_01,
                            phase_post_moment_20,
                            phase_post_moment_02,
                            phase_post_moment_11,
                            phase_post_moment_21,
                            phase_post_moment_12,
                            phase_post_moment_22,
                        )
                    )
                # Close the local zeroth moment through the rest population;
                # this leaves every transport moment unchanged. Accumulate in
                # f64 as in macroscopic phase recovery to avoid biased f32 sums.
                outgoing_sum = ti.cast(0.0, ti.f64)
                for q in range(1, D2Q9_DIRECTION_COUNT):
                    outgoing_sum += ti.cast(self.phase_stream_buffer[i, j, q], ti.f64)
                zero = capacity * self.water_phase[i, j]
                if ti.static(self._conservative_phase_enabled):
                    zero = phase_volume
                self.phase_stream_buffer[i, j, 0] = ti.cast(zero - outgoing_sum, self._phase_dtype)

    @ti.kernel
    def _stream(self):
        """Push each population to its unique destination after boundary preparation."""
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = lattice_direction(q)
                    neighbor_i, neighbor_j = i + direction.x, j + direction.y
                    outgoing_momentum = self.momentum_stream_buffer[i, j, q]
                    outgoing_phase = self.phase_stream_buffer[i, j, q]
                    if inside_grid(
                        neighbor_i, neighbor_j, self.nx, self.ny
                    ) and self._is_fluid_cell(neighbor_i, neighbor_j):
                        self.momentum_populations[neighbor_i, neighbor_j, q] = (
                            outgoing_momentum
                        )
                        self.phase_populations[neighbor_i, neighbor_j, q] = (
                            outgoing_phase
                        )
                    else:
                        opposite = opposite_direction_index(q)
                        self.momentum_populations[i, j, opposite] = outgoing_momentum
                        if (
                            inside_grid(neighbor_i, neighbor_j, self.nx, self.ny)
                            and self.solid_mask[neighbor_i, neighbor_j] == 1
                        ):
                            fraction = self._cut_link_fraction(
                                i, j, neighbor_i, neighbor_j
                            )
                            direction_float = ti.cast(direction, ti.f32)
                            boundary_velocity = self._body_velocity_at(
                                self._cell_point(i, j) + fraction * direction_float,
                                self.solid_body_id[neighbor_i, neighbor_j],
                            )
                            before_phase = outgoing_phase
                            outgoing_phase -= (
                                6.0
                                * lattice_weight(q)
                                * self.water_phase[i, j]
                                * direction_float.dot(boundary_velocity)
                            )
                            if ti.static(self._conservative_phase_enabled):
                                self.conservative_phase.record_wall(i, j, before_phase, outgoing_phase)
                        self.phase_populations[i, j, opposite] = outgoing_phase

    @ti.kernel
    def _stream_phase_only(self):
        """Stream h during phase warm-up without executing fluid coupling."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = lattice_direction(q)
                    ni = i + direction.x
                    nj = j + direction.y
                    outgoing = self.phase_stream_buffer[i, j, q]
                    if inside_grid(ni, nj, nx, ny) and self._is_fluid_cell(ni, nj):
                        self.phase_populations[ni, nj, q] = outgoing
                    elif inside_grid(ni, nj, nx, ny) and self.solid_mask[ni, nj] == 1:
                        eta = self._cut_link_fraction(i, j, ni, nj)
                        direction_f = ti.cast(direction, ti.f32)
                        boundary_velocity = self._body_velocity_at(
                            self._cell_point(i, j) + eta * direction_f,
                            self.solid_body_id[ni, nj],
                        )
                        reflected = outgoing - (
                            6.0 * lattice_weight(q) * self.water_phase[i, j]
                            * direction_f.dot(boundary_velocity)
                        )
                        if ti.static(self._conservative_phase_enabled):
                            self.conservative_phase.record_wall(i, j, outgoing, reflected)
                        self.phase_populations[i, j, opposite_direction_index(q)] = reflected
                    else:
                        self.phase_populations[i, j, opposite_direction_index(q)] = (
                            outgoing
                        )

    @ti.kernel
    def _update_streamed_macroscopic_fields(self):
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                water_phase = ti.cast(0.0, ti.f64)
                momentum = ti.Vector([0.0, 0.0])
                for q in range(D2Q9_DIRECTION_COUNT):
                    water_phase += ti.cast(self.phase_populations[i, j, q], ti.f64)
                    momentum += (
                        ti.cast(lattice_direction(q), ti.f32)
                        * self.momentum_populations[i, j, q]
                    )
                self.water_phase[i, j] = ti.cast(water_phase, self._phase_dtype)
                density = mixture_density(
                    ti.cast(water_phase, ti.f32), rho_water, rho_air
                )
                self.momentum_velocity_lattice[i, j] = momentum / ti.max(
                    density, 1.0e-12
                )
            else:
                velocity = ti.Vector([0.0, 0.0])
                if self.solid_mask[i, j] == 1:
                    velocity = self._body_velocity_at(
                        self._cell_point(i, j), ti.max(self.solid_body_id[i, j], 0)
                    )
                if self.wall_mask[i, j] == 1:
                    self.water_phase[i, j] = 0.0
                self.momentum_velocity_lattice[i, j] = velocity
                self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _update_streamed_phase(self):
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                water_phase = ti.cast(0.0, ti.f64)
                for q in range(D2Q9_DIRECTION_COUNT):
                    water_phase += ti.cast(self.phase_populations[i, j, q], ti.f64)
                self.water_phase[i, j] = ti.cast(water_phase, self._phase_dtype)
            else:
                if self.wall_mask[i, j] == 1:
                    self.water_phase[i, j] = 0.0

    def _warm_start_phase(self, steps):
        for _ in range(int(steps)):
            self._collide_phase()
            if self._conservative_phase_enabled:
                self.conservative_phase.prepare_transport()
                self.conservative_phase.begin_fluid(0)
            self._stream_phase_only()
            if self._conservative_phase_enabled:
                self.conservative_phase.limit_stream()
            self._update_streamed_phase()
            if self._conservative_phase_enabled:
                self.conservative_phase.finish_energy()
                self.conservative_phase.repair_bounds()

    @ti.kernel
    def _update_pressure(self):
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.pressure_lattice:
            if self._is_fluid_cell(i, j):
                moving_distribution_sum = 0.0
                for q in range(D2Q9_DIRECTION_COUNT):
                    if q != 0:
                        moving_distribution_sum += self.momentum_populations[i, j, q]
                water_phase = ti.cast(self.water_phase[i, j], ti.f32)
                material_density = mixture_density(water_phase, rho_water, rho_air)
                velocity = (
                    self.momentum_velocity_lattice[i, j]
                    + 0.5 * self.fluid_acceleration_lattice[i, j]
                )
                grad_phi = self._momentum_phase_gradient(i, j)
                grad_rho = (rho_water - rho_air) * grad_phi
                s0 = -1.5 * lattice_weight(0) * velocity.dot(velocity)
                dynamic_pressure = 0.6 * (
                    moving_distribution_sum
                    + 0.5 * velocity.dot(grad_rho)
                    + material_density * s0
                )
                reference_pressure = self.hydrostatic_reference_pressure[i, j]
                self.pressure_lattice[i, j] = reference_pressure + dynamic_pressure
            else:
                if self.wall_mask[i, j] == 1:
                    self.pressure_lattice[i, j] = 0.0

    # Rigid-body forcing and integration

    def _prepare_moving_body_contact_support(self):
        """Refresh the candidate SDF and reduce its wall support.

        Keep this small compatibility wrapper for callers that need to query
        contact support outside ``_solve_contact_and_rasterize``.  The
        rasterizer itself already has a current candidate pass and calls the
        device kernel directly; standalone callers must populate the shared
        SDF first.
        """

        self._calculate_moving_body_fraction()
        self._prepare_moving_body_contact_support_kernel(True)

    @ti.kernel
    def _prepare_moving_body_contact_support_kernel(
        self, reduce_support: ti.template()
    ):
        for b in range(self.body_count):
            self._reset_moving_body_contact_support(b)
        if ti.static(reduce_support):
            for b, i, j in self._body_fractions:
                self._reduce_moving_body_contact_extrema(b, i, j)
            if ti.static(self._conservative_phase_enabled):
                # Thresholded raster support can omit eroded material tiles.
                # Their real mass must also stay inside the container.
                for a, j in self.thermal.body_solid_mass:
                    b = self.thermal.material_body_id[a, j]
                    if b >= 0 and self.thermal.body_solid_mass[a, j] > 0:
                        angle = ti.cast(self.body_angle[b], ti.f64)
                        c, sn = ti.cos(angle), ti.sin(angle)
                        x = a - self.thermal.material_offset[b] + 0.5 - 0.5 * self.thermal.material_size[b].x
                        y = j + 0.5 - 0.5 * self.thermal.material_size[b].y
                        rx, ry = c * x - sn * y, sn * x + c * y
                        extent = 0.5 * (ti.abs(c) + ti.abs(sn)) + 2e-4
                        ti.atomic_min(self._body_contact_support_extrema[b, 0], ti.cast(rx - extent, ti.f32))
                        ti.atomic_max(self._body_contact_support_extrema[b, 1], ti.cast(rx + extent, ti.f32))
                        ti.atomic_min(self._body_contact_support_extrema[b, 2], ti.cast(ry - extent, ti.f32))
                        ti.atomic_max(self._body_contact_support_extrema[b, 3], ti.cast(ry + extent, ti.f32))
            for b in range(self.body_count):
                self._finalize_moving_body_contact_support(b)

    @ti.func
    def _reset_moving_body_contact_support(self, b=0):
        self._body_contact_support_extrema[b, 0] = 1.0e30
        self._body_contact_support_extrema[b, 1] = -1.0e30
        self._body_contact_support_extrema[b, 2] = 1.0e30
        self._body_contact_support_extrema[b, 3] = -1.0e30

    @ti.func
    def _reduce_moving_body_contact_extrema(self, b, i, j):
        """Find supports of the unclipped sharp LBM raster at this pose."""

        threshold = ti.static(float(self.config.thermal.solid_liquid_threshold))
        origin = self.body_reference_origin[b]
        fraction = self._body_fractions[b, i, j]
        # The fraction pass has already produced the same raw SDF used
        # by ``_update_solid_mask``.  Reusing its sign
        # and the thresholded coverage keeps contact consistent with the
        # sharp mask, including a legitimate rectangle-edge node whose
        # SDF is zero, while excluding an outside bilinear contributor or
        # an empty rectangle boundary at lower thresholds.
        if fraction >= threshold and self._body_level_at(b, i, j) <= 0.0:
            # The sharp solver treats a thresholded world node as one
            # cell-centred finite volume.  Its cell faces therefore give
            # the collision support, with the same one-cell topology as
            # ``solid`` and the cut-link boundary.
            minimum_x = ti.cast(i, ti.f32) - origin.x
            maximum_x = ti.cast(i + 1, ti.f32) - origin.x
            minimum_y = ti.cast(j, ti.f32) - origin.y
            maximum_y = ti.cast(j + 1, ti.f32) - origin.y
            ti.atomic_min(self._body_contact_support_extrema[b, 0], minimum_x)
            ti.atomic_max(self._body_contact_support_extrema[b, 1], maximum_x)
            ti.atomic_min(self._body_contact_support_extrema[b, 2], minimum_y)
            ti.atomic_max(self._body_contact_support_extrema[b, 3], maximum_y)

    @ti.func
    def _finalize_moving_body_contact_support(self, b=0):
        active = self._body_contact_support_extrema[b, 2] < 1.0e20
        if not active:
            for index in ti.static(range(4)):
                self._body_contact_support_extrema[b, index] = 0.0

    @ti.kernel
    def _integrate_rigid_ice(self):
        for b in range(self.body_count):
            if self.body_active[b] != 0 and self._body_has_resolved_boundary[b] != 0:
                mass = ti.cast(self.body_mass_lattice[b], ti.f32)
                inertia = ti.cast(self.body_inertia_lattice[b], ti.f32)
                # This branch admits only a mechanically resolved body, so these are
                # the same stored mass and inertia audited by the thermal transition.
                gravity = ti.Vector(
                    [
                        ti.static(self._gravity_lattice[0]),
                        ti.static(self._gravity_lattice[1]),
                    ]
                )
                total_impulse = (
                    ti.cast(self.hydrodynamic_impulse[b], ti.f32) + mass * gravity
                )
                total_torque = ti.cast(self.hydrodynamic_torque[b], ti.f32)

                velocity = ti.static(self.config.linear_damping) * (
                    self.body_velocity[b] + total_impulse / mass
                )
                omega = ti.static(self.config.angular_damping) * (
                    self.body_angular_velocity[b] + total_torque / inertia
                )
                local_com = ti.cast(self.body_local_center_of_mass[b], ti.f32)
                angle = self.body_angle[b] + omega
                center = self.body_center[b] + velocity

                cosine = ti.cos(angle)
                sine = ti.sin(angle)
                self.body_velocity[b] = velocity
                self.body_angle[b] = angle
                self.body_angular_velocity[b] = omega
                reference_offset = ti.Vector(
                    [
                        cosine * local_com.x - sine * local_com.y,
                        sine * local_com.x + cosine * local_com.y,
                    ]
                )
                self.body_reference_origin[b] = center - reference_offset
            else:
                # A subcell remnant has no cut links or wall contact. Integrating
                # gravity alone would launch invisible material through the walls.
                # Hold its pose at the resolution limit; retain its mass, coverage
                # and thermal exchange so melting can continue without deletion.
                self.body_velocity[b] = ti.Vector([0.0, 0.0])
                self.body_angular_velocity[b] = 0.0
            self.hydrodynamic_impulse[b] = ti.Vector([0.0, 0.0])
            self.hydrodynamic_torque[b] = 0.0

    @ti.kernel
    def _project_moving_body_inside_container(self):
        """Keep the whole body inside the walls and reject outward velocity."""
        for b in range(self.body_count):
            self._body_contact_projection_changed[b] = ti.cast(0, ti.i8)
            center = self.body_center[b]
            velocity = self.body_velocity[b]
            angle = self.body_angle[b]
            cosine = ti.cos(angle)
            sine = ti.sin(angle)
            local_com = ti.cast(self.body_local_center_of_mass[b], ti.f32)
            rotated_com = ti.Vector(
                [
                    cosine * local_com.x - sine * local_com.y,
                    sine * local_com.x + cosine * local_com.y,
                ]
            )

            if self.body_active[b] != 0 and self._body_contact_geometry_active[b] != 0:
                minimum_relative_x = (
                    self._body_contact_support_extrema[b, 0] - rotated_com.x
                )
                maximum_relative_x = (
                    self._body_contact_support_extrema[b, 1] - rotated_com.x
                )
                minimum_relative_y = (
                    self._body_contact_support_extrema[b, 2] - rotated_com.y
                )
                maximum_relative_y = (
                    self._body_contact_support_extrema[b, 3] - rotated_com.y
                )
                lower_boundary = ti.static(float(self.config.boundary_cells))
                upper_x_boundary = ti.static(
                    float(self.nx - self.config.boundary_cells)
                )
                upper_y_boundary = ti.static(
                    float(self.ny - self.config.boundary_cells)
                )
                lower_x = lower_boundary - minimum_relative_x
                upper_x = upper_x_boundary - maximum_relative_x
                lower_y = lower_boundary - minimum_relative_y
                upper_y = upper_y_boundary - maximum_relative_y
                if center.x <= lower_x:
                    if center.x < lower_x:
                        self._body_contact_projection_changed[b] = ti.cast(1, ti.i8)
                    center.x = lower_x
                    velocity.x = ti.max(velocity.x, 0.0)
                elif center.x >= upper_x:
                    if center.x > upper_x:
                        self._body_contact_projection_changed[b] = ti.cast(1, ti.i8)
                    center.x = upper_x
                    velocity.x = ti.min(velocity.x, 0.0)
                if center.y <= lower_y:
                    if center.y < lower_y:
                        self._body_contact_projection_changed[b] = ti.cast(1, ti.i8)
                    center.y = lower_y
                    velocity.y = ti.max(velocity.y, 0.0)
                elif center.y >= upper_y:
                    if center.y > upper_y:
                        self._body_contact_projection_changed[b] = ti.cast(1, ti.i8)
                    center.y = upper_y
                    velocity.y = ti.min(velocity.y, 0.0)

            self.body_velocity[b] = velocity
            # The fraction pass can be committed unchanged when contact only
            # rejects an outward velocity at an already admissible boundary.  In
            # that case leave the authoritative ALE origin bit-for-bit untouched;
            # rewrite it only after a positional projection has actually changed
            # the accepted pose.
            if self._body_contact_projection_changed[b] != 0:
                self.body_reference_origin[b] = center - rotated_com

    def _solve_contact_and_rasterize(self, *, resolve_contact=True):
        """Solve all contacts before atomically committing the union geometry."""
        self._calculate_moving_body_fraction()
        projected = False
        if resolve_contact:
            for _ in range(self.config.body_contact_iterations):
                self._prepare_moving_body_contact_support_kernel(True)
                self._project_moving_body_inside_container()
                wall_changed = bool(
                    np.any(self._body_contact_projection_changed.array())
                )
                if wall_changed:
                    projected = True
                    self._calculate_moving_body_fraction()
                    self._prepare_moving_body_contact_support_kernel(True)
                touching = self.contacts is not None and self.contacts.detect() > 0
                if not touching:
                    break
                self.contacts.project()
                projected = True
                self._calculate_moving_body_fraction()
            else:
                # The last pair correction can push a body back through a
                # container wall even when the pair itself is now separated.
                self._prepare_moving_body_contact_support_kernel(True)
                self._project_moving_body_inside_container()
                if np.any(self._body_contact_projection_changed.array()):
                    self._calculate_moving_body_fraction()
                    self._prepare_moving_body_contact_support_kernel(True)
                # Never silently commit interpenetrating rigid bodies.
                if self.contacts is not None and self.contacts.detect() > 0:
                    raise RuntimeError(
                        "ice contact projection did not converge; reduce the time step or initial crowding"
                    )
        else:
            self._body_contact_projection_changed.fill(0)
            self._prepare_moving_body_contact_support_kernel(False)
        self._commit_moving_body_raster()
        return projected

    # ------------------------------------------------------------------
    # Local phase bounds and output geometry

    @ti.kernel
    def _reduce_phase_change_geometry_volume(self) -> ti.f64:
        volume = ti.cast(0.0, ti.f64)
        for i, j in self.solid_mask:
            if self.solid_mask[i, j] != 0:
                volume += 1.0
        return volume

    def phase_change_solid_volume_cells(self):
        return float(self.phase_change_current_solid_volume[None])

    def phase_change_geometry_volume_cells(self):
        return float(self._reduce_phase_change_geometry_volume())

    def _bound_water_phase(self):
        if self._conservative_phase_enabled:
            return
        self._bound_water_phase_legacy()

    @ti.kernel
    def _bound_water_phase_legacy(self):
        """Repair only out-of-range phase values and their phase populations."""

        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                old_phase = self.water_phase[i, j]
                if old_phase < 0.0 or old_phase > 1.0:
                    bounded_phase = ti.min(1.0, ti.max(0.0, old_phase))
                    velocity = self.momentum_velocity_lattice[i, j]
                    # mixture_density already clamps phi, so repairing an
                    # overshoot does not change density or momentum populations.
                    # Lift only h's equilibrium part; valid cells, including
                    # interface tails and pure-phase endpoints, remain untouched.
                    lifted_sum = 0.0
                    for q in range(D2Q9_DIRECTION_COUNT):
                        lifted = (
                            self.phase_populations[i, j, q]
                            + phase_equilibrium(q, bounded_phase, velocity)
                            - phase_equilibrium(q, old_phase, velocity)
                        )
                        self.phase_populations[i, j, q] = lifted
                        lifted_sum += lifted
                    self.phase_populations[i, j, 0] += bounded_phase - lifted_sum
                    self.water_phase[i, j] = bounded_phase

    def sample_thermal_fields(self):
        return self.thermal.sample_world_fields(self.wall_mask.to_numpy())

    @ti.func
    def _wall_at(self, i, j):
        boundary = ti.static(self.config.boundary_cells)
        return ti.cast(
            i < boundary
            or i >= self.nx - boundary
            or j < boundary
            or j >= self.ny - boundary,
            ti.i8,
        )

    def _wall_numpy(self):
        x, y = np.ogrid[: self.nx, : self.ny]
        boundary = self.config.boundary_cells
        return (
            (x < boundary)
            | (x >= self.nx - boundary)
            | (y < boundary)
            | (y >= self.ny - boundary)
        ).astype(np.int8)

    @ti.func
    def _signed_distance_at(self, i, j):
        # A signed level set for cut-link interpolation, not a Euclidean SDF.
        # The material coverage is authoritative even on the original box edge.
        return (
            self.config.thermal.solid_liquid_threshold
            - self._moving_body_fraction[i, j]
        ) * self.config.dx

    def _signed_distance_numpy(self):
        return (
            (
                self.config.thermal.solid_liquid_threshold
                - self._moving_body_fraction.to_numpy()
            )
            * self.config.dx
        ).astype(np.float32)

    @ti.func
    def _reference_pressure_at(self, i, j):
        return self.reference_pressure_by_row[j]

    @ti.func
    def _reference_density_at(self, i, j):
        return self.reference_density_by_row[j]

    @ti.func
    def _physical_velocity_at(self, i, j):
        velocity = ti.Vector([0.0, 0.0])
        if self._is_fluid_cell(i, j):
            velocity = (
                self.momentum_velocity_lattice[i, j]
                + 0.5 * self.fluid_acceleration_lattice[i, j]
            )
        return velocity

    def _physical_velocity_numpy(self):
        velocity = (
            self.momentum_velocity_lattice.to_numpy()
            + np.float32(0.5) * self.fluid_acceleration_lattice.to_numpy()
        )
        velocity[
            (self.wall_mask.to_numpy() != 0) | (self.solid_mask.to_numpy() != 0)
        ] = 0.0
        return velocity

    @ti.func
    def _read_body_initial_mass_lattice(self, index):
        b = index
        return self._initial_masses[b]

    @ti.func
    def _read_body_active(self, index):
        b = index
        return ti.cast(self.body_inertia_lattice[b] > 0.0, ti.i8)

    @ti.func
    def _read_cumulative_melted_mass_lattice(self, index):
        b = index
        return self._initial_masses[b] - self.body_mass_lattice[b]

    @ti.func
    def _read_phase_change_initial_solid_volume(self, index):
        total = ti.cast(0.0, ti.f64)
        for b in range(self.body_count):
            total += self._initial_masses[b] * (
                self.config.rho_water / self.config.rho_ice
            )
        return total

    @ti.func
    def _read_phase_change_current_solid_volume(self, index):
        total = ti.cast(0.0, ti.f64)
        for b in range(self.body_count):
            total += self.body_mass_lattice[b] * (
                self.config.rho_water / self.config.rho_ice
            )
        return total

    @ti.func
    def _read__body_contact_geometry_active(self, index):
        b = index
        return ti.cast(
            self._body_contact_support_extrema[b, 1]
            > self._body_contact_support_extrema[b, 0],
            ti.i8,
        )

    @ti.func
    def _body_center_at(self, index):
        b = index
        angle = self.body_angle[b]
        cosine, sine = ti.cos(angle), ti.sin(angle)
        local = self.body_local_center_of_mass[b]
        offset = ti.Vector(
            [cosine * local.x - sine * local.y, sine * local.x + cosine * local.y]
        )
        return ti.cast(ti.cast(self.body_reference_origin[b], ti.f64) + offset, ti.f32)

    def _body_center_numpy(self):
        angle = self.body_angle.array()
        local = self.body_local_center_of_mass.array()
        c, s = np.cos(angle), np.sin(angle)
        offset = np.stack(
            (c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]),
            axis=-1,
        )
        centers = (
            self.body_reference_origin.array().astype(np.float64) + offset
        ).astype(np.float32)
        return centers[0] if self.body_count == 1 else centers

    def _set_body_center_host(self, index, value):
        if index is None:
            if self.body_count != 1:
                raise IndexError("multiple bodies require a body_id index")
            index = 0
        angle = float(self.body_angle[index])
        x, y = self.body_local_center_of_mass[index]
        offset = np.array(
            [
                np.cos(angle) * x - np.sin(angle) * y,
                np.sin(angle) * x + np.cos(angle) * y,
            ]
        )
        self.body_reference_origin[index] = np.asarray(value, dtype=np.float64) - offset

    @ti.func
    def _body_level_at(self, b, i, j):
        return (
            ti.static(self.config.thermal.solid_liquid_threshold)
            - self._body_fractions[b, i, j]
        ) * self.config.dx

    @ti.func
    def _cut_link_fraction(self, i, j, neighbor_i, neighbor_j):
        b = ti.max(self.solid_body_id[neighbor_i, neighbor_j], 0)
        fluid_distance = ti.max(self._body_level_at(b, i, j), self.config.dx * 1e-6)
        return ti.min(
            0.95,
            ti.max(
                0.05,
                fluid_distance
                / (
                    fluid_distance
                    - self._body_level_at(b, neighbor_i, neighbor_j)
                    + 1e-12
                ),
            ),
        )

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        """Apply Tao's rule while the original populations are still available.

        A cut-link's outgoing slot is otherwise unused by push streaming. Store
        its reflected population in that slot, after recording momentum exchange.
        This removes the full pre-collision non-equilibrium array without reading
        a population that another streaming thread may overwrite.
        """
        prepared = raw_post
        impulse_sum = ti.Vector.zero(ti.f64, 2)
        torque_sum = ti.cast(0.0, ti.f64)
        has_cut_link = False
        density = mixture_density(
            self.water_phase[i, j],
            self._water_density_lattice,
            self._air_density_lattice,
        )
        dynamic_pressure = (
            self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
        )
        velocity = (
            self.momentum_velocity_lattice[i, j]
            + 0.5 * self.fluid_acceleration_lattice[i, j]
        )
        for q in range(1, D2Q9_DIRECTION_COUNT):
            direction = lattice_direction(q)
            neighbor_i, neighbor_j = i + direction.x, j + direction.y
            if (
                inside_grid(neighbor_i, neighbor_j, self.nx, self.ny)
                and self.solid_mask[neighbor_i, neighbor_j] == 1
            ):
                b = self.solid_body_id[neighbor_i, neighbor_j]
                if ti.static(self.body_count > 1):
                    impulse_sum = ti.Vector.zero(ti.f64, 2)
                    torque_sum = ti.cast(0.0, ti.f64)
                fraction = self._cut_link_fraction(i, j, neighbor_i, neighbor_j)
                direction_float = ti.cast(direction, ti.f32)
                boundary_point = self._cell_point(i, j) + fraction * direction_float
                boundary_velocity = self._body_velocity_at(boundary_point, b)
                if ti.static(self._conservative_phase_enabled):
                    boundary_velocity += self.conservative_phase.boundary_offset(i, j, neighbor_i, neighbor_j, b)
                opposite = opposite_direction_index(q)
                nonequilibrium = self.momentum_populations[
                    i, j, q
                ] - momentum_equilibrium(q, dynamic_pressure, density, velocity)
                wall_equilibrium = momentum_equilibrium(
                    opposite, dynamic_pressure, density, boundary_velocity
                )
                reflected = (
                    wall_equilibrium + nonequilibrium + fraction * raw_post[opposite]
                ) / (1.0 + fraction)
                prepared[q] = reflected
                impulse = (direction_float - boundary_velocity) * raw_post[q] - (
                    -direction_float - boundary_velocity
                ) * reflected
                reference_pressure = (
                    1.0 - fraction
                ) * self.hydrostatic_reference_pressure[
                    i, j
                ] + fraction * self.hydrostatic_reference_pressure[
                    neighbor_i, neighbor_j
                ]
                impulse += (
                    6.0 * lattice_weight(q) * reference_pressure * direction_float
                )
                impulse64 = ti.cast(impulse, ti.f64)
                relative64 = ti.cast(boundary_point - self.body_center[b], ti.f64)
                impulse_sum += impulse64
                torque_sum += cross_product_2d(relative64, impulse64)
                if ti.static(self._surface_tension_lattice > 0.0):
                    # Momentum exchange contains pressure and viscous traction,
                    # but not the direct capillary traction at the contact line.
                    # Use the exact reaction of the fluid's blocked-link flux.
                    local_stress = self.capillary_stress_lattice[i, j]
                    ghost_stress = self._ice_capillary_stress(
                        i, j, neighbor_i, neighbor_j
                    )
                    capillary_impulse = ti.cast(
                        -3.0
                        * lattice_weight(q)
                        * self._capillary_traction(
                            local_stress + ghost_stress, direction_float
                        ),
                        ti.f64,
                    )
                    # These stress fluxes live at lattice-link midpoints, unlike
                    # interpolated no-slip reflection. The matching moment arm
                    # makes their symmetric stress conserve angular momentum.
                    capillary_arm = ti.cast(
                        self._cell_point(i, j)
                        + 0.5 * direction_float
                        - self.body_center[b],
                        ti.f64,
                    )
                    impulse_sum += capillary_impulse
                    torque_sum += cross_product_2d(capillary_arm, capillary_impulse)
                    # Complete the symmetric-stress moment when the ghost
                    # differs from the local tensor. Midpoint force moments
                    # alone leave +1.5*w*c x ((K_i-K_ghost)c) unbalanced.
                    delta_traction = self._capillary_traction(
                        local_stress - ghost_stress, direction_float
                    )
                    torque_sum -= ti.cast(
                        1.5
                        * lattice_weight(q)
                        * cross_product_2d(direction_float, delta_traction),
                        ti.f64,
                    )
                has_cut_link = True
                if ti.static(self.body_count > 1):
                    ti.atomic_add(self.hydrodynamic_impulse[b], impulse_sum)
                    ti.atomic_add(self.hydrodynamic_torque[b], torque_sum)
        if ti.static(self.body_count == 1):
            if has_cut_link:
                ti.atomic_add(self.hydrodynamic_impulse[0], impulse_sum)
                ti.atomic_add(self.hydrodynamic_torque[0], torque_sum)
        return prepared

    def _allocate_populations(self, dtype=ti.f32):
        """Keep (i,j,q) indexing with q-major physical storage for coalesced loads."""
        populations = ti.field(dtype)
        ti.root.dense(ti.k, D2Q9_DIRECTION_COUNT).dense(
            ti.ij, (self.nx, self.ny)
        ).place(populations)
        return populations
