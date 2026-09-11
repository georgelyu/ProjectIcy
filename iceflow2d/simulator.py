"""2D air--water LBM coupled to a falling, melting rigid ice body.

The two distribution functions use pressure--momentum and conservative phase
central-moment collision operators.  An oriented-box signed-distance field
excludes ice nodes from the fluid domain, and cut lattice links enforce an
impermeable rigid boundary.  Link momentum exchange drives the freely moving
ice body.

A material-frame ALE enthalpy solver keeps ice mass and sensible energy in
body coordinates, advects world-water volume and sensible energy with paired
fluxes, and feeds melt mass, energy, and momentum back to the flow and dynamic
rigid-body mass properties.
"""

import math
from typing import Any

import numpy as np
import taichi as ti

from .config import (
    IceFlowConfig,
    LatticeScales,
    MovingBodyThermalTotals,
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
)

_MAX_D2Q9_LATTICE_VELOCITY_L1 = 1.0
_MAX_D2Q9_BULK_PHASE_POPULATION_L1 = 2.0


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


@ti.func
def _warp_shuffle_down_f64(value, offset):
    """Shuffle the two 32-bit words without rounding an f64 reduction value."""
    bits = ti.bit_cast(value, ti.u64)
    low = ti.cast(bits, ti.i32)
    high = ti.cast(bits >> 32, ti.i32)
    low = ti.simt.warp.shfl_down_i32(ti.cast(-1, ti.u32), low, offset)
    high = ti.simt.warp.shfl_down_i32(ti.cast(-1, ti.u32), high, offset)
    shuffled = ti.cast(ti.cast(low, ti.u32), ti.u64) | (
        ti.cast(ti.cast(high, ti.u32), ti.u64) << 32
    )
    return ti.bit_cast(shuffled, ti.f64)


@ti.func
def _warp_reduce_f64(value, operation: ti.template()):
    """Reduce a full CUDA warp; callers pad their range to whole 128-thread blocks."""
    result = value
    for offset in ti.static((16, 8, 4, 2, 1)):
        neighbor = _warp_shuffle_down_f64(result, offset)
        if ti.static(operation == "sum"):
            result += neighbor
        elif ti.static(operation == "min"):
            result = ti.min(result, neighbor)
        else:
            result = ti.max(result, neighbor)
    return result


class _DerivedField:
    """Read-only, allocation-free field view; device reads inline arithmetic.

    Host reads materialize NumPy values on demand. This is deliberately not a
    Taichi field: no persistent device allocation, upload, or refresh kernel.
    """

    def __init__(self, device_reader, host_reader, *, host_writer=None):
        self._device_reader = device_reader
        self._host_reader = host_reader
        self._host_writer = host_writer

    def __getitem__(self, index):
        if ti.lang.impl.inside_kernel():
            indices = index if isinstance(index, tuple) else (index,)
            return self._device_reader(*indices)
        values = self._host_reader()
        return values if index is None else values[index]

    def __setitem__(self, index, value):
        if self._host_writer is None or ti.lang.impl.inside_kernel():
            raise TypeError("derived values are read-only; update the conserved state")
        self._host_writer(index, value)

    def to_numpy(self):
        return np.asarray(self._host_reader()).copy()


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
        self.body_nx = int(body_nx)
        self.body_ny = int(body_ny)
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
        self._body_melt_target = ti.field(ti.i32, shape=body_shape)
        self._body_melt_mass_step = ti.field(ti.f64, shape=body_shape)
        self._body_melt_sensible_step = ti.field(ti.f64, shape=body_shape)

        self.water_volume_m2 = ti.field(ti.f64, shape=world_shape)
        self.water_sensible_energy = ti.field(ti.f64, shape=world_shape)
        self._water_energy_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
        self._water_energy_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
        self._water_volume_flux_x = ti.field(ti.f64, shape=(self.nx + 1, self.ny))
        self._water_volume_flux_y = ti.field(ti.f64, shape=(self.nx, self.ny + 1))
        # Conservative projection between the extensive thermal water
        # state and the current LBM water aperture.  The target is built
        # from ``phi`` but is capped by one physical cell area, so newly
        # wetted cells receive a thermal state and no cell can hold more
        # water than its geometric capacity.
        self._phase_target_energy = ti.field(ti.f64, shape=world_shape)
        # One row per padded CUDA warp. These independent partial sums are
        # consumed by a second reduction, avoiding a contended global scalar.
        self._aperture_partial_count = ((self.nx * self.ny + 127) // 128) * 4
        self._aperture_reduction_partials = ti.field(
            ti.f64, shape=(self._aperture_partial_count, 6)
        )
        self._phase_volume_before = ti.field(ti.f64, shape=())
        self._phase_energy_before = ti.field(ti.f64, shape=())
        self._phase_base_capacity = ti.field(ti.f64, shape=())
        self._phase_full_capacity = ti.field(ti.f64, shape=())
        self._phase_specific_energy_min = ti.field(ti.f64, shape=())
        self._phase_specific_energy_max = ti.field(ti.f64, shape=())
        self._phase_reconstructed_energy = ti.field(ti.f64, shape=())
        self._phase_cooling_capacity = ti.field(ti.f64, shape=())
        self._phase_heating_capacity = ti.field(ti.f64, shape=())
        # The bounded energy projection is a deliberate low-cost
        # approximation to a fully local geometric transport solve.
        # Expose its size separately from the conservation residual.
        self.aperture_energy_correction_j_m = _DerivedField(
            self._read_aperture_energy_correction_j_m,
            lambda: self._phase_energy_before[None]
            - self._phase_reconstructed_energy[None],
        )
        self.aperture_energy_correction_abs_j_m = ti.field(ti.f64, shape=())
        self._phase_volume_after = ti.field(ti.f64, shape=())
        self._phase_energy_after = ti.field(ti.f64, shape=())
        # Snapshot of eligible cells used while body-cell threads scatter
        # newly melted water.  It is frozen in a separate kernel so one
        # thread cannot make a previously dry cell eligible while another
        # thread is still selecting its fallback target.
        self._melt_injection_eligible = ti.field(ti.i8, shape=world_shape)

        self.world_body_solid_fraction = ti.field(ti.f32, shape=world_shape)
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

        self.boundary_power = ti.field(ti.f64, shape=())
        self.boundary_heat_input = ti.field(ti.f64, shape=())
        self.boundary_heat_input_j_m = self.boundary_heat_input
        self.phase_aperture_volume_residual_m2 = _DerivedField(
            self._read_phase_aperture_volume_residual_m2,
            lambda: self._phase_volume_after[None] - self._phase_volume_before[None],
        )
        self.phase_aperture_energy_residual_j_m = _DerivedField(
            self._read_phase_aperture_energy_residual_j_m,
            lambda: self._phase_energy_after[None] - self._phase_energy_before[None],
        )
        self.phase_aperture_capacity_margin_m2 = _DerivedField(
            self._read_phase_aperture_capacity_margin_m2,
            lambda: self._phase_full_capacity[None] - self._phase_volume_after[None],
        )
        # Legacy output names now audit the single combined pose/phase
        # remap, rather than a separate swept-coverage operation.
        self.ale_water_volume_residual_m2 = self.phase_aperture_volume_residual_m2
        self.ale_water_energy_residual_j_m = self.phase_aperture_energy_residual_j_m
        # A material cell normally injects melt into its paired interface
        # water cell.  If that local topology disappears within a thermal
        # substep, all four extensive sources enter this conservative
        # fallback pool instead of being silently discarded.
        self._unassigned_melt_mass = ti.field(ti.f64, shape=())
        self._unassigned_melt_energy = ti.field(ti.f64, shape=())
        self._melt_fallback_free_weight = ti.field(ti.f64, shape=())
        self._melt_fallback_wet_weight = ti.field(ti.f64, shape=())
        self._interval_body_melt_mass = ti.field(ti.f64, shape=())
        self._interval_water_melt_mass = ti.field(ti.f64, shape=())
        self.melt_injection_mass_residual_kg_m = _DerivedField(
            self._read_melt_injection_mass_residual_kg_m,
            lambda: self._interval_water_melt_mass[None]
            - self._interval_body_melt_mass[None],
        )
        self._solid_body_mass_sum = ti.field(ti.f64, shape=())
        self._water_mass_sum = ti.field(ti.f64, shape=())
        self._body_sensible_sum = ti.field(ti.f64, shape=())
        self._water_sensible_sum = ti.field(ti.f64, shape=())
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
        self._reset_diagnostics()

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
        # One remap handles both the moving sharp mask and phase aperture.
        # A preceding swept-coverage remap would count the displacement
        # twice and mix the cold interface water into remote receivers.
        self.synchronize_water_aperture(water_phase, wall_mask, solid_mask)
        sub_dt = dt / substeps
        for substep_index in range(substeps):
            self._compute_water_advection_flux_x(
                velocity, water_phase, wall_mask, solid_mask
            )
            self._compute_water_advection_flux_y(
                velocity, water_phase, wall_mask, solid_mask
            )
            self._update_water_advection(sub_dt, wall_mask)
            # The upwind volume flux is conservative globally, while an
            # individual partial cell can temporarily outrun its current
            # phase aperture.  Reconcile after every Courant substep so
            # the next face flux always sees an admissible capacity.
            if substep_index + 1 < substeps:
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
        rasterize_world_callback: Any | None = None,
    ) -> int:
        """Advance conduction, wall heat, phase change, and melt injection."""

        self.body_reference_origin = body_reference_origin
        self.body_angle = body_angle
        dt = _positive("time_step_s", time_step_s)
        substeps = self.required_diffusion_substeps(dt)
        sub_dt = dt / substeps
        self._reset_interval_sources()
        for substep_index in range(substeps):

            self._accumulate_body_conduction(sub_dt)
            self._reset_boundary_power()
            self._compute_water_conduction_flux_x(water_phase, wall_mask, solid_mask)
            self._compute_water_conduction_flux_y(water_phase, wall_mask, solid_mask)
            self._update_water_conduction(sub_dt, wall_mask)
            self._accumulate_boundary_heat(sub_dt)

            self._exchange_interface_heat(sub_dt, water_phase, wall_mask, solid_mask)
            self._apply_body_heat_and_phase_change()
            self._freeze_melt_injection_eligibility(water_phase, wall_mask, solid_mask)
            self._inject_melt_water(
                body_reference_origin,
                body_angle,
            )
            self._distribute_unassigned_melt(water_phase, wall_mask, solid_mask)
            # Restore the per-cell aperture bound before the next heat
            # substep; this keeps V and S admissible even when one
            # coupling interval requires N_sub > 1.
            self.synchronize_water_aperture(water_phase, wall_mask, solid_mask)
            if substep_index + 1 < substeps:
                # A diffusion interval can melt enough material to alter
                # the continuous contact aperture before its next
                # substep.  Refresh the material-to-world weights at the
                # fixed SLOW pose; the parent solver rebuilds the sharp
                # LBM mask once after the complete interval.

                if rasterize_world_callback is not None:
                    rasterize_world_callback(resolve_contact=False)

        if rasterize_world_callback is not None:
            rasterize_world_callback(resolve_contact=False)
        self.steps += substeps
        self.time_s += dt
        return substeps

    def synchronize_water_aperture(
        self,
        water_phase: Any,
        wall_mask: Any,
        solid_mask: Any,
    ) -> None:
        """Conservatively align extensive water state with the LBM aperture.

        The bounded volume target follows ``dx**2 * phi`` and preserves
        the thermal water total, including density-converted melt. Wet
        cells retain their specific sensible energy; newly opened cells
        extrapolate it from nearby water. A bounded global energy
        projection closes the extensive sum without introducing new
        temperature extrema. This avoids a costly geometric flux solve
        at every LBM step, at the expense of nonlocal numerical heat
        redistribution, recorded by ``aperture_energy_correction_abs_j_m``.
        """

        measurement = self._measure_phase_aperture(water_phase, wall_mask, solid_mask)
        total_volume, full_capacity = float(measurement[0]), float(measurement[1])
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
        self._build_phase_aperture_targets(water_phase, wall_mask, solid_mask)
        self._apply_phase_aperture_transfer(water_phase, wall_mask, solid_mask)
        self._finish_phase_aperture_transfer()

    def mass_energy_totals(self) -> MovingBodyThermalTotals:
        """Synchronously reduce all extensive moving-body thermal fields."""

        self._reduce_totals()
        initial_body = self._initial_body_cell_mass * self.body_nx * self.body_ny
        solid_body = float(self._solid_body_mass_sum[None])
        melted = initial_body - solid_body
        water_mass = float(self._water_mass_sum[None])
        body_sensible = float(self._body_sensible_sum[None])
        water_sensible = float(self._water_sensible_sum[None])
        latent = self._latent_heat_j_kg * melted
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

    def total_enthalpy_j_m(self, wall_mask: Any | None = None) -> float:
        """Return conserved body/water energy per unit out-of-plane depth."""

        del wall_mask
        return self.mass_energy_totals().total_energy_j_m

    @ti.kernel
    def _initialize_body_state(self, cell_mass: ti.f64):
        specific_energy = ti.static(
            self._ice_specific_heat
            * (self.config.initial_ice_temperature_c - self._melting_temperature_c)
        )
        for i, j in self.body_solid_mass:
            self.body_solid_mass[i, j] = cell_mass
            self.body_sensible_energy[i, j] = cell_mass * specific_energy

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
            self.water_volume_m2[i, j] = volume
            self.water_sensible_energy[i, j] = (
                rho_cp * volume * (initial_temperature - melting)
            )

    @ti.kernel
    def _reset_diagnostics(self):
        self.boundary_power[None] = 0.0
        self.boundary_heat_input[None] = 0.0
        self.aperture_energy_correction_abs_j_m[None] = 0.0

    @ti.kernel
    def _reset_interval_sources(self):
        self._interval_body_melt_mass[None] = 0.0
        self._interval_water_melt_mass[None] = 0.0

    @ti.kernel
    def _reset_boundary_power(self):
        self.boundary_power[None] = 0.0

    @ti.kernel
    def _measure_aperture_partial_sums(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        """Reduce thermal totals and the legal water-side capacities."""

        self._phase_volume_before[None] = 0.0
        self._phase_energy_before[None] = 0.0
        self._phase_base_capacity[None] = 0.0
        self._phase_full_capacity[None] = 0.0
        self._phase_specific_energy_min[None] = 1.0e30
        self._phase_specific_energy_max[None] = -1.0e30
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        ti.loop_config(block_dim=128)
        for flat_index in range(((self.nx * self.ny + 127) // 128) * 128):
            thread_phase_volume_before = ti.cast(0.0, ti.f64)
            thread_phase_energy_before = ti.cast(0.0, ti.f64)
            thread_phase_specific_energy_min = ti.cast(1.0e30, ti.f64)
            thread_phase_specific_energy_max = ti.cast(-1.0e30, ti.f64)
            thread_phase_base_capacity = ti.cast(0.0, ti.f64)
            thread_phase_full_capacity = ti.cast(0.0, ti.f64)
            if flat_index < self.nx * self.ny:
                i, j = flat_index // self.ny, flat_index % self.ny
                volume = self.water_volume_m2[i, j]
                thread_phase_volume_before += volume
                thread_phase_energy_before += self.water_sensible_energy[i, j]
                if volume > 1.0e-30:
                    specific = self.water_sensible_energy[i, j] / volume
                    thread_phase_specific_energy_min = ti.min(
                        thread_phase_specific_energy_min, specific
                    )
                    thread_phase_specific_energy_max = ti.max(
                        thread_phase_specific_energy_max, specific
                    )
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                if (
                    wall_mask[i, j] == 0
                    and solid_mask[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                ):
                    thread_phase_base_capacity += area * ti.cast(phase, ti.f64)
                    thread_phase_full_capacity += area
            warp_phase_volume_before = _warp_reduce_f64(
                thread_phase_volume_before, "sum"
            )
            warp_phase_energy_before = _warp_reduce_f64(
                thread_phase_energy_before, "sum"
            )
            warp_phase_specific_energy_min = _warp_reduce_f64(
                thread_phase_specific_energy_min, "min"
            )
            warp_phase_specific_energy_max = _warp_reduce_f64(
                thread_phase_specific_energy_max, "max"
            )
            warp_phase_base_capacity = _warp_reduce_f64(
                thread_phase_base_capacity, "sum"
            )
            warp_phase_full_capacity = _warp_reduce_f64(
                thread_phase_full_capacity, "sum"
            )
            if flat_index % 32 == 0:
                self._aperture_reduction_partials[flat_index // 32, 0] = (
                    warp_phase_volume_before
                )
                self._aperture_reduction_partials[flat_index // 32, 1] = (
                    warp_phase_energy_before
                )
                self._aperture_reduction_partials[flat_index // 32, 2] = (
                    warp_phase_specific_energy_min
                )
                self._aperture_reduction_partials[flat_index // 32, 3] = (
                    warp_phase_specific_energy_max
                )
                self._aperture_reduction_partials[flat_index // 32, 4] = (
                    warp_phase_base_capacity
                )
                self._aperture_reduction_partials[flat_index // 32, 5] = (
                    warp_phase_full_capacity
                )

    @ti.kernel
    def _build_phase_aperture_targets(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        """Reconstruct local water temperature on the new volume target."""

        area = ti.cast(ti.static(self._cell_area), ti.f64)
        total = ti.max(0.0, self._phase_volume_before[None])
        base = self._phase_base_capacity[None]
        full = self._phase_full_capacity[None]
        self._phase_reconstructed_energy[None] = 0.0
        self._phase_cooling_capacity[None] = 0.0
        self._phase_heating_capacity[None] = 0.0
        low = self._phase_specific_energy_min[None]
        high = self._phase_specific_energy_max[None]
        base_scale = ti.cast(0.0, ti.f64)
        spare_scale = ti.cast(0.0, ti.f64)
        if total <= base and base > 1.0e-30:
            base_scale = total / base
        elif total > base:
            base_scale = 1.0
            if full > base + 1.0e-30:
                spare_scale = ti.min(1.0, (total - base) / (full - base))
        ti.loop_config(block_dim=128)
        for flat_index in range(((self.nx * self.ny + 127) // 128) * 128):
            thread_phase_reconstructed_energy = ti.cast(0.0, ti.f64)
            thread_phase_cooling_capacity = ti.cast(0.0, ti.f64)
            thread_phase_heating_capacity = ti.cast(0.0, ti.f64)
            if flat_index < self.nx * self.ny:
                i, j = flat_index // self.ny, flat_index % self.ny
                phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
                target = ti.cast(0.0, ti.f64)
                if (
                    wall_mask[i, j] == 0
                    and solid_mask[i, j] == 0
                    and phase > ti.static(self._water_phase_cutoff)
                ):
                    phase64 = ti.cast(phase, ti.f64)
                    target = area * (
                        base_scale * phase64 + spare_scale * (1.0 - phase64)
                    )
                target = ti.min(area, ti.max(0.0, target))
                volume = self.water_volume_m2[i, j]
                energy = ti.cast(0.0, ti.f64)
                if target > 0.0:
                    specific = ti.cast(0.0, ti.f64)
                    if volume > 1.0e-30:
                        specific = self.water_sensible_energy[i, j] / volume
                    else:
                        # Frozen source arrays: newly exposed nodes must not
                        # read other nodes being refilled in this GPU launch.
                        # Prefer the first wet ring, including covered donors
                        # which still carry the outgoing local water state.
                        weight = ti.cast(0.0, ti.f64)
                        for radius in ti.static(range(1, 4)):
                            if weight <= 1.0e-30:
                                for di, dj in ti.ndrange(
                                    (-radius, radius + 1), (-radius, radius + 1)
                                ):
                                    ni, nj = i + di, j + dj
                                    if (
                                        ti.max(ti.abs(di), ti.abs(dj)) == radius
                                        and 0 <= ni < ti.static(self.nx)
                                        and 0 <= nj < ti.static(self.ny)
                                        and wall_mask[ni, nj] == 0
                                    ):
                                        neighbor_volume = self.water_volume_m2[ni, nj]
                                        if neighbor_volume > 1.0e-30:
                                            distance2 = ti.cast(
                                                di * di + dj * dj, ti.f64
                                            )
                                            specific += (
                                                self.water_sensible_energy[ni, nj]
                                                / distance2
                                            )
                                            weight += neighbor_volume / distance2
                        if weight > 1.0e-30:
                            specific /= weight
                        elif total > 1.0e-30:
                            # Only disconnected/new components lack any
                            # local water stencil. Never use the configured
                            # initial bath or air temperature as a heat source.
                            specific = self._phase_energy_before[None] / total
                    specific = ti.min(high, ti.max(low, specific))
                    energy = target * specific
                    thread_phase_cooling_capacity += target * (specific - low)
                    thread_phase_heating_capacity += target * (high - specific)
                self._phase_target_energy[i, j] = energy
                thread_phase_reconstructed_energy += energy
            warp_phase_reconstructed_energy = _warp_reduce_f64(
                thread_phase_reconstructed_energy, "sum"
            )
            warp_phase_cooling_capacity = _warp_reduce_f64(
                thread_phase_cooling_capacity, "sum"
            )
            warp_phase_heating_capacity = _warp_reduce_f64(
                thread_phase_heating_capacity, "sum"
            )
            if flat_index % 32 == 0:
                ti.atomic_add(
                    self._phase_reconstructed_energy[None],
                    warp_phase_reconstructed_energy,
                )
                ti.atomic_add(
                    self._phase_cooling_capacity[None], warp_phase_cooling_capacity
                )
                ti.atomic_add(
                    self._phase_heating_capacity[None], warp_phase_heating_capacity
                )

    @ti.kernel
    def _apply_phase_aperture_transfer(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        correction = (
            self._phase_energy_before[None] - self._phase_reconstructed_energy[None]
        )
        self.aperture_energy_correction_abs_j_m[None] += ti.abs(correction)
        low = self._phase_specific_energy_min[None]
        high = self._phase_specific_energy_max[None]
        cooling = self._phase_cooling_capacity[None]
        heating = self._phase_heating_capacity[None]
        for i, j in self.water_volume_m2:
            target = self._water_aperture_target(
                i, j, water_phase, wall_mask, solid_mask
            )
            energy = self._phase_target_energy[i, j]
            if target > 0.0:
                if correction < 0.0 and cooling > 1.0e-30:
                    energy += ti.max(-1.0, correction / cooling) * ti.max(
                        0.0, energy - low * target
                    )
                elif correction > 0.0 and heating > 1.0e-30:
                    energy += ti.min(1.0, correction / heating) * ti.max(
                        0.0, high * target - energy
                    )
            self.water_volume_m2[i, j] = target
            self.water_sensible_energy[i, j] = energy

    @ti.kernel
    def _finish_phase_aperture_transfer(self):
        self._phase_volume_after[None] = 0.0
        self._phase_energy_after[None] = 0.0
        ti.loop_config(block_dim=128)
        for flat_index in range(((self.nx * self.ny + 127) // 128) * 128):
            thread_phase_volume_after = ti.cast(0.0, ti.f64)
            thread_phase_energy_after = ti.cast(0.0, ti.f64)
            if flat_index < self.nx * self.ny:
                i, j = flat_index // self.ny, flat_index % self.ny
                thread_phase_volume_after += self.water_volume_m2[i, j]
                thread_phase_energy_after += self.water_sensible_energy[i, j]
            warp_phase_volume_after = _warp_reduce_f64(thread_phase_volume_after, "sum")
            warp_phase_energy_after = _warp_reduce_f64(thread_phase_energy_after, "sum")
            if flat_index % 32 == 0:
                ti.atomic_add(self._phase_volume_after[None], warp_phase_volume_after)
                ti.atomic_add(self._phase_energy_after[None], warp_phase_energy_after)

    @ti.kernel
    def _accumulate_body_conduction(self, time_step_s: ti.f64):
        # Mass and sensible energy are immutable in this launch. Each face can
        # therefore be evaluated at both endpoints without a flux buffer or race.
        self._unassigned_melt_mass[None] = 0.0
        self._unassigned_melt_energy[None] = 0.0
        for i, j in self.body_solid_mass:
            self._body_melt_target[i, j] = -1
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
            boundary_inward = ti.cast(0.0, ti.f64)
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
                    boundary_inward = flux
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
                    boundary_inward = -flux
            self._water_energy_flux_x[face_i, j] = flux
            if boundary_inward != 0.0:
                ti.atomic_add(self.boundary_power[None], boundary_inward)

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
            boundary_inward = ti.cast(0.0, ti.f64)
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
                    boundary_inward = flux
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
                    boundary_inward = -flux
            self._water_energy_flux_y[i, face_j] = flux
            if boundary_inward != 0.0:
                ti.atomic_add(self.boundary_power[None], boundary_inward)

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
    def _accumulate_boundary_heat(self, time_step_s: ti.f64):
        self.boundary_heat_input[None] += time_step_s * self.boundary_power[None]

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
        and melt targets use atomics and are consumed by the next launch.
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
                requests = ti.Vector.zero(ti.f64, 20)
                targets = ti.Vector.zero(ti.i32, 20)
                requested_heat = ti.cast(0.0, ti.f64)
                water_aperture = ti.min(
                    1.0, self.water_volume_m2[i, j] / self._cell_area
                )
                water_temperature = self._water_temperature_at(i, j)
                for face in ti.static(range(5)):
                    offset = ti.static(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))[face])
                    world_i, world_j = i + offset[0], j + offset[1]
                    if 0 <= world_i < self.nx and 0 <= world_j < self.ny:
                        coverage = ti.cast(
                            self.world_body_solid_fraction[world_i, world_j], ti.f64
                        )
                        if coverage > 0.0:
                            contact = ti.min(
                                water_aperture,
                                ti.min(1.0, coverage / self._solid_fraction_threshold),
                            )
                            grid = self._body_grid_coordinates(world_i, world_j)
                            base_i, base_j = ti.cast(ti.floor(grid.x), ti.i32), ti.cast(
                                ti.floor(grid.y), ti.i32
                            )
                            fraction_x, fraction_y = ti.cast(
                                grid.x - base_i, ti.f64
                            ), ti.cast(grid.y - base_j, ti.f64)
                            for offset_i, offset_j in ti.static(ti.ndrange(2, 2)):
                                body_i, body_j = base_i + offset_i, base_j + offset_j
                                if (
                                    0 <= body_i < self.body_nx
                                    and 0 <= body_j < self.body_ny
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
                                                self._body_fraction_at(body_i, body_j),
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
                                        slot = ti.static(
                                            face * 4 + offset_i * 2 + offset_j
                                        )
                                        requests[slot] = heat
                                        targets[slot] = body_i * self.body_ny + body_j
                                        requested_heat += heat
                scale = ti.cast(0.0, ti.f64)
                if requested_heat > 0.0:
                    scale = ti.min(
                        1.0,
                        ti.max(0.0, self.water_sensible_energy[i, j]) / requested_heat,
                    )
                transferred_heat = ti.cast(0.0, ti.f64)
                for slot in ti.static(range(20)):
                    heat = scale * requests[slot]
                    if heat > 0.0:
                        body_i, body_j = (
                            targets[slot] // self.body_ny,
                            targets[slot] % self.body_ny,
                        )
                        ti.atomic_add(self._body_heat_delta[body_i, body_j], heat)
                        ti.atomic_max(
                            self._body_melt_target[body_i, body_j], i * self.ny + j
                        )
                        transferred_heat += heat
                self.water_sensible_energy[i, j] -= transferred_heat

    @ti.kernel
    def _apply_body_heat_and_phase_change(self):
        latent = ti.cast(ti.static(self._latent_heat_j_kg), ti.f64)
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
            self.body_solid_mass[i, j] = solid_mass
            self.body_sensible_energy[i, j] = sensible
            self._body_melt_mass_step[i, j] = melt_mass

            self._body_melt_sensible_step[i, j] = melt_sensible
            if melt_mass > 0.0:
                ti.atomic_add(self._interval_body_melt_mass[None], melt_mass)

    @ti.kernel
    def _freeze_melt_injection_eligibility(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        for i, j in self._melt_injection_eligible:
            eligible = (
                wall_mask[i, j] == 0
                and solid_mask[i, j] == 0
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
                local_x = ti.cast(body_i, ti.f64) + 0.5 - ti.static(0.5 * self.body_nx)
                local_y = ti.cast(body_j, ti.f64) + 0.5 - ti.static(0.5 * self.body_ny)
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
                volume = melt_mass / self._water_density_kg_m3
                energy = self._body_melt_sensible_step[body_i, body_j]
                if target_i >= 0 and target_j >= 0:
                    ti.atomic_add(self.water_volume_m2[target_i, target_j], volume)
                    ti.atomic_add(
                        self.water_sensible_energy[target_i, target_j], energy
                    )
                    ti.atomic_add(self._interval_water_melt_mass[None], melt_mass)
                else:
                    ti.atomic_add(self._unassigned_melt_mass[None], melt_mass)
                    ti.atomic_add(self._unassigned_melt_energy[None], energy)

    @ti.kernel
    def _measure_melt_fallback_weights(
        self,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        self._melt_fallback_free_weight[None] = 0.0
        self._melt_fallback_wet_weight[None] = 0.0
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        for i, j in self.water_volume_m2:
            phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
            if (
                wall_mask[i, j] == 0
                and solid_mask[i, j] == 0
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
        wall_mask: ti.template(),
        solid_mask: ti.template(),
        use_free_capacity: ti.i32,
    ):
        area = ti.cast(ti.static(self._cell_area), ti.f64)
        denominator = self._melt_fallback_wet_weight[None]
        if use_free_capacity != 0:
            denominator = self._melt_fallback_free_weight[None]
        mass = self._unassigned_melt_mass[None]
        volume = self._unassigned_melt_mass[None] / self._water_density_kg_m3
        energy = self._unassigned_melt_energy[None]
        if denominator > 1.0e-30 and mass > 0.0:
            self._interval_water_melt_mass[None] += mass
        for i, j in self.water_volume_m2:
            phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
            eligible = (
                wall_mask[i, j] == 0
                and solid_mask[i, j] == 0
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
                added_volume = fraction * volume
                added_energy = fraction * energy
                self.water_volume_m2[i, j] += added_volume
                self.water_sensible_energy[i, j] += added_energy

    def _distribute_unassigned_melt(
        self, water_phase: Any, wall_mask: Any, solid_mask: Any
    ) -> None:
        """Conservatively inject melt whose local interface target vanished."""

        unassigned_mass = float(self._unassigned_melt_mass[None])
        if unassigned_mass <= 0.0:
            return
        self._measure_melt_fallback_weights(water_phase, wall_mask, solid_mask)
        free_weight = float(self._melt_fallback_free_weight[None])
        wet_weight = float(self._melt_fallback_wet_weight[None])
        if free_weight > 1.0e-30:
            self._apply_melt_fallback(water_phase, wall_mask, solid_mask, 1)
        elif wet_weight > 1.0e-30:
            # The following aperture projection will either recover spare
            # capacity from other thermal donors or report infeasibility.
            self._apply_melt_fallback(water_phase, wall_mask, solid_mask, 0)
        else:
            raise RuntimeError(
                "melt water has no legal LBM water cell for conservative "
                "mass and energy injection"
            )

    @ti.kernel
    def _reduce_totals(self):
        self._solid_body_mass_sum[None] = 0.0
        self._water_mass_sum[None] = 0.0
        self._body_sensible_sum[None] = 0.0
        self._water_sensible_sum[None] = 0.0
        rho_water = ti.static(self._water_density_kg_m3)
        for i, j in self.body_solid_mass:
            ti.atomic_add(self._solid_body_mass_sum[None], self.body_solid_mass[i, j])
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
        if volume > 1.0e-30:
            temperature = self._melting_temperature_c + self.water_sensible_energy[
                i, j
            ] / (self._water_density_kg_m3 * self._water_specific_heat * volume)
        return temperature

    @ti.func
    def _body_grid_coordinates(self, i, j):
        origin = self.body_reference_origin[None]
        angle = self.body_angle[None]
        cosine, sine = ti.cos(angle), ti.sin(angle)
        relative_x = ti.cast(i, ti.f32) + 0.5 - origin.x
        relative_y = ti.cast(j, ti.f32) + 0.5 - origin.y
        return ti.Vector(
            [
                cosine * relative_x + sine * relative_y + 0.5 * self.body_nx - 0.5,
                -sine * relative_x + cosine * relative_y + 0.5 * self.body_ny - 0.5,
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
        total = ti.max(0.0, self._phase_volume_before[None])
        base, full = self._phase_base_capacity[None], self._phase_full_capacity[None]
        base_scale, spare_scale = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
        if total <= base and base > 1.0e-30:
            base_scale = total / base
        elif total > base:
            base_scale = 1.0
            if full > base + 1.0e-30:
                spare_scale = ti.min(1.0, (total - base) / (full - base))
        target = ti.cast(0.0, ti.f64)
        phase = ti.min(1.0, ti.max(0.0, water_phase[i, j]))
        if (
            wall_mask[i, j] == 0
            and solid_mask[i, j] == 0
            and phase > self._water_phase_cutoff
        ):
            phase64 = ti.cast(phase, ti.f64)
            target = self._cell_area * (
                base_scale * phase64 + spare_scale * (1.0 - phase64)
            )
        return ti.min(self._cell_area, ti.max(0.0, target))

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
        wet = volume > 1.0e-30
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
        wet = volume > 1.0e-30
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
        liquid = np.ones(volume.shape, dtype=np.float32)
        material = wet.copy()
        origin = np.asarray(self.body_reference_origin[None], dtype=np.float32)
        angle = np.float32(self.body_angle[None])
        cosine, sine = np.cos(angle), np.sin(angle)
        x, y = np.meshgrid(
            np.arange(self.nx, dtype=np.float32) + 0.5 - origin[0],
            np.arange(self.ny, dtype=np.float32) + 0.5 - origin[1],
            indexing="ij",
        )
        body_i = np.floor(cosine * x + sine * y + 0.5 * self.body_nx).astype(np.int32)
        body_j = np.floor(-sine * x + cosine * y + 0.5 * self.body_ny).astype(np.int32)
        ice = (
            (
                self.world_body_solid_fraction.to_numpy()
                >= self._solid_fraction_threshold
            )
            & (body_i >= 0)
            & (body_i < self.body_nx)
            & (body_j >= 0)
            & (body_j < self.body_ny)
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
        enthalpy[ice] = (
            selected_energy
            + np.maximum(0.0, self._initial_body_cell_mass - selected_mass)
            * self._latent_heat_j_kg
        ) / self._cell_area
        liquid[ice] = 1.0 - self._body_fraction_numpy()[body_i[ice], body_j[ice]]
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

    @ti.func
    def _read_phase_aperture_volume_residual_m2(self, index):
        return self._phase_volume_after[None] - self._phase_volume_before[None]

    @ti.func
    def _read_phase_aperture_energy_residual_j_m(self, index):
        return self._phase_energy_after[None] - self._phase_energy_before[None]

    @ti.func
    def _read_phase_aperture_capacity_margin_m2(self, index):
        return self._phase_full_capacity[None] - self._phase_volume_after[None]

    @ti.func
    def _read_melt_injection_mass_residual_kg_m(self, index):
        return (
            self._interval_water_melt_mass[None] - self._interval_body_melt_mass[None]
        )

    @ti.func
    def _read_aperture_energy_correction_j_m(self, index):
        return self._phase_energy_before[None] - self._phase_reconstructed_energy[None]

    def _measure_phase_aperture(self, water_phase, wall_mask, solid_mask):
        self._measure_aperture_partial_sums(water_phase, wall_mask, solid_mask)
        return self._finish_aperture_measurement()

    @ti.kernel
    def _finish_aperture_measurement(self) -> ti.types.vector(2, ti.f64):
        ti.loop_config(block_dim=128)
        for partial_index in range(((self._aperture_partial_count + 127) // 128) * 128):
            volume, energy = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
            minimum, maximum = ti.cast(1.0e30, ti.f64), ti.cast(-1.0e30, ti.f64)
            base, capacity = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
            if partial_index < self._aperture_partial_count:
                volume = self._aperture_reduction_partials[partial_index, 0]
                energy = self._aperture_reduction_partials[partial_index, 1]
                minimum = self._aperture_reduction_partials[partial_index, 2]
                maximum = self._aperture_reduction_partials[partial_index, 3]
                base = self._aperture_reduction_partials[partial_index, 4]
                capacity = self._aperture_reduction_partials[partial_index, 5]
            volume = _warp_reduce_f64(volume, "sum")
            energy = _warp_reduce_f64(energy, "sum")
            minimum = _warp_reduce_f64(minimum, "min")
            maximum = _warp_reduce_f64(maximum, "max")
            base = _warp_reduce_f64(base, "sum")
            capacity = _warp_reduce_f64(capacity, "sum")
            if partial_index % 32 == 0:
                ti.atomic_add(self._phase_volume_before[None], volume)
                ti.atomic_add(self._phase_energy_before[None], energy)
                ti.atomic_min(self._phase_specific_energy_min[None], minimum)
                ti.atomic_max(self._phase_specific_energy_max[None], maximum)
                ti.atomic_add(self._phase_base_capacity[None], base)
                ti.atomic_add(self._phase_full_capacity[None], capacity)
        return ti.Vector(
            [self._phase_volume_before[None], self._phase_full_capacity[None]],
            dt=ti.f64,
        )


@ti.data_oriented
class IceFlow2D:
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
        self._water_viscosity_lattice = float(
            config.viscosity_water * u_ref_lattice / (config.dx * u_ref)
        )
        self._air_viscosity_lattice = float(
            config.viscosity_air * u_ref_lattice / (config.dx * u_ref)
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
        configured_interface_tau = config.air_interface_relaxation_time
        self._air_interface_relaxation_time = (
            0.5 if configured_interface_tau is None else float(configured_interface_tau)
        )
        self._body_half_width = 0.5 * float(config.ice_width)
        self._body_half_height = 0.5 * float(config.ice_height)
        self._initial_body_mass_lattice = float(config.ice_mass_lattice)
        self._initial_body_inertia_lattice = float(config.ice_inertia_lattice)
        cutoff = float(config.volume_projection_interface_cutoff)
        profile_radius = (
            0.25 * float(config.interface_width) * math.log((1.0 - cutoff) / cutoff)
        )
        # Ping-pong dilation finishes in the primary mask after an even
        # number of passes.  The radius covers the equilibrium logistic
        # profile down to the configured bulk cutoff.
        self._volume_projection_band_radius = max(2, int(math.ceil(profile_radius)))
        if self._volume_projection_band_radius % 2:
            self._volume_projection_band_radius += 1

        shape_xy = (self.nx, self.ny)

        # The momentum distribution carries pressure and momentum:
        # sum(c*f)=rho(phi)*u and its second moment contains pressure, while
        # sum(f) is deliberately not material density.  h is the conservative
        # Allen--Cahn phase distribution.
        self.momentum_populations = self._allocate_populations()
        self.momentum_stream_buffer = self._allocate_populations()
        self.phase_populations = self._allocate_populations()
        # After phase streaming, phase_stream_buffer is dead until the next phase
        # collision.  Its q=0 and q=1 planes are reused as the two projection
        # masks, avoiding dedicated device arrays without aliasing a live
        # value or reading and writing the same plane in one kernel.
        self.phase_stream_buffer = self._allocate_populations()
        self.water_phase = ti.field(ti.f32, shape=shape_xy)
        self.momentum_velocity_lattice = ti.Vector.field(2, ti.f32, shape=shape_xy)
        # Retained across streaming because pressure reconstruction uses the
        # half-force velocity at the next time level.
        self.fluid_acceleration_lattice = ti.Vector.field(2, ti.f32, shape=shape_xy)
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

        # Rigid state and the only load accumulator consumed by integration.
        self.body_center = _DerivedField(
            self._body_center_at,
            self._body_center_numpy,
            host_writer=self._set_body_center_host,
        )
        self.body_velocity = ti.Vector.field(2, ti.f32, shape=())
        self.body_angle = ti.field(ti.f32, shape=())
        self.body_angular_velocity = ti.field(ti.f32, shape=())
        # Dynamic mass properties let the material-frame thermal path erode
        # the rigid remnant.
        self.body_initial_mass_lattice = _DerivedField(
            self._read_body_initial_mass_lattice,
            lambda: self._initial_body_mass_lattice,
        )
        self.body_mass_lattice = ti.field(ti.f64, shape=())
        self.body_inertia_lattice = ti.field(ti.f64, shape=())
        self.body_local_center_of_mass = ti.Vector.field(2, ti.f64, shape=())
        self.body_reference_origin = ti.Vector.field(2, ti.f32, shape=())
        self.body_active = _DerivedField(
            self._read_body_active, lambda: int(self.body_inertia_lattice[None] > 0.0)
        )
        self.cumulative_melted_mass_lattice = _DerivedField(
            self._read_cumulative_melted_mass_lattice,
            lambda: self._initial_body_mass_lattice - self.body_mass_lattice[None],
        )
        self.cumulative_melted_momentum_lattice = ti.Vector.field(2, ti.f64, shape=())
        self.cumulative_fluid_melt_momentum_lattice = _DerivedField(
            self._read_cumulative_fluid_melt_momentum_lattice,
            lambda: self.cumulative_fluid_melt_carrier_momentum_lattice[None]
            + self.cumulative_fluid_melt_correction_momentum_lattice[None],
        )
        self.cumulative_fluid_melt_carrier_momentum_lattice = ti.Vector.field(
            2, ti.f64, shape=()
        )
        self.cumulative_fluid_melt_correction_momentum_lattice = ti.Vector.field(
            2, ti.f64, shape=()
        )
        self.melt_momentum_residual_lattice = _DerivedField(
            self._read_melt_momentum_residual_lattice,
            lambda: self.cumulative_fluid_melt_momentum_lattice[None]
            - self.cumulative_melted_momentum_lattice[None],
        )
        self.cumulative_melted_angular_momentum_lattice = ti.field(ti.f64, shape=())
        self.cumulative_fluid_melt_angular_momentum_lattice = _DerivedField(
            self._read_cumulative_fluid_melt_angular_momentum_lattice,
            lambda: self.cumulative_fluid_melt_carrier_angular_momentum_lattice[None]
            + self.cumulative_fluid_melt_correction_angular_momentum_lattice[None],
        )
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice = ti.field(
            ti.f64, shape=()
        )
        self.cumulative_fluid_melt_correction_angular_momentum_lattice = ti.field(
            ti.f64, shape=()
        )
        self.melt_angular_momentum_residual_lattice = _DerivedField(
            self._read_melt_angular_momentum_residual_lattice,
            lambda: self.cumulative_fluid_melt_angular_momentum_lattice[None]
            - self.cumulative_melted_angular_momentum_lattice[None],
        )
        self.ale_water_residual_cells = _DerivedField(
            self._read_ale_water_residual_cells,
            lambda: self.thermal.phase_aperture_volume_residual_m2[None]
            / (self.config.dx * self.config.dx),
        )
        self.ale_energy_residual_j_m = _DerivedField(
            self._read_ale_energy_residual_j_m,
            lambda: self.thermal.phase_aperture_energy_residual_j_m[None],
        )
        self.phase_aperture_water_residual_cells = _DerivedField(
            self._read_phase_aperture_water_residual_cells,
            lambda: self.thermal.phase_aperture_volume_residual_m2[None]
            / (self.config.dx * self.config.dx),
        )
        self.phase_aperture_energy_residual_j_m = _DerivedField(
            self._read_phase_aperture_energy_residual_j_m,
            lambda: self.thermal.phase_aperture_energy_residual_j_m[None],
        )
        self.phase_aperture_capacity_margin_cells = _DerivedField(
            self._read_phase_aperture_capacity_margin_cells,
            lambda: self.thermal.phase_aperture_capacity_margin_m2[None]
            / (self.config.dx * self.config.dx),
        )
        self._body_reduced_mass = ti.field(ti.f64, shape=())
        self._body_reduced_first_moment = ti.Vector.field(2, ti.f64, shape=())
        self._body_reduced_inertia_origin = ti.field(ti.f64, shape=())
        # End-to-end first-moment audit for one moving thermal interval.  The
        # provisional change includes sharp-node refill and phase-density
        # lifting; a zero-mass D2Q9 correction then closes it to the momentum
        # lost by the eroding rigid body.
        self._thermal_fluid_momentum_before = ti.Vector.field(2, ti.f64, shape=())
        self._thermal_fluid_momentum_provisional = ti.Vector.field(2, ti.f64, shape=())
        self._thermal_fluid_momentum_after = ti.Vector.field(2, ti.f64, shape=())
        self._thermal_body_momentum_before = ti.Vector.field(2, ti.f64, shape=())
        self._thermal_fluid_angular_momentum_before = ti.field(ti.f64, shape=())
        self._thermal_fluid_angular_momentum_provisional = ti.field(ti.f64, shape=())
        self._thermal_fluid_angular_momentum_after = ti.field(ti.f64, shape=())
        self._thermal_body_angular_momentum_before = ti.field(ti.f64, shape=())
        self._melt_momentum_correction = _DerivedField(
            self._current_melt_momentum_correction,
            lambda: self.cumulative_melted_momentum_lattice[None]
            - self._thermal_body_momentum_before[None]
            - (
                self._thermal_fluid_momentum_after[None]
                - self._thermal_fluid_momentum_before[None]
            ),
        )
        self._melt_angular_momentum_correction = _DerivedField(
            self._current_melt_angular_correction,
            lambda: self.cumulative_melted_angular_momentum_lattice[None]
            - self._thermal_body_angular_momentum_before[None]
            - (
                self._thermal_fluid_angular_momentum_after[None]
                - self._thermal_fluid_angular_momentum_before[None]
            ),
        )
        self._melt_momentum_wet_weight = ti.field(ti.f64, shape=())
        self._melt_momentum_wet_target = ti.field(ti.i32, shape=())
        self._melt_momentum_wet_target_max = ti.field(ti.i32, shape=())
        self._melt_momentum_weight_first_moment = ti.Vector.field(2, ti.f64, shape=())
        self._melt_momentum_weight_second_moment = ti.field(ti.f64, shape=())
        self._melt_momentum_weight_centroid = _DerivedField(
            self._wet_support_centroid, lambda: self._wet_support_centroid_numpy()
        )
        self._melt_momentum_weight_polar_moment = _DerivedField(
            self._wet_support_polar_moment,
            lambda: max(
                0.0,
                float(self._melt_momentum_weight_second_moment[None])
                - float(self._melt_momentum_wet_weight[None])
                * float(
                    np.dot(
                        self._wet_support_centroid_numpy(),
                        self._wet_support_centroid_numpy(),
                    )
                ),
            ),
        )
        # For an eroding body, container contact is reconstructed from the
        # current unclipped sharp world raster rather than the initial rectangle.
        # The four entries are min-x, max-x, min-y and max-y in the world
        # orientation but relative to ``body_reference_origin``.
        self._body_contact_support_extrema = ti.field(ti.f32, shape=4)
        self._body_contact_geometry_active = _DerivedField(
            self._read__body_contact_geometry_active,
            lambda: int(
                self._body_contact_support_extrema[1]
                > self._body_contact_support_extrema[0]
            ),
        )
        # The continuous body coverage is first evaluated without applying
        # the container mask.  Contact reduction consumes this scratch field
        # so the same fraction evaluation can be reused by the world
        # rasterizer; a second evaluation is needed only when projection
        # changes the accepted pose.
        self._body_contact_projection_changed = ti.field(ti.i8, shape=())
        # Cut-link loads are signed sums with strong cancellation.  CUDA is
        # free to order global atomics differently as block scheduling
        # changes, so reduce in f64 and round only at rigid integration.
        self.hydrodynamic_impulse = ti.Vector.field(2, ti.f64, shape=())
        self.hydrodynamic_torque = ti.field(ti.f64, shape=())
        # Water-volume constraint and interface-projection state.  It remains
        # constant in the mechanical model and becomes a density-aware target
        # when thermal phase change is enabled.
        self.water_volume_target = _DerivedField(
            self._water_volume_target_at,
            lambda: self.phase_change_initial_water_volume[None]
            + self._initial_body_mass_lattice
            - self.body_mass_lattice[None],
            host_writer=self._set_water_volume_target_host,
        )
        self.water_volume_current = ti.field(ti.f64, shape=())
        self.water_projection_derivative = ti.field(ti.f64, shape=())

        self._thermal_lbm_steps_pending = 0
        self._moving_melt_momentum_pending = False
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
            water_phase_cutoff=config.volume_projection_interface_cutoff,
        )
        # Share references to the authoritative pose and the one coverage
        # allocation; the thermal component owns no second pose or raster.
        self.thermal.body_reference_origin = self.body_reference_origin
        self.thermal.body_angle = self.body_angle
        self._moving_body_fraction = self.thermal.world_body_solid_fraction
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
        self.phase_change_initial_water_volume = ti.field(ti.f64, shape=())
        self.phase_change_initial_solid_volume = _DerivedField(
            self._read_phase_change_initial_solid_volume,
            lambda: float(self.config.ice_width * self.config.ice_height),
        )
        self.phase_change_current_solid_volume = _DerivedField(
            self._read_phase_change_current_solid_volume,
            lambda: self.body_mass_lattice[None]
            * (self.config.rho_water / self.config.rho_ice),
        )

        self._initialize_body(
            self._initial_body_mass_lattice, self._initial_body_inertia_lattice
        )
        self._initialize_geometry()
        # Seed the material fractions before the first world raster so the
        # canonical SDF and the initial sharp mask are derived from the same
        # geometry.  The full thermal state is initialized later, after the
        # diffuse phase warm-up has reached its final composition.
        self.thermal._initialize_body_state(self.thermal._initial_body_cell_mass)
        self._solve_contact_and_rasterize()
        self._initialize_moving_body_geometry()
        self._initialize_fluid()
        # The no-melting invariant is the sharp initial water volume.  Phase
        # warm-up is a numerical preparation step and must not redefine it.
        self._initialize_water_volume()
        if config.phase_warmup_steps > 0:
            self._warm_start_phase(config.phase_warmup_steps)
        self._recover_initial_fluid_moments()
        self._correct_water_volume()
        # Initialize enthalpy only after the diffuse water/air interface has
        # reached its projected initial state.  Otherwise the first thermal
        # recovery would combine pre-warm-up composition energy with the
        # post-warm-up phase fraction.
        self.thermal.initialize(self.water_phase, self.wall_mask, self.solid_mask)

        self._update_moving_body_mass_properties(initialize=True)
        self._build_hydrostatic_reference()
        self._initialize_hydrostatic_equilibrium()

    def step(self, num_steps=1):
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, (int, np.integer))
            or num_steps < 0
        ):
            raise ValueError("num_steps must be a non-negative integer")
        for _ in range(num_steps):
            self._collide_momentum()
            self._collide_phase()
            self._stream()
            self._update_streamed_macroscopic_fields()
            self._update_pressure()
            self._integrate_rigid_ice()
            # Rasterization and the whole-body wall projection form one
            # geometry transaction.  The fraction pass is shared by contact
            # reduction and the final shared SDF/thermal coverage fields.
            self._solve_contact_and_rasterize()
            self._update_solid_mask()
            self._refill_changed_nodes()
            self._advance_moving_thermal_fast(target_step=self.steps + 1)
            self._thermal_lbm_steps_pending += 1
            if (
                self._thermal_lbm_steps_pending
                >= self.config.thermal.update_interval_lbm_steps
            ):
                self._advance_moving_thermal_slow(
                    self._thermal_lbm_steps_pending,
                )
                self._thermal_lbm_steps_pending = 0
            self._correct_water_volume()
            self._finalize_melt_momentum_coupling()
            self._synchronize_moving_water_aperture()
            self.steps += 1

    @property
    def physical_time_s(self):
        return self.steps * self._time_step_s

    def _advance_moving_thermal_fast(self, *, target_step):
        """Remap the moving aperture and advect water/energy at every LBM step."""

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
        """Commit accumulated conduction, melting, and melt momentum."""

        count = int(lbm_steps)
        if count <= 0:
            raise ValueError("lbm_steps must be positive")
        if self._moving_melt_momentum_pending:
            raise RuntimeError(
                "a moving thermal momentum interval is still pending phase projection"
            )
        # FAST deliberately defers its final aperture reconciliation to the
        # post-projection synchronization on ordinary LBM steps.  SLOW needs
        # an admissible extensive water state before evaluating conduction
        # and interface heat, so reconcile once here only on SLOW steps.
        self.thermal.synchronize_water_aperture(
            self.water_phase, self.wall_mask, self.solid_mask
        )
        self._snapshot_melt_momentum_coupling()
        self.thermal.advance_slow(
            count * self._time_step_s,
            self.water_phase,
            self.wall_mask,
            self.solid_mask,
            self.body_reference_origin,
            self.body_angle,
            rasterize_world_callback=self._solve_contact_and_rasterize,
        )
        self._update_moving_body_mass_properties()
        self._update_solid_mask()
        # A node exposed by erosion is not necessarily liquid: the falling
        # body can melt on its air-facing side as well as below the free
        # surface.  Reuse the schedule-independent moving-boundary
        # extrapolation so the released node inherits the surrounding phase,
        # pressure, and velocity.  The melt volume itself has already been
        # injected conservatively by ``advance_slow`` and the following
        # global projection restores its exact target volume.
        self._refill_changed_nodes()
        self._moving_melt_momentum_pending = True

    def _synchronize_moving_water_aperture(self):
        """Align thermal extensive water after the LBM phase projection."""

        self.thermal.synchronize_water_aperture(
            self.water_phase, self.wall_mask, self.solid_mask
        )

        # Views are current by construction; no device refresh is necessary.

    # Geometry and initialization

    @ti.func
    def _is_fluid_cell(self, i, j):
        return self.wall_mask[i, j] == 0 and self.solid_mask[i, j] == 0

    @ti.kernel
    def _measure_water_outflow_rate(self):
        """Bound the sum of outward face speeds for upwind volume positivity."""
        self.maximum_water_outflow_rate_lattice[None] = 0.0
        for i, j in self.water_phase:
            if (
                self._is_fluid_cell(i, j)
                and self.water_phase[i, j]
                > self.config.volume_projection_interface_cutoff
            ):
                velocity = self._physical_velocity_at(i, j)
                outflow = 0.0
                for offset_i, offset_j in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    neighbor_i, neighbor_j = i + offset_i, j + offset_j
                    if inside_grid(neighbor_i, neighbor_j, self.nx, self.ny):
                        if (
                            self._is_fluid_cell(neighbor_i, neighbor_j)
                            and self.water_phase[neighbor_i, neighbor_j]
                            > self.config.volume_projection_interface_cutoff
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
    def _body_velocity_at(self, point):
        relative = point - self.body_center[None]
        omega = self.body_angular_velocity[None]
        return self.body_velocity[None] + omega * ti.Vector([-relative.y, relative.x])

    @ti.func
    def _cell_point(self, i, j):
        return ti.Vector([ti.cast(i, ti.f32) + 0.5, ti.cast(j, ti.f32) + 0.5])

    @ti.func
    def _phase_neighbor(self, i, j, step_x, step_y, distance):
        value = self.water_phase[i, j]
        ni1 = i + step_x
        nj1 = j + step_y
        if inside_grid(
            ni1, nj1, ti.static(self.nx), ti.static(self.ny)
        ) and self._is_fluid_cell(ni1, nj1):
            value = self.water_phase[ni1, nj1]
            if distance == 2:
                ni2 = i + 2 * step_x
                nj2 = j + 2 * step_y
                if inside_grid(
                    ni2, nj2, ti.static(self.nx), ti.static(self.ny)
                ) and self._is_fluid_cell(ni2, nj2):
                    value = self.water_phase[ni2, nj2]
                else:
                    value = self.water_phase[i, j]
        return value

    # ------------------------------------------------------------------
    # Moving-body world rasterization

    @ti.func
    def _body_local_coordinates(self, point):
        """Transform a world cell centre into material-frame coordinates."""

        center = self.body_reference_origin[None]
        angle = self.body_angle[None]
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
    def _nearest_body_index(self, local):
        body_i = ti.cast(
            ti.floor(local.x + ti.static(0.5 * self.config.ice_width)), ti.i32
        )
        body_j = ti.cast(
            ti.floor(local.y + ti.static(0.5 * self.config.ice_height)), ti.i32
        )
        return ti.Vector([body_i, body_j])

    @ti.func
    def _sample_body_fraction(self, local):
        """Bilinearly sample the eroded material grid without wall clipping."""

        grid_x = local.x + ti.static(0.5 * self.config.ice_width - 0.5)
        grid_y = local.y + ti.static(0.5 * self.config.ice_height - 0.5)
        base_i = ti.cast(ti.floor(grid_x), ti.i32)
        base_j = ti.cast(ti.floor(grid_y), ti.i32)
        fraction = ti.cast(0.0, ti.f32)
        for di, dj in ti.static(ti.ndrange(2, 2)):
            body_i = base_i + di
            body_j = base_j + dj
            if 0 <= body_i < ti.static(
                self.config.ice_width
            ) and 0 <= body_j < ti.static(self.config.ice_height):
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
        """Rasterize unmasked material coverage once for contact and heat."""
        for i, j in self._moving_body_fraction:
            local = self._body_local_coordinates(self._cell_point(i, j))
            self._moving_body_fraction[i, j] = self._sample_body_fraction(local)

    @ti.kernel
    def _commit_moving_body_raster(self):
        """Clip the accepted coverage in place; contact has finished reading it."""
        for i, j in self._moving_body_fraction:
            if self.wall_mask[i, j] != 0:
                self._moving_body_fraction[i, j] = 0.0

    @ti.kernel
    def _initialize_body(self, initial_mass: ti.f64, initial_inertia: ti.f64):
        center_x = ti.static(float(self.config.ice_initial_center[0]))
        center_y = ti.static(float(self.config.ice_initial_center[1]))
        velocity_x = ti.static(float(self.config.ice_initial_velocity[0]))
        velocity_y = ti.static(float(self.config.ice_initial_velocity[1]))
        self.body_velocity[None] = ti.Vector([velocity_x, velocity_y])
        initial_angle = ti.static(float(self.config.ice_initial_angle))
        initial_omega = ti.static(float(self.config.ice_initial_angular_velocity))
        self.body_angle[None] = initial_angle
        self.body_angular_velocity[None] = initial_omega
        self.body_mass_lattice[None] = initial_mass
        self.body_inertia_lattice[None] = initial_inertia
        self.body_local_center_of_mass[None] = ti.Vector([0.0, 0.0])
        self.body_reference_origin[None] = ti.Vector([center_x, center_y])
        self.cumulative_melted_momentum_lattice[None] = ti.Vector([0.0, 0.0])
        self.cumulative_fluid_melt_carrier_momentum_lattice[None] = ti.Vector(
            [0.0, 0.0]
        )
        self.cumulative_fluid_melt_correction_momentum_lattice[None] = ti.Vector(
            [0.0, 0.0]
        )
        self._thermal_fluid_momentum_before[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_momentum_provisional[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_momentum_after[None] = ti.Vector([0.0, 0.0])
        self._thermal_body_momentum_before[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_before[None] = 0.0
        self._thermal_fluid_angular_momentum_provisional[None] = 0.0
        self._thermal_fluid_angular_momentum_after[None] = 0.0
        self._thermal_body_angular_momentum_before[None] = 0.0
        self._melt_momentum_wet_weight[None] = 0.0
        self._melt_momentum_wet_target[None] = -1
        self._melt_momentum_wet_target_max[None] = -1
        self._melt_momentum_weight_first_moment[None] = ti.Vector([0.0, 0.0])
        self._melt_momentum_weight_second_moment[None] = 0.0
        self.cumulative_melted_angular_momentum_lattice[None] = 0.0
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice[None] = 0.0
        self.cumulative_fluid_melt_correction_angular_momentum_lattice[None] = 0.0
        self.hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.hydrodynamic_torque[None] = 0.0
        self._body_contact_projection_changed[None] = ti.cast(0, ti.i8)

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
                velocity = self._body_velocity_at(self._cell_point(i, j))
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
                water_phase = self.water_phase[i, j]
                material_density = mixture_density(water_phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] = rest
                self.pressure_lattice[i, j] = hydrostatic_pressure
                for q in range(D2Q9_DIRECTION_COUNT):
                    # Only p_dyn is carried by the populations.  The frozen
                    # p_H traction remains in the Guo completion.
                    self.momentum_populations[i, j, q] = momentum_equilibrium(
                        q, 0.0, material_density, rest
                    )
                    self.phase_populations[i, j, q] = phase_equilibrium(
                        q, water_phase, rest
                    )
            else:
                boundary_velocity = ti.Vector([0.0, 0.0])
                if self.solid_mask[i, j] == 1:
                    boundary_velocity = self._body_velocity_at(point)
                self.momentum_velocity_lattice[i, j] = boundary_velocity
                self.pressure_lattice[i, j] = 0.0

    @ti.kernel
    def _initialize_moving_body_geometry(self):
        """Seed the sharp mask from the initial rasterized canonical SDF."""

        for i, j in self.solid_mask:
            active_body = self.body_active[None] != 0
            is_solid = (
                active_body
                and self.wall_mask[i, j] == 0
                and self.body_signed_distance_m[i, j] <= 0.0
            )
            self.solid_mask[i, j] = ti.cast(1 if is_solid else 0, ti.i8)
            self.previous_solid_mask[i, j] = self.solid_mask[i, j]

    @ti.kernel
    def _update_solid_mask(self):
        """Commit the already-rasterized SDF to the sharp solid mask.

        Record the previous committed mask and derive the new mask from
        accepted coverage. Candidate/contact and thermal substeps may update
        coverage while the committed topology remains frozen.
        """

        for i, j in self.solid_mask:
            self.previous_solid_mask[i, j] = self.solid_mask[i, j]
            distance = self.body_signed_distance_m[i, j]
            active_body = self.body_active[None] != 0
            is_solid = active_body and self.wall_mask[i, j] == 0 and distance <= 0.0
            self.solid_mask[i, j] = ti.cast(1 if is_solid else 0, ti.i8)

    def _update_moving_body_mass_properties(self, *, initialize: bool = False):
        """Reduce the eroding material grid and update the rigid COM state."""

        self._reduce_moving_body_mass_properties()
        self._apply_moving_body_mass_properties(1 if initialize else 0)

    @ti.kernel
    def _snapshot_melt_momentum_coupling(self):
        """Record fluid/body linear and angular momenta before thermal work."""

        self._thermal_fluid_momentum_before[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_before[None] = 0.0
        self._thermal_body_momentum_before[None] = (
            self.cumulative_melted_momentum_lattice[None]
        )
        self._thermal_body_angular_momentum_before[None] = (
            self.cumulative_melted_angular_momentum_lattice[None]
        )
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(D2Q9_DIRECTION_COUNT):
                    momentum += ti.cast(lattice_direction(q), ti.f64) * ti.cast(
                        self.momentum_populations[i, j, q], ti.f64
                    )
                ti.atomic_add(self._thermal_fluid_momentum_before[None].x, momentum.x)
                ti.atomic_add(self._thermal_fluid_momentum_before[None].y, momentum.y)
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                ti.atomic_add(
                    self._thermal_fluid_angular_momentum_before[None],
                    point.x * momentum.y - point.y * momentum.x,
                )

    @ti.kernel
    def _reduce_provisional_melt_momentum(self):
        """Measure provisional momentum and the wet support used for correction."""
        self._thermal_fluid_momentum_provisional[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_provisional[None] = 0.0
        self._melt_momentum_wet_weight[None] = 0.0
        self._melt_momentum_wet_target[None] = self.nx * self.ny
        self._melt_momentum_wet_target_max[None] = -1
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(D2Q9_DIRECTION_COUNT):
                    momentum += ti.cast(lattice_direction(q), ti.f64) * ti.cast(
                        self.momentum_populations[i, j, q], ti.f64
                    )
                ti.atomic_add(self._thermal_fluid_momentum_provisional[None], momentum)
                ti.atomic_add(
                    self._thermal_fluid_angular_momentum_provisional[None],
                    (ti.cast(i, ti.f64) + 0.5) * momentum.y
                    - (ti.cast(j, ti.f64) + 0.5) * momentum.x,
                )
                phase = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
                if phase > self.config.volume_projection_interface_cutoff:
                    ti.atomic_add(
                        self._melt_momentum_wet_weight[None], ti.cast(phase, ti.f64)
                    )
                    ti.atomic_min(self._melt_momentum_wet_target[None], i * self.ny + j)
                    ti.atomic_max(
                        self._melt_momentum_wet_target_max[None], i * self.ny + j
                    )
        self._thermal_fluid_momentum_after[None] = (
            self._thermal_fluid_momentum_provisional[None]
        )
        self._thermal_fluid_angular_momentum_after[None] = (
            self._thermal_fluid_angular_momentum_provisional[None]
        )

    @ti.kernel
    def _reduce_melt_momentum_weight_geometry(self) -> ti.i32:
        self._melt_momentum_weight_first_moment[None] = ti.Vector([0.0, 0.0])
        self._melt_momentum_weight_second_moment[None] = 0.0
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        for i, j in self.water_phase:
            phase = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
            weight = ti.cast(0.0, ti.f64)
            if self._is_fluid_cell(i, j) and phase > cutoff:
                weight = ti.cast(phase, ti.f64)
            if weight > 0.0:
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                ti.atomic_add(
                    self._melt_momentum_weight_first_moment[None].x,
                    weight * point.x,
                )
                ti.atomic_add(
                    self._melt_momentum_weight_first_moment[None].y,
                    weight * point.y,
                )
                ti.atomic_add(
                    self._melt_momentum_weight_second_moment[None],
                    weight * point.dot(point),
                )
        correction = self._current_melt_momentum_correction(None)
        angular = self._current_melt_angular_correction(None)
        weight = self._melt_momentum_wet_weight[None]
        status = 0
        if weight <= 1.0e-30:
            if correction.norm() > 1.0e-12 or ti.abs(angular) > 1.0e-12:
                status = -1
        else:
            centroid = self._wet_support_centroid(None)
            couple = angular - cross_product_2d(centroid, correction)
            if (
                self._wet_support_polar_moment(None) <= 1.0e-30
                and ti.abs(couple) > 1.0e-12
            ):
                status = -2
            elif correction.norm() > 1.0e-14 or ti.abs(angular) > 1.0e-14:
                status = 1
        return status

    @ti.kernel
    def _apply_melt_momentum_correction(self):
        """Apply a zero-mass D2Q9 lift satisfying three moment constraints."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        # Compute tiny derived vectors inside the parallel task, avoiding
        # cross-task vector temporaries in Taichi 1.7.4 CUDA code generation.
        for i, j in self.water_phase:
            denominator = self._melt_momentum_wet_weight[None]
            correction = self._melt_momentum_correction[None]
            angular_correction = self._melt_angular_momentum_correction[None]
            centroid = self._melt_momentum_weight_centroid[None]
            polar = self._melt_momentum_weight_polar_moment[None]
            rotational_scale = ti.cast(0.0, ti.f64)
            if polar > 1.0e-30:
                translation_angular = (
                    centroid.x * correction.y - centroid.y * correction.x
                )
                rotational_scale = (angular_correction - translation_angular) / polar
            phase = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
            eligible = self._is_fluid_cell(i, j) and phase > cutoff
            weight = ti.cast(0.0, ti.f64)
            if eligible:
                weight = ti.cast(phase, ti.f64)
            if weight > 0.0 and denominator > 1.0e-30:
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                relative = point - centroid
                rotational_direction = ti.Vector([-relative.y, relative.x])
                delta_momentum_f64 = weight * (
                    correction / denominator + rotational_scale * rotational_direction
                )
                delta_momentum = ti.cast(delta_momentum_f64, ti.f32)
                density = mixture_density(phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] += delta_momentum / ti.max(
                    density, 1.0e-12
                )
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = ti.cast(lattice_direction(q), ti.f32)
                    self.momentum_populations[i, j, q] += (
                        3.0 * lattice_weight(q) * direction.dot(delta_momentum)
                    )

    @ti.kernel
    def _reduce_final_melt_momentum(self):
        self._thermal_fluid_momentum_after[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_after[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(D2Q9_DIRECTION_COUNT):
                    momentum += ti.cast(lattice_direction(q), ti.f64) * ti.cast(
                        self.momentum_populations[i, j, q], ti.f64
                    )
                ti.atomic_add(self._thermal_fluid_momentum_after[None].x, momentum.x)
                ti.atomic_add(self._thermal_fluid_momentum_after[None].y, momentum.y)
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                ti.atomic_add(
                    self._thermal_fluid_angular_momentum_after[None],
                    point.x * momentum.y - point.y * momentum.x,
                )

    @ti.kernel
    def _apply_local_melt_momentum_correction(self):
        """Close f32 remainders with a two-cell force/couple pair."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        ny = ti.static(self.ny)
        encoded_1 = self._melt_momentum_wet_target[None]
        encoded_2 = self._melt_momentum_wet_target_max[None]
        if (
            0 <= encoded_1 < ti.static(self.nx * self.ny)
            and 0 <= encoded_2 < ti.static(self.nx * self.ny)
            and encoded_1 != encoded_2
        ):
            i1 = encoded_1 // ny
            j1 = encoded_1 - i1 * ny
            i2 = encoded_2 // ny
            j2 = encoded_2 - i2 * ny
            point_1 = ti.Vector([ti.cast(i1, ti.f64) + 0.5, ti.cast(j1, ti.f64) + 0.5])
            point_2 = ti.Vector([ti.cast(i2, ti.f64) + 0.5, ti.cast(j2, ti.f64) + 0.5])
            correction = self._melt_momentum_correction[None]
            angular_correction = self._melt_angular_momentum_correction[None]
            difference = point_1 - point_2
            midpoint = 0.5 * (point_1 + point_2)
            couple = angular_correction - (
                midpoint.x * correction.y - midpoint.y * correction.x
            )
            difference_norm2 = difference.dot(difference)
            q_couple = (
                couple
                * ti.Vector([-difference.y, difference.x])
                / ti.max(difference_norm2, 1.0e-30)
            )
            delta_1 = 0.5 * correction + q_couple
            delta_2 = 0.5 * correction - q_couple
            for target in ti.static(range(2)):
                i = i1
                j = j1
                delta_momentum_f64 = delta_1
                if ti.static(target == 1):
                    i = i2
                    j = j2
                    delta_momentum_f64 = delta_2
                delta_momentum = ti.cast(delta_momentum_f64, ti.f32)
                phase = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
                density = mixture_density(phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] += delta_momentum / ti.max(
                    density, 1.0e-12
                )
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = ti.cast(lattice_direction(q), ti.f32)
                    self.momentum_populations[i, j, q] += (
                        3.0 * lattice_weight(q) * direction.dot(delta_momentum)
                    )

    @ti.kernel
    def _finish_melt_momentum_coupling(self):
        carrier_change = (
            self._thermal_fluid_momentum_provisional[None]
            - self._thermal_fluid_momentum_before[None]
        )
        correction_change = (
            self._thermal_fluid_momentum_after[None]
            - self._thermal_fluid_momentum_provisional[None]
        )
        self.cumulative_fluid_melt_carrier_momentum_lattice[None] += carrier_change
        self.cumulative_fluid_melt_correction_momentum_lattice[
            None
        ] += correction_change
        carrier_angular_change = (
            self._thermal_fluid_angular_momentum_provisional[None]
            - self._thermal_fluid_angular_momentum_before[None]
        )
        correction_angular_change = (
            self._thermal_fluid_angular_momentum_after[None]
            - self._thermal_fluid_angular_momentum_provisional[None]
        )
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice[
            None
        ] += carrier_angular_change
        self.cumulative_fluid_melt_correction_angular_momentum_lattice[
            None
        ] += correction_angular_change

    def _finalize_melt_momentum_coupling(self):
        """Close thermal, refill and projection momentum changes to body loss."""
        if not self._moving_melt_momentum_pending:
            return
        self._reduce_provisional_melt_momentum()
        # Distribute over all wet cells: a few dilute melt-source cells can give
        # an ill-conditioned rotational correction at a moving contact line.
        status = self._reduce_melt_momentum_weight_geometry()
        if status == -1:
            raise RuntimeError(
                "melt momentum has no legal wet LBM cell for conservative linear/angular-momentum closure"
            )
        if status == -2:
            raise RuntimeError(
                "melt momentum correction requires two distinct wet cells"
            )
        if status == 1:
            self._apply_melt_momentum_correction()
        self._reduce_final_melt_momentum()
        for _ in range(2):
            self._apply_local_melt_momentum_correction()
            self._reduce_final_melt_momentum()
        self._finish_melt_momentum_coupling()
        self._moving_melt_momentum_pending = False

    @ti.kernel
    def _reduce_moving_body_mass_properties(self):
        self._body_reduced_mass[None] = 0.0
        self._body_reduced_first_moment[None] = ti.Vector([0.0, 0.0])
        self._body_reduced_inertia_origin[None] = 0.0
        inverse_reference_cell_mass = ti.cast(
            ti.static(1.0 / (self.config.rho_water * self.config.dx * self.config.dx)),
            ti.f64,
        )
        half_x = ti.static(0.5 * self.config.ice_width)
        half_y = ti.static(0.5 * self.config.ice_height)
        for i, j in self.thermal.body_solid_mass:
            mass = self.thermal.body_solid_mass[i, j] * inverse_reference_cell_mass
            fraction = ti.cast(self.thermal.body_solid_fraction[i, j], ti.f64)
            local = ti.Vector(
                [
                    ti.cast(i, ti.f64) + 0.5 - half_x,
                    ti.cast(j, ti.f64) + 0.5 - half_y,
                ]
            )
            ti.atomic_add(self._body_reduced_mass[None], mass)
            ti.atomic_add(self._body_reduced_first_moment[None].x, mass * local.x)
            ti.atomic_add(self._body_reduced_first_moment[None].y, mass * local.y)
            # A partial finite volume is represented by the equal-area square
            # of side sqrt(f) used by contact and thermal rasterization.  Its
            # intrinsic polar moment is therefore m*f/6; the parallel-axis
            # contribution remains m*|x|^2.
            ti.atomic_add(
                self._body_reduced_inertia_origin[None],
                mass * (local.dot(local) + fraction / 6.0),
            )

    @ti.kernel
    def _apply_moving_body_mass_properties(self, initialize: ti.i32):
        reduced_mass = self._body_reduced_mass[None]
        old_mass = self.body_mass_lattice[None]
        initial_mass = self.body_initial_mass_lattice[None]
        # Material-frame phase change is monotone.  A parallel f64 reduction
        # can nevertheless move by a few ulps when its atomic order changes;
        # project that numerically unresolved change onto the previous rigid
        # mass properties.  A genuine terminal zero is always accepted.
        mass_change_tolerance = 1.0e-12 * ti.max(initial_mass, 1.0)
        new_mass = reduced_mass
        hold_previous_properties = False
        if initialize == 0 and reduced_mass > 0.0:
            unresolved_change = ti.abs(reduced_mass - old_mass) <= (
                mass_change_tolerance
            )
            nonphysical_increase = reduced_mass > old_mass
            if unresolved_change or nonphysical_increase:
                new_mass = old_mass
                hold_previous_properties = True
        inactive_mass_threshold = ti.max(1.0e-12, 1.0e-14 * ti.max(initial_mass, 1.0))
        inactive_inertia_threshold = 1.0e-14 * ti.max(
            ti.static(self._initial_body_inertia_lattice), 1.0
        )
        has_resolved_mass = new_mass > inactive_mass_threshold
        was_active = self.body_active[None] != 0
        old_local_com = self.body_local_center_of_mass[None]
        new_local_com = old_local_com
        raw_inertia = ti.cast(0.0, ti.f64)
        if has_resolved_mass:
            if hold_previous_properties:
                raw_inertia = self.body_inertia_lattice[None]
            else:
                new_local_com = self._body_reduced_first_moment[None] / new_mass
                raw_inertia = self._body_reduced_inertia_origin[None] - new_mass * (
                    new_local_com.dot(new_local_com)
                )
            raw_inertia = ti.max(raw_inertia, 0.0)
        mechanically_resolved = (
            has_resolved_mass
            and raw_inertia > inactive_inertia_threshold
            and (initialize != 0 or was_active)
        )
        # Thermal mass can outlive the mechanically resolved remnant.  Once
        # inertia falls below the grid-scaled threshold, freeze that sub-grid
        # parcel at its last pose and transfer its stored momentum exactly;
        # mass and sensible/latent energy remain in the thermal material grid.
        new_inertia = raw_inertia if mechanically_resolved else 0.0

        angle = self.body_angle[None]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        old_local_world = ti.Vector(
            [
                cosine * old_local_com.x - sine * old_local_com.y,
                sine * old_local_com.x + cosine * old_local_com.y,
            ]
        )
        # The material-frame origin is the authoritative ALE pose.  Rebuilding
        # it from the f32 COM would accumulate a quantization walk at every
        # thermal interval, even when no material was lost.
        origin = ti.cast(self.body_reference_origin[None], ti.f64)
        new_local_world = ti.Vector(
            [
                cosine * new_local_com.x - sine * new_local_com.y,
                sine * new_local_com.x + cosine * new_local_com.y,
            ]
        )

        retiring = was_active and not mechanically_resolved
        if initialize == 0 and (new_mass < old_mass or retiring):
            ti.max(old_mass - new_mass, 0.0)
            old_velocity = ti.cast(self.body_velocity[None], ti.f64)
            omega = ti.cast(self.body_angular_velocity[None], ti.f64)
            # The remaining material keeps the rigid velocity evaluated at
            # its new COM; this is the no-ejection-recoil limit.
            com_shift = new_local_world - old_local_world
            shifted_velocity = ti.Vector([0.0, 0.0], dt=ti.f64)
            if mechanically_resolved:
                shifted_velocity = old_velocity + omega * ti.Vector(
                    [-com_shift.y, com_shift.x]
                )
            # Momentum diagnostics close the actually stored discrete rigid
            # state, including its f32 position/velocity quantization.
            shifted_velocity_stored_f32 = ti.cast(shifted_velocity, ti.f32)
            shifted_velocity_stored = ti.cast(shifted_velocity_stored_f32, ti.f64)
            new_center_stored_f32 = ti.cast(origin + new_local_world, ti.f32)
            new_center_stored = ti.cast(new_center_stored_f32, ti.f64)
            old_center_stored = ti.cast(self.body_center[None], ti.f64)
            lost_momentum = old_mass * old_velocity - new_mass * shifted_velocity_stored
            self.cumulative_melted_momentum_lattice[None] += lost_momentum
            old_body_momentum = old_mass * old_velocity
            new_body_momentum = new_mass * shifted_velocity_stored
            old_angular_momentum = (
                old_center_stored.x * old_body_momentum.y
                - old_center_stored.y * old_body_momentum.x
                + self.body_inertia_lattice[None] * omega
            )
            new_angular_momentum = (
                new_center_stored.x * new_body_momentum.y
                - new_center_stored.y * new_body_momentum.x
                + new_inertia * omega
            )
            self.cumulative_melted_angular_momentum_lattice[None] += (
                old_angular_momentum - new_angular_momentum
            )
            self.body_velocity[None] = shifted_velocity_stored_f32

        self.body_local_center_of_mass[None] = new_local_com
        self.body_mass_lattice[None] = new_mass
        self.body_inertia_lattice[None] = new_inertia
        if not mechanically_resolved:
            self.body_velocity[None] = ti.Vector([0.0, 0.0])
            self.body_angular_velocity[None] = 0.0
            for index in ti.static(range(4)):
                self._body_contact_support_extrema[index] = 0.0

    @ti.kernel
    def _refill_changed_nodes(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            became_solid = (
                self.previous_solid_mask[i, j] == 0
                and self.solid_mask[i, j] == 1
                and self.wall_mask[i, j] == 0
            )
            became_fluid = (
                self.previous_solid_mask[i, j] == 1
                and self.solid_mask[i, j] == 0
                and self.wall_mask[i, j] == 0
            )
            if became_solid:
                covered_phase = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
                reservoir_pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                velocity = self._body_velocity_at(self._cell_point(i, j))
                self.water_phase[i, j] = covered_phase
                self.momentum_velocity_lattice[i, j] = velocity
                self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])
                self.pressure_lattice[i, j] = reservoir_pressure
            elif became_fluid:
                phi_sum = 0.0
                pressure_sum = 0.0
                velocity_sum = ti.Vector([0.0, 0.0])
                count = 0.0
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    ni = i + di
                    nj = j + dj
                    # Only read nodes that were fluid before this raster
                    # update.  Reading another fresh node in the same kernel
                    # would make the extrapolation depend on GPU scheduling.
                    if (
                        (di != 0 or dj != 0)
                        and inside_grid(ni, nj, nx, ny)
                        and self._is_fluid_cell(ni, nj)
                        and self.previous_solid_mask[ni, nj] == 0
                    ):
                        phi_sum += ti.min(1.0, ti.max(0.0, self.water_phase[ni, nj]))
                        neighbor_pressure = (
                            self.pressure_lattice[ni, nj]
                            - self.hydrostatic_reference_pressure[ni, nj]
                        )
                        pressure_sum += neighbor_pressure
                        velocity_sum += self.momentum_velocity_lattice[ni, nj]
                        count += 1.0
                phi0 = ti.min(1.0, ti.max(0.0, self.water_phase[i, j]))
                pressure0 = self.pressure_lattice[i, j]
                velocity0 = self._body_velocity_at(self._cell_point(i, j))
                if count > 0.0:
                    phi0 = phi_sum / count
                    pressure0 = pressure_sum / count
                    velocity0 = velocity_sum / count
                pressure0 += self.hydrostatic_reference_pressure[i, j]
                fresh_density = mixture_density(phi0, rho_water, rho_air)
                self.water_phase[i, j] = phi0
                self.momentum_velocity_lattice[i, j] = velocity0
                self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])
                self.pressure_lattice[i, j] = pressure0
                dynamic_pressure = pressure0 - self.hydrostatic_reference_pressure[i, j]
                for q in range(D2Q9_DIRECTION_COUNT):
                    self.momentum_populations[i, j, q] = momentum_equilibrium(
                        q, dynamic_pressure, fresh_density, velocity0
                    )
                    self.phase_populations[i, j, q] = phase_equilibrium(
                        q, phi0, velocity0
                    )

    # ------------------------------------------------------------------
    # Two-phase pressure--momentum LBM

    @ti.func
    def _fluid_force_viscosity_and_gradient(self, i, j):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        gravity = ti.Vector(
            [ti.static(self._gravity_lattice[0]), ti.static(self._gravity_lattice[1])]
        )
        bulk_energy_coefficient = ti.static(
            12.0 * self._surface_tension_lattice / self.config.interface_width
        )
        gradient_energy_coefficient = ti.static(
            1.5 * self._surface_tension_lattice * self.config.interface_width
        )
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        smagorinsky_constant = ti.static(self.config.smagorinsky_constant)
        water_phase = self.water_phase[i, j]
        bounded_phi = ti.min(1.0, ti.max(0.0, water_phase))
        density = mixture_density(bounded_phi, rho_water, rho_air)

        gradient = ti.Vector([0.0, 0.0])
        source_gradient = ti.Vector([0.0, 0.0])
        laplacian = 0.0
        alpha = ti.static(1.0 / 3.0)
        for q in range(D2Q9_DIRECTION_COUNT):
            direction_i = lattice_direction(q)
            direction = ti.cast(direction_i, ti.f32)
            phi1 = self._phase_neighbor(i, j, direction_i.x, direction_i.y, 1)
            phi2 = self._phase_neighbor(i, j, direction_i.x, direction_i.y, 2)
            source_gradient += (
                3.0 * lattice_weight(q) * direction * (phi1 - water_phase)
            )
            gradient += (
                (1.0 - alpha)
                * 3.0
                * lattice_weight(q)
                * direction
                * (phi1 - water_phase)
            )
            gradient += (
                alpha
                * 1.5
                * lattice_weight(q)
                * direction
                * (4.0 * phi1 - phi2 - 3.0 * water_phase)
            )
            laplacian += 6.0 * lattice_weight(q) * (phi1 - water_phase)

        # Assemble gravity sources as force densities.  The frozen
        # hydrostatic reference removes only the base far-field load; the
        # temperature-dependent density anomaly must remain additive.
        reference_density = self.hydrostatic_reference_density[i, j]
        gravity_force_density = (density - reference_density) * gravity
        reference_temperature = ti.static(
            float(self.config.thermal.buoyancy_reference_temperature_c)
        )
        temperature = ti.cast(self.thermal._water_temperature_at(i, j), ti.f32)
        expansion = ti.static(float(self.config.thermal.thermal_expansion_water_1_k))
        density_anomaly_ratio = -expansion * (temperature - reference_temperature)

        # The thermal field deliberately treats the water/air interface as
        # adiabatic and uses the air temperature for dry thermal cells. A smooth
        # water-side gate prevents that unrelated air value from forcing the
        # diffuse layer.
        water_weight = ti.min(1.0, ti.max(0.0, 2.0 * bounded_phi - 1.0))
        gravity_force_density += (
            water_weight * rho_water * density_anomaly_ratio * gravity
        )
        force = gravity_force_density / density
        chemical = (
            4.0
            * bulk_energy_coefficient
            * bounded_phi
            * (bounded_phi - 1.0)
            * (bounded_phi - 0.5)
            - gradient_energy_coefficient * laplacian
        )
        force += chemical * gradient / density

        velocity = self.momentum_velocity_lattice[i, j]
        velocity_x_gradient_x = 0.0
        velocity_x_gradient_y = 0.0
        velocity_y_gradient_x = 0.0
        velocity_y_gradient_y = 0.0
        for q in range(D2Q9_DIRECTION_COUNT):
            direction_i = lattice_direction(q)
            direction = ti.cast(direction_i, ti.f32)
            ni = i + direction_i.x
            nj = j + direction_i.y
            neighbor_velocity = ti.Vector([0.0, 0.0])
            if inside_grid(ni, nj, nx, ny) and self._is_fluid_cell(ni, nj):
                neighbor_velocity = self.momentum_velocity_lattice[ni, nj]
            elif inside_grid(ni, nj, nx, ny) and self.solid_mask[ni, nj] == 1:
                neighbor_velocity = self._body_velocity_at(self._cell_point(ni, nj))
            difference = neighbor_velocity - velocity
            velocity_x_gradient_x += (
                3.0 * lattice_weight(q) * direction.x * difference.x
            )
            velocity_x_gradient_y += (
                3.0 * lattice_weight(q) * direction.y * difference.x
            )
            velocity_y_gradient_x += (
                3.0 * lattice_weight(q) * direction.x * difference.y
            )
            velocity_y_gradient_y += (
                3.0 * lattice_weight(q) * direction.y * difference.y
            )
        artificial_viscosity = (
            smagorinsky_constant
            * smagorinsky_constant
            * ti.sqrt(
                2.0
                * (
                    velocity_x_gradient_x * velocity_x_gradient_x
                    + velocity_y_gradient_y * velocity_y_gradient_y
                    + 0.5 * (velocity_x_gradient_y + velocity_y_gradient_x) ** 2
                )
            )
        )
        return ti.Vector(
            [
                force.x,
                force.y,
                artificial_viscosity,
                source_gradient.x,
                source_gradient.y,
            ]
        )

    @ti.kernel
    def _collide_momentum(self):
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        nu_water = ti.static(self._water_viscosity_lattice)
        nu_air = ti.static(self._air_viscosity_lattice)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                force_data = self._fluid_force_viscosity_and_gradient(i, j)
                force = ti.Vector([force_data[0], force_data[1]])
                artificial_vis = force_data[2]
                grad_phi = ti.Vector([force_data[3], force_data[4]])
                self.fluid_acceleration_lattice[i, j] = force
                water_phase = self.water_phase[i, j]
                velocity = self.momentum_velocity_lattice[i, j] + 0.5 * force
                phase_fraction = ti.min(1.0, ti.max(0.0, water_phase))
                nu_local = (
                    phase_fraction * nu_water
                    + (1.0 - phase_fraction) * nu_air
                    + artificial_vis
                )
                physical_tau = 0.5 + 3.0 * nu_local
                # A low-density gas and sharp moving cut links amplify
                # unresolved shear modes near the ice/free-surface contact
                # line.  Extend the same non-conserved shear envelope by one
                # lattice cell around the sharp ice geometry; mass, momentum,
                # the pressure trace, and pure-water cells away from the ice
                # retain their original relaxation rates.
                interface_tau = ti.static(self._air_interface_relaxation_time)
                sharp_solid_neighbor = 0.0
                for q in ti.static(range(1, D2Q9_DIRECTION_COUNT)):
                    direction = lattice_direction(q)
                    ni = i + direction.x
                    nj = j + direction.y
                    if inside_grid(ni, nj, self.nx, self.ny):
                        if self.solid_mask[ni, nj] == 1:
                            sharp_solid_neighbor = 1.0
                shear_indicator = ti.max(1.0 - phase_fraction, sharp_solid_neighbor)
                tau_floor = 0.5 + (interface_tau - 0.5) * shear_indicator
                shear_tau = ti.max(physical_tau, tau_floor)
                # Liang et al. PRE 97, 033309 (2018), Eqs. (15)--(23).
                material_density = mixture_density(water_phase, rho_water, rho_air)
                dynamic_pressure = (
                    self.pressure_lattice[i, j]
                    - self.hydrostatic_reference_pressure[i, j]
                )
                force_density = material_density * force
                delta_rho = rho_water - rho_air
                raw_moment_00 = 0.0
                raw_moment_10 = 0.0
                raw_moment_01 = 0.0
                raw_moment_20 = 0.0
                raw_moment_02 = 0.0
                raw_moment_11 = 0.0
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
                for q in range(D2Q9_DIRECTION_COUNT):
                    direction = ti.cast(lattice_direction(q), ti.f32)
                    cx = direction.x
                    cy = direction.y
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
                    raw_moment_00 += value
                    raw_moment_10 += cx * value
                    raw_moment_01 += cy * value
                    raw_moment_20 += cx * cx * value
                    raw_moment_02 += cy * cy * value
                    raw_moment_11 += cx * cy * value
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

                # MRT extension of Liang's BGK equation.  The two deviatoric
                # stresses use the shear envelope; lower-order hydrodynamic
                # moments retain the physical BGK rate, while the trace and
                # ghost modes relax in one step.  The trapezoidal source
                # prefactor is applied per moment.
                physical_omega = 1.0 / physical_tau
                shear_omega = 1.0 / shear_tau
                source_physical = 1.0 - 0.5 * physical_omega
                source_shear = 1.0 - 0.5 * shear_omega
                post_moment_00 = (
                    raw_moment_00
                    - physical_omega * (raw_moment_00 - equilibrium_moment_00)
                    + source_physical * force_moment_00
                )
                post_moment_10 = (
                    raw_moment_10
                    - physical_omega * (raw_moment_10 - equilibrium_moment_10)
                    + source_physical * force_moment_10
                )
                post_moment_01 = (
                    raw_moment_01
                    - physical_omega * (raw_moment_01 - equilibrium_moment_01)
                    + source_physical * force_moment_01
                )
                trace_eq = equilibrium_moment_20 + equilibrium_moment_02
                trace_src = force_moment_20 + force_moment_02
                post_trace = trace_eq + 0.5 * trace_src
                difference = raw_moment_20 - raw_moment_02
                difference_eq = equilibrium_moment_20 - equilibrium_moment_02
                difference_src = force_moment_20 - force_moment_02
                post_difference = (
                    difference
                    - shear_omega * (difference - difference_eq)
                    + source_shear * difference_src
                )
                post_moment_20 = 0.5 * (post_trace + post_difference)
                post_moment_02 = 0.5 * (post_trace - post_difference)
                post_moment_11 = (
                    raw_moment_11
                    - shear_omega * (raw_moment_11 - equilibrium_moment_11)
                    + source_shear * force_moment_11
                )
                post_moment_21 = equilibrium_moment_21 + 0.5 * force_moment_21
                post_moment_12 = equilibrium_moment_12 + 0.5 * force_moment_12
                post_moment_22 = equilibrium_moment_22 + 0.5 * force_moment_22
                raw_post = ti.Vector.zero(ti.f32, D2Q9_DIRECTION_COUNT)
                for q in range(D2Q9_DIRECTION_COUNT):
                    raw_post[q] = reconstruct_central_moment_population(
                        lattice_direction(q).x,
                        lattice_direction(q).y,
                        0.0,
                        0.0,
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
            if self._is_fluid_cell(i, j):
                water_phase = self.water_phase[i, j]
                velocity = (
                    self.momentum_velocity_lattice[i, j]
                    + 0.5 * self.fluid_acceleration_lattice[i, j]
                )
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
                compression = 4.0 * water_phase * (1.0 - water_phase) / interface_width

                phase_raw_moment_10 = 0.0
                phase_raw_moment_01 = 0.0
                phase_equilibrium_moment_00 = 0.0
                phase_equilibrium_moment_10 = 0.0
                phase_equilibrium_moment_01 = 0.0
                phase_equilibrium_moment_20 = 0.0
                phase_equilibrium_moment_02 = 0.0
                phase_equilibrium_moment_11 = 0.0
                phase_equilibrium_moment_21 = 0.0
                phase_equilibrium_moment_12 = 0.0
                phase_equilibrium_moment_22 = 0.0
                phase_force_moment_00 = 0.0
                phase_force_moment_10 = 0.0
                phase_force_moment_01 = 0.0
                phase_force_moment_20 = 0.0
                phase_force_moment_02 = 0.0
                phase_force_moment_11 = 0.0
                phase_force_moment_21 = 0.0
                phase_force_moment_12 = 0.0
                phase_force_moment_22 = 0.0
                for q in range(D2Q9_DIRECTION_COUNT):
                    cx = ti.cast(lattice_direction(q).x, ti.f32)
                    cy = ti.cast(lattice_direction(q).y, ti.f32)
                    rx = cx - velocity.x
                    ry = cy - velocity.y
                    rx2 = rx * rx
                    ry2 = ry * ry
                    h_value = self.phase_populations[i, j, q]
                    h_equilibrium = phase_equilibrium(q, water_phase, velocity)
                    phase_force = (
                        lattice_weight(q)
                        * compression
                        * ti.Vector([cx, cy]).dot(normal)
                    )
                    phase_raw_moment_10 += h_value * rx
                    phase_raw_moment_01 += h_value * ry
                    phase_equilibrium_moment_00 += h_equilibrium
                    phase_equilibrium_moment_10 += h_equilibrium * rx
                    phase_equilibrium_moment_01 += h_equilibrium * ry
                    phase_equilibrium_moment_20 += h_equilibrium * rx2
                    phase_equilibrium_moment_02 += h_equilibrium * ry2
                    phase_equilibrium_moment_11 += h_equilibrium * rx * ry
                    phase_equilibrium_moment_21 += h_equilibrium * rx2 * ry
                    phase_equilibrium_moment_12 += h_equilibrium * rx * ry2
                    phase_equilibrium_moment_22 += h_equilibrium * rx2 * ry2
                    phase_force_moment_00 += phase_force
                    phase_force_moment_10 += phase_force * rx
                    phase_force_moment_01 += phase_force * ry
                    phase_force_moment_20 += phase_force * rx2
                    phase_force_moment_02 += phase_force * ry2
                    phase_force_moment_11 += phase_force * rx * ry
                    phase_force_moment_21 += phase_force * rx2 * ry
                    phase_force_moment_12 += phase_force * rx * ry2
                    phase_force_moment_22 += phase_force * rx2 * ry2
                phase_post_moment_00 = (
                    phase_equilibrium_moment_00 + 0.5 * phase_force_moment_00
                )
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
                        reconstruct_central_moment_population(
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
                                self._cell_point(i, j) + fraction * direction_float
                            )
                            outgoing_phase -= (
                                6.0
                                * lattice_weight(q)
                                * self.water_phase[i, j]
                                * direction_float.dot(boundary_velocity)
                            )
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
                        sdf_fluid = ti.max(self.body_signed_distance_m[i, j], 1.0e-6)
                        eta = ti.min(
                            0.95,
                            ti.max(
                                0.05,
                                sdf_fluid
                                / (
                                    sdf_fluid
                                    - self.body_signed_distance_m[ni, nj]
                                    + 1.0e-12
                                ),
                            ),
                        )
                        direction_f = ti.cast(direction, ti.f32)
                        boundary_velocity = self._body_velocity_at(
                            self._cell_point(i, j) + eta * direction_f
                        )
                        self.phase_populations[i, j, opposite_direction_index(q)] = (
                            outgoing
                            - (
                                6.0
                                * lattice_weight(q)
                                * self.water_phase[i, j]
                                * direction_f.dot(boundary_velocity)
                            )
                        )
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
                water_phase = 0.0
                momentum = ti.Vector([0.0, 0.0])
                for q in range(D2Q9_DIRECTION_COUNT):
                    water_phase += self.phase_populations[i, j, q]
                    momentum += (
                        ti.cast(lattice_direction(q), ti.f32)
                        * self.momentum_populations[i, j, q]
                    )
                self.water_phase[i, j] = water_phase
                density = mixture_density(water_phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] = momentum / ti.max(
                    density, 1.0e-12
                )
            else:
                velocity = ti.Vector([0.0, 0.0])
                if self.solid_mask[i, j] == 1:
                    velocity = self._body_velocity_at(self._cell_point(i, j))
                if self.wall_mask[i, j] == 1:
                    self.water_phase[i, j] = 0.0
                self.momentum_velocity_lattice[i, j] = velocity
                self.fluid_acceleration_lattice[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _update_streamed_phase(self):
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                water_phase = 0.0
                for q in range(D2Q9_DIRECTION_COUNT):
                    water_phase += self.phase_populations[i, j, q]
                self.water_phase[i, j] = water_phase
            else:
                if self.wall_mask[i, j] == 1:
                    self.water_phase[i, j] = 0.0

    def _warm_start_phase(self, steps):
        for _ in range(int(steps)):
            self._collide_phase()
            self._stream_phase_only()
            self._update_streamed_phase()

    @ti.kernel
    def _recover_initial_fluid_moments(self):
        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                water_phase = 0.0
                momentum = ti.Vector([0.0, 0.0])
                for q in range(D2Q9_DIRECTION_COUNT):
                    water_phase += self.phase_populations[i, j, q]
                    momentum += (
                        ti.cast(lattice_direction(q), ti.f32)
                        * self.momentum_populations[i, j, q]
                    )
                self.water_phase[i, j] = water_phase
                material_density = mixture_density(water_phase, rho_water, rho_air)
                self.momentum_velocity_lattice[i, j] = momentum / ti.max(
                    material_density, 1.0e-12
                )
            elif self.solid_mask[i, j] == 1:
                self.momentum_velocity_lattice[i, j] = self._body_velocity_at(
                    self._cell_point(i, j)
                )
            else:
                self.water_phase[i, j] = 0.0
                self.momentum_velocity_lattice[i, j] = ti.Vector([0.0, 0.0])

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
                water_phase = self.water_phase[i, j]
                material_density = mixture_density(water_phase, rho_water, rho_air)
                velocity = (
                    self.momentum_velocity_lattice[i, j]
                    + 0.5 * self.fluid_acceleration_lattice[i, j]
                )
                grad_phi = ti.Vector([0.0, 0.0])
                for q in range(D2Q9_DIRECTION_COUNT):
                    integer_direction = lattice_direction(q)
                    direction = ti.cast(integer_direction, ti.f32)
                    neighbor = self._phase_neighbor(
                        i, j, integer_direction.x, integer_direction.y, 1
                    )
                    grad_phi += (
                        3.0 * lattice_weight(q) * direction * (neighbor - water_phase)
                    )
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
        """Run the contact-support transaction in one device launch.

        ``reduce_support`` is a compile-time template flag.  Keeping the
        choice static lets CUDA lower the reduction loop as a top-level
        ``struct_for`` while allowing fixed-pose refreshes to clear the
        previous support state without rebuilding contact extrema.
        """

        self._reset_moving_body_contact_support()
        if ti.static(reduce_support):
            for i, j in self._moving_body_fraction:
                self._reduce_moving_body_contact_extrema(i, j)
            self._finalize_moving_body_contact_support()

    @ti.func
    def _reset_moving_body_contact_support(self):
        self._body_contact_support_extrema[0] = 1.0e30
        self._body_contact_support_extrema[1] = -1.0e30
        self._body_contact_support_extrema[2] = 1.0e30
        self._body_contact_support_extrema[3] = -1.0e30

    @ti.func
    def _reduce_moving_body_contact_extrema(self, i, j):
        """Find supports of the unclipped sharp LBM raster at this pose."""

        threshold = ti.static(float(self.config.thermal.solid_liquid_threshold))
        origin = self.body_reference_origin[None]
        fraction = self._moving_body_fraction[i, j]
        # The fraction pass has already produced the same raw SDF used
        # by ``_update_solid_mask``.  Reusing its sign
        # and the thresholded coverage keeps contact consistent with the
        # sharp mask, including a legitimate rectangle-edge node whose
        # SDF is zero, while excluding an outside bilinear contributor or
        # an empty rectangle boundary at lower thresholds.
        if fraction >= threshold and self.body_signed_distance_m[i, j] <= 0.0:
            # The sharp solver treats a thresholded world node as one
            # cell-centred finite volume.  Its cell faces therefore give
            # the collision support, with the same one-cell topology as
            # ``solid`` and the cut-link boundary.
            minimum_x = ti.cast(i, ti.f32) - origin.x
            maximum_x = ti.cast(i + 1, ti.f32) - origin.x
            minimum_y = ti.cast(j, ti.f32) - origin.y
            maximum_y = ti.cast(j + 1, ti.f32) - origin.y
            ti.atomic_min(self._body_contact_support_extrema[0], minimum_x)
            ti.atomic_max(self._body_contact_support_extrema[1], maximum_x)
            ti.atomic_min(self._body_contact_support_extrema[2], minimum_y)
            ti.atomic_max(self._body_contact_support_extrema[3], maximum_y)

    @ti.func
    def _finalize_moving_body_contact_support(self):
        active = self._body_contact_support_extrema[2] < 1.0e20
        if not active:
            for index in ti.static(range(4)):
                self._body_contact_support_extrema[index] = 0.0

    @ti.kernel
    def _integrate_rigid_ice(self):
        if self.body_active[None] != 0:
            mass = ti.cast(self.body_mass_lattice[None], ti.f32)
            inertia = ti.cast(self.body_inertia_lattice[None], ti.f32)
            # This branch admits only a mechanically resolved body, so these are
            # the same stored mass and inertia audited by the thermal transition.
            gravity = ti.Vector(
                [
                    ti.static(self._gravity_lattice[0]),
                    ti.static(self._gravity_lattice[1]),
                ]
            )
            total_impulse = (
                ti.cast(self.hydrodynamic_impulse[None], ti.f32) + mass * gravity
            )
            total_torque = ti.cast(self.hydrodynamic_torque[None], ti.f32)

            velocity = ti.static(self.config.linear_damping) * (
                self.body_velocity[None] + total_impulse / mass
            )
            omega = ti.static(self.config.angular_damping) * (
                self.body_angular_velocity[None] + total_torque / inertia
            )
            local_com = ti.cast(self.body_local_center_of_mass[None], ti.f32)
            angle = self.body_angle[None] + omega
            center = self.body_center[None] + velocity

            cosine = ti.cos(angle)
            sine = ti.sin(angle)
            self.body_velocity[None] = velocity
            self.body_angle[None] = angle
            self.body_angular_velocity[None] = omega
            reference_offset = ti.Vector(
                [
                    cosine * local_com.x - sine * local_com.y,
                    sine * local_com.x + cosine * local_com.y,
                ]
            )
            self.body_reference_origin[None] = center - reference_offset
        else:
            self.body_velocity[None] = ti.Vector([0.0, 0.0])
            self.body_angular_velocity[None] = 0.0
        self.hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.hydrodynamic_torque[None] = 0.0

    @ti.kernel
    def _project_moving_body_inside_container(self):
        """Keep the whole body inside the walls and reject outward velocity."""

        self._body_contact_projection_changed[None] = ti.cast(0, ti.i8)
        center = self.body_center[None]
        velocity = self.body_velocity[None]
        angle = self.body_angle[None]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        local_com = ti.cast(self.body_local_center_of_mass[None], ti.f32)
        rotated_com = ti.Vector(
            [
                cosine * local_com.x - sine * local_com.y,
                sine * local_com.x + cosine * local_com.y,
            ]
        )

        if (
            self.body_active[None] != 0
            and self._body_contact_geometry_active[None] != 0
        ):
            minimum_relative_x = self._body_contact_support_extrema[0] - rotated_com.x
            maximum_relative_x = self._body_contact_support_extrema[1] - rotated_com.x
            minimum_relative_y = self._body_contact_support_extrema[2] - rotated_com.y
            maximum_relative_y = self._body_contact_support_extrema[3] - rotated_com.y
            lower_boundary = ti.static(float(self.config.boundary_cells))
            upper_x_boundary = ti.static(float(self.nx - self.config.boundary_cells))
            upper_y_boundary = ti.static(float(self.ny - self.config.boundary_cells))
            lower_x = lower_boundary - minimum_relative_x
            upper_x = upper_x_boundary - maximum_relative_x
            lower_y = lower_boundary - minimum_relative_y
            upper_y = upper_y_boundary - maximum_relative_y
            if center.x <= lower_x:
                if center.x < lower_x:
                    self._body_contact_projection_changed[None] = ti.cast(1, ti.i8)
                center.x = lower_x
                velocity.x = ti.max(velocity.x, 0.0)
            elif center.x >= upper_x:
                if center.x > upper_x:
                    self._body_contact_projection_changed[None] = ti.cast(1, ti.i8)
                center.x = upper_x
                velocity.x = ti.min(velocity.x, 0.0)
            if center.y <= lower_y:
                if center.y < lower_y:
                    self._body_contact_projection_changed[None] = ti.cast(1, ti.i8)
                center.y = lower_y
                velocity.y = ti.max(velocity.y, 0.0)
            elif center.y >= upper_y:
                if center.y > upper_y:
                    self._body_contact_projection_changed[None] = ti.cast(1, ti.i8)
                center.y = upper_y
                velocity.y = ti.min(velocity.y, 0.0)

        self.body_velocity[None] = velocity
        # The fraction pass can be committed unchanged when contact only
        # rejects an outward velocity at an already admissible boundary.  In
        # that case leave the authoritative ALE origin bit-for-bit untouched;
        # rewrite it only after a positional projection has actually changed
        # the accepted pose.
        if self._body_contact_projection_changed[None] != 0:
            self.body_reference_origin[None] = center - rotated_com

    def _solve_contact_and_rasterize(
        self,
        *,
        resolve_contact: bool = True,
    ):
        """Rasterize the moving material and resolve whole-body wall contact.

        A single unclipped fraction pass feeds both the contact support
        reduction and the shared world SDF/thermal coverage fields.  If
        contact projection does not move the body, that pass is committed
        directly.  If projection clamps a penetrated pose, the fraction is
        evaluated once more at the corrected pose before committing it.
        The committed LBM mask retains its old time level until refill. A
        mechanically inactive remnant is rasterized but never projected.

        The body reference origin, angle, and wall are always read from this
        simulator instance. ``resolve_contact=False`` refreshes eroded
        coverage at the fixed thermal pose. The return value reports whether
        contact projection changed the accepted pose.
        """

        self._calculate_moving_body_fraction()
        projected = False

        if resolve_contact:
            self._prepare_moving_body_contact_support_kernel(True)
            self._project_moving_body_inside_container()
            projected = int(self._body_contact_projection_changed[None]) != 0
            if projected:
                self._calculate_moving_body_fraction()
                self._prepare_moving_body_contact_support_kernel(True)
        else:
            # Keep the diagnostic flag scoped to one rasterization
            # transaction when callers intentionally skip contact handling.
            self._body_contact_projection_changed[None] = 0
            self._prepare_moving_body_contact_support_kernel(False)

        self._commit_moving_body_raster()
        return projected

    # ------------------------------------------------------------------
    # Bounded, topology-constrained water-volume projection

    @ti.kernel
    def _initialize_water_volume(self):
        self.phase_change_initial_water_volume[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                ti.atomic_add(
                    self.phase_change_initial_water_volume[None],
                    ti.cast(self.water_phase[i, j], ti.f64),
                )
        self.water_volume_current[None] = self.phase_change_initial_water_volume[None]

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

    @ti.kernel
    def _clip_water_phase_for_projection(self) -> ti.types.vector(2, ti.f64):
        """Restore phase bounds and reduce the resulting active water volume."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        self.water_volume_current[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                old_phase = self.water_phase[i, j]
                bounded_phase = ti.min(1.0, ti.max(0.0, old_phase))
                # Values beyond the resolved logistic band are numerical bulk
                # tails, not a physical interface.  Canonicalizing both sides
                # prevents one-sided clipping/roundoff from accumulating into
                # disconnected droplets or holes.  The resulting mass change
                # is included in the subsequent constrained interface shift.
                if bounded_phase <= cutoff:
                    bounded_phase = 0.0
                elif bounded_phase >= 1.0 - cutoff:
                    bounded_phase = 1.0
                if bounded_phase != old_phase:
                    velocity = self.momentum_velocity_lattice[i, j]
                    dynamic_pressure = (
                        self.pressure_lattice[i, j]
                        - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = mixture_density(old_phase, rho_water, rho_air)
                    new_density = mixture_density(bounded_phase, rho_water, rho_air)
                    for q in range(D2Q9_DIRECTION_COUNT):
                        self.momentum_populations[i, j, q] += momentum_equilibrium(
                            q, dynamic_pressure, new_density, velocity
                        ) - momentum_equilibrium(
                            q, dynamic_pressure, old_density, velocity
                        )
                velocity = self.momentum_velocity_lattice[i, j]
                phase_population_l1 = 0.0
                for q in range(D2Q9_DIRECTION_COUNT):
                    phase_population_l1 += ti.abs(self.phase_populations[i, j, q])
                regularize_bulk = (
                    bounded_phase == 0.0 or bounded_phase == 1.0
                ) and phase_population_l1 > ti.static(
                    _MAX_D2Q9_BULK_PHASE_POPULATION_L1
                )
                if regularize_bulk:
                    # A large L1 norm with a canonical zeroth moment is an
                    # ill-conditioned cancellation of kinetic ghost modes,
                    # not a resolved interface.  Rebuilding h preserves phase
                    # mass and equilibrium flux without damping healthy bulk
                    # populations on every projection.
                    phase_velocity = (
                        self.momentum_velocity_lattice[i, j]
                        + 0.5 * self.fluid_acceleration_lattice[i, j]
                    )
                    reconstructed_sum = 0.0
                    for q in range(D2Q9_DIRECTION_COUNT):
                        reconstructed = phase_equilibrium(
                            q, bounded_phase, phase_velocity
                        )
                        self.phase_populations[i, j, q] = reconstructed
                        reconstructed_sum += reconstructed
                    self.phase_populations[i, j, 0] += bounded_phase - reconstructed_sum
                elif bounded_phase != old_phase:
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
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(bounded_phase, ti.f64)
                )
        return ti.Vector(
            [self.water_volume_target[None], self.water_volume_current[None]], dt=ti.f64
        )

    @ti.kernel
    def _seed_water_projection_interface(self):
        """Seed candidates connected to a local phi=1/2 crossing."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        for i, j in self.water_phase:
            seed = 0
            if self._is_fluid_cell(i, j):
                phase = self.water_phase[i, j]
                if cutoff < phase < 1.0 - cutoff:
                    if phase == 0.5:
                        seed = 1
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ni = i + di
                        nj = j + dj
                        if (
                            (di != 0 or dj != 0)
                            and inside_grid(ni, nj, nx, ny)
                            and self._is_fluid_cell(ni, nj)
                        ):
                            neighbor = self.water_phase[ni, nj]
                            if (phase < 0.5 <= neighbor) or (neighbor < 0.5 <= phase):
                                seed = 1
            self.phase_stream_buffer[i, j, 0] = ti.cast(seed, ti.f32)
            self.phase_stream_buffer[i, j, 1] = 0.0

    @ti.kernel
    def _dilate_water_projection_interface(self, primary_to_next: ti.i32):
        """Dilate between the two scratch planes through candidate cells."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        for i, j in self.water_phase:
            value = 0
            phase = self.water_phase[i, j]
            if self._is_fluid_cell(i, j) and cutoff < phase < 1.0 - cutoff:
                if primary_to_next == 1:
                    value = ti.cast(self.phase_stream_buffer[i, j, 0], ti.i32)
                else:
                    value = ti.cast(self.phase_stream_buffer[i, j, 1], ti.i32)
                if value == 0:
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ni = i + di
                        nj = j + dj
                        if (di != 0 or dj != 0) and inside_grid(ni, nj, nx, ny):
                            neighbor = 0.0
                            if primary_to_next == 1:
                                neighbor = self.phase_stream_buffer[ni, nj, 0]
                            else:
                                neighbor = self.phase_stream_buffer[ni, nj, 1]
                            if neighbor == 1.0:
                                value = 1
            if primary_to_next == 1:
                self.phase_stream_buffer[i, j, 1] = ti.cast(value, ti.f32)
            else:
                self.phase_stream_buffer[i, j, 0] = ti.cast(value, ti.f32)

    @ti.kernel
    def _evaluate_water_projection(
        self, lagrange_multiplier: ti.f64
    ) -> ti.types.vector(2, ti.f64):
        """Evaluate volume and derivative using f64 reductions.

        The mapped value is rounded to the f32 phase storage type before the
        reduction.  Consequently this is the same discrete volume that the
        application kernel and the conservation reduction will observe.
        """

        self.water_volume_current[None] = 0.0
        self.water_projection_derivative[None] = 0.0
        exponential = ti.exp(lagrange_multiplier)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        ti.loop_config(block_dim=128)
        for flat_index in range(((self.nx * self.ny + 127) // 128) * 128):
            threadwater_volume_current = ti.cast(0.0, ti.f64)
            threadwater_projection_derivative = ti.cast(0.0, ti.f64)
            if flat_index < self.nx * self.ny:
                i, j = flat_index // self.ny, flat_index % self.ny
                if self._is_fluid_cell(i, j):
                    phase64 = ti.cast(self.water_phase[i, j], ti.f64)
                    mapped64 = phase64
                    adjustable = self.phase_stream_buffer[i, j, 0] == 1.0
                    if adjustable:
                        mapped64 = (
                            phase64
                            * exponential
                            / (1.0 - phase64 + phase64 * exponential)
                        )
                    mapped32 = ti.cast(mapped64, ti.f32)
                    # Endpoint canonicalization is part of the projection map,
                    # rather than an unaccounted post-processing clip.  This is
                    # an active-set map: once an interface value enters an
                    # unresolved bulk tail it contributes exactly zero or one
                    # to the mass equation and has zero local derivative.
                    if mapped32 <= cutoff:
                        mapped32 = 0.0
                    elif mapped32 >= 1.0 - cutoff:
                        mapped32 = 1.0
                    mapped64 = ti.cast(mapped32, ti.f64)
                    threadwater_volume_current += mapped64
                    if adjustable and cutoff < mapped32 < 1.0 - cutoff:
                        threadwater_projection_derivative += mapped64 * (1.0 - mapped64)
            warpwater_volume_current = _warp_reduce_f64(
                threadwater_volume_current, "sum"
            )
            warpwater_projection_derivative = _warp_reduce_f64(
                threadwater_projection_derivative, "sum"
            )
            if flat_index % 32 == 0:
                ti.atomic_add(self.water_volume_current[None], warpwater_volume_current)
                ti.atomic_add(
                    self.water_projection_derivative[None],
                    warpwater_projection_derivative,
                )
        return ti.Vector(
            [self.water_volume_current[None], self.water_projection_derivative[None]],
            dt=ti.f64,
        )

    @ti.kernel
    def _apply_water_projection(self, lagrange_multiplier: ti.f64) -> ti.f64:
        """Apply the entropic map, lift h, and reduce the final volume."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        exponential = ti.exp(lagrange_multiplier)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        self.water_volume_current[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                old_phase = self.water_phase[i, j]
                new_phase = old_phase
                if self.phase_stream_buffer[i, j, 0] == 1.0:
                    old_phase64 = ti.cast(old_phase, ti.f64)
                    mapped64 = (
                        old_phase64
                        * exponential
                        / (1.0 - old_phase64 + old_phase64 * exponential)
                    )
                    new_phase = ti.cast(mapped64, ti.f32)
                if new_phase <= cutoff:
                    new_phase = 0.0
                elif new_phase >= 1.0 - cutoff:
                    new_phase = 1.0
                if new_phase != old_phase:
                    velocity = self.momentum_velocity_lattice[i, j]
                    dynamic_pressure = (
                        self.pressure_lattice[i, j]
                        - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = mixture_density(old_phase, rho_water, rho_air)
                    new_density = mixture_density(new_phase, rho_water, rho_air)
                    for q in range(D2Q9_DIRECTION_COUNT):
                        self.momentum_populations[i, j, q] += momentum_equilibrium(
                            q, dynamic_pressure, new_density, velocity
                        ) - momentum_equilibrium(
                            q, dynamic_pressure, old_density, velocity
                        )
                    lifted_sum = 0.0
                    for q in range(D2Q9_DIRECTION_COUNT):
                        lifted = (
                            self.phase_populations[i, j, q]
                            + phase_equilibrium(q, new_phase, velocity)
                            - phase_equilibrium(q, old_phase, velocity)
                        )
                        self.phase_populations[i, j, q] = lifted
                        lifted_sum += lifted
                    self.phase_populations[i, j, 0] += new_phase - lifted_sum
                    self.water_phase[i, j] = new_phase
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(new_phase, ti.f64)
                )
        return self.water_volume_current[None]

    @ti.kernel
    def _measure_water_projection_residual_weight(self, residual: ti.f64) -> ti.f64:
        """Measure the free-set metric for a storage-level mass closure."""

        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        # Four global f32 ulps are conservative at both phase endpoints and
        # keep the subsequent cast strictly outside the canonicalized tails.
        storage_margin = ti.static(4.0 * float(np.finfo(np.float32).eps))
        magnitude = ti.abs(residual)
        self.water_projection_derivative[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j) and self.phase_stream_buffer[i, j, 0] == 1.0:
                phase = ti.cast(self.water_phase[i, j], ti.f64)
                free = cutoff < phase < 1.0 - cutoff
                if residual > 0.0:
                    free = free and 1.0 - cutoff - phase > magnitude + storage_margin
                else:
                    free = free and phase - cutoff > magnitude + storage_margin
                if free:
                    ti.atomic_add(
                        self.water_projection_derivative[None],
                        phase * (1.0 - phase),
                    )
        return self.water_projection_derivative[None]

    @ti.kernel
    def _apply_water_projection_residual(
        self, residual: ti.f64, total_weight: ti.f64
    ) -> ti.f64:
        """Close a scalar f32 residual on the safe entropic free set."""

        rho_water = ti.static(self._water_density_lattice)
        rho_air = ti.static(self._air_density_lattice)
        cutoff = ti.static(self.config.volume_projection_interface_cutoff)
        storage_margin = ti.static(4.0 * float(np.finfo(np.float32).eps))
        magnitude = ti.abs(residual)
        self.water_volume_current[None] = 0.0
        for i, j in self.water_phase:
            if self._is_fluid_cell(i, j):
                old_phase = self.water_phase[i, j]
                old_phase64 = ti.cast(old_phase, ti.f64)
                new_phase = old_phase
                free = self.phase_stream_buffer[i, j, 0] == 1.0 and (
                    cutoff < old_phase64 < 1.0 - cutoff
                )
                if residual > 0.0:
                    free = free and (
                        1.0 - cutoff - old_phase64 > magnitude + storage_margin
                    )
                else:
                    free = free and (old_phase64 - cutoff > magnitude + storage_margin)
                if free and total_weight > 0.0:
                    weight = old_phase64 * (1.0 - old_phase64)
                    new_phase = ti.cast(
                        old_phase64 + residual * weight / total_weight, ti.f32
                    )
                if new_phase != old_phase:
                    velocity = self.momentum_velocity_lattice[i, j]
                    dynamic_pressure = (
                        self.pressure_lattice[i, j]
                        - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = mixture_density(old_phase, rho_water, rho_air)
                    new_density = mixture_density(new_phase, rho_water, rho_air)
                    for q in range(D2Q9_DIRECTION_COUNT):
                        self.momentum_populations[i, j, q] += momentum_equilibrium(
                            q, dynamic_pressure, new_density, velocity
                        ) - momentum_equilibrium(
                            q, dynamic_pressure, old_density, velocity
                        )
                    lifted_sum = 0.0
                    for q in range(D2Q9_DIRECTION_COUNT):
                        lifted = (
                            self.phase_populations[i, j, q]
                            + phase_equilibrium(q, new_phase, velocity)
                            - phase_equilibrium(q, old_phase, velocity)
                        )
                        self.phase_populations[i, j, q] = lifted
                        lifted_sum += lifted
                    self.phase_populations[i, j, 0] += new_phase - lifted_sum
                    self.water_phase[i, j] = new_phase
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(new_phase, ti.f64)
                )
        return self.water_volume_current[None]

    def _correct_water_volume(self):
        """Project onto the melt-adjusted volume, returning each reduction once."""
        initial = self._clip_water_phase_for_projection()
        target, current = float(initial[0]), float(initial[1])
        tolerance = self.config.volume_projection_tolerance * max(1.0, abs(target))
        initial_error = target - current
        if abs(initial_error) <= tolerance:
            return
        self._seed_water_projection_interface()
        for pass_index in range(self._volume_projection_band_radius):
            self._dilate_water_projection_interface(1 if pass_index % 2 == 0 else 0)
        lambda_limit = (
            4.0 * self.config.volume_projection_max_shift / self.config.interface_width
        )
        lower_mass = float(self._evaluate_water_projection(-lambda_limit)[0])
        upper_mass = float(self._evaluate_water_projection(lambda_limit)[0])
        if abs(upper_mass - lower_mass) <= tolerance:
            raise RuntimeError(
                "water-volume projection is infeasible: no adjustable phi=0.5-connected interface is available"
            )
        if target < lower_mass - tolerance or target > upper_mass + tolerance:
            raise RuntimeError(
                "water-volume projection exceeds the permitted interface shift: "
                f"target={target:.17g}, reachable=[{lower_mass:.17g}, {upper_mass:.17g}]"
            )
        lower_lambda, upper_lambda = -lambda_limit, lambda_limit
        lagrange_multiplier, best_lambda = 0.0, 0.0
        best_residual = abs(initial_error)
        for _ in range(self.config.volume_projection_max_iterations):
            evaluation = self._evaluate_water_projection(lagrange_multiplier)
            mass, derivative = float(evaluation[0]), float(evaluation[1])
            residual = mass - target
            if abs(residual) < best_residual:
                best_residual, best_lambda = abs(residual), lagrange_multiplier
            if abs(residual) <= tolerance:
                best_lambda = lagrange_multiplier
                break
            if residual < 0.0:
                lower_lambda = lagrange_multiplier
            else:
                upper_lambda = lagrange_multiplier
            proposal = math.nan
            if derivative > 0.0 and math.isfinite(derivative):
                proposal = lagrange_multiplier - residual / derivative
            if (
                not math.isfinite(proposal)
                or not lower_lambda < proposal < upper_lambda
            ):
                proposal = 0.5 * (lower_lambda + upper_lambda)
            lagrange_multiplier = proposal
        current = float(self._apply_water_projection(best_lambda))
        # The rounded endpoint map can have small gaps; close only on interface
        # cells safely separated from either canonicalized pure phase.
        for _ in range(8):
            residual = target - current
            if abs(residual) <= tolerance:
                break
            weight = float(self._measure_water_projection_residual_weight(residual))
            if not math.isfinite(weight) or weight <= 0.0:
                break
            current = float(self._apply_water_projection_residual(residual, weight))
        error = abs(target - current)
        if error > tolerance:
            raise RuntimeError(
                "water-volume projection cannot close its endpoint active set: "
                f"residual={error:.6e}, tolerance={tolerance:.6e}"
            )

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
        local = self._body_local_coordinates(self._cell_point(i, j))
        index = self._nearest_body_index(local)
        delta = ti.abs(local) - ti.Vector(
            [self._body_half_width, self._body_half_height]
        )
        outside = ti.max(delta, 0.0)
        distance = outside.norm() + ti.min(ti.max(delta.x, delta.y), 0.0)
        if (
            0 <= index.x < self.config.ice_width
            and 0 <= index.y < self.config.ice_height
        ):
            distance = (
                self.config.thermal.solid_liquid_threshold
                - self._moving_body_fraction[i, j]
            )
        return distance * self.config.dx

    def _signed_distance_numpy(self):
        origin = np.asarray(self.body_reference_origin[None], dtype=np.float32)
        angle = np.float32(self.body_angle[None])
        cosine, sine = np.cos(angle), np.sin(angle)
        x, y = np.meshgrid(
            np.arange(self.nx, dtype=np.float32) + 0.5 - origin[0],
            np.arange(self.ny, dtype=np.float32) + 0.5 - origin[1],
            indexing="ij",
        )
        local_x, local_y = cosine * x + sine * y, -sine * x + cosine * y
        delta_x, delta_y = (
            np.abs(local_x) - self._body_half_width,
            np.abs(local_y) - self._body_half_height,
        )
        distance = np.hypot(
            np.maximum(delta_x, 0.0), np.maximum(delta_y, 0.0)
        ) + np.minimum(np.maximum(delta_x, delta_y), 0.0)
        inside = (
            (local_x >= -self._body_half_width)
            & (local_x < self._body_half_width)
            & (local_y >= -self._body_half_height)
            & (local_y < self._body_half_height)
        )
        distance[inside] = (
            self.config.thermal.solid_liquid_threshold
            - self._moving_body_fraction.to_numpy()[inside]
        )
        return (distance * self.config.dx).astype(np.float32)

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
        return ti.cast(ti.static(self._initial_body_mass_lattice), ti.f64)

    @ti.func
    def _read_body_active(self, index):
        return ti.cast(self.body_inertia_lattice[None] > 0.0, ti.i8)

    @ti.func
    def _read_cumulative_melted_mass_lattice(self, index):
        return self._initial_body_mass_lattice - self.body_mass_lattice[None]

    @ti.func
    def _read_cumulative_fluid_melt_momentum_lattice(self, index):
        return (
            self.cumulative_fluid_melt_carrier_momentum_lattice[None]
            + self.cumulative_fluid_melt_correction_momentum_lattice[None]
        )

    @ti.func
    def _read_melt_momentum_residual_lattice(self, index):
        return (
            self.cumulative_fluid_melt_momentum_lattice[None]
            - self.cumulative_melted_momentum_lattice[None]
        )

    @ti.func
    def _read_cumulative_fluid_melt_angular_momentum_lattice(self, index):
        return (
            self.cumulative_fluid_melt_carrier_angular_momentum_lattice[None]
            + self.cumulative_fluid_melt_correction_angular_momentum_lattice[None]
        )

    @ti.func
    def _read_melt_angular_momentum_residual_lattice(self, index):
        return (
            self.cumulative_fluid_melt_angular_momentum_lattice[None]
            - self.cumulative_melted_angular_momentum_lattice[None]
        )

    @ti.func
    def _read_ale_water_residual_cells(self, index):
        return self.thermal.phase_aperture_volume_residual_m2[None] / (
            self.config.dx * self.config.dx
        )

    @ti.func
    def _read_ale_energy_residual_j_m(self, index):
        return self.thermal.phase_aperture_energy_residual_j_m[None]

    @ti.func
    def _read_phase_aperture_water_residual_cells(self, index):
        return self.thermal.phase_aperture_volume_residual_m2[None] / (
            self.config.dx * self.config.dx
        )

    @ti.func
    def _read_phase_aperture_energy_residual_j_m(self, index):
        return self.thermal.phase_aperture_energy_residual_j_m[None]

    @ti.func
    def _read_phase_aperture_capacity_margin_cells(self, index):
        return self.thermal.phase_aperture_capacity_margin_m2[None] / (
            self.config.dx * self.config.dx
        )

    @ti.func
    def _read_phase_change_initial_solid_volume(self, index):
        return ti.cast(
            ti.static(float(self.config.ice_width * self.config.ice_height)), ti.f64
        )

    @ti.func
    def _read_phase_change_current_solid_volume(self, index):
        return self.body_mass_lattice[None] * (
            self.config.rho_water / self.config.rho_ice
        )

    @ti.func
    def _read__body_contact_geometry_active(self, index):
        return ti.cast(
            self._body_contact_support_extrema[1]
            > self._body_contact_support_extrema[0],
            ti.i8,
        )

    @ti.func
    def _body_center_at(self, index):
        angle = self.body_angle[None]
        cosine, sine = ti.cos(angle), ti.sin(angle)
        local = self.body_local_center_of_mass[None]
        offset = ti.Vector(
            [cosine * local.x - sine * local.y, sine * local.x + cosine * local.y]
        )
        return ti.cast(
            ti.cast(self.body_reference_origin[None], ti.f64) + offset, ti.f32
        )

    def _body_center_numpy(self):
        angle = np.float32(self.body_angle[None])
        cosine, sine = np.cos(angle), np.sin(angle)
        local = np.asarray(self.body_local_center_of_mass[None], dtype=np.float64)
        offset = np.array(
            [cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1]]
        )
        return (
            np.asarray(self.body_reference_origin[None], dtype=np.float64) + offset
        ).astype(np.float32)

    def _set_body_center_host(self, index, value):
        if index is not None:
            raise IndexError("body_center is a scalar vector; use [None]")
        angle = np.float32(self.body_angle[None])
        cosine, sine = np.cos(angle), np.sin(angle)
        local = np.asarray(self.body_local_center_of_mass[None], dtype=np.float64)
        offset = np.array(
            [cosine * local[0] - sine * local[1], sine * local[0] + cosine * local[1]]
        )
        self.body_reference_origin[None] = np.asarray(value, dtype=np.float64) - offset

    @ti.func
    def _cut_link_fraction(self, i, j, neighbor_i, neighbor_j):
        fluid_distance = ti.max(self.body_signed_distance_m[i, j], 1.0e-6)
        return ti.min(
            0.95,
            ti.max(
                0.05,
                fluid_distance
                / (
                    fluid_distance
                    - self.body_signed_distance_m[neighbor_i, neighbor_j]
                    + 1.0e-12
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
                fraction = self._cut_link_fraction(i, j, neighbor_i, neighbor_j)
                direction_float = ti.cast(direction, ti.f32)
                boundary_point = self._cell_point(i, j) + fraction * direction_float
                boundary_velocity = self._body_velocity_at(boundary_point)
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
                relative64 = ti.cast(boundary_point - self.body_center[None], ti.f64)
                impulse_sum += impulse64
                torque_sum += cross_product_2d(relative64, impulse64)
                has_cut_link = True
        if has_cut_link:
            ti.atomic_add(self.hydrodynamic_impulse[None], impulse_sum)
            ti.atomic_add(self.hydrodynamic_torque[None], torque_sum)
        return prepared

    def _allocate_populations(self):
        """Keep (i,j,q) indexing with q-major physical storage for coalesced loads."""
        populations = ti.field(ti.f32)
        ti.root.dense(ti.k, D2Q9_DIRECTION_COUNT).dense(
            ti.ij, (self.nx, self.ny)
        ).place(populations)
        return populations

    @ti.func
    def _current_melt_momentum_correction(self, index):
        return (
            self.cumulative_melted_momentum_lattice[None]
            - self._thermal_body_momentum_before[None]
            - (
                self._thermal_fluid_momentum_after[None]
                - self._thermal_fluid_momentum_before[None]
            )
        )

    @ti.func
    def _current_melt_angular_correction(self, index):
        return (
            self.cumulative_melted_angular_momentum_lattice[None]
            - self._thermal_body_angular_momentum_before[None]
            - (
                self._thermal_fluid_angular_momentum_after[None]
                - self._thermal_fluid_angular_momentum_before[None]
            )
        )

    @ti.func
    def _wet_support_centroid(self, index):
        centroid = ti.Vector.zero(ti.f64, 2)
        weight = self._melt_momentum_wet_weight[None]
        if weight > 1.0e-30:
            centroid = self._melt_momentum_weight_first_moment[None] / weight
        return centroid

    @ti.func
    def _wet_support_polar_moment(self, index):
        centroid = self._wet_support_centroid(None)
        return ti.max(
            0.0,
            self._melt_momentum_weight_second_moment[None]
            - self._melt_momentum_wet_weight[None] * centroid.dot(centroid),
        )

    def _wet_support_centroid_numpy(self):
        weight = float(self._melt_momentum_wet_weight[None])
        if weight <= 1.0e-30:
            return np.zeros(2, dtype=np.float64)
        return (
            np.asarray(self._melt_momentum_weight_first_moment[None], dtype=np.float64)
            / weight
        )

    @ti.func
    def _water_volume_target_at(self, index):
        return (
            self.phase_change_initial_water_volume[None]
            + self._initial_body_mass_lattice
            - self.body_mass_lattice[None]
        )

    def _set_water_volume_target_host(self, index, value):
        if index is not None:
            raise IndexError("water_volume_target is scalar; use [None]")
        self.phase_change_initial_water_volume[None] = (
            float(value)
            - self._initial_body_mass_lattice
            + float(self.body_mass_lattice[None])
        )
