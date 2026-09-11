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

import numpy as np
import taichi as ti

from .config import IceFlowConfig
from .lattice import (
    Q,
    _c,
    _cross2,
    _heq,
    _inside,
    _opp,
    _pressure_eq,
    _reconstruct_central,
    _rho_mix,
    _w,
)
from .thermal import LatticeScales, MovingBodyThermal2D


_MAX_D2Q9_LATTICE_VELOCITY_L1 = 1.0
_MAX_D2Q9_BULK_PHASE_POPULATION_L1 = 2.0


def ensure_taichi_cuda():
    """Initialize the one supported backend without resetting an active runtime."""

    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        try:
            ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
        except Exception as exc:  # pragma: no cover - depends on host GPU
            raise RuntimeError(
                "IceFlow2D requires Taichi CUDA; no CPU fallback is provided"
            ) from exc
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("IceFlow2D requires an existing or new Taichi CUDA runtime")


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
        self.cfg = config
        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.steps = 0
        self.scales = LatticeScales.from_iceflow_config(config)

        # Dimensional-to-lattice conversion.
        u_ref_lattice = self.scales.reference_lattice_velocity
        u_ref = self.scales.reference_velocity_m_s
        c_u = self.scales.velocity_scale_m_s
        self._time_step_s = self.scales.dt_s
        self._rho_water_l = 1.0
        self._rho_air_l = float(config.rho_air / config.rho_water)
        self._nu_water_l = float(
            config.viscosity_water * u_ref_lattice / (config.dx * u_ref)
        )
        self._nu_air_l = float(
            config.viscosity_air * u_ref_lattice / (config.dx * u_ref)
        )
        self._gravity_l = (
            float(config.gravity[0] * config.dx / (c_u * c_u)),
            float(config.gravity[1] * config.dx / (c_u * c_u)),
        )
        self._sigma_l = float(
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
        self._body_mass = float(config.ice_mass_lattice)
        self._body_inertia = float(config.ice_inertia_lattice)
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

        shape_q = (self.nx, self.ny, Q)
        shape_xy = (self.nx, self.ny)

        # f is Liang et al.'s pressure--momentum distribution:
        # sum(c*f)=rho(phi)*u and its second moment contains pressure, while
        # sum(f) is deliberately not material density.  h is the conservative
        # Allen--Cahn phase distribution.
        self.f = ti.field(ti.f32, shape=shape_q)
        self.f_post = ti.field(ti.f32, shape=shape_q)
        # Tao et al.'s one-point curved boundary condition needs the
        # pre-collision non-equilibrium population.  Keep it separate from f:
        # push streaming overwrites f in place, so reading f from _stream
        # would otherwise race with a neighboring fluid cell.
        self.f_pre_collision_neq = ti.field(ti.f32, shape=shape_q)
        self.h = ti.field(ti.f32, shape=shape_q)
        # After phase streaming, h_post is dead until the next phase
        # collision.  Its q=0 and q=1 planes are reused as the two projection
        # masks, avoiding dedicated device arrays without aliasing a live
        # value or reading and writing the same plane in one kernel.
        self.h_post = ti.field(ti.f32, shape=shape_q)
        self.phi = ti.field(ti.f32, shape=shape_xy)
        self.u = ti.Vector.field(2, ti.f32, shape=shape_xy)
        # Retained across streaming because pressure reconstruction uses the
        # half-force velocity at the next time level.
        self.fluid_force = ti.Vector.field(2, ti.f32, shape=shape_xy)
        self.p = ti.field(ti.f32, shape=shape_xy)
        self.hydrostatic_reference_pressure = ti.field(ti.f32, shape=shape_xy)
        self.hydrostatic_reference_density = ti.field(ti.f32, shape=shape_xy)

        # Static container and the single world signed-distance field.  The
        # rasterizer fills this field with the current candidate pose; after
        # the contact transaction, the LBM kernels consume the same field.
        self.wall = ti.field(ti.i8, shape=shape_xy)
        self.solid = ti.field(ti.i8, shape=shape_xy)
        self.sdf = ti.field(ti.f32, shape=shape_xy)
        self.solid_prev = ti.field(ti.i8, shape=shape_xy)

        # Rigid state and the only load accumulator consumed by integration.
        self.body_center = ti.Vector.field(2, ti.f32, shape=())
        self.body_velocity = ti.Vector.field(2, ti.f32, shape=())
        self.body_angle = ti.field(ti.f32, shape=())
        self.body_angular_velocity = ti.field(ti.f32, shape=())
        # Dynamic mass properties let the material-frame thermal path erode
        # the rigid remnant.
        self.body_initial_mass_lattice = ti.field(ti.f64, shape=())
        self.body_mass_lattice = ti.field(ti.f64, shape=())
        self.body_inertia_lattice = ti.field(ti.f64, shape=())
        self.body_local_center_of_mass = ti.Vector.field(2, ti.f64, shape=())
        self.body_reference_origin = ti.Vector.field(2, ti.f32, shape=())
        self.body_active = ti.field(ti.i8, shape=())
        self.cumulative_melted_mass_lattice = ti.field(ti.f64, shape=())
        self.cumulative_melted_momentum_lattice = ti.Vector.field(2, ti.f64, shape=())
        self.cumulative_fluid_melt_momentum_lattice = ti.Vector.field(
            2, ti.f64, shape=()
        )
        self.cumulative_fluid_melt_carrier_momentum_lattice = ti.Vector.field(
            2, ti.f64, shape=()
        )
        self.cumulative_fluid_melt_correction_momentum_lattice = ti.Vector.field(
            2, ti.f64, shape=()
        )
        self.melt_momentum_residual_lattice = ti.Vector.field(2, ti.f64, shape=())
        self.cumulative_melted_angular_momentum_lattice = ti.field(ti.f64, shape=())
        self.cumulative_fluid_melt_angular_momentum_lattice = ti.field(ti.f64, shape=())
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice = ti.field(
            ti.f64, shape=()
        )
        self.cumulative_fluid_melt_correction_angular_momentum_lattice = ti.field(
            ti.f64, shape=()
        )
        self.melt_angular_momentum_residual_lattice = ti.field(ti.f64, shape=())
        self.ale_water_residual_cells = ti.field(ti.f64, shape=())
        self.ale_energy_residual_j_m = ti.field(ti.f64, shape=())
        self.phase_aperture_water_residual_cells = ti.field(ti.f64, shape=())
        self.phase_aperture_energy_residual_j_m = ti.field(ti.f64, shape=())
        self.phase_aperture_capacity_margin_cells = ti.field(ti.f64, shape=())
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
        self._melt_momentum_correction = ti.Vector.field(2, ti.f64, shape=())
        self._melt_angular_momentum_correction = ti.field(ti.f64, shape=())
        self._melt_momentum_source_weight = ti.field(ti.f64, shape=())
        self._melt_momentum_wet_weight = ti.field(ti.f64, shape=())
        self._melt_momentum_source_target = ti.field(ti.i32, shape=())
        self._melt_momentum_wet_target = ti.field(ti.i32, shape=())
        self._melt_momentum_source_target_max = ti.field(ti.i32, shape=())
        self._melt_momentum_wet_target_max = ti.field(ti.i32, shape=())
        self._melt_momentum_weight_first_moment = ti.Vector.field(2, ti.f64, shape=())
        self._melt_momentum_weight_second_moment = ti.field(ti.f64, shape=())
        self._melt_momentum_weight_centroid = ti.Vector.field(2, ti.f64, shape=())
        self._melt_momentum_weight_polar_moment = ti.field(ti.f64, shape=())
        # For an eroding body, container contact is reconstructed from the
        # current unclipped sharp world raster rather than the initial rectangle.
        # The four entries are min-x, max-x, min-y and max-y in the world
        # orientation but relative to ``body_reference_origin``.
        self._body_contact_support_extrema = ti.field(ti.f32, shape=4)
        self._body_contact_geometry_active = ti.field(ti.i8, shape=())
        # The continuous body coverage is first evaluated without applying
        # the container mask.  Contact reduction consumes this scratch field
        # so the same fraction evaluation can be reused by the world
        # rasterizer; a second evaluation is needed only when projection
        # changes the accepted pose.
        self._moving_body_fraction = ti.field(ti.f32, shape=shape_xy)
        self._body_contact_projection_changed = ti.field(ti.i8, shape=())
        # Cut-link loads are signed sums with strong cancellation.  CUDA is
        # free to order global atomics differently as block scheduling
        # changes, so reduce in f64 and round only at rigid integration.
        self.hydrodynamic_impulse = ti.Vector.field(2, ti.f64, shape=())
        self.hydrodynamic_torque = ti.field(ti.f64, shape=())
        # Water-volume constraint and interface-projection state.  It remains
        # constant in the mechanical model and becomes a density-aware target
        # when thermal phase change is enabled.
        self.water_volume_target = ti.field(ti.f64, shape=())
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
        # ``MovingBodyThermal2D`` no longer owns world rasterization.  Keep a
        # narrow delegate for older callers while making the simulator the
        # single implementation of the geometry/contact transaction.
        self.thermal._rasterize_world_callback = self._solve_contact_and_rasterize
        self.thermal_enthalpy = self.thermal.enthalpy_j_m3
        self.temperature = self.thermal.temperature_c
        self.liquid_fraction = self.thermal.liquid_fraction
        self.phase_change_material = self.thermal.ice_material
        self.thermal_advection_velocity = ti.Vector.field(2, ti.f32, shape=shape_xy)
        self.thermal_max_velocity_l1 = ti.field(ti.f32, shape=())
        self.thermal_max_fluid_velocity_l1 = ti.field(ti.f32, shape=())
        self.thermal_state_invalid = ti.field(ti.i32, shape=())
        self.phase_change_initial_water_volume = ti.field(ti.f64, shape=())
        self.phase_change_initial_solid_volume = ti.field(ti.f64, shape=())
        self.phase_change_current_solid_volume = ti.field(ti.f64, shape=())
        self.phase_change_current_geometry_volume = ti.field(ti.f64, shape=())

        self._initialize_body()
        self._initialize_geometry()
        # Seed the material fractions before the first world raster so the
        # canonical SDF and the initial sharp mask are derived from the same
        # geometry.  The full thermal state is initialized later, after the
        # diffuse phase warm-up has reached its final composition.
        self.thermal._initialize_body_state()
        self._solve_contact_and_rasterize(initialize_previous=True)
        self._initialize_moving_body_geometry()
        self._initialize_fluid()
        # The no-melting invariant is the sharp initial water volume.  Phase
        # warm-up is a numerical preparation step and must not redefine it.
        self._initialize_water_volume()
        if config.phase_warmup_steps > 0:
            self._warm_start_phase(config.phase_warmup_steps)
        self._update_fluid_macro()
        self._correct_water_volume()
        # Initialize enthalpy only after the diffuse water/air interface has
        # reached its projected initial state.  Otherwise the first thermal
        # recovery would combine pre-warm-up composition energy with the
        # post-warm-up phase fraction.
        self.thermal.initialize(self.phi, self.wall, self.solid)
        self.thermal._compose_world_temperature(self.wall)
        self._update_moving_body_mass_properties(initialize=True)
        self._initialize_phase_change_reference()
        self._build_hydrostatic_reference()
        self._initialize_hydrostatic_equilibrium()

    def step(self, num_steps=1):
        for _ in range(int(num_steps)):
            self._collide_velocity()
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
            if self._thermal_lbm_steps_pending >= self.cfg.thermal.update_interval_lbm_steps:
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
        """Advance pose ALE and paired water advection at every LBM step."""

        self._prepare_thermal_advection_velocity()
        maximum_velocity_l1 = float(self.thermal_max_velocity_l1[None])
        if self._thermal_stability_check_enabled:
            self._check_thermal_advection_stability(target_step=target_step)
        self.thermal.advance_fast(
            self._time_step_s,
            self.thermal_advection_velocity,
            self.phi,
            self.wall,
            self.solid,
            max_velocity_lattice_l1=maximum_velocity_l1,
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
            self.phi, self.wall, self.solid, refresh_derived=False
        )
        self._snapshot_melt_momentum_coupling()
        self.thermal.advance_slow(
            count * self._time_step_s,
            self.phi,
            self.wall,
            self.solid,
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
        self._update_phase_change_water_target()
        self._moving_melt_momentum_pending = True

    def _synchronize_moving_water_aperture(self):
        """Align thermal extensive water after the LBM phase projection."""

        self.thermal.synchronize_water_aperture(self.phi, self.wall, self.solid)

    def synchronize_diagnostics(self):
        """Refresh output-only ALE diagnostics at a snapshot boundary."""

        self._synchronize_moving_thermal_diagnostics()

    # Geometry and initialization

    @ti.func
    def _active(self, i, j):
        return self.wall[i, j] == 0 and self.solid[i, j] == 0

    @ti.kernel
    def _prepare_thermal_advection_velocity(self):
        """Store the half-force velocity and water-side CFL maximum.

        Stability diagnostics are deliberately collected by a separate,
        opt-in pass.  This kernel is still required by the fast thermal
        solver: it supplies the velocity field used by water advection and
        the water-side maximum used to choose its Courant substeps.
        """

        self.thermal_max_velocity_l1[None] = 0.0
        for i, j in self.u:
            velocity = ti.Vector([0.0, 0.0])
            if self._active(i, j):
                velocity = self.u[i, j] + 0.5 * self.fluid_force[i, j]
            self.thermal_advection_velocity[i, j] = velocity
            include_in_cfl = self._active(i, j)
            include_in_cfl = include_in_cfl and self.phi[i, j] > ti.static(
                self.cfg.volume_projection_interface_cutoff
            )
            if include_in_cfl:
                ti.atomic_max(
                    self.thermal_max_velocity_l1[None],
                    ti.abs(velocity.x) + ti.abs(velocity.y),
                )

    @ti.kernel
    def _collect_thermal_stability_metrics(self):
        """Collect optional finite-state and full-fluid speed diagnostics."""

        self.thermal_max_fluid_velocity_l1[None] = 0.0
        self.thermal_state_invalid[None] = 0
        for i, j in self.u:
            if self._active(i, j):
                velocity = self.thermal_advection_velocity[i, j]
                state_invalid = (
                    ti.math.isnan(velocity.x)
                    or ti.math.isinf(velocity.x)
                    or ti.math.isnan(velocity.y)
                    or ti.math.isinf(velocity.y)
                    or ti.math.isnan(self.phi[i, j])
                    or ti.math.isinf(self.phi[i, j])
                )
                for q in range(Q):
                    state_invalid = (
                        state_invalid
                        or ti.math.isnan(self.h[i, j, q])
                        or ti.math.isinf(self.h[i, j, q])
                    )
                if state_invalid:
                    ti.atomic_max(self.thermal_state_invalid[None], 1)
                ti.atomic_max(
                    self.thermal_max_fluid_velocity_l1[None],
                    ti.abs(velocity.x) + ti.abs(velocity.y),
                )

    def _check_thermal_advection_stability(self, *, target_step):
        """Run the opt-in LBM sanity scan before thermal subcycling."""

        self._collect_thermal_stability_metrics()
        maximum_water_velocity_l1 = float(self.thermal_max_velocity_l1[None])
        maximum_fluid_velocity_l1 = float(self.thermal_max_fluid_velocity_l1[None])
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

        velocity = self.thermal_advection_velocity.to_numpy()
        phase = self.phi.to_numpy()
        populations = self.h.to_numpy()
        active = (self.wall.to_numpy() == 0) & (self.solid.to_numpy() == 0)
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
    def _box_sdf(self, point):
        center = self.body_center[None]
        angle = self.body_angle[None]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        relative = point - center
        local = ti.Vector(
            [
                cosine * relative.x + sine * relative.y,
                -sine * relative.x + cosine * relative.y,
            ]
        )
        delta = ti.Vector(
            [
                ti.abs(local.x) - ti.static(self._body_half_width),
                ti.abs(local.y) - ti.static(self._body_half_height),
            ]
        )
        outside = ti.Vector([ti.max(delta.x, 0.0), ti.max(delta.y, 0.0)])
        return outside.norm() + ti.min(ti.max(delta.x, delta.y), 0.0)

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
        value = self.phi[i, j]
        ni1 = i + step_x
        nj1 = j + step_y
        if _inside(ni1, nj1, ti.static(self.nx), ti.static(self.ny)) and self._active(
            ni1, nj1
        ):
            value = self.phi[ni1, nj1]
            if distance == 2:
                ni2 = i + 2 * step_x
                nj2 = j + 2 * step_y
                if _inside(
                    ni2, nj2, ti.static(self.nx), ti.static(self.ny)
                ) and self._active(ni2, nj2):
                    value = self.phi[ni2, nj2]
                else:
                    value = self.phi[i, j]
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
            ti.floor(local.x + ti.static(0.5 * self.cfg.ice_width)), ti.i32
        )
        body_j = ti.cast(
            ti.floor(local.y + ti.static(0.5 * self.cfg.ice_height)), ti.i32
        )
        return ti.Vector([body_i, body_j])

    @ti.func
    def _sample_body_fraction(self, local):
        """Bilinearly sample the eroded material grid without wall clipping."""

        grid_x = local.x + ti.static(0.5 * self.cfg.ice_width - 0.5)
        grid_y = local.y + ti.static(0.5 * self.cfg.ice_height - 0.5)
        base_i = ti.cast(ti.floor(grid_x), ti.i32)
        base_j = ti.cast(ti.floor(grid_y), ti.i32)
        fraction = ti.cast(0.0, ti.f32)
        for di, dj in ti.static(ti.ndrange(2, 2)):
            body_i = base_i + di
            body_j = base_j + dj
            if 0 <= body_i < ti.static(self.cfg.ice_width) and 0 <= body_j < ti.static(
                self.cfg.ice_height
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
        """Evaluate the unclipped coverage and pose-dependent interpolation data.

        This pass deliberately ignores ``wall``.  The contact reduction must
        see a candidate body even when its predicted pose crosses a container
        plane.  The result is kept in ``_moving_body_fraction`` and the single
        simulator SDF is populated with the corresponding candidate geometry;
        the sharp ``solid`` mask is committed only after contact projection
        has accepted the pose.
        """

        half_x = ti.static(0.5 * self.cfg.ice_width)
        half_y = ti.static(0.5 * self.cfg.ice_height)
        dx = ti.static(self.cfg.dx)
        threshold = ti.static(self.cfg.thermal.solid_liquid_threshold)
        for i, j in self._moving_body_fraction:
            point = self._cell_point(i, j)
            local = self._body_local_coordinates(point)
            fraction = self._sample_body_fraction(local)

            # index of body_solid_fraction[body_i, body_j]
            index = self._nearest_body_index(local)
            grid_x = local.x + half_x - 0.5
            grid_y = local.y + half_y - 0.5
            base_i = ti.cast(ti.floor(grid_x), ti.i32)
            base_j = ti.cast(ti.floor(grid_y), ti.i32)
            valid = 0 <= index.x < ti.static(self.cfg.ice_width) and 0 <= index.y < ti.static(
                self.cfg.ice_height
            )
            self._moving_body_fraction[i, j] = fraction
            self.thermal.world_body_local_i[i, j] = index.x if valid else -1
            self.thermal.world_body_local_j[i, j] = index.y if valid else -1
            self.thermal.world_body_interp_base_i[i, j] = base_i
            self.thermal.world_body_interp_base_j[i, j] = base_j
            self.thermal.world_body_interp_fraction_x[i, j] = ti.cast(
                grid_x - ti.cast(base_i, ti.f32), ti.f32
            )
            self.thermal.world_body_interp_fraction_y[i, j] = ti.cast(
                grid_y - ti.cast(base_j, ti.f32), ti.f32
            )

            delta_x = ti.abs(local.x) - half_x
            delta_y = ti.abs(local.y) - half_y
            outside_x = ti.max(delta_x, 0.0)
            outside_y = ti.max(delta_y, 0.0)
            rectangle_distance = ti.sqrt(
                outside_x * outside_x + outside_y * outside_y
            ) + ti.min(ti.max(delta_x, delta_y), 0.0)
            distance = rectangle_distance * dx
            if valid:
                distance = (threshold - fraction) * dx
            self.sdf[i, j] = ti.cast(distance, ti.f32)

    @ti.kernel
    def _commit_moving_body_raster(self, previous_mode: ti.i32):
        """Commit a calculated coverage after contact has chosen the pose.

        ``previous_mode`` is 2 during initialization (old and new are the
        same), and 1 for a normal accepted time-level transition.
        """

        dx = ti.static(self.cfg.dx)
        threshold = ti.static(self.cfg.thermal.solid_liquid_threshold)
        for i, j in self._moving_body_fraction:
            fraction = self._moving_body_fraction[i, j]
            if self.wall[i, j] != 0:
                fraction = 0.0
            if previous_mode == 2:
                self.thermal.world_body_indicator_prev[i, j] = fraction
            elif previous_mode == 1:
                self.thermal.world_body_indicator_prev[i, j] = (
                    self.thermal.world_body_indicator[i, j]
                )
            self.thermal.world_body_indicator[i, j] = fraction
            if (
                self.thermal.world_body_local_i[i, j] >= 0
                and self.thermal.world_body_local_j[i, j] >= 0
            ):
                self.sdf[i, j] = (threshold - fraction) * dx

    @ti.kernel
    def _initialize_body(self):
        center_x = ti.static(float(self.cfg.ice_initial_center[0]))
        center_y = ti.static(float(self.cfg.ice_initial_center[1]))
        velocity_x = ti.static(float(self.cfg.ice_initial_velocity[0]))
        velocity_y = ti.static(float(self.cfg.ice_initial_velocity[1]))
        self.body_center[None] = ti.Vector([center_x, center_y])
        self.body_velocity[None] = ti.Vector([velocity_x, velocity_y])
        initial_angle = ti.static(float(self.cfg.ice_initial_angle))
        initial_omega = ti.static(float(self.cfg.ice_initial_angular_velocity))
        self.body_angle[None] = initial_angle
        self.body_angular_velocity[None] = initial_omega
        initial_mass = ti.cast(ti.static(self._body_mass), ti.f64)
        initial_inertia = ti.cast(ti.static(self._body_inertia), ti.f64)
        self.body_initial_mass_lattice[None] = initial_mass
        self.body_mass_lattice[None] = initial_mass
        self.body_inertia_lattice[None] = initial_inertia
        self.body_local_center_of_mass[None] = ti.Vector([0.0, 0.0])
        self.body_reference_origin[None] = ti.Vector([center_x, center_y])
        self.body_active[None] = ti.cast(1, ti.i8)
        self.cumulative_melted_mass_lattice[None] = 0.0
        self.cumulative_melted_momentum_lattice[None] = ti.Vector([0.0, 0.0])
        self.cumulative_fluid_melt_momentum_lattice[None] = ti.Vector([0.0, 0.0])
        self.cumulative_fluid_melt_carrier_momentum_lattice[None] = ti.Vector(
            [0.0, 0.0]
        )
        self.cumulative_fluid_melt_correction_momentum_lattice[None] = ti.Vector(
            [0.0, 0.0]
        )
        self.melt_momentum_residual_lattice[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_momentum_before[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_momentum_provisional[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_momentum_after[None] = ti.Vector([0.0, 0.0])
        self._thermal_body_momentum_before[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_before[None] = 0.0
        self._thermal_fluid_angular_momentum_provisional[None] = 0.0
        self._thermal_fluid_angular_momentum_after[None] = 0.0
        self._thermal_body_angular_momentum_before[None] = 0.0
        self._melt_momentum_correction[None] = ti.Vector([0.0, 0.0])
        self._melt_angular_momentum_correction[None] = 0.0
        self._melt_momentum_source_weight[None] = 0.0
        self._melt_momentum_wet_weight[None] = 0.0
        self._melt_momentum_source_target[None] = -1
        self._melt_momentum_wet_target[None] = -1
        self._melt_momentum_source_target_max[None] = -1
        self._melt_momentum_wet_target_max[None] = -1
        self._melt_momentum_weight_first_moment[None] = ti.Vector([0.0, 0.0])
        self._melt_momentum_weight_second_moment[None] = 0.0
        self._melt_momentum_weight_centroid[None] = ti.Vector([0.0, 0.0])
        self._melt_momentum_weight_polar_moment[None] = 0.0
        self.cumulative_melted_angular_momentum_lattice[None] = 0.0
        self.cumulative_fluid_melt_angular_momentum_lattice[None] = 0.0
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice[None] = 0.0
        self.cumulative_fluid_melt_correction_angular_momentum_lattice[None] = 0.0
        self.melt_angular_momentum_residual_lattice[None] = 0.0
        self.ale_water_residual_cells[None] = 0.0
        self.ale_energy_residual_j_m[None] = 0.0
        self.phase_aperture_water_residual_cells[None] = 0.0
        self.phase_aperture_energy_residual_j_m[None] = 0.0
        self.phase_aperture_capacity_margin_cells[None] = 0.0
        self.hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.hydrodynamic_torque[None] = 0.0
        self._body_contact_projection_changed[None] = ti.cast(0, ti.i8)

    @ti.kernel
    def _initialize_geometry(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        boundary = ti.static(self.cfg.boundary_cells)
        dx = ti.static(self.cfg.dx)
        for i, j in self.wall:
            is_wall = (
                i < boundary or i >= nx - boundary or j < boundary or j >= ny - boundary
            )
            self.wall[i, j] = ti.cast(1 if is_wall else 0, ti.i8)
            # Keep the canonical SDF in physical-length units, matching the
            # moving-body rasterizer and the public diagnostic field.
            distance = self._box_sdf(self._cell_point(i, j)) * dx
            is_solid = not is_wall and distance <= 0.0
            self.sdf[i, j] = distance
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)
            self.solid_prev[i, j] = self.solid[i, j]

    @ti.kernel
    def _initialize_fluid(self):
        water_width = ti.static(self.cfg.water_width)
        water_height = ti.static(self.cfg.water_height)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.phi:
            water_indicator = 1.0 if i < water_width and j < water_height else 0.0
            active = self._active(i, j)
            phi0 = water_indicator if active else 0.0
            velocity = ti.Vector([0.0, 0.0])
            if self.solid[i, j] == 1:
                velocity = self._body_velocity_at(self._cell_point(i, j))
            # In moving-body runs, inactive solid cells retain the covered
            # phase directly in phi until that cell is exposed again.
            reservoir_phase = phi0
            if self.solid[i, j] == 1:
                reservoir_phase = water_indicator
            self.phi[i, j] = reservoir_phase
            self.u[i, j] = velocity
            self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
            self.p[i, j] = 0.0
            material_density = _rho_mix(phi0, rho_water, rho_air)
            for q in range(Q):
                self.f[i, j, q] = _pressure_eq(q, 0.0, material_density, velocity)
                self.h[i, j, q] = _heq(q, phi0, velocity) if active else 0.0

    def _build_hydrostatic_reference(self):
        """Freeze a gauge-consistent hydrostatic density and pressure field.

        A full liquid domain retains the original two-dimensional analytic
        reference.  For a water--air pool, the warmed phase field is averaged
        over each active row and integrated vertically from the initial free
        surface.  Integrating ``rho_ref * g`` keeps pressure continuous across
        the diffuse interface; multiplying a local density by distance from a
        gauge point would instead introduce a spurious pressure jump.
        """

        full_liquid = self.cfg.water_height == self.ny - self.cfg.boundary_cells
        gravity = np.asarray(self._gravity_l, dtype=np.float64)
        if full_liquid:
            x, y = np.meshgrid(
                np.arange(self.nx, dtype=np.float64) + 0.5,
                np.arange(self.ny, dtype=np.float64) + 0.5,
                indexing="ij",
            )
            gauge_x = 0.5 * self.nx
            gauge_y = 0.5 * self.ny
            reference_density = np.full(
                (self.nx, self.ny), self._rho_water_l, dtype=np.float64
            )
            reference_pressure = self._rho_water_l * (
                gravity[0] * (x - gauge_x) + gravity[1] * (y - gauge_y)
            )
        else:
            phase = np.clip(self.phi.to_numpy().astype(np.float64), 0.0, 1.0)
            active = (self.wall.to_numpy() == 0) & (self.solid.to_numpy() == 0)
            phase_profile = np.empty(self.ny, dtype=np.float64)
            for j in range(self.ny):
                row_active = active[:, j]
                if np.any(row_active):
                    phase_profile[j] = float(np.mean(phase[row_active, j]))
                else:
                    phase_profile[j] = 1.0 if j < self.cfg.water_height else 0.0

            density_profile = (
                self._rho_air_l + (self._rho_water_l - self._rho_air_l) * phase_profile
            )
            pressure_profile = np.zeros(self.ny, dtype=np.float64)
            surface = int(self.cfg.water_height)
            gy = float(gravity[1])

            # Cell-centred finite-volume integration with p_H(surface)=0.
            # This makes adjacent-cell differences exactly the trapezoidal
            # integral of rho_ref*g over one lattice spacing.
            pressure_profile[surface - 1] = -0.5 * gy * density_profile[surface - 1]
            for j in range(surface - 2, -1, -1):
                pressure_profile[j] = pressure_profile[j + 1] - 0.5 * gy * (
                    density_profile[j] + density_profile[j + 1]
                )
            pressure_profile[surface] = 0.5 * gy * density_profile[surface]
            for j in range(surface + 1, self.ny):
                pressure_profile[j] = pressure_profile[j - 1] + 0.5 * gy * (
                    density_profile[j - 1] + density_profile[j]
                )

            reference_density = np.broadcast_to(
                density_profile[np.newaxis, :], (self.nx, self.ny)
            ).copy()
            reference_pressure = np.broadcast_to(
                pressure_profile[np.newaxis, :], (self.nx, self.ny)
            ).copy()

        if not (
            np.isfinite(reference_density).all()
            and np.isfinite(reference_pressure).all()
        ):
            raise RuntimeError(
                "hydrostatic reference construction produced non-finite values"
            )
        if self.hydrostatic_reference_density is not None:
            self.hydrostatic_reference_density.from_numpy(
                reference_density.astype(np.float32)
            )
        self.hydrostatic_reference_pressure.from_numpy(
            reference_pressure.astype(np.float32)
        )

    @ti.kernel
    def _initialize_hydrostatic_equilibrium(self):
        """Set the two-phase fluid to the frozen-reference rest state."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        rest = ti.Vector([0.0, 0.0])
        for i, j in self.phi:
            point = self._cell_point(i, j)
            hydrostatic_pressure = self.hydrostatic_reference_pressure[i, j]
            # In hydrostatic mode this reservoir stores only the dynamic
            # pressure residual.  The spatial reference is always added back
            # at the newly exposed cell, so rigid motion cannot advect a
            # hydrostatic gauge value with the body.
            if self._active(i, j):
                phi = self.phi[i, j]
                material_density = _rho_mix(phi, rho_water, rho_air)
                self.u[i, j] = rest
                self.p[i, j] = hydrostatic_pressure
                for q in range(Q):
                    # Only p_dyn is carried by the populations.  The frozen
                    # p_H traction remains in the Guo completion.
                    self.f[i, j, q] = _pressure_eq(q, 0.0, material_density, rest)
                    self.h[i, j, q] = _heq(q, phi, rest)
            else:
                boundary_velocity = ti.Vector([0.0, 0.0])
                if self.solid[i, j] == 1:
                    boundary_velocity = self._body_velocity_at(point)
                self.u[i, j] = boundary_velocity
                self.p[i, j] = 0.0

    @ti.kernel
    def _initialize_moving_body_geometry(self):
        """Seed the sharp mask from the initial rasterized canonical SDF."""

        for i, j in self.solid:
            active_body = self.body_active[None] != 0
            is_solid = active_body and self.wall[i, j] == 0 and self.sdf[i, j] <= 0.0
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)
            self.solid_prev[i, j] = self.solid[i, j]

    @ti.kernel
    def _update_solid_mask(self):
        """Commit the already-rasterized SDF to the sharp solid mask.

        ``_solve_contact_and_rasterize`` has already evaluated the accepted
        pose into the simulator-owned SDF.  This pass records the previous
        sharp mask and derives the new mask from that same field; no second
        SDF storage or copy is needed.
        """

        for i, j in self.solid:
            self.solid_prev[i, j] = self.solid[i, j]
            distance = self.sdf[i, j]
            active_body = self.body_active[None] != 0
            is_solid = active_body and self.wall[i, j] == 0 and distance <= 0.0
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)

    def _update_moving_body_mass_properties(self, *, initialize: bool = False):
        """Reduce the eroding material grid and update the rigid COM state."""

        self._reduce_moving_body_mass_properties()
        self._apply_moving_body_mass_properties(1 if initialize else 0)

    @ti.kernel
    def _synchronize_moving_thermal_diagnostics(self):
        """Expose the current conservative ALE residuals to common output."""

        inverse_cell_area = ti.static(1.0 / (self.cfg.dx * self.cfg.dx))
        self.ale_water_residual_cells[None] = (
            self.thermal.ale_water_volume_residual_m2[None] * inverse_cell_area
        )
        self.ale_energy_residual_j_m[None] = self.thermal.ale_water_energy_residual_j_m[
            None
        ]
        self.phase_aperture_water_residual_cells[None] = (
            self.thermal.phase_aperture_volume_residual_m2[None] * inverse_cell_area
        )
        self.phase_aperture_energy_residual_j_m[None] = (
            self.thermal.phase_aperture_energy_residual_j_m[None]
        )
        self.phase_aperture_capacity_margin_cells[None] = (
            self.thermal.phase_aperture_capacity_margin_m2[None] * inverse_cell_area
        )

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
        for i, j in self.phi:
            if self._active(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(Q):
                    momentum += ti.cast(_c(q), ti.f64) * ti.cast(
                        self.f[i, j, q], ti.f64
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
        """Reduce carrier/topology momentum and correction weights."""

        self._thermal_fluid_momentum_provisional[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_provisional[None] = 0.0
        self._melt_momentum_source_weight[None] = 0.0
        self._melt_momentum_wet_weight[None] = 0.0
        self._melt_momentum_source_target[None] = self.nx * self.ny
        self._melt_momentum_wet_target[None] = self.nx * self.ny
        self._melt_momentum_source_target_max[None] = -1
        self._melt_momentum_wet_target_max[None] = -1
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        ny = ti.static(self.ny)
        for i, j in self.phi:
            if self._active(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(Q):
                    momentum += ti.cast(_c(q), ti.f64) * ti.cast(
                        self.f[i, j, q], ti.f64
                    )
                ti.atomic_add(
                    self._thermal_fluid_momentum_provisional[None].x,
                    momentum.x,
                )
                ti.atomic_add(
                    self._thermal_fluid_momentum_provisional[None].y,
                    momentum.y,
                )
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                ti.atomic_add(
                    self._thermal_fluid_angular_momentum_provisional[None],
                    point.x * momentum.y - point.y * momentum.x,
                )
                phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                if phase > cutoff:
                    encoded = i * ny + j
                    source_weight = ti.max(
                        0.0, self.thermal.water_melt_mass_source[i, j]
                    )
                    ti.atomic_add(
                        self._melt_momentum_source_weight[None], source_weight
                    )
                    ti.atomic_add(
                        self._melt_momentum_wet_weight[None],
                        ti.cast(phase, ti.f64),
                    )
                    ti.atomic_min(self._melt_momentum_wet_target[None], encoded)
                    ti.atomic_max(self._melt_momentum_wet_target_max[None], encoded)
                    if source_weight > 0.0:
                        ti.atomic_min(self._melt_momentum_source_target[None], encoded)
                        ti.atomic_max(
                            self._melt_momentum_source_target_max[None], encoded
                        )

    @ti.kernel
    def _reduce_melt_momentum_weight_geometry(self, use_source: ti.i32):
        self._melt_momentum_weight_first_moment[None] = ti.Vector([0.0, 0.0])
        self._melt_momentum_weight_second_moment[None] = 0.0
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        for i, j in self.phi:
            phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
            weight = ti.cast(0.0, ti.f64)
            if self._active(i, j) and phase > cutoff:
                if use_source != 0:
                    weight = ti.max(0.0, self.thermal.water_melt_mass_source[i, j])
                else:
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

    @ti.kernel
    def _finish_melt_momentum_weight_geometry(self, use_source: ti.i32):
        weight = self._melt_momentum_wet_weight[None]
        if use_source != 0:
            weight = self._melt_momentum_source_weight[None]
        centroid = ti.Vector([0.0, 0.0], dt=ti.f64)
        polar = ti.cast(0.0, ti.f64)
        if weight > 1.0e-30:
            centroid = self._melt_momentum_weight_first_moment[None] / weight
            polar = self._melt_momentum_weight_second_moment[None] - weight * (
                centroid.dot(centroid)
            )
        self._melt_momentum_weight_centroid[None] = centroid
        self._melt_momentum_weight_polar_moment[None] = ti.max(0.0, polar)

    @ti.kernel
    def _prepare_melt_momentum_correction(self):
        carrier_change = (
            self._thermal_fluid_momentum_provisional[None]
            - self._thermal_fluid_momentum_before[None]
        )
        expected_change = (
            self.cumulative_melted_momentum_lattice[None]
            - self._thermal_body_momentum_before[None]
        )
        self._melt_momentum_correction[None] = expected_change - carrier_change
        carrier_angular_change = (
            self._thermal_fluid_angular_momentum_provisional[None]
            - self._thermal_fluid_angular_momentum_before[None]
        )
        expected_angular_change = (
            self.cumulative_melted_angular_momentum_lattice[None]
            - self._thermal_body_angular_momentum_before[None]
        )
        self._melt_angular_momentum_correction[None] = (
            expected_angular_change - carrier_angular_change
        )

    @ti.kernel
    def _apply_melt_momentum_correction(self, use_source: ti.i32):
        """Apply a zero-mass D2Q9 lift satisfying three moment constraints."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        denominator = self._melt_momentum_wet_weight[None]
        if use_source != 0:
            denominator = self._melt_momentum_source_weight[None]
        correction = self._melt_momentum_correction[None]
        angular_correction = self._melt_angular_momentum_correction[None]
        centroid = self._melt_momentum_weight_centroid[None]
        polar = self._melt_momentum_weight_polar_moment[None]
        rotational_scale = ti.cast(0.0, ti.f64)
        if polar > 1.0e-30:
            translation_angular = centroid.x * correction.y - centroid.y * correction.x
            rotational_scale = (angular_correction - translation_angular) / polar
        for i, j in self.phi:
            phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
            eligible = self._active(i, j) and phase > cutoff
            weight = ti.cast(0.0, ti.f64)
            if eligible:
                if use_source != 0:
                    weight = ti.max(0.0, self.thermal.water_melt_mass_source[i, j])
                else:
                    weight = ti.cast(phase, ti.f64)
            if weight > 0.0 and denominator > 1.0e-30:
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                relative = point - centroid
                rotational_direction = ti.Vector([-relative.y, relative.x])
                delta_momentum_f64 = weight * (
                    correction / denominator + rotational_scale * rotational_direction
                )
                delta_momentum = ti.cast(delta_momentum_f64, ti.f32)
                density = _rho_mix(phase, rho_water, rho_air)
                self.u[i, j] += delta_momentum / ti.max(density, 1.0e-12)
                for q in range(Q):
                    direction = ti.cast(_c(q), ti.f32)
                    self.f[i, j, q] += 3.0 * _w(q) * direction.dot(delta_momentum)

    @ti.kernel
    def _reduce_final_melt_momentum(self):
        self._thermal_fluid_momentum_after[None] = ti.Vector([0.0, 0.0])
        self._thermal_fluid_angular_momentum_after[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                momentum = ti.Vector([0.0, 0.0], dt=ti.f64)
                for q in range(Q):
                    momentum += ti.cast(_c(q), ti.f64) * ti.cast(
                        self.f[i, j, q], ti.f64
                    )
                ti.atomic_add(self._thermal_fluid_momentum_after[None].x, momentum.x)
                ti.atomic_add(self._thermal_fluid_momentum_after[None].y, momentum.y)
                point = ti.Vector([ti.cast(i, ti.f64) + 0.5, ti.cast(j, ti.f64) + 0.5])
                ti.atomic_add(
                    self._thermal_fluid_angular_momentum_after[None],
                    point.x * momentum.y - point.y * momentum.x,
                )

    @ti.kernel
    def _prepare_remaining_melt_momentum_correction(self):
        expected_change = (
            self.cumulative_melted_momentum_lattice[None]
            - self._thermal_body_momentum_before[None]
        )
        actual_change = (
            self._thermal_fluid_momentum_after[None]
            - self._thermal_fluid_momentum_before[None]
        )
        self._melt_momentum_correction[None] = expected_change - actual_change
        expected_angular_change = (
            self.cumulative_melted_angular_momentum_lattice[None]
            - self._thermal_body_angular_momentum_before[None]
        )
        actual_angular_change = (
            self._thermal_fluid_angular_momentum_after[None]
            - self._thermal_fluid_angular_momentum_before[None]
        )
        self._melt_angular_momentum_correction[None] = (
            expected_angular_change - actual_angular_change
        )

    @ti.kernel
    def _apply_local_melt_momentum_correction(self, use_source: ti.i32):
        """Close f32 remainders with a two-cell force/couple pair."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        ny = ti.static(self.ny)
        encoded_1 = self._melt_momentum_wet_target[None]
        encoded_2 = self._melt_momentum_wet_target_max[None]
        if use_source != 0:
            encoded_1 = self._melt_momentum_source_target[None]
            encoded_2 = self._melt_momentum_source_target_max[None]
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
                phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                density = _rho_mix(phase, rho_water, rho_air)
                self.u[i, j] += delta_momentum / ti.max(density, 1.0e-12)
                for q in range(Q):
                    direction = ti.cast(_c(q), ti.f32)
                    self.f[i, j, q] += 3.0 * _w(q) * direction.dot(delta_momentum)

    @ti.kernel
    def _finish_melt_momentum_coupling(self):
        carrier_change = (
            self._thermal_fluid_momentum_provisional[None]
            - self._thermal_fluid_momentum_before[None]
        )
        actual_change = (
            self._thermal_fluid_momentum_after[None]
            - self._thermal_fluid_momentum_before[None]
        )
        correction_change = (
            self._thermal_fluid_momentum_after[None]
            - self._thermal_fluid_momentum_provisional[None]
        )
        self.cumulative_fluid_melt_carrier_momentum_lattice[None] += carrier_change
        self.cumulative_fluid_melt_correction_momentum_lattice[None] += (
            correction_change
        )
        self.cumulative_fluid_melt_momentum_lattice[None] += actual_change
        self.melt_momentum_residual_lattice[None] = (
            self.cumulative_fluid_melt_momentum_lattice[None]
            - self.cumulative_melted_momentum_lattice[None]
        )
        carrier_angular_change = (
            self._thermal_fluid_angular_momentum_provisional[None]
            - self._thermal_fluid_angular_momentum_before[None]
        )
        actual_angular_change = (
            self._thermal_fluid_angular_momentum_after[None]
            - self._thermal_fluid_angular_momentum_before[None]
        )
        correction_angular_change = (
            self._thermal_fluid_angular_momentum_after[None]
            - self._thermal_fluid_angular_momentum_provisional[None]
        )
        self.cumulative_fluid_melt_carrier_angular_momentum_lattice[None] += (
            carrier_angular_change
        )
        self.cumulative_fluid_melt_correction_angular_momentum_lattice[None] += (
            correction_angular_change
        )
        self.cumulative_fluid_melt_angular_momentum_lattice[None] += (
            actual_angular_change
        )
        self.melt_angular_momentum_residual_lattice[None] = (
            self.cumulative_fluid_melt_angular_momentum_lattice[None]
            - self.cumulative_melted_angular_momentum_lattice[None]
        )

    def _finalize_melt_momentum_coupling(self):
        """Close all thermal/refill/projection first-moment changes to the body."""

        if not self._moving_melt_momentum_pending:
            return
        self._reduce_provisional_melt_momentum()
        wet_weight = float(self._melt_momentum_wet_weight[None])
        # The residual closed here is not only the physical momentum of the
        # newly melted mass.  It also contains carrier changes introduced by
        # moving-node refill and the global phase-volume projection.  Forcing
        # that global residual through the few instantaneous melt-source
        # cells can create an arbitrarily large couple when one source lies in
        # the dilute contact line.  The phase-weighted wet support is the
        # minimum-energy admissible linear/angular-momentum projection and
        # remains well conditioned as the local melt stencil changes.
        use_source = 0
        self._reduce_melt_momentum_weight_geometry(use_source)
        self._finish_melt_momentum_weight_geometry(use_source)
        self._prepare_melt_momentum_correction()
        correction = np.asarray(self._melt_momentum_correction[None], dtype=np.float64)
        angular_correction = float(self._melt_angular_momentum_correction[None])
        if wet_weight <= 1.0e-30:
            if (
                float(np.linalg.norm(correction)) > 1.0e-12
                or abs(angular_correction) > 1.0e-12
            ):
                raise RuntimeError(
                    "melt momentum has no legal wet LBM cell for conservative "
                    "linear/angular-momentum closure"
                )
        else:
            centroid = np.asarray(
                self._melt_momentum_weight_centroid[None], dtype=np.float64
            )
            independent_couple = angular_correction - (
                centroid[0] * correction[1] - centroid[1] * correction[0]
            )
            polar = float(self._melt_momentum_weight_polar_moment[None])
            if polar <= 1.0e-30 and abs(independent_couple) > 1.0e-12:
                raise RuntimeError(
                    "melt momentum correction requires two distinct wet cells"
                )
            if (
                float(np.linalg.norm(correction)) > 1.0e-14
                or abs(angular_correction) > 1.0e-14
            ):
                self._apply_melt_momentum_correction(use_source)
        self._reduce_final_melt_momentum()
        # A distributed f32 lift leaves a few ulps after the f64 reduction.
        # Two localized remainder passes make the reported residual an actual
        # end-to-end first-moment audit rather than an arithmetic artifact.
        for _ in range(2):
            self._prepare_remaining_melt_momentum_correction()
            self._apply_local_melt_momentum_correction(use_source)
            self._reduce_final_melt_momentum()
        self._finish_melt_momentum_coupling()
        self._moving_melt_momentum_pending = False

    @ti.kernel
    def _reduce_moving_body_mass_properties(self):
        self._body_reduced_mass[None] = 0.0
        self._body_reduced_first_moment[None] = ti.Vector([0.0, 0.0])
        self._body_reduced_inertia_origin[None] = 0.0
        inverse_reference_cell_mass = ti.cast(
            ti.static(1.0 / (self.cfg.rho_water * self.cfg.dx * self.cfg.dx)),
            ti.f64,
        )
        half_x = ti.static(0.5 * self.cfg.ice_width)
        half_y = ti.static(0.5 * self.cfg.ice_height)
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
            ti.static(self._body_inertia), 1.0
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
            lost_mass = ti.max(old_mass - new_mass, 0.0)
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
            self.cumulative_melted_mass_lattice[None] += lost_mass
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
        self.body_center[None] = ti.cast(origin + new_local_world, ti.f32)
        self.body_mass_lattice[None] = new_mass
        self.body_inertia_lattice[None] = new_inertia
        if mechanically_resolved:
            self.body_active[None] = ti.cast(1, ti.i8)
        else:
            self.body_active[None] = ti.cast(0, ti.i8)
            self.body_velocity[None] = ti.Vector([0.0, 0.0])
            self.body_angular_velocity[None] = 0.0
            self._body_contact_geometry_active[None] = ti.cast(0, ti.i8)
            for index in ti.static(range(4)):
                self._body_contact_support_extrema[index] = 0.0

    @ti.kernel
    def _refill_changed_nodes(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.phi:
            became_solid = (
                self.solid_prev[i, j] == 0
                and self.solid[i, j] == 1
                and self.wall[i, j] == 0
            )
            became_fluid = (
                self.solid_prev[i, j] == 1
                and self.solid[i, j] == 0
                and self.wall[i, j] == 0
            )
            if became_solid:
                covered_phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                reservoir_pressure = (
                    self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                )
                velocity = self._body_velocity_at(self._cell_point(i, j))
                self.phi[i, j] = covered_phase
                self.u[i, j] = velocity
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                self.p[i, j] = reservoir_pressure
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
                        and _inside(ni, nj, nx, ny)
                        and self._active(ni, nj)
                        and self.solid_prev[ni, nj] == 0
                    ):
                        phi_sum += ti.min(1.0, ti.max(0.0, self.phi[ni, nj]))
                        neighbor_pressure = (
                            self.p[ni, nj] - self.hydrostatic_reference_pressure[ni, nj]
                        )
                        pressure_sum += neighbor_pressure
                        velocity_sum += self.u[ni, nj]
                        count += 1.0
                phi0 = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                pressure0 = self.p[i, j]
                velocity0 = self._body_velocity_at(self._cell_point(i, j))
                if count > 0.0:
                    phi0 = phi_sum / count
                    pressure0 = pressure_sum / count
                    velocity0 = velocity_sum / count
                pressure0 += self.hydrostatic_reference_pressure[i, j]
                fresh_density = _rho_mix(phi0, rho_water, rho_air)
                self.phi[i, j] = phi0
                self.u[i, j] = velocity0
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                self.p[i, j] = pressure0
                dynamic_pressure = pressure0 - self.hydrostatic_reference_pressure[i, j]
                for q in range(Q):
                    self.f[i, j, q] = _pressure_eq(
                        q, dynamic_pressure, fresh_density, velocity0
                    )
                    self.h[i, j, q] = _heq(q, phi0, velocity0)

    # ------------------------------------------------------------------
    # Two-phase pressure--momentum LBM

    @ti.func
    def _fluid_force_viscosity_and_gradient(self, i, j):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        gravity = ti.Vector(
            [ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])]
        )
        beta = ti.static(12.0 * self._sigma_l / self.cfg.interface_width)
        kapa = ti.static(1.5 * self._sigma_l * self.cfg.interface_width)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        cd = ti.static(self.cfg.cd)
        phi = self.phi[i, j]
        bounded_phi = ti.min(1.0, ti.max(0.0, phi))
        density = _rho_mix(bounded_phi, rho_water, rho_air)

        gradient = ti.Vector([0.0, 0.0])
        source_gradient = ti.Vector([0.0, 0.0])
        laplacian = 0.0
        alpha = ti.static(1.0 / 3.0)
        for q in range(Q):
            direction_i = _c(q)
            direction = ti.cast(direction_i, ti.f32)
            phi1 = self._phase_neighbor(i, j, direction_i.x, direction_i.y, 1)
            phi2 = self._phase_neighbor(i, j, direction_i.x, direction_i.y, 2)
            source_gradient += 3.0 * _w(q) * direction * (phi1 - phi)
            gradient += (1.0 - alpha) * 3.0 * _w(q) * direction * (phi1 - phi)
            gradient += (
                alpha * 1.5 * _w(q) * direction * (4.0 * phi1 - phi2 - 3.0 * phi)
            )
            laplacian += 6.0 * _w(q) * (phi1 - phi)

        # Assemble gravity sources as force densities.  The frozen
        # hydrostatic reference removes only the base far-field load; the
        # temperature-dependent density anomaly must remain additive.
        reference_density = self.hydrostatic_reference_density[i, j]
        gravity_force_density = (density - reference_density) * gravity
        reference_temperature = ti.static(
            float(self.cfg.thermal.buoyancy_reference_temperature_c)
        )
        temperature = ti.cast(self.temperature[i, j], ti.f32)
        expansion = ti.static(float(self.cfg.thermal.thermal_expansion_water_1_k))
        density_anomaly_ratio = -expansion * (temperature - reference_temperature)

        # The thermal field deliberately treats the water/air interface as
        # adiabatic and stores the air temperature on phi < 0.5.  A smooth
        # water-side gate prevents that unrelated air value from forcing the
        # diffuse layer.
        water_weight = ti.min(1.0, ti.max(0.0, 2.0 * bounded_phi - 1.0))
        gravity_force_density += (
            water_weight * rho_water * density_anomaly_ratio * gravity
        )
        force = gravity_force_density / density
        chemical = (
            4.0 * beta * bounded_phi * (bounded_phi - 1.0) * (bounded_phi - 0.5)
            - kapa * laplacian
        )
        force += chemical * gradient / density

        velocity = self.u[i, j]
        dux = 0.0
        duy = 0.0
        dvx = 0.0
        dvy = 0.0
        for q in range(Q):
            direction_i = _c(q)
            direction = ti.cast(direction_i, ti.f32)
            ni = i + direction_i.x
            nj = j + direction_i.y
            neighbor_velocity = ti.Vector([0.0, 0.0])
            if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                neighbor_velocity = self.u[ni, nj]
            elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                neighbor_velocity = self._body_velocity_at(self._cell_point(ni, nj))
            difference = neighbor_velocity - velocity
            dux += 3.0 * _w(q) * direction.x * difference.x
            duy += 3.0 * _w(q) * direction.y * difference.x
            dvx += 3.0 * _w(q) * direction.x * difference.y
            dvy += 3.0 * _w(q) * direction.y * difference.y
        artificial_viscosity = (
            cd * cd * ti.sqrt(2.0 * (dux * dux + dvy * dvy + 0.5 * (duy + dvx) ** 2))
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
    def _collide_velocity(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nu_water = ti.static(self._nu_water_l)
        nu_air = ti.static(self._nu_air_l)
        for i, j in self.phi:
            if self._active(i, j):
                force_data = self._fluid_force_viscosity_and_gradient(i, j)
                force = ti.Vector([force_data[0], force_data[1]])
                artificial_vis = force_data[2]
                grad_phi = ti.Vector([force_data[3], force_data[4]])
                self.fluid_force[i, j] = force
                phi = self.phi[i, j]
                velocity = self.u[i, j] + 0.5 * force
                phase_fraction = ti.min(1.0, ti.max(0.0, phi))
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
                for q in ti.static(range(1, Q)):
                    direction = _c(q)
                    ni = i + direction.x
                    nj = j + direction.y
                    if _inside(ni, nj, self.nx, self.ny):
                        if self.solid[ni, nj] == 1:
                            sharp_solid_neighbor = 1.0
                shear_indicator = ti.max(1.0 - phase_fraction, sharp_solid_neighbor)
                tau_floor = 0.5 + (interface_tau - 0.5) * shear_indicator
                shear_tau = ti.max(physical_tau, tau_floor)
                # Liang et al. PRE 97, 033309 (2018), Eqs. (15)--(23).
                material_density = _rho_mix(phi, rho_water, rho_air)
                dynamic_pressure = (
                    self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                )
                force_density = material_density * force
                delta_rho = rho_water - rho_air
                m00 = 0.0
                m10 = 0.0
                m01 = 0.0
                m20 = 0.0
                m02 = 0.0
                m11 = 0.0
                m21 = 0.0
                m12 = 0.0
                m22 = 0.0
                meq00 = 0.0
                meq10 = 0.0
                meq01 = 0.0
                meq20 = 0.0
                meq02 = 0.0
                meq11 = 0.0
                meq21 = 0.0
                meq12 = 0.0
                meq22 = 0.0
                src00 = 0.0
                src10 = 0.0
                src01 = 0.0
                src20 = 0.0
                src02 = 0.0
                src11 = 0.0
                src21 = 0.0
                src12 = 0.0
                src22 = 0.0
                for q in range(Q):
                    cc = ti.cast(_c(q), ti.f32)
                    cx = cc.x
                    cy = cc.y
                    value = self.f[i, j, q]
                    equilibrium = _pressure_eq(
                        q,
                        dynamic_pressure,
                        material_density,
                        velocity,
                    )
                    self.f_pre_collision_neq[i, j, q] = value - equilibrium
                    base_source = (
                        _w(q)
                        * 3.0
                        * (
                            cc.dot(force_density)
                            + delta_rho * cc.dot(velocity) * cc.dot(grad_phi)
                        )
                    )
                    m00 += value
                    m10 += cx * value
                    m01 += cy * value
                    m20 += cx * cx * value
                    m02 += cy * cy * value
                    m11 += cx * cy * value
                    m21 += cx * cx * cy * value
                    m12 += cx * cy * cy * value
                    m22 += cx * cx * cy * cy * value
                    meq00 += equilibrium
                    meq10 += cx * equilibrium
                    meq01 += cy * equilibrium
                    meq20 += cx * cx * equilibrium
                    meq02 += cy * cy * equilibrium
                    meq11 += cx * cy * equilibrium
                    meq21 += cx * cx * cy * equilibrium
                    meq12 += cx * cy * cy * equilibrium
                    meq22 += cx * cx * cy * cy * equilibrium
                    src00 += base_source
                    src10 += cx * base_source
                    src01 += cy * base_source
                    src20 += cx * cx * base_source
                    src02 += cy * cy * base_source
                    src11 += cx * cy * base_source
                    src21 += cx * cx * cy * base_source
                    src12 += cx * cy * cy * base_source
                    src22 += cx * cx * cy * cy * base_source

                # MRT extension of Liang's BGK equation.  The two deviatoric
                # stresses use the shear envelope; lower-order hydrodynamic
                # moments retain the physical BGK rate, while the trace and
                # ghost modes relax in one step.  The trapezoidal source
                # prefactor is applied per moment.
                physical_omega = 1.0 / physical_tau
                shear_omega = 1.0 / shear_tau
                source_physical = 1.0 - 0.5 * physical_omega
                source_shear = 1.0 - 0.5 * shear_omega
                post00 = m00 - physical_omega * (m00 - meq00) + source_physical * src00
                post10 = m10 - physical_omega * (m10 - meq10) + source_physical * src10
                post01 = m01 - physical_omega * (m01 - meq01) + source_physical * src01
                trace_eq = meq20 + meq02
                trace_src = src20 + src02
                post_trace = trace_eq + 0.5 * trace_src
                difference = m20 - m02
                difference_eq = meq20 - meq02
                difference_src = src20 - src02
                post_difference = (
                    difference
                    - shear_omega * (difference - difference_eq)
                    + source_shear * difference_src
                )
                post20 = 0.5 * (post_trace + post_difference)
                post02 = 0.5 * (post_trace - post_difference)
                post11 = m11 - shear_omega * (m11 - meq11) + source_shear * src11
                post21 = meq21 + 0.5 * src21
                post12 = meq12 + 0.5 * src12
                post22 = meq22 + 0.5 * src22
                for q in range(Q):
                    self.f_post[i, j, q] = _reconstruct_central(
                        _c(q).x,
                        _c(q).y,
                        0.0,
                        0.0,
                        post00,
                        post10,
                        post01,
                        post20,
                        post02,
                        post11,
                        post21,
                        post12,
                        post22,
                    )

    @ti.kernel
    def _collide_phase(self):
        interface_width = ti.static(self.cfg.interface_width)
        tau_inv_phi = ti.static(1.0 / (3.0 * self.cfg.mobility + 0.5))
        for i, j in self.phi:
            if self._active(i, j):
                phi = self.phi[i, j]
                velocity = self.u[i, j] + 0.5 * self.fluid_force[i, j]
                grad_phi = ti.Vector([0.0, 0.0])
                alpha = ti.static(1.0 / 3.0)
                for qq in range(Q):
                    cc_i = _c(qq)
                    cc = ti.cast(cc_i, ti.f32)
                    phi1 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 1)
                    phi2 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 2)
                    grad_phi += (1.0 - alpha) * 3.0 * _w(qq) * cc * (phi1 - phi)
                    grad_phi += (
                        alpha * 1.5 * _w(qq) * cc * (4.0 * phi1 - phi2 - 3.0 * phi)
                    )
                normal = grad_phi / (grad_phi.norm() + 1.0e-14)
                compression = 4.0 * phi * (1.0 - phi) / interface_width

                hm00 = 0.0
                hm10 = 0.0
                hm01 = 0.0
                hm20 = 0.0
                hm02 = 0.0
                hm11 = 0.0
                hm21 = 0.0
                hm12 = 0.0
                hm22 = 0.0
                he00 = 0.0
                he10 = 0.0
                he01 = 0.0
                he20 = 0.0
                he02 = 0.0
                he11 = 0.0
                he21 = 0.0
                he12 = 0.0
                he22 = 0.0
                hf00 = 0.0
                hf10 = 0.0
                hf01 = 0.0
                hf20 = 0.0
                hf02 = 0.0
                hf11 = 0.0
                hf21 = 0.0
                hf12 = 0.0
                hf22 = 0.0
                for q in range(Q):
                    cx = ti.cast(_c(q).x, ti.f32)
                    cy = ti.cast(_c(q).y, ti.f32)
                    rx = cx - velocity.x
                    ry = cy - velocity.y
                    rx2 = rx * rx
                    ry2 = ry * ry
                    h_value = self.h[i, j, q]
                    h_equilibrium = _heq(q, phi, velocity)
                    phase_force = _w(q) * compression * ti.Vector([cx, cy]).dot(normal)
                    hm00 += h_value
                    hm10 += h_value * rx
                    hm01 += h_value * ry
                    hm20 += h_value * rx2
                    hm02 += h_value * ry2
                    hm11 += h_value * rx * ry
                    hm21 += h_value * rx2 * ry
                    hm12 += h_value * rx * ry2
                    hm22 += h_value * rx2 * ry2
                    he00 += h_equilibrium
                    he10 += h_equilibrium * rx
                    he01 += h_equilibrium * ry
                    he20 += h_equilibrium * rx2
                    he02 += h_equilibrium * ry2
                    he11 += h_equilibrium * rx * ry
                    he21 += h_equilibrium * rx2 * ry
                    he12 += h_equilibrium * rx * ry2
                    he22 += h_equilibrium * rx2 * ry2
                    hf00 += phase_force
                    hf10 += phase_force * rx
                    hf01 += phase_force * ry
                    hf20 += phase_force * rx2
                    hf02 += phase_force * ry2
                    hf11 += phase_force * rx * ry
                    hf21 += phase_force * rx2 * ry
                    hf12 += phase_force * rx * ry2
                    hf22 += phase_force * rx2 * ry2
                kh00 = he00 + 0.5 * hf00
                kh10 = (
                    hm10
                    - tau_inv_phi * (hm10 - he10)
                    + (1.0 - 0.5 * tau_inv_phi) * hf10
                )
                kh01 = (
                    hm01
                    - tau_inv_phi * (hm01 - he01)
                    + (1.0 - 0.5 * tau_inv_phi) * hf01
                )
                kh20 = he20 + 0.5 * hf20
                kh02 = he02 + 0.5 * hf02
                kh11 = he11 + 0.5 * hf11
                kh21 = he21 + 0.5 * hf21
                kh12 = he12 + 0.5 * hf12
                kh22 = he22 + 0.5 * hf22
                for q in range(Q):
                    self.h_post[i, j, q] = _reconstruct_central(
                        _c(q).x,
                        _c(q).y,
                        velocity.x,
                        velocity.y,
                        kh00,
                        kh10,
                        kh01,
                        kh20,
                        kh02,
                        kh11,
                        kh21,
                        kh12,
                        kh22,
                    )

    @ti.kernel
    def _stream(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.phi:
            if self._active(i, j):
                # Sum all cut links owned by this cell locally.  Besides
                # fixing their order, this changes three global atomics per
                # link into three atomics per boundary cell.
                cell_impulse = ti.Vector([0.0, 0.0], dt=ti.f64)
                cell_torque = ti.cast(0.0, ti.f64)
                has_cut_link = 0
                for q in range(Q):
                    direction = _c(q)
                    direction_f = ti.cast(direction, ti.f32)
                    ni = i + direction.x
                    nj = j + direction.y
                    outgoing_f = self.f_post[i, j, q]
                    outgoing_h = self.h_post[i, j, q]
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        self.f[ni, nj, q] = outgoing_f
                        self.h[ni, nj, q] = outgoing_h
                    elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                        sdf_fluid = ti.max(self.sdf[i, j], 1.0e-6)
                        eta = ti.min(
                            0.95,
                            ti.max(
                                0.05,
                                sdf_fluid / (sdf_fluid - self.sdf[ni, nj] + 1.0e-12),
                            ),
                        )
                        boundary_point = self._cell_point(i, j) + eta * direction_f
                        boundary_velocity = self._body_velocity_at(boundary_point)
                        density = _rho_mix(
                            ti.min(1.0, ti.max(0.0, self.phi[i, j])),
                            rho_water,
                            rho_air,
                        )
                        # Tao et al. (2018), Eqs. (13)--(16), is a true
                        # single-node rule: every population term comes from
                        # the fluid node that owns this cut link; no back-node
                        # population is read.  Apply it uniformly, including
                        # links in the air--water--ice contact band.
                        opposite = _opp(q)
                        dynamic_pressure = (
                            self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                        )
                        wall_equilibrium = _pressure_eq(
                            opposite,
                            dynamic_pressure,
                            density,
                            boundary_velocity,
                        )
                        reflected_f = (
                            wall_equilibrium
                            + self.f_pre_collision_neq[i, j, q]
                            + eta * self.f_post[i, j, opposite]
                        ) / (1.0 + eta)
                        self.f[i, j, opposite] = reflected_f
                        self.h[i, j, opposite] = outgoing_h - (
                            6.0
                            * _w(q)
                            * self.phi[i, j]
                            * direction_f.dot(boundary_velocity)
                        )

                        opposite_direction = -direction_f
                        impulse = (direction_f - boundary_velocity) * outgoing_f - (
                            opposite_direction - boundary_velocity
                        ) * reflected_f
                        hydrostatic_pressure = (
                            1.0 - eta
                        ) * self.hydrostatic_reference_pressure[
                            i, j
                        ] + eta * self.hydrostatic_reference_pressure[ni, nj]
                        impulse += 6.0 * _w(q) * hydrostatic_pressure * direction_f
                        relative = boundary_point - self.body_center[None]
                        impulse64 = ti.cast(impulse, ti.f64)
                        relative64 = ti.cast(relative, ti.f64)
                        cell_impulse += impulse64
                        cell_torque += _cross2(relative64, impulse64)
                        has_cut_link = 1
                    else:
                        opposite = _opp(q)
                        self.f[i, j, opposite] = outgoing_f
                        self.h[i, j, opposite] = outgoing_h
                if has_cut_link == 1:
                    ti.atomic_add(self.hydrodynamic_impulse[None].x, cell_impulse.x)
                    ti.atomic_add(self.hydrodynamic_impulse[None].y, cell_impulse.y)
                    ti.atomic_add(self.hydrodynamic_torque[None], cell_torque)

    @ti.kernel
    def _stream_phase_only(self):
        """Stream h during phase warm-up without executing fluid coupling."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.phi:
            if self._active(i, j):
                for q in range(Q):
                    direction = _c(q)
                    ni = i + direction.x
                    nj = j + direction.y
                    outgoing = self.h_post[i, j, q]
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        self.h[ni, nj, q] = outgoing
                    elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                        sdf_fluid = ti.max(self.sdf[i, j], 1.0e-6)
                        eta = ti.min(
                            0.95,
                            ti.max(
                                0.05,
                                sdf_fluid / (sdf_fluid - self.sdf[ni, nj] + 1.0e-12),
                            ),
                        )
                        direction_f = ti.cast(direction, ti.f32)
                        boundary_velocity = self._body_velocity_at(
                            self._cell_point(i, j) + eta * direction_f
                        )
                        self.h[i, j, _opp(q)] = outgoing - (
                            6.0
                            * _w(q)
                            * self.phi[i, j]
                            * direction_f.dot(boundary_velocity)
                        )
                    else:
                        self.h[i, j, _opp(q)] = outgoing

    @ti.kernel
    def _update_streamed_macroscopic_fields(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.phi:
            if self._active(i, j):
                phi = 0.0
                momentum = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    phi += self.h[i, j, q]
                    momentum += ti.cast(_c(q), ti.f32) * self.f[i, j, q]
                self.phi[i, j] = phi
                density = _rho_mix(phi, rho_water, rho_air)
                self.u[i, j] = momentum / ti.max(density, 1.0e-12)
            else:
                velocity = ti.Vector([0.0, 0.0])
                if self.solid[i, j] == 1:
                    velocity = self._body_velocity_at(self._cell_point(i, j))
                if self.wall[i, j] == 1:
                    self.phi[i, j] = 0.0
                self.u[i, j] = velocity
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _update_streamed_phase(self):
        for i, j in self.phi:
            if self._active(i, j):
                phi = 0.0
                for q in range(Q):
                    phi += self.h[i, j, q]
                self.phi[i, j] = phi
            else:
                if self.wall[i, j] == 1:
                    self.phi[i, j] = 0.0

    def _warm_start_phase(self, steps):
        for _ in range(int(steps)):
            self._collide_phase()
            self._stream_phase_only()
            self._update_streamed_phase()

    @ti.kernel
    def _update_fluid_macro(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.phi:
            if self._active(i, j):
                phi = 0.0
                momentum = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    phi += self.h[i, j, q]
                    momentum += ti.cast(_c(q), ti.f32) * self.f[i, j, q]
                self.phi[i, j] = phi
                material_density = _rho_mix(phi, rho_water, rho_air)
                self.u[i, j] = momentum / ti.max(material_density, 1.0e-12)
            elif self.solid[i, j] == 1:
                self.u[i, j] = self._body_velocity_at(self._cell_point(i, j))
            else:
                self.phi[i, j] = 0.0
                self.u[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _update_pressure(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.p:
            if self._active(i, j):
                moving_distribution_sum = 0.0
                for q in range(Q):
                    if q != 0:
                        moving_distribution_sum += self.f[i, j, q]
                phi = self.phi[i, j]
                material_density = _rho_mix(phi, rho_water, rho_air)
                velocity = self.u[i, j] + 0.5 * self.fluid_force[i, j]
                grad_phi = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    cc_i = _c(q)
                    cc = ti.cast(cc_i, ti.f32)
                    neighbor = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 1)
                    grad_phi += 3.0 * _w(q) * cc * (neighbor - phi)
                grad_rho = (rho_water - rho_air) * grad_phi
                s0 = -1.5 * _w(0) * velocity.dot(velocity)
                dynamic_pressure = 0.6 * (
                    moving_distribution_sum
                    + 0.5 * velocity.dot(grad_rho)
                    + material_density * s0
                )
                reference_pressure = self.hydrostatic_reference_pressure[i, j]
                self.p[i, j] = reference_pressure + dynamic_pressure
            else:
                if self.wall[i, j] == 1:
                    self.p[i, j] = 0.0

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
    def _prepare_moving_body_contact_support_kernel(self, reduce_support: ti.template()):
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
        self._body_contact_geometry_active[None] = ti.cast(0, ti.i8)

    @ti.func
    def _reduce_moving_body_contact_extrema(self, i, j):
        """Find supports of the unclipped sharp LBM raster at this pose."""

        threshold = ti.static(float(self.cfg.thermal.solid_liquid_threshold))
        origin = self.body_reference_origin[None]
        fraction = self._moving_body_fraction[i, j]
        # The fraction pass has already produced the same raw SDF used
        # by ``_update_solid_mask``.  Reusing its sign
        # and the thresholded coverage keeps contact consistent with the
        # sharp mask, including a legitimate rectangle-edge node whose
        # SDF is zero, while excluding an outside bilinear contributor or
        # an empty rectangle boundary at lower thresholds.
        if fraction >= threshold and self.sdf[i, j] <= 0.0:
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
        self._body_contact_geometry_active[None] = ti.cast(active, ti.i8)
        if not active:
            for index in ti.static(range(4)):
                self._body_contact_support_extrema[index] = 0.0

    def _integrate_rigid_ice(self):
        """Advance the rigid state by one LBM step.

        World rasterization owns the subsequent contact transaction.  Keeping
        this method limited to the dynamic update ensures that its fraction
        pass can be shared with contact support reduction.
        """

        if int(self.body_active[None]) == 0:
            # Preserve the last finite pose of a completely melted body.
            # Cut-link loads are empty once its geometry vanishes, but clear
            # their accumulators explicitly so an inactive body cannot retain
            # or integrate a stale impulse.
            self.body_velocity[None] = (0.0, 0.0)
            self.body_angular_velocity[None] = 0.0
            self.hydrodynamic_impulse[None] = (0.0, 0.0)
            self.hydrodynamic_torque[None] = 0.0
            return
        self._integrate_rigid_ice_kernel()

    @ti.kernel
    def _integrate_rigid_ice_kernel(self):
        mass = ti.cast(self.body_mass_lattice[None], ti.f32)
        inertia = ti.cast(self.body_inertia_lattice[None], ti.f32)
        # The wrapper admits only a mechanically resolved body, so these are
        # the same stored mass and inertia audited by the thermal transition.
        gravity = ti.Vector(
            [ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])]
        )
        total_impulse = (
            ti.cast(self.hydrodynamic_impulse[None], ti.f32)
            + mass * gravity
        )
        total_torque = ti.cast(self.hydrodynamic_torque[None], ti.f32)

        velocity = ti.static(self.cfg.linear_damping) * (
            self.body_velocity[None] + total_impulse / mass
        )
        omega = ti.static(self.cfg.angular_damping) * (
            self.body_angular_velocity[None] + total_torque / inertia
        )
        local_com = ti.cast(self.body_local_center_of_mass[None], ti.f32)
        angle = self.body_angle[None] + omega
        center = self.body_center[None] + velocity

        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        self.body_center[None] = center
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
        # This kernel is the sole consumer of the accumulated cut-link load.
        # Consume-and-clear ownership leaves the following stream kernel with
        # one job: add the next step's locally reduced boundary loads.
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

        if self._body_contact_geometry_active[None] != 0:
            minimum_relative_x = self._body_contact_support_extrema[0] - rotated_com.x
            maximum_relative_x = self._body_contact_support_extrema[1] - rotated_com.x
            minimum_relative_y = self._body_contact_support_extrema[2] - rotated_com.y
            maximum_relative_y = self._body_contact_support_extrema[3] - rotated_com.y
            lower_boundary = ti.static(float(self.cfg.boundary_cells))
            upper_x_boundary = ti.static(float(self.nx - self.cfg.boundary_cells))
            upper_y_boundary = ti.static(float(self.ny - self.cfg.boundary_cells))
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

        self.body_center[None] = center
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
        initialize_previous: bool = False,
        *,
        resolve_contact: bool = True,
    ):
        """Rasterize the moving material and resolve whole-body wall contact.

        A single unclipped fraction pass feeds both the contact support
        reduction and the shared world SDF/thermal coverage fields.  If
        contact projection does not move the body, that pass is committed
        directly.  If projection clamps a penetrated pose, the fraction is
        evaluated once more at the corrected pose before committing it.
        Normal calls advance the ALE old/new pair, initialization seeds both
        values identically, and a mechanically inactive remnant is rasterized
        but is never projected.

        The body reference origin, angle, and wall are always read from this
        simulator instance.  ``initialize_previous`` is used only when the
        initial ALE old/new pair is seeded; ``resolve_contact=False`` is used
        by fixed-pose thermal refreshes that must not run the mechanical
        projection.  The return value reports whether contact projection
        changed the accepted pose.
        """

        self._calculate_moving_body_fraction()
        projected = False

        if resolve_contact and int(self.body_active[None]) != 0:
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

        if initialize_previous:
            previous_mode = 2
        else:
            previous_mode = 1
        self._commit_moving_body_raster(previous_mode)
        return projected

    # ------------------------------------------------------------------
    # Bounded, topology-constrained water-volume projection

    @ti.kernel
    def _initialize_water_volume(self):
        self.water_volume_target[None] = 0.0
        self.water_volume_current[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                phase = self.phi[i, j]
                ti.atomic_add(self.water_volume_target[None], ti.cast(phase, ti.f64))
        self.water_volume_current[None] = self.water_volume_target[None]

    @ti.kernel
    def _reduce_phase_change_solid_volume(self):
        self.phase_change_current_solid_volume[None] = 0.0
        density_ratio = ti.static(float(self.cfg.rho_ice / self.cfg.rho_water))
        self.phase_change_current_solid_volume[None] = (
            self.body_mass_lattice[None] / density_ratio
        )

    @ti.kernel
    def _reduce_phase_change_geometry_volume(self):
        self.phase_change_current_geometry_volume[None] = 0.0
        for i, j in self.solid:
            if self.solid[i, j] != 0:
                ti.atomic_add(self.phase_change_current_geometry_volume[None], 1.0)

    def _initialize_phase_change_reference(self):
        self._reduce_phase_change_solid_volume()
        self.phase_change_initial_water_volume[None] = float(
            self.water_volume_target[None]
        )
        initial_solid = float(self.phase_change_current_solid_volume[None])
        self.phase_change_initial_solid_volume[None] = initial_solid

    @ti.kernel
    def _apply_phase_change_water_target(self):
        density_ratio = ti.static(float(self.cfg.rho_ice / self.cfg.rho_water))
        solid_volume_change = (
            self.phase_change_current_solid_volume[None]
            - self.phase_change_initial_solid_volume[None]
        )
        # Melt is detached and injected into world water immediately.  Sharp
        # mask-count changes from rigid translation/rotation do not affect the
        # density-converted target.
        self.water_volume_target[None] = (
            self.phase_change_initial_water_volume[None]
            - density_ratio * solid_volume_change
        )

    def _update_phase_change_water_target(self):
        """Apply exact ice-mass/water-volume conversion for the thermal state."""

        self._reduce_phase_change_solid_volume()
        self._apply_phase_change_water_target()

    def phase_change_solid_volume_cells(self):
        self._reduce_phase_change_solid_volume()
        return float(self.phase_change_current_solid_volume[None])

    def phase_change_geometry_volume_cells(self):
        self._reduce_phase_change_geometry_volume()
        return float(self.phase_change_current_geometry_volume[None])

    @ti.kernel
    def _clip_water_phase_for_projection(self):
        """Restore phase bounds and reduce the resulting active water volume."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        self.water_volume_current[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                old_phase = self.phi[i, j]
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
                    velocity = self.u[i, j]
                    dynamic_pressure = (
                        self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = _rho_mix(old_phase, rho_water, rho_air)
                    new_density = _rho_mix(bounded_phase, rho_water, rho_air)
                    for q in range(Q):
                        self.f[i, j, q] += _pressure_eq(
                            q, dynamic_pressure, new_density, velocity
                        ) - _pressure_eq(q, dynamic_pressure, old_density, velocity)
                velocity = self.u[i, j]
                phase_population_l1 = 0.0
                for q in range(Q):
                    phase_population_l1 += ti.abs(self.h[i, j, q])
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
                    phase_velocity = self.u[i, j] + 0.5 * self.fluid_force[i, j]
                    reconstructed_sum = 0.0
                    for q in range(Q):
                        reconstructed = _heq(q, bounded_phase, phase_velocity)
                        self.h[i, j, q] = reconstructed
                        reconstructed_sum += reconstructed
                    self.h[i, j, 0] += bounded_phase - reconstructed_sum
                elif bounded_phase != old_phase:
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = (
                            self.h[i, j, q]
                            + _heq(q, bounded_phase, velocity)
                            - _heq(q, old_phase, velocity)
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    self.h[i, j, 0] += bounded_phase - lifted_sum
                self.phi[i, j] = bounded_phase
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(bounded_phase, ti.f64)
                )

    @ti.kernel
    def _seed_water_projection_interface(self):
        """Seed candidates connected to a local phi=1/2 crossing."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        for i, j in self.phi:
            seed = 0
            if self._active(i, j):
                phase = self.phi[i, j]
                if cutoff < phase < 1.0 - cutoff:
                    if phase == 0.5:
                        seed = 1
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ni = i + di
                        nj = j + dj
                        if (
                            (di != 0 or dj != 0)
                            and _inside(ni, nj, nx, ny)
                            and self._active(ni, nj)
                        ):
                            neighbor = self.phi[ni, nj]
                            if (phase < 0.5 <= neighbor) or (neighbor < 0.5 <= phase):
                                seed = 1
            self.h_post[i, j, 0] = ti.cast(seed, ti.f32)
            self.h_post[i, j, 1] = 0.0

    @ti.kernel
    def _dilate_water_projection_interface(self, primary_to_next: ti.i32):
        """Dilate between the two scratch planes through candidate cells."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        for i, j in self.phi:
            value = 0
            phase = self.phi[i, j]
            if self._active(i, j) and cutoff < phase < 1.0 - cutoff:
                if primary_to_next == 1:
                    value = ti.cast(self.h_post[i, j, 0], ti.i32)
                else:
                    value = ti.cast(self.h_post[i, j, 1], ti.i32)
                if value == 0:
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ni = i + di
                        nj = j + dj
                        if (di != 0 or dj != 0) and _inside(ni, nj, nx, ny):
                            neighbor = 0.0
                            if primary_to_next == 1:
                                neighbor = self.h_post[ni, nj, 0]
                            else:
                                neighbor = self.h_post[ni, nj, 1]
                            if neighbor == 1.0:
                                value = 1
            if primary_to_next == 1:
                self.h_post[i, j, 1] = ti.cast(value, ti.f32)
            else:
                self.h_post[i, j, 0] = ti.cast(value, ti.f32)

    @ti.kernel
    def _evaluate_water_projection(self, lagrange_multiplier: ti.f64):
        """Evaluate volume and derivative using f64 reductions.

        The mapped value is rounded to the f32 phase storage type before the
        reduction.  Consequently this is the same discrete volume that the
        application kernel and the conservation reduction will observe.
        """

        self.water_volume_current[None] = 0.0
        self.water_projection_derivative[None] = 0.0
        exponential = ti.exp(lagrange_multiplier)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        for i, j in self.phi:
            if self._active(i, j):
                phase64 = ti.cast(self.phi[i, j], ti.f64)
                mapped64 = phase64
                adjustable = self.h_post[i, j, 0] == 1.0
                if adjustable:
                    mapped64 = (
                        phase64 * exponential / (1.0 - phase64 + phase64 * exponential)
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
                ti.atomic_add(self.water_volume_current[None], mapped64)
                if adjustable and cutoff < mapped32 < 1.0 - cutoff:
                    ti.atomic_add(
                        self.water_projection_derivative[None],
                        mapped64 * (1.0 - mapped64),
                    )

    @ti.kernel
    def _apply_water_projection(self, lagrange_multiplier: ti.f64):
        """Apply the entropic map, lift h, and reduce the final volume."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        exponential = ti.exp(lagrange_multiplier)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        self.water_volume_current[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                old_phase = self.phi[i, j]
                new_phase = old_phase
                if self.h_post[i, j, 0] == 1.0:
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
                    velocity = self.u[i, j]
                    dynamic_pressure = (
                        self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = _rho_mix(old_phase, rho_water, rho_air)
                    new_density = _rho_mix(new_phase, rho_water, rho_air)
                    for q in range(Q):
                        self.f[i, j, q] += _pressure_eq(
                            q, dynamic_pressure, new_density, velocity
                        ) - _pressure_eq(q, dynamic_pressure, old_density, velocity)
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = (
                            self.h[i, j, q]
                            + _heq(q, new_phase, velocity)
                            - _heq(q, old_phase, velocity)
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    self.h[i, j, 0] += new_phase - lifted_sum
                    self.phi[i, j] = new_phase
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(new_phase, ti.f64)
                )

    @ti.kernel
    def _measure_water_projection_residual_weight(self, residual: ti.f64):
        """Measure the free-set metric for a storage-level mass closure."""

        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        # Four global f32 ulps are conservative at both phase endpoints and
        # keep the subsequent cast strictly outside the canonicalized tails.
        storage_margin = ti.static(4.0 * float(np.finfo(np.float32).eps))
        magnitude = ti.abs(residual)
        self.water_projection_derivative[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j) and self.h_post[i, j, 0] == 1.0:
                phase = ti.cast(self.phi[i, j], ti.f64)
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

    @ti.kernel
    def _apply_water_projection_residual(self, residual: ti.f64, total_weight: ti.f64):
        """Close a scalar f32 residual on the safe entropic free set."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        storage_margin = ti.static(4.0 * float(np.finfo(np.float32).eps))
        magnitude = ti.abs(residual)
        self.water_volume_current[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                old_phase = self.phi[i, j]
                old_phase64 = ti.cast(old_phase, ti.f64)
                new_phase = old_phase
                free = self.h_post[i, j, 0] == 1.0 and (
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
                    velocity = self.u[i, j]
                    dynamic_pressure = (
                        self.p[i, j] - self.hydrostatic_reference_pressure[i, j]
                    )
                    old_density = _rho_mix(old_phase, rho_water, rho_air)
                    new_density = _rho_mix(new_phase, rho_water, rho_air)
                    for q in range(Q):
                        self.f[i, j, q] += _pressure_eq(
                            q, dynamic_pressure, new_density, velocity
                        ) - _pressure_eq(q, dynamic_pressure, old_density, velocity)
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = (
                            self.h[i, j, q]
                            + _heq(q, new_phase, velocity)
                            - _heq(q, old_phase, velocity)
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    self.h[i, j, 0] += new_phase - lifted_sum
                    self.phi[i, j] = new_phase
                ti.atomic_add(
                    self.water_volume_current[None], ti.cast(new_phase, ti.f64)
                )

    def _correct_water_volume(self):
        """Project the active phase field onto the exact no-melting volume.

        A scalar Bernoulli-relative-entropy projection translates the existing
        logistic interface.  Its support is restricted to the diffuse band
        connected to phi=1/2, so disconnected minority-phase noise and exact
        bulk values cannot receive an additive volume source.
        """

        self._clip_water_phase_for_projection()
        target = float(self.water_volume_target[None])
        tolerance = float(self.cfg.volume_projection_tolerance) * max(1.0, abs(target))
        initial_error = target - float(self.water_volume_current[None])
        if abs(initial_error) <= tolerance:
            return

        self._seed_water_projection_interface()
        for pass_index in range(self._volume_projection_band_radius):
            self._dilate_water_projection_interface(1 if pass_index % 2 == 0 else 0)

        lambda_limit = (
            4.0
            * float(self.cfg.volume_projection_max_shift)
            / float(self.cfg.interface_width)
        )
        self._evaluate_water_projection(-lambda_limit)
        lower_mass = float(self.water_volume_current[None])
        self._evaluate_water_projection(lambda_limit)
        upper_mass = float(self.water_volume_current[None])
        if abs(upper_mass - lower_mass) <= tolerance:
            raise RuntimeError(
                "water-volume projection is infeasible: no adjustable "
                "phi=0.5-connected interface is available"
            )
        if target < lower_mass - tolerance or target > upper_mass + tolerance:
            raise RuntimeError(
                "water-volume projection exceeds the permitted interface shift: "
                f"target={target:.17g}, reachable=[{lower_mass:.17g}, {upper_mass:.17g}]"
            )

        lower_lambda = -lambda_limit
        upper_lambda = lambda_limit
        lagrange_multiplier = 0.0
        best_lambda = 0.0
        best_residual = abs(initial_error)
        for _ in range(int(self.cfg.volume_projection_max_iterations)):
            self._evaluate_water_projection(lagrange_multiplier)
            mass = float(self.water_volume_current[None])
            derivative = float(self.water_projection_derivative[None])
            residual = mass - target
            if abs(residual) < best_residual:
                best_residual = abs(residual)
                best_lambda = lagrange_multiplier
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

        self._apply_water_projection(best_lambda)
        # Endpoint canonicalization makes the scalar f32 map discontinuous.
        # If the exact target lies in one of its small gaps, close the
        # remaining volume on interface cells that are safely separated from
        # either endpoint.  The Bernoulli entropy Hessian is
        # 1/[phi(1-phi)], so this weighted correction is the minimum
        # quadratic-entropy perturbation on the current active set.
        for _ in range(8):
            residual = target - float(self.water_volume_current[None])
            if abs(residual) <= tolerance:
                break
            self._measure_water_projection_residual_weight(residual)
            total_weight = float(self.water_projection_derivative[None])
            if not math.isfinite(total_weight) or total_weight <= 0.0:
                break
            self._apply_water_projection_residual(residual, total_weight)
        final_error = abs(target - float(self.water_volume_current[None]))
        if final_error > tolerance:
            raise RuntimeError(
                "water-volume projection cannot close its endpoint active set: "
                f"residual={final_error:.6e}, tolerance={tolerance:.6e}"
            )
