"""CUDA/CPU Taichi D3Q27 flow, moving ice, and material-frame melting.

Pressure populations carry dynamic pressure, not an isothermal rho/3.
Material density is independently interpolated from the bounded water phase.
"""
# ruff: noqa: E741 -- I and J are Taichi grouped grid indices.

import math
import numpy as np
import taichi as ti

from .config import IceFlowConfig, LatticeScales, integer
from .lattice import (
    Q,
    REST,
    W0,
    density,
    direction,
    inverse_basis,
    mixture_kinematic_viscosity,
    momentum_equilibrium,
    momentum_rest_velocity,
    phase_equilibrium,
    transform,
    weight,
)
from .gradients import MirrorGradient3D
from .rigid import RigidBodies
from .thermal import Thermal3D
from .viscosity import air_dynamic_viscosity_pa_s


def initialize_taichi(backend):
    arch = ti.cuda if backend == "cuda" else ti.cpu
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(
            arch=arch,
            enable_fallback=False,
            default_fp=ti.f32,
            cpu_max_num_threads=4,
            offline_cache=True,
        )
    elif ti.lang.impl.current_cfg().arch != arch:
        raise RuntimeError("Taichi is already initialized with a different backend")


@ti.data_oriented
class IceFlow3D(MirrorGradient3D, Thermal3D):
    # Scratch bounds are reconstructed by rasterization, not restart state.
    _transient_field_names = frozenset(
        ("_gradient_geometry_lower", "_gradient_geometry_upper")
    )

    def __init__(self, config: IceFlowConfig):
        if not isinstance(config, IceFlowConfig):
            raise TypeError("config must be IceFlowConfig")
        if config.conserve_phase_volume:
            raise ValueError(
                "global phase-volume correction is commented out; "
                "set conserve_phase_volume=False"
            )
        initialize_taichi(config.backend)
        self.config, self.scales = config, LatticeScales.from_iceflow_config(config)
        self.shape = (config.nx, config.ny, config.nz)
        self.nx, self.ny, self.nz = self.shape
        self.steps = 0
        self.body_count = len(config.bodies)
        self.rho_air = config.rho_air / config.rho_water
        self.gravity = tuple(
            np.array(config.gravity) * self.scales.dt_s**2 / self.scales.dx_m
        )
        self.sigma = (
            config.surface_tension_n_m
            * self.scales.dt_s**2
            / (config.rho_water * self.scales.dx_m**3)
        )
        self.mu_air = air_dynamic_viscosity_pa_s(
            config.thermal.initial_air_temperature_c
        )
        self.nu_air = (
            self.mu_air / config.rho_air * self.scales.dt_s / self.scales.dx_m**2
        )
        self.f, self.f_post, self.h, self.h_post = [ti.field(ti.f32) for _ in range(4)]
        for population in (self.f, self.f_post, self.h, self.h_post):
            ti.root.dense(ti.l, Q).dense(ti.ijk, self.shape).place(population)
        for name in (
            "water_phase",
            "pressure_lattice",
            "old_pressure",
            "coverage",
            "density_source_scale",
        ):
            setattr(self, name, ti.field(ti.f32, shape=self.shape))
        for name in (
            "velocity_lattice",
            "old_velocity",
            "force_density",
            "phase_gradient_lattice",
            "phase_mass_flux",
        ):
            setattr(self, name, ti.Vector.field(3, ti.f32, shape=self.shape))
        self.capillary_stress = ti.Matrix.field(3, 3, ti.f32, shape=self.shape)
        self.moving_population_sum = ti.field(ti.f64, shape=self.shape)
        # IDs: -1 fluid, -2 container, -3 unresolved newly exposed cavity, >=0 ice.
        self.solid_body_id = ti.field(ti.i32, shape=self.shape)
        self.previous_body_id = ti.field(ti.i32, shape=self.shape)
        self.coverage_body_id = ti.field(ti.i32, shape=self.shape)
        self.refill_state = ti.Vector.field(5, ti.f32, shape=self.shape)
        self.refill_next = ti.Vector.field(5, ti.f32, shape=self.shape)
        self.refill_valid = ti.field(ti.i8, shape=self.shape)
        self.refill_next_valid = ti.field(ti.i8, shape=self.shape)
        self.openings = ti.field(ti.i32, shape=())
        self.remaining = ti.field(ti.i32, shape=())
        self.refill_lower = ti.Vector.field(3, ti.i32, shape=())
        self.refill_upper = ti.Vector.field(3, ti.i32, shape=())
        self._gradient_geometry_lower = ti.Vector.field(3, ti.i32, shape=())
        self._gradient_geometry_upper = ti.Vector.field(3, ti.i32, shape=())
        self.reference_density = ti.field(ti.f32, shape=self.ny)
        self.reference_pressure = ti.field(ti.f32, shape=self.ny)
        self.body_origin = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.body_rotation = ti.Matrix.field(3, 3, ti.f32, shape=self.body_count)
        self.body_center = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.body_velocity = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.body_angular_velocity = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.hydrodynamic_force = ti.Vector.field(3, ti.f64, shape=self.body_count)
        self.hydrodynamic_torque = ti.Vector.field(3, ti.f64, shape=self.body_count)
        self.reduced_mass = ti.field(ti.f64, shape=self.body_count)
        self.reduced_first = ti.Vector.field(3, ti.f64, shape=self.body_count)
        self.reduced_second = ti.Matrix.field(3, 3, ti.f64, shape=self.body_count)
        self.reduced_lower = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.reduced_upper = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.invalid_index = ti.field(ti.i32, shape=())
        self.maximum_speed = ti.field(ti.f32, shape=())
        self.peak_speed = ti.field(ti.f32, shape=())
        self.repair_count = ti.field(ti.i32, shape=())
        self.phase_clip_volume = ti.field(ti.f64, shape=())
        self.phase_balance_sum = ti.field(ti.f64, shape=())
        self.phase_balance_capacity = ti.field(ti.f64, shape=())
        self.phase_volume_projection_m3 = ti.field(ti.f64, shape=())
        self.phase_projection_momentum_n_s = ti.Vector.field(3, ti.f64, shape=())
        self.pending_cells = 0
        self._allocate_thermal()
        self.rigid = RigidBodies(self)
        self._initialize_material()
        self.rigid.update_mass()
        self._rasterize()
        self._initialize_phase()
        for _ in range(config.phase_warmup_steps):
            self._gradients()
            self._collide_phase()
            self._stream(True)
            self._recover_phase()
        self._initialize_water_heat()
        self._build_reference()
        self._gradients()
        self._forces(False)
        self._seed_populations()
        self.initial_ice_mass_kg = self.rigid.mass.copy() * self.scales.mass_kg
        self.initial_water_volume_m3 = float(self.water_volume.to_numpy().sum())
        self.initial_phase_volume_m3 = float(
            self.water_phase.to_numpy().sum(dtype=np.float64) * self.cell_volume
        )
        self._thermal_pending = 0
        # The final initialization gradient is also the next collision input.
        self._gradient_cache_valid = True

    def invalidate_gradient_cache(self):
        """Call after externally editing phase, coverage, or fluid/solid masks.

        Normal stepping maintains the cache, including the changed-geometry
        halo. Checkpoint loading invalidates it automatically.
        """
        self._gradient_cache_valid = False

    @property
    def physical_time_s(self):
        return self.steps * self.scales.dt_s

    @ti.func
    def _inside(self, I):
        return 0 <= I.x < self.nx and 0 <= I.y < self.ny and 0 <= I.z < self.nz

    @ti.func
    def _fluid(self, I):
        result = False
        if self._inside(I):
            result = self.solid_body_id[I] == -1
        return result

    @ti.func
    def _connected(self, I, c):
        """A face-connected path within this 2x2x2 stencil; never cross ice."""
        J = I + c
        valid = False
        if self._fluid(J):
            distance = ti.abs(c).sum()
            if distance <= 1:
                valid = True
            else:
                for a in ti.static(range(3)):
                    ca = ti.Vector.zero(ti.i32, 3)
                    ca[a] = c[a]
                    if c[a] != 0 and self._fluid(I + ca):
                        if distance == 2:
                            valid = True
                        else:
                            for b in ti.static(range(3)):
                                if ti.static(a != b):
                                    cb = ca
                                    cb[b] = c[b]
                                    if self._fluid(I + cb):
                                        valid = True
        return valid

    @ti.func
    def _material_grid(self, point, b):
        return (
            self.body_rotation[b].transpose() @ (point - self.body_origin[b])
            + 0.5 * ti.cast(self.material_size[b], ti.f32)
            - 0.5
        )

    @ti.func
    def _material_inside(self, J, b):
        return (
            0 <= J.x < self.material_size[b].x
            and 0 <= J.y < self.material_size[b].y
            and 0 <= J.z < self.material_size[b].z
        )

    @ti.func
    def _fraction(self, point, b):
        grid = self._material_grid(point, b)
        base = ti.cast(ti.floor(grid), ti.i32)
        part = grid - ti.cast(base, ti.f32)
        value = 0.0
        for a, c, d in ti.static(ti.ndrange(2, 2, 2)):
            J = base + ti.Vector([a, c, d])
            if self._material_inside(J, b):
                w = (
                    (part.x if a else 1 - part.x)
                    * (part.y if c else 1 - part.y)
                    * (part.z if d else 1 - part.z)
                )
                value += w * ti.cast(
                    self.material_mass[b, J.x, J.y, J.z] / self.initial_cell_mass,
                    ti.f32,
                )
        return value

    @ti.func
    def _body_velocity_at(self, point, b):
        return self.body_velocity[b] + self.body_angular_velocity[b].cross(
            point - self.body_center[b]
        )

    @ti.kernel
    def _rasterize(self):
        bnd = ti.static(self.config.boundary_cells)
        self._gradient_geometry_lower[None] = ti.Vector(self.shape)
        self._gradient_geometry_upper[None] = ti.Vector([-1, -1, -1])
        for I in ti.grouped(self.water_phase):
            self.previous_body_id[I] = self.solid_body_id[I]
            wall = (
                I.x < bnd
                or I.x >= self.nx - bnd
                or I.y < bnd
                or I.y >= self.ny - bnd
                or I.z < bnd
                or I.z >= self.nz - bnd
            )
            fraction, owner = 0.0, -1
            if not wall:
                for b in range(self.body_count):
                    f = self._fraction(ti.cast(I, ti.f32) + 0.5, b)
                    if f > fraction:
                        fraction, owner = f, b
            new_id = (
                -2
                if wall
                else (
                    -1
                    if fraction < self.config.thermal.solid_liquid_threshold
                    else owner
                )
            )
            if fraction != self.coverage[I] or new_id != self.solid_body_id[I]:
                ti.atomic_min(self._gradient_geometry_lower[None], I)
                ti.atomic_max(self._gradient_geometry_upper[None], I)
            self.coverage[I] = fraction
            self.coverage_body_id[I] = owner
            self.solid_body_id[I] = new_id

    @ti.kernel
    def _initialize_phase(self):
        for I in ti.grouped(self.water_phase):
            phi = 0.0
            if self._fluid(I):
                # Smooth union of the column and downstream pool.
                column = ti.min(
                    self.config.water_width - I.x - 0.5,
                    self.config.water_height - I.y - 0.5,
                )
                distance = column
                if ti.static(self.config.floor_water_depth_cells > 0):
                    distance = ti.max(
                        column,
                        self.config.boundary_cells
                        + self.config.floor_water_depth_cells
                        - I.y
                        - 0.5,
                    )
                phi = 0.5 + 0.5 * ti.tanh(2 * distance / self.config.interface_width)
            self.water_phase[I] = phi
            for q in range(Q):
                self.h[I, q] = weight(q) * phi

    def _build_reference(self):
        phase = self.water_phase.to_numpy()
        ids = self.solid_body_id.to_numpy()
        ref = (
            np.arange(self.ny)
            < self.config.boundary_cells + self.config.floor_water_depth_cells
        ).astype(float)
        if self.config.water_width == self.nx - self.config.boundary_cells:
            ref = (np.arange(self.ny) < self.config.water_height).astype(float)
            mask = ids == -1
        else:
            mask = ids == -1
            mask[: self.config.water_width] = False
        if (
            self.config.floor_water_depth_cells
            or self.config.water_width == self.nx - self.config.boundary_cells
        ):
            counts = mask.sum(axis=(0, 2))
            np.divide(
                np.where(mask, phase, 0).sum(axis=(0, 2), dtype=float),
                counts,
                out=ref,
                where=counts > 0,
            )
        rho = self.rho_air + (1 - self.rho_air) * ref
        p = -self.gravity[1] * (np.cumsum(rho[::-1])[::-1] - 0.5 * rho)
        excess = np.where(
            ids == -1, (phase - ref[None, :, None]) * (1 - self.rho_air), 0.0
        )
        dynamic = -self.gravity[1] * (
            np.cumsum(excess[:, ::-1, :], axis=1)[:, ::-1, :] - 0.5 * excess
        )
        self.reference_density.from_numpy(rho.astype(np.float32))
        self.reference_pressure.from_numpy(p.astype(np.float32))
        self.pressure_lattice.from_numpy(dynamic.astype(np.float32))

    @ti.kernel
    def _seed_populations(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                # Keep the relaxed phase populations from phase warmup.
                self._set_momentum_equilibrium(
                    I,
                    self.water_phase[I],
                    self.pressure_lattice[I],
                    ti.Vector.zero(ti.f32, 3),
                )

    @ti.func
    def _set_momentum_equilibrium(self, I, phi, p, u):
        rho = density(phi, self.rho_air)
        force = self.force_density[I]
        grad = self.phase_gradient_lattice[I]
        for q in range(Q):
            c = ti.cast(direction(q), ti.f32)
            correction = (
                -1.5
                * weight(q)
                * (c.dot(force) + (1 - self.rho_air) * c.dot(u) * c.dot(grad))
            )
            self.f[I, q] = momentum_equilibrium(q, p, rho, u) + correction
        self.velocity_lattice[I] = u

    @ti.func
    def _set_equilibrium(self, I, phi, p, u):
        self._set_momentum_equilibrium(I, phi, p, u)
        for q in range(Q):
            self.h[I, q] = phase_equilibrium(q, phi, u)

    @ti.kernel
    def _collide_phase(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                phi, u = self.water_phase[I], self.velocity_lattice[I]
                g = self.phase_gradient_lattice[I]
                normal = g / ti.sqrt(
                    g.dot(g) + self.config.phase_normal_epsilon**2 + 1e-28
                )
                base = ti.Vector.zero(ti.f32, Q)
                residual_first = ti.Vector.zero(ti.f32, 3)
                for q in ti.static(range(Q)):
                    c = ti.cast(direction(q), ti.f32)
                    eq = phase_equilibrium(q, phi, u)
                    src = (
                        weight(q)
                        * 4
                        * phi
                        * (1 - phi)
                        / self.config.interface_width
                        * c.dot(normal)
                    )
                    base[q] = eq + 0.5 * src
                    residual_first += (c - u) * (self.h[I, q] - eq + 0.5 * src)
                # By linearity, T^-1(T(base) + delta) = base + T^-1(delta).
                # Avoid a forward transform of an already known population.
                moments = ti.Vector.zero(ti.f32, Q)
                rate = ti.static(1 / (0.5 + 3 * self.config.mobility))
                moments[9] += (1 - rate) * residual_first.x
                moments[3] += (1 - rate) * residual_first.y
                moments[1] += (1 - rate) * residual_first.z
                post = base + transform(moments, u, True)
                outgoing = ti.cast(0.0, ti.f64)
                flux = ti.Vector.zero(ti.f32, 3)
                for q in ti.static(range(Q)):
                    if ti.static(q != REST):
                        outgoing += ti.cast(post[q], ti.f64)
                    self.h_post[I, q] = post[q]
                    flux += (
                        0.5 * (self.h[I, q] + post[q]) * ti.cast(direction(q), ti.f32)
                    )
                self.h_post[I, REST] = ti.cast(ti.cast(phi, ti.f64) - outgoing, ti.f32)
                self.phase_mass_flux[I] = (1 - self.rho_air) * (flux - phi * u)

    @ti.func
    def _capillary_neighbor(self, I, c):
        J = I + c
        stress = self.capillary_stress[I]
        if self._fluid(J):
            stress = self.capillary_stress[J]
        elif self._inside(J) and self.solid_body_id[J] == -2:
            # Even reflection across a planar wall; flip mixed tensor entries.
            R = ti.Matrix.identity(ti.f32, 3)
            b = ti.static(self.config.boundary_cells)
            for a in ti.static(range(3)):
                if J[a] < b:
                    J[a] = 2 * b - 1 - J[a]
                    R[a, a] = -1.0
                elif J[a] >= ti.static(self.shape[a]) - b:
                    J[a] = 2 * (ti.static(self.shape[a]) - b) - 1 - J[a]
                    R[a, a] = -1.0
            if self._fluid(J):
                stress = R @ self.capillary_stress[J] @ R
        return stress

    @ti.func
    def _force_at(self, I, with_flux: ti.template()):
        force = ti.Vector.zero(ti.f32, 3)
        if self._fluid(I):
            phi = self.water_phase[I]
            rho = density(phi, self.rho_air)
            gravity = ti.Vector(self.gravity)
            force = (rho - self.reference_density[I.y]) * gravity
            force -= (
                ti.max(0.0, 2 * phi - 1)
                * self.config.thermal.thermal_expansion_water_1_k
                * (
                    self._water_temperature(I)
                    - self.config.thermal.initial_water_temperature_c
                )
                * gravity
            )
            if ti.static(self.sigma > 0):
                for q in range(Q):
                    if q != REST:
                        c = direction(q)
                        force += (
                            3
                            * weight(q)
                            * (
                                self.capillary_stress[I]
                                + self._capillary_neighbor(I, c)
                            )
                            @ ti.cast(c, ti.f32)
                        )
            if ti.static(with_flux):
                for a, sign in ti.static(ti.ndrange(3, 2)):
                    c = ti.Vector.zero(ti.i32, 3)
                    c[a] = 2 * sign - 1
                    J = I + c
                    if self._fluid(J):
                        # Conservative donor transport for the phase mass flux.
                        # The added face-jump term dissipates kinetic energy and
                        # vanishes for uniform velocity, without changing the
                        # physical shear viscosity or pure-phase transport.
                        mass_flux = 0.5 * (
                            self.phase_mass_flux[I][a] + self.phase_mass_flux[J][a]
                        )
                        donor = self.velocity_lattice[J]
                        if (2 * sign - 1) * mass_flux >= 0:
                            donor = self.velocity_lattice[I]
                        flux = mass_flux * donor
                        if ti.static(self.config.phase_flux_acceleration_limit > 0):
                            budget = (
                                self.config.phase_flux_acceleration_limit
                                / 6
                                * ti.min(
                                    rho, density(self.water_phase[J], self.rho_air)
                                )
                            )
                            flux *= ti.min(1.0, budget / (ti.abs(flux).sum() + 1e-20))
                        force -= (2 * sign - 1) * flux
        return force

    @ti.kernel
    def _forces(self, with_flux: ti.template()):
        for I in ti.grouped(self.water_phase):
            self.force_density[I] = self._force_at(I, with_flux)

    @ti.kernel
    def _collide_momentum(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                phi = self.water_phase[I]
                rho = density(phi, self.rho_air)
                u = self.velocity_lattice[I]
                self.old_velocity[I], self.old_pressure[I] = u, self.pressure_lattice[I]
                g = self.phase_gradient_lattice[I]
                # Compatibility diagnostic: no separate density limiter.
                self.density_source_scale[I] = 1.0
                nu = (
                    mixture_kinematic_viscosity(
                        phi,
                        self._water_temperature(I),
                        self.config.rho_water,
                        self.config.rho_air,
                        self.mu_air,
                    )
                    * self.scales.dt_s
                    / self.scales.dx_m**2
                )
                rate = 1 / (0.5 + 3 * nu)
                base, residual = ti.Vector.zero(ti.f32, Q), ti.Vector.zero(ti.f32, Q)
                old_moving, eq_moving, source_moving = 0.0, 0.0, 0.0
                for q in ti.static(range(Q)):
                    c = ti.cast(direction(q), ti.f32)
                    eq = momentum_equilibrium(q, self.pressure_lattice[I], rho, u)
                    source = (
                        3
                        * weight(q)
                        * (
                            c.dot(self.force_density[I])
                            + (1 - self.rho_air) * c.dot(u) * c.dot(g)
                        )
                    )
                    base[q] = eq + 0.5 * source
                    residual[q] = self.f[I, q] - eq + 0.5 * source
                    if ti.static(q != REST):
                        old_moving += self.f[I, q]
                        eq_moving += eq
                        source_moving += source
                moments = ti.Vector.zero(ti.f32, Q)
                non_eq = transform(residual, u, False)
                bulk = ti.static(self.config.cm_acoustic_relaxation_rate)
                trace = (non_eq[18] + non_eq[6] + non_eq[2]) / 3
                reconstructed = ti.Matrix.zero(ti.f32, 3, 3)
                if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                    reconstructed = self._regularized_stress(I, rho, 1 / rate)
                    reconstructed -= (
                        reconstructed.trace() / 3 * ti.Matrix.identity(ti.f32, 3)
                    )
                    kinetic = ti.Matrix(
                        [
                            [non_eq[18] - trace, non_eq[12], non_eq[10]],
                            [non_eq[12], non_eq[6] - trace, non_eq[4]],
                            [non_eq[10], non_eq[4], non_eq[2] - trace],
                        ]
                    )
                    blend = ti.static(self.config.hrr_kinetic_blend)
                    reconstructed = blend * kinetic + (1 - blend) * reconstructed
                if ti.static(self.config.momentum_collision_model == "cm_mrt"):
                    # Kinetic deviatoric stress: no velocity-gradient estimate
                    # or HRR blending enters CM-MRT. It also defines the
                    # second-order content of the central fourth moments.
                    reconstructed = ti.Matrix(
                        [
                            [non_eq[18] - trace, non_eq[12], non_eq[10]],
                            [non_eq[12], non_eq[6] - trace, non_eq[4]],
                            [non_eq[10], non_eq[4], non_eq[2] - trace],
                        ]
                    )
                for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                    k = ti.static(a * 9 + b * 3 + c)
                    if ti.static((a, b, c) in ((2, 0, 0), (0, 2, 0), (0, 0, 2))):
                        dev = non_eq[k] - trace
                        if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                            axis = ti.static(0 if a == 2 else (1 if b == 2 else 2))
                            dev = reconstructed[axis, axis]
                        moments[k] += (1 - rate) * dev + (1 - bulk) * trace
                    elif ti.static(a + b + c == 2):
                        dev = non_eq[k]
                        if ti.static(self.config.momentum_collision_model == "cm_hrr"):
                            axis0 = ti.static(0 if a else 1)
                            axis1 = ti.static(2 if c else 1)
                            dev = reconstructed[axis0, axis1]
                        moments[k] += (1 - rate) * dev
                    elif ti.static(a + b + c == 4):
                        # Relax independent fourth-order modes after removing
                        # their second-order shear content (central Hermite
                        # basis). Retain that content at the molecular shear
                        # rate. Setting monomial fourth moments to equilibrium
                        # instead couples ghost modes back into kinetic shear.
                        # In 3D even 220/202/022 retain deviatoric content too.
                        high = 0.0
                        if ti.static(a == 0):
                            high = (reconstructed[1, 1] + reconstructed[2, 2]) / 3
                        elif ti.static(b == 0):
                            high = (reconstructed[0, 0] + reconstructed[2, 2]) / 3
                        elif ti.static(c == 0):
                            high = (reconstructed[0, 0] + reconstructed[1, 1]) / 3
                        elif ti.static(a == 2):
                            high = reconstructed[1, 2] / 3
                        elif ti.static(b == 2):
                            high = reconstructed[0, 2] / 3
                        else:
                            high = reconstructed[0, 1] / 3
                        moments[k] += (1 - rate) * high
                        if ti.static(a % 2 == 0 and b % 2 == 0 and c % 2 == 0):
                            moments[k] += (1 - bulk) * (non_eq[k] - high)
                    elif ti.static(
                        a % 2 == 0 and b % 2 == 0 and c % 2 == 0 and a + b + c >= 4
                    ):
                        # All even pressure modes share the acoustic rate.
                        moments[k] += (1 - bulk) * non_eq[k]
                if ti.static(
                    self.config.momentum_collision_model == "cm_hrr"
                    and self.config.hrr_cubic_correction
                ):
                    # D3Q27 aliases c_a^3=c_a and omits rho*u_a^3 from
                    # the diagonal third equilibrium moment. HRR supplies the
                    # physical shear explicitly; cancel the remaining
                    # half-step streaming defect in the post-collision stress.
                    cubic = self._hrr_cubic_correction(I)
                    moments[18] += cubic.x
                    moments[6] += cubic.y
                    moments[2] += cubic.z
                    moments[24] += (cubic.x + cubic.y) / 3
                    moments[20] += (cubic.x + cubic.z) / 3
                    moments[8] += (cubic.y + cubic.z) / 3
                    moments[26] += cubic.sum() / 9
                post = base + transform(moments, u, True)
                if ti.static(True):
                    # Close the raw pressure mode independently of the shear
                    # relaxation in both collision models. A pure 222 lift changes sum(q != rest) by +delta
                    # and preserves all lower moments, including momentum.
                    actual = 0.0
                    for q in ti.static(range(Q)):
                        if ti.static(q != REST):
                            actual += post[q]
                    target = (
                        old_moving
                        - bulk * (old_moving - eq_moving)
                        + (1 - 0.5 * bulk) * source_moving
                    )
                    for q in ti.static(range(Q)):
                        c = direction(q)
                        post[q] += (
                            (target - actual)
                            * inverse_basis(c.x, 0.0, 2)
                            * inverse_basis(c.y, 0.0, 2)
                            * inverse_basis(c.z, 0.0, 2)
                        )
                # Enforce the local forced first moment to roundoff accuracy.
                # rho*u and the equilibrium are rounded independently in f32;
                # repeatedly replacing the incoming momentum by that rounded
                # equilibrium biases low-viscosity shear, especially at mixed
                # densities. An antisymmetric axis-pair lift changes only the
                # first raw moment among orders <=2 and leaves pressure intact.
                defect = ti.cast(self.force_density[I], ti.f64)
                for q in ti.static(range(Q)):
                    defect += ti.cast(direction(q), ti.f64) * (
                        ti.cast(self.f[I, q], ti.f64) - ti.cast(post[q], ti.f64)
                    )
                for q in ti.static(range(Q)):
                    c = direction(q)
                    correction = ti.cast(0.0, ti.f64)
                    if ti.abs(c).sum() == 1:
                        correction = 0.5 * ti.cast(c, ti.f64).dot(defect)
                    self.f_post[I, q] = ti.cast(
                        ti.cast(post[q], ti.f64) + correction, ti.f32
                    )

    @ti.func
    def _stress_neighbor_velocity(self, I, c):
        center = self.velocity_lattice[I]
        J = I + c
        neighbor = -center
        if self._fluid(J):
            neighbor = self.velocity_lattice[J]
        elif self._inside(J) and self.solid_body_id[J] >= 0:
            eta = ti.min(
                0.95,
                ti.max(
                    0.05,
                    (self.config.thermal.solid_liquid_threshold - self.coverage[I])
                    / ti.max(self.coverage[J] - self.coverage[I], 1e-6),
                ),
            )
            ub = self._body_velocity_at(
                ti.cast(I, ti.f32) + 0.5 + eta * ti.cast(c, ti.f32),
                self.solid_body_id[J],
            )
            # Extrapolate from the wall and the opposite fluid node. Unlike
            # center+(ub-center)/eta, coefficients remain bounded as eta->0.
            if self._connected(I, -c):
                neighbor = (2 * ub + (eta - 1) * self.velocity_lattice[I - c]) / (
                    1 + eta
                )
            else:
                neighbor = 2 * ub - center
        return neighbor

    @ti.func
    def _regularized_stress(self, I, rho, tau):
        """3D form of v3's HRR finite-difference, time-centered shear stress."""
        gradient = ti.Matrix.zero(ti.f32, 3, 3)
        for q in range(Q):
            if q != REST:
                c = direction(q)
                neighbor = self._stress_neighbor_velocity(I, c)
                gradient += 3 * weight(q) * neighbor.outer_product(ti.cast(c, ti.f32))
        return -rho * tau / 3 * (gradient + gradient.transpose())

    @ti.func
    def _hrr_cubic_correction(self, I):
        derivative = ti.Vector.zero(ti.f32, 3)
        for q in range(Q):
            if q != REST:
                c = direction(q)
                J = I + c
                rho = density(self.water_phase[I], self.rho_air)
                if self._fluid(J):
                    rho = density(self.water_phase[J], self.rho_air)
                neighbor = self._stress_neighbor_velocity(I, c)
                derivative += 3 * weight(q) * ti.cast(c, ti.f32) * rho * neighbor**3
        return -0.5 * derivative

    @ti.func
    def _touches_ice(self, I):
        near = False
        for q in range(Q):
            J = I + direction(q)
            if self._inside(J) and self.solid_body_id[J] >= 0:
                near = True
        return near

    @ti.kernel
    def _prepare_boundaries(self):
        """Read old populations before *any* streaming thread overwrites them."""
        for I in ti.grouped(self.water_phase):
            if self._fluid(I) and self._touches_ice(I):
                raw = ti.Vector.zero(ti.f32, Q)
                for q in ti.static(range(Q)):
                    raw[q] = self.f_post[I, q]
                for q in range(Q):
                    c = direction(q)
                    J = I + c
                    if self._inside(J) and self.solid_body_id[J] >= 0:
                        b = self.solid_body_id[J]
                        cf = ti.cast(c, ti.f32)
                        eta = self._boundary_link_fraction(I, J)
                        point = ti.cast(I, ti.f32) + 0.5 + eta * cf
                        ub = self._body_velocity_at(point, b)
                        rho = density(self.water_phase[I], self.rho_air)
                        neq = self.f[I, q] - momentum_equilibrium(
                            q, self.pressure_lattice[I], rho, self.velocity_lattice[I]
                        )
                        reflected = (
                            momentum_equilibrium(
                                26 - q, self.pressure_lattice[I], rho, ub
                            )
                            + neq
                            + eta * raw[26 - q]
                        ) / (1 + eta)
                        impulse = (cf - ub) * raw[q] - (-cf - ub) * reflected
                        pref = (1 - eta) * self.reference_pressure[
                            I.y
                        ] + eta * self.reference_pressure[J.y]
                        impulse += 6 * weight(q) * pref * cf
                        capillary = (
                            -3
                            * weight(q)
                            * (
                                self.capillary_stress[I]
                                + self._capillary_neighbor(I, c)
                            )
                            @ cf
                        )
                        ti.atomic_add(
                            self.hydrodynamic_force[b],
                            ti.cast(impulse + capillary, ti.f64),
                        )
                        torque = (point - self.body_center[b]).cross(impulse)
                        torque += (
                            ti.cast(I, ti.f32) + 0.5 + 0.5 * cf - self.body_center[b]
                        ).cross(capillary)
                        ti.atomic_add(
                            self.hydrodynamic_torque[b], ti.cast(torque, ti.f64)
                        )
                        self.f_post[I, q] = reflected
                        self.h_post[I, q] -= (
                            6 * weight(q) * self.water_phase[I] * cf.dot(ub)
                        )

    @ti.kernel
    def _stream(self, phase_only: ti.template()):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                for q in ti.static(range(Q)):
                    J = I + direction(q)
                    if self._fluid(J):
                        self.h[J, q] = self.h_post[I, q]
                        if ti.static(not phase_only):
                            self.f[J, q] = self.f_post[I, q]
                    else:
                        self.h[I, 26 - q] = self.h_post[I, q]
                        if ti.static(not phase_only):
                            self.f[I, 26 - q] = self.f_post[I, q]

    @ti.kernel
    def _stream_recover(self, block_size: ti.template()):
        """Pull both populations and recover phase/momentum in one memory pass.

        Pressure alone waits for the new phase gradient. Accumulations retain
        the f64 precision of the separate recovery kernels.
        """
        ti.loop_config(block_dim=block_size)
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                total = ti.cast(0.0, ti.f64)
                momentum = ti.Vector.zero(ti.f64, 3)
                moving = ti.cast(0.0, ti.f64)
                for q in ti.static(range(Q)):
                    J = I - direction(q)
                    hp, fp = 0.0, 0.0
                    if self._fluid(J):
                        hp, fp = self.h_post[J, q], self.f_post[J, q]
                    else:
                        hp, fp = self.h_post[I, 26 - q], self.f_post[I, 26 - q]
                    self.h[I, q], self.f[I, q] = hp, fp
                    total += ti.cast(hp, ti.f64)
                    momentum += ti.cast(direction(q), ti.f64) * ti.cast(fp, ti.f64)
                    if ti.static(q != REST):
                        moving += ti.cast(fp, ti.f64)
                bounded = total
                if not ti.math.isnan(total) and not ti.math.isinf(total):
                    bounded = ti.min(1.0, ti.max(0.0, total))
                self.h[I, REST] += ti.cast(bounded - total, ti.f32)
                phi = ti.cast(bounded, ti.f32)
                self.water_phase[I] = phi
                ti.atomic_add(self.phase_clip_volume[None], bounded - total)
                self.velocity_lattice[I] = ti.cast(
                    (momentum + 0.5 * ti.cast(self.force_density[I], ti.f64))
                    / density(phi, self.rho_air),
                    ti.f32,
                )
                self.moving_population_sum[I] = moving
            else:
                self.water_phase[I] = 0.0
                self.velocity_lattice[I] = ti.Vector.zero(ti.f32, 3)
                if self.solid_body_id[I] >= 0:
                    self.velocity_lattice[I] = self._body_velocity_at(
                        ti.cast(I, ti.f32) + 0.5, self.solid_body_id[I]
                    )

    @ti.kernel
    def _recover_pressure(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                u = self.velocity_lattice[I]
                rho = density(self.water_phase[I], self.rho_air)
                gradient = self.phase_gradient_lattice[I]
                self.pressure_lattice[I] = ti.cast(
                    (
                        self.moving_population_sum[I]
                        + 0.5 * (1 - self.rho_air) * u.dot(gradient)
                        + momentum_rest_velocity(rho, u)
                    )
                    / (3 * (1 - W0)),
                    ti.f32,
                )

    @ti.kernel
    def _recover_phase(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                total = ti.cast(0.0, ti.f64)
                for q in range(Q):
                    total += ti.cast(self.h[I, q], ti.f64)
                bounded = total
                if not ti.math.isnan(total) and not ti.math.isinf(total):
                    bounded = ti.min(1.0, ti.max(0.0, total))
                self.h[I, REST] += ti.cast(bounded - total, ti.f32)
                self.water_phase[I] = ti.cast(bounded, ti.f32)
                ti.atomic_add(self.phase_clip_volume[None], bounded - total)
            else:
                self.water_phase[I] = 0.0

    @ti.kernel
    def _recover_momentum(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                rho = density(self.water_phase[I], self.rho_air)
                momentum = ti.Vector.zero(ti.f64, 3)
                moving = ti.cast(0.0, ti.f64)
                for q in range(Q):
                    momentum += ti.cast(direction(q), ti.f64) * ti.cast(
                        self.f[I, q], ti.f64
                    )
                    if q != REST:
                        moving += ti.cast(self.f[I, q], ti.f64)
                u = ti.cast(
                    (momentum + 0.5 * ti.cast(self.force_density[I], ti.f64)) / rho,
                    ti.f32,
                )
                repaired, pressure = False, 0.0
                if ti.static(self.config.gas_state_repair):
                    speed = ti.abs(u).sum()
                    jump = ti.abs(u - self.old_velocity[I]).sum()
                    if (
                        (self.water_phase[I] < 0.5 or speed > 1.0)
                        and speed > 0.25
                        and jump > 0.15
                    ):
                        total, count, peak, psum = (
                            ti.Vector.zero(ti.f32, 3),
                            0.0,
                            0.0,
                            0.0,
                        )
                        for q in range(Q):
                            c = direction(q)
                            if q != REST and self._connected(I, c):
                                previous = self.old_velocity[I + c]
                                peak = ti.max(peak, ti.abs(previous).sum())
                                if ti.abs(previous).sum() <= 0.25:
                                    total += previous
                                    psum += self.old_pressure[I + c]
                                    count += 1
                        if (
                            speed > ti.max(0.25, 2 * peak + 0.05)
                            or speed > 1
                            and jump > 0.5
                        ):
                            u = total / ti.max(count, 1.0)
                            pressure = psum / ti.max(count, 1.0)
                            self._set_equilibrium(I, self.water_phase[I], pressure, u)
                            ti.atomic_add(self.repair_count[None], 1)
                            repaired = True
                if not repaired:
                    gradient = self.phase_gradient_lattice[I]
                    pressure = (
                        moving
                        + 0.5 * (1 - self.rho_air) * u.dot(gradient)
                        + momentum_rest_velocity(rho, u)
                    ) / (3 * (1 - W0))
                self.velocity_lattice[I], self.pressure_lattice[I] = u, pressure
            else:
                self.velocity_lattice[I] = ti.Vector.zero(ti.f32, 3)
                if self.solid_body_id[I] >= 0:
                    self.velocity_lattice[I] = self._body_velocity_at(
                        ti.cast(I, ti.f32) + 0.5, self.solid_body_id[I]
                    )

    @ti.kernel
    def _seed_refill(self) -> ti.types.vector(2, ti.i32):
        self.openings[None] = 0
        covered = 0
        self.refill_lower[None] = ti.Vector(self.shape)
        self.refill_upper[None] = ti.Vector([-1, -1, -1])
        for I in ti.grouped(self.water_phase):
            valid = self._fluid(I) and self.previous_body_id[I] == -1
            u = self.velocity_lattice[I]
            self.refill_state[I] = ti.Vector(
                [self.water_phase[I], self.pressure_lattice[I], u.x, u.y, u.z]
            )
            self.refill_valid[I] = ti.cast(valid, ti.i8)
            if self._fluid(I) and not valid:
                ti.atomic_add(self.openings[None], 1)
                ti.atomic_min(self.refill_lower[None], I)
                ti.atomic_max(self.refill_upper[None], I)
            if self.previous_body_id[I] == -1 and not self._fluid(I):
                covered += 1
                # Cached stresses at newly covered nodes and their neighbors
                # must be refreshed even if no new fluid node was exposed.
                ti.atomic_min(self.refill_lower[None], I)
                ti.atomic_max(self.refill_upper[None], I)
        return ti.Vector([self.openings[None], covered])

    @ti.kernel
    def _propagate_refill(self):
        self.remaining[None] = 0
        for I in ti.grouped(self.water_phase):
            valid, value = self.refill_valid[I], self.refill_state[I]
            if self._fluid(I) and valid == 0:
                total, count = ti.Vector.zero(ti.f32, 5), 0.0
                for q in range(Q):
                    c = direction(q)
                    if q != REST and self._connected(I, c):
                        if self.refill_valid[I + c]:
                            total += weight(q) * self.refill_state[I + c]
                            count += weight(q)
                if count > 0:
                    value, valid = total / count, ti.cast(1, ti.i8)
                else:
                    ti.atomic_add(self.remaining[None], 1)
            self.refill_next[I], self.refill_next_valid[I] = value, valid

    @ti.kernel
    def _commit_refill(self):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I) and self.previous_body_id[I] != -1:
                if self.refill_valid[I]:
                    state = self.refill_state[I]
                    self.water_phase[I], self.pressure_lattice[I] = state[0], state[1]
                    self.velocity_lattice[I] = ti.Vector([state[2], state[3], state[4]])
                else:
                    self.solid_body_id[I] = -3
            if not self._fluid(I):
                self.water_phase[I] = 0.0
                self.velocity_lattice[I] = ti.Vector.zero(ti.f32, 3)
                if self.solid_body_id[I] >= 0:
                    self.velocity_lattice[I] = self._body_velocity_at(
                        ti.cast(I, ti.f32) + 0.5, self.solid_body_id[I]
                    )

    @ti.kernel
    def _project_refill(self):
        lo, hi = self.refill_lower[None], self.refill_upper[None] + 1
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            if self._fluid(I) and self.previous_body_id[I] != -1:
                self.force_density[I] = self._force_at(I, False)
                self._set_equilibrium(
                    I,
                    self.water_phase[I],
                    self.pressure_lattice[I],
                    self.velocity_lattice[I],
                )

    # Global water-inventory correction disabled on 2026-10-21.
    # Keep the implementation here for a reversible ablation.
    # @ti.kernel
    # def _measure_phase_balance(self):
    #     self.phase_balance_sum[None] = 0.0
    #     self.phase_balance_capacity[None] = 0.0
    #     for I in ti.grouped(self.water_phase):
    #         if self._fluid(I):
    #             phi = ti.cast(self.water_phase[I], ti.f64)
    #             ti.atomic_add(self.phase_balance_sum[None], phi)
    #             ti.atomic_add(self.phase_balance_capacity[None], phi * (1 - phi))
    #
    # @ti.kernel
    # def _project_phase_volume(self, coefficient: ti.f64):
    #     for I in ti.grouped(self.water_phase):
    #         if self._fluid(I):
    #             before = self.water_phase[I]
    #             after = ti.cast(
    #                 ti.min(
    #                     1.0, ti.max(0.0, before + coefficient * before * (1 - before))
    #                 ),
    #                 ti.f32,
    #             )
    #             delta = after - before
    #             if delta != 0:
    #                 self.water_phase[I] = after
    #                 u = self.velocity_lattice[I]
    #                 delta_rho = (1 - self.rho_air) * delta
    #                 # Change the transported amount while retaining local
    #                 # velocity and dynamic pressure. Keep the existing
    #                 # nonequilibrium populations rather than resetting them.
    #                 for q in range(Q):
    #                     self.h[I, q] += phase_equilibrium(q, delta, u)
    #                     self.f[I, q] += momentum_equilibrium(q, 0.0, delta_rho, u)
    #                 ti.atomic_add(
    #                     self.phase_volume_projection_m3[None],
    #                     ti.cast(delta, ti.f64) * self.cell_volume,
    #                 )
    #                 ti.atomic_add(
    #                     self.phase_projection_momentum_n_s[None],
    #                     ti.cast(delta_rho * u, ti.f64)
    #                     * self.scales.mass_kg
    #                     * self.scales.velocity_m_s,
    #                 )
    #
    # def _conserve_phase_volume(self):
    #     """Bounded interface redistribution for this closed water inventory.
    #
    #     A sharp moving mask plus local extrapolation does not conserve volume.
    #     Redistribute its defect over diffuse interfaces, with weights phi(1-phi),
    #     instead of leaving artificial air holes next to melting ice. The amount
    #     and associated momentum adjustment are reported explicitly.
    #     """
    #     target = self.initial_phase_volume_m3 / self.cell_volume
    #     # A uniform float32 phase cannot represent arbitrary total volumes:
    #     # rounding all equal cells has a systematic, not statistical, error.
    #     tolerance = max(1e-6, np.finfo(np.float32).eps * target)
    #     for _ in range(4):
    #         self._measure_phase_balance()
    #         error = target - float(self.phase_balance_sum[None])
    #         if abs(error) <= tolerance:
    #             return
    #         capacity = float(self.phase_balance_capacity[None])
    #         if not math.isfinite(error) or capacity <= 1e-20:
    #             raise RuntimeError(
    #                 f"cannot conserve phase volume at step {self.steps + 1}"
    #             )
    #         self._project_phase_volume(max(-1.0, min(1.0, error / capacity)))
    #     self._measure_phase_balance()
    #     if abs(target - float(self.phase_balance_sum[None])) > tolerance:
    #         raise RuntimeError(
    #             f"phase volume projection did not converge at step {self.steps + 1}"
    #         )
    #

    def _refresh_geometry(self, phase_changed=True):
        self._rasterize()
        remaining, covered = (int(value) for value in self._seed_refill())
        has_openings = remaining > 0
        self.pending_cells = 0
        while remaining:
            self._propagate_refill()
            new = int(self.remaining[None])
            # Kernels cache self.field identities. Attribute swapping would
            # repeatedly read the first buffer and strand multi-layer cavities.
            self.refill_state.copy_from(self.refill_next)
            self.refill_valid.copy_from(self.refill_next_valid)
            if new == remaining:
                self.pending_cells = new
                break
            remaining = new
        self._commit_refill()
        # Global water-inventory correction disabled for the unconstrained run.
        # if self.config.conserve_phase_volume:
        #     self._conserve_phase_volume()
        # Also refresh when sub-cell movement/melting changes mirror geometry
        # without changing labels. An unchanged geometry yields an empty box.
        self._refill_gradients()
        if has_openings or covered:
            self._project_refill()
        # Conduction changes energy, not phi. If no fluid/solid cells changed,
        # volumes are already synchronized and a remap is the identity.
        if phase_changed or has_openings or covered:
            self._synchronize_water()

    @ti.kernel
    def _reduce_mass(self):
        for b in range(self.body_count):
            self.reduced_mass[b] = 0.0
            self.reduced_first[b] = ti.Vector.zero(ti.f64, 3)
            self.reduced_second[b] = ti.Matrix.zero(ti.f64, 3, 3)
            self.reduced_lower[b] = ti.Vector([1e9, 1e9, 1e9])
            self.reduced_upper[b] = ti.Vector([-1e9, -1e9, -1e9])
        # One register reduction per material column, then one set of global
        # atomics. This removes contention without new scratch buffers or
        # changing the precision of mass, center of mass or inertia.
        ti.loop_config(block_dim=64)
        for b, i, j in ti.ndrange(
            self.body_count, self.material_mass.shape[1], self.material_mass.shape[2]
        ):
            total = ti.cast(0.0, ti.f64)
            first = ti.Vector.zero(ti.f64, 3)
            second = ti.Matrix.zero(ti.f64, 3, 3)
            lower = ti.Vector([1e9, 1e9, 1e9])
            upper = ti.Vector([-1e9, -1e9, -1e9])
            for k in range(self.material_mass.shape[3]):
                mass = self.material_mass[b, i, j, k] / self.scales.mass_kg
                if mass > 0:
                    point = ti.Vector(
                        [i + 0.5, j + 0.5, k + 0.5], dt=ti.f64
                    ) - 0.5 * ti.cast(self.material_size[b], ti.f64)
                    total += mass
                    first += mass * point
                    second += mass * (
                        (point.dot(point) + 1 / 6) * ti.Matrix.identity(ti.f64, 3)
                        - point.outer_product(point)
                    )
                    lower = ti.min(lower, ti.cast(point - 0.5, ti.f32))
                    upper = ti.max(upper, ti.cast(point + 0.5, ti.f32))
            if total > 0:
                ti.atomic_add(self.reduced_mass[b], total)
                ti.atomic_add(self.reduced_first[b], first)
                ti.atomic_add(self.reduced_second[b], second)
                ti.atomic_min(self.reduced_lower[b], lower)
                ti.atomic_max(self.reduced_upper[b], upper)

    @ti.kernel
    def _scan_stability(self):
        self.invalid_index[None] = self.nx * self.ny * self.nz
        self.maximum_speed[None] = 0.0
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                speed = ti.abs(self.velocity_lattice[I]).sum()
                ti.atomic_max(self.maximum_speed[None], speed)
                invalid = (
                    ti.math.isnan(speed)
                    or ti.math.isinf(speed)
                    or speed > self.config.maximum_velocity_lattice
                )
                p = self.pressure_lattice[I]
                invalid = invalid or ti.math.isnan(p) or ti.math.isinf(p)
                phi = self.water_phase[I]
                invalid = invalid or ti.math.isnan(phi) or ti.math.isinf(phi)
                if invalid:
                    ti.atomic_min(
                        self.invalid_index[None], (I.x * self.ny + I.y) * self.nz + I.z
                    )
        self.peak_speed[None] = ti.max(self.peak_speed[None], self.maximum_speed[None])

    def check_stability(self):
        self._scan_stability()
        index = int(self.invalid_index[None])
        if index < math.prod(self.shape):
            I = (
                index // (self.ny * self.nz),
                index // self.nz % self.ny,
                index % self.nz,
            )
            raise RuntimeError(
                f"unstable fluid at step {self.steps + 1}, cell {I}: "
                f"phi={self.water_phase[I]}, velocity={self.velocity_lattice[I]}"
            )
        return float(self.maximum_speed[None])

    def step(self, num_steps=1):
        num_steps = integer("num_steps", num_steps, 0)
        for _ in range(num_steps):
            if not self._gradient_cache_valid:
                self._gradients()
                self._gradient_cache_valid = True
            self._collide_phase()
            self._forces(True)
            self._collide_momentum()
            self.hydrodynamic_force.fill(0)
            self.hydrodynamic_torque.fill(0)
            self._prepare_boundaries()
            if self.config.gas_state_repair:
                self._stream(False)
                self._recover_phase()
                self._gradients()
                self._recover_momentum()
            else:
                self._stream_recover(256)
                self._gradients()
                self._recover_pressure()
            # Check before changed geometry can hide a divergent fluid node.
            speed_bound = None
            if self.config.check_stability:
                speed_bound = self.check_stability()
            self.rigid.advance()
            self._refresh_geometry()
            self._advance_advection(speed_bound)
            self._thermal_pending += 1
            if self._thermal_pending >= self.config.thermal.update_interval_lbm_steps:
                self._advance_conduction(self._thermal_pending * self.scales.dt_s)
                self._thermal_pending = 0
            self.steps += 1

    def flush_thermal(self):
        """Resolve the final partial thermal interval at the end of a run."""
        if self._thermal_pending:
            self._advance_conduction(self._thermal_pending * self.scales.dt_s)
            self._thermal_pending = 0

    def snapshot(self):
        ids = self.solid_body_id.to_numpy()
        phi = self.water_phase.to_numpy()
        volume = self.water_volume.to_numpy()
        energy = self.water_energy.to_numpy()
        cp = (
            self.config.rho_water
            * self.config.thermal.properties.specific_heat_water_j_kg_k
        )
        temp = np.full(
            self.shape, self.config.thermal.initial_air_temperature_c, np.float32
        )
        wet = volume > 1e-30
        temp[wet] = (
            energy[wet] / (cp * volume[wet])
            + self.config.thermal.properties.melting_temperature_c
        ).astype(np.float32)
        self._sample_ice_temperature()
        ice_temp = self.sampled_ice_temperature.to_numpy()
        temp[ids >= 0] = ice_temp[ids >= 0]
        return {
            "water_phase": phi,
            "solid_body_id": ids,
            "ice_fraction": self.coverage.to_numpy(),
            "velocity_m_s": self.velocity_lattice.to_numpy() * self.scales.velocity_m_s,
            "pressure_pa": (
                self.pressure_lattice.to_numpy()
                + self.reference_pressure.to_numpy()[None, :, None]
            )
            * self.scales.pressure_pa,
            "temperature_c": temp,
        }

    def diagnostics(self):
        self._scan_stability()
        volume = self.water_volume.to_numpy()
        rho = self.rigid
        centers = self.body_center.to_numpy() * self.scales.dx_m
        result = {
            "step": self.steps,
            "time_s": self.physical_time_s,
            "water_volume_m3": float(volume.sum()),
            "water_volume_change_ratio": (
                float(volume.sum() / self.initial_water_volume_m3 - 1)
                if self.initial_water_volume_m3 > 0
                else None
            ),
            "water_energy_j": float(self.water_energy.to_numpy().sum()),
            "ice_sensible_energy_j": float(self.material_energy.to_numpy().sum()),
            "maximum_fluid_speed_lattice": float(self.maximum_speed[None]),
            "peak_fluid_speed_lattice": float(self.peak_speed[None]),
            "phase_volume_change_ratio": float(
                self.water_phase.to_numpy().sum(dtype=np.float64)
                * self.cell_volume
                / self.initial_phase_volume_m3
                - 1
            ),
            "phase_volume_projection_m3": float(self.phase_volume_projection_m3[None]),
            **{
                f"phase_projection_momentum_{axis}_n_s": float(
                    self.phase_projection_momentum_n_s[None][a]
                )
                for a, axis in enumerate("xyz")
            },
            "ice_mass_kg": float(rho.mass.sum() * self.scales.mass_kg),
            "melted_mass_kg": float(
                self.initial_ice_mass_kg.sum() - rho.mass.sum() * self.scales.mass_kg
            ),
            "gas_repair_count": int(self.repair_count[None]),
            "phase_clip_volume_m3": float(self.phase_clip_volume[None])
            * self.scales.dx_m**3,
            "thermal_remap_energy_j": float(self.remap_energy[None]),
            "discarded_melt_heat_j": float(self.discarded_heat[None]),
            "pending_refill_cells": self.pending_cells,
        }
        for b in range(self.body_count):
            result[f"ice_{b}_mass_kg"] = float(rho.mass[b] * self.scales.mass_kg)
            for a, label in enumerate("xyz"):
                result[f"ice_{b}_{label}_m"] = float(centers[b, a])
                result[f"ice_{b}_v{label}_m_s"] = float(
                    rho.velocity[b, a] * self.scales.velocity_m_s
                )
                result[f"ice_{b}_omega_{label}_s_1"] = float(
                    rho.omega[b, a] / self.scales.dt_s
                )
            for a, label in enumerate("wxyz"):
                result[f"ice_{b}_q{label}"] = float(rho.q[b, a])
        return result
