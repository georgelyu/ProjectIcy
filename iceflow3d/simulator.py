"""HOME central-moment flow, D3Q7 in-place phase, moving and melting ice."""
# ruff: noqa: E741 -- I and J are Taichi grouped grid indices.

import math
import numpy as np
import taichi as ti

from .config import IceFlowConfig, LatticeScales, integer
from .lattice import (
    Q,
    REST,
    density,
    direction,
    weight,
)
from .home import HomeMomentum
from .phase import InPlacePhase, phase_equilibrium7
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
    elif ti.lang.impl.current_cfg().default_fp != ti.f32:
        raise RuntimeError("IceFlow3D requires Taichi default_fp=ti.f32")


@ti.data_oriented
class IceFlow3D(HomeMomentum, InPlacePhase, MirrorGradient3D, Thermal3D):
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
        if config.gas_state_repair:
            raise ValueError("HOME does not use gas-state repair")
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
        self.home = ti.Vector.field(10, ti.f32, shape=self.shape, layout=ti.Layout.SOA)
        self.home_post = ti.Vector.field(
            10, ti.f32, shape=self.shape, layout=ti.Layout.SOA
        )
        # Rebuilt by every HOME stream; scratch masks are not restart state.
        self._home_boundary = ti.field(ti.i8, shape=self.shape)
        self._transient_field_names = self._transient_field_names | {"_home_boundary"}
        self.h = ti.field(ti.f32)
        ti.root.dense(ti.l, 7).dense(ti.ijk, self.shape).place(self.h)
        self.phase_parity = ti.field(ti.i32, shape=())
        for name in (
            "water_phase",
            "pressure_lattice",
            "old_pressure",
            "coverage",
        ):
            setattr(self, name, ti.field(ti.f32, shape=self.shape))
        for name in (
            "velocity_lattice",
            "old_velocity",
            "force_density",
            "home_acceleration",
            "phase_gradient_lattice",
            "phase_mass_flux",
        ):
            setattr(self, name, ti.Vector.field(3, ti.f32, shape=self.shape))
        self.capillary_stress = ti.Matrix.field(3, 3, ti.f32, shape=self.shape)
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
        self.hydrodynamic_force = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.hydrodynamic_torque = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.reduced_mass = ti.field(ti.f32, shape=self.body_count)
        self.reduced_first = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.reduced_second = ti.Matrix.field(3, 3, ti.f32, shape=self.body_count)
        self.reduced_lower = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.reduced_upper = ti.Vector.field(3, ti.f32, shape=self.body_count)
        self.invalid_index = ti.field(ti.i32, shape=())
        self.maximum_speed = ti.field(ti.f32, shape=())
        self.peak_speed = ti.field(ti.f32, shape=())
        self.repair_count = ti.field(ti.i32, shape=())
        self.phase_clip_volume = ti.field(ti.f32, shape=())
        self.phase_balance_sum = ti.field(ti.f32, shape=())
        self.phase_balance_capacity = ti.field(ti.f32, shape=())
        self.phase_volume_projection_m3 = ti.field(ti.f32, shape=())
        self.phase_projection_momentum_n_s = ti.Vector.field(3, ti.f32, shape=())
        self.pending_cells = 0
        self._allocate_thermal()
        self.rigid = RigidBodies(self)
        self._initialize_material()
        self.rigid.update_mass()
        self._rasterize()
        self._initialize_phase()
        for _ in range(config.phase_warmup_steps):
            self._gradients()
            self._collide_phase(moving_walls=False)
            self._recover_phase()
        self._initialize_water_heat()
        self._build_reference()
        self._gradients()
        self._forces(False)
        self._seed_populations()
        self.initial_ice_mass_kg = self.rigid.mass.copy() * self.scales.mass_kg
        self.initial_water_volume_m3 = float(
            self.water_volume.to_numpy().sum(dtype=float)
        )
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
    def _upload_rigid_state(self, values: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for b in range(self.body_count):
            for a in ti.static(range(3)):
                self.body_origin[b][a] = values[b, a]
                self.body_center[b][a] = values[b, 12 + a]
                self.body_velocity[b][a] = values[b, 15 + a]
                self.body_angular_velocity[b][a] = values[b, 18 + a]
                for c in ti.static(range(3)):
                    self.body_rotation[b][a, c] = values[b, 3 + 3 * a + c]

    @ti.kernel
    def _download_rigid_state(
        self, values: ti.types.ndarray(dtype=ti.f32, ndim=2), mass: ti.template()
    ):
        for b in range(self.body_count):
            if ti.static(mass):
                values[b, 0] = self.reduced_mass[b]
                for a in ti.static(range(3)):
                    values[b, 1 + a] = self.reduced_first[b][a]
                    values[b, 13 + a] = self.reduced_lower[b][a]
                    values[b, 16 + a] = self.reduced_upper[b][a]
                    for c in ti.static(range(3)):
                        values[b, 4 + 3 * a + c] = self.reduced_second[b][a, c]
            else:
                for a in ti.static(range(3)):
                    values[b, a] = self.hydrodynamic_force[b][a]
                    values[b, 3 + a] = self.hydrodynamic_torque[b][a]

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
            for q in ti.static(range(7)):
                self.h[I, q] = phase_equilibrium7(q, phi, ti.Vector.zero(ti.f32, 3))

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
    def _force_at(
        self,
        I,
        with_flux: ti.template(),
        interior: ti.template() = False,
        tank: ti.template() = False,
    ):
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
                wall = False
                # The center-stress contribution sums to zero by quadrature
                # symmetry. Specialize the fluid stencil; handle reflected
                # wall tensors only on links actually touching a solid.
                for q in ti.static(range(Q)):
                    if ti.static(q != REST):
                        c = direction(q)
                        g = ti.Vector.zero(ti.f32, 3)
                        valid = False
                        if ti.static(tank):
                            # A one-link ghost of a planar tank wall reflects
                            # to the nearest fluid layer. P(R*g)=R*P(g)*R.
                            J = I + c
                            b = ti.static(self.config.boundary_cells)
                            K = ti.max(b, ti.min(ti.Vector(self.shape) - b - 1, J))
                            g = self.phase_gradient_lattice[K]
                            for a in ti.static(range(3)):
                                if J[a] != K[a]:
                                    g[a] = -g[a]
                            valid = True
                        else:
                            if self._force_link_fluid(I + c, interior, False):
                                g = self.phase_gradient_lattice[I + c]
                                valid = True
                            else:
                                wall = True
                        if valid:
                            # P*c = K*(|grad(phi)|^2*c - (grad(phi).c)*grad(phi)).
                            # Read three gradient scalars rather than a 3x3
                            # cached tensor, retaining the same stress law.
                            cf = ti.cast(c, ti.f32)
                            force += (
                                4.5
                                * self.sigma
                                * self.config.interface_width
                                * weight(q)
                            ) * (g.dot(g) * cf - g.dot(cf) * g)
                if not ti.static(interior or tank) and wall:
                    for q in range(Q):
                        if q != REST:
                            c = direction(q)
                            if not self._fluid(I + c):
                                force += (
                                    3
                                    * weight(q)
                                    * self._capillary_neighbor(I, c)
                                    @ ti.cast(c, ti.f32)
                                )
            if ti.static(with_flux):
                for a, sign in ti.static(ti.ndrange(3, 2)):
                    c = ti.Vector.zero(ti.i32, 3)
                    c[a] = 2 * sign - 1
                    J = I + c
                    if self._force_link_fluid(J, interior, tank):
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
                        # Velocity formulation: -div(J u) + u div(J).
                        # Uniform velocity is invariant even as phase diffuses.
                        flux = mass_flux * (donor - self.velocity_lattice[I])
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

    def _forces(self, with_flux, moment_pressure=False):
        self._classify_home()
        self._forces_bulk(with_flux, moment_pressure)
        self._forces_tank(with_flux, moment_pressure)
        self._forces_boundary(with_flux, moment_pressure)

    @ti.kernel
    def _forces_bulk(self, with_flux: ti.template(), moment_pressure: ti.template()):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 0:
                force = self._force_at(I, with_flux, True)
                self.force_density[I] = force
                self.home_acceleration[I] = self._home_acceleration_at(
                    I, force, True, moment_pressure
                )

    @ti.kernel
    def _forces_tank(self, with_flux: ti.template(), moment_pressure: ti.template()):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 1:
                force = self._force_at(I, with_flux, False, True)
                self.force_density[I] = force
                self.home_acceleration[I] = self._home_acceleration_at(
                    I, force, False, moment_pressure, True
                )

    @ti.kernel
    def _forces_boundary(
        self, with_flux: ti.template(), moment_pressure: ti.template()
    ):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 2:
                force = self._force_at(I, with_flux)
                self.force_density[I] = force
                self.home_acceleration[I] = self._home_acceleration_at(
                    I, force, False, moment_pressure
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

    def _seed_refill(self):
        counts = self._find_refill_changes()
        if int(counts[0]) > 0:
            self._seed_refill_region()
        return counts

    @ti.kernel
    def _find_refill_changes(self) -> ti.types.vector(2, ti.i32):
        self.openings[None] = 0
        covered = 0
        self.refill_lower[None] = ti.Vector(self.shape)
        self.refill_upper[None] = ti.Vector([-1, -1, -1])
        for I in ti.grouped(self.water_phase):
            valid = self._fluid(I) and self.previous_body_id[I] == -1
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
    def _seed_refill_region(self):
        # Only newly exposed cells consume the donor cache. Include their
        # complete one-link halo; untouched regions never enter propagation.
        lo = ti.max(self.refill_lower[None] - 1, 0)
        hi = ti.min(self.refill_upper[None] + 2, ti.Vector(self.shape))
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            u = self.velocity_lattice[I]
            self.refill_state[I] = ti.Vector(
                [self.water_phase[I], self.pressure_lattice[I], u.x, u.y, u.z]
            )
            self.refill_valid[I] = ti.cast(
                self._fluid(I) and self.previous_body_id[I] == -1, ti.i8
            )

    @ti.kernel
    def _propagate_refill(self):
        self.remaining[None] = 0
        lo = ti.max(self.refill_lower[None] - 1, 0)
        hi = ti.min(self.refill_upper[None] + 2, ti.Vector(self.shape))
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
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
    def _accept_refill_iteration(self):
        lo = ti.max(self.refill_lower[None] - 1, 0)
        hi = ti.min(self.refill_upper[None] + 2, ti.Vector(self.shape))
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            self.refill_state[I] = self.refill_next[I]
            self.refill_valid[I] = self.refill_next_valid[I]

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

    def _project_refill(self):
        self._refill_forces()
        self._write_refill_equilibrium()

    @ti.kernel
    def _refill_forces(self):
        lo, hi = self.refill_lower[None], self.refill_upper[None] + 1
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            if self._fluid(I) and self.previous_body_id[I] != -1:
                self.force_density[I] = self._force_at(I, False)
                self.home_acceleration[I] = self._home_acceleration_at(
                    I, self.force_density[I]
                )

    @ti.kernel
    def _write_refill_equilibrium(self):
        lo, hi = self.refill_lower[None], self.refill_upper[None] + 1
        for i, j, k in ti.ndrange((lo.x, hi.x), (lo.y, hi.y), (lo.z, hi.z)):
            I = ti.Vector([i, j, k])
            if self._fluid(I) and self.previous_body_id[I] != -1:
                self._set_equilibrium(
                    I,
                    self.water_phase[I],
                    self.pressure_lattice[I],
                    self.velocity_lattice[I],
                )

    def _refresh_geometry(self, phase_changed=True):
        self._rasterize()
        self._repair_phase_layout()
        remaining, covered = (int(value) for value in self._seed_refill())
        has_openings = remaining > 0
        self.pending_cells = 0
        while remaining:
            self._propagate_refill()
            new = int(self.remaining[None])
            # Kernels cache self.field identities. Attribute swapping would
            # repeatedly read the first buffer and strand multi-layer cavities.
            self._accept_refill_iteration()
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
            self.reduced_first[b] = ti.Vector.zero(ti.f32, 3)
            self.reduced_second[b] = ti.Matrix.zero(ti.f32, 3, 3)
            self.reduced_lower[b] = ti.Vector([1e9, 1e9, 1e9])
            self.reduced_upper[b] = ti.Vector([-1e9, -1e9, -1e9])
        # One register reduction per material column, then one set of global
        # atomics, all in FP32. Host pose/contact solves retain NumPy FP64.
        ti.loop_config(block_dim=64)
        for b, i, j in ti.ndrange(
            self.body_count, self.material_mass.shape[1], self.material_mass.shape[2]
        ):
            total = ti.cast(0.0, ti.f32)
            first = ti.Vector.zero(ti.f32, 3)
            second = ti.Matrix.zero(ti.f32, 3, 3)
            lower = ti.Vector([1e9, 1e9, 1e9])
            upper = ti.Vector([-1e9, -1e9, -1e9])
            for k in range(self.material_mass.shape[3]):
                mass = self.material_mass[b, i, j, k] / self.scales.mass_kg
                if mass > 0:
                    point = ti.Vector(
                        [i + 0.5, j + 0.5, k + 0.5], dt=ti.f32
                    ) - 0.5 * ti.cast(self.material_size[b], ti.f32)
                    total += mass
                    first += mass * point
                    second += mass * (
                        (point.dot(point) + 1 / 6) * ti.Matrix.identity(ti.f32, 3)
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

    @ti.func
    def _scan_stability_at(self, I, phi, p):
        speed = ti.abs(self.velocity_lattice[I]).sum()
        ti.atomic_max(self.maximum_speed[None], speed)
        invalid = (
            ti.math.isnan(speed)
            or ti.math.isinf(speed)
            or speed > self.config.maximum_velocity_lattice
            or ti.math.isnan(p)
            or ti.math.isinf(p)
            or ti.math.isnan(phi)
            or ti.math.isinf(phi)
        )
        if invalid:
            ti.atomic_min(
                self.invalid_index[None], (I.x * self.ny + I.y) * self.nz + I.z
            )

    @ti.kernel
    def _scan_stability(self):
        self.invalid_index[None] = self.nx * self.ny * self.nz
        self.maximum_speed[None] = 0.0
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                self._scan_stability_at(
                    I, self.water_phase[I], self.pressure_lattice[I]
                )
        self.peak_speed[None] = ti.max(self.peak_speed[None], self.maximum_speed[None])

    def check_stability(self, scanned=False):
        if not scanned:
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
            self._forces(True, moment_pressure=True)
            self._collide_momentum()
            self.hydrodynamic_force.fill(0)
            self.hydrodynamic_torque.fill(0)
            self._stream_home(classified=True)
            # AA populations, HOME pressure and the stability reduction share
            # a single grid traversal. Geometry has not changed yet.
            self._recover_phase(recover_flow=True)
            self._gradients(classified=True)
            # Check before changed geometry can hide a divergent fluid node.
            speed_bound = None
            if self.config.check_stability:
                speed_bound = self.check_stability(scanned=True)
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
            "water_volume_m3": float(volume.sum(dtype=float)),
            "water_volume_change_ratio": (
                float(volume.sum(dtype=float) / self.initial_water_volume_m3 - 1)
                if self.initial_water_volume_m3 > 0
                else None
            ),
            "water_energy_j": float(self.water_energy.to_numpy().sum(dtype=float)),
            "ice_sensible_energy_j": float(
                self.material_energy.to_numpy().sum(dtype=float)
            ),
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
