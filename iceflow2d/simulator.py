"""2D air--water LBM coupled to a sharp rigid ice body and optional heat.

The two distribution functions use pressure--momentum and conservative phase
central-moment collision operators.  An oriented-box signed-distance field
excludes ice nodes from the fluid domain, and cut lattice links enforce an
impermeable rigid boundary.  Link momentum exchange drives the freely moving
ice body or reports the load on a fixed one.

When ``IceFlowConfig.thermal`` is present, a Taichi finite-volume enthalpy
component advances temperature and phase change on the same CUDA lattice.
The first coupled model keeps the ice pose fixed while its phase-fraction
zero contour changes the sharp LBM boundary.
"""

import json
import math
import sys
import time
from pathlib import Path

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
from .thermal import EnthalpyFV2D, LatticeScales


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _TerminalProgress:
    """Small dependency-free progress display, matching the existing demo."""

    def __init__(self, total, *, enabled, stream=None, width=24):
        self.total = max(0, int(total))
        self.enabled = bool(enabled) and self.total > 0
        self.stream = stream if stream is not None else sys.stderr
        self.width = max(1, int(width))
        self.completed = 0
        self._started_at = None
        self._last_line_length = 0
        self._closed = False

    def __enter__(self):
        if self.enabled:
            self._started_at = time.monotonic()
            self._render(self._started_at)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def advance(self):
        self.completed = min(self.total, self.completed + 1)
        if self.enabled:
            self._render(time.monotonic())

    def close(self):
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, now):
        started_at = self._started_at if self._started_at is not None else now
        elapsed = max(0.0, now - started_at)
        fraction = self.completed / self.total
        filled = min(self.width, int(fraction * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.completed / elapsed if self.completed > 0 and elapsed > 0 else 0.0
        eta = (self.total - self.completed) / rate if rate > 0 else None
        line = (
            f"Simulating rigid ice [{bar}] {fraction:6.1%} | "
            f"{self.completed}/{self.total} frames | elapsed {_format_duration(elapsed)} | "
            f"ETA {_format_duration(eta)}"
        )
        padding = " " * max(0, self._last_line_length - len(line))
        self.stream.write(f"\r{line}{padding}")
        self.stream.flush()
        self._last_line_length = len(line)


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
    """Air--water phase-field LBM with optional fixed-ice phase change."""

    def __init__(self, config):
        if not isinstance(config, IceFlowConfig):
            raise TypeError("config must be an IceFlowConfig")
        ensure_taichi_cuda()
        self.cfg = config
        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.frame = 0
        self.steps = 0
        self._use_unified_boundary = config.rigid_boundary_scheme == "unified"
        self._thermal_enabled = config.thermal is not None
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
        # Hydrostatic fields are absent from configurations that never read
        # them, so ordinary runs do not reserve their device memory.
        self.hydrostatic_reference_pressure = None
        if config.well_balanced_hydrostatics:
            self.hydrostatic_reference_pressure = ti.field(ti.f32, shape=shape_xy)
        self.hydrostatic_reference_density = None
        if config.well_balanced_hydrostatics:
            self.hydrostatic_reference_density = ti.field(ti.f32, shape=shape_xy)

        # Static container and dynamic sharp ice geometry.
        self.wall = ti.field(ti.i8, shape=shape_xy)
        self.solid = ti.field(ti.i8, shape=shape_xy)
        self.sdf = ti.field(ti.f32, shape=shape_xy)
        self.solid_prev = None
        if not config.ice_fixed or self._thermal_enabled:
            self.solid_prev = ti.field(ti.i8, shape=shape_xy)

        # Rigid state and the only load accumulator consumed by integration.
        self.body_center = ti.Vector.field(2, ti.f32, shape=())
        self.body_velocity = ti.Vector.field(2, ti.f32, shape=())
        self.body_angle = ti.field(ti.f32, shape=())
        self.body_angular_velocity = ti.field(ti.f32, shape=())
        self.hydrodynamic_impulse = None
        self.hydrodynamic_torque = None
        if not config.ice_fixed:
            # Cut-link loads are signed sums with strong cancellation.  CUDA
            # is free to order global atomics differently as block scheduling
            # changes, so accumulate these two scalars in f64 and round only
            # once when the rigid integrator consumes them.  This costs just
            # twelve extra bytes compared with f32 scalar storage.
            self.hydrodynamic_impulse = ti.Vector.field(2, ti.f64, shape=())
            self.hydrodynamic_torque = ti.field(ti.f64, shape=())
        # Water-volume constraint and interface-projection state.  It remains
        # constant in the mechanical model and becomes a density-aware target
        # when thermal phase change is enabled.
        self.water_volume_target = ti.field(ti.f64, shape=())
        self.water_volume_current = ti.field(ti.f64, shape=())
        self.water_projection_derivative = ti.field(ti.f64, shape=())

        self.thermal = None
        self.thermal_enthalpy = None
        self.temperature = None
        self.liquid_fraction = None
        self.phase_change_material = None
        self.thermal_advection_velocity = None
        self.thermal_max_velocity_l1 = None
        self.phase_change_initial_water_volume = None
        self.phase_change_initial_solid_volume = None
        self.phase_change_current_solid_volume = None
        self.phase_change_initial_geometry_volume = None
        self.phase_change_current_geometry_volume = None
        self._thermal_lbm_steps_pending = 0
        if self._thermal_enabled:
            self.thermal = EnthalpyFV2D(
                self.nx,
                self.ny,
                config.thermal,
                self.scales,
                density_water_kg_m3=config.rho_water,
                density_air_kg_m3=config.rho_air,
                density_ice_kg_m3=config.rho_ice,
            )
            self.thermal_enthalpy = self.thermal.enthalpy_j_m3
            self.temperature = self.thermal.temperature_c
            self.liquid_fraction = self.thermal.liquid_fraction
            self.phase_change_material = self.thermal.ice_material
            self.thermal_advection_velocity = ti.Vector.field(2, ti.f32, shape=shape_xy)
            self.thermal_max_velocity_l1 = ti.field(ti.f32, shape=())
            self.phase_change_initial_water_volume = ti.field(ti.f64, shape=())
            self.phase_change_initial_solid_volume = ti.field(ti.f64, shape=())
            self.phase_change_current_solid_volume = ti.field(ti.f64, shape=())
            self.phase_change_initial_geometry_volume = ti.field(ti.f64, shape=())
            self.phase_change_current_geometry_volume = ti.field(ti.f64, shape=())

        self._initialize_body()
        self._initialize_geometry()
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
        if self._thermal_enabled:
            self.thermal.initialize(self.phi, self.wall, self.solid)
            self._initialize_phase_change_reference()
        if config.well_balanced_hydrostatics:
            self._build_hydrostatic_reference()
            self._initialize_hydrostatic_equilibrium()
        else:
            self._update_pressure()

    def step(self, num_steps=1):
        for _ in range(int(num_steps)):
            self._collide_velocity()
            self._collide_phase()
            self._stream()
            self._update_streamed_macroscopic_fields()
            self._update_pressure()
            if not self.cfg.ice_fixed:
                self._integrate_rigid_ice()
                self._update_ice_geometry()
                self._refill_changed_nodes()
            if self._thermal_enabled:
                self._thermal_lbm_steps_pending += 1
                interval = self.cfg.thermal.update_interval_lbm_steps
                if self._thermal_lbm_steps_pending >= interval:
                    self._advance_thermal_coupling(self._thermal_lbm_steps_pending)
                    self._thermal_lbm_steps_pending = 0
            self._correct_water_volume()
            self.steps += 1

    @property
    def physical_time_s(self):
        return self.steps * self._time_step_s

    def _advance_thermal_coupling(self, lbm_steps):
        """Advance heat, synchronize the phase boundary, and exchange water."""

        if not self._thermal_enabled:
            raise RuntimeError("thermal coupling is disabled")
        count = int(lbm_steps)
        if count <= 0:
            raise ValueError("lbm_steps must be positive")
        self._prepare_thermal_advection_velocity()
        maximum_velocity_l1 = float(self.thermal_max_velocity_l1[None])
        self.thermal.advance(
            count * self._time_step_s,
            self.thermal_advection_velocity,
            self.phi,
            self.wall,
            self.solid,
            max_velocity_lattice_l1=maximum_velocity_l1,
        )
        self._update_phase_change_geometry()
        self._refill_phase_change_nodes()
        self._update_phase_change_water_target()

    def synchronize_thermal(self):
        """Advance a pending partial thermal interval to the LBM time."""

        if not self._thermal_enabled or self._thermal_lbm_steps_pending == 0:
            return
        self._advance_thermal_coupling(self._thermal_lbm_steps_pending)
        self._thermal_lbm_steps_pending = 0
        self._correct_water_volume()

    def run(self, frames, steps_per_frame, output_dir=None, *, show_progress=False):
        total_frames = int(frames)
        if total_frames < 0:
            raise ValueError("frames must be non-negative")
        if int(steps_per_frame) <= 0:
            raise ValueError("steps_per_frame must be positive")
        out = Path(output_dir or self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.write_metadata(out)
        gui = None
        if self.cfg.show_gui:
            gui = ti.GUI(
                "IceFlow2D rigid ice / two-phase flow",
                res=(self.nx, self.ny),
                show_gui=True,
            )
        with _TerminalProgress(total_frames, enabled=show_progress) as progress:
            for _ in range(total_frames):
                self.step(steps_per_frame)
                self.synchronize_thermal()
                self.save_frame(out / f"frame_{self.frame:05d}.png", gui=gui)
                if self.cfg.save_npz:
                    self.save_state_npz(out / f"state_{self.frame:05d}.npz")
                self.frame += 1
                progress.advance()
        self.write_metadata(out)

    def save_frame(self, path, gui=None):
        phase = np.clip(self.phi.to_numpy(), 0.0, 1.0)[..., np.newaxis]
        air = np.asarray([0.97, 0.98, 0.99], dtype=np.float32)
        water = np.asarray([0.06, 0.38, 0.88], dtype=np.float32)
        native = air * (1.0 - phase) + water * phase
        solid = self.solid.to_numpy() == 1
        if np.any(solid):
            center_y = float(self.body_center[None].y)
            local_height = (
                np.arange(self.ny, dtype=np.float32)[np.newaxis, :] - center_y
            ) / (2.0 * self._body_half_height + 1.0e-6)
            tint = np.clip(0.55 + 0.25 * local_height, 0.0, 1.0)[..., np.newaxis]
            ice = (
                np.asarray([0.62, 0.84, 0.96], dtype=np.float32) * (1.0 - tint)
                + np.asarray([0.92, 0.98, 1.0], dtype=np.float32) * tint
            )
            native[solid] = np.broadcast_to(ice, native.shape)[solid]
        native[self.wall.to_numpy() == 1] = np.asarray(
            [0.12, 0.13, 0.14], dtype=np.float32
        )
        file_image = np.flip(np.transpose(native, (1, 0, 2)), axis=0)
        file_image = np.clip(file_image * 255.0, 0.0, 255.0).astype(np.uint8)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, file_image)
        except Exception:
            import matplotlib.pyplot as plt

            plt.imsave(path, file_image)
        if gui is not None:
            # Taichi GUI expects its native (nx, ny, channels) layout.  File
            # output above deliberately uses the conventional (ny, nx, 3).
            gui.set_image(native)
            gui.show()

    def save_state_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        phi = self.phi.to_numpy()
        pressure = self.p.to_numpy()
        solid = self.solid.to_numpy()
        # Moving-solid cells use phi/p themselves as refill reservoirs.  They
        # are masked here because those values are not part of the fluid state.
        phi = np.where(solid == 1, 0.0, phi)
        pressure = np.where(solid == 1, 0.0, pressure)
        state = {
            "phi": phi,
            "rho": self._rho_air_l
            + (self._rho_water_l - self._rho_air_l) * np.clip(phi, 0.0, 1.0),
            "u": self.u.to_numpy(),
            "p": pressure,
            "solid": solid,
            "sdf": self.sdf.to_numpy(),
            "body_center": np.asarray(self.body_center[None]),
            "body_velocity": np.asarray(self.body_velocity[None]),
            "body_angle": float(self.body_angle[None]),
            "body_angular_velocity": float(self.body_angular_velocity[None]),
            "physical_time_s": float(self.physical_time_s),
        }
        if self._thermal_enabled:
            state.update(
                {
                    "thermal_enthalpy_j_m3": self.thermal_enthalpy.to_numpy(),
                    "temperature_c": self.temperature.to_numpy(),
                    "liquid_fraction": self.liquid_fraction.to_numpy(),
                    "phase_change_material": self.phase_change_material.to_numpy(),
                }
            )
        if self.hydrostatic_reference_density is not None:
            state["hydrostatic_reference_density"] = (
                self.hydrostatic_reference_density.to_numpy()
            )
        if self.hydrostatic_reference_pressure is not None:
            state["hydrostatic_reference_pressure"] = (
                self.hydrostatic_reference_pressure.to_numpy()
            )
        np.savez_compressed(path, **state)

    def write_metadata(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        thermal_results = None
        if self._thermal_enabled:
            initial_solid = float(self.phase_change_initial_solid_volume[None])
            current_solid = self.phase_change_solid_volume_cells()
            current_geometry = self.phase_change_geometry_volume_cells()
            melted_solid = initial_solid - current_solid
            density_ratio = float(self.cfg.rho_ice / self.cfg.rho_water)
            thermal_results = {
                "physical_time_s": float(self.physical_time_s),
                "thermal_time_s": float(self.thermal.time_s),
                "thermal_substeps": int(self.thermal.steps),
                "initial_solid_volume_cells": initial_solid,
                "current_solid_volume_cells": current_solid,
                "initial_sharp_geometry_volume_cells": float(
                    self.phase_change_initial_geometry_volume[None]
                ),
                "current_sharp_geometry_volume_cells": current_geometry,
                "melted_fraction": self.phase_change_melted_fraction,
                "generated_water_volume_cells": density_ratio * melted_solid,
                "total_liquid_water_volume_cells": float(
                    self.phase_change_initial_water_volume[None]
                )
                + density_ratio * melted_solid,
                "phase_change_volume_contraction_cells": (1.0 - density_ratio)
                * melted_solid,
                "active_lbm_water_volume_target_cells": float(
                    self.water_volume_target[None]
                ),
                "water_volume_current_cells": float(self.water_volume_current[None]),
                "total_enthalpy_j_m": self.thermal.total_enthalpy_j_m(self.wall),
                "boundary_heat_input_j_m": float(
                    self.thermal.boundary_heat_input_j_m[None]
                ),
            }

        data = {
            "model": (
                "thermal fixed-ice/two-phase flow"
                if self._thermal_enabled
                else "non-thermal rigid-ice/two-phase flow"
            ),
            "hydrodynamic_population_moments": (
                "Liang P-rho*u: zeroth=0, first=rho(phi)*u, second=rho*u*u+p_dyn*I"
            ),
            "mechanics": (
                "fixed pose with phase-changing shape"
                if self._thermal_enabled
                else ("fixed rigid body" if self.cfg.ice_fixed else "moving rigid body")
            ),
            "coupling": (
                "liquid-fraction zero-contour cut links with density-aware mass exchange"
                if self._thermal_enabled
                else (
                    "sharp SDF cut-link bounce-back with pressure-completed GIMEM"
                    if self.cfg.well_balanced_hydrostatics
                    else "sharp SDF cut-link bounce-back with dynamic GIMEM"
                )
            ),
            "well_balanced_hydrostatics": bool(self.cfg.well_balanced_hydrostatics),
            "thermal_model": (
                "conservative finite-volume enthalpy with LBM velocity advection"
                if self._thermal_enabled
                else None
            ),
            "phase_change": bool(self._thermal_enabled),
            "thermal_results": thermal_results,
            "backend": "cuda",
            "taichi_version": ".".join(map(str, ti.__version__)),
            "steps": self.steps,
            "frame": self.frame,
            "lattice_scaling": {
                "rho_water": self._rho_water_l,
                "rho_air": self._rho_air_l,
                "nu_water": self._nu_water_l,
                "nu_air": self._nu_air_l,
                "gravity": self._gravity_l,
                "sigma": self._sigma_l,
                "dx_m": self.scales.dx_m,
                "dt_s": self.scales.dt_s,
                "velocity_scale_m_s": self.scales.velocity_scale_m_s,
                "reference_lattice_velocity": (
                    self.scales.reference_lattice_velocity
                ),
                "reference_velocity_m_s": self.scales.reference_velocity_m_s,
            },
            "config": self.cfg.to_dict(),
        }
        (out / "metadata.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Geometry and initialization

    @ti.func
    def _active(self, i, j):
        return self.wall[i, j] == 0 and self.solid[i, j] == 0

    @ti.kernel
    def _prepare_thermal_advection_velocity(self):
        """Store the same half-force velocity used by the LBM equilibria."""

        self.thermal_max_velocity_l1[None] = 0.0
        for i, j in self.u:
            velocity = ti.Vector([0.0, 0.0])
            if self._active(i, j):
                velocity = self.u[i, j] + 0.5 * self.fluid_force[i, j]
            self.thermal_advection_velocity[i, j] = velocity
            ti.atomic_max(
                self.thermal_max_velocity_l1[None],
                ti.abs(velocity.x) + ti.abs(velocity.y),
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
        if ti.static(not self.cfg.ice_fixed):
            self.hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
            self.hydrodynamic_torque[None] = 0.0

    @ti.kernel
    def _initialize_geometry(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        boundary = ti.static(self.cfg.boundary_cells)
        for i, j in self.wall:
            is_wall = (
                i < boundary or i >= nx - boundary or j < boundary or j >= ny - boundary
            )
            self.wall[i, j] = ti.cast(1 if is_wall else 0, ti.i8)
            distance = self._box_sdf(self._cell_point(i, j))
            is_solid = not is_wall and distance <= 0.0
            self.sdf[i, j] = distance
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)
            if ti.static(not self.cfg.ice_fixed):
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
            if ti.static(not self.cfg.ice_fixed):
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
    def _update_ice_geometry(self):
        for i, j in self.solid:
            self.solid_prev[i, j] = self.solid[i, j]
            distance = self._box_sdf(self._cell_point(i, j))
            is_solid = self.wall[i, j] == 0 and distance <= 0.0
            self.sdf[i, j] = distance
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)

    @ti.kernel
    def _update_phase_change_geometry(self):
        """Use the thermal liquid-fraction zero contour as the cut-link field.

        ``sdf`` only needs a signed, linearly interpolable zero crossing for
        the fixed-body cut-link formula.  ``lambda-threshold`` supplies that
        crossing directly and avoids converting the diffuse thermal interface
        to a stair-step rectangle.
        """

        threshold = ti.static(float(self.cfg.thermal.solid_liquid_threshold))
        for i, j in self.solid:
            self.solid_prev[i, j] = self.solid[i, j]
            indicator = 0.5
            is_solid = False
            if self.wall[i, j] == 0 and self.phase_change_material[i, j] != 0:
                indicator = self.liquid_fraction[i, j] - threshold
                is_solid = indicator <= 0.0
            self.sdf[i, j] = indicator
            self.solid[i, j] = ti.cast(1 if is_solid else 0, ti.i8)

    @ti.kernel
    def _refill_phase_change_nodes(self):
        """Rebuild LBM state on melted nodes and absorb frozen-node momentum."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
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
                reservoir_pressure = self.p[i, j]
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    reservoir_pressure -= self.hydrostatic_reference_pressure[i, j]
                self.phi[i, j] = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                self.u[i, j] = self._body_velocity_at(self._cell_point(i, j))
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                # In well-balanced mode an inactive phase-change cell stores
                # only p_dyn, just like a covered moving-body reservoir.
                self.p[i, j] = reservoir_pressure
            elif became_fluid:
                pressure_sum = 0.0
                count = 0.0
                for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    ni = i + di
                    nj = j + dj
                    if (
                        (di != 0 or dj != 0)
                        and _inside(ni, nj, nx, ny)
                        and self.wall[ni, nj] == 0
                        and self._active(ni, nj)
                        and self.solid_prev[ni, nj] == 0
                    ):
                        neighbor_pressure = self.p[ni, nj]
                        if ti.static(self.cfg.well_balanced_hydrostatics):
                            neighbor_pressure -= self.hydrostatic_reference_pressure[
                                ni, nj
                            ]
                        pressure_sum += neighbor_pressure
                        count += 1.0
                # An inactive thermal cell carries a pressure reservoir.  It
                # is p_dyn in well-balanced mode and ordinary pressure
                # otherwise, so the same value initializes the populations.
                pressure0 = self.p[i, j]
                if count > 0.0:
                    pressure0 = pressure_sum / count
                velocity0 = self._body_velocity_at(self._cell_point(i, j))
                self.phi[i, j] = 1.0
                self.u[i, j] = velocity0
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                total_pressure0 = pressure0
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    total_pressure0 += self.hydrostatic_reference_pressure[i, j]
                self.p[i, j] = total_pressure0
                for q in range(Q):
                    self.f[i, j, q] = _pressure_eq(q, pressure0, rho_water, velocity0)
                    self.h[i, j, q] = _heq(q, 1.0, velocity0)

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
                reservoir_pressure = self.p[i, j]
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    reservoir_pressure -= self.hydrostatic_reference_pressure[i, j]
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
                        neighbor_pressure = self.p[ni, nj]
                        if ti.static(self.cfg.well_balanced_hydrostatics):
                            neighbor_pressure -= self.hydrostatic_reference_pressure[
                                ni, nj
                            ]
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
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    pressure0 += self.hydrostatic_reference_pressure[i, j]
                fresh_density = _rho_mix(phi0, rho_water, rho_air)
                self.phi[i, j] = phi0
                self.u[i, j] = velocity0
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                self.p[i, j] = pressure0
                dynamic_pressure = pressure0
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    dynamic_pressure -= self.hydrostatic_reference_pressure[i, j]
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
        gravity_force_density = density * gravity
        if ti.static(self.cfg.well_balanced_hydrostatics):
            reference_density = self.hydrostatic_reference_density[i, j]
            gravity_force_density = (density - reference_density) * gravity
        if ti.static(self._thermal_enabled):
            reference_temperature = ti.static(
                float(self.cfg.thermal.buoyancy_reference_temperature_c)
            )
            temperature = ti.cast(self.temperature[i, j], ti.f32)
            density_anomaly_ratio = 0.0
            if ti.static(self.cfg.thermal.water_buoyancy_model == "linear"):
                expansion = ti.static(
                    float(self.cfg.thermal.thermal_expansion_water_1_k)
                )
                density_anomaly_ratio = -expansion * (
                    temperature - reference_temperature
                )
            else:
                density_beta = ti.static(
                    float(
                        self.cfg.thermal.freshwater_density_quadratic_coefficient_1_k2
                    )
                )
                maximum_density_temperature = ti.static(
                    float(
                        self.cfg.thermal.freshwater_density_max_temperature_c
                    )
                )
                reference_offset = (
                    reference_temperature - maximum_density_temperature
                )
                local_offset = temperature - maximum_density_temperature
                reference_factor = ti.static(
                    1.0
                    - float(
                        self.cfg.thermal.freshwater_density_quadratic_coefficient_1_k2
                    )
                    * (
                        float(self.cfg.thermal.buoyancy_reference_temperature_c)
                        - float(
                            self.cfg.thermal.freshwater_density_max_temperature_c
                        )
                    )
                    ** 2
                )
                density_anomaly_ratio = density_beta * (
                    reference_offset * reference_offset
                    - local_offset * local_offset
                ) / reference_factor

            # The thermal field deliberately treats the water/air interface
            # as adiabatic and stores the air temperature on phi < 0.5.  A
            # smooth water-side gate prevents that unrelated air value from
            # producing an anomalous freshwater force in the diffuse layer.
            water_weight = ti.min(1.0, ti.max(0.0, 2.0 * bounded_phi - 1.0))
            gravity_force_density += (
                water_weight
                * rho_water
                * density_anomaly_ratio
                * gravity
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
                tau_local = 0.5 + 3.0 * nu_local
                # Liang et al. PRE 97, 033309 (2018), Eqs. (15)--(23).
                material_density = _rho_mix(phi, rho_water, rho_air)
                dynamic_pressure = self.p[i, j]
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    dynamic_pressure -= self.hydrostatic_reference_pressure[i, j]
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

                # MRT extension of Liang's BGK equation.  Hydrodynamic
                # shear modes retain omega=1/tau (and therefore the target
                # viscosity); bulk and ghost modes relax in one step.  The
                # trapezoidal source prefactor is applied per moment.
                omega = 1.0 / tau_local
                source_shear = 1.0 - 0.5 * omega
                post00 = m00 - omega * (m00 - meq00) + source_shear * src00
                post10 = m10 - omega * (m10 - meq10) + source_shear * src10
                post01 = m01 - omega * (m01 - meq01) + source_shear * src01
                trace_eq = meq20 + meq02
                trace_src = src20 + src02
                post_trace = trace_eq + 0.5 * trace_src
                difference = m20 - m02
                difference_eq = meq20 - meq02
                difference_src = src20 - src02
                post_difference = (
                    difference
                    - omega * (difference - difference_eq)
                    + source_shear * difference_src
                )
                post20 = 0.5 * (post_trace + post_difference)
                post02 = 0.5 * (post_trace - post_difference)
                post11 = m11 - omega * (m11 - meq11) + source_shear * src11
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
                        wall_correction = (
                            6.0 * _w(q) * density * direction_f.dot(boundary_velocity)
                        )
                        reflected_f = outgoing_f - wall_correction
                        if ti.static(self._use_unified_boundary):
                            back_i = i - direction.x
                            back_j = j - direction.y
                            if _inside(back_i, back_j, nx, ny) and self._active(
                                back_i, back_j
                            ):
                                reflected_f = (
                                    eta * self.f_post[i, j, _opp(q)]
                                    + (1.0 - eta) * self.f_post[back_i, back_j, q]
                                    + eta * outgoing_f
                                    - wall_correction
                                ) / (1.0 + eta)
                        opposite = _opp(q)
                        self.f[i, j, opposite] = reflected_f
                        self.h[i, j, opposite] = outgoing_h - (
                            6.0
                            * _w(q)
                            * self.phi[i, j]
                            * direction_f.dot(boundary_velocity)
                        )

                        if ti.static(not self.cfg.ice_fixed):
                            opposite_direction = -direction_f
                            impulse = (direction_f - boundary_velocity) * outgoing_f - (
                                opposite_direction - boundary_velocity
                            ) * reflected_f
                            if ti.static(self.cfg.well_balanced_hydrostatics):
                                hydrostatic_pressure = (
                                    1.0 - eta
                                ) * self.hydrostatic_reference_pressure[
                                    i, j
                                ] + eta * self.hydrostatic_reference_pressure[ni, nj]
                                impulse += (
                                    6.0 * _w(q) * hydrostatic_pressure * direction_f
                                )
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
                if ti.static(not self.cfg.ice_fixed):
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
                if ti.static(self.cfg.ice_fixed):
                    self.phi[i, j] = 0.0
                elif self.wall[i, j] == 1:
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
                if ti.static(self.cfg.ice_fixed):
                    self.phi[i, j] = 0.0
                elif self.wall[i, j] == 1:
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
                if ti.static(self.cfg.ice_fixed):
                    self.phi[i, j] = 0.0
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
                reference_pressure = 0.0
                if ti.static(self.cfg.well_balanced_hydrostatics):
                    reference_pressure = self.hydrostatic_reference_pressure[i, j]
                self.p[i, j] = reference_pressure + dynamic_pressure
            else:
                if ti.static(self.cfg.ice_fixed):
                    if ti.static(self._thermal_enabled):
                        # Preserve the dynamic-pressure reservoir stored when
                        # a phase-change node freezes.  Static container walls
                        # do not need such a reservoir.
                        if self.wall[i, j] == 1:
                            self.p[i, j] = 0.0
                    else:
                        self.p[i, j] = 0.0
                elif self.wall[i, j] == 1:
                    self.p[i, j] = 0.0

    # Rigid-body forcing and integration

    @ti.kernel
    def _integrate_rigid_ice(self):
        mass = ti.static(self._body_mass)
        inertia = ti.static(self._body_inertia)
        gravity = ti.Vector(
            [ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])]
        )
        total_impulse = (
            ti.cast(self.hydrodynamic_impulse[None], ti.f32) + mass * gravity
        )
        total_torque = ti.cast(self.hydrodynamic_torque[None], ti.f32)

        velocity = ti.static(self.cfg.linear_damping) * (
            self.body_velocity[None] + total_impulse / mass
        )
        speed = velocity.norm()
        max_speed = ti.static(self.cfg.max_ice_speed)
        if speed > max_speed:
            velocity *= max_speed / speed
        omega = ti.static(self.cfg.angular_damping) * (
            self.body_angular_velocity[None] + total_torque / inertia
        )
        max_omega = ti.static(self.cfg.max_ice_angular_speed)
        omega = ti.min(max_omega, ti.max(-max_omega, omega))

        # Bottom contact is an impulse constraint, not component-wise velocity
        # damping.  Solve the four rectangle corners as a projected
        # Gauss--Seidel contact manifold.  Each accumulated tangential impulse
        # is projected onto the Coulomb cone |J_t| <= mu J_n, so weak forcing
        # can stick while stronger dam-break forcing produces sliding.
        old_center = self.body_center[None]
        old_angle = self.body_angle[None]
        cosine_old = ti.cos(old_angle)
        sine_old = ti.sin(old_angle)
        floor_y = ti.static(float(self.cfg.boundary_cells))
        inv_mass = 1.0 / mass
        inv_inertia = 1.0 / inertia
        contact_slop = ti.static(1.0e-3)
        restitution_threshold = ti.static(1.0e-4)
        bottom_friction = ti.static(self.cfg.bottom_wall_friction)
        lever_x = ti.Vector([0.0, 0.0, 0.0, 0.0])
        lever_y = ti.Vector([0.0, 0.0, 0.0, 0.0])
        normal_target = ti.Vector([0.0, 0.0, 0.0, 0.0])
        normal_impulse = ti.Vector([0.0, 0.0, 0.0, 0.0])
        tangent_impulse = ti.Vector([0.0, 0.0, 0.0, 0.0])
        active_contact = ti.Vector([0, 0, 0, 0])
        for corner in ti.static(range(4)):
            local_x = ti.static(self._body_half_width) * (
                -1.0 if ti.static(corner % 2 == 0) else 1.0
            )
            local_y = ti.static(self._body_half_height) * (
                -1.0 if ti.static(corner < 2) else 1.0
            )
            relative = ti.Vector(
                [
                    cosine_old * local_x - sine_old * local_y,
                    sine_old * local_x + cosine_old * local_y,
                ]
            )
            lever_x[corner] = relative.x
            lever_y[corner] = relative.y
            gap = old_center.y + relative.y - floor_y
            normal_velocity = velocity.y + omega * relative.x
            if gap <= contact_slop or gap + ti.min(normal_velocity, 0.0) <= 0.0:
                active_contact[corner] = 1
                if normal_velocity < -restitution_threshold:
                    normal_target[corner] = (
                        -ti.static(self.cfg.wall_restitution) * normal_velocity
                    )

        # A handful of iterations is ample for this four-contact scalar LCP
        # and avoids choosing an arbitrary corner for an initially flat base.
        for _ in ti.static(range(12)):
            for corner in ti.static(range(4)):
                if active_contact[corner] == 1:
                    rx = lever_x[corner]
                    normal_velocity = velocity.y + omega * rx
                    effective_inverse_mass = inv_mass + rx * rx * inv_inertia
                    delta_impulse = (
                        normal_target[corner] - normal_velocity
                    ) / effective_inverse_mass
                    old_impulse = normal_impulse[corner]
                    new_impulse = ti.max(0.0, old_impulse + delta_impulse)
                    applied_impulse = new_impulse - old_impulse
                    normal_impulse[corner] = new_impulse
                    velocity.y += applied_impulse * inv_mass
                    omega += rx * applied_impulse * inv_inertia

                    # Tangent t=(1, 0): v_t=v_x-omega*r_y and
                    # r x (J_t t)=-r_y J_t.  Accumulated projection gives a
                    # single-coefficient Coulomb stick/slip model.
                    ry = lever_y[corner]
                    tangent_velocity = velocity.x - omega * ry
                    tangent_inverse_mass = inv_mass + ry * ry * inv_inertia
                    delta_tangent_impulse = -tangent_velocity / tangent_inverse_mass
                    old_tangent_impulse = tangent_impulse[corner]
                    friction_limit = bottom_friction * new_impulse
                    new_tangent_impulse = ti.min(
                        friction_limit,
                        ti.max(
                            -friction_limit, old_tangent_impulse + delta_tangent_impulse
                        ),
                    )
                    applied_tangent_impulse = new_tangent_impulse - old_tangent_impulse
                    tangent_impulse[corner] = new_tangent_impulse
                    velocity.x += applied_tangent_impulse * inv_mass
                    omega -= ry * applied_tangent_impulse * inv_inertia

        # Do not clip omega after the constraint solve: doing so would destroy
        # the just-enforced contact velocity and break angular-impulse balance.
        angle = self.body_angle[None] + omega
        center = self.body_center[None] + velocity

        # Split position impulses remove any residual corner penetration
        # without changing physical velocity or injecting kinetic energy.  In
        # contrast to an AABB snap, a one-corner correction changes both the
        # centre height and angle according to the same generalized inverse
        # mass used by the velocity constraint.
        position_slop = ti.static(1.0e-5)
        position_beta = ti.static(0.8)
        for _ in ti.static(range(12)):
            for corner in ti.static(range(4)):
                cosine_position = ti.cos(angle)
                sine_position = ti.sin(angle)
                local_x = ti.static(self._body_half_width) * (
                    -1.0 if ti.static(corner % 2 == 0) else 1.0
                )
                local_y = ti.static(self._body_half_height) * (
                    -1.0 if ti.static(corner < 2) else 1.0
                )
                relative = ti.Vector(
                    [
                        cosine_position * local_x - sine_position * local_y,
                        sine_position * local_x + cosine_position * local_y,
                    ]
                )
                penetration = floor_y - (center.y + relative.y)
                if penetration > position_slop:
                    effective_inverse_mass = (
                        inv_mass + relative.x * relative.x * inv_inertia
                    )
                    split_impulse = (
                        position_beta
                        * (penetration - position_slop)
                        / effective_inverse_mass
                    )
                    center.y += split_impulse * inv_mass
                    angle += relative.x * split_impulse * inv_inertia

        cosine = ti.abs(ti.cos(angle))
        sine = ti.abs(ti.sin(angle))
        extent_x = cosine * ti.static(self._body_half_width) + sine * ti.static(
            self._body_half_height
        )
        extent_y = sine * ti.static(self._body_half_width) + cosine * ti.static(
            self._body_half_height
        )
        lower_x = ti.static(float(self.cfg.boundary_cells)) + extent_x
        upper_x = ti.static(float(self.nx - self.cfg.boundary_cells)) - extent_x
        upper_y = ti.static(float(self.ny - self.cfg.boundary_cells)) - extent_y
        restitution = ti.static(self.cfg.wall_restitution)
        tangential = 1.0 - ti.static(self.cfg.wall_friction)
        if center.x < lower_x:
            center.x = lower_x
            if velocity.x < 0.0:
                velocity.x = -restitution * velocity.x
                velocity.y *= tangential
                omega *= tangential
        elif center.x > upper_x:
            center.x = upper_x
            if velocity.x > 0.0:
                velocity.x = -restitution * velocity.x
                velocity.y *= tangential
                omega *= tangential
        if center.y > upper_y:
            center.y = upper_y
            if velocity.y > 0.0:
                velocity.y = -restitution * velocity.y
                velocity.x *= tangential
                omega *= tangential
        self.body_center[None] = center
        self.body_velocity[None] = velocity
        self.body_angle[None] = angle
        self.body_angular_velocity[None] = omega
        # This kernel is the sole consumer of the accumulated cut-link load.
        # Consume-and-clear ownership leaves the following stream kernel with
        # one job: add the next step's locally reduced boundary loads.
        self.hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.hydrodynamic_torque[None] = 0.0

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
        for i, j in self.liquid_fraction:
            if self.phase_change_material[i, j] != 0:
                fraction = ti.min(1.0, ti.max(0.0, self.liquid_fraction[i, j]))
                ti.atomic_add(
                    self.phase_change_current_solid_volume[None],
                    1.0 - ti.cast(fraction, ti.f64),
                )

    @ti.kernel
    def _reduce_phase_change_geometry_volume(self):
        self.phase_change_current_geometry_volume[None] = 0.0
        for i, j in self.solid:
            if self.solid[i, j] != 0:
                ti.atomic_add(self.phase_change_current_geometry_volume[None], 1.0)

    def _initialize_phase_change_reference(self):
        self._reduce_phase_change_solid_volume()
        self._reduce_phase_change_geometry_volume()
        self.phase_change_initial_water_volume[None] = float(
            self.water_volume_target[None]
        )
        initial_solid = float(self.phase_change_current_solid_volume[None])
        self.phase_change_initial_solid_volume[None] = initial_solid
        self.phase_change_initial_geometry_volume[None] = float(
            self.phase_change_current_geometry_volume[None]
        )

    @ti.kernel
    def _apply_phase_change_water_target(self):
        density_ratio = ti.static(float(self.cfg.rho_ice / self.cfg.rho_water))
        solid_volume_change = (
            self.phase_change_current_solid_volume[None]
            - self.phase_change_initial_solid_volume[None]
        )
        sharp_geometry_change = (
            self.phase_change_current_geometry_volume[None]
            - self.phase_change_initial_geometry_volume[None]
        )
        # water_volume_target counts only active LBM water.  The continuous
        # thermodynamic volume change supplies the physical expansion or
        # contraction, while the sharp-mask term exactly compensates every
        # whole node added to or removed from the active fluid lattice.
        self.water_volume_target[None] = (
            self.phase_change_initial_water_volume[None]
            + (1.0 - density_ratio) * solid_volume_change
            - sharp_geometry_change
        )

    def _update_phase_change_water_target(self):
        """Apply exact ice-mass/water-volume conversion for the thermal state."""

        self._reduce_phase_change_solid_volume()
        self._reduce_phase_change_geometry_volume()
        self._apply_phase_change_water_target()

    def phase_change_solid_volume_cells(self):
        if not self._thermal_enabled:
            raise RuntimeError("thermal coupling is disabled")
        self._reduce_phase_change_solid_volume()
        return float(self.phase_change_current_solid_volume[None])

    def phase_change_geometry_volume_cells(self):
        if not self._thermal_enabled:
            raise RuntimeError("thermal coupling is disabled")
        self._reduce_phase_change_geometry_volume()
        return float(self.phase_change_current_geometry_volume[None])

    @property
    def phase_change_melted_fraction(self):
        if not self._thermal_enabled:
            return 0.0
        initial = float(self.phase_change_initial_solid_volume[None])
        if initial <= 0.0:
            return 0.0
        return 1.0 - self.phase_change_solid_volume_cells() / initial

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
                if bounded_phase < cutoff:
                    bounded_phase = 0.0
                elif bounded_phase > 1.0 - cutoff:
                    bounded_phase = 1.0
                if bounded_phase != old_phase:
                    velocity = self.u[i, j]
                    dynamic_pressure = self.p[i, j]
                    if ti.static(self.cfg.well_balanced_hydrostatics):
                        dynamic_pressure -= self.hydrostatic_reference_pressure[i, j]
                    old_density = _rho_mix(old_phase, rho_water, rho_air)
                    new_density = _rho_mix(bounded_phase, rho_water, rho_air)
                    for q in range(Q):
                        self.f[i, j, q] += _pressure_eq(
                            q, dynamic_pressure, new_density, velocity
                        ) - _pressure_eq(q, dynamic_pressure, old_density, velocity)
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = (
                            self.h[i, j, q]
                            + _heq(q, bounded_phase, velocity)
                            - _heq(q, old_phase, velocity)
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    # Close the f32 zeroth moment without changing any bulk
                    # cell that was already exactly zero or one.
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
        for i, j in self.phi:
            if self._active(i, j):
                phase64 = ti.cast(self.phi[i, j], ti.f64)
                mapped64 = phase64
                if self.h_post[i, j, 0] == 1.0:
                    mapped64 = (
                        phase64 * exponential / (1.0 - phase64 + phase64 * exponential)
                    )
                mapped32 = ti.cast(mapped64, ti.f32)
                mapped64 = ti.cast(mapped32, ti.f64)
                ti.atomic_add(self.water_volume_current[None], mapped64)
                if self.h_post[i, j, 0] == 1.0:
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
                if new_phase != old_phase:
                    velocity = self.u[i, j]
                    dynamic_pressure = self.p[i, j]
                    if ti.static(self.cfg.well_balanced_hydrostatics):
                        dynamic_pressure -= self.hydrostatic_reference_pressure[i, j]
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
        converged = False
        iterations = 0
        for iterations in range(1, int(self.cfg.volume_projection_max_iterations) + 1):
            self._evaluate_water_projection(lagrange_multiplier)
            mass = float(self.water_volume_current[None])
            derivative = float(self.water_projection_derivative[None])
            residual = mass - target
            if abs(residual) < best_residual:
                best_residual = abs(residual)
                best_lambda = lagrange_multiplier
            if abs(residual) <= tolerance:
                best_lambda = lagrange_multiplier
                converged = True
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

        if not converged:
            # With f32 phase storage the volume map is piecewise constant.
            # Accept the best representable state only when it satisfies the
            # declared conservation tolerance.
            self._evaluate_water_projection(best_lambda)
            best_residual = abs(float(self.water_volume_current[None]) - target)
            if best_residual > tolerance:
                raise RuntimeError(
                    "water-volume projection did not converge to storage precision: "
                    f"residual={best_residual:.6e}, tolerance={tolerance:.6e}"
                )

        self._apply_water_projection(best_lambda)
        final_error = abs(target - float(self.water_volume_current[None]))
        if final_error > tolerance:
            raise RuntimeError(
                "water-volume projection application disagrees with its f64 reduction: "
                f"residual={final_error:.6e}, tolerance={tolerance:.6e}"
            )
