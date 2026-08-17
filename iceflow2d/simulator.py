"""Non-thermal 2D air--water flow coupled to one sharp rigid ice body.

The two distribution functions and their central-moment collision operators
are a deliberately small fork of the pure-fluid path in ``mixture2d``.  The
porous LBM--MPM terms are not used.  Instead, an oriented-box signed-distance
field excludes ice nodes from the fluid domain and cut lattice links enforce
an impermeable moving boundary.  Link momentum exchange drives the freely
translating and rotating ice body.

This module intentionally contains no temperature, enthalpy, melting, or
freezing state.  It is the mechanical baseline on which conservative phase
change can be added later.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import taichi as ti

from .config import IceFlowConfig
from .lattice import (
    Q,
    _axis_c,
    _c,
    _cross2,
    _feq,
    _heq,
    _inside,
    _opp,
    _reconstruct_central,
    _rho_mix,
    _tau_mix,
    _viscosity_mix,
    _w,
)


_TAICHI_INITIALIZED = False


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

    def __init__(self, total, *, label, enabled, stream=None, width=24):
        self.total = max(0, int(total))
        self.label = str(label)
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

    global _TAICHI_INITIALIZED
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        try:
            ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
        except Exception as exc:  # pragma: no cover - depends on host GPU
            raise RuntimeError("IceFlow2D requires Taichi CUDA; no CPU fallback is provided") from exc
    if ti.lang.impl.current_cfg().arch != ti.cuda:
        raise RuntimeError("IceFlow2D requires an existing or new Taichi CUDA runtime")
    _TAICHI_INITIALIZED = True


@ti.data_oriented
class IceFlow2D:
    """Air--water phase-field LBM with a single sharp rigid ice rectangle."""

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

        # Same dimensional-to-lattice conversion as mixture2d Sec. 5.1.
        g_ref = abs(float(config.gravity[1])) if abs(float(config.gravity[1])) > 0.0 else 9.8
        u_ref = math.sqrt(g_ref * config.reference_length_cells * config.dx * 4.0)
        u_ref_lattice = 0.1
        c_u = u_ref / u_ref_lattice
        self._rho_water_l = 1.0
        self._rho_air_l = float(config.rho_air / config.rho_water)
        self._rho_ice_l = float(config.rho_ice / config.rho_water)
        self._nu_water_l = float(config.viscosity_water * u_ref_lattice / (config.dx * u_ref))
        self._nu_air_l = float(config.viscosity_air * u_ref_lattice / (config.dx * u_ref))
        self._gravity_l = (
            float(config.gravity[0] * config.dx / (c_u * c_u)),
            float(config.gravity[1] * config.dx / (c_u * c_u)),
        )
        self._sigma_l = float(
            config.sigma * u_ref_lattice * u_ref_lattice / (u_ref * u_ref) / config.dx / config.rho_water
        )
        self._body_half_width = 0.5 * float(config.ice_width)
        self._body_half_height = 0.5 * float(config.ice_height)
        self._body_mass = float(config.ice_mass_lattice)
        self._body_inertia = float(config.ice_inertia_lattice)
        cutoff = float(config.volume_projection_interface_cutoff)
        profile_radius = 0.25 * float(config.interface_width) * math.log((1.0 - cutoff) / cutoff)
        # Ping-pong dilation finishes in the primary mask after an even
        # number of passes.  The radius covers the equilibrium logistic
        # profile down to the configured bulk cutoff.
        self._volume_projection_band_radius = max(2, int(math.ceil(profile_radius)))
        if self._volume_projection_band_radius % 2:
            self._volume_projection_band_radius += 1

        shape_q = (self.nx, self.ny, Q)
        shape_xy = (self.nx, self.ny)

        # Pure two-phase LBM state.  f is the unit-density velocity/pressure
        # distribution and h is the conservative Allen--Cahn phase field.
        self.f = ti.field(ti.f32, shape=shape_q)
        self.f_post = ti.field(ti.f32, shape=shape_q)
        self.f_next = ti.field(ti.f32, shape=shape_q)
        self.h = ti.field(ti.f32, shape=shape_q)
        self.h_post = ti.field(ti.f32, shape=shape_q)
        self.h_next = ti.field(ti.f32, shape=shape_q)
        self.rho = ti.field(ti.f32, shape=shape_xy)
        self.phi = ti.field(ti.f32, shape=shape_xy)
        self.u = ti.Vector.field(2, ti.f32, shape=shape_xy)
        self.u_temp = ti.Vector.field(2, ti.f32, shape=shape_xy)
        self.fluid_force = ti.Vector.field(2, ti.f32, shape=shape_xy)
        self.p = ti.field(ti.f32, shape=shape_xy)
        self.p_temp = ti.field(ti.f32, shape=shape_xy)
        self.artificial_vis = ti.field(ti.f32, shape=shape_xy)

        # Static container and dynamic sharp ice geometry.
        self.wall = ti.field(ti.i32, shape=shape_xy)
        self.solid = ti.field(ti.i32, shape=shape_xy)
        self.solid_prev = ti.field(ti.i32, shape=shape_xy)
        self.sdf = ti.field(ti.f32, shape=shape_xy)
        # Fractional geometry is deliberately kept separate from ``solid``.
        # The latter remains the cell-centre topology used by the node-based
        # LBM, whereas these fields resolve the area swept by the moving box.
        # They provide a geometric-conservation ledger without pretending
        # that the present streaming operator is already a cut-cell method.
        self.solid_fraction = ti.field(ti.f64, shape=shape_xy)
        self.solid_fraction_prev = ti.field(ti.f64, shape=shape_xy)
        self.stored_phi = ti.field(ti.f32, shape=shape_xy)
        self.stored_pressure = ti.field(ti.f32, shape=shape_xy)

        # Rigid state and exact action--reaction link diagnostics.
        self.body_center = ti.Vector.field(2, ti.f32, shape=())
        self.body_velocity = ti.Vector.field(2, ti.f32, shape=())
        self.body_angle = ti.field(ti.f32, shape=())
        self.body_angular_velocity = ti.field(ti.f32, shape=())
        self.raw_hydrodynamic_impulse = ti.Vector.field(2, ti.f32, shape=())
        self.raw_hydrodynamic_torque = ti.field(ti.f32, shape=())
        self.filtered_hydrodynamic_impulse = ti.Vector.field(2, ti.f32, shape=())
        self.filtered_hydrodynamic_torque = ti.field(ti.f32, shape=())
        self.buoyancy_impulse = ti.Vector.field(2, ti.f32, shape=())
        self.buoyancy_torque = ti.field(ti.f32, shape=())
        self.total_body_impulse = ti.Vector.field(2, ti.f32, shape=())
        self.total_body_torque = ti.field(ti.f32, shape=())
        self.bottom_contact_impulse = ti.field(ti.f32, shape=())
        self.bottom_contact_torque_impulse = ti.field(ti.f32, shape=())
        self.bottom_friction_impulse = ti.field(ti.f32, shape=())
        self.bottom_friction_abs_impulse = ti.field(ti.f32, shape=())
        self.bottom_friction_torque_impulse = ti.field(ti.f32, shape=())
        self.bottom_position_correction = ti.field(ti.f32, shape=())
        self.wall_contact = ti.field(ti.i32, shape=())
        self.cut_link_count = ti.field(ti.i32, shape=())
        self.solid_cell_count = ti.field(ti.i32, shape=())
        self.fresh_cell_count = ti.field(ti.i32, shape=())
        self.covered_cell_count = ti.field(ti.i32, shape=())
        self.displaced_water_volume = ti.field(ti.f64, shape=())

        # Fractional-occupancy/GCL diagnostics.  Each cut-cell fraction is the
        # exact OBB--unit-square polygon intersection (up to floating-point
        # roundoff), rather than a sampled/smoothed Heaviside.  The remap
        # ledger measures the geometry/refill substep in isolation, before a
        # later global phase projection can hide its defect.
        self.geometric_solid_area = ti.field(ti.f64, shape=())
        self.geometric_solid_area_error = ti.field(ti.f64, shape=())
        self.geometric_fluid_area = ti.field(ti.f64, shape=())
        self.geometric_gcl_residual = ti.field(ti.f64, shape=())
        self.geometric_water_before_remap = ti.field(ti.f64, shape=())
        self.geometric_water_after_remap = ti.field(ti.f64, shape=())
        self.geometric_water_remap_error = ti.field(ti.f64, shape=())
        self.geometric_swept_water_volume = ti.field(ti.f64, shape=())
        self.binary_covered_water = ti.field(ti.f64, shape=())
        self.binary_fresh_water = ti.field(ti.f64, shape=())
        self.binary_remap_error = ti.field(ti.f64, shape=())

        # No-melting phase-volume invariant and interface-projection state.
        # This ledger covers transport, moving-boundary, clipping, and binary
        # remap defects; it is intentionally independent of the fractional
        # geometry diagnostics above.
        self.water_volume_target = ti.field(ti.f64, shape=())
        self.water_volume_current = ti.field(ti.f64, shape=())
        self.water_volume_bounded = ti.field(ti.f64, shape=())
        self.water_volume_raw = ti.field(ti.f64, shape=())
        self.water_volume_error = ti.field(ti.f64, shape=())
        self.water_projection_mass = ti.field(ti.f64, shape=())
        self.water_projection_derivative = ti.field(ti.f64, shape=())
        self.water_projection_lambda = ti.field(ti.f64, shape=())
        self.water_projection_shift = ti.field(ti.f64, shape=())
        self.water_projection_feasible = ti.field(ti.i32, shape=())
        self.water_projection_interface_cells = ti.field(ti.i32, shape=())
        self.water_projection_iterations = ti.field(ti.i32, shape=())
        self.water_projection_candidate = ti.field(ti.i32, shape=shape_xy)
        self.water_projection_mask = ti.field(ti.i32, shape=shape_xy)
        self.water_projection_mask_next = ti.field(ti.i32, shape=shape_xy)

        self.image = ti.Vector.field(3, ti.f32, shape=shape_xy)

        self._initialize_body()
        self._initialize_geometry()
        self._update_fractional_geometry()
        self._initialize_fractional_geometry_history()
        self._initialize_fluid()
        # The no-melting invariant is the sharp initial water volume.  Phase
        # warm-up is a numerical preparation step and must not redefine it.
        self._initialize_water_volume()
        if config.phase_warmup_steps > 0:
            self._warm_start_phase(config.phase_warmup_steps)
        self._update_fluid_macro()
        self._correct_water_volume()
        self._initialize_geometric_water_ledger()
        self._update_pressure()
        self._average_pressure()

    def step(self, num_steps=1):
        for _ in range(int(num_steps)):
            self._compute_fluid_force(self.steps)
            self._collide_velocity()
            self._collide_phase()
            self._clear_stream_targets()
            self._stream_velocity()
            self._stream_phase()
            self._copy_lbm_next()
            self._update_fluid_macro()
            self._update_pressure()
            self._average_pressure()
            self._compute_hydrostatic_buoyancy()
            self._advance_rigid_ice()
            self._measure_geometry_water_before_remap()
            self._save_previous_geometry()
            self._rasterize_ice()
            self._update_fractional_geometry()
            self._measure_geometry_swept_water()
            self._refill_changed_nodes()
            self._measure_geometry_water_after_remap()
            self._update_fluid_macro()
            self._correct_water_volume()
            self._measure_water_volume()
            self.steps += 1

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
            gui = ti.GUI("IceFlow2D rigid ice / two-phase flow", res=(self.nx, self.ny), show_gui=True)
        with _TerminalProgress(total_frames, label="coupled ice", enabled=show_progress) as progress:
            for _ in range(total_frames):
                self.step(steps_per_frame)
                self.save_frame(out / f"frame_{self.frame:05d}.png", gui=gui)
                if self.cfg.save_npz:
                    self.save_state_npz(out / f"state_{self.frame:05d}.npz")
                self.frame += 1
                progress.advance()
        self.write_metadata(out)

    def save_frame(self, path, gui=None):
        self._render()
        native = self.image.to_numpy()
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
        np.savez_compressed(
            path,
            phi=self.phi.to_numpy(),
            rho=self.rho.to_numpy(),
            u=self.u.to_numpy(),
            p=self.p.to_numpy(),
            solid=self.solid.to_numpy(),
            solid_fraction=self.solid_fraction.to_numpy(),
            sdf=self.sdf.to_numpy(),
            body_center=np.asarray(self.body_center[None]),
            body_velocity=np.asarray(self.body_velocity[None]),
            body_angle=float(self.body_angle[None]),
            body_angular_velocity=float(self.body_angular_velocity[None]),
        )

    def write_metadata(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": "coupled",
            "material": "ice",
            "mechanics": "variable-pose rigid body (no MPM)",
            "coupling": "sharp SDF cut-link moving bounce-back with momentum exchange",
            "geometry_accounting": "exact OBB--unit-cell fractional occupancy with a GCL/remap ledger",
            "phase_control_volume": "binary active-node LBM (fractional occupancy is diagnostic only)",
            "thermal_model": None,
            "phase_change": False,
            "backend": "cuda",
            "taichi_version": ".".join(map(str, ti.__version__)),
            "steps": self.steps,
            "frame": self.frame,
            "lattice_scaling": {
                "rho_water": self._rho_water_l,
                "rho_air": self._rho_air_l,
                "rho_ice": self._rho_ice_l,
                "nu_water": self._nu_water_l,
                "nu_air": self._nu_air_l,
                "gravity": self._gravity_l,
                "sigma": self._sigma_l,
            },
            "diagnostics": self.diagnostics(),
            "config": self.cfg.to_dict(),
        }
        (out / "metadata.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def diagnostics(self):
        arrays = (
            self.f.to_numpy(),
            self.f_post.to_numpy(),
            self.f_next.to_numpy(),
            self.h.to_numpy(),
            self.h_post.to_numpy(),
            self.h_next.to_numpy(),
            self.phi.to_numpy(),
            self.rho.to_numpy(),
            self.u.to_numpy(),
            self.u_temp.to_numpy(),
            self.fluid_force.to_numpy(),
            self.p.to_numpy(),
            self.p_temp.to_numpy(),
            self.artificial_vis.to_numpy(),
            self.sdf.to_numpy(),
            self.solid_fraction.to_numpy(),
        )
        phi = arrays[6]
        velocity = arrays[8]
        center = np.asarray(self.body_center[None], dtype=np.float64)
        body_velocity = np.asarray(self.body_velocity[None], dtype=np.float64)
        angle = float(self.body_angle[None])
        omega = float(self.body_angular_velocity[None])
        scalar_values = np.asarray(
            [
                *center,
                *body_velocity,
                angle,
                omega,
                *np.asarray(self.raw_hydrodynamic_impulse[None]),
                float(self.raw_hydrodynamic_torque[None]),
                *np.asarray(self.filtered_hydrodynamic_impulse[None]),
                float(self.filtered_hydrodynamic_torque[None]),
                *np.asarray(self.buoyancy_impulse[None]),
                float(self.buoyancy_torque[None]),
                *np.asarray(self.total_body_impulse[None]),
                float(self.total_body_torque[None]),
                float(self.bottom_contact_impulse[None]),
                float(self.bottom_contact_torque_impulse[None]),
                float(self.bottom_friction_impulse[None]),
                float(self.bottom_friction_abs_impulse[None]),
                float(self.bottom_friction_torque_impulse[None]),
                float(self.bottom_position_correction[None]),
                float(self.water_volume_target[None]),
                float(self.water_volume_current[None]),
                float(self.water_volume_bounded[None]),
                float(self.water_volume_error[None]),
                float(self.water_projection_lambda[None]),
                float(self.water_projection_shift[None]),
                float(self.displaced_water_volume[None]),
                float(self.geometric_solid_area[None]),
                float(self.geometric_solid_area_error[None]),
                float(self.geometric_fluid_area[None]),
                float(self.geometric_gcl_residual[None]),
                float(self.geometric_water_before_remap[None]),
                float(self.geometric_water_after_remap[None]),
                float(self.geometric_water_remap_error[None]),
                float(self.geometric_swept_water_volume[None]),
                float(self.binary_covered_water[None]),
                float(self.binary_fresh_water[None]),
                float(self.binary_remap_error[None]),
            ],
            dtype=np.float64,
        )
        c = abs(math.cos(angle))
        s = abs(math.sin(angle))
        extent_x = c * self._body_half_width + s * self._body_half_height
        extent_y = s * self._body_half_width + c * self._body_half_height
        b = float(self.cfg.boundary_cells)
        body_inside = bool(
            center[0] - extent_x >= b - 1.0e-4
            and center[0] + extent_x <= self.nx - b + 1.0e-4
            and center[1] - extent_y >= b - 1.0e-4
            and center[1] + extent_y <= self.ny - b + 1.0e-4
        )
        active = (self.wall.to_numpy() == 0) & (self.solid.to_numpy() == 0)
        active_phi = phi[active]
        max_fluid_speed = float(np.nanmax(np.linalg.norm(velocity[active], axis=1))) if active.any() else 0.0
        return {
            "fluid_finite": bool(all(np.isfinite(values).all() for values in arrays)),
            "rigid_finite": bool(np.isfinite(scalar_values).all()),
            "body_inside": body_inside,
            "phi_min": float(np.nanmin(active_phi)) if active_phi.size else 0.0,
            "phi_max": float(np.nanmax(active_phi)) if active_phi.size else 0.0,
            "max_fluid_speed": max_fluid_speed,
            "body_center_x": float(center[0]),
            "body_center_y": float(center[1]),
            "body_velocity_x": float(body_velocity[0]),
            "body_velocity_y": float(body_velocity[1]),
            "body_angle": angle,
            "body_angular_velocity": omega,
            "body_mass": self._body_mass,
            "body_inertia": self._body_inertia,
            "solid_cells": int(self.solid_cell_count[None]),
            "cut_links": int(self.cut_link_count[None]),
            "fresh_cells": int(self.fresh_cell_count[None]),
            "covered_cells": int(self.covered_cell_count[None]),
            "wall_contact": bool(self.wall_contact[None]),
            "hydrodynamic_impulse_x": float(self.raw_hydrodynamic_impulse[None].x),
            "hydrodynamic_impulse_y": float(self.raw_hydrodynamic_impulse[None].y),
            "hydrodynamic_torque": float(self.raw_hydrodynamic_torque[None]),
            "filtered_hydrodynamic_torque": float(self.filtered_hydrodynamic_torque[None]),
            "buoyancy_impulse_x": float(self.buoyancy_impulse[None].x),
            "buoyancy_impulse_y": float(self.buoyancy_impulse[None].y),
            "buoyancy_torque": float(self.buoyancy_torque[None]),
            "total_body_impulse_x": float(self.total_body_impulse[None].x),
            "total_body_impulse_y": float(self.total_body_impulse[None].y),
            "total_body_torque": float(self.total_body_torque[None]),
            "bottom_contact_impulse": float(self.bottom_contact_impulse[None]),
            "bottom_contact_torque_impulse": float(self.bottom_contact_torque_impulse[None]),
            "bottom_friction_impulse": float(self.bottom_friction_impulse[None]),
            "bottom_friction_abs_impulse": float(self.bottom_friction_abs_impulse[None]),
            "bottom_friction_torque_impulse": float(self.bottom_friction_torque_impulse[None]),
            "bottom_position_correction": float(self.bottom_position_correction[None]),
            "displaced_water_volume": float(self.displaced_water_volume[None]),
            "water_volume_target": float(self.water_volume_target[None]),
            "water_volume_current": float(self.water_volume_current[None]),
            "water_volume_bounded": float(self.water_volume_bounded[None]),
            "water_volume_raw": float(self.water_volume_raw[None]),
            "water_volume_error": float(self.water_volume_error[None]),
            "water_projection_lambda": float(self.water_projection_lambda[None]),
            "water_projection_shift": float(self.water_projection_shift[None]),
            "water_projection_feasible": bool(self.water_projection_feasible[None]),
            "water_projection_interface_cells": int(self.water_projection_interface_cells[None]),
            "water_projection_iterations": int(self.water_projection_iterations[None]),
            "geometric_solid_area": float(self.geometric_solid_area[None]),
            "geometric_solid_area_error": float(self.geometric_solid_area_error[None]),
            "geometric_fluid_area": float(self.geometric_fluid_area[None]),
            "geometric_gcl_residual": float(self.geometric_gcl_residual[None]),
            "geometric_water_before_remap": float(self.geometric_water_before_remap[None]),
            "geometric_water_after_remap": float(self.geometric_water_after_remap[None]),
            "geometric_water_remap_error": float(self.geometric_water_remap_error[None]),
            "geometric_swept_water_volume": float(self.geometric_swept_water_volume[None]),
            "binary_covered_water": float(self.binary_covered_water[None]),
            "binary_fresh_water": float(self.binary_fresh_water[None]),
            "binary_remap_error": float(self.binary_remap_error[None]),
        }

    # ------------------------------------------------------------------
    # Geometry and initialization

    @ti.func
    def _active(self, i, j):
        return self.wall[i, j] == 0 and self.solid[i, j] == 0

    @ti.func
    def _box_sdf(self, point):
        center = self.body_center[None]
        angle = self.body_angle[None]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        relative = point - center
        local = ti.Vector([
            cosine * relative.x + sine * relative.y,
            -sine * relative.x + cosine * relative.y,
        ])
        delta = ti.Vector([
            ti.abs(local.x) - ti.static(self._body_half_width),
            ti.abs(local.y) - ti.static(self._body_half_height),
        ])
        outside = ti.Vector([ti.max(delta.x, 0.0), ti.max(delta.y, 0.0)])
        return outside.norm() + ti.min(ti.max(delta.x, delta.y), 0.0)

    @ti.func
    def _cell_solid_fraction_exact(self, i, j):
        """Area of the oriented ice rectangle inside one unit cell.

        Sutherland--Hodgman clips the cell, expressed in body coordinates,
        against the four half-planes of the rectangle.  The fixed eight-
        vertex buffers are sufficient for the intersection of two convex
        quadrilaterals.  Buffer reads/writes are selected with static loops so
        this remains compatible with Taichi runtimes that disable dynamic
        indexing of local matrices.
        """

        vertices = ti.Matrix.zero(ti.f64, 8, 2)
        scratch = ti.Matrix.zero(ti.f64, 8, 2)
        center = ti.cast(self.body_center[None], ti.f64)
        angle = ti.cast(self.body_angle[None], ti.f64)
        cosine = ti.cos(angle)
        sine = ti.sin(angle)

        # Counter-clockwise unit-cell corners transformed into body space.
        for corner in ti.static(range(4)):
            offset_x = ti.static((0.0, 1.0, 1.0, 0.0)[corner])
            offset_y = ti.static((0.0, 0.0, 1.0, 1.0)[corner])
            relative_x = ti.cast(i, ti.f64) + offset_x - center.x
            relative_y = ti.cast(j, ti.f64) + offset_y - center.y
            vertices[corner, 0] = cosine * relative_x + sine * relative_y
            vertices[corner, 1] = -sine * relative_x + cosine * relative_y

        vertex_count = 4
        half_width = ti.cast(ti.static(float(self._body_half_width)), ti.f64)
        half_height = ti.cast(ti.static(float(self._body_half_height)), ti.f64)
        for boundary in ti.static(range(4)):
            output_count = 0
            for edge in ti.static(range(8)):
                if edge < vertex_count:
                    start_x = vertices[edge, 0]
                    start_y = vertices[edge, 1]
                    next_index = edge + 1
                    if next_index >= vertex_count:
                        next_index = 0
                    end_x = vertices[0, 0]
                    end_y = vertices[0, 1]
                    for candidate in ti.static(range(8)):
                        if next_index == candidate:
                            end_x = vertices[candidate, 0]
                            end_y = vertices[candidate, 1]

                    start_distance = start_x + half_width
                    end_distance = end_x + half_width
                    if ti.static(boundary == 0):
                        start_distance = start_x + half_width
                        end_distance = end_x + half_width
                    elif ti.static(boundary == 1):
                        start_distance = half_width - start_x
                        end_distance = half_width - end_x
                    elif ti.static(boundary == 2):
                        start_distance = start_y + half_height
                        end_distance = end_y + half_height
                    else:
                        start_distance = half_height - start_y
                        end_distance = half_height - end_y

                    start_inside = start_distance >= 0.0
                    end_inside = end_distance >= 0.0
                    if start_inside != end_inside:
                        interpolation = start_distance / (start_distance - end_distance)
                        intersection_x = start_x + interpolation * (end_x - start_x)
                        intersection_y = start_y + interpolation * (end_y - start_y)
                        for slot in ti.static(range(8)):
                            if output_count == slot:
                                scratch[slot, 0] = intersection_x
                                scratch[slot, 1] = intersection_y
                        output_count += 1
                    if end_inside:
                        for slot in ti.static(range(8)):
                            if output_count == slot:
                                scratch[slot, 0] = end_x
                                scratch[slot, 1] = end_y
                        output_count += 1

            vertex_count = output_count
            for slot in ti.static(range(8)):
                vertices[slot, 0] = scratch[slot, 0]
                vertices[slot, 1] = scratch[slot, 1]

        twice_area = vertices[0, 0] * 0.0
        for edge in ti.static(range(8)):
            if edge < vertex_count:
                start_x = vertices[edge, 0]
                start_y = vertices[edge, 1]
                next_index = edge + 1
                if next_index >= vertex_count:
                    next_index = 0
                end_x = vertices[0, 0]
                end_y = vertices[0, 1]
                for candidate in ti.static(range(8)):
                    if next_index == candidate:
                        end_x = vertices[candidate, 0]
                        end_y = vertices[candidate, 1]
                twice_area += start_x * end_y - start_y * end_x
        return ti.min(1.0, ti.max(0.0, 0.5 * ti.abs(twice_area)))

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
        if _inside(ni1, nj1, ti.static(self.nx), ti.static(self.ny)) and self._active(ni1, nj1):
            value = self.phi[ni1, nj1]
            if distance == 2:
                ni2 = i + 2 * step_x
                nj2 = j + 2 * step_y
                if _inside(ni2, nj2, ti.static(self.nx), ti.static(self.ny)) and self._active(ni2, nj2):
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
        self.raw_hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.raw_hydrodynamic_torque[None] = 0.0
        self.filtered_hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.filtered_hydrodynamic_torque[None] = 0.0
        self.buoyancy_impulse[None] = ti.Vector([0.0, 0.0])
        self.buoyancy_torque[None] = 0.0
        self.total_body_impulse[None] = ti.Vector([0.0, 0.0])
        self.total_body_torque[None] = 0.0
        self.bottom_contact_impulse[None] = 0.0
        self.bottom_contact_torque_impulse[None] = 0.0
        self.bottom_friction_impulse[None] = 0.0
        self.bottom_friction_abs_impulse[None] = 0.0
        self.bottom_friction_torque_impulse[None] = 0.0
        self.bottom_position_correction[None] = 0.0
        self.wall_contact[None] = 0

    @ti.kernel
    def _initialize_geometry(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        boundary = ti.static(self.cfg.boundary_cells)
        self.solid_cell_count[None] = 0
        for i, j in self.wall:
            is_wall = i < boundary or i >= nx - boundary or j < boundary or j >= ny - boundary
            self.wall[i, j] = 1 if is_wall else 0
            distance = self._box_sdf(self._cell_point(i, j))
            is_solid = not is_wall and distance <= 0.0
            self.sdf[i, j] = distance
            self.solid[i, j] = 1 if is_solid else 0
            self.solid_prev[i, j] = self.solid[i, j]
            if is_solid:
                ti.atomic_add(self.solid_cell_count[None], 1)

    @ti.kernel
    def _compute_fractional_geometry(self):
        self.geometric_solid_area[None] = 0.0
        self.geometric_fluid_area[None] = 0.0
        self.geometric_gcl_residual[None] = 0.0
        half_diagonal = ti.cast(ti.static(math.sqrt(0.5)), ti.f64)
        for i, j in self.solid_fraction:
            fraction = ti.cast(0.0, ti.f64)
            if self.wall[i, j] == 0:
                # Signed distance is one-Lipschitz.  Cells farther than half
                # a diagonal from the interface can therefore bypass polygon
                # clipping; only the O(perimeter) cut band pays that cost.
                center_distance = ti.cast(self.sdf[i, j], ti.f64)
                if center_distance <= -half_diagonal:
                    fraction = 1.0
                elif center_distance < half_diagonal:
                    fraction = self._cell_solid_fraction_exact(i, j)
            self.solid_fraction[i, j] = fraction
            if self.wall[i, j] == 0:
                ti.atomic_add(self.geometric_solid_area[None], fraction)
                ti.atomic_add(self.geometric_fluid_area[None], 1.0 - fraction)
                ti.atomic_add(
                    self.geometric_gcl_residual[None],
                    self.solid_fraction_prev[i, j] - fraction,
                )
        self.geometric_solid_area_error[None] = self.geometric_solid_area[None] - ti.static(
            float(self.cfg.ice_width * self.cfg.ice_height)
        )

    def _update_fractional_geometry(self):
        """Refresh exact cut-cell occupancy after the rigid pose changes."""

        self._compute_fractional_geometry()

    @ti.kernel
    def _initialize_fractional_geometry_history(self):
        for i, j in self.solid_fraction:
            self.solid_fraction_prev[i, j] = self.solid_fraction[i, j]
        self.geometric_gcl_residual[None] = 0.0
        self.geometric_water_before_remap[None] = 0.0
        self.geometric_water_after_remap[None] = 0.0
        self.geometric_water_remap_error[None] = 0.0
        self.geometric_swept_water_volume[None] = 0.0
        self.binary_covered_water[None] = 0.0
        self.binary_fresh_water[None] = 0.0
        self.binary_remap_error[None] = 0.0

    @ti.kernel
    def _initialize_fluid(self):
        water_width = ti.static(self.cfg.water_width)
        water_height = ti.static(self.cfg.water_height)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            water_indicator = 1.0 if i < water_width and j < water_height else 0.0
            phi0 = water_indicator if self._active(i, j) else 0.0
            velocity = ti.Vector([0.0, 0.0])
            if self.solid[i, j] == 1:
                velocity = self._body_velocity_at(self._cell_point(i, j))
            self.stored_phi[i, j] = water_indicator
            self.stored_pressure[i, j] = 0.0
            self.phi[i, j] = phi0
            self.rho[i, j] = _rho_mix(phi0, rho_water, rho_air)
            self.u[i, j] = velocity
            self.u_temp[i, j] = velocity
            self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
            self.p[i, j] = 0.0
            self.p_temp[i, j] = 0.0
            self.artificial_vis[i, j] = 0.0
            for q in range(Q):
                self.f[i, j, q] = _feq(q, 1.0, velocity)
                self.f_post[i, j, q] = self.f[i, j, q]
                self.f_next[i, j, q] = self.f[i, j, q]
                self.h[i, j, q] = _heq(q, phi0, velocity)
                self.h_post[i, j, q] = self.h[i, j, q]
                self.h_next[i, j, q] = self.h[i, j, q]

    @ti.kernel
    def _save_previous_geometry(self):
        for i, j in self.solid:
            self.solid_prev[i, j] = self.solid[i, j]
            self.solid_fraction_prev[i, j] = self.solid_fraction[i, j]

    @ti.kernel
    def _measure_geometry_water_before_remap(self):
        self.geometric_water_before_remap[None] = 0.0
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                # ``stored_phi`` extends phase through centre-solid cut cells;
                # it is diagnostic state, not an LBM population reservoir.
                phase = self.phi[i, j]
                if self.solid[i, j] == 1:
                    phase = self.stored_phi[i, j]
                phase = ti.min(1.0, ti.max(0.0, phase))
                fluid_fraction = 1.0 - self.solid_fraction[i, j]
                ti.atomic_add(
                    self.geometric_water_before_remap[None],
                    fluid_fraction * ti.cast(phase, ti.f64),
                )

    @ti.kernel
    def _measure_geometry_swept_water(self):
        self.geometric_swept_water_volume[None] = 0.0
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                old_phase = self.phi[i, j]
                if self.solid_prev[i, j] == 1:
                    old_phase = self.stored_phi[i, j]
                old_phase = ti.min(1.0, ti.max(0.0, old_phase))
                # Positive means that rigid motion exposes fluid volume.
                swept_fluid_fraction = self.solid_fraction_prev[i, j] - self.solid_fraction[i, j]
                ti.atomic_add(
                    self.geometric_swept_water_volume[None],
                    swept_fluid_fraction * ti.cast(old_phase, ti.f64),
                )

    @ti.kernel
    def _measure_geometry_water_after_remap(self):
        self.geometric_water_after_remap[None] = 0.0
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                phase = self.phi[i, j]
                if self.solid[i, j] == 1:
                    phase = self.stored_phi[i, j]
                phase = ti.min(1.0, ti.max(0.0, phase))
                fluid_fraction = 1.0 - self.solid_fraction[i, j]
                ti.atomic_add(
                    self.geometric_water_after_remap[None],
                    fluid_fraction * ti.cast(phase, ti.f64),
                )
        self.geometric_water_remap_error[None] = (
            self.geometric_water_after_remap[None] - self.geometric_water_before_remap[None]
        )

    @ti.kernel
    def _initialize_geometric_water_ledger(self):
        self.geometric_water_before_remap[None] = 0.0
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                phase = self.phi[i, j]
                if self.solid[i, j] == 1:
                    phase = self.stored_phi[i, j]
                phase = ti.min(1.0, ti.max(0.0, phase))
                ti.atomic_add(
                    self.geometric_water_before_remap[None],
                    (1.0 - self.solid_fraction[i, j]) * ti.cast(phase, ti.f64),
                )
        self.geometric_water_after_remap[None] = self.geometric_water_before_remap[None]
        self.geometric_water_remap_error[None] = 0.0
        self.geometric_swept_water_volume[None] = 0.0

    @ti.kernel
    def _rasterize_ice(self):
        self.solid_cell_count[None] = 0
        for i, j in self.solid:
            distance = self._box_sdf(self._cell_point(i, j))
            is_solid = self.wall[i, j] == 0 and distance <= 0.0
            self.sdf[i, j] = distance
            self.solid[i, j] = 1 if is_solid else 0
            if is_solid:
                ti.atomic_add(self.solid_cell_count[None], 1)

    @ti.kernel
    def _refill_changed_nodes(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        self.fresh_cell_count[None] = 0
        self.covered_cell_count[None] = 0
        self.binary_covered_water[None] = 0.0
        self.binary_fresh_water[None] = 0.0
        for i, j in self.phi:
            became_solid = self.solid_prev[i, j] == 0 and self.solid[i, j] == 1 and self.wall[i, j] == 0
            became_fluid = self.solid_prev[i, j] == 1 and self.solid[i, j] == 0 and self.wall[i, j] == 0
            if became_solid:
                covered_phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
                ti.atomic_add(self.binary_covered_water[None], ti.cast(covered_phase, ti.f64))
                self.stored_phi[i, j] = covered_phase
                self.stored_pressure[i, j] = self.p_temp[i, j]
                velocity = self._body_velocity_at(self._cell_point(i, j))
                self.phi[i, j] = 0.0
                self.rho[i, j] = rho_air
                self.u[i, j] = velocity
                self.u_temp[i, j] = velocity
                self.p[i, j] = 0.0
                self.p_temp[i, j] = 0.0
                for q in range(Q):
                    self.f[i, j, q] = _feq(q, 1.0, velocity)
                    self.h[i, j, q] = 0.0
                ti.atomic_add(self.covered_cell_count[None], 1)
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
                        pressure_sum += self.p_temp[ni, nj]
                        velocity_sum += self.u[ni, nj]
                        count += 1.0
                phi0 = ti.min(1.0, ti.max(0.0, self.stored_phi[i, j]))
                pressure0 = self.stored_pressure[i, j]
                velocity0 = self._body_velocity_at(self._cell_point(i, j))
                if count > 0.0:
                    phi0 = phi_sum / count
                    pressure0 = pressure_sum / count
                    velocity0 = velocity_sum / count
                self.phi[i, j] = phi0
                self.rho[i, j] = _rho_mix(phi0, rho_water, rho_air)
                self.u[i, j] = velocity0
                self.u_temp[i, j] = velocity0
                self.p[i, j] = pressure0
                self.p_temp[i, j] = pressure0
                for q in range(Q):
                    self.f[i, j, q] = _feq(q, 1.0, velocity0)
                    self.h[i, j, q] = _heq(q, phi0, velocity0)
                ti.atomic_add(self.binary_fresh_water[None], ti.cast(phi0, ti.f64))
                ti.atomic_add(self.fresh_cell_count[None], 1)
        self.binary_remap_error[None] = self.binary_fresh_water[None] - self.binary_covered_water[None]

    # ------------------------------------------------------------------
    # Pure two-phase fluid: the minimally changed mixture2d LBM path

    @ti.kernel
    def _compute_fluid_force(self, steps: ti.i32):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        gravity = ti.Vector([ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])])
        beta = ti.static(12.0 * self._sigma_l / self.cfg.interface_width)
        kapa = ti.static(1.5 * self._sigma_l * self.cfg.interface_width)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nu_water = ti.static(self._nu_water_l)
        nu_air = ti.static(self._nu_air_l)
        delta_rho = ti.static(self._rho_water_l - self._rho_air_l)
        cd = ti.static(self.cfg.cd)
        for i, j in self.rho:
            if not self._active(i, j):
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                self.artificial_vis[i, j] = 0.0
                point = self._cell_point(i, j)
                self.u_temp[i, j] = self._body_velocity_at(point) if self.solid[i, j] == 1 else ti.Vector([0.0, 0.0])
            else:
                phi = self.phi[i, j]
                phi_force = ti.min(1.0, ti.max(0.0, phi))
                rho = _rho_mix(phi_force, rho_water, rho_air)

                grad = ti.Vector([0.0, 0.0])
                lap = 0.0
                alpha = ti.static(1.0 / 3.0)
                for qq in range(Q):
                    cc_i = _c(qq)
                    cc = ti.cast(cc_i, ti.f32)
                    phi1 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 1)
                    phi2 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 2)
                    grad += (1.0 - alpha) * 3.0 * _w(qq) * cc * (phi1 - phi)
                    grad += alpha * 1.5 * _w(qq) * cc * (4.0 * phi1 - phi2 - 3.0 * phi)
                    lap += 6.0 * _w(qq) * (phi1 - phi)

                force = gravity
                chemical = 4.0 * beta * phi_force * (phi_force - 1.0) * (phi_force - 0.5) - kapa * lap
                force += chemical * grad / rho

                u0 = self.u[i, j]
                pressure = self.p_temp[i, j]
                dux = 0.0
                duy = 0.0
                dvx = 0.0
                dvy = 0.0
                grad_p = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    cc = ti.cast(_c(q), ti.f32)
                    ni = i + _c(q).x
                    nj = j + _c(q).y
                    mx = 3.0 * _w(q) * cc.x
                    my = 3.0 * _w(q) * cc.y
                    u1 = ti.Vector([0.0, 0.0])
                    p1 = pressure
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        u1 = self.u[ni, nj]
                        p1 = self.p_temp[ni, nj]
                    elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                        u1 = self._body_velocity_at(self._cell_point(ni, nj))
                    diff_u = u1 - u0
                    dux += mx * diff_u.x
                    duy += my * diff_u.x
                    dvx += mx * diff_u.y
                    dvy += my * diff_u.y
                    grad_p += _w(q) * cc * 3.0 * (p1 - pressure)

                art_vis = cd * cd * ti.sqrt(
                    2.0 * (dux * dux + dvy * dvy + 0.5 * (duy + dvx) * (duy + dvx))
                )
                vis_k = _viscosity_mix(phi_force, rho_water, rho_air, nu_water, nu_air)
                vis_eff = vis_k + art_vis
                grad_rho = delta_rho * grad
                strain_xy = duy + dvx
                force.x += vis_eff * (2.0 * dux * grad_rho.x + strain_xy * grad_rho.y) / rho
                force.y += vis_eff * (strain_xy * grad_rho.x + 2.0 * dvy * grad_rho.y) / rho
                if phi_force < 0.1 and steps < 120 * 600:
                    grad_p *= 0.1
                force -= grad_p / rho

                # Same low-Re advection stabilization used by mixture2d.
                fa = ti.Vector([0.0, 0.0])
                for a in range(4):
                    axis = _axis_c(a)
                    ni = i + axis.x
                    nj = j + axis.y
                    diff = ti.Vector([0.0, 0.0])
                    u_abs = 0.0
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        u1 = self.u[ni, nj]
                        diff = u1 - u0 if a < 2 else u0 - u1
                        u_abs = ti.abs(u0.x + u1.x) * 0.5 if a == 0 or a == 2 else ti.abs(u0.y + u1.y) * 0.5
                        if u_abs / (vis_k + vis_k) < 4.0:
                            diff = ti.Vector([0.0, 0.0])
                    fa += (1.0 if a < 2 else -1.0) * diff * u_abs
                force += 0.5 * fa

                self.u_temp[i, j] = u0 + 0.5 * force
                self.artificial_vis[i, j] = art_vis
                self.fluid_force[i, j] = force

    @ti.kernel
    def _collide_velocity(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nu_water = ti.static(self._nu_water_l)
        nu_air = ti.static(self._nu_air_l)
        for i, j in self.rho:
            if self._active(i, j):
                phi = self.phi[i, j]
                velocity = self.u_temp[i, j]
                force = self.fluid_force[i, j]
                tau_local = _tau_mix(
                    phi, self.artificial_vis[i, j], rho_water, rho_air, nu_water, nu_air
                )
                m00 = 0.0
                m10 = 0.0
                m01 = 0.0
                m20 = 0.0
                m02 = 0.0
                m11 = 0.0
                for q in range(Q):
                    cx = ti.cast(_c(q).x, ti.f32)
                    cy = ti.cast(_c(q).y, ti.f32)
                    value = self.f[i, j, q]
                    m00 += value
                    m10 += value * cx
                    m01 += value * cy
                    m20 += value * cx * cx
                    m02 += value * cy * cy
                    m11 += value * cx * cy
                ux = velocity.x
                uy = velocity.y
                k20 = m20 - 2.0 * ux * m10 + ux * ux * m00
                k02 = m02 - 2.0 * uy * m01 + uy * uy * m00
                k11 = m11 - ux * m01 - uy * m10 + ux * uy * m00
                shear_difference = (k20 - k02) * (1.0 - 1.0 / tau_local)
                shear_xy = k11 * (1.0 - 1.0 / tau_local)
                k00 = 1.0
                k10 = 0.5 * force.x
                k01 = 0.5 * force.y
                k20 = 0.5 * (2.0 / 3.0 + shear_difference)
                k02 = 0.5 * (2.0 / 3.0 - shear_difference)
                k11 = shear_xy
                k21 = 0.5 * force.y / 3.0
                k12 = 0.5 * force.x / 3.0
                k22 = 1.0 / 9.0
                for q in range(Q):
                    self.f_post[i, j, q] = _reconstruct_central(
                        _c(q).x, _c(q).y, ux, uy, k00, k10, k01, k20, k02, k11, k21, k12, k22
                    )
            else:
                for q in range(Q):
                    self.f_post[i, j, q] = self.f[i, j, q]

    @ti.kernel
    def _collide_phase(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        interface_width = ti.static(self.cfg.interface_width)
        tau_inv_phi = ti.static(1.0 / (3.0 * self.cfg.mobility + 0.5))
        for i, j in self.phi:
            if self._active(i, j):
                phi = self.phi[i, j]
                velocity = self.u_temp[i, j]
                grad_phi = ti.Vector([0.0, 0.0])
                alpha = ti.static(1.0 / 3.0)
                for qq in range(Q):
                    cc_i = _c(qq)
                    cc = ti.cast(cc_i, ti.f32)
                    phi1 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 1)
                    phi2 = self._phase_neighbor(i, j, cc_i.x, cc_i.y, 2)
                    grad_phi += (1.0 - alpha) * 3.0 * _w(qq) * cc * (phi1 - phi)
                    grad_phi += alpha * 1.5 * _w(qq) * cc * (4.0 * phi1 - phi2 - 3.0 * phi)
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
                kh10 = hm10 - tau_inv_phi * (hm10 - he10) + (1.0 - 0.5 * tau_inv_phi) * hf10
                kh01 = hm01 - tau_inv_phi * (hm01 - he01) + (1.0 - 0.5 * tau_inv_phi) * hf01
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
            else:
                for q in range(Q):
                    self.h_post[i, j, q] = self.h[i, j, q]

    @ti.kernel
    def _clear_stream_targets(self):
        self.raw_hydrodynamic_impulse[None] = ti.Vector([0.0, 0.0])
        self.raw_hydrodynamic_torque[None] = 0.0
        self.cut_link_count[None] = 0
        for i, j, q in self.f_next:
            self.f_next[i, j, q] = 0.0
            self.h_next[i, j, q] = 0.0

    @ti.kernel
    def _clear_h_next(self):
        for i, j, q in self.h_next:
            self.h_next[i, j, q] = 0.0

    @ti.kernel
    def _stream_velocity(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            if self._active(i, j):
                for q in range(Q):
                    direction = _c(q)
                    ni = i + direction.x
                    nj = j + direction.y
                    outgoing = self.f_post[i, j, q]
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        self.f_next[ni, nj, q] = outgoing
                    elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                        sdf_fluid = ti.max(self.sdf[i, j], 1.0e-6)
                        sdf_solid = self.sdf[ni, nj]
                        eta = ti.min(0.95, ti.max(0.05, sdf_fluid / (sdf_fluid - sdf_solid + 1.0e-12)))
                        direction_f = ti.cast(direction, ti.f32)
                        fluid_point = self._cell_point(i, j)
                        boundary_point = fluid_point + eta * direction_f
                        boundary_velocity = self._body_velocity_at(boundary_point)
                        reflected = outgoing - 6.0 * _w(q) * direction_f.dot(boundary_velocity)
                        if ti.static(self._use_unified_boundary):
                            back_i = i - direction.x
                            back_j = j - direction.y
                            if _inside(back_i, back_j, nx, ny) and self._active(back_i, back_j):
                                reflected = (
                                    eta * self.f_post[i, j, _opp(q)]
                                    + (1.0 - eta) * self.f_post[back_i, back_j, q]
                                    + eta * outgoing
                                    - 6.0 * _w(q) * direction_f.dot(boundary_velocity)
                                ) / (1.0 + eta)
                        self.f_next[i, j, _opp(q)] = reflected

                        # Galilean-invariant momentum exchange.  f carries
                        # unit reference density, hence rho(phi) appears only
                        # here, never in the moving-wall population correction.
                        rho_link = _rho_mix(self.phi[i, j], rho_water, rho_air)
                        opposite_direction = -direction_f
                        impulse = rho_link * (
                            (direction_f - boundary_velocity) * outgoing
                            - (opposite_direction - boundary_velocity) * reflected
                            - 2.0 * _w(q) * direction_f
                        )
                        relative = boundary_point - self.body_center[None]
                        ti.atomic_add(self.raw_hydrodynamic_impulse[None].x, impulse.x)
                        ti.atomic_add(self.raw_hydrodynamic_impulse[None].y, impulse.y)
                        ti.atomic_add(self.raw_hydrodynamic_torque[None], _cross2(relative, impulse))
                        ti.atomic_add(self.cut_link_count[None], 1)
                    else:
                        # Static container wall, retaining the small damping
                        # used in mixture2d for the lower-index directions.
                        opposite = _opp(q)
                        reflected = outgoing
                        # if opposite < q:
                        #     reflected = 0.9 * outgoing + 0.1 * _feq(
                        #         opposite, 1.0, ti.Vector([0.0, 0.0])
                        #     )
                        self.f_next[i, j, opposite] = reflected

    @ti.kernel
    def _stream_phase(self):
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
                        self.h_next[ni, nj, q] = outgoing
                    elif _inside(ni, nj, nx, ny) and self.solid[ni, nj] == 1:
                        sdf_fluid = ti.max(self.sdf[i, j], 1.0e-6)
                        sdf_solid = self.sdf[ni, nj]
                        eta = ti.min(0.95, ti.max(0.05, sdf_fluid / (sdf_fluid - sdf_solid + 1.0e-12)))
                        direction_f = ti.cast(direction, ti.f32)
                        boundary_point = self._cell_point(i, j) + eta * direction_f
                        boundary_velocity = self._body_velocity_at(boundary_point)
                        # Moving no-flux boundary (Liu et al. Eq. 51 in the
                        # present push-stream convention).  It preserves a
                        # uniform phase field under Galilean co-motion.
                        reflected = outgoing - 6.0 * _w(q) * self.phi[i, j] * direction_f.dot(boundary_velocity)
                        self.h_next[i, j, _opp(q)] = reflected
                    else:
                        # Neutral wetting/no phase flux at both static and
                        # moving solid walls.  Static walls have zero velocity.
                        self.h_next[i, j, _opp(q)] = outgoing

    @ti.kernel
    def _copy_lbm_next(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            if self._active(i, j):
                phi = 0.0
                for q in range(Q):
                    self.f[i, j, q] = self.f_next[i, j, q]
                    self.h[i, j, q] = self.h_next[i, j, q]
                    phi += self.h_next[i, j, q]
                self.phi[i, j] = phi
                self.rho[i, j] = _rho_mix(phi, rho_water, rho_air)
            else:
                velocity = ti.Vector([0.0, 0.0])
                if self.solid[i, j] == 1:
                    velocity = self._body_velocity_at(self._cell_point(i, j))
                self.phi[i, j] = 0.0
                self.rho[i, j] = rho_air
                for q in range(Q):
                    self.f[i, j, q] = _feq(q, 1.0, velocity)
                    self.h[i, j, q] = 0.0

    @ti.kernel
    def _copy_h_next(self):
        for i, j in self.phi:
            if self._active(i, j):
                phi = 0.0
                for q in range(Q):
                    self.h[i, j, q] = self.h_next[i, j, q]
                    phi += self.h_next[i, j, q]
                self.phi[i, j] = phi
            else:
                self.phi[i, j] = 0.0
                for q in range(Q):
                    self.h[i, j, q] = 0.0

    @ti.kernel
    def _update_phi_from_h(self):
        for i, j in self.phi:
            if self._active(i, j):
                value = 0.0
                for q in range(Q):
                    value += self.h[i, j, q]
                self.phi[i, j] = value
            else:
                self.phi[i, j] = 0.0

    def _warm_start_phase(self, steps):
        for _ in range(int(steps)):
            self._update_phi_from_h()
            self._collide_phase()
            self._clear_h_next()
            self._stream_phase()
            self._copy_h_next()

    @ti.kernel
    def _update_fluid_macro(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            if self._active(i, j):
                phi = 0.0
                momentum = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    phi += self.h[i, j, q]
                    momentum += ti.cast(_c(q), ti.f32) * self.f[i, j, q]
                self.phi[i, j] = phi
                self.rho[i, j] = _rho_mix(phi, rho_water, rho_air)
                self.u[i, j] = momentum
            elif self.solid[i, j] == 1:
                self.phi[i, j] = 0.0
                self.rho[i, j] = rho_air
                self.u[i, j] = self._body_velocity_at(self._cell_point(i, j))
                self.u_temp[i, j] = self.u[i, j]
            else:
                self.phi[i, j] = 0.0
                self.rho[i, j] = rho_air
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.u_temp[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _update_pressure(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.p:
            if self._active(i, j):
                distribution_sum = 0.0
                for q in range(Q):
                    distribution_sum += self.f[i, j, q]
                rho = _rho_mix(self.phi[i, j], rho_water, rho_air)
                self.p[i, j] = self.p_temp[i, j] + rho / 3.0 * (distribution_sum - 1.0)
            else:
                self.p[i, j] = 0.0

    @ti.kernel
    def _average_pressure(self):
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.p_temp:
            if self._active(i, j):
                pressure = 0.0
                for q in range(Q):
                    ni = i + _c(q).x
                    nj = j + _c(q).y
                    neighbor = self.p[i, j]
                    if _inside(ni, nj, nx, ny) and self._active(ni, nj):
                        neighbor = self.p[ni, nj]
                    pressure += _w(q) * neighbor
                self.p_temp[i, j] = pressure
            else:
                self.p_temp[i, j] = 0.0

    # ------------------------------------------------------------------
    # Rigid-body forcing and integration

    @ti.func
    def _sample_phase_outside_box(self, point):
        """Extend phase along a world-horizontal chord of the ice body.

        Hydrostatic pressure varies vertically, so samples at the left and
        right intersections of the same horizontal line recover the immersed
        area of a body under a horizontal free surface.  Interpolating those
        samples also retains left/right asymmetry during the dam-break impact.
        """
        center = self.body_center[None]
        angle = self.body_angle[None]
        cosine = ti.cos(angle)
        sine = ti.sin(angle)
        relative = point - center
        local = ti.Vector([
            cosine * relative.x + sine * relative.y,
            -sine * relative.x + cosine * relative.y,
        ])
        # World direction (1, 0) expressed in body coordinates.
        direction_local = ti.Vector([cosine, -sine])
        t_min = -1.0e6
        t_max = 1.0e6
        half_width = ti.static(self._body_half_width)
        half_height = ti.static(self._body_half_height)
        if ti.abs(direction_local.x) > 1.0e-7:
            tx0 = (-half_width - local.x) / direction_local.x
            tx1 = (half_width - local.x) / direction_local.x
            t_min = ti.max(t_min, ti.min(tx0, tx1))
            t_max = ti.min(t_max, ti.max(tx0, tx1))
        if ti.abs(direction_local.y) > 1.0e-7:
            ty0 = (-half_height - local.y) / direction_local.y
            ty1 = (half_height - local.y) / direction_local.y
            t_min = ti.max(t_min, ti.min(ty0, ty1))
            t_max = ti.min(t_max, ti.max(ty0, ty1))

        fallback_i = ti.min(ti.static(self.nx - 1), ti.max(0, ti.cast(ti.floor(point.x), ti.i32)))
        fallback_j = ti.min(ti.static(self.ny - 1), ti.max(0, ti.cast(ti.floor(point.y), ti.i32)))
        phase_left = self.stored_phi[fallback_i, fallback_j]
        phase_right = phase_left
        found_left = 0
        found_right = 0
        for sample_index in ti.static(range(3)):
            offset = 0.75 + ti.cast(sample_index, ti.f32)
            sample_left = ti.Vector([point.x + t_min - offset, point.y])
            sample_right = ti.Vector([point.x + t_max + offset, point.y])
            li = ti.min(ti.static(self.nx - 1), ti.max(0, ti.cast(ti.floor(sample_left.x), ti.i32)))
            lj = ti.min(ti.static(self.ny - 1), ti.max(0, ti.cast(ti.floor(sample_left.y), ti.i32)))
            ri = ti.min(ti.static(self.nx - 1), ti.max(0, ti.cast(ti.floor(sample_right.x), ti.i32)))
            rj = ti.min(ti.static(self.ny - 1), ti.max(0, ti.cast(ti.floor(sample_right.y), ti.i32)))
            if found_left == 0 and self._active(li, lj):
                phase_left = self.phi[li, lj]
                found_left = 1
            if found_right == 0 and self._active(ri, rj):
                phase_right = self.phi[ri, rj]
                found_right = 1
        chord_fraction = ti.min(1.0, ti.max(0.0, -t_min / (t_max - t_min + 1.0e-12)))
        phase = (1.0 - chord_fraction) * phase_left + chord_fraction * phase_right
        return ti.min(1.0, ti.max(0.0, phase))

    @ti.kernel
    def _compute_hydrostatic_buoyancy(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        gravity = ti.Vector([ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])])
        self.buoyancy_impulse[None] = ti.Vector([0.0, 0.0])
        # The chord extension gives a robust zeroth-order displaced volume,
        # but its first moment is not a reliable transient centre of pressure
        # during dam impact.  Apply this explicit Archimedes correction through
        # the centre of mass; resolved rotation comes from cut-link momentum
        # exchange until pressure traction replaces this volume approximation.
        self.buoyancy_torque[None] = 0.0
        self.displaced_water_volume[None] = 0.0
        cell_weight = ti.cast(
            ti.static(float(self.cfg.ice_width * self.cfg.ice_height))
            / ti.max(1, self.solid_cell_count[None]),
            ti.f32,
        )
        for i, j in self.solid:
            if self.solid[i, j] == 1:
                point = self._cell_point(i, j)
                phase = self._sample_phase_outside_box(point)
                displaced_density = _rho_mix(phase, rho_water, rho_air)
                cell_impulse = -cell_weight * displaced_density * gravity
                ti.atomic_add(self.buoyancy_impulse[None].x, cell_impulse.x)
                ti.atomic_add(self.buoyancy_impulse[None].y, cell_impulse.y)
                ti.atomic_add(self.displaced_water_volume[None], ti.cast(cell_weight * phase, ti.f64))

    @ti.kernel
    def _advance_rigid_ice(self):
        relaxation = ti.static(self.cfg.force_relaxation)
        hydro_scale = ti.static(self.cfg.hydrodynamic_force_scale)
        buoyancy_scale = ti.static(self.cfg.hydrostatic_buoyancy_scale)
        mass = ti.static(self._body_mass)
        inertia = ti.static(self._body_inertia)
        gravity = ti.Vector([ti.static(self._gravity_l[0]), ti.static(self._gravity_l[1])])
        filtered_impulse = (
            (1.0 - relaxation) * self.filtered_hydrodynamic_impulse[None]
            + relaxation * self.raw_hydrodynamic_impulse[None]
        )
        filtered_torque = (
            (1.0 - relaxation) * self.filtered_hydrodynamic_torque[None]
            + relaxation * self.raw_hydrodynamic_torque[None]
        )
        self.filtered_hydrodynamic_impulse[None] = filtered_impulse
        self.filtered_hydrodynamic_torque[None] = filtered_torque

        total_impulse = hydro_scale * filtered_impulse + buoyancy_scale * self.buoyancy_impulse[None] + mass * gravity
        total_torque = hydro_scale * filtered_torque
        self.total_body_impulse[None] = total_impulse
        self.total_body_torque[None] = total_torque

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
            local_x = ti.static(self._body_half_width) * (-1.0 if ti.static(corner % 2 == 0) else 1.0)
            local_y = ti.static(self._body_half_height) * (-1.0 if ti.static(corner < 2) else 1.0)
            relative = ti.Vector([
                cosine_old * local_x - sine_old * local_y,
                sine_old * local_x + cosine_old * local_y,
            ])
            lever_x[corner] = relative.x
            lever_y[corner] = relative.y
            gap = old_center.y + relative.y - floor_y
            normal_velocity = velocity.y + omega * relative.x
            if gap <= contact_slop or gap + ti.min(normal_velocity, 0.0) <= 0.0:
                active_contact[corner] = 1
                if normal_velocity < -restitution_threshold:
                    normal_target[corner] = -ti.static(self.cfg.wall_restitution) * normal_velocity

        # A handful of iterations is ample for this four-contact scalar LCP
        # and avoids choosing an arbitrary corner for an initially flat base.
        for _ in ti.static(range(12)):
            for corner in ti.static(range(4)):
                if active_contact[corner] == 1:
                    rx = lever_x[corner]
                    normal_velocity = velocity.y + omega * rx
                    effective_inverse_mass = inv_mass + rx * rx * inv_inertia
                    delta_impulse = (normal_target[corner] - normal_velocity) / effective_inverse_mass
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
                        ti.max(-friction_limit, old_tangent_impulse + delta_tangent_impulse),
                    )
                    applied_tangent_impulse = new_tangent_impulse - old_tangent_impulse
                    tangent_impulse[corner] = new_tangent_impulse
                    velocity.x += applied_tangent_impulse * inv_mass
                    omega -= ry * applied_tangent_impulse * inv_inertia

        contact_impulse = 0.0
        contact_torque_impulse = 0.0
        friction_impulse = 0.0
        friction_abs_impulse = 0.0
        friction_torque_impulse = 0.0
        bottom_active = 0
        for corner in ti.static(range(4)):
            contact_impulse += normal_impulse[corner]
            contact_torque_impulse += lever_x[corner] * normal_impulse[corner]
            friction_impulse += tangent_impulse[corner]
            friction_abs_impulse += ti.abs(tangent_impulse[corner])
            friction_torque_impulse -= lever_y[corner] * tangent_impulse[corner]
            bottom_active = ti.max(bottom_active, active_contact[corner])
        self.bottom_contact_impulse[None] = contact_impulse
        self.bottom_contact_torque_impulse[None] = contact_torque_impulse
        self.bottom_friction_impulse[None] = friction_impulse
        self.bottom_friction_abs_impulse[None] = friction_abs_impulse
        self.bottom_friction_torque_impulse[None] = friction_torque_impulse

        # Do not clip omega after the constraint solve: doing so would destroy
        # the just-enforced contact velocity and break angular-impulse balance.
        angle = self.body_angle[None] + omega
        center = self.body_center[None] + velocity

        # Split position impulses remove any residual corner penetration
        # without changing physical velocity or injecting kinetic energy.  In
        # contrast to an AABB snap, a one-corner correction changes both the
        # centre height and angle according to the same generalized inverse
        # mass used by the velocity constraint.
        predicted_center_y = center.y
        position_contact = 0
        position_slop = ti.static(1.0e-5)
        position_beta = ti.static(0.8)
        for _ in ti.static(range(12)):
            for corner in ti.static(range(4)):
                cosine_position = ti.cos(angle)
                sine_position = ti.sin(angle)
                local_x = ti.static(self._body_half_width) * (-1.0 if ti.static(corner % 2 == 0) else 1.0)
                local_y = ti.static(self._body_half_height) * (-1.0 if ti.static(corner < 2) else 1.0)
                relative = ti.Vector([
                    cosine_position * local_x - sine_position * local_y,
                    sine_position * local_x + cosine_position * local_y,
                ])
                penetration = floor_y - (center.y + relative.y)
                if penetration > position_slop:
                    effective_inverse_mass = inv_mass + relative.x * relative.x * inv_inertia
                    split_impulse = position_beta * (penetration - position_slop) / effective_inverse_mass
                    center.y += split_impulse * inv_mass
                    angle += relative.x * split_impulse * inv_inertia
                    position_contact = 1
        self.bottom_position_correction[None] = center.y - predicted_center_y

        cosine = ti.abs(ti.cos(angle))
        sine = ti.abs(ti.sin(angle))
        extent_x = cosine * ti.static(self._body_half_width) + sine * ti.static(self._body_half_height)
        extent_y = sine * ti.static(self._body_half_width) + cosine * ti.static(self._body_half_height)
        lower_x = ti.static(float(self.cfg.boundary_cells)) + extent_x
        upper_x = ti.static(float(self.nx - self.cfg.boundary_cells)) - extent_x
        upper_y = ti.static(float(self.ny - self.cfg.boundary_cells)) - extent_y
        restitution = ti.static(self.cfg.wall_restitution)
        tangential = 1.0 - ti.static(self.cfg.wall_friction)
        contact = ti.max(bottom_active, position_contact)
        if center.x < lower_x:
            center.x = lower_x
            if velocity.x < 0.0:
                velocity.x = -restitution * velocity.x
                velocity.y *= tangential
                omega *= tangential
            contact = 1
        elif center.x > upper_x:
            center.x = upper_x
            if velocity.x > 0.0:
                velocity.x = -restitution * velocity.x
                velocity.y *= tangential
                omega *= tangential
            contact = 1
        if center.y > upper_y:
            center.y = upper_y
            if velocity.y > 0.0:
                velocity.y = -restitution * velocity.y
                velocity.x *= tangential
                omega *= tangential
            contact = 1
        self.body_center[None] = center
        self.body_velocity[None] = velocity
        self.body_angle[None] = angle
        self.body_angular_velocity[None] = omega
        self.wall_contact[None] = contact

    # ------------------------------------------------------------------
    # Bounded, topology-constrained water-volume projection

    @ti.kernel
    def _initialize_water_volume(self):
        self.water_volume_target[None] = 0.0
        self.water_volume_current[None] = 0.0
        self.water_volume_bounded[None] = 0.0
        self.water_volume_raw[None] = 0.0
        self.water_volume_error[None] = 0.0
        self.water_projection_mass[None] = 0.0
        self.water_projection_derivative[None] = 0.0
        self.water_projection_lambda[None] = 0.0
        self.water_projection_shift[None] = 0.0
        self.water_projection_feasible[None] = 1
        self.water_projection_interface_cells[None] = 0
        self.water_projection_iterations[None] = 0
        for i, j in self.phi:
            if self._active(i, j):
                phase = self.phi[i, j]
                ti.atomic_add(self.water_volume_target[None], ti.cast(phase, ti.f64))
            self.water_projection_candidate[i, j] = 0
            self.water_projection_mask[i, j] = 0
            self.water_projection_mask_next[i, j] = 0
        self.water_volume_current[None] = self.water_volume_target[None]
        self.water_volume_bounded[None] = self.water_volume_target[None]
        self.water_volume_raw[None] = self.water_volume_target[None]

    @ti.kernel
    def _measure_water_volume(self):
        self.water_volume_current[None] = 0.0
        self.water_volume_bounded[None] = 0.0
        self.water_volume_raw[None] = 0.0
        for i, j in self.phi:
            if self._active(i, j):
                phase = self.phi[i, j]
                ti.atomic_add(self.water_volume_raw[None], ti.cast(phase, ti.f64))
                ti.atomic_add(self.water_volume_current[None], ti.cast(phase, ti.f64))
                ti.atomic_add(self.water_volume_bounded[None], ti.cast(ti.min(1.0, ti.max(0.0, phase)), ti.f64))
        self.water_volume_error[None] = self.water_volume_target[None] - self.water_volume_current[None]

    @ti.kernel
    def _clip_water_phase_for_projection(self):
        """Restore bounds/pure bulks while retaining h's nonequilibrium part."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
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
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = self.h[i, j, q] + _heq(q, bounded_phase, velocity) - _heq(
                            q, old_phase, velocity
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    # Close the f32 zeroth moment without changing any bulk
                    # cell that was already exactly zero or one.
                    self.h[i, j, 0] += bounded_phase - lifted_sum
                self.phi[i, j] = bounded_phase
                self.rho[i, j] = _rho_mix(bounded_phase, rho_water, rho_air)

    @ti.kernel
    def _seed_water_projection_interface(self):
        """Seed candidates connected to a local phi=1/2 crossing."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        cutoff = ti.static(self.cfg.volume_projection_interface_cutoff)
        for i, j in self.phi:
            candidate = 0
            seed = 0
            if self._active(i, j):
                phase = self.phi[i, j]
                if cutoff < phase < 1.0 - cutoff:
                    candidate = 1
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
            self.water_projection_candidate[i, j] = candidate
            self.water_projection_mask[i, j] = seed
            self.water_projection_mask_next[i, j] = 0

    @ti.kernel
    def _dilate_water_projection_interface(self, primary_to_next: ti.i32):
        """One deterministic 8-connected dilation through candidate cells."""

        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.phi:
            value = 0
            if self.water_projection_candidate[i, j] == 1:
                if primary_to_next == 1:
                    value = self.water_projection_mask[i, j]
                else:
                    value = self.water_projection_mask_next[i, j]
                if value == 0:
                    for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        ni = i + di
                        nj = j + dj
                        if (di != 0 or dj != 0) and _inside(ni, nj, nx, ny):
                            neighbor = 0
                            if primary_to_next == 1:
                                neighbor = self.water_projection_mask[ni, nj]
                            else:
                                neighbor = self.water_projection_mask_next[ni, nj]
                            if neighbor == 1:
                                value = 1
            if primary_to_next == 1:
                self.water_projection_mask_next[i, j] = value
            else:
                self.water_projection_mask[i, j] = value

    @ti.kernel
    def _count_water_projection_interface(self):
        self.water_projection_interface_cells[None] = 0
        for i, j in self.phi:
            if self.water_projection_mask[i, j] == 1:
                ti.atomic_add(self.water_projection_interface_cells[None], 1)

    @ti.kernel
    def _evaluate_water_projection(self, lagrange_multiplier: ti.f64):
        """Evaluate volume and derivative using f64 reductions.

        The mapped value is rounded to the f32 phase storage type before the
        reduction.  Consequently this is the same discrete volume that the
        application kernel and the public ledger will observe.
        """

        self.water_projection_mass[None] = 0.0
        self.water_projection_derivative[None] = 0.0
        exponential = ti.exp(lagrange_multiplier)
        for i, j in self.phi:
            if self._active(i, j):
                phase64 = ti.cast(self.phi[i, j], ti.f64)
                mapped64 = phase64
                if self.water_projection_mask[i, j] == 1:
                    mapped64 = phase64 * exponential / (
                        1.0 - phase64 + phase64 * exponential
                    )
                mapped32 = ti.cast(mapped64, ti.f32)
                mapped64 = ti.cast(mapped32, ti.f64)
                ti.atomic_add(self.water_projection_mass[None], mapped64)
                if self.water_projection_mask[i, j] == 1:
                    ti.atomic_add(
                        self.water_projection_derivative[None], mapped64 * (1.0 - mapped64)
                    )

    @ti.kernel
    def _apply_water_projection(self, lagrange_multiplier: ti.f64):
        """Apply the entropic map and lift h by the equilibrium difference."""

        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        exponential = ti.exp(lagrange_multiplier)
        for i, j in self.phi:
            if self._active(i, j) and self.water_projection_mask[i, j] == 1:
                old_phase = self.phi[i, j]
                old_phase64 = ti.cast(old_phase, ti.f64)
                mapped64 = old_phase64 * exponential / (
                    1.0 - old_phase64 + old_phase64 * exponential
                )
                new_phase = ti.cast(mapped64, ti.f32)
                if new_phase != old_phase:
                    velocity = self.u[i, j]
                    lifted_sum = 0.0
                    for q in range(Q):
                        lifted = self.h[i, j, q] + _heq(q, new_phase, velocity) - _heq(
                            q, old_phase, velocity
                        )
                        self.h[i, j, q] = lifted
                        lifted_sum += lifted
                    self.h[i, j, 0] += new_phase - lifted_sum
                    self.phi[i, j] = new_phase
                    self.rho[i, j] = _rho_mix(new_phase, rho_water, rho_air)

    def _correct_water_volume(self):
        """Project the active phase field onto the exact no-melting volume.

        A scalar Bernoulli-relative-entropy projection translates the existing
        logistic interface.  Its support is restricted to the diffuse band
        connected to phi=1/2, so disconnected minority-phase noise and exact
        bulk values cannot receive an additive volume source.
        """

        self.water_projection_lambda[None] = 0.0
        self.water_projection_shift[None] = 0.0
        self.water_projection_iterations[None] = 0
        self.water_projection_interface_cells[None] = 0
        self.water_projection_feasible[None] = 1

        self._clip_water_phase_for_projection()
        self._measure_water_volume()
        target = float(self.water_volume_target[None])
        tolerance = float(self.cfg.volume_projection_tolerance) * max(1.0, abs(target))
        initial_error = target - float(self.water_volume_current[None])
        if abs(initial_error) <= tolerance:
            return

        self._seed_water_projection_interface()
        for pass_index in range(self._volume_projection_band_radius):
            self._dilate_water_projection_interface(1 if pass_index % 2 == 0 else 0)
        self._count_water_projection_interface()
        interface_cells = int(self.water_projection_interface_cells[None])
        if interface_cells == 0:
            self.water_projection_feasible[None] = 0
            raise RuntimeError(
                "water-volume projection is infeasible: nonzero volume error "
                "but no phi=0.5-connected interface cells are available"
            )

        lambda_limit = 4.0 * float(self.cfg.volume_projection_max_shift) / float(
            self.cfg.interface_width
        )
        self._evaluate_water_projection(-lambda_limit)
        lower_mass = float(self.water_projection_mass[None])
        self._evaluate_water_projection(lambda_limit)
        upper_mass = float(self.water_projection_mass[None])
        if target < lower_mass - tolerance or target > upper_mass + tolerance:
            self.water_projection_feasible[None] = 0
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
            mass = float(self.water_projection_mass[None])
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
            if not math.isfinite(proposal) or not lower_lambda < proposal < upper_lambda:
                proposal = 0.5 * (lower_lambda + upper_lambda)
            lagrange_multiplier = proposal

        self.water_projection_iterations[None] = iterations
        if not converged:
            # With f32 phase storage the volume map is piecewise constant.
            # Accept the best representable state only when it satisfies the
            # declared conservation tolerance.
            self._evaluate_water_projection(best_lambda)
            best_residual = abs(float(self.water_projection_mass[None]) - target)
            if best_residual > tolerance:
                self.water_projection_feasible[None] = 0
                raise RuntimeError(
                    "water-volume projection did not converge to storage precision: "
                    f"residual={best_residual:.6e}, tolerance={tolerance:.6e}"
                )

        self.water_projection_lambda[None] = best_lambda
        # Report a non-negative displacement magnitude; lambda retains the
        # direction of the normal translation.
        self.water_projection_shift[None] = (
            0.25 * float(self.cfg.interface_width) * abs(best_lambda)
        )
        self._apply_water_projection(best_lambda)
        self._measure_water_volume()
        final_error = abs(float(self.water_volume_error[None]))
        if final_error > tolerance:
            self.water_projection_feasible[None] = 0
            raise RuntimeError(
                "water-volume projection application disagrees with its f64 ledger: "
                f"residual={final_error:.6e}, tolerance={tolerance:.6e}"
            )

    # ------------------------------------------------------------------
    # Visualization

    @ti.kernel
    def _render(self):
        for i, j in self.image:
            phase = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
            air = ti.Vector([0.97, 0.98, 0.99])
            water = ti.Vector([0.06, 0.38, 0.88])
            color = air * (1.0 - phase) + water * phase
            if self.solid[i, j] == 1:
                # Subtle blue-white gradient makes rotation visible without
                # using particles or implying a porous interior.
                local_height = (ti.cast(j, ti.f32) - self.body_center[None].y) / (
                    2.0 * ti.static(self._body_half_height) + 1.0e-6
                )
                tint = ti.min(1.0, ti.max(0.0, 0.55 + 0.25 * local_height))
                color = ti.Vector([0.62, 0.84, 0.96]) * (1.0 - tint) + ti.Vector([0.92, 0.98, 1.0]) * tint
            if self.wall[i, j] == 1:
                color = ti.Vector([0.12, 0.13, 0.14])
            self.image[i, j] = color


# Familiar alias for callers migrating from mixture2d.
Simulator2D = IceFlow2D
