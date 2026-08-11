"""2D Taichi implementation of the paper's LBM-MPM mixture solver.

This file is an open-source 2D dam-break reduction of the method in
"Volume-Preserving LBM-MPM Coupling for Air-Water-Sand Mixtures".
Comments below point to the relevant paper sections/equations. The paper's
full solver is 3D, uses D3Q27/D3Q7 lattices, in-place streaming, mesh SDFs,
and broader production features; this version keeps only the CUDA 2D dam-break
case, uses D2Q9 distributions and explicit double buffers, and omits mesh/SDF
solid coupling.
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import taichi as ti

from .config import DamBreakConfig


Q = 9
_TAICHI_INITIALIZED = False


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    total_seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _TerminalProgress:
    """Small dependency-free frame progress bar for the command-line demo."""

    def __init__(
        self,
        total: int,
        *,
        label: str,
        enabled: bool,
        stream: TextIO | None = None,
        width: int = 24,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.enabled = bool(enabled) and self.total > 0
        self.stream = stream if stream is not None else sys.stderr
        self.width = max(1, int(width))
        self.completed = 0
        self._started_at: float | None = None
        self._last_line_length = 0
        self._closed = False

    def __enter__(self) -> "_TerminalProgress":
        if self.enabled:
            self._started_at = time.monotonic()
            self._render(self._started_at)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def advance(self) -> None:
        self.completed = min(self.total, self.completed + 1)
        if self.enabled:
            self._render(time.monotonic())

    def close(self) -> None:
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, now: float) -> None:
        started_at = self._started_at if self._started_at is not None else now
        elapsed = max(0.0, now - started_at)
        fraction = self.completed / self.total
        filled = min(self.width, int(fraction * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self.completed / elapsed if self.completed > 0 and elapsed > 0 else 0.0
        eta = (self.total - self.completed) / rate if rate > 0 else None
        line = (
            f"Simulating {self.label} [{bar}] {fraction:6.1%} | "
            f"{self.completed}/{self.total} frames | elapsed {_format_duration(elapsed)} | "
            f"ETA {_format_duration(eta)}"
        )
        padding = " " * max(0, self._last_line_length - len(line))
        self.stream.write(f"\r{line}{padding}")
        self.stream.flush()
        self._last_line_length = len(line)


def ensure_taichi_cuda() -> None:
    global _TAICHI_INITIALIZED
    if not _TAICHI_INITIALIZED:
        try:
            ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
        except Exception as exc:  # pragma: no cover - depends on host GPU
            raise RuntimeError("Mixture2D requires Taichi CUDA. No CPU fallback is provided.") from exc
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError("Mixture2D requires Taichi CUDA. No CPU fallback is provided.")
        _TAICHI_INITIALIZED = True


@ti.func
def _c(q):
    out = ti.Vector([0, 0], dt=ti.i32)
    if q == 1:
        out = ti.Vector([1, 0])
    elif q == 2:
        out = ti.Vector([0, 1])
    elif q == 3:
        out = ti.Vector([-1, 0])
    elif q == 4:
        out = ti.Vector([0, -1])
    elif q == 5:
        out = ti.Vector([1, 1])
    elif q == 6:
        out = ti.Vector([-1, 1])
    elif q == 7:
        out = ti.Vector([-1, -1])
    elif q == 8:
        out = ti.Vector([1, -1])
    return out


@ti.func
def _w(q):
    out = ti.cast(4.0 / 9.0, ti.f32)
    if 1 <= q <= 4:
        out = ti.cast(1.0 / 9.0, ti.f32)
    elif q >= 5:
        out = ti.cast(1.0 / 36.0, ti.f32)
    return out


@ti.func
def _opp(q):
    out = 0
    if q == 1:
        out = 3
    elif q == 2:
        out = 4
    elif q == 3:
        out = 1
    elif q == 4:
        out = 2
    elif q == 5:
        out = 7
    elif q == 6:
        out = 8
    elif q == 7:
        out = 5
    elif q == 8:
        out = 6
    return out


@ti.func
def _feq(q, rho, u):
    cq = ti.cast(_c(q), ti.f32)
    cu = cq.dot(u)
    uu = u.dot(u)
    return _w(q) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)


@ti.func
def _heq(q, phi, u):
    cq = ti.cast(_c(q), ti.f32)
    cu = cq.dot(u)
    uu = u.dot(u)
    return _w(q) * phi * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)


@ti.func
def _inside(i, j, nx, ny):
    return 0 <= i < nx and 0 <= j < ny


@ti.func
def _rho_mix(phi, rho_water, rho_air):
    p = ti.min(1.0, ti.max(0.0, phi))
    return rho_water * p + rho_air * (1.0 - p)


@ti.func
def _viscosity_mix(phi, rho_water, rho_air, nu_water, nu_air):
    p = ti.min(1.0, ti.max(0.0, phi))
    rho = _rho_mix(p, rho_water, rho_air)
    return (rho_water * nu_water * p + rho_air * nu_air * (1.0 - p)) / rho


@ti.func
def _tau_mix(phi, artificial_vis, rho_water, rho_air, nu_water, nu_air):
    return 3.0 * (_viscosity_mix(phi, rho_water, rho_air, nu_water, nu_air) + artificial_vis) + 0.5


@ti.func
def _axis_c(idx):
    out = ti.Vector([0, 0], dt=ti.i32)
    if idx == 0:
        out = ti.Vector([1, 0])
    elif idx == 1:
        out = ti.Vector([0, 1])
    elif idx == 2:
        out = ti.Vector([-1, 0])
    else:
        out = ti.Vector([0, -1])
    return out


@ti.func
def _inv1d(c, u, k):
    out = 0.0
    if c == 0:
        if k == 0:
            out = 1.0 - u * u
        elif k == 1:
            out = -2.0 * u
        else:
            out = -1.0
    elif c == 1:
        if k == 0:
            out = 0.5 * u * (u + 1.0)
        elif k == 1:
            out = u + 0.5
        else:
            out = 0.5
    else:
        if k == 0:
            out = 0.5 * u * (u - 1.0)
        elif k == 1:
            out = u - 0.5
        else:
            out = 0.5
    return out


@ti.func
def _reconstruct_central(cxi, cyi, ux, uy, k00, k10, k01, k20, k02, k11, k21, k12, k22):
    ix0 = _inv1d(cxi, ux, 0)
    ix1 = _inv1d(cxi, ux, 1)
    ix2 = _inv1d(cxi, ux, 2)
    iy0 = _inv1d(cyi, uy, 0)
    iy1 = _inv1d(cyi, uy, 1)
    iy2 = _inv1d(cyi, uy, 2)
    return (
        ix0 * iy0 * k00
        + ix1 * iy0 * k10
        + ix0 * iy1 * k01
        + ix2 * iy0 * k20
        + ix0 * iy2 * k02
        + ix1 * iy1 * k11
        + ix2 * iy1 * k21
        + ix1 * iy2 * k12
        + ix2 * iy2 * k22
    )


@ti.data_oriented
class Simulator2D:
    def __init__(self, config: DamBreakConfig):
        ensure_taichi_cuda()
        if config.mode not in ("fluid", "sand", "coupled"):
            raise ValueError("mode must be one of: fluid, sand, coupled")
        self.cfg = config
        self.nx = int(config.nx)
        self.ny = int(config.ny)
        self.n_particles = int(config.particle_count)
        self.frame = 0
        self.steps = 0

        # Paper Sec. 5.1: dimensionless conversion between physical and LBM
        # units. The user-facing velocity scale is encoded through u_ref; the
        # lattice reference velocity remains 0.1 as in the reference code.
        g_ref = abs(float(config.gravity[1])) if abs(float(config.gravity[1])) > 0 else 9.8
        u_ref = math.sqrt(g_ref * config.reference_length_cells * config.dx * 4.0)
        u_ref_lattice = 0.1
        c_u = u_ref / u_ref_lattice
        self._rho_water_l = 1.0
        self._rho_air_l = float(config.rho_air / config.rho_water)
        self._nu_water_l = float(config.viscosity_water * u_ref_lattice / (config.dx * u_ref))
        self._nu_air_l = float(config.viscosity_air * u_ref_lattice / (config.dx * u_ref))
        self._gravity_l = float(g_ref * config.dx * u_ref_lattice * u_ref_lattice / (u_ref * u_ref))
        self._sigma_l = float(config.sigma * u_ref_lattice * u_ref_lattice / (u_ref * u_ref) / config.dx / config.rho_water)
        sand_mu_phy = config.sand_youngs_modulus / (2.0 * (1.0 + config.sand_poisson_ratio))
        sand_lambda_phy = (
            config.sand_youngs_modulus
            * config.sand_poisson_ratio
            / ((1.0 + config.sand_poisson_ratio) * (1.0 - 2.0 * config.sand_poisson_ratio))
        )
        self._sand_density_l = float(config.sand_density / config.rho_water)
        self._sand_mu_l = float(sand_mu_phy / (config.rho_water * c_u * c_u))
        self._sand_lambda_l = float(sand_lambda_phy / (config.rho_water * c_u * c_u))
        self._sand_gravity_l = float(config.gravity[1] * config.dx / (c_u * c_u))
        self._mpm_dt_l = float(config.mpm_dt)

        # Fluid fields. f solves the velocity/pressure LBM (paper Sec. 4.1,
        # Eqs. 8-23), while h solves the conservative phase-field LBM (paper
        # Sec. 4.4, Eqs. 46-50). Both use D2Q9 in this 2D release.
        self.f = ti.field(ti.f32, shape=(self.nx, self.ny, Q))
        self.f_next = ti.field(ti.f32, shape=(self.nx, self.ny, Q))
        self.h = ti.field(ti.f32, shape=(self.nx, self.ny, Q))
        self.h_next = ti.field(ti.f32, shape=(self.nx, self.ny, Q))
        self.rho = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.phi = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.u = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.u_temp = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.fluid_force = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.p = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.p_temp = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.artificial_vis = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.wall = ti.field(ti.i32, shape=(self.nx, self.ny))

        # Sand/MPM fields. These are the 2D particle states used in paper
        # Sec. 4.2: position x_p, velocity v_p, APIC affine C_p, deformation
        # gradient F_p, hardening q/alpha, and bound water from Sec. 4.5.
        self.p_x = ti.Vector.field(2, ti.f32, shape=self.n_particles)
        self.p_v = ti.Vector.field(2, ti.f32, shape=self.n_particles)
        self.p_C = ti.Matrix.field(2, 2, ti.f32, shape=self.n_particles)
        self.p_F = ti.Matrix.field(2, 2, ti.f32, shape=self.n_particles)
        self.p_q = ti.field(ti.f32, shape=self.n_particles)
        self.p_vcs = ti.field(ti.f32, shape=self.n_particles)
        self.p_alpha = ti.field(ti.f32, shape=self.n_particles)
        self.p_bound_water = ti.field(ti.f32, shape=self.n_particles)
        self.p_water_content = ti.field(ti.f32, shape=self.n_particles)
        self.p_state = ti.field(ti.i32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.sand_grid_force = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.grid_m = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.delta = ti.field(ti.f32, shape=(self.nx, self.ny))

        # Coupling fields. delta is the sediment volume fraction alpha_s
        # (paper Eq. 25), epsinon is the fluid porosity epsilon = 1 - alpha_s
        # (Eq. 33), and epsinon_src is the effective porosity used by the
        # volume-preserving phase source (Secs. 4.4-4.5).
        self.grid_bound_water = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.grid_bound_water_post = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.grid_expand_ratio = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.phi_absorbed = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.epsinon = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.epsinon_src = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.epsinon_post = ti.field(ti.f32, shape=(self.nx, self.ny))
        # Reference uses double precision for integrated water volumes to
        # reduce accumulation drift in the volume controller.
        self.sum_vol0 = ti.field(ti.f64, shape=())
        self.sum_vol = ti.field(ti.f64, shape=())
        self.delta_vol = ti.field(ti.f32, shape=())
        self.phase_source_ratio = ti.field(ti.f32, shape=())

        # Shared visualization buffer.
        self.image = ti.Vector.field(3, ti.f32, shape=(self.nx, self.ny))

        self._init_all()
        if self.cfg.mode in ("fluid", "coupled") and self.cfg.phase_warmup_steps > 0:
            self._warm_start_phase(self.cfg.phase_warmup_steps)
        self._init_phase_source_ratio()
        if self.cfg.mode == "coupled":
            self._clear_mpm_grid()
            self._p2g()
            self._update_sand_grid(1)
            self._clear_fluid_force()
            self._init_reference_fluid_volume()

    def step(self, num_steps: int = 1) -> None:
        for _ in range(int(num_steps)):
            if self.cfg.mode == "fluid":
                self._fluid_step(coupled=0)
            elif self.cfg.mode == "sand":
                self._sand_step(coupled=0)
            else:
                self._coupled_step()
            self.steps += 1

    def run(
        self,
        frames: int,
        steps_per_frame: int,
        output_dir: str | Path | None = None,
        *,
        show_progress: bool = False,
    ) -> None:
        total_frames = int(frames)
        out = Path(output_dir or self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.write_metadata(out)
        gui = ti.GUI("Mixture2D Dam Break", res=(self.nx, self.ny), show_gui=self.cfg.show_gui)
        with _TerminalProgress(total_frames, label=self.cfg.mode, enabled=show_progress) as progress:
            for _ in range(total_frames):
                self.step(steps_per_frame)
                frame_path = out / f"frame_{self.frame:05d}.png"
                self.save_frame(frame_path, gui=gui if self.cfg.show_gui else None)
                if self.cfg.save_npz and self.cfg.mode in ("sand", "coupled"):
                    self.save_particles_npz(out / f"particles_{self.frame:05d}.npz")
                self.frame += 1
                progress.advance()
        self.write_metadata(out)

    def save_frame(self, path: str | Path, gui: Any | None = None) -> None:
        self._render_background()
        if self.cfg.mode in ("sand", "coupled"):
            self._render_particles()
        img = self.image.to_numpy()
        img = np.flip(np.transpose(img, (1, 0, 2)), axis=0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if gui is not None:
            gui.set_image(img)
            gui.show(str(path))
            return
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, img)
        except Exception:
            import matplotlib.pyplot as plt

            plt.imsave(path, img)

    def save_particles_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=self.p_x.to_numpy(),
            v=self.p_v.to_numpy(),
            state=self.p_state.to_numpy(),
            bound_water=self.p_bound_water.to_numpy(),
            water_content=self.p_water_content.to_numpy(),
        )

    def write_metadata(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.cfg.mode,
            "taichi_version": ".".join(map(str, ti.__version__)),
            "backend": "cuda",
            "steps": self.steps,
            "config": self.cfg.to_dict(),
        }
        (out / "metadata.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def diagnostics(self) -> dict[str, bool | float]:
        phi = self.phi.to_numpy()
        u = self.u.to_numpy()
        out: dict[str, bool | float] = {
            "fluid_finite": bool(np.isfinite(phi).all() and np.isfinite(u).all()),
            "phi_min": float(np.nanmin(phi)),
            "phi_max": float(np.nanmax(phi)),
        }
        if self.cfg.mode in ("sand", "coupled"):
            x = self.p_x.to_numpy()
            out["particles_finite"] = bool(np.isfinite(x).all())
            out["particles_inside"] = bool(
                (x[:, 0] >= 0).all() and (x[:, 0] < self.nx).all() and (x[:, 1] >= 0).all() and (x[:, 1] < self.ny).all()
            )
            out["grid_mass"] = float(self.grid_m.to_numpy().sum())
            out["particle_mass"] = float(self.n_particles * self._particle_mass())
            eps = self.epsinon.to_numpy()
            out["eps_min"] = float(np.nanmin(eps))
            out["eps_max"] = float(np.nanmax(eps))
            bw = self.p_bound_water.to_numpy()
            out["bound_water_finite"] = bool(np.isfinite(bw).all())
            out["bound_water_min"] = float(np.nanmin(bw))
            out["bound_water_max"] = float(np.nanmax(bw))
        return out

    def _particle_vol(self) -> float:
        ppc = max(1, self.cfg.particles_per_cell)
        return float(self.cfg.particle_volume_fraction / (ppc * ppc))

    def _particle_mass(self) -> float:
        return float(self._particle_vol() * self._sand_density_l)

    def _fluid_step(self, coupled: int) -> None:
        # Paper Alg. 1, LBM stages: update volume controller, assemble forces,
        # collide/stream velocity and phase distributions, then recover macro
        # variables and pressure. This version uses explicit double buffers.
        if coupled == 1:
            self._update_phase_source_ratio()
        self._compute_fluid_force(coupled, self.steps)
        self._clear_lbm_next()
        self._collide_velocity_and_stream(coupled)
        self._collide_phase_and_stream(coupled)
        self._copy_lbm_next()
        self._update_fluid_macro()
        self._update_pressure(coupled)
        self._average_pressure()

    def _warm_start_phase(self, steps: int) -> None:
        for _ in range(int(steps)):
            self._update_phi_from_h()
            self._clear_h_next()
            self._collide_phase_and_stream(0)
            self._copy_h_next()

    @ti.kernel
    def _init_phase_source_ratio(self):
        self.sum_vol0[None] = 0.0
        self.sum_vol[None] = 0.0
        self.delta_vol[None] = 0.0
        self.phase_source_ratio[None] = 1.0

    @ti.kernel
    def _init_reference_fluid_volume(self):
        # Paper Sec. 4.4, Eq. 52: record the initial fluid volume V0 for the
        # proportional source-term controller.
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                ti.atomic_add(self.sum_vol0[None], ti.cast(self.phi[i, j] * self.epsinon_src[i, j], ti.f64))

    @ti.kernel
    def _update_phase_source_ratio(self):
        # Paper Sec. 4.4, Eqs. 45 and 52. The source term compensates the
        # porosity time derivative and u dot grad(epsilon); the global ratio
        # steers the integrated fluid volume back toward V0. With retention,
        # bound water contributes to the conserved total water volume.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        self.sum_vol[None] = 0.0
        self.delta_vol[None] = 0.0
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                phi = self.phi[i, j]
                eps_src = self.epsinon_src[i, j]
                grad_eps = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    cc = ti.cast(_c(q), ti.f32)
                    ni = i + _c(q)[0]
                    nj = j + _c(q)[1]
                    eps1 = eps_src
                    if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                        eps1 = self.epsinon_src[ni, nj]
                    grad_eps += _w(q) * cc * 3.0 * (eps1 - eps_src)
                source = -phi / eps_src * (self.u[i, j].dot(grad_eps) + eps_src - self.epsinon_post[i, j])
                cell_water = phi * eps_src
                if ti.static(self.cfg.water_retention):
                    cell_water += self.grid_bound_water_post[i, j]
                ti.atomic_add(self.sum_vol[None], ti.cast(cell_water, ti.f64))
                ti.atomic_add(self.delta_vol[None], source * eps_src)

        ratio = 1.0
        sum0 = self.sum_vol0[None]
        sumv = self.sum_vol[None]
        deltav = self.delta_vol[None]
        if ti.abs(deltav) > 1.0e-9 and ti.abs(sumv - sum0) > 0.04 * sum0:
            ratio = ti.cast((sum0 - sumv) / ti.cast(deltav, ti.f64), ti.f32)
        self.phase_source_ratio[None] = ti.max(-1.0, ti.min(1.0, ratio))

    def _sand_step(self, coupled: int) -> None:
        # Paper Sec. 4.2 MPM loop: P2G (Eq. 24), grid update (Eqs. 25-29),
        # and G2P/advection (Eq. 30). In sand-only mode coupling forces vanish.
        self._clear_mpm_grid()
        self._p2g()
        self._update_sand_grid(coupled)
        self._g2p()

    def _coupled_step(self) -> None:
        # Paper Alg. 1 and reference implementation order. The LBM step uses
        # coupling quantities assembled during the previous MPM grid update;
        # the current MPM pass then writes force/retention fields for the next
        # fluid step. The delayed fluid start mirrors the reference dam-break
        # setup and is important for the water-retention case.
        if self.steps > self.cfg.coupled_fluid_start_step:
            self._fluid_step(coupled=1)
        self._clear_mpm_grid()
        self._clear_fluid_force()
        self._p2g()
        self._update_sand_grid(1)
        self._g2p()

    @ti.kernel
    def _init_all(self):
        # Dam-break initialization used for this 2D release: phase field phi is
        # a left water block, sand is a centered rectangular block, and walls
        # are axis-aligned container boundaries. Mesh SDFs from the full paper
        # implementation are intentionally omitted.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        b = ti.static(self.cfg.boundary_cells)
        water_w = ti.static(self.cfg.water_width)
        water_h = ti.static(self.cfg.water_height)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            is_wall = i < b or i >= nx - b or j < b or j >= ny - b
            self.wall[i, j] = 1 if is_wall else 0
            phi0 = 0.0
            if ti.static(self.cfg.mode != "sand"):
                if not is_wall and i < water_w and j < water_h:
                    phi0 = 1.0
            rho0 = _rho_mix(phi0, rho_water, rho_air)
            self.rho[i, j] = rho0
            self.phi[i, j] = phi0
            self.u[i, j] = ti.Vector([0.0, 0.0])
            self.u_temp[i, j] = ti.Vector([0.0, 0.0])
            self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
            self.p[i, j] = 0.0
            self.p_temp[i, j] = 0.0
            self.artificial_vis[i, j] = 0.0
            self.grid_v[i, j] = ti.Vector([0.0, 0.0])
            self.sand_grid_force[i, j] = ti.Vector([0.0, 0.0])
            self.grid_m[i, j] = 0.0
            self.delta[i, j] = 0.0
            self.grid_bound_water[i, j] = 0.0
            self.grid_bound_water_post[i, j] = 0.0
            self.grid_expand_ratio[i, j] = 0.0
            self.phi_absorbed[i, j] = 0.0
            self.epsinon[i, j] = 1.0
            self.epsinon_src[i, j] = 1.0
            self.epsinon_post[i, j] = 1.0
            for q in range(Q):
                self.f[i, j, q] = _feq(q, 1.0, ti.Vector([0.0, 0.0]))
                self.f_next[i, j, q] = self.f[i, j, q]
                self.h[i, j, q] = _heq(q, phi0, ti.Vector([0.0, 0.0]))
                self.h_next[i, j, q] = self.h[i, j, q]

        cells_x = ti.static(max(1, self.cfg.sand_width))
        cells_y = ti.static(max(1, self.cfg.sand_height))
        particle_res_x = ti.static(max(1, self.cfg.sand_width) * max(1, self.cfg.particles_per_cell))
        particle_res_y = ti.static(max(1, self.cfg.sand_height) * max(1, self.cfg.particles_per_cell))
        x0 = ti.static(0.5 * self.nx - 0.5 * cells_x)
        y0 = ti.static(self.cfg.sand_base_y_cells)
        for p in self.p_x:
            px = p % particle_res_x
            py = p // particle_res_x
            self.p_x[p] = ti.Vector([
                ti.cast(x0, ti.f32) + ti.cast(px, ti.f32) / ti.cast(particle_res_x, ti.f32) * ti.cast(cells_x, ti.f32),
                ti.cast(y0, ti.f32) + ti.cast(py, ti.f32) / ti.cast(particle_res_y, ti.f32) * ti.cast(cells_y, ti.f32),
            ])
            self.p_v[p] = ti.Vector([0.0, 0.0])
            self.p_C[p] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            self.p_F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            self.p_q[p] = 0.0
            self.p_vcs[p] = 0.0
            self.p_alpha[p] = self.cfg.sand_initial_alpha
            self.p_bound_water[p] = 0.0
            self.p_water_content[p] = 0.0
            self.p_state[p] = 0

    @ti.kernel
    def _clear_lbm_next(self):
        for i, j, q in self.f_next:
            self.f_next[i, j, q] = 0.0
            self.h_next[i, j, q] = 0.0

    @ti.kernel
    def _clear_h_next(self):
        for i, j, q in self.h_next:
            self.h_next[i, j, q] = 0.0

    @ti.kernel
    def _collide_velocity_and_stream(self, coupled: int):
        # Paper Sec. 4.1, Eqs. 8-11 and 13-15. This is the velocity-based LBM
        # collision/streaming step in central-moment form. The paper uses
        # NOCM-MRT on D3Q27; here we keep the same moment-space idea on D2Q9.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nu_water = ti.static(self._nu_water_l)
        nu_air = ti.static(self._nu_air_l)
        for i, j in self.rho:
            if self.wall[i, j] == 0:
                phi = self.phi[i, j]
                u = self.u_temp[i, j]
                force = self.fluid_force[i, j]
                tau_local = _tau_mix(phi, self.artificial_vis[i, j], rho_water, rho_air, nu_water, nu_air)

                m00 = 0.0
                m10 = 0.0
                m01 = 0.0
                m20 = 0.0
                m02 = 0.0
                m11 = 0.0
                m21 = 0.0
                m12 = 0.0
                m22 = 0.0
                for q in range(Q):
                    cx = ti.cast(_c(q)[0], ti.f32)
                    cy = ti.cast(_c(q)[1], ti.f32)
                    fval = self.f[i, j, q]
                    cx2 = cx * cx
                    cy2 = cy * cy
                    m00 += fval
                    m10 += fval * cx
                    m01 += fval * cy
                    m20 += fval * cx2
                    m02 += fval * cy2
                    m11 += fval * cx * cy
                    m21 += fval * cx2 * cy
                    m12 += fval * cx * cy2
                    m22 += fval * cx2 * cy2

                ux = u.x
                uy = u.y
                ux2 = ux * ux
                uy2 = uy * uy
                kf20 = m20 - 2.0 * ux * m10 + ux2 * m00
                kf02 = m02 - 2.0 * uy * m01 + uy2 * m00
                kf11 = m11 - ux * m01 - uy * m10 + ux * uy * m00
                bf4 = kf20 - kf02
                bf5 = kf11
                s_shear = 1.0 / tau_local
                bf4 = bf4 - s_shear * bf4
                bf5 = bf5 - s_shear * bf5
                kf00 = 1.0
                kf10 = 0.5 * force.x
                kf01 = 0.5 * force.y
                kf20 = 0.5 * (2.0 / 3.0 + bf4)
                kf02 = 0.5 * (2.0 / 3.0 - bf4)
                kf11 = bf5
                kf21 = 0.5 * force.y / 3.0
                kf12 = 0.5 * force.x / 3.0
                kf22 = 1.0 / 9.0

                for q in range(Q):
                    cx_i = _c(q)[0]
                    cy_i = _c(q)[1]
                    f_post = _reconstruct_central(cx_i, cy_i, ux, uy, kf00, kf10, kf01, kf20, kf02, kf11, kf21, kf12, kf22)
                    ni = i + _c(q)[0]
                    nj = j + _c(q)[1]
                    if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                        self.f_next[ni, nj, q] = f_post
                    else:
                        iq = _opp(q)
                        if iq < q:
                            eq = _feq(iq, 1.0, ti.Vector([0.0, 0.0]))
                            self.f_next[i, j, iq] = 0.9 * f_post + 0.1 * eq
                        else:
                            self.f_next[i, j, iq] = f_post

    @ti.kernel
    def _collide_phase_and_stream(self, coupled: int):
        # Paper Sec. 3.1 Eq. 5 and Sec. 4.4 Eqs. 46-50. The Allen-Cahn-like
        # phase LBM transports phi and applies interface compression. In
        # coupled mode, source follows Eq. 45; with retention it subtracts the
        # absorbed free water from Eq. 57.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        interface_width = ti.static(self.cfg.interface_width)
        tau_inv_phi = ti.static(1.0 / (3.0 * self.cfg.mobility + 0.5))
        for i, j in self.phi:
            if self.wall[i, j] == 0:
                phi = self.phi[i, j]
                u = self.u_temp[i, j]
                grad_phi = ti.Vector([0.0, 0.0])
                alpha = ti.static(1.0 / 3.0)
                for qq in range(Q):
                    cc = ti.cast(_c(qq), ti.f32)
                    ni1 = i + _c(qq)[0]
                    nj1 = j + _c(qq)[1]
                    ni2 = i + 2 * _c(qq)[0]
                    nj2 = j + 2 * _c(qq)[1]
                    phi1 = phi
                    phi2 = phi
                    if _inside(ni1, nj1, nx, ny) and self.wall[ni1, nj1] == 0:
                        phi1 = self.phi[ni1, nj1]
                    if _inside(ni2, nj2, nx, ny) and self.wall[ni2, nj2] == 0:
                        phi2 = self.phi[ni2, nj2]
                    grad_phi += (1.0 - alpha) * 3.0 * _w(qq) * cc * (phi1 - phi)
                    grad_phi += alpha * 1.5 * _w(qq) * cc * (4.0 * phi1 - phi2)
                normal = ti.Vector([0.0, 0.0])
                grad_norm = grad_phi.norm()
                normal = grad_phi / (grad_norm + 1.0e-14)
                compression = 4.0 * phi * (1.0 - phi) / interface_width
                source = 0.0
                if coupled == 1:
                    eps_src = self.epsinon_src[i, j]
                    grad_eps = ti.Vector([0.0, 0.0])
                    for qq in range(Q):
                        cc = ti.cast(_c(qq), ti.f32)
                        ni = i + _c(qq)[0]
                        nj = j + _c(qq)[1]
                        eps1 = eps_src
                        if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                            eps1 = self.epsinon_src[ni, nj]
                        grad_eps += _w(qq) * cc * 3.0 * (eps1 - eps_src)
                    time_deviation = eps_src - self.epsinon_post[i, j]
                    source = -phi / eps_src * (self.u[i, j].dot(grad_eps) + time_deviation)
                    source *= self.phase_source_ratio[None]
                    if ti.static(self.cfg.water_retention):
                        water_absorbed = self.phi_absorbed[i, j] / eps_src
                        max_l = ti.max(0.0, phi + source)
                        water_absorbed = ti.max(0.0, ti.min(max_l, water_absorbed))
                        source -= water_absorbed

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
                hs00 = 0.0
                hs10 = 0.0
                hs01 = 0.0
                hs20 = 0.0
                hs02 = 0.0
                hs11 = 0.0
                hs21 = 0.0
                hs12 = 0.0
                hs22 = 0.0
                for q in range(Q):
                    cx = ti.cast(_c(q)[0], ti.f32)
                    cy = ti.cast(_c(q)[1], ti.f32)
                    rx = cx - u.x
                    ry = cy - u.y
                    rx2 = rx * rx
                    ry2 = ry * ry
                    cq = ti.Vector([cx, cy])
                    hval = self.h[i, j, q]
                    h_eq = _heq(q, phi, u)
                    phi_force_q = _w(q) * compression * cq.dot(normal)
                    source_q = _w(q) * source
                    hm00 += hval
                    hm10 += hval * rx
                    hm01 += hval * ry
                    hm20 += hval * rx2
                    hm02 += hval * ry2
                    hm11 += hval * rx * ry
                    hm21 += hval * rx2 * ry
                    hm12 += hval * rx * ry2
                    hm22 += hval * rx2 * ry2
                    he00 += h_eq
                    he10 += h_eq * rx
                    he01 += h_eq * ry
                    he20 += h_eq * rx2
                    he02 += h_eq * ry2
                    he11 += h_eq * rx * ry
                    he21 += h_eq * rx2 * ry
                    he12 += h_eq * rx * ry2
                    he22 += h_eq * rx2 * ry2
                    hf00 += phi_force_q
                    hf10 += phi_force_q * rx
                    hf01 += phi_force_q * ry
                    hf20 += phi_force_q * rx2
                    hf02 += phi_force_q * ry2
                    hf11 += phi_force_q * rx * ry
                    hf21 += phi_force_q * rx2 * ry
                    hf12 += phi_force_q * rx * ry2
                    hf22 += phi_force_q * rx2 * ry2
                    hs00 += source_q
                    hs10 += source_q * rx
                    hs01 += source_q * ry
                    hs20 += source_q * rx2
                    hs02 += source_q * ry2
                    hs11 += source_q * rx * ry
                    hs21 += source_q * rx2 * ry
                    hs12 += source_q * rx * ry2
                    hs22 += source_q * rx2 * ry2

                kh00 = he00 + 0.5 * hf00 + hs00
                kh10 = hm10 - tau_inv_phi * (hm10 - he10) + (1.0 - 0.5 * tau_inv_phi) * hf10 + hs10
                kh01 = hm01 - tau_inv_phi * (hm01 - he01) + (1.0 - 0.5 * tau_inv_phi) * hf01 + hs01
                kh20 = he20 + 0.5 * hf20 + hs20
                kh02 = he02 + 0.5 * hf02 + hs02
                kh11 = he11 + 0.5 * hf11 + hs11
                kh21 = he21 + 0.5 * hf21 + hs21
                kh12 = he12 + 0.5 * hf12 + hs12
                kh22 = he22 + 0.5 * hf22 + hs22

                for q in range(Q):
                    cx_i = _c(q)[0]
                    cy_i = _c(q)[1]
                    h_post = _reconstruct_central(cx_i, cy_i, u.x, u.y, kh00, kh10, kh01, kh20, kh02, kh11, kh21, kh12, kh22)
                    ni = i + _c(q)[0]
                    nj = j + _c(q)[1]
                    if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                        self.h_next[ni, nj, q] = h_post
                    else:
                        self.h_next[i, j, _opp(q)] = h_post

    @ti.kernel
    def _clear_fluid_force(self):
        for i, j in self.fluid_force:
            self.fluid_force[i, j] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def _clear_mpm_grid(self):
        for i, j in self.grid_m:
            self.grid_m[i, j] = 0.0
            self.grid_v[i, j] = ti.Vector([0.0, 0.0])
            self.sand_grid_force[i, j] = ti.Vector([0.0, 0.0])
            self.delta[i, j] = 0.0
            self.grid_bound_water[i, j] = 0.0
            self.grid_bound_water_post[i, j] = 0.0
            self.grid_expand_ratio[i, j] = 0.0
            self.phi_absorbed[i, j] = 0.0

    @ti.kernel
    def _update_fluid_macro(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            if self.wall[i, j] == 1:
                self.phi[i, j] = 0.0
                self.rho[i, j] = rho_air
                self.u[i, j] = ti.Vector([0.0, 0.0])
                self.u_temp[i, j] = ti.Vector([0.0, 0.0])
            else:
                phi = 0.0
                mom = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    hq = self.h[i, j, q]
                    fq = self.f[i, j, q]
                    phi += hq
                    mom += ti.cast(_c(q), ti.f32) * fq
                self.phi[i, j] = phi
                self.rho[i, j] = _rho_mix(phi, rho_water, rho_air)
                self.u[i, j] = mom

    @ti.kernel
    def _update_pressure(self, coupled: int):
        # Paper Sec. 4.3, pressure update from mixture volume conservation
        # (Eqs. 32-40). The p1/p2/p3 terms account for porosity gradients,
        # sand-grid velocity divergence, and relative sand/fluid motion.
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.p:
            if self.wall[i, j] == 1:
                self.p[i, j] = 0.0
            else:
                f_sum = 0.0
                for q in range(Q):
                    f_sum += self.f[i, j, q]
                rho = _rho_mix(self.phi[i, j], rho_water, rho_air)
                p_base = self.p_temp[i, j]
                e = 1.0
                if coupled == 1:
                    if ti.static(self.cfg.water_retention):
                        e = self.epsinon_src[i, j]
                    else:
                        e = self.epsinon[i, j]
                    delta = 1.0 - self.epsinon[i, j]
                    de = ti.Vector([0.0, 0.0])
                    us = self.grid_v[i, j]
                    thresh = 0.3
                    us_norm = ti.sqrt(us.dot(us))
                    if us_norm > thresh:
                        us = us / us_norm * thresh
                    div_us = 0.0
                    for qq in range(Q):
                        cc = ti.cast(_c(qq), ti.f32)
                        ni = i + _c(qq)[0]
                        nj = j + _c(qq)[1]
                        e1 = 1.0
                        us1 = ti.Vector([0.0, 0.0])
                        if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                            if ti.static(self.cfg.water_retention):
                                e1 = self.epsinon_src[ni, nj]
                            else:
                                e1 = self.epsinon[ni, nj]
                            us1 = self.grid_v[ni, nj]
                        us1_norm = ti.sqrt(us1.dot(us1))
                        if us1_norm > thresh:
                            us1 = us1 / us1_norm * thresh
                        diff_us = us1 - us
                        de += _w(qq) * cc * 3.0 * (e1 - e)
                        div_us += _w(qq) * 3.0 * (cc.x * diff_us.x + cc.y * diff_us.y)
                    p0 = p_base + rho / 3.0 * (f_sum - 1.0)
                    p1 = -rho / 3.0 * self.u[i, j].dot(de) / e
                    p2 = -rho / 3.0 * delta * div_us / e
                    p3 = rho / 3.0 * us.dot(de) / e
                    self.p[i, j] = p0 + p1 + p2 + p3
                else:
                    self.p[i, j] = p_base + rho / 3.0 * (f_sum - 1.0)

    @ti.kernel
    def _average_pressure(self):
        # Paper Sec. 4.3 Eq. 40: pressure filter after the explicit pressure
        # update. The D2Q9 weights are the 2D counterpart of the paper lattice.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.p_temp:
            if self.wall[i, j] == 1:
                self.p_temp[i, j] = 0.0
            else:
                pressure = 0.0
                for q in range(Q):
                    ni = i + _c(q)[0]
                    nj = j + _c(q)[1]
                    p1 = pressure
                    if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                        p1 = self.p[ni, nj]
                    pressure += _w(q) * p1
                self.p_temp[i, j] = pressure

    @ti.kernel
    def _compute_fluid_force(self, coupled: int, steps: int):
        # Paper Sec. 4.1 Eqs. 15-23: force assembly for the velocity-based LBM,
        # including viscosity-gradient, pressure, gravity/body force, surface
        # tension/chemical-potential terms, plus the sand reaction force from
        # Sec. 4.3 when coupled.
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        grav = ti.static(self._gravity_l)
        beta = ti.static(12.0 * self._sigma_l / self.cfg.interface_width)
        kapa = ti.static(1.5 * self._sigma_l * self.cfg.interface_width)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        nu_water = ti.static(self._nu_water_l)
        nu_air = ti.static(self._nu_air_l)
        delta_rho = ti.static(self._rho_water_l - self._rho_air_l)
        cd = ti.static(self.cfg.cd)
        for i, j in self.rho:
            if self.wall[i, j] == 1:
                self.fluid_force[i, j] = ti.Vector([0.0, 0.0])
                self.artificial_vis[i, j] = 0.0
                self.u_temp[i, j] = ti.Vector([0.0, 0.0])
            else:
                phi = self.phi[i, j]
                phi_force = ti.min(1.0, ti.max(0.0, phi))
                rho = _rho_mix(phi_force, rho_water, rho_air)

                grad = ti.Vector([0.0, 0.0])
                lap = 0.0
                alpha = ti.static(1.0 / 3.0)
                for qq in range(Q):
                    cc = ti.cast(_c(qq), ti.f32)
                    ni1 = i + _c(qq)[0]
                    nj1 = j + _c(qq)[1]
                    ni2 = i + 2 * _c(qq)[0]
                    nj2 = j + 2 * _c(qq)[1]
                    phi1 = phi
                    phi2 = phi
                    if _inside(ni1, nj1, nx, ny) and self.wall[ni1, nj1] == 0:
                        phi1 = self.phi[ni1, nj1]
                    if _inside(ni2, nj2, nx, ny) and self.wall[ni2, nj2] == 0:
                        phi2 = self.phi[ni2, nj2]
                    grad += (1.0 - alpha) * 3.0 * _w(qq) * cc * (phi1 - phi)
                    grad += alpha * 1.5 * _w(qq) * cc * (4.0 * phi1 - phi2)
                    lap += 6.0 * _w(qq) * (phi1 - phi)

                force = ti.Vector([0.0, -grav])
                capillary = 4.0 * beta * phi_force * (phi_force - 1.0) * (phi_force - 0.5) - kapa * lap
                force += capillary * grad / rho

                u0 = self.u[i, j]
                pressure = self.p_temp[i, j]
                dux = 0.0
                duy = 0.0
                dvx = 0.0
                dvy = 0.0
                eps = 1.0
                if coupled == 1:
                    if ti.static(self.cfg.water_retention):
                        eps = self.epsinon_src[i, j]
                    else:
                        eps = self.epsinon[i, j]
                grad_eps = ti.Vector([0.0, 0.0])
                grad_p = ti.Vector([0.0, 0.0])
                for q in range(Q):
                    cc = ti.cast(_c(q), ti.f32)
                    ni = i + _c(q)[0]
                    nj = j + _c(q)[1]
                    mx = 3.0 * _w(q) * cc.x
                    my = 3.0 * _w(q) * cc.y
                    u1 = ti.Vector([0.0, 0.0])
                    p1 = pressure
                    if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                        u1 = self.u[ni, nj]
                        p1 = self.p_temp[ni, nj]
                    diff_u = u1 - u0
                    dux += mx * diff_u.x
                    duy += my * diff_u.x
                    dvx += mx * diff_u.y
                    dvy += my * diff_u.y
                    if coupled == 1:
                        eps1 = eps
                        if _inside(ni, nj, nx, ny) and self.wall[ni, nj] == 0:
                            if ti.static(self.cfg.water_retention):
                                eps1 = self.epsinon_src[ni, nj]
                            else:
                                eps1 = self.epsinon[ni, nj]
                        grad_eps += _w(q) * cc * 3.0 * (eps1 - eps)
                    grad_p += _w(q) * cc * 3.0 * (p1 - pressure)

                art_vis = cd * cd * ti.sqrt(2.0 * (dux * dux + dvy * dvy + 0.5 * (duy + dvx) * (duy + dvx)))
                vis_k = _viscosity_mix(phi_force, rho_water, rho_air, nu_water, nu_air)
                vis_eff = vis_k + art_vis
                eps_safe = ti.max(eps, 1.0e-6)
                grad_rho = delta_rho * grad
                strain_xy = duy + dvx
                force.x += vis_eff * (
                    (2.0 * dux * grad_rho.x + strain_xy * grad_rho.y) / rho
                    + (2.0 * dux * grad_eps.x + strain_xy * grad_eps.y) / eps_safe
                )
                force.y += vis_eff * (
                    (strain_xy * grad_rho.x + 2.0 * dvy * grad_rho.y) / rho
                    + (strain_xy * grad_eps.x + 2.0 * dvy * grad_eps.y) / eps_safe
                )
                if phi_force < 0.1 and steps < 120 * 600:
                    grad_p *= 0.1
                force -= grad_p / rho

                fa = ti.Vector([0.0, 0.0])
                for a in range(4):
                    axis = _axis_c(a)
                    nb_i = i + axis[0]
                    nb_j = j + axis[1]
                    diff = ti.Vector([0.0, 0.0])
                    u_abs = 0.0
                    if _inside(nb_i, nb_j, nx, ny) and self.wall[nb_i, nb_j] == 0:
                        u1 = self.u[nb_i, nb_j]
                        if a == 0 or a == 1:
                            diff = u1 - u0
                        else:
                            diff = u0 - u1
                        if a == 0 or a == 2:
                            u_abs = ti.abs(u0.x + u1.x) * 0.5
                        else:
                            u_abs = ti.abs(u0.y + u1.y) * 0.5
                        if u_abs / (vis_k + vis_k) < 4.0:
                            diff = ti.Vector([0.0, 0.0])
                    sign = 1.0
                    if a >= 2:
                        sign = -1.0
                    fa += sign * diff * u_abs
                force += 0.5 * fa

                if coupled == 1:
                    eps = self.epsinon[i, j]
                    rho_e = eps * rho
                    force += -self.fluid_force[i, j] / rho_e

                self.u_temp[i, j] = u0 + 0.5 * force
                self.artificial_vis[i, j] = art_vis
                self.fluid_force[i, j] = force

    @ti.kernel
    def _update_phi_from_h(self):
        for i, j in self.phi:
            if self.wall[i, j] == 1:
                self.phi[i, j] = 0.0
            else:
                phi = 0.0
                for q in range(Q):
                    phi += self.h[i, j, q]
                self.phi[i, j] = phi

    @ti.kernel
    def _copy_h_next(self):
        for i, j in self.phi:
            if self.wall[i, j] == 1:
                self.phi[i, j] = 0.0
                for q in range(Q):
                    self.h[i, j, q] = _heq(q, 0.0, ti.Vector([0.0, 0.0]))
            else:
                phi = 0.0
                for q in range(Q):
                    self.h[i, j, q] = self.h_next[i, j, q]
                    phi += self.h_next[i, j, q]
                self.phi[i, j] = phi

    @ti.kernel
    def _copy_lbm_next(self):
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        for i, j in self.rho:
            if self.wall[i, j] == 1:
                phi0 = 0.0
                rho0 = rho_air
                for q in range(Q):
                    self.f[i, j, q] = _feq(q, 1.0, ti.Vector([0.0, 0.0]))
                    self.h[i, j, q] = _heq(q, phi0, ti.Vector([0.0, 0.0]))
            else:
                for q in range(Q):
                    self.f[i, j, q] = self.f_next[i, j, q]
                    self.h[i, j, q] = self.h_next[i, j, q]
                phi = 0.0
                for q in range(Q):
                    phi += self.h[i, j, q]
                self.phi[i, j] = phi
                self.rho[i, j] = _rho_mix(self.phi[i, j], rho_water, rho_air)

    @ti.kernel
    def _p2g(self):
        # Paper Sec. 4.2 Eq. 24: APIC particle-to-grid transfer of mass and
        # momentum with quadratic B-splines. The stress contribution implements
        # the elastic force from Eq. 27. With water retention, particle bound
        # water is rasterized to the grid as in Eq. 53.
        p_mass = ti.static(self._particle_mass())
        p_vol = ti.static(self._particle_vol())
        mu = ti.static(self._sand_mu_l)
        la = ti.static(self._sand_lambda_l)
        for p in self.p_x:
            Xp = self.p_x[p]
            base = ti.cast(Xp - 0.5, ti.i32)
            if base.x >= 0 and base.x < ti.static(self.nx - 2) and base.y >= 0 and base.y < ti.static(self.ny - 2):
                fx = Xp - ti.cast(base, ti.f32)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                F = self.p_F[p]
                U, sig, V = ti.svd(F)
                s0 = sig[0, 0]
                s1 = sig[1, 1]
                log0 = ti.log(s0)
                log1 = ti.log(s1)
                trace_log = log0 + log1
                sig_inv_log = ti.Matrix([
                    [(2.0 * mu * log0 + la * trace_log) / s0, 0.0],
                    [0.0, (2.0 * mu * log1 + la * trace_log) / s1],
                ])
                stress = U @ sig_inv_log @ V.transpose()
                stress = -4.0 * p_vol * stress @ F.transpose()
                for a, b in ti.static(ti.ndrange(3, 3)):
                    offset = ti.Vector([a, b])
                    node = base + offset
                    weight = w[a][0] * w[b][1]
                    dpos = ti.cast(offset, ti.f32) - fx
                    grid_force = stress @ dpos
                    affine_momentum = p_mass * self.p_C[p] @ dpos
                    ti.atomic_add(self.grid_m[node[0], node[1]], weight * p_mass)
                    ti.atomic_add(self.grid_bound_water[node[0], node[1]], weight * self.p_bound_water[p])
                    ti.atomic_add(self.grid_v[node[0], node[1]], weight * (p_mass * self.p_v[p] + affine_momentum))
                    ti.atomic_add(self.sand_grid_force[node[0], node[1]], weight * grid_force)

    @ti.kernel
    def _update_sand_grid(self, coupled: int):
        # Paper Sec. 4.2-4.3: compute sediment volume fraction (Eq. 25),
        # update sand grid momentum (Eqs. 26 and 29), add drag and buoyancy
        # (Sec. 4.3, Eq. 31), and write the opposite drag density for the fluid.
        # The retention block below follows Sec. 4.5, Eqs. 54-56.
        dt = ti.static(self._mpm_dt_l)
        gy = ti.static(self._sand_gravity_l)
        rho_s = ti.static(self._sand_density_l)
        grav_l = ti.static(self._gravity_l)
        rho_water = ti.static(self._rho_water_l)
        rho_air = ti.static(self._rho_air_l)
        delta_max = ti.static(self.cfg.delta_max)
        ratio_max = ti.static(self.cfg.ratio_max)
        ci = ti.static(self.cfg.ci)
        chi = ti.static(self.cfg.chi)
        bnd = ti.static(self.cfg.boundary_cells)
        mu_wall = ti.static(self.cfg.wall_friction)
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for i, j in self.grid_m:
            m = self.grid_m[i, j]
            if m > 0.0:
                vel = self.grid_v[i, j] / m
                force = self.sand_grid_force[i, j]
                vol = m / rho_s
                delta_raw = vol
                self.delta[i, j] = delta_raw
                delta = ti.min(delta_raw, delta_max)
                eps = 1.0 - delta
                old_eps = self.epsinon[i, j]
                self.epsinon[i, j] = eps
                if ti.static(self.cfg.water_retention):
                    # Paper Sec. 4.5. bw_old is grid bound-water fraction R_i
                    # from Eq. 53; eps_src is effective porosity epsilon-hat
                    # from Eq. 54; water_should_absorbed is Delta R_i from
                    # Eq. 55; bw_new applies Eq. 56.
                    bw_old = self.grid_bound_water[i, j]
                    cap_volume = delta * ratio_max
                    bw_eff_old = ti.min(bw_old, cap_volume)
                    av = ti.max(0.0, cap_volume - bw_eff_old)
                    eps_src = eps - bw_eff_old
                    phi_f = ti.max(0.0, ti.min(1.0, self.phi[i, j]))
                    avail_water = ti.max(0.0, eps_src) * phi_f
                    rate_cap = 3.0e-3 * cap_volume
                    water_should_absorbed = ti.max(0.0, ti.min(ti.min(av, rate_cap), avail_water))
                    bw_new = bw_old + water_should_absorbed
                    bw_eff_new = ti.min(bw_new, cap_volume)
                    eps_src_new = eps - bw_eff_new
                    self.grid_bound_water_post[i, j] = bw_new
                    self.epsinon_post[i, j] = self.epsinon_src[i, j]
                    self.epsinon_src[i, j] = eps_src_new
                    self.phi_absorbed[i, j] = water_should_absorbed
                    self.grid_expand_ratio[i, j] = bw_eff_new / ratio_max / delta
                else:
                    self.epsinon_post[i, j] = old_eps
                    self.epsinon_src[i, j] = eps

                fd = ti.Vector([0.0, 0.0])
                rhof = 0.0
                if coupled == 1:
                    phi = ti.max(0.0, ti.min(1.0, self.phi[i, j]))
                    rhof = _rho_mix(phi, rho_water, rho_air)
                    rel = self.u[i, j] - vel
                    rel_norm = ti.sqrt(rel.dot(rel))
                    fd = rhof * rel_norm * rel
                    area = 2.0 * ti.sqrt(vol / math.pi)
                    fd *= 0.5 * ci * area * ti.pow(eps, -chi)

                    force_l = fd
                    force_norm = ti.sqrt(force_l.dot(force_l))
                    rho_e = eps * _rho_mix(self.phi[i, j], rho_water, rho_air)
                    thresh_force = 1.0e4 * grav_l * rho_e
                    fluid_force_l = force_l
                    if force_norm > thresh_force:
                        fluid_force_l = force_l * thresh_force / force_norm
                        fd *= ti.sqrt(fluid_force_l.dot(fluid_force_l)) / force_norm
                    self.fluid_force[i, j] += fluid_force_l

                vel += dt * force / ti.max(m, 1.0e-6)
                if coupled == 1:
                    vel += dt * (fd + ti.Vector([0.0, gy]) * (m - vol * rhof)) / ti.max(m, 1.0e-6)
                else:
                    vel.y += dt * gy
                normal = ti.Vector([0.0, 0.0])
                if i < bnd and vel.x < 0.0:
                    normal = ti.Vector([1.0, 0.0])
                if i >= nx - bnd and vel.x > 0.0:
                    normal = ti.Vector([-1.0, 0.0])
                if j < bnd and vel.y < 0.0:
                    normal = ti.Vector([0.0, 1.0])
                if j >= ny - bnd and vel.y > 0.0:
                    normal = ti.Vector([0.0, -1.0])
                if normal.x != 0.0 or normal.y != 0.0:
                    s = normal.dot(vel)
                    if s <= 0.0:
                        v_normal = s * normal
                        v_tangent = vel - v_normal
                        vt_norm = ti.sqrt(v_tangent.dot(v_tangent))
                        if vt_norm > 1.0e-12:
                            vel = v_tangent - ti.min(vt_norm, -mu_wall * s) * v_tangent / vt_norm
                        else:
                            vel = ti.Vector([0.0, 0.0])
                self.grid_v[i, j] = vel
            else:
                self.sand_grid_force[i, j] = ti.Vector([0.0, 0.0])
                self.fluid_force[i, j] += ti.Vector([0.0, 0.0])
                self.epsinon[i, j] = 1.0
                self.grid_expand_ratio[i, j] = 0.0
                self.epsinon_post[i, j] = self.epsinon_src[i, j]
                self.epsinon_src[i, j] = 1.0

    @ti.kernel
    def _g2p(self):
        # Paper Sec. 4.2 Eq. 30: gather grid velocity back to particles and
        # advect them. The deformation gradient is projected with the
        # saturation-dependent Drucker-Prager model (Eq. 28). Retained water is
        # transferred back to particles during G2P. In retention mode, cohesion
        # follows the configured piecewise curve evaluated at the free-plus-bound
        # water fraction from Eq. 59.
        dt = ti.static(self._mpm_dt_l)
        mu = ti.static(self._sand_mu_l)
        la = ti.static(self._sand_lambda_l)
        cohesion = ti.static(self.cfg.sand_cohesion_factor)
        cohesion_c0 = ti.static(self.cfg.retention_cohesion_c[0])
        cohesion_c1 = ti.static(self.cfg.retention_cohesion_c[1])
        cohesion_c2 = ti.static(self.cfg.retention_cohesion_c[2])
        cohesion_phi0 = ti.static(self.cfg.retention_cohesion_phi[0])
        cohesion_phi1 = ti.static(self.cfg.retention_cohesion_phi[1])
        cohesion_phi2 = ti.static(self.cfg.retention_cohesion_phi[2])
        delta_max = ti.static(self.cfg.delta_max)
        p_vol_dim = ti.static(self._particle_vol())
        nx = ti.static(self.nx)
        ny = ti.static(self.ny)
        for p in self.p_x:
            Xp = self.p_x[p]
            base = ti.cast(Xp - 0.5, ti.i32)
            if base.x >= 0 and base.x < nx - 2 and base.y >= 0 and base.y < ny - 2:
                fx = Xp - ti.cast(base, ti.f32)
                w = [
                    0.5 * (1.5 - fx) ** 2,
                    0.75 - (fx - 1.0) ** 2,
                    0.5 * (fx - 0.5) ** 2,
                ]
                new_v = ti.Vector([0.0, 0.0])
                new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                epsinon_p = 0.0
                epsinon_src_p = 0.0
                diff_bound_water = 0.0
                r_cur = 0.0
                for a, b in ti.static(ti.ndrange(3, 3)):
                    offset = ti.Vector([a, b])
                    node = base + offset
                    weight = w[a][0] * w[b][1]
                    g_v = self.grid_v[node[0], node[1]]
                    dpos = ti.cast(offset, ti.f32) - fx
                    new_v += weight * g_v
                    new_C += 4.0 * weight * g_v.outer_product(dpos)
                    phi_j = ti.max(0.0, ti.min(1.0, self.phi[node[0], node[1]]))
                    epsinon_p += weight * self.epsinon[node[0], node[1]] * phi_j
                    if ti.static(self.cfg.water_retention):
                        epsinon_src_p += weight * self.epsinon_src[node[0], node[1]] * phi_j
                        old_bound_water = self.grid_bound_water[node[0], node[1]]
                        post_bound_water = self.grid_bound_water_post[node[0], node[1]]
                        delta_j = self.delta[node[0], node[1]]
                        diff_bound_water += weight * (post_bound_water - old_bound_water) * p_vol_dim / ti.max(1.0e-10, delta_j)
                        r_cur += weight * post_bound_water

                Xp += dt * new_v
                if ti.static(self.cfg.water_retention):
                    new_bound_water = self.p_bound_water[p] + diff_bound_water
                    cap = ti.static(self.cfg.ratio_max) * p_vol_dim
                    new_bound_water = ti.max(0.0, ti.min(cap, new_bound_water))
                    self.p_bound_water[p] = new_bound_water
                    self.p_water_content[p] = new_bound_water
                else:
                    self.p_water_content[p] = epsinon_p * p_vol_dim

                F = (ti.Matrix.identity(ti.f32, 2) + dt * new_C) @ self.p_F[p]
                elastic_F = F
                U, sig, V = ti.svd(F)
                e0 = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
                e = e0 + 0.5 * self.p_vcs[p] * ti.Vector([1.0, 1.0])
                cohesion_strength = cohesion
                if ti.static(self.cfg.water_retention):
                    phi_s = ti.max(0.0, ti.min(cohesion_phi2, epsinon_src_p + r_cur))
                    if phi_s < cohesion_phi0:
                        cohesion_strength = cohesion_c0 + (cohesion_c1 - cohesion_c0) * phi_s / cohesion_phi0
                    elif phi_s < cohesion_phi1:
                        cohesion_strength = cohesion_c1 + (cohesion_c2 - cohesion_c1) * (phi_s - cohesion_phi0) / (cohesion_phi1 - cohesion_phi0)
                    else:
                        cohesion_strength = cohesion_c2 - cohesion_c2 * (phi_s - cohesion_phi1) / (cohesion_phi2 - cohesion_phi1)
                else:
                    phi_s = 1.0 - epsinon_p / (1.0 - delta_max)
                    phi_s = ti.max(0.0, ti.min(1.0, phi_s))
                    cohesion_strength = cohesion * phi_s
                e += (-cohesion_strength) / (2.0 * self.p_alpha[p]) * ti.Vector([1.0, 1.0])
                tr = e.x + e.y
                ehat = e - 0.5 * tr * ti.Vector([1.0, 1.0])
                norm_e = ti.sqrt(ehat.dot(ehat))
                coeff = (2.0 * la + 2.0 * mu) / (2.0 * mu)
                delta_gamma = norm_e + coeff * tr * self.p_alpha[p]
                e_new = e0
                delta_q = 0.0
                state = 0
                if norm_e <= 0.0 or tr > 0.0:
                    e_new = ti.Vector([0.0, 0.0])
                    delta_q = ti.sqrt(e.dot(e))
                    state = 1
                elif delta_gamma <= 0.0:
                    e_new = e0
                    state = 0
                else:
                    e_new = e - delta_gamma / norm_e * ehat
                    delta_q = delta_gamma
                    state = 2
                sig_new = ti.Matrix([[ti.exp(e_new.x), 0.0], [0.0, ti.exp(e_new.y)]])
                new_F = U @ sig_new @ V.transpose()
                self.p_F[p] = new_F
                self.p_vcs[p] += -ti.log(new_F.determinant()) + ti.log(elastic_F.determinant())
                q = self.p_q[p] + delta_q
                self.p_q[p] = q
                phi_angle = 35.0 + (9.0 * q - 10.0) * ti.exp(-0.2 * q)
                phi_angle = phi_angle / 180.0 * math.pi
                sin_phi = ti.sin(phi_angle)
                self.p_alpha[p] = ti.sqrt(2.0 / 3.0) * (2.0 * sin_phi) / (3.0 - sin_phi)
                self.p_x[p] = Xp
                self.p_v[p] = new_v
                self.p_C[p] = new_C
                self.p_state[p] = state

    @ti.kernel
    def _render_background(self):
        for i, j in self.image:
            phi = ti.min(1.0, ti.max(0.0, self.phi[i, j]))
            air = ti.Vector([0.97, 0.98, 0.98])
            water = ti.Vector([0.08, 0.42, 0.90])
            sand_bg = ti.Vector([0.98, 0.96, 0.90])
            col = air * (1.0 - phi) + water * phi
            if ti.static(self.cfg.mode == "sand"):
                col = sand_bg
            if self.wall[i, j] == 1:
                col = ti.Vector([0.12, 0.13, 0.14])
            self.image[i, j] = col

    @ti.kernel
    def _render_particles(self):
        # Paper Sec. 5.1: retained water drives wet/dry sand appearance. This
        # lightweight renderer maps p_water_content to an orange-to-blue ramp;
        # with retention enabled p_water_content is the particle bound water.
        ratio_max = ti.static(self.cfg.ratio_max)
        p_vol_dim = ti.static(self._particle_vol())
        for p in self.p_x:
            ix = ti.cast(self.p_x[p].x, ti.i32)
            iy = ti.cast(self.p_x[p].y, ti.i32)
            wet = ti.max(0.0, ti.min(1.0, self.p_water_content[p] / (ratio_max * p_vol_dim)))
            dry_col = ti.Vector([0.92, 0.64, 0.12])
            wet_col = ti.Vector([0.05, 0.18, 0.95])
            col = dry_col * (1.0 - wet) + wet_col * wet
            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                x = ix + ox
                y = iy + oy
                if _inside(x, y, ti.static(self.nx), ti.static(self.ny)):
                    self.image[x, y] = col
