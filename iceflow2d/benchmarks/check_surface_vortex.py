"""Reproducible falling/melting controls and read-only flow diagnostics.

Production kernels are unchanged. Every 0.1 s, eight actual steps are split
into the same production stages to measure discrete energy/momentum changes.
All other steps use IceFlow2D.step directly. NPZ arrays use (x, y[, component]).
"""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from iceflow2d import IceFlow2D  # noqa: E402
from iceflow2d.examples.coupled_falling_ice_melting_2d import (  # noqa: E402
    create_config,
    parse_args,
    _capture_snapshot,
    _history_row,
    HISTORY_COLUMNS,
)

CASES = [
    "baseline",
    "no_buoyancy",
    "shear_065",
    "width_4",
    "grid_200",
    "cutoff_1e5",
]


def configuration(case):
    args = []
    if case == "no_buoyancy":
        args += ["--thermal-expansion-water-1-k", "0"]
    if case == "width_4":
        args += ["--interface-width", "4"]
    if case == "grid_200":
        # Keep physical geometry, interface width, mobility, thermal interval,
        # phase warmup time, angular damping time and shear viscosity floor.
        args += [
            "--resolution-x",
            "200",
            "--resolution-y",
            "400",
            "--boundary-cells",
            "6",
            "--interface-width",
            "10",
            "--mobility",
            "0.2",
            "--phase-warmup-steps",
            "1000",
            "--thermal-update-interval",
            "16",
        ]
    cfg = create_config(parse_args(args))
    if case == "cutoff_1e5":
        cfg = replace(cfg, volume_projection_interface_cutoff=1e-5)
    if case == "shear_065":
        cfg = replace(cfg, air_interface_relaxation_time=0.65)
    if case == "grid_200":
        cfg = replace(
            cfg,
            air_interface_relaxation_time=1.1,
            angular_damping=0.999**0.5,
            volume_projection_max_shift=1.0,
            smagorinsky_constant=0.2,
        )
    return cfg


@ti.kernel
def sample_force(s: ti.template(), out: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i, j in s.water_phase:
        if s._is_fluid_cell(i, j):
            ph = ti.min(1.0, ti.max(0.0, s.water_phase[i, j]))
            rho = (
                s._air_density_lattice
                + (s._water_density_lattice - s._air_density_lattice) * ph
            )
            g = ti.Vector([s._gravity_lattice[0], s._gravity_lattice[1]])
            fg = (rho - s.reference_density_by_row[j]) * g
            temp = ti.cast(s.thermal._water_temperature_at(i, j), ti.f32)
            fb = (
                ti.max(0.0, 2.0 * ph - 1.0)
                * s._water_density_lattice
                * (
                    -s.config.thermal.thermal_expansion_water_1_k
                    * (temp - s.config.thermal.buoyancy_reference_temperature_c)
                )
                * g
            )
            data = s._fluid_force_viscosity_and_gradient(i, j)
            ft = rho * ti.Vector([data[0], data[1]])
            fc = ft - fg - fb
            near = 0.0
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                ni, nj = i + di, j + dj
                if 0 <= ni < s.nx and 0 <= nj < s.ny:
                    if s.solid_mask[ni, nj] != 0:
                        near = 1.0
            nu = (
                ph * s._water_viscosity_lattice
                + (1.0 - ph) * s._air_viscosity_lattice
                + data[2]
            )
            tau = ti.max(
                0.5 + 3.0 * nu,
                0.5 + (s._air_interface_relaxation_time - 0.5) * ti.max(1.0 - ph, near),
            )
            out[i, j, 0] = fg.x
            out[i, j, 1] = fg.y
            out[i, j, 2] = fb.x
            out[i, j, 3] = fb.y
            out[i, j, 4] = fc.x
            out[i, j, 5] = fc.y
            out[i, j, 6] = tau
            out[i, j, 7] = temp
            out[i, j, 8] = rho
            out[i, j, 9] = data[2]


def fields(s):
    return {
        "phi": s.water_phase.to_numpy(),
        "solid": s.solid_mask.to_numpy(),
        "velocity": s.physical_velocity_lattice.to_numpy()
        * s.scales.velocity_scale_m_s,
        "momentum_velocity": s.momentum_velocity_lattice.to_numpy()
        * s.scales.velocity_scale_m_s,
    }


def kinetic(s, state):
    ph = np.clip(state["phi"].astype(float), 0, 1)
    active = (state["solid"] == 0) & (s.wall_mask.to_numpy() == 0)
    rho = s.config.rho_air + (s.config.rho_water - s.config.rho_air) * ph
    rw = s.config.rho_water * ph
    v = state["velocity"].astype(float)
    v2 = np.sum(v * v, axis=-1)
    area = s.config.dx**2
    return {
        "mixture_ke": float(np.sum(np.where(active, 0.5 * rho * v2, 0)) * area),
        "water_ke": float(np.sum(np.where(active, 0.5 * rw * v2, 0)) * area),
        "mixture_px": float(np.sum(np.where(active, rho * v[:, :, 0], 0)) * area),
        "mixture_py": float(np.sum(np.where(active, rho * v[:, :, 1], 0)) * area),
        "water_px": float(np.sum(np.where(active, rw * v[:, :, 0], 0)) * area),
        "water_py": float(np.sum(np.where(active, rw * v[:, :, 1], 0)) * area),
    }


def diagnostic_step(s, rows, maps, save_maps=False):
    """Identical update order to production step; only add state reads."""
    t = s.physical_time_s
    previous = fields(s)
    previous_k = kinetic(s, previous)

    def measure(stage):
        nonlocal previous, previous_k
        state = fields(s)
        k = kinetic(s, state)
        rows.append(
            {
                "time_s": t,
                "stage": stage,
                **{key: k[key] - previous_k[key] for key in k},
            }
        )
        if save_maps:
            area = s.config.dx**2

            def energy(a):
                rho = s.config.rho_air + (
                    s.config.rho_water - s.config.rho_air
                ) * np.clip(a["phi"], 0, 1)
                return np.where(
                    (a["solid"] == 0) & (s.wall_mask.to_numpy() == 0),
                    0.5 * rho * np.sum(a["velocity"] ** 2, axis=-1) * area,
                    0,
                )

            if stage not in maps:
                maps[stage] = np.zeros((s.nx, s.ny), dtype=np.float64)
            maps[stage] += energy(state) - energy(previous)
            # Separate density bookkeeping from actual velocity changes.
            for suffix, delta in [
                ("delta_phi", state["phi"] - previous["phi"]),
                (
                    "delta_ux",
                    state["velocity"][:, :, 0] - previous["velocity"][:, :, 0],
                ),
                (
                    "delta_uy",
                    state["velocity"][:, :, 1] - previous["velocity"][:, :, 1],
                ),
            ]:
                name = stage + "__" + suffix
                if name not in maps:
                    maps[name] = np.zeros((s.nx, s.ny), dtype=np.float64)
                maps[name] += delta
        previous = state
        previous_k = k

    s._collide_momentum()
    s._collide_phase()
    s._stream()
    s._update_streamed_macroscopic_fields()
    s._update_pressure()
    measure("hydrodynamics")
    s._integrate_rigid_ice()
    s._solve_contact_and_rasterize()
    s._update_solid_mask()
    s._refill_changed_nodes()
    measure("body_motion_refill")
    s._advance_moving_thermal_fast(target_step=s.steps + 1)
    s._thermal_lbm_steps_pending += 1
    if s._thermal_lbm_steps_pending >= s.config.thermal.update_interval_lbm_steps:
        s._advance_moving_thermal_slow(s._thermal_lbm_steps_pending)
        s._thermal_lbm_steps_pending = 0
    measure("thermal_melt_refill")
    # Observe inside the existing projection without changing its operations.
    original_clip = s._clip_water_phase_for_projection

    def observed_clip():
        result = original_clip()
        measure("phase_clip")
        return result

    s._clip_water_phase_for_projection = observed_clip
    try:
        s._correct_water_volume()
    finally:
        s._clip_water_phase_for_projection = original_clip
    measure("phase_volume_shift")
    s._finalize_melt_momentum_coupling()
    measure("melt_momentum")
    s._synchronize_moving_water_aperture()
    measure("thermal_sync")
    s.steps += 1


def diagnostics(s, state, force):
    ph = state["phi"]
    v = state["velocity"]
    dx = s.config.dx
    active = (state["solid"] == 0) & (s.wall_mask.to_numpy() == 0)
    water = active & (ph >= 0.5)
    xx = (np.arange(s.nx) + 0.5) * dx
    yy = (np.arange(s.ny) + 0.5) * dx
    roi = (
        water
        & (xx[:, None] >= 0.009)
        & (xx[:, None] <= 0.0238)
        & (yy[None, :] >= 0.029)
        & (yy[None, :] <= 0.0318)
    )
    upper = (
        water
        & (xx[:, None] >= 0.009)
        & (xx[:, None] <= 0.0238)
        & (yy[None, :] >= 0.025)
        & (yy[None, :] <= 0.034)
    )
    speed = np.linalg.norm(v, axis=-1)
    pos = np.unravel_index(np.argmax(np.where(roi, speed, -1)), speed.shape)
    area = dx * dx
    rho = s.config.rho_air + (s.config.rho_water - s.config.rho_air) * np.clip(ph, 0, 1)
    omega = np.gradient(v[:, :, 1], dx, axis=0) - np.gradient(v[:, :, 0], dx, axis=1)
    k = kinetic(s, state)
    r = {
        "time_s": s.physical_time_s,
        **k,
        "water_umax": float(speed[water].max()),
        "jet_speed": float(speed[roi].max()),
        "jet_ux_max": float(v[:, :, 0][roi].max()),
        "jet_ux_mean": float(v[:, :, 0][roi].mean()),
        "jet_peak_x": xx[pos[0]],
        "jet_peak_y": yy[pos[1]],
        "upper_right_circulation": float(omega[upper].sum() * area),
        "jet_kinetic_energy": float((0.5 * rho * speed**2)[roi].sum() * area),
        "force_snapshot_max": float(
            np.max(np.linalg.norm(force[:, :, :6].reshape(s.nx, s.ny, 3, 2), axis=-1))
        ),
    }
    for n, sl in [
        ("residual_gravity", slice(0, 2)),
        ("buoyancy", slice(2, 4)),
        ("capillary", slice(4, 6)),
    ]:
        dot = np.sum(v * force[:, :, sl], axis=-1)
        r[n + "_power_mixture"] = float(dot[active].sum() * area)
        r[n + "_power_water_region"] = float(dot[water].sum() * area)
        r[n + "_power_jet"] = float(dot[roi].sum() * area)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--end", type=float, default=3.8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=0.02)
    p.add_argument("--window-start", type=float, default=1.8)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "fields").mkdir()
    cfg = configuration(a.case)
    manifest = {
        "case": a.case,
        "config": asdict(cfg),
        "end_s": a.end,
        "interval_s": a.interval,
        "window_start_s": a.window_start,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                ROOT / "iceflow2d/simulator.py",
                ROOT / "iceflow2d/config.py",
                Path(__file__),
            ]
        },
        "notes": [
            "Production solver unchanged. Diagnostic stages preserve production order.",
            "Force sampled using current phi/T; production stored half-force velocity uses preceding collision force.",
            "Stage budgets sample one full thermal cycle before each 0.1 s output; not continuous time integrals.",
            "Grid control preserves physical interface width, mobility, thermal cadence, damping time and shear viscosity floor.",
            "Grid control retains one-cell ice stabilization stencil, so its physical thickness halves.",
            "NPZ array order x,y; force densities N/m3, pressure Pa, velocity m/s, energy per out-of-plane metre.",
            "Map names stage_delta_ke_* are energy except __delta_phi (dimensionless) and __delta_ux/__delta_uy (m/s).",
        ],
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    s = IceFlow2D(cfg)
    dt = s.scales.dt_s
    initial_h = float(s.thermal.total_enthalpy_j_m(s.wall_mask))
    initial_s = float(s.phase_change_solid_volume_cells())
    stage_rows = []
    metric_rows = []
    hfile = (a.output / "history.csv").open("w")
    hw = csv.writer(hfile)
    hw.writerow(HISTORY_COLUMNS)
    last_status = 0
    try:
        targets = sorted(
            set(
                [
                    round(t / dt)
                    for t in np.arange(0, a.end + a.interval * 0.1, a.interval)
                ]
                + [round(a.end / dt)]
            )
        )
        for target in targets:
            sample_cycle = (
                target > 0
                and target * dt >= a.window_start - 1e-8
                and abs(target * dt * 10 - round(target * dt * 10)) < 1e-6
            )
            cycles = cfg.thermal.update_interval_lbm_steps if sample_cycle else 0
            s.step(target - s.steps - cycles)
            maps = {}
            for _ in range(cycles):
                diagnostic_step(s, stage_rows, maps, save_maps=True)
            snap = _capture_snapshot(
                s,
                initial_total_enthalpy_j_m=initial_h,
                initial_solid_volume_cells=initial_s,
            )
            hw.writerow(_history_row(snap))
            hfile.flush()
            state = fields(s)
            force = np.zeros((s.nx, s.ny, 10), dtype=np.float32)
            sample_force(s, force)
            force[:, :, :6] *= cfg.rho_water * cfg.dx / dt**2
            metric_rows.append(diagnostics(s, state, force))
            if target * dt >= a.window_start - 1e-8 or target == 0:
                pressure = (
                    s.pressure_lattice.to_numpy() * cfg.rho_water * (cfg.dx / dt) ** 2
                )
                ref = (
                    s.reference_pressure_by_row.to_numpy()
                    * cfg.rho_water
                    * (cfg.dx / dt) ** 2
                )
                np.savez_compressed(
                    a.output / "fields" / f"frame_{target:09d}.npz",
                    time_s=target * dt,
                    dx=cfg.dx,
                    dt=dt,
                    **state,
                    temperature=force[:, :, 7],
                    pressure=pressure,
                    dynamic_pressure=pressure - ref[None, :],
                    force_gravity=force[:, :, :2],
                    force_buoyancy=force[:, :, 2:4],
                    force_capillary=force[:, :, 4:6],
                    shear_tau=force[:, :, 6],
                    sgs_nu_lattice=force[:, :, 9],
                    wall=s.wall_mask.to_numpy(),
                    **{"stage_delta_ke_" + k: v for k, v in maps.items()},
                )
            if abs(target * dt - 1.8) < dt * 0.1:
                # Preserve authoritative device and host state for paired restarts.
                checkpoint = {}
                seen = set()
                for prefix, owner in [("flow", s), ("thermal", s.thermal)]:
                    for name, value in vars(owner).items():
                        if (
                            isinstance(value, (ti.Field, ti.MatrixField))
                            and id(value) not in seen
                        ):
                            checkpoint[prefix + "__" + name] = value.to_numpy()
                            seen.add(id(value))
                np.savez_compressed(a.output / "checkpoint_1p8.npz", **checkpoint)
                host = {}
                for prefix, owner in [("flow", s), ("thermal", s.thermal)]:
                    host[prefix] = {
                        name: value
                        for name, value in vars(owner).items()
                        if isinstance(value, (bool, int, float, str))
                    }
                (a.output / "checkpoint_1p8_host.json").write_text(
                    json.dumps(host, indent=2)
                )
            for filename, rows in [
                ("metrics.csv", metric_rows),
                ("stage_budgets.csv", stage_rows),
            ]:
                if rows:
                    with (a.output / filename).open("w") as f:
                        w = csv.DictWriter(f, fieldnames=rows[0].keys())
                        w.writeheader()
                        w.writerows(rows)
            status = {
                "case": a.case,
                "time_s": s.physical_time_s,
                "wall_seconds": time.monotonic() - start,
                "melted_fraction": snap.melted_fraction,
                "body_active": snap.body_active,
                "jet_speed": metric_rows[-1]["jet_speed"],
                "water_umax": metric_rows[-1]["water_umax"],
            }
            (a.output / "status.json").write_text(json.dumps(status, indent=2))
            if time.monotonic() - last_status >= 20 or target == targets[-1]:
                print(json.dumps(status), flush=True)
                last_status = time.monotonic()
            if not np.isfinite(state["velocity"]).all():
                raise RuntimeError("Nonfinite velocity")
            center = np.asarray(s.body_center[None]) * cfg.dx
            if snap.body_active and (
                center[0] < 0
                or center[0] > s.nx * cfg.dx
                or center[1] < 0
                or center[1] > s.ny * cfg.dx
            ):
                raise RuntimeError(
                    "Unresolved active ice has left the container; later flow is not a valid control"
                )
        status["complete"] = True
    except BaseException:
        status = {
            "case": a.case,
            "time_s": s.physical_time_s,
            "wall_seconds": time.monotonic() - start,
            "complete": False,
            "error": traceback.format_exc(),
        }
        raise
    finally:
        hfile.close()
        (a.output / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
