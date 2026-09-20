"""Summarize completed vortex controls; reconstruct force/energy budgets."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return (
        {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "stage"}
        if rows
        else {}
    )


def bulk_budget(f):
    a = np.load(f)
    dx = float(a["dx"])
    v = a["velocity"].astype(float)
    p = a["dynamic_pressure"].astype(float)
    phi = a["phi"]
    tau = a["shear_tau"]
    dt = float(a["dt"])
    rho = 1.25 + 998.75 * np.clip(phi, 0, 1)
    dux = np.gradient(v[:, :, 0], dx, axis=0)
    duy = np.gradient(v[:, :, 0], dx, axis=1)
    dvx = np.gradient(v[:, :, 1], dx, axis=0)
    dvy = np.gradient(v[:, :, 1], dx, axis=1)
    nu = (tau - 0.5) / 3 * dx * dx / dt
    stressxx = 2 * rho * nu * dux
    stressyy = 2 * rho * nu * dvy
    stressxy = rho * nu * (duy + dvx)
    visc = np.stack(
        [
            np.gradient(stressxx, dx, axis=0) + np.gradient(stressxy, dx, axis=1),
            np.gradient(stressxy, dx, axis=0) + np.gradient(stressyy, dx, axis=1),
        ],
        axis=-1,
    )
    pf = -np.stack(np.gradient(p, dx, axis=(0, 1)), axis=-1)
    conv = np.stack(
        [
            np.gradient(rho * v[:, :, 0] ** 2, dx, axis=0)
            + np.gradient(rho * v[:, :, 0] * v[:, :, 1], dx, axis=1),
            np.gradient(rho * v[:, :, 0] * v[:, :, 1], dx, axis=0)
            + np.gradient(rho * v[:, :, 1] ** 2, dx, axis=1),
        ],
        axis=-1,
    )
    # Exclude diffuse interface, ice and wall plus a two-cell stencil margin.
    valid = (phi > 0.999) & (a["solid"] == 0) & (a["wall"] == 0)
    for _ in range(2):
        valid &= (
            np.roll(valid, 1, 0)
            & np.roll(valid, -1, 0)
            & np.roll(valid, 1, 1)
            & np.roll(valid, -1, 1)
        )
    valid[:2] = False
    valid[-2:] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    x = (np.arange(phi.shape[0]) + 0.5) * dx
    y = (np.arange(phi.shape[1]) + 0.5) * dx
    roi = (
        valid
        & (x[:, None] > 0.009)
        & (x[:, None] < 0.0238)
        & (y[None, :] > 0.029)
        & (y[None, :] < 0.0318)
    )
    fields = {
        "pressure": pf,
        "viscous_approx": visc,
        "convective": -conv,
        "gravity": a["force_gravity"],
        "buoyancy": a["force_buoyancy"],
        "capillary": a["force_capillary"],
    }
    row = {"time_s": float(a["time_s"]), "bulk_jet_cells": int(roi.sum())}
    for k, F in fields.items():
        row[k + "_power"] = float(np.sum(np.sum(v * F, axis=-1)[roi]) * dx**2)
        row[k + "_fx"] = float(F[:, :, 0][roi].sum() * dx**2)
    mom = rho[:, :, None] * v
    return row, mom, sum(fields.values()), roi, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--frame-time", type=float, default=3.0)
    a = p.parse_args()
    cases = [
        x
        for x in [
            "baseline",
            "no_buoyancy",
            "shear_065",
            "width_4",
            "grid_200",
            "cutoff_1e5",
        ]
        if (a.root / x / "metrics.csv").exists()
    ]
    labels = {
        "baseline": "Baseline",
        "no_buoyancy": "No thermal buoyancy",
        "shear_065": "Shear tau=0.65",
        "width_4": "Interface W=4",
        "grid_200": "200 x 400 (matched scales)",
        "cutoff_1e5": "Phase cutoff 1e-5",
    }
    colors = plt.get_cmap("tab10").colors
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    summary = {}
    for ci, c in enumerate(cases):
        d = read(a.root / c / "metrics.csv")
        h = read(a.root / c / "history.csv")
        t = d["time_s"]
        mask = t >= 1.8
        for ax, key in zip(
            axs.flat,
            ["jet_speed", "jet_ux_mean", "water_ke", "upper_right_circulation"],
        ):
            ax.plot(t, d[key], label=labels[c], color=colors[ci])
            ax.set_xlim(1.8, 3.8)
            ax.grid(alpha=0.25)
            ax.set_xlabel("Time (s)")
        if mask.any():
            idx = np.where(mask)[0][np.argmax(d["jet_speed"][mask])]
            summary[c] = {
                "peak_speed": float(d["jet_speed"][idx]),
                "peak_t": float(t[idx]),
                "peak_y": float(d["jet_peak_y"][idx]),
                "final_melt": float(h["melted_fraction"][-1]),
                "final_t": float(t[-1]),
            }
        if (a.root / c / "stage_budgets.csv").exists():
            with (a.root / c / "stage_budgets.csv").open() as f:
                r = list(csv.DictReader(f))
            stages = {}
            for s in dict.fromkeys(x["stage"] for x in r):
                rr = [x for x in r if x["stage"] == s]
                duration = len(rr) * (6.25e-6 if c != "grid_200" else 3.125e-6)
                stages[s] = {
                    key + "_mean_rate": sum(float(x[key]) for x in rr) / duration
                    for key in ["mixture_ke", "water_ke", "mixture_px", "mixture_py"]
                }
            if c in summary:
                summary[c]["sampled_stage_rates"] = stages
    for ax, title in zip(
        axs.flat,
        [
            "Speed maximum in upper band (m/s)",
            "Mean horizontal speed in upper band (m/s)",
            "Water kinetic energy (J/m)",
            "Upper-right circulation (m²/s)",
        ],
    ):
        ax.set_title(title)
    axs[0, 0].legend(fontsize=8)
    fig.savefig(a.root / "comparison.png", dpi=170)
    plt.close(fig)
    (a.root / "comparison.json").write_text(json.dumps(summary, indent=2))
    # Flow panels at 3 s (or nearest available raw snapshot).
    fig, axs = plt.subplots(
        3,
        len(cases),
        figsize=(3.3 * len(cases), 10),
        squeeze=False,
        constrained_layout=True,
    )
    for ci, c in enumerate(cases):
        files = list((a.root / c / "fields").glob("*.npz"))
        dt = 3.125e-6 if c == "grid_200" else 6.25e-6
        f = min(files, key=lambda f: abs(int(f.stem.split("_")[1]) * dt - a.frame_time))
        b = np.load(f)
        dx = float(b["dx"])
        v = b["velocity"]
        water = (b["phi"] >= 0.5) & (b["solid"] == 0) & (b["wall"] == 0)
        speed = np.linalg.norm(v, axis=-1)
        omega = np.gradient(v[:, :, 1], dx, axis=0) - np.gradient(
            v[:, :, 0], dx, axis=1
        )
        x = (np.arange(water.shape[0]) + 0.5) * dx
        y = (np.arange(water.shape[1]) + 0.5) * dx
        for row, (values, cmap, vmin, vmax) in enumerate(
            [
                (speed, "viridis", 0, 0.1),
                (b["temperature"], "inferno", 0, 90),
                (omega, "RdBu_r", -150, 150),
            ]
        ):
            ax = axs[row, ci]
            im = ax.imshow(
                np.where(water, values, np.nan).T,
                origin="lower",
                extent=[0, 0.025, 0, 0.05],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.contour(x, y, b["phi"].T, [0.5], colors="gray", linewidths=0.6)
            if b["solid"].any():
                ax.contour(x, y, b["solid"].T, [0.5], colors="cyan", linewidths=0.7)
            ax.set_ylim(0.015, 0.036)
            ax.set_xlim(0, 0.025)
            ax.set_title(
                labels[c] + f"\nt={float(b['time_s']):.2f}s" if row == 0 else ""
            )
            if row == 0:
                stride = max(1, round(0.002 / dx))
                xx, yy = np.meshgrid(x[::stride], y[::stride], indexing="ij")
                uv = np.where(water[:, :, None], v, np.nan)[::stride, ::stride]
                ax.quiver(
                    xx,
                    yy,
                    uv[:, :, 0],
                    uv[:, :, 1],
                    scale=1.0,
                    color="white",
                    width=0.006,
                )
            if ci == 0:
                ax.set_ylabel(
                    ["Speed (m/s)", "Temperature (°C)", "Vorticity (1/s)"][row]
                )
        fig.colorbar(im, ax=axs[2, ci], shrink=0.6)
    fig.savefig(a.root / "flow_comparison.png", dpi=160)
    plt.close(fig)
    # Approximate continuum balance only in fully wet resolved stencil.
    for c in cases:
        fs = sorted((a.root / c / "fields").glob("*.npz"))
        fs = [f for f in fs if int(f.stem.split("_")[1]) > 0]
        if len(fs) < 3:
            continue
        buf = []
        rows = []
        for f in fs:
            buf.append(bulk_budget(f))
            if len(buf) < 3:
                continue
            left, mid, right = buf
            row, mom, rhs, mask, v = mid
            dmdt = (right[1] - left[1]) / (right[0]["time_s"] - left[0]["time_s"])
            residual = dmdt - rhs
            row["residual_rms_N_m3"] = (
                float(np.sqrt(np.mean(np.sum(residual[mask] ** 2, axis=-1))))
                if mask.any()
                else None
            )
            row["rhs_rms_N_m3"] = (
                float(np.sqrt(np.mean(np.sum(rhs[mask] ** 2, axis=-1))))
                if mask.any()
                else None
            )
            rows.append(row)
            buf.pop(0)
        if rows:
            with (a.root / c / "bulk_balance.csv").open("w") as f:
                w = csv.DictWriter(f, fieldnames=rows[0])
                w.writeheader()
                w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
