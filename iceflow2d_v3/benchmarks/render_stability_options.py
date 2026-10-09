"""Render the independent option runs without rerunning the simulations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    names = {
        "baseline": "Original",
        "hrr": "Pressure HRR",
        "consistent_flux": "Consistent flux",
        "refill": "Stress refill",
    }
    colors = {
        "baseline": "#697386",
        "hrr": "#d97924",
        "consistent_flux": "#16847c",
        "refill": "#8660b4",
    }
    series = {
        name: [
            json.loads(line)
            for line in (args.root / name / "metrics.jsonl").read_text().splitlines()
        ]
        for name in names
    }
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), layout="constrained")
    ax = axes[0, 0]
    for name, rows in series.items():
        rows = [r for r in rows if r["time_s"] <= 0.12]
        t = [r["time_s"] for r in rows]
        speed = [max(r["max_l1"], 1e-7) for r in rows]
        ax.semilogy(t, speed, color=colors[name], label=names[name], lw=1.8)
        if name != "consistent_flux":
            ax.plot(t[-1], speed[-1], "x", color=colors[name], ms=8, mew=2)
    ax.set(
        xlabel="Time (s)",
        ylabel="Max |ux| + |uy| (LU)",
        xlim=(0, 0.12),
        ylim=(1e-4, 1e5),
        title="A  Entry: three runs diverge near 0.10 s",
    )
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.15)
    rows = series["consistent_flux"]
    t = np.array([r["time_s"] for r in rows])
    ax = axes[0, 1]
    ax.semilogy(
        t,
        np.maximum([r["max_l1"] for r in rows], 1e-7),
        color=colors["consistent_flux"],
        label="Full domain",
    )
    ax.semilogy(
        t,
        np.maximum([r["water_max_l1"] for r in rows], 1e-7),
        color="#246cb1",
        label="Water (phi > 0.5)",
        alpha=0.8,
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Max |ux| + |uy| (LU)",
        xlim=(0, 5),
        ylim=(1e-4, 1),
        title="B  Consistent flux: 600,000 steps completed",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    ax = axes[1, 0]
    volume = 100 * np.array([r["phase_relative_change"] for r in rows])
    ice = np.array([sum(r["body_mass"]) for r in rows])
    ax.plot(t, volume, color="#ba522d", label="Liquid volume change")
    ax.set(
        xlabel="Time (s)",
        ylabel="Liquid volume change (%)",
        xlim=(0, 5),
        ylim=(0, 11),
        title="C  Stability does not establish mass accuracy",
    )
    twin = ax.twinx()
    twin.plot(t, 100 * ice / ice[0], color="#697386", ls="--", label="Remaining ice")
    twin.set(ylabel="Remaining ice (%)", ylim=(0, 110))
    ax.annotate(
        f"+{volume[-1]:.2f}%",
        (5, volume[-1]),
        xytext=(-8, -15),
        textcoords="offset points",
        ha="right",
        color="#ba522d",
        weight="bold",
    )
    ax.legend(handles=[*ax.lines, *twin.lines], frameon=False, loc="center right")
    ax.grid(alpha=0.15)
    ax = axes[1, 1]
    with np.load(args.root / "consistent_flux/final.npz") as data:
        phi = np.ma.masked_where(data["wall"] != 0, data["phi"])
    im = ax.imshow(
        phi.T,
        origin="lower",
        extent=(0, 15, 0, 5),
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title="D  Phase at 5 s; all four ice blocks melted",
    )
    fig.colorbar(im, ax=ax, shrink=0.65, label="Water fraction phi")
    ax.text(
        0.02,
        0.92,
        f"Air-side mean phi = {rows[-1]['gas_mean_phi']:.4g}\nLiquid volume drift = +{volume[-1]:.2f}%",
        transform=ax.transAxes,
        va="top",
        color="#313b4a",
        fontsize=9,
    )
    fig.suptitle(
        "Independent stability options | W=5 | 450 x 150 | dt=8.333 us",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(args.root / "comparison.png", dpi=180)
    fig.savefig(args.root / "comparison.pdf")


if __name__ == "__main__":
    main()
