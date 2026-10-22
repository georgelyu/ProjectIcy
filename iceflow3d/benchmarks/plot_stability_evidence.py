"""Plot recorded ablations and a completed run; never rerun the solver."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_rows(directory):
    verified = directory / "verified_timeline.jsonl"
    timeline = verified if verified.exists() else directory / "timeline.jsonl"
    return [json.loads(line) for line in timeline.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", default="final_150")
    args = parser.parse_args()
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, label in (
        ("current_150", "Both old limiters"),
        ("flux_unlimited_150", "Only flux limiter removed"),
        ("reference8_150", "Old limiters, half timestep"),
        ("density_unlimited_150", "Density-source limiter removed"),
        ("unlimited_150", "Both limiters removed"),
    ):
        rows = read_rows(args.root / name)
        meta = json.loads((args.root / name / "diagnostic.json").read_text())
        if meta["status"] == "failed":
            rows.append(meta["failure"])
        axes[0].plot(
            [r["time_s"] for r in rows],
            [r["maximum_fluid_speed_lattice"] for r in rows],
            label=label,
        )
    axes[0].axhline(
        0.5, color="black", linestyle=":", label="Original failure threshold"
    )
    axes[0].set(
        xlabel="Physical time (s)",
        ylabel="Maximum L1 speed (LU)",
        title="Source-limiter ablation (150 x 50 x 30)",
    )
    axes[0].legend(fontsize=8)
    for name, label in (
        ("full8_unlimited_150", "Local geometry refill"),
        (args.run, "Constrained water inventory"),
    ):
        rows = read_rows(args.root / name)
        axes[1].plot(
            [r["time_s"] for r in rows],
            [100 * r["water_volume_change_ratio"] for r in rows],
            label=label,
        )
    axes[1].set(
        xlabel="Physical time (s)",
        ylabel="Water-volume change (%)",
        title="Geometry-remapping mass defect (150 x 50 x 30)",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.root / "root_cause_evidence.png", dpi=180)
    plt.close(fig)
    rows = read_rows(args.root / args.run)
    t = np.array([r["time_s"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    configuration = json.loads((args.root / args.run / "diagnostic.json").read_text())[
        "configuration"
    ]
    fig.suptitle(
        f"Recorded diagnostics: {configuration['nx']} x {configuration['ny']} x "
        f"{configuration['nz']}, through {t[-1]:g} s"
    )
    axes[0, 0].semilogy(
        t, np.maximum(1e-9, [r["maximum_fluid_speed_lattice"] for r in rows])
    )
    axes[0, 0].axhline(0.5, color="black", linestyle=":")
    axes[0, 0].set(
        ylabel="Maximum L1 speed (LU)", title="Recorded speed and failure threshold"
    )
    axes[0, 1].plot(t, [1e6 * r["phase_volume_change_ratio"] for r in rows])
    axes[0, 1].set(ylabel="Phase-volume change (ppm)", title="Closed water inventory")
    for body in range(4):
        axes[1, 0].plot(
            t, [1000 * r[f"ice_{body}_mass_kg"] for r in rows], label=f"Ice {body + 1}"
        )
    axes[1, 0].set(
        ylabel="Remaining ice mass (g)", title="Four independently melting bodies"
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(t, [r["thermal_remap_energy_j"] for r in rows])
    axes[1, 1].set(
        ylabel="Accumulated remap energy (J)",
        title="Explicit numerical heat adjustment",
    )
    for ax in axes.flat:
        ax.set_xlabel("Physical time (s)")
        ax.grid(alpha=0.2)
    fig.savefig(args.root / args.run / "diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
