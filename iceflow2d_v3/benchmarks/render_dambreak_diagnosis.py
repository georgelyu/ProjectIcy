"""Render the scalar-only dam-break controls; never read/write raw grid files."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parents[1]
    root = package / "outputs"
    ghost = json.loads(
        (root / "dambreak_ghost_controls/matched_ghost_operator.json").read_text()
    )
    modes = json.loads(
        (root / "dambreak_hrr_controls/uniform_air_modes_and_shear.json").read_text()
    )
    stencils = json.loads(
        (root / "dambreak_ghost_controls/ghost_stencil_detail.json").read_text()
    )
    bad = next(s for s in stencils if s["direction"] == [-1.0, 1.0])
    controls = []
    for folder in (
        "dambreak_root_cause",
        "dambreak_ghost_controls",
        "dambreak_density_gradient_controls",
        "dambreak_limited_gradient_1s",
        "dambreak_hrr_controls",
        "dambreak_linear_pressure_controls",
    ):
        for path in sorted((root / folder).glob("*_summary.json")):
            result = json.loads(path.read_text())
            if "mode" not in result:
                continue
            failed = re.search(r"at step (\d+)", result["error"] or "")
            controls.append(
                dict(
                    mode=result["mode"],
                    completed=result["completed"],
                    requested_s=result["requested_s"],
                    failure_step=int(failed[1]) if failed else None,
                    failure_time_s=int(failed[1]) / 120000 if failed else None,
                    completed_steps=result["last"]["step"],
                    source=str(path),
                )
            )
    summary = dict(
        controls=controls,
        bad_ghost=bad,
        matched_ghost=[{k: v for k, v in g.items() if k != "history"} for g in ghost],
        uniform_air_modes=[
            {k: v for k, v in m.items() if k != "samples"} for m in modes["modes"]
        ],
        shear_viscosity=modes["shear"],
        source_sha256={
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in (
                "simulator.py",
                "stability_options.py",
                "lattice.py",
                "config.py",
                "examples/dambreak_four_ice_melting_2d.py",
                "benchmarks/diagnose_dambreak_instability.py",
            )
        },
    )
    translation = (
        root / "dambreak_translation_controls/uniform_translation_controls.json"
    )
    if translation.exists():
        summary["uniform_translation"] = [
            {k: v for k, v in t.items() if k != "history"}
            for t in json.loads(translation.read_text())
        ]
    (args.output_dir / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), layout="constrained")
    ax = axes[0]
    cells = [
        (178, 3, "#8dccbd", "Donor\n$\\phi=0.23368$"),
        (179, 3, "#abd4ee", "ICE"),
        (180, 3, "#fff3d0", "New air cell\n$\\phi=4.71\\times10^{-7}$"),
        (181, 3, "#fff3d0", "Air donor\n$\\phi=4.71\\times10^{-7}$"),
        (179, 4, "#abd4ee", "ICE\nblocked $q_6$"),
        (180, 4, "#abd4ee", "ICE"),
        (181, 4, "#abd4ee", "ICE"),
    ]
    for x, y, color, label in cells:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((177.5, 1.5), 4, 1, facecolor="#d9dde3", edgecolor="white"))
    ax.text(179.5, 1.93, "FLOOR (j < 3)", ha="center", color="#555")
    ax.plot(
        *bad["mirror"], marker="x", color="#c64444", markersize=10, markeredgewidth=2
    )
    ax.annotate(
        "Mirror stencil touches floor",
        xy=bad["mirror"],
        xytext=(177.7, 1.25),
        fontsize=9,
        color="#b22c2c",
        arrowprops={"arrowstyle": "->", "color": "#b22c2c"},
    )
    ax.annotate(
        "",
        xy=(178, 3.52),
        xytext=(180, 3.52),
        arrowprops={
            "arrowstyle": "->",
            "color": "#c64444",
            "linestyle": "--",
            "linewidth": 2,
        },
    )
    ax.text(
        179,
        3.68,
        "ghost reads across solid separator",
        ha="center",
        fontsize=8,
        color="#b22c2c",
    )
    ax.set(
        xlim=(177.45, 181.55),
        ylim=(0.9, 4.6),
        xticks=[178, 179, 180, 181],
        yticks=[2, 3, 4],
        xlabel="Grid i",
        ylabel="Grid j",
        title="A. Proven ghost sampling error",
    )
    ax.set_aspect("equal")

    ax = axes[1]
    labels = ["Original mirror", "Local Neumann control", "Connected donors control"]
    colors = ["#c64444", "#3864a4", "#218575"]
    for g, label, color in zip(ghost, labels, colors, strict=True):
        rows = g["history"]
        ax.semilogy(
            [r["step"] - 8544 for r in rows],
            [r["point_speed_l1"] for r in rows],
            color=color,
            label=label,
            linewidth=1.8,
        )
    ax.annotate(
        "Fails in 5 steps",
        xy=(5, ghost[0]["last"]["point_speed_l1"]),
        xytext=(32, 2),
        color=colors[0],
        arrowprops={"arrowstyle": "->", "color": colors[0]},
    )
    ax.set(
        xlabel="Fluid steps from identical state at step 8544",
        ylabel="$|u_x|+|u_y|$ at (180,3) [LU]",
        title="B. Same-state causal test",
        xlim=(0, 245),
        ylim=(1e-5, 100),
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.03,
        0.04,
        "Frozen geometry; stationary ice walls\nPhysical viscosity and dt unchanged",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = axes[2]
    for m in modes["modes"]:
        if m["speed"] != 0.2:
            continue
        rows = m["samples"]
        amplitude = np.array([r["amplitude"] for r in rows])
        label = f"{m['model']}: {m['measured_radius']:.5f} / step"
        ax.semilogy(
            [r["step"] for r in rows],
            amplitude / amplitude[0],
            label=label,
            linewidth=1.8,
            color="#c64444" if m["model"] == "cm_mrt" else "#3864a4",
        )
    ax.axhline(1, color="#777", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="LBM step",
        ylabel="Perturbation amplitude / initial amplitude",
        title="C. Independent pure-air short wave",
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(
        0.32,
        0.04,
        "Uniform diagonal speed: 0.2 LU\n$\\nu_{air}$ = 1.45065e-5 m²/s\nNo ice, thermal update, or phase gradient\nHRR still fails the full dam-break case",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.suptitle(
        "Dam-break diagnosis | W=5 | 450×150 | dt=8.333 µs | density ratio=800",
        fontsize=14,
    )
    fig.savefig(args.output_dir / "diagnosis.png", dpi=180)
    plt.close(fig)
    print(args.output_dir.resolve() / "diagnosis.png")
    print(args.output_dir.resolve() / "evidence_summary.json")


if __name__ == "__main__":
    main()
