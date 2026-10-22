"""Summarize saved complete-step timings and numerical comparisons; no GPU run."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def group_kernels(measurement):
    result = defaultdict(float)
    for row in measurement["kernels"]:
        name = row["name"]
        if name.startswith("_stream_home"):
            group = "HOME streaming"
        elif name.startswith("_forces") or name == "_classify_home":
            group = "Forces + classification"
        elif name == "_collide_momentum":
            group = "CM momentum collision"
        elif name.startswith("_collide_phase"):
            group = "D3Q7 AA collision"
        elif name.startswith("_gradients"):
            group = "Phase gradient"
        elif name.startswith("_recover_") or name == "_scan_stability":
            group = "Macro recovery + stability"
        elif name in {
            "_thermal_fluxes",
            "_apply_thermal_fluxes",
            "_build_water_targets",
            "_commit_water_targets",
            "_body_conduction",
            "_exchange_heat",
            "_apply_melting",
            "_check_heat",
            "_measure_outflow",
        }:
            group = "Thermal kernels"
        elif name in {
            "_rasterize",
            "_repair_phase_layout",
            "_find_refill_changes",
            "_seed_refill",
            "_seed_refill_region",
            "_propagate_refill",
            "_accept_refill_iteration",
            "_commit_refill",
            "_project_refill",
            "_refill_gradients",
        } or name.startswith(
            ("field_copy", "_refill_", "_classify_refill_", "_write_refill_")
        ):
            group = "Geometry + refill"
        else:
            group = "Other GPU tasks"
        result[group] += row["gpu_ms_per_step"]
    return dict(result)


def array_comparison(before, after):
    with np.load(before) as a, np.load(after) as b:
        result = {}
        fluid = (a["solid_id"] == -1) & (b["solid_id"] == -1)
        for key in a.files:
            x, y = a[key].astype(float), b[key].astype(float)
            difference = y - x
            norm = np.linalg.norm(x.ravel())
            result[key] = dict(
                max_abs=float(np.max(abs(difference))),
                rms=float(np.sqrt(np.mean(difference**2))),
                relative_l2=float(np.linalg.norm(difference.ravel()) / norm)
                if norm
                else None,
                finite=bool(np.isfinite(x).all() and np.isfinite(y).all()),
            )
            if key == "solid_id":
                result[key]["different_cells"] = int(np.count_nonzero(x != y))
            elif key in ("phi", "velocity", "pressure", "water_energy", "water_volume"):
                d, reference = difference[fluid], x[fluid]
                norm = np.linalg.norm(reference.ravel())
                result[key]["common_fluid"] = dict(
                    max_abs=float(np.max(abs(d))),
                    rms=float(np.sqrt(np.mean(d**2))),
                    relative_l2=float(np.linalg.norm(d.ravel()) / norm)
                    if norm
                    else None,
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    result = {
        "protocol": {
            "grid": [450, 150, 90],
            "device": "NVIDIA RTX 3090",
            "baseline": "r00: unoptimized HOME+D3Q7 AA working tree at task start",
            "sample": "one full step: thermal pending 7 to 0; conduction/melting dt=8*flow_dt",
            "excluded": "initialization, JIT, snapshot restore, diagnostics and file output",
            "wall": "unprofiled process; independent restored samples; median",
            "gpu": "CUDA-event sum from a separate profiler process",
            "stages": "separate synchronized attribution, not additive to production wall time",
            "validation": "256 coupled steps after step31, followed by thermal flush; sections plus global scalar diagnostics",
        },
        "wall": {},
        "gpu": {},
        "validation": {},
    }
    for case in ("initial", "moving"):
        before = read(folder, f"r00_{case}_wall")
        after = read(folder, f"final_{case}_wall")
        a, b = before["measurement"], after["measurement"]
        assert before["configuration"] == after["configuration"]
        result["wall"][case] = dict(
            before_ms=a["wall_median_ms"],
            after_ms=b["wall_median_ms"],
            speedup=a["wall_median_ms"] / b["wall_median_ms"],
            reduction_percent=100 * (1 - b["wall_median_ms"] / a["wall_median_ms"]),
            before_samples_ms=a["wall_ms"],
            after_samples_ms=b["wall_ms"],
        )
        validation_before = read(folder, f"r00_{case}_validation")
        va = validation_before["measurement"]["validation"]
        vb = after["measurement"]["validation"]
        assert va["step"] == vb["step"]
        scalar_keys = (
            "water_volume_m3",
            "water_energy_j",
            "ice_mass_kg",
            "melted_mass_kg",
            "maximum_fluid_speed_lattice",
            "peak_fluid_speed_lattice",
            "phase_volume_change_ratio",
            "pending_refill_cells",
            "gas_repair_count",
        )
        result["validation"][case] = dict(
            step=va["step"],
            time_s=va["time_s"],
            sections_and_bodies=array_comparison(
                folder / f"r00_{case}_validation.validation.npz",
                folder / f"final_{case}_wall.validation.npz",
            ),
            global_diagnostics={
                k: dict(before=va[k], after=vb[k], difference=vb[k] - va[k])
                for k in scalar_keys
            },
        )
    before = read(folder, "r00_initial_kernels")["measurement"]
    after = read(folder, "final_initial_kernels")["measurement"]
    ga, gb = group_kernels(before), group_kernels(after)
    result["gpu"] = dict(
        before_ms=before["gpu_ms_per_step"],
        after_ms=after["gpu_ms_per_step"],
        groups=[
            dict(name=k, before_ms=ga.get(k, 0), after_ms=gb.get(k, 0)) for k in ga | gb
        ],
        top_final=after["kernels"][:12],
    )
    result["stages"] = {
        label: [r for r in measurement["stages"] if len(r["path"]) == 1]
        for label, measurement in (("before", before), ("after", after))
    }
    result["physics"] = read(folder, "final_physics")
    audit = read(folder, "final_gpu_ptx")
    result["gpu_precision"] = dict(
        modules=len(audit["modules"]),
        fp64_mentions=audit["fp64_mentions"],
        passed=audit["passed"],
    )
    result["source_sha256"] = read(folder, "final_initial_wall")["source_sha256"]
    if (folder / "regression_queue.json").exists():
        jobs = json.loads((folder / "regression_queue.json").read_text())
        logs = ["final_home_tests.log"] + [row["log"] for row in jobs]
        counts = []
        for name in logs:
            log = (folder / name).read_text()
            match = re.search(r"Ran (\d+) tests? in", log)
            counts.append(
                dict(
                    log=name,
                    count=int(match[1]) if match else 0,
                    passed="\nOK\n" in log,
                )
            )
        total = sum(row["count"] for row in counts)
        result["regressions"] = dict(
            expected=53,
            completed=total,
            logs=counts,
            passed=total == 53
            and all(row["passed"] for row in counts)
            and all(row["exit_code"] == 0 for row in jobs),
        )
    result["limitations"] = [
        "Short coupled regression, not a multi-second dam-break stability certificate.",
        "Field comparisons use the middle-z section; global diagnostics cover volume, energy, ice mass and stability.",
        "Pressure in solid cells is an unused retained value; assess pressure differences on the common-fluid mask.",
        "Six-face centered velocity derivative in the variable-density viscous force is the deliberate spatial approximation; CM shear relaxation is unchanged.",
        "Atomic FP32 reductions and reordered quadrature are not bitwise reproducible.",
        "Remaining timings depend on interface complexity, contacts and thermal/advection substeps.",
    ]
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.8]}
    )
    colors = ["#8c9ca8", "#007e87"]
    for i, (case, row) in enumerate(result["wall"].items()):
        for j, key in enumerate(("before_ms", "after_ms")):
            y = i * 3 + j
            axes[0].barh(y, row[key], color=colors[j], height=0.7)
            axes[0].text(row[key] + 0.3, y, f"{row[key]:.2f} ms", va="center")
    axes[0].set_yticks([0.5, 3.5], ["Dam-break", "Moving / tilted ice"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(r["before_ms"] for r in result["wall"].values()) * 1.3)
    axes[0].set_title("Complete-step wall time (median)", loc="left", fontweight="bold")
    axes[0].set_xlabel("milliseconds / complete step")
    groups = sorted(result["gpu"]["groups"], key=lambda r: r["before_ms"], reverse=True)
    y = np.arange(len(groups))
    for j, key in enumerate(("before_ms", "after_ms")):
        bars = axes[1].barh(
            y + (j - 0.5) * 0.35,
            [r[key] for r in groups],
            height=0.32,
            color=colors[j],
            label=("Before", "After")[j],
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set_yticks(y, [r["name"] for r in groups])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(r["before_ms"] for r in groups) * 1.15)
    axes[1].set_title("GPU kernel time by component", loc="left", fontweight="bold")
    axes[1].set_xlabel("milliseconds / complete step (separate CUDA profiler run)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "HOME + D3Q7 AA optimization | 450 × 150 × 90 | RTX 3090",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.025,
        "W=5  ·  physical water/air viscosity  ·  dt=1/120000 s  ·  central-moment collisions  ·  full thermal update included",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=2)
    fig.savefig(folder / "performance.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result["wall"], indent=2))


if __name__ == "__main__":
    main()
