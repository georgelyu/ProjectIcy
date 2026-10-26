"""Compare saved dry-ice trajectories and matching physical-time slices.

This checks preservation of a numerical solution, not experimental accuracy.
The reference may be a longer run; only matching saved steps are compared.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def history(path):
    with (path / "history.csv").open() as stream:
        return {
            int(row["step"]): {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)
        }


def difference(actual, reference):
    delta = actual.astype(float) - reference.astype(float)
    return {
        "rms": float(np.sqrt(np.mean(delta**2))),
        "max_absolute": float(np.max(np.abs(delta))),
        "relative_l2": float(
            np.linalg.norm(delta.ravel())
            / max(np.linalg.norm(reference.ravel()), 1e-30)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-label", default="Optimized")
    args = parser.parse_args()
    paths = (args.reference, args.candidate)
    metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
    if metadata[0]["configuration"] != metadata[1]["configuration"]:
        raise ValueError("physical/numerical configurations differ")
    rows = [history(p) for p in paths]
    steps = sorted(rows[1])
    if any(step not in rows[0] for step in steps):
        raise ValueError("candidate output times absent from reference")
    times = np.array([rows[1][step]["time_s"] for step in steps])
    diagnostics = {}
    for key in rows[1][steps[0]]:
        a, b = (np.array([row[step][key] for step in steps]) for row in rows)
        diagnostics[key] = {
            "max_absolute_difference": float(np.max(np.abs(b - a))),
            "reference_final": float(a[-1]),
            "candidate_final": float(b[-1]),
        }
    positions = [
        np.array([[row[step][f"ice_0_{axis}_m"] for axis in "xyz"] for step in steps])
        for row in rows
    ]
    position_error = np.linalg.norm(positions[1] - positions[0], axis=1)
    slices = []
    last = None
    for path in sorted((args.candidate / "slices").glob("slice_*.npz")):
        with np.load(path) as data:
            candidate = {k: data[k] for k in data.files}
        with np.load(args.reference / "slices" / path.name) as data:
            reference = {k: data[k] for k in data.files}
        assert candidate["step"] == reference["step"]
        common = (reference["solid_body_id"] == -1) & (candidate["solid_body_id"] == -1)
        wet = (
            common
            & (reference["water_phase"] >= 0.5)
            & (candidate["water_phase"] >= 0.5)
        )
        gas = (
            common & (reference["water_phase"] < 0.5) & (candidate["water_phase"] < 0.5)
        )
        fields = {}
        for field, mask in (
            ("water_phase", common),
            ("velocity_m_s", common),
            ("water_temperature_c", wet),
        ):
            fields[field] = difference(candidate[field][mask], reference[field][mask])
            assert np.isfinite(candidate[field][mask]).all(), (path, field)
        for phase, mask in (("water", wet), ("gas", gas)):
            fields[phase + "_velocity_m_s"] = difference(
                candidate["velocity_m_s"][mask], reference["velocity_m_s"][mask]
            )
        slices.append(
            {
                "step": int(candidate["step"]),
                "time_s": float(candidate["time_s"]),
                "solid_label_changes": int(
                    np.count_nonzero(
                        candidate["solid_body_id"] != reference["solid_body_id"]
                    )
                ),
                "fields": fields,
            }
        )
        last = (reference, candidate)
    result = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_label": args.candidate_label,
        "configuration_identical": True,
        "end_time_s": float(times[-1]),
        "frames": len(slices),
        "body_position_max_norm_difference_m": float(position_error.max()),
        "diagnostics": diagnostics,
        "slices": slices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for index, (label, style) in enumerate(
        (("Original", "-"), (args.candidate_label, "--"))
    ):
        for ax, key, factor, title in (
            (axes[0, 0], "ice_0_y_m", 1000, "Dry-ice center y (mm, array coordinates)"),
            (axes[0, 1], "sublimated_co2_mass_kg", 1e6, "Cumulative sublimation (mg)"),
        ):
            ax.plot(
                times, [rows[index][s][key] * factor for s in steps], style, label=label
            )
            ax.set(xlabel="Time (s)", title=title)
            ax.grid(alpha=0.2)
            ax.legend()
    axes[0, 2].plot(
        [s["time_s"] for s in slices],
        [s["fields"]["water_phase"]["rms"] for s in slices],
        label="Phase RMS",
    )
    axes[0, 2].set(
        title="Central-slice difference", xlabel="Time (s)", ylabel="Phase RMS"
    )
    axes[0, 2].grid(alpha=0.2)
    for ax, data, title in zip(axes[1, :2], last, ("Original", args.candidate_label)):
        x, y = data["x_m"] * 1000, data["y_m"] * 1000
        ax.pcolormesh(
            x,
            y,
            np.ma.masked_where(data["solid_body_id"] != -1, data["water_phase"]),
            cmap="Blues",
            vmin=0,
            vmax=1,
            shading="nearest",
        )
        ax.contour(x, y, data["solid_body_id"] >= 0, levels=[0.5], colors="black")
        ax.set(
            xlabel="x (mm)",
            ylabel="y (mm)",
            title=f"{title}, t={times[-1]:.2f} s",
            xlim=(15, 57),
            ylim=(0, 30),
            aspect="equal",
        )
    reference, candidate = last
    delta = candidate["water_phase"] - reference["water_phase"]
    mask = (candidate["solid_body_id"] != -1) | (reference["solid_body_id"] != -1)
    vmax = max(float(np.max(np.abs(delta[~mask]))), 1e-6)
    im = axes[1, 2].pcolormesh(
        x,
        y,
        np.ma.masked_where(mask, delta),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="nearest",
    )
    axes[1, 2].set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="Phase difference",
        xlim=(15, 57),
        ylim=(0, 30),
        aspect="equal",
    )
    fig.colorbar(im, ax=axes[1, 2])
    n = metadata[1]["configuration"]["nx"] - 2
    fig.suptitle(f"{n}³ dry-ice comparison: unchanged physical / numerical parameters")
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {k: result[k] for k in ("end_time_s", "frames", "configuration_identical")}
        )
    )
    print("saved", args.output)


if __name__ == "__main__":
    main()
