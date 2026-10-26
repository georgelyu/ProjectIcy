"""Compare full-grid validation exports made by profile_dryice.py."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate):
    a = json.loads(reference.read_text())
    b = json.loads(candidate.read_text())
    if a["configuration"] != b["configuration"]:
        raise ValueError("configuration mismatch")
    if (
        a["validation_steps"] != b["validation_steps"]
        or a["warmup_steps"] != b["warmup_steps"]
    ):
        raise ValueError("validation interval mismatch")
    result = dict(
        reference=str(reference.resolve()),
        candidate=str(candidate.resolve()),
        speedup=a["wall_median_ms"] / b["wall_median_ms"],
        fields={},
        bodies={},
    )
    with (
        np.load(reference.with_suffix(".npz")) as x,
        np.load(candidate.with_suffix(".npz")) as y,
    ):
        fluid = (x["solid_id"] == -1) & (y["solid_id"] == -1)
        result["solid_label_changes"] = int(
            np.count_nonzero(x["solid_id"] != y["solid_id"])
        )
        for key in ("phi", "velocity", "water_energy", "water_volume"):
            xa, ya = x[key][fluid].astype(float), y[key][fluid].astype(float)
            diff = ya - xa
            result["fields"][key] = dict(
                max_absolute=float(abs(diff).max()),
                rms_absolute=float(np.sqrt(np.mean(diff**2))),
                relative_l2=float(
                    np.sqrt(np.sum(diff**2) / max(np.sum(xa**2), 1e-300))
                ),
                finite=bool(np.isfinite(ya).all()),
            )
        for key in ("body_origin", "body_q", "body_velocity", "body_mass"):
            result["bodies"][key] = float(abs(y[key] - x[key]).max())
        dx = a["configuration"]["domain_size_m"][0] / a["configuration"]["nx"]
        result["body_position_max_difference_m"] = float(
            np.linalg.norm(y["body_origin"] - x["body_origin"], axis=1).max() * dx
        )
        result["body_mass_max_difference_kg"] = (
            result["bodies"]["body_mass"] * a["configuration"]["rho_water"] * dx**3
        )
        result["velocity_max_difference_m_s"] = (
            result["fields"]["velocity"]["max_absolute"] * dx / a["dt_s"]
        )
    result["diagnostic_differences"] = {
        k: b["validation"][k] - v
        for k, v in a["validation"].items()
        if isinstance(v, (int, float))
        and isinstance(b["validation"].get(k), (int, float))
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
