"""Exact production replay with a sparse link budget of the opening ice gap."""

import argparse
import json
from pathlib import Path

import numpy as np

from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    PointProbe,
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config


def sparse_state(s, stage, attempted):
    names = (
        "water_phase",
        "momentum_velocity_lattice",
        "fluid_acceleration_lattice",
        "pressure_lattice",
        "hydrostatic_reference_pressure",
        "solid_body_id",
        "previous_body_id",
        "_moving_body_fraction",
        "phase_gradient_lattice",
        "source_gradient_lattice",
        "phase_gradient_status",
        "phase_donor_mask",
        "phase_populations",
        "momentum_populations",
        "phase_stream_buffer",
        "momentum_stream_buffer",
        "phase_mass_flux",
        "phase_wall_geometry",
    )
    arrays = {name: getattr(s, name).to_numpy() for name in names}
    points = []
    for i in range(174, 178):
        for j in range(3, 18):
            row = dict(cell=[i, j])
            for name, values in arrays.items():
                row[name] = values[i, j].tolist()
            row["dynamic_pressure"] = (
                row["pressure_lattice"] - row["hydrostatic_reference_pressure"]
            )
            row["rho"] = 0.00125 + 0.99875 * np.clip(row["water_phase"], 0, 1)
            row["physical_velocity"] = (
                np.array(row["momentum_velocity_lattice"])
                + 0.5 * np.array(row["fluid_acceleration_lattice"])
            ).tolist()
            points.append(row)
    return dict(step=attempted, stage=stage, state=state_summary(s), points=points)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s = PointProbe(create_config())
    s._thermal_stability_check_enabled = True
    s.step(8270)
    snapshot = (device_snapshot(s), host_snapshot(s))
    rows = []
    failure = None
    for attempted in range(8271, 8289):
        try:
            rows.append(sparse_state(s, "before", attempted))
            s._collide_momentum()
            rows.append(sparse_state(s, "collision", attempted))
            s._stream()
            s._update_streamed_macroscopic_fields()
            rows.append(sparse_state(s, "macros", attempted))
            s._bound_water_phase()
            s._prepare_phase_gradients()
            s._update_pressure()
            rows.append(sparse_state(s, "pressure", attempted))
            s._advance_body()
            rows.append(sparse_state(s, "body", attempted))
            s._advance_thermal()
            s.steps += 1
        except RuntimeError as exc:
            failure = str(exc)
            break
    result = dict(records=rows, failure=failure, raw_fields_saved=False)
    (args.output_dir / "sparse_gap_trace.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(dict(failure=failure)), flush=True)
    # Determine whether the newly opening throat itself injects the first jump.
    controls = []
    for mode in (
        "baseline",
        "fixed_pose_after_8277",
        "stationary_after_8277",
        "suppress_new_throat_1767",
    ):
        restore(s, snapshot)
        history = []
        result = dict(mode=mode, history=history)
        try:
            for k in range(200):
                step = s.steps + 1
                s._advance_fluid()
                if not (
                    mode in ("fixed_pose_after_8277", "stationary_after_8277")
                    and step >= 8278
                ):
                    s._advance_body()
                    if mode == "suppress_new_throat_1767" and step >= 8277:
                        s.solid_mask[176, 7] = 1
                        s.solid_body_id[176, 7] = 1
                        s._prepare_phase_geometry()
                else:
                    if mode == "stationary_after_8277":
                        s.body_velocity.fill(0)
                        s.body_angular_velocity.fill(0)
                if mode not in ("fixed_pose_after_8277", "stationary_after_8277"):
                    s._advance_thermal()
                s.steps += 1
                if k % 5 == 0 or k >= 195:
                    state = state_summary(s)
                    history.append(state)
                    if state["speed_l1_max"] > 1:
                        break
        except RuntimeError as exc:
            result["error"] = str(exc)
        result["completed_steps"] = s.steps - 8270
        controls.append(result)
        (args.output_dir / "throat_controls.json").write_text(
            json.dumps(controls, indent=2) + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "history"}
                | dict(last=state_summary(s))
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
