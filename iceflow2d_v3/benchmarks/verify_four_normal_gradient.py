"""Check whether the underresolved normal closure alone causes the first burst."""

import argparse
import json
from pathlib import Path
import numpy as np

from iceflow2d_v3.benchmarks.verify_four_ice_opening_controls import OpeningProbe
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    simulation = OpeningProbe(create_config())
    simulation._thermal_stability_check_enabled = True
    simulation.step(8520)
    snapshot = (device_snapshot(simulation), host_snapshot(simulation))
    original_prepare = simulation._prepare_phase_gradients
    mode = "baseline"

    def prepare():
        original_prepare()
        for field_name, enabled in (
            (
                "source_gradient_lattice",
                mode in ("both_normal_zero", "source_normal_zero"),
            ),
            (
                "phase_gradient_lattice",
                mode in ("both_normal_zero", "phase_normal_zero"),
            ),
        ):
            if enabled:
                field = getattr(simulation, field_name)
                values = field.to_numpy()
                values[148:162, 3, 1] = 0
                field.from_numpy(values)

    simulation._prepare_phase_gradients = prepare
    results = []
    for mode in (
        "baseline",
        "both_normal_zero",
        "source_normal_zero",
        "phase_normal_zero",
    ):
        restore(simulation, snapshot)
        prepare()
        first_bad = None
        error = None
        for k in range(600):
            try:
                step = simulation.steps + 1
                simulation._collide_momentum()
                simulation._stream()
                simulation._update_streamed_macroscopic_fields()
                state = state_summary(simulation)
                if state["speed_l1_max"] > 1 or not np.isfinite(state["speed_l1_max"]):
                    first_bad = {"attempted_step": step, "state_before_geometry": state}
                    break
                simulation._bound_water_phase()
                simulation._prepare_phase_gradients()
                simulation._update_pressure()
                simulation._advance_body()
                simulation._advance_thermal()
                simulation.steps += 1
            except RuntimeError as exc:
                error = str(exc)
                break
        result = {
            "mode": mode,
            "completed_steps": simulation.steps - 8520,
            "first_bad": first_bad,
            "error": error,
            "last": state_summary(simulation),
        }
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
