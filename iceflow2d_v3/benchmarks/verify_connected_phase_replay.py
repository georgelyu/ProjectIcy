"""Same-state check of the diagnosed ghost error; all grid copies stay in RAM."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    PointProbe,
    device_snapshot,
)
from iceflow2d_v3.body_state import BodyField


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original = PointProbe(replace(create_config(), phase_gradient_scheme="legacy"))
    original._thermal_stability_check_enabled = True
    original.step(8544)
    before = original.point_rows()[0]
    state = device_snapshot(original)
    repaired = PointProbe(
        replace(
            original.config, phase_gradient_scheme="connected_ls", phase_warmup_steps=0
        )
    )
    for prefix, name, values in state:
        owner = repaired if prefix == "flow" else repaired.thermal
        field = getattr(owner, name)
        if isinstance(field, BodyField):
            field = field.field
        field.from_numpy(values)
    repaired.steps = 8544
    repaired._prepare_phase_geometry()
    repaired._prepare_phase_gradients()
    gradient = np.array(repaired.source_gradient_lattice[180, 3])
    assert np.linalg.norm(gradient) < 1e-5
    assert np.hypot(before["grad_phi_x"], before["grad_phi_y"]) > 0.01
    repaired.body_velocity.fill(0)
    repaired.body_angular_velocity.fill(0)
    rows = []
    for k in range(240):
        repaired._advance_fluid()
        repaired.steps += 1
        if k % 24 == 0 or k == 239:
            u = repaired.physical_velocity_lattice.to_numpy()
            active = (repaired.wall_mask.to_numpy() == 0) & (
                repaired.solid_mask.to_numpy() == 0
            )
            maximum = float(np.abs(u[active]).sum(-1).max())
            assert np.isfinite(u).all() and maximum < 1.0
            rows.append(
                dict(
                    step=repaired.steps,
                    max_speed_l1=maximum,
                    point_velocity_lattice=u[180, 3].tolist(),
                )
            )
    result = dict(
        original_step=8544,
        original_point=before,
        repaired_source_gradient_lattice=gradient.tolist(),
        matched_frozen_geometry_steps=240,
        completed=True,
        history=rows,
        raw_fields_saved=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("original_point", "history")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
