#!/usr/bin/env python3
"""Verify the figure's masks and numbers with the unmodified Taichi kernels."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from iceflow2d_v3.phase_reconstruction import PhaseReconstruction
from build_method_figures import (
    analytic_cases, CORNER_MASK, DIRECTIONS, EXACT, HALF_MASK, PHI0,
)


@ti.data_oriented
class StencilProbe(PhaseReconstruction):
    """Minimal fields plus a rotated, bilinearly representable ice stripe.

    In material coordinates the coverage is the linear interpolant of a
    single occupied column (s=1), with empty neighboring columns (s=0).
    The gradient and connectivity methods themselves are inherited verbatim.
    """

    def __init__(self):
        self.nx = self.ny = 7
        self.water_phase = ti.field(ti.f32, shape=(7, 7))
        self.solid = ti.field(ti.i32, shape=(7, 7))
        self.body_count = 1
        self.body_active = ti.field(ti.i32, shape=1)
        self.thermal = SimpleNamespace(material_size=ti.Vector.field(2, ti.i32, shape=1))
        self.thermal.material_size[0] = [3, 9]
        self.config = SimpleNamespace(thermal=SimpleNamespace(solid_liquid_threshold=.5))
        self._initialize_phase_reconstruction()

    @ti.func
    def _is_fluid_cell(self, i, j):
        return self.solid[i, j] == 0

    @ti.func
    def _cell_point(self, i, j):
        return ti.Vector([ti.cast(i, ti.f32) - 3, ti.cast(j, ti.f32) - 3])

    @ti.func
    def _body_local_coordinates(self, point, b):
        return ti.Vector([(point.x + point.y - 1) / ti.sqrt(2.),
                          (point.y - point.x) / ti.sqrt(2.)])

    @ti.func
    def _sample_body_fraction(self, local, b):
        return ti.max(0., 1. - ti.abs(local.x))


def run_case(probe, name, accepted, remote=False):
    solid = np.ones((7, 7), dtype=np.int32)
    phi = np.full((7, 7), PHI0, dtype=np.float32)
    solid[3, 3] = 0
    for (dx, dy), keep in zip(DIRECTIONS, accepted):
        if keep:
            solid[3 + dx, 3 + dy] = 0
            phi[3 + dx, 3 + dy] = PHI0 + EXACT @ [dx, dy]
    if remote:
        solid[4, 4], phi[4, 4] = 0, .9
    probe.body_active[0] = int(remote)
    probe.solid.from_numpy(solid)
    probe.water_phase.from_numpy(phi)
    probe._prepare_phase_geometry()
    probe._prepare_phase_gradients()
    mask = int(probe.phase_donor_mask[3, 3])
    expected_mask = 1 << 4
    for (dx, dy), keep in zip(DIRECTIONS, accepted):
        if keep:
            expected_mask |= 1 << ((int(dx) + 1) * 3 + int(dy) + 1)
    assert mask == expected_mask, (name, mask, expected_mask)
    gradient = np.asarray(probe.phase_gradient_lattice[3, 3])
    expected = EXACT if name != "rank_one" else np.array([EXACT[0], 0.])
    np.testing.assert_allclose(gradient, expected, atol=5e-8, rtol=0)
    expected_rank = 1 if name == "rank_one" else 2
    assert int(probe.phase_gradient_data_rank[3, 3]) == expected_rank
    return {"case": name, "donor_mask": mask, "rank": expected_rank,
            "gradient_fp32": gradient.tolist(), "expected": expected.tolist(),
            "absolute_error": float(np.linalg.norm(gradient - expected)),
            "checks_passed": True}


def main():
    ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=2)
    analytic_cases()  # Independently checks both analytic fits and the controls.
    probe = StencilProbe()
    cases = [run_case(probe, "truncated", HALF_MASK),
             run_case(probe, "separated", CORNER_MASK, remote=True),
             run_case(probe, "rank_one", DIRECTIONS[:, 1] == 0)]
    source = ROOT / "iceflow2d_v3/phase_reconstruction.py"
    result = {
        "scope": "Actual connectivity and gradient kernels on manufactured local data",
        "backend": "Taichi CPU, float32",
        "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
    }
    output = ROOT / "paper/figures/gradient_kernel_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
