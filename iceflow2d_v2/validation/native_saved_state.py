"""Diagnostic native-graph probe on a recorded state, with explicit provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import taichi as ti

from iceflow2d_v2.checkpoint import Snapshot, fields
from iceflow2d_v2.config import Config
from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
from iceflow2d_v2.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = Snapshot.load(args.checkpoint)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    sim = Simulation(Config(**saved.metadata['config']))
    saved.restore(sim, check_source=False)  # Explicit diagnostic, never continuation.
    sim.enable_graph()
    before = Snapshot(sim)
    native = CudaKernelSchedule(sim)
    unchanged = all(np.array_equal(field.to_numpy(), before.arrays[name]) for name, field in fields(sim).items())
    assert unchanged and sim.steps == saved.metadata['steps']
    sim.schedule = native
    sim.step(16)
    diagnostics = sim.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'diagnostic_only': True, 'recording_preserved_every_state_array': unchanged,
              'input_checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'input_sources': saved.metadata['source_sha256'], 'output_sources': sim.source_sha256,
              'execution': native.metadata, 'diagnostics': diagnostics}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    native.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
