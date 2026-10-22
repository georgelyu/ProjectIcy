"""Compare launch sizes on unchanged post-collision inputs (CUDA events)."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceFlow3D  # noqa: E402
from iceflow3d.checkpoint import source_hashes  # noqa: E402
from iceflow3d.examples.dambreak_four_ice_melting_3d import create_config, parse_args  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ti.init(
        arch=ti.cuda, enable_fallback=False, kernel_profiler=True, offline_cache=True
    )
    s = IceFlow3D(create_config(parse_args(["--resolution-x", "450"])))
    s.step(31)
    sizes = (32, 64, 128, 256)
    for size in sizes:
        s._stream_recover(size)
    records = []
    for size in (*sizes, *reversed(sizes)):
        for _ in range(20):
            s._stream_recover(size)
        ti.sync()
        ti.profiler.clear_kernel_profiler_info()
        for _ in range(30):
            s._stream_recover(size)
        ti.sync()
        prog = ti.lang.impl.get_runtime().prog
        prog.sync_kernel_profiler()
        prog.update_kernel_profiler()
        rows = prog.get_kernel_profiler_records()
        record = {
            "block_size": size,
            "median_ms": statistics.median(r.kernel_time for r in rows),
            "registers": rows[0].register_per_thread,
            "active_blocks_per_sm": rows[0].active_blocks_per_multiprocessor,
        }
        print(record, flush=True)
        records.append(record)
    a.output.write_text(
        json.dumps(
            {"source_sha256": source_hashes(), "measurements": records}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
