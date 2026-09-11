"""Report warmed CUDA step time and unique Taichi field payload (no output I/O)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iceflow2d import IceFlow2D
from iceflow2d.examples.coupled_falling_ice_melting_2d import create_config, parse_args


def field_inventory(simulation):
    seen = set()
    inventory = []
    for prefix, owner in (("flow", simulation), ("thermal", simulation.thermal)):
        for name, value in vars(owner).items():
            if not isinstance(value, (ti.Field, ti.MatrixField)) or id(value) in seen:
                continue
            seen.add(id(value))
            component_count = getattr(value, "n", 1) * getattr(value, "m", 1)
            element_count = component_count
            for dimension in value.shape:
                element_count *= dimension
            inventory.append(
                {
                    "name": f"{prefix}.{name}",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "bytes": element_count * int(str(value.dtype)[1:]) // 8,
                }
            )
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution-x", type=int, default=100)
    parser.add_argument("--resolution-y", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    simulation = IceFlow2D(
        create_config(
            parse_args(
                [
                    "--resolution-x",
                    str(args.resolution_x),
                    "--resolution-y",
                    str(args.resolution_y),
                ]
            )
        )
    )
    simulation.step(32)
    ti.sync()
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        simulation.step(args.steps)
        ti.sync()
        durations.append((time.perf_counter() - started) / args.steps)
    inventory = field_inventory(simulation)
    result = {
        "grid": list(simulation.config.resolution),
        "warmup_steps": 32,
        "steps_per_repeat": args.steps,
        "seconds_per_step": durations,
        "median_seconds_per_step": statistics.median(durations),
        "field_count": len(inventory),
        "field_payload_bytes": sum(item["bytes"] for item in inventory),
        "fields": inventory,
    }
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
