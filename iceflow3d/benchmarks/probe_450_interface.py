"""Replay a complete checkpoint and record the downstream interface locally.

Overrides are diagnostic ablations, explicitly recorded; they are not a
production restart or a substitute for a full run from a consistent state.
"""
# ruff: noqa: E741 -- I is a Taichi vector grid index.

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti


@ti.kernel
def extract_region(
    sim: ti.template(), macro: ti.types.ndarray(), populations: ti.types.ndarray()
):
    for x, y, z in ti.ndrange(19, 39, 9):
        I = ti.Vector([x + 428, y + 82, z + 40])
        macro[x, y, z, 0] = sim.water_phase[I]
        macro[x, y, z, 1] = sim.pressure_lattice[I]
        macro[x, y, z, 2] = sim.old_pressure[I]
        for a in ti.static(range(3)):
            macro[x, y, z, 3 + a] = sim.velocity_lattice[I][a]
            macro[x, y, z, 6 + a] = sim.old_velocity[I][a]
            macro[x, y, z, 9 + a] = sim.force_density[I][a]
            macro[x, y, z, 12 + a] = sim.phase_gradient_lattice[I][a]
            macro[x, y, z, 15 + a] = sim.phase_mass_flux[I][a]
        macro[x, y, z, 18] = sim.solid_body_id[I]
        for q in range(27):
            populations[x, y, z, q, 0] = sim.f[I, q]
            populations[x, y, z, q, 1] = sim.f_post[I, q]
            populations[x, y, z, q, 2] = sim.h[I, q]
            populations[x, y, z, q, 3] = sim.h_post[I, q]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--overrides", default="{}")
    p.add_argument("--end-time-s", type=float, default=0.16)
    p.add_argument("--local-from-step", type=int, default=17280)
    p.add_argument("--local-stride", type=int, default=4)
    p.add_argument("--save-checkpoint-step", type=int, default=16800)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from iceflow3d.checkpoint import save_checkpoint, source_hashes
    from iceflow3d.config import (
        IceBodyConfig,
        IceFlowConfig,
        PhaseChangeProperties,
        ThermalConfig,
    )
    from iceflow3d.simulator import IceFlow3D

    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    overrides = json.loads(args.overrides)
    record = {
        "status": "initializing",
        "source_root": str(args.source_root.resolve()),
        "source_sha256": source_hashes(),
        "checkpoint": str(args.checkpoint.resolve()),
        "diagnostic_overrides": overrides,
        "local_region_origin": [428, 82, 40],
        "local_macro_channels": [
            "phi",
            "pressure",
            "old_pressure",
            "ux",
            "uy",
            "uz",
            "old_ux",
            "old_uy",
            "old_uz",
            "Fx",
            "Fy",
            "Fz",
            "grad_phi_x",
            "grad_phi_y",
            "grad_phi_z",
            "Jx",
            "Jy",
            "Jz",
            "body_id",
        ],
        "local_population_channels": ["f", "f_post", "h", "h_post"],
    }

    def publish():
        record["wall_time_s"] = time.perf_counter() - start
        temp = args.output / "probe.json.tmp"
        temp.write_text(json.dumps(record, indent=2))
        temp.replace(args.output / "probe.json")

    publish()
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        values = dict(metadata["configuration"])
        thermal = dict(values["thermal"])
        thermal["properties"] = PhaseChangeProperties(**thermal["properties"])
        values["thermal"] = ThermalConfig(**thermal)
        values["ice_bodies"] = tuple(IceBodyConfig(**v) for v in values["ice_bodies"])
        config = replace(IceFlowConfig(**values), **overrides)
        sim = IceFlow3D(replace(config, phase_warmup_steps=0))
        sim.config = config
        for name in metadata["fields"]:
            getattr(sim, name).from_numpy(archive["field/" + name])
        for name in metadata["rigid"]:
            setattr(sim.rigid, name, archive["rigid/" + name].copy())
        sim.steps = metadata["steps"]
        sim._thermal_pending = metadata["thermal_pending"]
        sim.pending_cells = metadata["pending_cells"]
        sim.initial_water_volume_m3 = metadata["initial_water_volume_m3"]
        sim.initial_phase_volume_m3 = metadata["initial_phase_volume_m3"]
        sim.initial_ice_mass_kg = archive["initial_ice_mass_kg"].copy()
    record["configuration"] = config.to_dict()
    record["status"] = "running"
    macro = np.zeros((19, 39, 9, 19), np.float32)
    populations = np.zeros((19, 39, 9, 27, 4), np.float32)

    def local(name):
        extract_region(sim, macro, populations)
        np.savez_compressed(
            args.output / name, macro=macro, populations=populations, step=sim.steps
        )

    target = round(args.end_time_s / sim.scales.dt_s)
    try:
        with (args.output / "timeline.jsonl").open("w") as stream:
            while sim.steps < target:
                if sim.steps % 120 == 0:
                    row = sim.diagnostics()
                    row["wall_time_s"] = time.perf_counter() - start
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    record["last_diagnostics"] = row
                    publish()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "step",
                                    "time_s",
                                    "maximum_fluid_speed_lattice",
                                    "wall_time_s",
                                )
                            }
                        ),
                        flush=True,
                    )
                if sim.steps == args.save_checkpoint_step:
                    save_checkpoint(sim, args.output / "replay_checkpoint.npz")
                if sim.steps >= args.local_from_step:
                    local(f"local_{sim.steps:09d}.npz")
                    count = min(args.local_stride, target - sim.steps)
                else:
                    count = min(
                        120, target - sim.steps, args.local_from_step - sim.steps
                    )
                    if sim.steps < args.save_checkpoint_step:
                        count = min(count, args.save_checkpoint_step - sim.steps)
                sim.step(count)
        record["status"] = "completed"
        record["last_diagnostics"] = sim.diagnostics()
        save_checkpoint(sim, args.output / "final_checkpoint.npz")
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["last_diagnostics"] = sim.diagnostics()
        local("failure_local.npz")
        np.savez_compressed(
            args.output / "failure_macro.npz",
            water_phase=sim.water_phase.to_numpy(),
            velocity=sim.velocity_lattice.to_numpy(),
            pressure=sim.pressure_lattice.to_numpy(),
            force=sim.force_density.to_numpy(),
            phase_mass_flux=sim.phase_mass_flux.to_numpy(),
            gradient=sim.phase_gradient_lattice.to_numpy(),
            solid_body_id=sim.solid_body_id.to_numpy(),
        )
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
