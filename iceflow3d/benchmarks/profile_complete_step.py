"""Time a complete fine-grid step whose thermal conduction/melting is due.

Every sample restores the same in-memory state. JIT, initialization, diagnostics,
and transfers are excluded. The thermal interval stays at eight LBM steps: this
measures the expensive eighth step, never an eight-step amortized average.
Only 2D diagnostic sections and scalar JSON are written to disk.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceFlow3D  # noqa: E402
from iceflow3d.checkpoint import source_hashes  # noqa: E402
from iceflow3d.examples.dambreak_four_ice_melting_3d import create_config, parse_args  # noqa: E402
from iceflow3d.benchmarks.profile_step import (  # noqa: E402
    StageTimer,
    capture,
    restore,
    timed_step,
    kernel_profile,
    condition_device,
)


def section(sim, path):
    k = sim.nz // 2
    np.savez_compressed(
        path,
        phi=sim.water_phase.to_numpy()[:, :, k],
        velocity=sim.velocity_lattice.to_numpy()[:, :, k],
        pressure=sim.pressure_lattice.to_numpy()[:, :, k],
        water_energy=sim.water_energy.to_numpy()[:, :, k],
        water_volume=sim.water_volume.to_numpy()[:, :, k],
        solid_id=sim.solid_body_id.to_numpy()[:, :, k],
        body_origin=sim.rigid.origin,
        body_quaternion=sim.rigid.q,
        body_velocity=sim.rigid.velocity,
        body_omega=sim.rigid.omega,
        body_mass=sim.rigid.mass,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("wall", "kernels"), default="kernels")
    p.add_argument("--resolution-x", type=int, default=450)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup-steps", type=int, default=31)
    p.add_argument("--validation-steps", type=int, default=0)
    p.add_argument("--case", choices=("initial", "moving"), default="initial")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        default_fp=ti.f32,
        offline_cache=True,
        kernel_profiler=a.mode == "kernels",
    )
    hashes = source_hashes()
    cfg = create_config(parse_args(["--resolution-x", str(a.resolution_x)]))
    if a.case == "moving":
        # A separate stress case, not a change to the dam-break defaults.
        # Tilted, translating and rotating ice exposes nodes during the timed
        # step, exercising the geometry/refill path even at this early time.
        cfg = replace(
            cfg,
            ice_bodies=tuple(
                replace(
                    b,
                    quaternion_wxyz=(np.cos(0.09), 0.0, np.sin(0.09), 0.0),
                    velocity_lattice=(0.01, 0.0, 0.003),
                    angular_velocity_lattice=(0.0, 0.0003, 0.0),
                )
                for b in cfg.bodies
            ),
        )
    assert (
        a.warmup_steps % cfg.thermal.update_interval_lbm_steps
        == cfg.thermal.update_interval_lbm_steps - 1
    )
    start = time.perf_counter()
    s = IceFlow3D(cfg)
    s.step(a.warmup_steps)
    state = capture(s)
    s.step()
    ti.sync()
    assert s._thermal_pending == 0
    restore(s, state)
    print(
        f"ready: {s.shape}, start={s.steps}, pending={s._thermal_pending}, setup={time.perf_counter() - start:.2f}s",
        flush=True,
    )
    if a.mode == "kernels":
        result = kernel_profile(s, state, 1, a.repeats, condition=True)
    else:
        samples = []
        for i in range(a.repeats):
            restore(s, state)
            condition_device(s)
            samples.append(timed_step(s, 1))
            assert s._thermal_pending == 0
            print(f"full step {i}: {samples[-1]:.4f} ms", flush=True)
        result = dict(wall_ms=samples, wall_median_ms=statistics.median(samples))
    # A separate synchronized attribution; do not confuse this instrumentation
    # with the production wall time above or add it to CUDA event time.
    timer = StageTimer(s)
    restore(s, state)
    condition_device(s)
    timer.install()
    timed_step(s, 1)
    timer.remove()
    result["stages"] = [
        dict(path=list(k), calls=v[0], inclusive_ms=v[1], exclusive_ms=v[2])
        for k, v in sorted(timer.records.items(), key=lambda x: -x[1][1])
    ]
    assert s._thermal_pending == 0
    for name in (
        "_advance_advection",
        "_advance_conduction",
        "_exchange_heat",
        "_apply_melting",
    ):
        assert any(row["path"][-1] == "flow." + name for row in result["stages"]), name
    section(s, a.output.with_suffix(".npz"))
    diagnostics = s.diagnostics()
    if a.validation_steps:
        restore(s, state)
        s.step(a.validation_steps)
        s.flush_thermal()
        section(s, a.output.with_suffix(".validation.npz"))
        result["validation"] = s.diagnostics()
    assert hashes == source_hashes()
    record = dict(
        configuration=cfg.to_dict(),
        source_sha256=hashes,
        mode=a.mode,
        case=a.case,
        start_step=a.warmup_steps,
        thermal_pending_before=cfg.thermal.update_interval_lbm_steps - 1,
        thermal_pending_after=0,
        thermal_dt_s=cfg.thermal.update_interval_lbm_steps * s.scales.dt_s,
        device_conditioning="32 untimed collisions after restoring each state; output caches overwritten by step",
        measurement=result,
        diagnostics=diagnostics,
        validation_steps=a.validation_steps,
    )
    a.output.write_text(json.dumps(record, indent=2) + "\n")
    if a.mode == "kernels":
        print(json.dumps(result["kernels"][:12], indent=2), flush=True)
    print("saved", a.output, flush=True)


if __name__ == "__main__":
    main()
