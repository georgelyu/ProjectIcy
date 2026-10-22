"""Inventory GPU fields and compare converted modules to an FP64 reference.

The reference restores FP64 thermal state/flux arithmetic and rigid force,
mass and inertia reductions, the modules converted from FP64 in this change.
Flow, phase and geometry stay FP32 in both runs; molecular flow precision is
checked separately against the FP64 eigenmode reference in validate_home.py.
Production files are never changed. No reference path exists in the solver.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def reference_source(name, value):
    if name == "thermal.py":
        value = value.replace("ti.f32", "ti.f64").replace("np.float32", "np.float64")
        # Preserve the old f32 geometry and temperature sampling interfaces.
        value = value.replace(
            "self.sampled_ice_temperature = ti.field(ti.f64",
            "self.sampled_ice_temperature = ti.field(ti.f32",
        )
        value = value.replace(
            "self.max_outflow = ti.field(ti.f64", "self.max_outflow = ti.field(ti.f32"
        )
        for expression in ("temperature", "J", "base", "self.material_size[b]", "t"):
            value = value.replace(
                f"ti.cast({expression}, ti.f64)", f"ti.cast({expression}, ti.f32)"
            )
    elif name == "simulator.py":
        for field in (
            "hydrodynamic_force",
            "hydrodynamic_torque",
            "reduced_mass",
            "reduced_first",
            "reduced_second",
        ):
            value = re.sub(rf"(self\.{field} = [^\n]*)ti.f32", r"\1ti.f64", value)
        start = value.index("    @ti.kernel\n    def _reduce_mass")
        end = value.index("    @ti.func\n    def _scan_stability_at", start)
        reduction = value[start:end].replace("ti.f32", "ti.f64")
        reduction = reduction.replace(
            "ti.cast(point - 0.5, ti.f64)", "ti.cast(point - 0.5, ti.f32)"
        )
        reduction = reduction.replace(
            "ti.cast(point + 0.5, ti.f64)", "ti.cast(point + 0.5, ti.f32)"
        )
        value = value[:start] + reduction + value[end:]
        # Preserve the precision of the reference reductions in the batched
        # host download as well. Geometry uploads intentionally remain FP32.
        value = value.replace(
            "self, values: ti.types.ndarray(dtype=ti.f32, ndim=2), mass:",
            "self, values: ti.types.ndarray(dtype=ti.f64, ndim=2), mass:",
        )
    elif name == "rigid.py":
        for columns in (6, 19):
            value = value.replace(
                f"np.empty((s.body_count, {columns}), np.float32)",
                f"np.empty((s.body_count, {columns}), np.float64)",
            )
    return value


def worker(args):
    import numpy as np
    import taichi as ti
    from iceflow3d import IceFlow3D
    from iceflow3d.checkpoint import _fields, source_hashes
    from iceflow3d.examples.dambreak_four_ice_melting_3d import (
        create_config,
        parse_args,
    )

    cfg = create_config(parse_args(["--resolution-x", str(args.resolution)]))
    s = IceFlow3D(cfg)
    inventory = []
    for name, field in _fields(s).items():
        n = int(np.prod(field.shape)) * getattr(field, "n", 1) * getattr(field, "m", 1)
        dtype = str(field.dtype)
        size = 8 if dtype in ("f64", "i64") else (1 if dtype in ("i8", "u8") else 4)
        inventory.append(
            dict(name=name, dtype=dtype, shape=field.shape, scalars=n, bytes=n * size)
        )
    samples = []
    for _ in range(0, args.steps, 200):
        s.step(min(200, args.steps - s.steps))
        samples.append(s.diagnostics())
        print(
            json.dumps(
                dict(
                    precision=args.worker,
                    step=s.steps,
                    speed=samples[-1]["maximum_fluid_speed_lattice"],
                )
            ),
            flush=True,
        )
    s.flush_thermal()
    ti.sync()
    fields = s.snapshot()
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **fields,
        body_origin=s.rigid.origin,
        body_mass=s.rigid.mass,
        body_velocity=s.rigid.velocity,
    )
    result = dict(
        precision=args.worker,
        configuration=cfg.to_dict(),
        source_sha256=source_hashes(),
        dt_s=s.scales.dt_s,
        inventory=inventory,
        diagnostics=s.diagnostics(),
        samples=samples,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, default=150)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=("fp32", "fp64"))
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[2]
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="iceflow-home-precision-") as tmp:
        for precision in ("fp32", "fp64"):
            destination = Path(tmp) / precision
            for source in (root / "iceflow3d").rglob("*.py"):
                if any(p in source.parts for p in ("outputs", "__pycache__")):
                    continue
                target = destination / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                value = source.read_text()
                if precision == "fp64" and source.parent == root / "iceflow3d":
                    value = reference_source(source.name, value)
                target.write_text(value)
            output = args.output.with_name(args.output.stem + "_" + precision + ".json")
            command = [
                sys.executable,
                "-m",
                "iceflow3d.benchmarks.audit_gpu_precision",
                "--worker",
                precision,
                "--resolution",
                str(args.resolution),
                "--steps",
                str(args.steps),
                "--output",
                str(output),
            ]
            subprocess.run(command, cwd=destination, check=True)
            outputs[precision] = output
    import numpy as np

    left, right = [np.load(outputs[k].with_suffix(".npz")) for k in ("fp32", "fp64")]
    errors = {}
    for name in left.files:
        a, b = left[name].astype(float), right[name].astype(float)
        difference = a - b
        errors[name] = dict(
            max_abs=float(abs(difference).max(initial=0)),
            relative_l2=float(
                np.linalg.norm(difference.ravel())
                / max(np.linalg.norm(b.ravel()), 1e-30)
            ),
        )
    fluid = (left["solid_body_id"] == -1) & (right["solid_body_id"] == -1)
    fluid_errors = {}
    for name in ("water_phase", "velocity_m_s", "pressure_pa", "temperature_c"):
        a, b = left[name][fluid].astype(float), right[name][fluid].astype(float)
        fluid_errors[name] = dict(
            max_abs=float(abs(a - b).max(initial=0)),
            relative_l2=float(
                np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30)
            ),
        )
    inventory = json.loads(outputs["fp32"].read_text())["inventory"]
    result = dict(
        steps=args.steps,
        resolution=args.resolution,
        comparisons=errors,
        valid_fluid_comparisons=fluid_errors,
        gpu_fp64_fields=[r for r in inventory if r["dtype"] == "f64"],
        gpu_field_bytes=sum(r["bytes"] for r in inventory),
        fp32_result=str(outputs["fp32"]),
        fp64_reference=str(outputs["fp64"]),
    )
    result["passed"] = (
        errors["water_phase"]["relative_l2"] < 0.01
        and errors["velocity_m_s"]["relative_l2"] < 0.02
        and errors["temperature_c"]["max_abs"] < 0.05
        and errors["body_mass"]["relative_l2"] < 0.001
        and fluid_errors["pressure_pa"]["relative_l2"] < 0.01
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
