"""Compile a complete CUDA step and inspect PTX for hidden FP64 arithmetic."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def worker():
    import taichi as ti
    from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
    from iceflow3d.checkpoint import source_hashes

    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        default_fp=ti.f32,
        offline_cache=False,
        print_kernel_asm=True,
    )
    cfg = IceFlowConfig(
        nx=30,
        ny=20,
        nz=12,
        domain_size_m=(0.03, 0.02, 0.012),
        boundary_cells=1,
        water_width=12,
        water_height=16,
        floor_water_depth_cells=5,
        ice_bodies=(IceBodyConfig((20, 5, 6), (4, 4, 4)),),
        phase_warmup_steps=2,
        backend="cuda",
    )
    s = IceFlow3D(cfg)
    s.step(8)
    s.rigid.origin[0] += [0.8, 0.25, 0.2]
    s.rigid.upload()
    s._refresh_geometry()
    s._project_refill()
    s._measure_outflow()
    s.snapshot()
    ti.sync()
    Path("source.json").write_text(json.dumps(source_hashes()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", action="store_true")
    args = p.parse_args()
    if args.worker:
        worker()
        return
    if args.output is None:
        p.error("--output is required")
    # Run with a private working directory: PTX dumps never clutter the repo.
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="iceflow-home-ptx-") as directory:
        folder = Path(directory)
        with (folder / "compile.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(script), "--worker"],
                cwd=folder,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        modules = []
        for file in sorted(folder.glob("*.ptx")):
            text = file.read_text()
            modules.append(
                dict(
                    name=file.name,
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    entry_points=re.findall(r"\.entry\s+(\w+)", text),
                    fp32_mentions=len(re.findall(r"\.f32\b", text)),
                    fp64_mentions=len(re.findall(r"\.f64\b", text)),
                )
            )
        result = dict(
            source_sha256=json.loads((folder / "source.json").read_text()),
            modules=modules,
            fp64_mentions=sum(row["fp64_mentions"] for row in modules),
        )
        result["passed"] = bool(modules) and result["fp64_mentions"] == 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        args.output.with_suffix(".log").write_text((folder / "compile.log").read_text())
        print(
            json.dumps(
                dict(
                    modules=len(modules),
                    fp64_mentions=result["fp64_mentions"],
                    passed=result["passed"],
                )
            )
        )
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
