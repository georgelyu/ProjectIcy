"""Finish the running dry-ice simulation, render, encode, and verify it.

This coordinator never starts a simulation or changes its parameters. A failed
simulation is rendered only through its last complete saved frame and reported
as partial, never silently promoted to a six-second result.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import psutil
from PIL import Image

from .dryice_export import sha256, write_json
from .dryice_finish import finish


def alive(pid, term):
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE and term in " ".join(p.cmdline())
    except psutil.Error:
        return False


def validate_simulation(root, sim, expected_end):
    frames = sim["frames"]
    dt, dx = sim["dt_s"], sim["dx_m"]
    stride = round(.01 / dt)
    expected_steps = round(expected_end / dt)
    expected_count = expected_steps // stride + 1
    complete = sim["status"] == "completed"
    assert sim["physical_grid"] == [216] * 3
    assert sim["configuration"]["interface_width"] == 5
    assert abs(dt - 1 / 120000) < 1e-14
    assert stride == 1200
    if complete:
        assert sim["current_step"] == expected_steps
        assert len(frames) == expected_count
    assert frames and [f["index"] for f in frames] == list(range(len(frames)))
    source = Path(sim["source_root"]) / "iceflow3d"
    for path, digest in sim["source_sha256"].items():
        assert sha256(source / path) == digest, path
    summary = dict(status="validating", complete_requested_run=complete,
                   expected_end_time_s=expected_end, expected_frames=expected_count,
                   saved_frames=len(frames), checked_frames=0, errors=[],
                   source_hashes_match=True, finite_fluid_fields=True,
                   gas_temperature_is_prescribed=True)
    path = root / "simulation" / "verification.json"
    for index, frame in enumerate(frames):
        field_path = root / "simulation" / frame["path"]
        assert sha256(field_path) == frame["sha256"], field_path
        assert frame["step"] == index * stride
        assert abs(frame["time_s"] - index * .01) < 1e-12
        with h5py.File(field_path, "r") as f:
            assert f.attrs["axis_order"] == "zyx"
            assert int(f.attrs["step"]) == frame["step"]
            assert abs(float(f.attrs["dx_m"]) - dx) < 1e-15
            ids, phase = f["fields/solid_body_id"][:], f["fields/water_phase"][:]
            assert ids.shape == phase.shape == (216, 216, 216)
            fluid = ids == -1
            assert np.all((phase[fluid] >= -1e-6) & (phase[fluid] <= 1.000001))
            for name, dataset in f["fields"].items():
                a = dataset[:]
                assert a.shape == ((216, 216, 216, 3) if name == "velocity_m_s" else (216, 216, 216))
                assert np.isfinite(a[fluid]).all(), (frame["index"], name)
                if name == "temperature_c":
                    assert np.allclose(a[fluid & (phase < .5)], f.attrs["gas_temperature_c"], rtol=0, atol=1e-5)
            for dataset in f["body"].values():
                assert np.isfinite(dataset[:]).all()
            for rotation in f["body/rotation"][:]:
                assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        summary["checked_frames"] = index + 1
        if index % 20 == 0:
            write_json(path, summary)
            print(f"VERIFIED_DATA {index + 1}/{len(frames)}", flush=True)
    summary.update(status="passed", last_step=frames[-1]["step"], last_time_s=frames[-1]["time_s"],
                   final_diagnostics=sim.get("last_diagnostics"),
                   limitation="Data-integrity checks, not quantitative validation of conservation, gas heat transfer, or bubble physics.")
    write_json(path, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--blender", default="/usr/local/blender-5.0.1/blender")
    p.add_argument("--expected-end-time-s", type=float, default=6)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--science-samples", type=int, default=48)
    args = p.parse_args()
    root = args.root.resolve()
    source_root = Path(__file__).resolve().parents[2]
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = root / "pipeline.json"
    state = dict(status="waiting_for_simulation", pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat(), source_root=str(source_root))
    write_json(path, state)
    try:
        process_file = root / "processes.json"
        processes = json.loads(process_file.read_text()) if process_file.exists() else {}
        while True:
            metadata = root / "simulation" / "simulation.json"
            if metadata.exists():
                sim = json.loads(metadata.read_text())
                if sim["status"] in ("completed", "failed"):
                    break
            if processes.get("simulation_pid") and not alive(processes["simulation_pid"], "dryice_export"):
                # Keep original metadata intact; record process loss here.
                sim = json.loads(metadata.read_text())
                sim["status"] = "failed"
                sim["error"] = "simulation process exited without a terminal status; inspect logs/simulation.log"
                break
            time.sleep(10)
        state.update(status="validating_simulation", simulation_status=sim["status"], simulation_error=sim.get("error"))
        write_json(path, state)
        verification = validate_simulation(root, sim, args.expected_end_time_s)
        # Allow the existing CPU mesher to publish its final frame before any
        # recovery invocation, avoiding concurrent writers to the mesh cache.
        for pid_key, module in (("meshes_pid", "dryice_mesh"), ("portable_pid", "dryice_portable")):
            if sim["status"] == "failed" and json.loads(metadata.read_text())["status"] not in ("completed", "failed"):
                if alive(processes.get(pid_key), module):
                    psutil.Process(processes[pid_key]).terminate()
            while alive(processes.get(pid_key), module):
                time.sleep(5)
        state["status"] = "completing_meshes"
        write_json(path, state)
        with (logs / "meshes_final.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "iceflow3d.rendering.dryice_mesh",
                "--simulation", str(root / "simulation"), "--output", str(root / "meshes")],
                cwd=source_root, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "completing_portable_archive"
        write_json(path, state)
        with (logs / "portable_final.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "-m", "iceflow3d.rendering.dryice_portable",
                "--simulation", str(root / "simulation"), "--output", str(root / "paraview")],
                cwd=source_root, stdout=log, stderr=subprocess.STDOUT, check=True)
        for frame in sim["frames"]:
            mesh = root / "meshes" / f"frame_{frame['index']:06d}.npz"
            info = json.loads(mesh.with_suffix(".json").read_text())
            assert info["source_sha256"] == frame["sha256"]
            assert info["mesh_sha256"] == sha256(mesh)
            assert info["step"] == frame["step"]
        state["status"] = "rendering"
        write_json(path, state)
        command = [args.blender, "--factory-startup", "-b", "--python-exit-code", "1",
                   "--python", str(Path(__file__).with_name("dryice_blender.py")), "--",
                   "--cache", str(root / "meshes"), "--output", str(root / "render"),
                   "--luts", str(root / "colormaps.json"), "--width", str(args.width),
                   "--height", str(args.height), "--samples", str(args.samples),
                   "--science-samples", str(args.science_samples), "--device", "OPTIX", "--all"]
        with (logs / "blender.log").open("a") as log:
            child = subprocess.Popen(command, cwd=source_root, stdout=log, stderr=subprocess.STDOUT)
            state.update(blender_pid=child.pid, blender_command=command)
            write_json(path, state)
            status = child.wait()
            if status:
                raise RuntimeError(f"Blender exited {status}; see {logs / 'blender.log'}")
        count = len(sim["frames"])
        for kind in ("beauty", "science"):
            images = sorted((root / "render" / (kind + "_raw") / "frames").glob("frame_*.png"))
            assert len(images) == count
            for i, image_path in enumerate(images):
                assert image_path.stem == f"frame_{i:06d}"
                with image_path.open("rb") as f:
                    assert f.read(25)[24] == 16
                with Image.open(image_path) as im:
                    assert im.size == (args.width, args.height)
                    im.verify()
        state["status"] = "compositing_and_encoding"
        write_json(path, state)
        delivery = finish(root, root / "render")
        assert all(n == count for n in delivery["frame_count"].values())
        state.update(status="completed" if verification["complete_requested_run"] else "partial_completed",
                     frame_count=count, physical_time_s=sim["frames"][-1]["time_s"],
                     delivery=str(root / "render" / "delivery.json"),
                     verification=str(root / "simulation" / "verification.json"))
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, state)


if __name__ == "__main__":
    main()
