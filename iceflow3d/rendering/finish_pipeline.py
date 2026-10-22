"""Wait for the full simulation, audit it, render in Cycles, then encode 50 fps."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulate import write_json


def audit_simulation(root):
    directory = root / "simulation"
    record = json.loads((directory / "simulation.json").read_text())
    assert record["status"] == "completed", record["status"]
    cfg = record["configuration"]
    assert (cfg["nx"], cfg["ny"], cfg["nz"]) == (450, 150, 90)
    assert cfg["check_stability"]
    end = round(record["requested_end_time_s"] / record["dt_s"])
    stride = round(record["frame_interval_s"] / record["dt_s"])
    count = end // stride + 1
    assert end % stride == 0
    assert not cfg["gas_state_repair"]
    assert record["current_step"] == record["target_steps"] == end
    assert abs(record["current_time_s"] - record["requested_end_time_s"]) < 1e-12
    assert len(record["frames"]) == count
    rows = [
        json.loads(x) for x in (directory / "timeline.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == list(range(0, end + 1, stride))
    assert all(np.isfinite(value).all() for row in rows for value in row.values())
    assert all(row["gas_repair_count"] == 0 for row in rows)
    assert all(row["pending_refill_cells"] == 0 for row in rows)
    peak = max(row["peak_fluid_speed_lattice"] for row in rows)
    assert peak < cfg["maximum_velocity_lattice"]
    volume_error = max(abs(row["phase_volume_change_ratio"]) for row in rows)
    if cfg["conserve_phase_volume"]:
        assert volume_error < 2e-7
    else:
        assert all(row["phase_volume_projection_m3"] == 0 for row in rows)
    assert all(
        b["ice_mass_kg"] <= a["ice_mass_kg"] + 1e-12 for a, b in zip(rows, rows[1:])
    )
    initial = rows[0]
    latent = cfg["thermal"]["properties"]["latent_heat_j_kg"]
    energy_residual = max(
        abs(
            row["water_energy_j"]
            + row["ice_sensible_energy_j"]
            + row["melted_mass_kg"] * latent
            + row["discarded_melt_heat_j"]
            - row["thermal_remap_energy_j"]
            - initial["water_energy_j"]
            - initial["ice_sensible_energy_j"]
        )
        for row in rows
    )
    assert energy_residual < 1e-6 * initial["water_energy_j"]
    for session in record["sessions"]:
        source = Path(session["source_root"]) / "iceflow3d"
        for name, expected in session["source_sha256"].items():
            assert (
                hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
            ), f"source changed after simulation: {source / name}"
    checkpoint = directory / record["latest_checkpoint"]["path"]
    assert record["latest_checkpoint"]["step"] == end
    with np.load(checkpoint, allow_pickle=False) as state:
        checkpoint_metadata = json.loads(str(state["metadata"]))
        assert checkpoint_metadata["steps"] == end
        assert checkpoint_metadata["thermal_pending"] == 0
        assert checkpoint_metadata["pending_cells"] == 0
        for name in state.files:
            if name == "metadata":
                continue
            values = state[name]
            assert np.isfinite(values).all(), f"nonfinite final checkpoint: {name}"
    for index, frame in enumerate(record["frames"]):
        assert frame["index"] == index and frame["step"] == index * stride
        assert abs(frame["time_s"] - index * record["frame_interval_s"]) < 1e-12
        path = directory / frame["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["sha256"]
        with np.load(path, allow_pickle=False) as fields:
            assert all(np.isfinite(fields[name]).all() for name in fields.files)
            assert fields["water_phase"].shape == (450, 150, 90)
            assert 0 <= fields["water_phase"].min() <= fields["water_phase"].max() <= 1
            assert fields["ice_material_fraction"].min() >= 0
            assert fields["ice_material_fraction"].max() <= 1 + 1e-6
            np.testing.assert_allclose(
                np.linalg.norm(fields["body_quaternion_wxyz"], axis=-1), 1, atol=1e-6
            )
    result = {
        "status": "passed",
        "grid": [450, 150, 90],
        "steps": end,
        "physical_time_s": record["requested_end_time_s"],
        "frame_interval_s": record["frame_interval_s"],
        "frames": count,
        "maximum_speed_over_every_step_lattice": peak,
        "maximum_sampled_phase_volume_relative_error": volume_error,
        "maximum_accounted_energy_residual_j": energy_residual,
        "thermal_remap_energy_j": rows[-1]["thermal_remap_energy_j"],
        "final_ice_mass_kg": rows[-1]["ice_mass_kg"],
        "total_simulation_wall_time_s": record["total_simulation_wall_time_s"],
        "final_checkpoint": str(checkpoint),
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "interface_width_cells": cfg["interface_width"],
        "interface_width_m": cfg["interface_width"] * record["dx_m"],
    }
    write_json(directory / "verification.json", result)
    return result


def run_logged(command, path):
    with path.open("a") as log:
        log.write("\n" + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def timing_effort(root, production_time_s):
    """Keep useful production timing separate from failed runs and diagnosis."""
    other = []
    for path in sorted((root / "attempts").glob("*/simulation/simulation.json")):
        run = json.loads(path.read_text())
        other.append(
            {
                "type": "previous_production_attempt",
                "metadata": str(path),
                "status": run["status"],
                "wall_time_s": run["total_simulation_wall_time_s"],
            }
        )
    for path in sorted((root / "stability_trials").glob("*/probe.json")):
        run = json.loads(path.read_text())
        other.append(
            {
                "type": "diagnostic_replay",
                "metadata": str(path),
                "status": run["status"],
                "wall_time_s": run["wall_time_s"],
            }
        )
    result = {
        "production_simulation_wall_time_s": production_time_s,
        "earlier_attempts_and_diagnostics": other,
        "sum_of_simulation_and_diagnostic_process_times_s": production_time_s
        + sum(item["wall_time_s"] for item in other),
        "excludes": "human investigation gaps, surface meshing, look development, rendering, encoding",
    }
    write_json(root / "simulation_effort.json", result)
    return result


def audit_render(root, width, height, count, video, interval):
    render = root / "render"
    metadata = json.loads((render / "render.json").read_text())
    assert metadata["status"] == "completed"
    assert sorted(f["index"] for f in metadata["frames"]) == list(range(count))
    for index in range(count):
        path = render / "frames" / f"frame_{index:06d}.png"
        with path.open("rb") as stream:
            header = stream.read(25)
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[24] == 16
        with Image.open(path) as im:
            assert im.size == (width, height)
            im.verify()
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"]) == (width, height)
    assert stream["r_frame_rate"] == "50/1" and int(stream["nb_frames"]) == count
    assert abs(float(stream["duration"]) - count / 50) < 1e-3
    write_json(render / "video_verification.json", probe)
    sheet = Image.new("RGB", (3 * 640, 3 * 392), "#151c24")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    for cell, index in enumerate(np.linspace(0, count - 1, 9, dtype=int)):
        with Image.open(render / "frames" / f"frame_{index:06d}.png") as image:
            preview = image.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS)
        x, y = (cell % 3) * 640, (cell // 3) * 392
        sheet.paste(preview, (x, y))
        draw.text(
            (x + 14, y + 364), f"t = {index * interval:.2f} s", fill="#e6edf2", font=font
        )
    sheet.save(render / "contact_sheet.jpg", quality=95)
    return metadata, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--blender", default="/usr/local/blender-5.0.1/blender")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--render-settings", type=Path)
    args = p.parse_args()
    render_settings = json.loads(args.render_settings.read_text()) if args.render_settings else {}
    render_options = [str(value) for name, value in render_settings.items() for value in ("--"+name.replace("_", "-"), value)]
    root = args.root.resolve()
    scripts = Path(__file__).resolve().parent
    status_path = root / "pipeline.json"
    state = {
        "status": "running",
        "stage": "waiting_for_simulation",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "render_resolution": [args.width, args.height],
        "cycles_samples": args.samples,
    }

    def stage(name):
        state["stage"] = name
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)

    try:
        stage("waiting_for_simulation")
        while True:
            path = root / "simulation/simulation.json"
            if path.exists():
                sim = json.loads(path.read_text())
                if sim["status"] == "failed":
                    raise RuntimeError(f"simulation failed: {sim.get('error')}")
                if sim["status"] == "completed":
                    break
                pid = sim["sessions"][-1]["pid"]
                command_path = Path(f"/proc/{pid}/cmdline")
                if (
                    not command_path.exists()
                    or b"simulate.py" not in command_path.read_bytes()
                ):
                    raise RuntimeError(
                        "simulation process exited without a completed manifest"
                    )
                state["simulation_step"] = sim.get("current_step", 0)
                state["simulation_time_s"] = sim.get("current_time_s", 0)
                stage("waiting_for_simulation")
            time.sleep(15)
        stage("auditing_simulation")
        simulation = audit_simulation(root)
        stage("finishing_meshes")
        # Let the streaming CPU worker finish its last atomic cache first.
        worker = json.loads((root / "meshing_process.json").read_text())["pid"]
        while Path(f"/proc/{worker}/cmdline").exists():
            command = Path(f"/proc/{worker}/cmdline").read_bytes()
            if b"mesh_cache.py" not in command:
                break
            time.sleep(5)
        run_logged(
            [
                sys.executable,
                str(scripts / "mesh_cache.py"),
                "--simulation",
                str(root / "simulation"),
                "--output",
                str(root / "meshes"),
            ],
            root / "logs/meshing.log",
        )
        stage("rendering")
        run_logged(
            [
                args.blender,
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "-t",
                "16",
                "--python",
                str(scripts / "blender_scene.py"),
                "--",
                "--cache",
                str(root / "meshes"),
                "--output",
                str(root / "render"),
                "--device",
                "OPTIX",
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--samples",
                str(args.samples),
                "--threads",
                "16",
                "--all",
                *render_options,
            ],
            root / "logs/render.log",
        )
        stage("encoding")
        count = simulation["frames"]
        duration = simulation["physical_time_s"]
        video = root / f"render/dambreak_four_ice_450_{duration:g}s_{args.height}p_50fps.mp4"
        encoding_start = time.perf_counter()
        run_logged(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-framerate",
                "50",
                "-start_number",
                "0",
                "-i",
                str(root / "render/frames/frame_%06d.png"),
                "-frames:v",
                str(count),
                "-vf",
                "format=gbrp16le,zscale=primariesin=709:transferin=iec61966-2-1:"
                "matrixin=gbr:rangein=full:primaries=709:transfer=709:matrix=709:"
                "range=limited,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "16",
                "-r",
                "50",
                "-pix_fmt",
                "yuv420p",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-movflags",
                "+faststart",
                str(video),
            ],
            root / "logs/encoding.log",
        )
        encoding_s = time.perf_counter() - encoding_start
        stage("auditing_delivery")
        render, video_info = audit_render(root, args.width, args.height, count, video, simulation["frame_interval_s"])
        delivery = {
            "status": "completed",
            "simulation": simulation,
            "simulation_timing": str(root / "simulation/timing.json"),
            "simulation_effort": timing_effort(
                root, simulation["total_simulation_wall_time_s"]
            ),
            "images_directory": str(root / "render/frames"),
            "video": str(video),
            "video_info": video_info,
            "blend_file": str(root / "render/dambreak_four_ice.blend"),
            "contact_sheet": str(root / "render/contact_sheet.jpg"),
            "render_time_s": render["total_frame_render_time_s"],
            "encoding_time_s": encoding_s,
            "frame_count": count,
            "fps": 50,
            "duration_note": f"{count} frames include t=0 and t={duration:g} seconds; 50 fps video duration is {count/50:g} seconds.",
            "quality_review": "machine checks complete; inspect contact sheet and video before final user report",
        }
        write_json(root / "delivery.json", delivery)
        state["status"] = "completed"
        stage("completed")
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        stage(state["stage"])
        raise


if __name__ == "__main__":
    main()
