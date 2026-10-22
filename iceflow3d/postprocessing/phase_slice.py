"""Plot saved water-phase midplane slices with a fixed physical phase scale."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_overview(output, frames):
    times = [0, .1, .2, .5, 1, 2, 3, 4, 5]
    canvas = Image.new("RGB", (1920, 1050), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for i, t in enumerate(times):
        row = min(frames, key=lambda row: abs(row["time_s"] - t))
        x, y = (i % 3) * 640, (i // 3) * 350
        draw.text((x + 20, y + 5), f"t = {row['time_s']:.2f} s", font=font, fill="#263547")
        with Image.open(row["image"]) as im:
            canvas.paste(im.convert("RGB").resize((640, 320), Image.Resampling.LANCZOS), (x, y + 30))
    canvas.save(output / "overview.jpg", quality=95)


def plot_frames(output, simulation, cache, result):
    cfg = simulation["configuration"]
    nx, ny = cfg["nx"], cfg["ny"]
    dx_mm = simulation["dx_m"] * 1000
    lx, ly = np.array(cfg["domain_size_m"][:2]) * 1000
    xs, ys = (np.arange(nx) + .5) * dx_mm, (np.arange(ny) + .5) * dx_mm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelcolor": "#263547", "text.color": "#263547", "axes.edgecolor": "#8493a5"})
    fig = plt.figure(figsize=(16, 8), dpi=120, facecolor="#f7f9fc")
    ax = fig.add_axes([.065, .195, .805, .58])
    cax = fig.add_axes([.9, .195, .014, .58])
    fig.text(.065, .93, "Dambreak with four ice blocks", fontsize=24, weight="bold")
    fig.text(.065, .876, f"MID-PLANE WATER PHASE   /   z = {result['plane']['coordinate_m'] * 1000:.3f} mm   /   {nx} x {ny} x {cfg['nz']} cells", fontsize=12, color="#64748b")
    clock = fig.text(.935, .926, "", ha="right", fontsize=20, weight="bold")
    counter = fig.text(.935, .875, "", ha="right", fontsize=11, color="#64748b")
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad((0, 0, 0, 0))
    artist = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=[0, lx, 0, ly],
                       interpolation="nearest", cmap=cmap, vmin=0, vmax=1, aspect="equal")
    solids = ax.imshow(np.ma.masked_all((ny, nx)), origin="lower", extent=[0, lx, 0, ly],
                       interpolation="nearest", cmap=ListedColormap(["#cdd5df", "#536477"]), vmin=0, vmax=1)
    bar = fig.colorbar(artist, cax=cax, ticks=np.linspace(0, 1, 6))
    bar.set_label("Water phase (0 = air, 1 = water)", labelpad=13)
    bar.outline.set_linewidth(.5)
    ax.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(0, lx), ylim=(0, ly))
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Line2D([], [], color="#5a6678", lw=2, label="Water interface (phase = 0.5)"),
                       Patch(facecolor="#cdd5df", label="Ice"), Patch(facecolor="#536477", label="Tank wall")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=10,
              borderaxespad=0, handlelength=1.8)
    fig.text(.065, .065, "Color: stored water phase. The same 0-1 scale is used for every frame; the liquid phase is not rescaled to 1.", fontsize=11)
    fig.text(.065, .033, "Exact z mid-plane: linear interpolation between cell centers 44 and 45. Solid cells are shown in gray.", fontsize=10, color="#64748b")
    maximum = fig.text(.87, .1, "", ha="right", fontsize=10, color="#64748b")
    contour = None
    frames = []
    started = time.perf_counter()
    for index, cached in enumerate(cache["frames"]):
        source = simulation["frames"][index]
        assert cached["index"] == source["index"] == index
        assert cached["source_sha256_from_verified_manifest"] == source["sha256"]
        slice_path = Path(cached["slice"])
        with np.load(slice_path, allow_pickle=False) as fields:
            phase, ids = fields["water_phase"], fields["solid_body_id"]
            assert float(fields["time_s"]) == source["time_s"]
            assert np.isclose(float(fields["z_m"]), result["plane"]["coordinate_m"])
        assert phase.shape == ids.shape == (nx, ny)
        assert np.isfinite(phase).all() and 0 <= phase.min() <= phase.max() <= 1
        fluid = ids == -1
        artist.set_data(np.ma.masked_where(~fluid.T, phase.T))
        solids.set_data(np.ma.masked_where(fluid.T, np.where(ids >= 0, 0, 1).T))
        if contour is not None:
            contour.remove()
            contour = None
        if np.any(fluid & (phase < .5)) and np.any(fluid & (phase >= .5)):
            contour = ax.contour(xs, ys, np.ma.masked_where(~fluid.T, phase.T), levels=[.5], colors="white", linewidths=1.05)
            contour.set_path_effects([pe.Stroke(linewidth=2, foreground="#344154"), pe.Normal()])
        clock.set_text(f"t = {source['time_s']:.2f} s")
        counter.set_text(f"Frame {index:03d} / {len(cache['frames']) - 1:03d}")
        maximum.set_text(f"Maximum phase in fluid cells: {float(phase[fluid].max()):.6f}")
        image_path = output / "frames" / f"frame_{index:06d}.png"
        temporary = image_path.with_suffix(".tmp.png")
        fig.savefig(temporary, dpi=120, facecolor=fig.get_facecolor())
        temporary.replace(image_path)
        frames.append({"index": index, "time_s": source["time_s"], "source": cached["source"],
                       "source_sha256_from_verified_manifest": source["sha256"],
                       "slice": str(slice_path), "slice_sha256": sha256(slice_path),
                       "min_phase_in_fluid": float(phase[fluid].min()), "max_phase_in_fluid": float(phase[fluid].max()),
                       "image": str(image_path)})
        if len(frames) % 25 == 0 or len(frames) == len(cache["frames"]):
            print(f"PLOTTED {len(frames)}/{len(cache['frames'])}", flush=True)
            result.update(frames=frames, rendered_frames=len(frames))
            write_json(output / "visualization.json", result)
    plt.close(fig)
    result["plot_wall_time_s"] = time.perf_counter() - started


def encode_and_verify(output, result):
    frames = result["frames"]
    for row in frames:
        path = Path(row["image"])
        with Image.open(path) as im:
            assert list(im.size) == result["image_size"]
            im.verify()
        row["image_sha256"] = sha256(path)
    video = output / "phase_z_mid_50fps.mp4"
    command = ["ffmpeg", "-hide_banner", "-y", "-framerate", "50", "-start_number", "0",
               "-i", str(output / "frames/frame_%06d.png"), "-frames:v", str(len(frames)), "-an",
               "-vf", "format=rgb24,zscale=primariesin=709:transferin=iec61966-2-1:matrixin=gbr:rangein=full:primaries=709:transfer=709:matrix=709:range=limited,format=yuv420p",
               "-c:v", "libx264", "-threads", "8", "-preset", "medium", "-crf", "16",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(video)]
    started = time.perf_counter()
    with (output / "encoding.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result["encoding_wall_time_s"] = time.perf_counter() - started
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,duration", "-of", "json", str(video)]))
    stream = info["streams"][0]
    assert int(stream["nb_read_frames"]) == len(frames)
    assert [stream["width"], stream["height"]] == result["image_size"]
    assert stream["r_frame_rate"] == "50/1"
    assert abs(float(stream["duration"]) - len(frames) / 50) < 1e-5
    with (output / "decode_verification.log").open("w") as log:
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], stdout=log, stderr=subprocess.STDOUT, check=True)
    result.update(video=str(video), video_sha256=sha256(video), video_verification=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    simulation_path = root / "simulation/simulation.json"
    cache_path = root / "velocity_z_mid/visualization.json"
    simulation = json.loads(simulation_path.read_text())
    cache = json.loads(cache_path.read_text())
    assert simulation["status"] == cache["status"] == "completed"
    assert cache["simulation_manifest_sha256"] == sha256(simulation_path)
    assert len(simulation["frames"]) == len(cache["frames"]) == 251
    assert cache["plane"]["source_cell_indices_zero_based"] == [44, 45]
    output = root / "phase_z_mid"
    (output / "frames").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"status": "plotting", "started_utc": utc_now(), "simulation": str(root / "simulation"),
              "simulation_manifest_sha256": sha256(simulation_path), "slice_cache_manifest": str(cache_path),
              "slice_cache_manifest_sha256": sha256(cache_path), "plane": cache["plane"],
              "field": "water_phase", "color_limits": [0., 1.], "image_size": [1920, 960],
              "fps": 50, "script_sha256": sha256(Path(__file__)),
              "definition": "Original saved phase with only spatial mid-plane interpolation; no clipping, smoothing, per-frame color normalization or correction of the liquid endpoint.",
              "temperature": "Not generated, per user instruction."}
    plot_frames(output, simulation, cache, result)
    make_overview(output, result["frames"])
    result["status"] = "encoding"
    write_json(output / "visualization.json", result)
    encode_and_verify(output, result)
    result.update(status="completed", completed_utc=utc_now(), wall_time_s=time.perf_counter() - started)
    write_json(output / "visualization.json", result)
    (output / "phase_slice.py").write_bytes(Path(__file__).read_bytes())
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
