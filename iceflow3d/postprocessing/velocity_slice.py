"""Plot the exact z-midplane from saved fields, without importing the solver."""

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
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


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nice_limit(value):
    if value <= 0:
        return 1.0
    quantum = 10 ** math.floor(math.log10(value)) / 10
    return math.ceil(value / quantum) * quantum


def extract_slices(simulation, output, manifest, status):
    records = []
    cfg = manifest["configuration"]
    shape = (cfg["nx"], cfg["ny"], cfg["nz"])
    dx = manifest["dx_m"]
    z_mid = cfg["domain_size_m"][2] / 2
    cell_coordinate = z_mid / dx - 0.5
    k0 = int(math.floor(cell_coordinate + 1e-10))
    k1 = int(math.ceil(cell_coordinate - 1e-10))
    weight = cell_coordinate - k0 if k1 != k0 else 0.0
    assert 0 <= k0 <= k1 < shape[2]
    status["plane"] = {
        "axis": "z",
        "coordinate_m": z_mid,
        "source_cell_indices_zero_based": [k0, k1],
        "source_cell_centers_m": [(k0 + 0.5) * dx, (k1 + 0.5) * dx],
        "upper_layer_weight": weight,
        "interpolation": "Linear interpolation of velocity and water phase between cell centers.",
        "solid_mask": "Mask if either interpolation layer is solid; wall takes priority over ice.",
    }
    started = time.perf_counter()
    for row in manifest["frames"]:
        source = simulation / row["path"]
        with np.load(source, allow_pickle=False) as fields:
            assert float(fields["time_s"]) == row["time_s"]
            assert np.isclose(float(fields["dx_m"]), dx)
            velocity = fields["velocity_m_s"]
            assert velocity.shape == shape + (3,)
            v = ((1 - weight) * velocity[:, :, k0] + weight * velocity[:, :, k1]).copy()
            del velocity
            phase = fields["water_phase"]
            assert phase.shape == shape
            phi = ((1 - weight) * phase[:, :, k0] + weight * phase[:, :, k1]).copy()
            del phase
            ids = fields["solid_body_id"]
            assert ids.shape == shape
            a, b = ids[:, :, k0], ids[:, :, k1]
            # Categorical values are never interpolated or averaged.
            solid = np.where((a < -1) | (b < -1), -2, np.where(a >= 0, a, b)).copy()
        assert np.isfinite(v).all() and np.isfinite(phi).all()
        assert float(phi.min()) >= 0 and float(phi.max()) <= 1
        speed = np.linalg.norm(v, axis=-1)
        fluid = solid == -1
        water = fluid & (phi >= 0.5)
        record = {
            "index": row["index"],
            "time_s": row["time_s"],
            "source": str(source),
            "source_sha256_from_verified_manifest": row["sha256"],
            "max_fluid_speed_m_s": float(speed[fluid].max(initial=0)),
            "max_water_speed_m_s": float(speed[water].max(initial=0)),
        }
        target = output / "slices" / f"frame_{row['index']:06d}.npz"
        np.savez_compressed(
            target, velocity_m_s=v, water_phase=phi, solid_body_id=solid,
            speed_m_s=speed, time_s=row["time_s"], z_m=z_mid, dx_m=dx,
        )
        record["slice"] = str(target)
        records.append(record)
        if len(records) % 25 == 0 or len(records) == len(manifest["frames"]):
            print(f"EXTRACTED {len(records)}/{len(manifest['frames'])}", flush=True)
    status["extraction_wall_time_s"] = time.perf_counter() - started
    status["frames"] = records
    status["color_limits_m_s"] = [0.0, nice_limit(max(r["max_fluid_speed_m_s"] for r in records))]
    status["maximum_fluid_speed_m_s"] = max(r["max_fluid_speed_m_s"] for r in records)
    status["field_definition"] = {
        "color": "Euclidean speed sqrt(ux^2 + uy^2 + uz^2), in m/s, in all non-solid fluid cells (air and water).",
        "arrows": "In-plane (ux, uy) vector, fixed length scale across all frames; sampling every 12 cells.",
        "interface": "Interpolated water_phase=0.5; contour masked in solid cells.",
        "time_color_scale": "One linear scale determined by the maximum over all saved slices; no clipping or per-frame normalization.",
    }
    write_json(output / "visualization.json", status)


def render_frames(output, manifest, status):
    cfg = manifest["configuration"]
    nx, ny = cfg["nx"], cfg["ny"]
    dx_mm = manifest["dx_m"] * 1000
    lx, ly = np.array(cfg["domain_size_m"][:2]) * 1000
    xs, ys = (np.arange(nx) + 0.5) * dx_mm, (np.arange(ny) + 0.5) * dx_mm
    vmax = status["color_limits_m_s"][1]
    width, height = status["image_size"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelcolor": "#263547", "text.color": "#263547", "axes.edgecolor": "#8493a5"})
    fig = plt.figure(figsize=(width / 120, height / 120), dpi=120, facecolor="#f7f9fc")
    ax = fig.add_axes([0.065, 0.195, 0.805, 0.58])
    cax = fig.add_axes([0.9, 0.195, 0.014, 0.58])
    fig.text(0.065, 0.93, "Dambreak with four ice blocks", fontsize=24, weight="bold")
    fig.text(0.065, 0.876, f"MID-PLANE VELOCITY   /   z = {status['plane']['coordinate_m'] * 1000:.3f} mm   /   {nx} x {ny} x {cfg['nz']} cells", fontsize=12, color="#64748b")
    clock_text = fig.text(0.935, 0.926, "", ha="right", fontsize=20, weight="bold")
    count_text = fig.text(0.935, 0.875, "", ha="right", fontsize=11, color="#64748b")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0, 0, 0, 0))
    artist = ax.imshow(np.zeros((ny, nx)), origin="lower", extent=[0, lx, 0, ly],
                       interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    solid_image = ax.imshow(np.ma.masked_all((ny, nx)), origin="lower", extent=[0, lx, 0, ly],
                            interpolation="nearest", cmap=ListedColormap(["#d9e0e8", "#536477"]), vmin=0, vmax=1)
    bar = fig.colorbar(artist, cax=cax)
    bar.set_label("Speed |u| [m/s]", labelpad=13)
    bar.outline.set_linewidth(0.5)
    ax.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(0, lx), ylim=(0, ly))
    ax.tick_params(labelsize=11)
    ax.legend(handles=[Line2D([], [], color="#5a6678", lw=2, label="Water interface (phase = 0.5)"),
                       Patch(facecolor="#d9e0e8", label="Ice"), Patch(facecolor="#536477", label="Tank wall")],
              loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=10,
              borderaxespad=0, handlelength=1.8)
    fig.text(0.065, 0.065, "Color: full 3D speed in water and air. Arrows: in-plane velocity (ux, uy).", fontsize=11)
    fig.text(0.065, 0.033, "Exact mid-plane interpolated between adjacent cell centers. Fixed color and arrow scales for the full sequence.", fontsize=10, color="#64748b")
    ij = np.ix_(np.arange(6, nx, 12), np.arange(6, ny, 12))
    xq, yq = np.meshgrid(xs[6::12], ys[6::12], indexing="ij")
    vectors = contour = key = None
    started = time.perf_counter()
    for number, row in enumerate(status["frames"]):
        with np.load(row["slice"], allow_pickle=False) as fields:
            v, phi, solid, speed = [fields[name] for name in ("velocity_m_s", "water_phase", "solid_body_id", "speed_m_s")]
        fluid = solid == -1
        artist.set_data(np.ma.masked_where(~fluid.T, speed.T))
        categories = np.where(solid >= 0, 0, 1)
        solid_image.set_data(np.ma.masked_where(fluid.T, categories.T))
        if contour is not None:
            contour.remove()
            contour = None
        if np.any(fluid & (phi < .5)) and np.any(fluid & (phi >= .5)):
            contour = ax.contour(xs, ys, np.ma.masked_where(~fluid.T, phi.T), levels=[.5], colors="white", linewidths=1.05)
            contour.set_path_effects([pe.Stroke(linewidth=2.0, foreground="#344154"), pe.Normal()])
        if vectors is not None:
            vectors.remove()
            key.remove()
        u, w = v[ij][..., 0], v[ij][..., 1]
        hidden = ~fluid[ij] | (np.hypot(u, w) < vmax * .01)
        vectors = ax.quiver(xq, yq, np.ma.masked_where(hidden, u), np.ma.masked_where(hidden, w),
                            angles="xy", scale_units="xy", scale=vmax / 4.0, pivot="mid",
                            color="white", edgecolor="#243044", linewidth=.2, width=.0012,
                            headwidth=3.5, headlength=4.5, minlength=0)
        key = ax.quiverkey(vectors, X=.895, Y=.1, U=vmax / 2, label=f"{vmax / 2:g} m/s", labelpos="W", coordinates="figure", color="#263547", labelcolor="#263547", fontproperties={"size": 10})
        clock_text.set_text(f"t = {row['time_s']:.2f} s")
        count_text.set_text(f"Frame {row['index']:03d} / {len(status['frames']) - 1:03d}")
        target = output / "frames" / f"frame_{row['index']:06d}.png"
        temporary = target.with_suffix(".tmp.png")
        fig.savefig(temporary, dpi=120, facecolor=fig.get_facecolor())
        temporary.replace(target)
        row["image"] = str(target)
        if (number + 1) % 25 == 0 or number + 1 == len(status["frames"]):
            status["rendered_frames"] = number + 1
            write_json(output / "visualization.json", status)
            print(f"PLOTTED {number + 1}/{len(status['frames'])}", flush=True)
    plt.close(fig)
    status["plot_wall_time_s"] = time.perf_counter() - started


def overview(output, rows):
    times = [0, .1, .2, .5, 1, 2, 3, 4, 5]
    selected = [min(rows, key=lambda row: abs(row["time_s"] - t)) for t in times]
    thumb_w, thumb_h = 640, 320
    canvas = Image.new("RGB", (thumb_w * 3, (thumb_h + 30) * 3), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for i, row in enumerate(selected):
        x, y = (i % 3) * thumb_w, (i // 3) * (thumb_h + 30)
        draw.text((x + 20, y + 5), f"t = {row['time_s']:.2f} s", font=font, fill="#263547")
        with Image.open(row["image"]) as im:
            canvas.paste(im.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y + 30))
    canvas.save(output / "overview.jpg", quality=95)


def encode_and_verify(output, status):
    rows = status["frames"]
    for row in rows:
        path = Path(row["image"])
        with path.open("rb") as stream:
            header = stream.read(33)
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        assert list(struct.unpack(">II", header[16:24])) == status["image_size"]
        with Image.open(path) as im:
            im.verify()
        row["image_sha256"] = sha256(path)
    video = output / "velocity_z_mid_50fps.mp4"
    command = [shutil.which("ffmpeg"), "-hide_banner", "-y", "-framerate", str(status["fps"]),
               "-start_number", "0", "-i", str(output / "frames/frame_%06d.png"),
               "-frames:v", str(len(rows)), "-an", "-vf", "format=rgb24,zscale=primariesin=709:transferin=iec61966-2-1:matrixin=gbr:rangein=full:primaries=709:transfer=709:matrix=709:range=limited,format=yuv420p",
               "-c:v", "libx264", "-threads", "8", "-preset", "medium", "-crf", "16", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(video)]
    started = time.perf_counter()
    with (output / "encoding.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    status["encoding_wall_time_s"] = time.perf_counter() - started
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,duration", "-of", "json", str(video)]))
    stream = info["streams"][0]
    assert int(stream["nb_read_frames"]) == len(rows)
    assert [stream["width"], stream["height"]] == status["image_size"]
    assert stream["r_frame_rate"] == f"{status['fps']}/1"
    assert abs(float(stream["duration"]) - len(rows) / status["fps"]) < 1e-5
    with (output / "decode_verification.log").open("w") as log:
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], stdout=log, stderr=subprocess.STDOUT, check=True)
    status["video"] = str(video)
    status["video_verification"] = info
    status["video_sha256"] = sha256(video)
    status["video_duration_note"] = "251 endpoint-inclusive samples at 50 fps give 5.02 s; simulated physical time is 5.00 s."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-slices", action="store_true")
    args = parser.parse_args()
    simulation, output = args.simulation.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ["slices", "frames"]:
        (output / directory).mkdir(exist_ok=True)
    manifest = json.loads((simulation / "simulation.json").read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["frames"]) == 251 and manifest["frames"][-1]["time_s"] == 5.0
    source_sha = sha256(simulation / "simulation.json")
    started = time.perf_counter()
    if args.reuse_slices:
        status = json.loads((output / "visualization.json").read_text())
        assert status["simulation_manifest_sha256"] == source_sha
        assert len(status["frames"]) == len(manifest["frames"])
        status["reused_slice_cache"] = True
    else:
        status = {"status": "extracting", "started_utc": utc_now(), "simulation": str(simulation),
                  "simulation_manifest_sha256": source_sha, "image_size": [1920, 960], "fps": 50}
        extract_slices(simulation, output, manifest, status)
    status.update(status="plotting", script_sha256=sha256(Path(__file__)))
    write_json(output / "visualization.json", status)
    render_frames(output, manifest, status)
    status["status"] = "encoding"
    write_json(output / "visualization.json", status)
    overview(output, status["frames"])
    encode_and_verify(output, status)
    status.update(status="completed", completed_utc=utc_now(), invocation_wall_time_s=time.perf_counter() - started)
    write_json(output / "visualization.json", status)
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
