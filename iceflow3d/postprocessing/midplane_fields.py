"""Render saved 2D midplanes as velocity, temperature and phase PNG/MP4."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, SymLogNorm  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate(run, record):
    cfg = record["configuration"]
    assert record["status"] == "completed"
    assert record["completed_steps"] == record["target_steps"]
    assert not record["output_3d_fields"] and not record["checkpoint_output"]
    assert cfg["check_stability"] and not cfg["gas_state_repair"]
    assert not cfg["conserve_phase_volume"]
    assert record["frames"][0]["step"] == 0
    assert record["frames"][-1]["step"] == record["completed_steps"]
    shape = (cfg["nx"], cfg["ny"])
    for row in record["frames"]:
        with np.load(run / row["path"], allow_pickle=False) as f:
            assert f["water_phase"].shape == f["temperature_c"].shape == shape
            assert f["velocity_m_s"].shape == (*shape, 3)
            assert all(np.isfinite(f[name]).all() for name in f.files)
            assert f["water_phase"].min() >= 0 and f["water_phase"].max() <= 1
            assert int(f["step"]) == row["step"]
            assert abs(float(f["time_s"]) - row["time_s"]) < 1e-12
            assert abs(float(f["z_m"]) - record["plane"]["coordinate_m"]) < 1e-12
        row["sha256"] = hashlib.sha256((run / row["path"]).read_bytes()).hexdigest()
    with (run / "history.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    assert len(rows) == len(record["frames"])
    assert all(np.isfinite(v) for r in rows for v in r.values())
    assert all(r["gas_repair_count"] == r["pending_refill_cells"] == 0 for r in rows)
    assert all(r["phase_volume_projection_m3"] == 0 for r in rows)
    assert rows[-1]["peak_fluid_speed_lattice"] < cfg["maximum_velocity_lattice"]
    assert all(
        b["ice_mass_kg"] <= a["ice_mass_kg"] + 1e-12 for a, b in zip(rows, rows[1:])
    )
    initial = rows[0]
    latent = cfg["thermal"]["properties"]["latent_heat_j_kg"]
    residual = max(
        abs(
            r["water_energy_j"]
            + r["ice_sensible_energy_j"]
            + r["melted_mass_kg"] * latent
            + r["discarded_melt_heat_j"]
            - r["thermal_remap_energy_j"]
            - initial["water_energy_j"]
            - initial["ice_sensible_energy_j"]
        )
        for r in rows
    )
    assert residual < 1e-6 * initial["water_energy_j"]
    temperature_min = min(r["temperature_min_c"] for r in record["frames"])
    temperature_max = max(r["temperature_max_c"] for r in record["frames"])
    initial_t = [
        cfg["thermal"][f"initial_{p}_temperature_c"] for p in ("water", "ice", "air")
    ]
    assert temperature_min >= min(initial_t) - 1e-3
    assert temperature_max <= max(initial_t) + 1e-3
    return dict(
        passed=True,
        frame_count=len(rows),
        physical_time_s=rows[-1]["time_s"],
        completed_steps=rows[-1]["step"],
        peak_l1_speed_lattice=rows[-1]["peak_fluid_speed_lattice"],
        phase_volume_change_ratio=rows[-1]["phase_volume_change_ratio"],
        energy_balance_residual_j=residual,
        temperature_range_c=[temperature_min, temperature_max],
        final_ice_mass_kg=rows[-1]["ice_mass_kg"],
        final_contact_residuals=record["final_contact_residuals"],
        no_3d_field_output=True,
    )


def render(run, record, frame_limit=None):
    cfg, rows = record["configuration"], record["frames"]
    dx_mm = record["scales"]["dx_m"] * 1000
    nx, ny = cfg["nx"], cfg["ny"]
    lx, ly = np.array(cfg["domain_size_m"][:2]) * 1000
    xs, ys = (np.arange(nx) + 0.5) * dx_mm, (np.arange(ny) + 0.5) * dx_mm
    maximum = max(row["max_speed_m_s"] for row in rows)
    vmax = max(0.1, np.ceil(maximum * 10) / 10)
    speed_norm = SymLogNorm(linthresh=0.05, linscale=1.0, vmin=0, vmax=vmax)
    tmin = min(
        cfg["thermal"][f"initial_{p}_temperature_c"] for p in ("ice", "water", "air")
    )
    tmax = max(
        cfg["thermal"][f"initial_{p}_temperature_c"] for p in ("ice", "water", "air")
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#283747",
            "text.color": "#283747",
            "axes.edgecolor": "#8a98a6",
        }
    )
    fig = plt.figure(figsize=(12, 13.5), dpi=120, facecolor="#f7f9fc")
    fig.text(
        0.075,
        0.972,
        "Dam break with four melting ice blocks",
        fontsize=21,
        weight="bold",
    )
    clock = fig.text(0.93, 0.938, "", ha="right", fontsize=20, weight="bold")
    fig.text(
        0.075,
        0.94,
        f"CM-MRT  |  {nx} x {ny} x {cfg['nz']}  |  z = {record['plane']['coordinate_m'] * 1000:g} mm",
        fontsize=13,
    )
    fig.text(
        0.075,
        0.915,
        "Water interface: white/black line    Ice: cyan outline    Tank walls: gray",
        fontsize=10,
        color="#536477",
    )
    titles = [
        "VELOCITY  |  color: full 3D speed; arrows: in-plane direction",
        f"TEMPERATURE  |  water and ice; air prescribed at {cfg['thermal']['initial_air_temperature_c']:g} C",
        "PHASE FIELD  |  0 = air, 1 = water",
    ]
    axes, artists, overlays = [], [], []
    cmaps = [plt.get_cmap(name).copy() for name in ("viridis", "inferno", "Blues")]
    for i, bottom in enumerate((0.65, 0.355, 0.06)):
        ax = fig.add_axes([0.075, bottom, 0.78, 0.225])
        cax = fig.add_axes([0.885, bottom, 0.017, 0.225])
        ax.set_facecolor("#dde5ec")
        cmaps[i].set_bad((0, 0, 0, 0))
        norm = speed_norm if i == 0 else None
        image = ax.imshow(
            np.zeros((ny, nx)),
            origin="lower",
            extent=[0, lx, 0, ly],
            interpolation="nearest",
            cmap=cmaps[i],
            norm=norm,
            vmin=None if i == 0 else (tmin if i == 1 else 0),
            vmax=None if i == 0 else (tmax if i == 1 else 1),
            aspect="equal",
        )
        overlay = ax.imshow(
            np.ma.masked_all((ny, nx)),
            origin="lower",
            extent=[0, lx, 0, ly],
            interpolation="nearest",
            cmap=ListedColormap(["#dde5ec", "#657585"]),
            vmin=0,
            vmax=1,
        )
        ax.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(0, lx), ylim=(0, ly))
        ax.set_title(titles[i], loc="left", fontsize=12, weight="bold", pad=12)
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(
            ["Speed |u| [m/s]", "Temperature [degC]", "Water phase"][i], labelpad=12
        )
        if i == 0:
            ticks = [
                v for v in (0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if v <= vmax
            ]
            if vmax not in ticks:
                ticks.append(vmax)
            bar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        axes.append(ax)
        artists.append(image)
        overlays.append(overlay)
    fig.text(
        0.075,
        0.006,
        "Fixed scales across all frames. Speed uses a logarithmic scale above 0.05 m/s. No spatial smoothing.",
        fontsize=10,
        color="#536477",
    )
    folder = run / "frames"
    folder.mkdir(exist_ok=True)
    contours, arrows = [], None
    stride = max(1, nx // 25)
    selection = np.ix_(
        np.arange(stride // 2, nx, stride), np.arange(stride // 2, ny, stride)
    )
    xq, yq = np.meshgrid(
        xs[stride // 2 :: stride], ys[stride // 2 :: stride], indexing="ij"
    )
    images = []
    for row in rows[:frame_limit]:
        with np.load(run / row["path"], allow_pickle=False) as f:
            v, phi, temp, ids = [
                f[name]
                for name in (
                    "velocity_m_s",
                    "water_phase",
                    "temperature_c",
                    "solid_body_id",
                )
            ]
        fluid, ice, wall = ids == -1, ids >= 0, ids < -1
        speed = np.linalg.norm(v, axis=-1)
        for contour in contours:
            contour.remove()
        contours = []
        for i, (ax, values) in enumerate(zip(axes, (speed, temp, phi))):
            mask = wall if i == 1 else ~fluid
            artists[i].set_data(np.ma.masked_where(mask.T, values.T))
            categories = np.where(wall, 1, 0)
            overlays[i].set_data(np.ma.masked_where(~mask.T, categories.T))
            if np.any(fluid & (phi < 0.5)) and np.any(fluid & (phi >= 0.5)):
                contour = ax.contour(
                    xs,
                    ys,
                    np.ma.masked_where(~fluid.T, phi.T),
                    levels=[0.5],
                    colors="white",
                    linewidths=0.9,
                )
                contour.set_path_effects(
                    [pe.Stroke(linewidth=1.8, foreground="#243746"), pe.Normal()]
                )
                contours.append(contour)
            if ice.any():
                contours.append(
                    ax.contour(
                        xs,
                        ys,
                        ice.T.astype(float),
                        levels=[0.5],
                        colors="#39c5df",
                        linewidths=0.8,
                    )
                )
        if arrows is not None:
            arrows.remove()
        vx, vy = v[selection][..., 0], v[selection][..., 1]
        length = np.hypot(vx, vy)
        hidden = ~fluid[selection] | (length < 0.003)
        arrows = axes[0].quiver(
            xq,
            yq,
            np.ma.masked_where(hidden, vx / np.maximum(length, 1e-20)),
            np.ma.masked_where(hidden, vy / np.maximum(length, 1e-20)),
            angles="xy",
            scale_units="xy",
            scale=1 / (3 * dx_mm),
            pivot="mid",
            color="white",
            edgecolor="#243746",
            linewidth=0.2,
            width=0.0018,
            headwidth=3.4,
            headlength=4.2,
        )
        clock.set_text(f"t = {row['time_s']:.2f} s")
        path = folder / f"frame_{row['index']:06d}.png"
        fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
        with Image.open(path) as im:
            im.verify()
        images.append(
            dict(
                index=row["index"],
                time_s=row["time_s"],
                path=str(path.relative_to(run)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        if len(images) % 25 == 0 or len(images) == len(rows[:frame_limit]):
            print(f"Rendered {len(images)}/{len(rows[:frame_limit])}", flush=True)
    plt.close(fig)
    return dict(
        frames=images,
        image_size=[1440, 1620],
        speed_limits_m_s=[0, float(vmax)],
        speed_scale="Linear below 0.05 m/s, logarithmic above; fixed for the sequence.",
        arrow_definition="Normalized in-plane velocity direction; hidden below 0.003 m/s.",
        temperature_limits_c=[tmin, tmax],
        phase_limits=[0, 1],
    )


def encode(run, record, visualization):
    fps = round(1 / record["output_interval_s"])
    count = len(visualization["frames"])
    video = run / "dambreak_z_mid.mp4"
    with (run / "encoding.log").open("w") as log:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(run / "frames/frame_%06d.png"),
                "-frames:v",
                str(count),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "medium",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate,duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = probe["streams"][0]
    assert int(stream["nb_read_frames"]) == count
    assert [stream["width"], stream["height"]] == visualization["image_size"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
        check=True,
    )
    visualization.update(
        video=str(video.relative_to(run)),
        video_probe=probe,
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    shutil.copy2(run / visualization["frames"][-1]["path"], run / "final.png")
    chosen = [
        min(visualization["frames"], key=lambda r: abs(r["time_s"] - t))
        for t in (0, 0.2, 0.5, 1, 2, 4)
    ]
    montage = Image.new("RGB", (1440, 1080), "#f7f9fc")
    for i, row in enumerate(chosen):
        with Image.open(run / row["path"]) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((480, 540), Image.Resampling.LANCZOS)
            montage.paste(thumb, ((i % 3) * 480, (i // 3) * 540))
    montage.save(run / "overview.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--preview-frames", type=int)
    args = parser.parse_args()
    record = json.loads((args.run / "metadata.json").read_text())
    verification = validate(args.run, record)
    write_json(args.run / "verification.json", verification)
    visualization = render(args.run, record, args.preview_frames)
    if args.preview_frames is None:
        encode(args.run, record, visualization)
    write_json(args.run / "visualization.json", visualization)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
