"""Render saved dry-ice z slices with separate liquid/gas colour scales.

Usage: python -m iceflow3d.examples.render_dry_ice_slices RUN_DIRECTORY
All frames use the same scales, determined from the whole saved sequence.
Use --gas-speed-max-m-s to override only the gas speed colour scale.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np


def read_slice(path):
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    ids, phi = fields["solid_body_id"], fields["water_phase"]
    velocity, temperature = fields["velocity_m_s"], fields["temperature_c"]
    if phi.ndim != 2 or ids.shape != phi.shape or velocity.shape != (*phi.shape, 3):
        raise ValueError(f"invalid slice shapes: {path}")
    if temperature.shape != phi.shape:
        raise ValueError(f"invalid temperature shape: {path}")
    fluid = ids == -1
    if not all(
        np.isfinite(values[fluid]).all() for values in (phi, velocity, temperature)
    ):
        raise ValueError(f"nonfinite fluid data: {path}")
    fields["water_mask"] = fluid & (phi >= 0.5)
    fields["gas_mask"] = fluid & (phi < 0.5)
    # The magnitude includes the out-of-plane component; arrows show ux, uy.
    fields["speed_m_s"] = np.linalg.norm(velocity, axis=-1)
    return fields


def sequence_limits(paths):
    limits = {
        "water_speed_max_m_s": 0.0,
        "gas_speed_max_m_s": 0.0,
        "water_temperature_min_c": np.inf,
        "water_temperature_max_c": -np.inf,
        "gas_temperature_min_c": np.inf,
        "gas_temperature_max_c": -np.inf,
    }
    times, steps = [], []
    for path in paths:
        data = read_slice(path)
        times.append(float(data["time_s"]))
        steps.append(int(data["step"]))
        for phase in ("water", "gas"):
            mask = data[f"{phase}_mask"]
            if np.any(mask):
                key = f"{phase}_speed_max_m_s"
                limits[key] = max(limits[key], float(data["speed_m_s"][mask].max()))
                for suffix, reduce, compare in (
                    ("min", np.min, min),
                    ("max", np.max, max),
                ):
                    key = f"{phase}_temperature_{suffix}_c"
                    limits[key] = compare(
                        limits[key], float(reduce(data["temperature_c"][mask]))
                    )
    if np.any(np.diff(times) <= 0) or np.any(np.diff(steps) <= 0):
        raise ValueError("slice times and steps must be strictly increasing")
    return limits, times, steps


class SliceFigure:
    """Reuse artists so 601 high-resolution frames do not rebuild the layout."""

    def __init__(self, sample, limits, *, dpi=110, gas_speed_max_m_s=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, Normalize
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        self.plt, self.dpi = plt, dpi
        dx = float(sample["dx_m"]) * 1000
        self.x = sample["x_m"] * 1000
        self.y = sample["y_m"] * 1000
        self.extent = (0, len(self.x) * dx, 0, len(self.y) * dx)
        self.fig = plt.figure(figsize=(18, 7.4), facecolor="white")
        grid = self.fig.add_gridspec(
            3,
            3,
            left=0.05,
            right=0.975,
            bottom=0.115,
            top=0.86,
            height_ratios=(1, 0.035, 0.035),
            hspace=0.42,
            wspace=0.26,
        )
        self.axes = [self.fig.add_subplot(grid[0, col]) for col in range(3)]
        self.layers, self.walls, self.ice = [], [], []
        self.interfaces = []
        blank = np.ma.masked_all(sample["water_phase"].shape)
        for ax in self.axes:
            ax.set_facecolor("#f4f5f7")
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
            ax.set_xlim(self.extent[:2])
            ax.set_ylim(self.extent[2:])

        def layer(column, row, cmap_name, vmin, vmax, label, ticks=None):
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad((0, 0, 0, 0))
            plotted = self.axes[column].imshow(
                blank,
                origin="lower",
                extent=self.extent,
                cmap=cmap,
                norm=Normalize(vmin=vmin, vmax=vmax),
                interpolation="nearest",
            )
            self.fig.colorbar(
                plotted,
                cax=self.fig.add_subplot(grid[row, column]),
                orientation="horizontal",
                label=label,
                ticks=ticks,
            )
            self.layers.append(plotted)

        self.speed_maxima = {
            phase: max(limits[f"{phase}_speed_max_m_s"], 1e-6)
            for phase in ("water", "gas")
        }
        layer(0, 1, "viridis", 0, self.speed_maxima["water"], "Water |u| (m/s)")
        gas_color_max = (
            self.speed_maxima["gas"] if gas_speed_max_m_s is None else gas_speed_max_m_s
        )
        layer(
            0,
            2,
            "Oranges",
            0,
            gas_color_max,
            "CO2 |u| (m/s)",
            ticks=np.linspace(0, gas_color_max, 6)
            if gas_speed_max_m_s is not None
            else None,
        )
        for phase, row, cmap in (("water", 1, "inferno"), ("gas", 2, "Blues")):
            lo, hi = (limits[f"{phase}_temperature_{s}_c"] for s in ("min", "max"))
            if not np.isfinite([lo, hi]).all():
                lo, hi = 0, 1
            if phase == "water":
                lo, hi = math.floor(lo * 2) / 2, math.ceil(hi * 2) / 2
                if hi - lo < 0.5:
                    lo = hi - 0.5
            elif hi - lo < 1e-3:
                lo, hi = lo - 1, hi + 1
            label = "Water temperature (deg C)"
            if phase == "gas":
                label = f"CO2 temperature (deg C), fixed at {float(sample['gas_temperature_c']):g}"
            layer(1, row, cmap, lo, hi, label)
        layer(2, 1, "Blues", 0, 1, "Water phase phi: 0 = CO2, 1 = water")
        for ax in self.axes:
            for collection, color in ((self.walls, "#69717b"), (self.ice, "#8de0df")):
                collection.append(
                    ax.imshow(
                        blank,
                        origin="lower",
                        extent=self.extent,
                        cmap=ListedColormap([color]),
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                        zorder=4,
                    )
                )
        self.stride = max(1, min(sample["water_phase"].shape) // 22)
        selection = np.s_[:: self.stride, :: self.stride]
        xx, yy = np.meshgrid(self.x, self.y)
        self.selection = selection
        self.quiver = self.axes[0].quiver(
            xx[selection],
            yy[selection],
            np.zeros_like(xx[selection]),
            np.zeros_like(yy[selection]),
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="mid",
            color="#1b2733",
            width=0.003,
            zorder=6,
        )
        self.reference_length_mm = dx * self.stride * 0.8
        self.axes[0].set_title("Velocity: full |u|; arrows = (ux, uy)")
        self.axes[1].set_title("Temperature: solved water / isothermal CO2")
        self.axes[2].set_title("Phase field and solid geometry")
        self.title = self.fig.suptitle("", fontsize=16, y=0.965)
        self.fig.text(
            0.5,
            0.905,
            "216^3 run: fixed colour scales over the saved sequence; gas temperature is prescribed"
            if sample["water_phase"].shape == (216, 216)
            else "Fixed colour scales over the saved sequence; gas temperature is prescribed",
            ha="center",
            fontsize=10,
            color="#4b5563",
        )
        self.fig.legend(
            handles=[
                Patch(facecolor="#69717b", label="Adiabatic tank"),
                Patch(facecolor="#8de0df", label="Dry ice"),
                Line2D([0], [0], color="#424b57", lw=0.8, label="phi = 0.5"),
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        self.fig.text(
            0.02,
            0.015,
            "Research model: global conservation errors remain",
            fontsize=8,
            color="#666666",
        )

    def update(self, data):
        phase, ids = data["water_phase"], data["solid_body_id"]
        self.title.set_text(
            f"Dry ice in hot water | t = {float(data['time_s']):.2f} s | "
            f"z = {float(data['z_m']) * 1000:.3f} mm"
        )
        for index, (key, mask) in enumerate(
            (
                ("speed_m_s", "water_mask"),
                ("speed_m_s", "gas_mask"),
                ("temperature_c", "water_mask"),
                ("temperature_c", "gas_mask"),
            )
        ):
            self.layers[index].set_data(np.ma.masked_where(~data[mask], data[key]))
        self.layers[4].set_data(np.ma.masked_where(ids != -1, phase))
        for wall, ice in zip(self.walls, self.ice, strict=True):
            wall.set_data(np.ma.masked_where(ids != -2, np.ones_like(phase)))
            ice.set_data(np.ma.masked_where(ids < 0, np.ones_like(phase)))
        for contour in self.interfaces:
            contour.remove()
        self.interfaces = []
        visible = np.ma.masked_where(ids != -1, phase)
        if visible.count() and visible.min() < 0.5 < visible.max():
            for ax in self.axes:
                self.interfaces.append(
                    ax.contour(
                        self.x,
                        self.y,
                        visible,
                        levels=[0.5],
                        colors=["#424b57"],
                        linewidths=0.6,
                        zorder=5,
                    )
                )
        select = self.selection
        velocity = data["velocity_m_s"][select]
        scale = np.where(
            data["water_mask"][select],
            self.speed_maxima["water"],
            self.speed_maxima["gas"],
        )
        valid = (ids[select] == -1) & (
            np.linalg.norm(velocity[..., :2], axis=-1) > 1e-9
        )
        self.quiver.set_UVC(
            np.ma.masked_where(
                ~valid, velocity[..., 0] / scale * self.reference_length_mm
            ),
            np.ma.masked_where(
                ~valid, velocity[..., 1] / scale * self.reference_length_mm
            ),
        )

    def save(self, path):
        self.fig.savefig(path, dpi=self.dpi)

    def close(self):
        self.plt.close(self.fig)


def render_sequence(
    run_dir, *, fps=30, dpi=110, output_dir=None, gas_speed_max_m_s=None
):
    if gas_speed_max_m_s is not None and (
        not math.isfinite(gas_speed_max_m_s) or gas_speed_max_m_s <= 0
    ):
        raise ValueError("gas speed colour maximum must be finite and positive")
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    samples = [s for s in metadata["samples"] if "slice_npz" in s]
    if not samples:
        raise ValueError("run contains no saved slices; use --save-slices")
    paths = [run_dir / s["slice_npz"] for s in samples]
    output = Path(output_dir) if output_dir is not None else run_dir / "visualization"
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    limits, times, steps = sequence_limits(paths)
    report = {
        "simulation_status": metadata["status"],
        "frame_count": len(paths),
        "times_s": times,
        "steps": steps,
        "limits": limits,
        "speed_color_limits_m_s": {
            "water": [0.0, max(limits["water_speed_max_m_s"], 1e-6)],
            "gas": [
                0.0,
                max(limits["gas_speed_max_m_s"], 1e-6)
                if gas_speed_max_m_s is None
                else gas_speed_max_m_s,
            ],
        },
        "gas_speed_color_max_override_m_s": gas_speed_max_m_s,
        "speed_color_overflow": "values above the colour maximum use the top colour",
        "video_fps": fps,
        "colormaps": {
            "speed": {"water": "viridis", "gas": "Oranges"},
            "temperature": {"water": "inferno", "gas": "Blues"},
            "phase": "Blues",
        },
        "velocity_magnitude": "sqrt(ux^2 + uy^2 + uz^2)",
        "arrows": "in-plane ux,uy, scaled separately by each phase sequence maximum",
        "gas_temperature": "prescribed isothermal reservoir; not a solved temperature",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slice_sha256": {
            str(path.relative_to(run_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
    }
    figure = SliceFigure(
        read_slice(paths[0]), limits, dpi=dpi, gas_speed_max_m_s=gas_speed_max_m_s
    )
    try:
        for i, path in enumerate(paths):
            figure.update(read_slice(path))
            figure.save(frames / f"frame_{i:05d}.png")
            if i % 25 == 0 or i == len(paths) - 1:
                print(f"rendered {i + 1}/{len(paths)}; t={times[i]:.4f}s", flush=True)
        shutil.copy2(frames / f"frame_{len(paths) - 1:05d}.png", output / "final.png")
        for axis in figure.axes:
            config = metadata["configuration"]
            dx = float(read_slice(paths[0])["dx_m"]) * 1000
            lo, hi = config["tank_lower_cells"], config["tank_upper_cells"]
            axis.set_xlim((lo[0] - 1) * dx - 2, (hi[0] - 1) * dx + 2)
            axis.set_ylim(0, (hi[1] - 1) * dx + 4)
        figure.save(output / "final_tank_zoom.png")
    finally:
        figure.close()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("PNG sequence is ready; ffmpeg is required to encode MP4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame_%05d.png"),
            "-frames:v",
            str(len(paths)),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags",
            "+faststart",
            str(output / "central_z_fields.mp4"),
        ],
        check=True,
    )
    report["status"] = "completed"
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"visualization complete: {output}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gas-speed-max-m-s",
        "--air-speed-max-m-s",
        type=float,
        help="fixed gas speed colour upper limit; raw values and arrow scaling are retained",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.dpi < 50:
        parser.error("fps must be positive and dpi must be at least 50")
    return render_sequence(
        args.run_dir,
        fps=args.fps,
        dpi=args.dpi,
        output_dir=args.output_dir,
        gas_speed_max_m_s=args.gas_speed_max_m_s,
    )


if __name__ == "__main__":
    main()
