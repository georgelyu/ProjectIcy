"""Replay and render one exact production single-ice dam-break step.

Save a full-domain phase/velocity figure and a local phase/node-type figure.
Grid arrays stay in RAM; sparse point metadata records the displayed state.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.benchmarks.render_dambreak_step8544 import (
    ice_contours,
    velocity_layers,
)
from iceflow2d_v3.examples.dambreak_single_ice_melting_2d import create_config


COLORS = ["#8e9299", "#fff0c9", "#66c6b3", "#3b83c2", "#c0cce0"]
LABELS = [
    "容器墙 / 地板",
    "气相流体：φ < 0.001",
    "界面 / 混合流体：0.001 ≤ φ < 0.999",
    "水相流体：φ ≥ 0.999",
    "冰体格点",
]
MARK_COLOR = "#d12e68"
BOUNDS = (419.5, 447.5, 1.5, 8.5)


def setup_fonts():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )


def classify(data):
    phi = data["phi"]
    kinds = np.ones(phi.shape, dtype=np.int8)
    kinds[(phi >= 0.001) & (phi < 0.999)] = 2
    kinds[phi >= 0.999] = 3
    kinds[data["solid"] != 0] = 4
    kinds[data["wall"] != 0] = 0
    return kinds


def full_domain(data, info, out):
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), layout="constrained")
    nx, ny = data["phi"].shape
    extent = (0, nx * data["dx_mm"], 0, ny * data["dx_mm"])
    blocked = (data["solid"] != 0) | (data["wall"] != 0)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#aeb4ba")
    image = axes[0].imshow(
        np.ma.masked_where(blocked, data["phi"]).T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.025, pad=0.015, label="水相比例 φ")
    ice_contours(axes[0], data, physical=True)
    xs = (data["x"] + 0.5) * data["dx_mm"]
    ys = (data["y"] + 0.5) * data["dx_mm"]
    axes[0].contour(
        xs,
        ys,
        np.ma.masked_where(blocked, data["phi"]).T,
        levels=[0.5],
        colors="#38444e",
        linewidths=0.65,
        linestyles="--",
    )
    axes[0].set_title("完整计算域：相场（灰色为冰体与墙，虚线为 φ=0.5）")
    velocity_layers(axes[1], data, physical=True, stride=9)
    axes[1].set_title(
        "完整计算域：速度（水 0–0.8 m/s；空气 0–4 m/s，分别使用固定色标）"
    )
    point = (np.array(info["target_cell"]) + 0.5) * data["dx_mm"]
    xa, xb, ya, yb = BOUNDS
    for ax in axes:
        ax.add_patch(
            Rectangle(
                ((xa + 0.5) * data["dx_mm"], (ya + 0.5) * data["dx_mm"]),
                (xb - xa) * data["dx_mm"],
                (yb - ya) * data["dx_mm"],
                fill=False,
                edgecolor=MARK_COLOR,
                linestyle="--",
                lw=1.7,
                zorder=10,
            )
        )
        ax.plot(
            *point, "*", ms=13, color=MARK_COLOR, markeredgecolor="white", zorder=11
        )
        ax.annotate(
            "放大区域；标记后续失稳点 (439,3)",
            xy=point,
            xytext=(98, 22),
            color="#a21f51",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color=MARK_COLOR),
            zorder=12,
        )
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="x (mm)", ylabel="y (mm)")
    fig.suptitle(
        f"单冰块溃坝：第 {info['step']:,} 步结束，t={info['time_s']:.9f} s\n"
        "默认生产配置；此刻尚未失稳，后续在第 22,642 步触发失败",
        fontsize=15,
    )
    fig.supxlabel(
        f"实际最大速度模：水侧 {info['water_speed_max_m_s']:.3f} m/s，"
        f"空气侧 {info['air_speed_max_m_s']:.3f} m/s。"
        "色标与箭头长度的裁剪仅影响显示。",
        fontsize=10,
    )
    path = out / f"full_domain_step{info['step']:06d}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def zoom_view(data, info, out):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), layout="constrained")
    nx, ny = data["phi"].shape
    extent = (-0.5, nx - 0.5, -0.5, ny - 0.5)
    kinds = classify(data)
    fresh = data["fresh"]
    background = np.where(data["wall"] != 0, 0, np.where(data["solid"] != 0, 4, 1))
    axes[0].imshow(
        background.T,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(COLORS),
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad((0, 0, 0, 0))
    phase = np.ma.masked_where((data["wall"] != 0) | (data["solid"] != 0), data["phi"])
    image = axes[0].imshow(
        phase.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=0,
        vmax=0.3,
        interpolation="nearest",
    )
    fig.colorbar(
        image,
        ax=axes[0],
        fraction=0.022,
        pad=0.015,
        label="局部水相比例 φ（色标 0–0.3）",
        extend="max",
    )
    axes[1].imshow(
        kinds.T,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(COLORS),
        vmin=0,
        vmax=4,
        interpolation="nearest",
        alpha=0.72,
    )
    xa, xb, ya, yb = BOUNDS
    xx, yy = np.meshgrid(np.arange(420, 448), np.arange(2, 9), indexing="ij")
    for ax in axes:
        ice_contours(ax, data)
        ax.axhline(2.5, color="#333333", lw=1.3, zorder=6)
        for kind in range(5):
            keep = kinds[xx, yy] == kind
            ax.scatter(
                xx[keep],
                yy[keep],
                marker="s" if kind in (0, 4) else "o",
                s=25,
                facecolors=COLORS[kind],
                edgecolors="#344356",
                lw=0.65,
                zorder=7,
            )
        fx, fy = np.nonzero(fresh)
        keep = (fx >= xa) & (fx <= xb) & (fy >= ya) & (fy <= yb)
        ax.scatter(
            fx[keep],
            fy[keep],
            s=215,
            facecolors="none",
            edgecolors=MARK_COLOR,
            lw=1.5,
            zorder=8,
        )
        target = info["target_cell"]
        ax.plot(
            *target, "*", ms=17, color=MARK_COLOR, markeredgecolor="white", zorder=10
        )
        ax.annotate(
            "F=(439,3)\n后续第 22,642 步在此失稳",
            xy=target,
            xytext=(433, 6.6),
            color="#a21f51",
            fontsize=10,
            ha="center",
            bbox=dict(facecolor="white", edgecolor="#ead3df", alpha=0.93),
            arrowprops=dict(arrowstyle="->", color=MARK_COLOR, lw=1.2),
            zorder=11,
        )
        ax.set(
            xlim=(xa, xb),
            ylim=(ya, yb),
            xlabel="格点索引 i",
            ylabel="格点索引 j",
            aspect="equal",
        )
        ax.set_xticks(np.arange(420, 448, 2))
        ax.set_yticks(np.arange(2, 9))
        ax.set_xticks(np.arange(419.5, 448, 1), minor=True)
        ax.set_yticks(np.arange(1.5, 9, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.65, alpha=0.8)
        ax.tick_params(which="minor", length=0)
    axes[0].set_title("相场与实际冰面：本步打开单格气隙，湿、干补点在 i=434 附近拼接")
    for i in range(420, 447):
        if kinds[i, 3] not in (0, 4):
            phi = data["phi"][i, 3]
            label = f"{phi:.3f}" if phi >= 0.001 else f"{phi:.1e}"
            axes[0].text(
                i,
                3.42,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#233548",
                zorder=12,
            )
    axes[0].text(422, 5.8, "上方：冰体", fontsize=11, color="#223c59")
    axes[0].text(421, 2, "地板", fontsize=10, color="#ffffff", zorder=9)
    axes[1].set_title(
        "格点类型：颜色表示相 / 固体类型；红圈表示第 22,627 步新露出的流体点"
    )
    handles = [
        Patch(facecolor=c, edgecolor="#344356", label=t)
        for c, t in zip(COLORS, LABELS, strict=True)
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=MARK_COLOR,
            lw=0,
            label="本步新露出的流体格点",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color=MARK_COLOR,
            lw=0,
            markersize=11,
            label="后续失稳点 (439,3)",
        ),
        Line2D([], [], color="#102c4c", lw=1.5, label="实际冰体覆盖率 0.5 轮廓"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=4, fontsize=9, frameon=True
    )
    fig.suptitle(
        f"第 {info['step']:,} 步失稳点附近放大，t={info['time_s']:.9f} s\n"
        "类型阈值仅用于显示；圆点是流体，方点是固体。红圈是历史标记，可与相类型重叠。",
        fontsize=14,
    )
    path = out / f"zoom_phase_node_types_step{info['step']:06d}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=22627)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/single_dambreak_step22627_visualization"),
    )
    args = parser.parse_args()
    if args.step < 1:
        parser.error("step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    started = time.monotonic()
    config = create_config()
    simulation = IceFlow2D(config)
    simulation._thermal_stability_check_enabled = True
    simulation.step(args.step - 1)
    previous = simulation.solid_mask.to_numpy().copy()
    simulation.step()
    phi = simulation.water_phase.to_numpy()
    solid = simulation.solid_mask.to_numpy()
    wall = simulation.wall_mask.to_numpy()
    velocity = (
        simulation.physical_velocity_lattice.to_numpy()
        * simulation.scales.velocity_scale_m_s
    )
    fresh = (previous != 0) & (solid == 0) & (wall == 0)
    data = dict(
        phi=phi,
        solid=solid,
        wall=wall,
        velocity=velocity,
        previous_solid=previous,
        fresh=fresh,
        coverage=simulation._moving_body_fraction.to_numpy(),
        body_coverage=simulation._body_fractions.to_numpy(),
        x=np.arange(config.nx),
        y=np.arange(config.ny),
        dx_mm=config.dx * 1000,
    )
    fluid = (solid == 0) & (wall == 0)
    speed = np.linalg.norm(velocity, axis=-1)
    cell = (439, 3)
    info = dict(
        step=simulation.steps,
        time_s=simulation.physical_time_s,
        target_cell=list(cell),
        target_phi=float(phi[cell]),
        target_velocity_m_s=velocity[cell].tolist(),
        target_speed_m_s=float(speed[cell]),
        target_newly_exposed=bool(fresh[cell]),
        water_speed_max_m_s=float(speed[fluid & (phi >= 0.5)].max()),
        air_speed_max_m_s=float(speed[fluid & (phi < 0.5)].max()),
        body_center_cells=simulation.body_center.to_numpy().tolist(),
        body_angle_rad=simulation.body_angle.to_numpy().tolist(),
        new_floor_cells=np.nonzero(fresh[:, 3])[0].tolist(),
        configuration=config.to_dict(),
        raw_grid_fields_saved=False,
        node_type_thresholds=dict(air_below=0.001, water_at_or_above=0.999),
        stage="end of complete production step, after body/refill and thermal fast update",
        marker_description="(439,3) fails later at step 22642; it is not unstable at step 22627",
    )
    if args.step == 22627:
        assert info["new_floor_cells"] == list(range(423, 446)), info["new_floor_cells"]
        assert np.isclose(phi[cell], 0.0004322023596614599, rtol=0, atol=1e-9), phi[
            cell
        ]
    local = []
    for i in range(420, 448):
        for j in range(2, 9):
            local.append(
                dict(
                    cell=[i, j],
                    phi=float(phi[i, j]),
                    solid=int(solid[i, j]),
                    wall=int(wall[i, j]),
                    newly_exposed=bool(fresh[i, j]),
                    velocity_m_s=velocity[i, j].tolist(),
                )
            )
    info["local_points"] = local
    package = Path(__file__).resolve().parents[1]
    info["source_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "simulator.py",
            "phase_reconstruction.py",
            "stability_options.py",
            "examples/dambreak_single_ice_melting_2d.py",
        )
    }
    print(
        json.dumps(
            {
                key: info[key]
                for key in ("step", "time_s", "target_phi", "new_floor_cells")
            }
        ),
        flush=True,
    )
    outputs = [
        full_domain(data, info, args.output_dir),
        zoom_view(data, info, args.output_dir),
    ]
    info["outputs"] = [str(path.resolve()) for path in outputs]
    info["elapsed_s"] = time.monotonic() - started
    (args.output_dir / f"annotations_step{args.step:06d}.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    for path in outputs:
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
