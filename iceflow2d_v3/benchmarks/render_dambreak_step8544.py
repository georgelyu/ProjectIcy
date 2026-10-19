"""Replay the production dam break and draw its exact step-8544 ghost stencil.

Only figures and sparse annotation metadata are saved. Grid fields are held
in RAM; this probe never overrides a production solver or boundary operator.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config
from iceflow2d_v3.lattice import lattice_direction


TYPE_COLORS = ["#bbc1c9", "#fff4df", "#8ed8be", "#4d96ce", "#b9cde9"]
TYPE_LABELS = [
    "地板 / 容器墙",
    "空气侧：φ < 0.001",
    "含水空气 / 过渡：0.001 ≤ φ < 0.5",
    "水侧：φ ≥ 0.5",
    "冰格点",
]


@ti.data_oriented
class MirrorProbe(IceFlow2D):
    def __init__(self, config):
        super().__init__(config)
        self.mirror_probe = ti.Vector.field(14, ti.f64, shape=8)

    @ti.kernel
    def sample_mirrors(self, fi: ti.i32, fj: ti.i32):
        for k in range(8):
            c = lattice_direction(k + 1)
            index = self._phase_sample_inside_walls(fi + c.x, fj + c.y)
            si, sj = index.x, index.y
            v = ti.Vector.zero(ti.f64, 14)
            v[0], v[1] = c.x, c.y
            v[2], v[3] = si, sj
            v[4] = self.solid_body_id[si, sj]
            v[5] = self._phase_neighbor(fi, fj, c.x, c.y, 1)
            if self.solid_mask[si, sj] != 0:
                mirror = self._ice_mirror_sample(fi, fj, si, sj, si, sj)
                eta = self._cut_link_fraction(fi, fj, si, sj)
                crossing = ti.Vector([ti.cast(fi, ti.f32), ti.cast(fj, ti.f32)])
                crossing += eta * ti.Vector(
                    [ti.cast(si - fi, ti.f32), ti.cast(sj - fj, ti.f32)]
                )
                v[6], v[7] = mirror.x, mirror.y
                v[8], v[9] = mirror.z, mirror.w
                v[10], v[11] = crossing.x, crossing.y
                v[12], v[13] = eta, 1.0
            self.mirror_probe[k] = v


def ice_contours(ax, data, *, physical=False, fill=False):
    x, y = data["x"], data["y"]
    if physical:
        x, y = (x + 0.5) * data["dx_mm"], (y + 0.5) * data["dx_mm"]
    coverage = data["coverage"]
    if fill:
        ax.contourf(
            x, y, coverage.T, levels=[0.5, 1.01], colors=[TYPE_COLORS[4]], alpha=0.52
        )
    layers = data["body_coverage"] if physical else coverage[None, ...]
    for layer in layers:
        if layer.max() > 0.5:
            ax.contour(
                x, y, layer.T, levels=[0.5], colors="#102c4c", linewidths=1.5, zorder=6
            )


def velocity_layers(
    ax, data, *, physical=False, colorbars=True, stride=1, arrow_cap=None
):
    phi, wall, solid, u = (data[k] for k in ("phi", "wall", "solid", "velocity"))
    fluid = (wall == 0) & (solid == 0)
    speed = np.linalg.norm(u, axis=-1)
    water, air = fluid & (phi >= 0.5), fluid & (phi < 0.5)
    extent = [-0.5, phi.shape[0] - 0.5, -0.5, phi.shape[1] - 0.5]
    if physical:
        extent = [0, phi.shape[0] * data["dx_mm"], 0, phi.shape[1] * data["dx_mm"]]
    background = np.where(wall != 0, 0, np.where(solid != 0, 4, 1))
    ax.imshow(
        background.T,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(TYPE_COLORS),
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )
    for mask, cmap_name, cap, label in (
        (water, "viridis", 0.8, "水侧速度模 |u| (m/s)"),
        (air, "Oranges", 4.0, "空气侧速度模 |u| (m/s)"),
    ):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad((0, 0, 0, 0))
        im = ax.imshow(
            np.ma.masked_where(~mask, speed).T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=Normalize(0, cap, clip=True),
            interpolation="nearest",
        )
        if colorbars:
            ax.figure.colorbar(
                im, ax=ax, fraction=0.033, pad=0.02, label=label, extend="max"
            )
    xs, ys = np.arange(0, phi.shape[0], stride), np.arange(0, phi.shape[1], stride)
    sample = np.ix_(xs, ys)
    uu = u[sample]
    water_sample = water[sample]
    caps = np.where(water_sample, 0.8, 4.0)
    if arrow_cap is not None:
        caps = np.full_like(caps, arrow_cap)
    lengths = np.linalg.norm(uu, axis=-1)
    arrows = (
        uu
        / np.maximum(lengths[..., None], 1e-30)
        * np.minimum(lengths / caps, 1)[..., None]
    )
    length = 0.72 if not physical else 0.85 * stride * data["dx_mm"]
    arrows *= length
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    if physical:
        xx, yy = (xx + 0.5) * data["dx_mm"], (yy + 0.5) * data["dx_mm"]
    keep = fluid[sample] & (lengths / caps >= 0.025)
    ax.quiver(
        xx[keep],
        yy[keep],
        arrows[..., 0][keep],
        arrows[..., 1][keep],
        color=np.where(water_sample[keep, None], "#f7f7f7", "#6e3300").reshape(-1),
        angles="xy",
        scale_units="xy",
        scale=1,
        pivot="mid",
        width=0.0018 if physical else 0.0035,
        minlength=0,
        zorder=5,
    )
    ice_contours(ax, data, physical=physical)
    ax.set_aspect("equal")


def node_types(data):
    phi, wall, solid = (data[k] for k in ("phi", "wall", "solid"))
    kinds = np.ones_like(solid, dtype=np.int8)
    kinds[(phi >= 0.001) & (phi < 0.5)] = 2
    kinds[phi >= 0.5] = 3
    kinds[solid != 0] = 4
    kinds[wall != 0] = 0
    return kinds


def draw_types(ax, data, bounds, *, labels=False):
    nx, ny = data["phi"].shape
    kinds = node_types(data)
    ax.imshow(
        kinds.T,
        origin="lower",
        extent=[-0.5, nx - 0.5, -0.5, ny - 0.5],
        cmap=ListedColormap(TYPE_COLORS),
        vmin=0,
        vmax=4,
        interpolation="nearest",
        alpha=0.65,
    )
    ice_contours(ax, data)
    xa, xb, ya, yb = bounds
    for i in range(max(0, int(np.ceil(xa))), min(nx, int(np.floor(xb)) + 1)):
        for j in range(max(0, int(np.ceil(ya))), min(ny, int(np.floor(yb)) + 1)):
            kind = kinds[i, j]
            marker = "s" if kind in (0, 4) else "o"
            ax.scatter(
                i,
                j,
                s=23,
                marker=marker,
                facecolor=TYPE_COLORS[kind],
                edgecolor="#36465d",
                linewidth=0.5,
                zorder=7,
            )
            if labels:
                label = f"({i},{j})"
                if kind not in (0, 4):
                    label += f"\nφ={data['phi'][i, j]:.2g}"
                ax.text(
                    i, j - 0.2, label, ha="center", va="top", fontsize=6.8, zorder=8
                )
    fresh = (data["previous_solid"] != 0) & (data["solid"] == 0) & (data["wall"] == 0)
    x, y = np.nonzero(fresh)
    keep = (x >= xa) & (x <= xb) & (y >= ya) & (y <= yb)
    ax.scatter(
        x[keep],
        y[keep],
        s=125,
        facecolor="none",
        edgecolor="#bd3374",
        linewidth=1.5,
        zorder=9,
    )
    ax.set(xlim=(xa, xb), ylim=(ya, yb), xlabel="格点索引 i", ylabel="格点索引 j")
    ax.set_xticks(np.arange(np.ceil(xa), np.floor(xb) + 1), minor=False)
    ax.set_yticks(np.arange(np.ceil(ya), np.floor(yb) + 1), minor=False)
    ax.set_xticks(np.arange(np.ceil(xa) - 0.5, xb + 0.5), minor=True)
    ax.set_yticks(np.arange(np.ceil(ya) - 0.5, yb + 0.5), minor=True)
    ax.grid(which="minor", color="#fff", linewidth=0.6)
    ax.set(xlim=(xa, xb), ylim=(ya, yb))
    ax.set_aspect("equal")


def full_velocity(data, info, out):
    fig, ax = plt.subplots(figsize=(16, 6.3), layout="constrained")
    velocity_layers(ax, data, physical=True, stride=9)
    x, y = (np.array(info["fluid_cell"]) + 0.5) * data["dx_mm"]
    ax.plot(
        x,
        y,
        "*",
        markersize=15,
        markerfacecolor="#f52b53",
        markeredgecolor="white",
        zorder=10,
    )
    ax.annotate(
        "诊断格点 (180,3)",
        xy=(x, y),
        xytext=(x - 17, y + 13),
        color="#992238",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        arrowprops={"arrowstyle": "->", "color": "#a21f45", "lw": 1.5},
        zorder=11,
    )
    ax.add_patch(
        Rectangle(
            ((172 - 0.5 + 0.5) * data["dx_mm"], 0),
            18 * data["dx_mm"],
            13.5 * data["dx_mm"],
            fill=False,
            edgecolor="#b32956",
            linestyle="--",
            linewidth=1.5,
            zorder=9,
        )
    )
    for b, center in enumerate(data["body_centers"]):
        ax.text(
            *(center * data["dx_mm"]),
            f"冰 {b + 1}",
            color="#142c4b",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=10,
        )
    ax.set(
        xlim=(0, 150),
        ylim=(0, 50),
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"完整速度场：第 8544 步结束，t={info['time_s']:.6f} s\n"
        "水侧 viridis：0–0.8 m/s；空气侧 Oranges：0–4 m/s；冰轮廓为覆盖率 0.5 等值线",
    )
    fig.text(
        0.01,
        0.008,
        f"箭头按各相色标上限缩放；超色标范围仅影响显示。实际最大速度模：水 {info['water_max_m_s']:.4g} m/s，空气 {info['air_max_m_s']:.4g} m/s。",
        fontsize=9,
    )
    fig.savefig(out / "full_velocity_step008544.png", dpi=190)
    plt.close(fig)


def local_velocity(data, info, out):
    bounds = (171.5, 189.5, -0.5, 12.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.7), layout="constrained")
    ax = axes[0]
    velocity_layers(ax, data, stride=1, arrow_cap=0.8)
    ax.set(
        xlim=bounds[:2],
        ylim=bounds[2:],
        xlabel="格点索引 i",
        ylabel="格点索引 j",
        title="局部速度场：箭头 0.8 m/s 对应 0.72 格\n微小速度省略箭头；较大速度按上限裁短",
    )
    ax.scatter(
        180, 3, s=130, facecolor="none", edgecolor="#c51d4a", linewidth=1.5, zorder=10
    )
    ax.annotate(
        "(180,3)",
        xy=(180, 3),
        xytext=(181, 1),
        color="#a22345",
        arrowprops={"arrowstyle": "->", "color": "#a22345"},
        zorder=11,
    )
    ax.axhline(2.5, color="#43474e", linewidth=1.2)
    ax = axes[1]
    draw_types(ax, data, bounds)
    ax.set_title("离散格点类型与实际冰体覆盖率轮廓")
    label_offsets = {
        (178, 3): (-10, -28),
        (179, 3): (0, 14),
        (180, 3): (0, -28),
        (179, 4): (0, 12),
        (181, 3): (14, 14),
    }
    for p, offset in label_offsets.items():
        ax.annotate(
            str(p),
            xy=p,
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            zorder=11,
            arrowprops={"arrowstyle": "-", "color": "#3c4658", "lw": 0.5},
        )
    ax.legend(
        handles=[
            Patch(facecolor=c, edgecolor="#36465d", label=t)
            for c, t in zip(TYPE_COLORS, TYPE_LABELS, strict=True)
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                color="#bd3374",
                linestyle="none",
                label="本步新暴露的流体格点",
            ),
            Line2D([], [], color="#102c4c", label="冰体覆盖率 0.5 等值线"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
    )
    fig.suptitle(
        f"(180,3) 附近：第 8544 步结束，t={info['time_s']:.6f} s\n"
        f"目标点 φ={info['fluid_phi']:.7g}，|u|={info['fluid_speed_m_s']:.5f} m/s；格点坐标 (i,j) 对应物理位置 ((i+0.5)dx,(j+0.5)dx)",
        fontsize=13,
    )
    fig.savefig(out / "local_velocity_and_node_types_step008544.png", dpi=190)
    plt.close(fig)


def mirror_process(data, info, out):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1.1, 1]},
        layout="constrained",
    )
    ax = axes[0]
    bounds = (176.8, 182.2, 1.25, 5.65)
    draw_types(ax, data, bounds)
    ax.set_title("真实格点几何 + 算法估计的镜像几何", pad=12)
    fluid, solid, crossing, mirror, normal = (
        np.array(info[k])
        for k in ("fluid_cell", "blocked_cell", "crossing", "mirror", "normal")
    )
    tangent = np.array([normal[1], -normal[0]])
    line = np.array([crossing - 4 * tangent, crossing + 4 * tangent])
    ax.plot(
        *line.T,
        color="#156d4c",
        linestyle="--",
        lw=2,
        zorder=12,
        label="算法估计冰面（局部直线）",
    )
    ax.annotate(
        "",
        xy=crossing + 0.95 * normal,
        xytext=crossing,
        arrowprops={"arrowstyle": "-|>", "color": "#156d4c", "lw": 2},
        zorder=14,
    )
    ax.text(
        *(crossing + 1.05 * normal),
        "n（指向冰内）",
        color="#156d4c",
        ha="left",
        fontsize=9,
        zorder=14,
    )
    ax.plot(
        *np.array([solid, mirror]).T, color="#8043ac", linestyle="-.", lw=1.8, zorder=13
    )
    foot = (solid + mirror) / 2
    ax.plot(*foot, "o", color="#8043ac", markersize=4, zorder=14)
    ax.annotate(
        "",
        xy=solid,
        xytext=fluid,
        arrowprops={"arrowstyle": "->", "color": "#304b70", "lw": 1.5},
        zorder=13,
    )
    ax.plot(*fluid, "*", markersize=17, color="#be224d", zorder=15)
    ax.plot(*solid, "s", markersize=9, color="#304b70", zorder=15)
    ax.plot(*crossing, "D", markersize=6, color="#156d4c", zorder=16)
    ax.plot(
        *mirror, "x", markersize=12, color="#8043ac", markeredgewidth=2.2, zorder=16
    )
    annotations = [
        (fluid, "F=(180,3)\n新暴露空气点", (181.05, 2.98), "#be224d"),
        (solid, "S=(179,4)\n被冰阻挡的采样点", (177.75, 4.9), "#304b70"),
        (crossing, "切链交点 B", (181.08, 3.63), "#156d4c"),
        (mirror, "镜像点 M", (177.0, 1.45), "#8043ac"),
    ]
    for point, label, text, color in annotations:
        ax.annotate(
            label,
            xy=point,
            xytext=text,
            fontsize=9,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
            zorder=17,
        )
    ax.add_patch(
        Rectangle(
            (178, 2),
            1,
            1,
            fill=False,
            edgecolor="#8043ac",
            linestyle=":",
            lw=1.8,
            zorder=13,
        )
    )
    ax.axhline(2.5, color="#515760", lw=1.5, zorder=10)
    ax.text(
        180.0,
        2.37,
        "地板反射轴 j=2.5",
        fontsize=8,
        color="#515760",
        ha="center",
        zorder=14,
    )
    for x, color in ((178, "#a12c28"), (179, "#666")):
        ax.annotate(
            "",
            xy=(x, 3),
            xytext=(x, 2),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
            zorder=14,
        )
    ax.scatter(
        [178], [3], s=170, facecolor="none", edgecolor="#a12c28", lw=2, zorder=15
    )
    ax.annotate(
        "有效供体 (178,3)\nφ=0.23368",
        xy=(178, 3),
        xytext=(176.95, 3.7),
        fontsize=9,
        color="#a12c28",
        arrowprops={"arrowstyle": "->", "color": "#a12c28"},
        bbox={"facecolor": "white", "alpha": 0.87, "edgecolor": "none"},
        zorder=17,
    )
    ax.legend(
        handles=[
            Line2D([], [], color="#102c4c", label="覆盖率 0.5：求解器冰轮廓"),
            Line2D(
                [],
                [],
                color="#156d4c",
                linestyle="--",
                label="切链交点 + 法向：估计冰面",
            ),
            Line2D(
                [],
                [],
                color="#8043ac",
                linestyle="-.",
                label="S → M：关于估计冰面的镜像",
            ),
            Line2D([], [], color="#a12c28", label="墙内供体 j=2 → j=3"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.93,
    )

    inset = ax.inset_axes([0.71, 0.015, 0.27, 0.21])
    inset.set_facecolor("#fff9ee")
    ice_contours(inset, data, fill=True)
    inset.plot(*line.T, color="#156d4c", linestyle="--", lw=1.2)
    inset.plot(*np.array([fluid, crossing]).T, color="#304b70", lw=0.8)
    inset.plot(*fluid, "*", color="#be224d", markersize=8)
    inset.plot(*crossing, "D", color="#156d4c", markersize=4)
    inset.annotate(
        "",
        xy=crossing + 0.065 * normal,
        xytext=crossing,
        arrowprops={"arrowstyle": "->", "color": "#156d4c", "lw": 1},
    )
    inset.text(*(fluid + np.array([0.012, -0.012])), "F", fontsize=7, color="#be224d")
    inset.text(
        *(crossing + np.array([-0.02, -0.012])), "B", fontsize=7, color="#156d4c"
    )
    inset.set(
        xlim=(179.85, 180.11),
        ylim=(2.955, 3.15),
        xticks=[179.9, 180.0],
        yticks=[3.0, 3.1],
    )
    inset.tick_params(labelsize=6, pad=1)
    inset.ticklabel_format(useOffset=False, style="plain", axis="x")
    inset.set_title("F 与 B 放大：η=0.05", fontsize=8, pad=2)
    inset.set_aspect("equal")

    ax = axes[1]
    ax.axis("off")
    ax.text(
        0, 0.99, "直接读取生产内核的几何量", fontsize=14, fontweight="bold", va="top"
    )
    mx, my = mirror
    nx, ny = normal
    bx, by = crossing
    ax.text(
        0,
        0.92,
        f"方向 q6=(-1,+1)，冰块编号 {info['body_id'] + 1}\n"
        f"切链比例 η={info['eta']:.7f}（本处触及代码下限 0.05）\n"
        f"B=({bx:.6f}, {by:.6f})\n"
        f"n=({nx:.6f}, {ny:.6f})\n"
        f"估计冰面：n·(x−B)=0\n"
        f"镜像点 M=({mx:.6f}, {my:.6f})\n"
        "镜像关系：M=S−2[(S−B)·n]n",
        fontsize=11,
        va="top",
        linespacing=1.65,
    )
    rows = [
        [
            str(tuple(s["raw_cell"])),
            str(tuple(s["mapped_cell"])),
            f"{s['weight']:.6f}",
            "接受：流体" if s["accepted"] else "剔除：冰",
        ]
        for s in info["donors"]
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["原始供体", "地板反射后", "原始权重", "有效性"],
        cellLoc="center",
        bbox=[0, 0.31, 0.98, 0.25],
        colWidths=[0.23, 0.23, 0.22, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dde3e9")
        if row == 0:
            cell.set_facecolor("#e9eef5")
        else:
            cell.set_facecolor(
                "#edf7f0" if info["donors"][row - 1]["accepted"] else "#eceef1"
            )
    ax.text(
        0,
        0.24,
        f"有效权重合计 = {info['accepted_weight']:.6f}\n"
        f"归一化后 φghost = {info['ghost_phi']:.8f}\n"
        f"F 的真实相场 φF = {info['fluid_phi']:.8g}\n"
        "供体只检查自身是不是流体；未检查到 F 的局部路径。\n"
        "(178,3) 与 F 之间的 (179,3) 是冰，因而读入了\n"
        "冰壁另一侧的相场值。",
        fontsize=11,
        va="top",
        linespacing=1.6,
    )
    fig.suptitle(
        f"冰壁镜像插值：第 8544 步结束的实际状态，t={info['time_s']:.6f} s\n"
        "坐标采用格点索引；点 M 高于地板反射轴，但其双线性插值模板包含 j=2",
        fontsize=13,
    )
    fig.savefig(out / "ice_mirror_interpolation_step008544.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chinese_font = next(
        name
        for name in ("Noto Sans CJK SC", "Noto Sans CJK JP")
        if name in available_fonts
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    s = MirrorProbe(replace(create_config(), phase_gradient_scheme="legacy"))
    s._thermal_stability_check_enabled = True
    s.step(8543)
    previous_solid = s.solid_mask.to_numpy().copy()
    s.step(1)
    s.sample_mirrors(180, 3)
    row = s.mirror_probe.to_numpy()[5]
    assert tuple(row[:2]) == (-1, 1) and row[13] == 1
    phi, solid, wall = (
        s.water_phase.to_numpy(),
        s.solid_mask.to_numpy(),
        s.wall_mask.to_numpy(),
    )
    velocity = s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s
    data = dict(
        phi=phi,
        solid=solid,
        wall=wall,
        velocity=velocity,
        previous_solid=previous_solid,
        coverage=s._moving_body_fraction.to_numpy(),
        body_coverage=s._body_fractions.to_numpy(),
        x=np.arange(s.nx),
        y=np.arange(s.ny),
        dx_mm=s.config.dx * 1000,
        body_centers=s.body_center.to_numpy(),
    )
    mx, my = row[6:8]
    base = np.floor([mx, my]).astype(int)
    donors = []
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        i, j = base + [di, dj]
        weight = max(0, 1 - abs(mx - i)) * max(0, 1 - abs(my - j))
        ri, rj = int(i), int(j)
        lower = s.config.boundary_cells
        if ri < lower:
            ri = 2 * lower - 1 - ri
        elif ri >= s.nx - lower:
            ri = 2 * (s.nx - lower) - 1 - ri
        if rj < lower:
            rj = 2 * lower - 1 - rj
        elif rj >= s.ny - lower:
            rj = 2 * (s.ny - lower) - 1 - rj
        donors.append(
            dict(
                raw_cell=[int(i), int(j)],
                mapped_cell=[ri, rj],
                weight=float(weight),
                phi=float(phi[ri, rj]),
                accepted=bool(solid[ri, rj] == 0 and wall[ri, rj] == 0),
            )
        )
    fluid = (solid == 0) & (wall == 0)
    speed = np.linalg.norm(velocity, axis=-1)
    accepted_weight = sum(d["weight"] for d in donors if d["accepted"])
    expected_phi = (
        sum(d["weight"] * d["phi"] for d in donors if d["accepted"]) / accepted_weight
    )
    normal = row[8:10]
    crossing = row[10:12]
    solid_point = row[2:4]
    independent_mirror = (
        solid_point - 2 * np.dot(solid_point - crossing, normal) * normal
    )
    assert np.allclose(independent_mirror, row[6:8], atol=3e-5)
    assert np.isclose(expected_phi, row[5], atol=5e-8)
    assert previous_solid[180, 3] == 1 and solid[180, 3] == 0
    assert np.allclose([mx, my], [178.670349, 2.640753], atol=2e-4), (
        "Replay differs from the diagnosed state"
    )
    info = dict(
        step=s.steps,
        time_s=s.physical_time_s,
        state_stage="after full flow/body/thermal step",
        coordinate_convention="cell-index centers; physical position=(index+0.5)*dx",
        fluid_cell=[180, 3],
        blocked_cell=solid_point.astype(int).tolist(),
        body_id=int(row[4]),
        mirror=[float(mx), float(my)],
        normal=normal.tolist(),
        crossing=crossing.tolist(),
        eta=float(row[12]),
        ghost_phi=float(row[5]),
        fluid_phi=float(phi[180, 3]),
        fluid_speed_m_s=float(speed[180, 3]),
        fluid_velocity_m_s=velocity[180, 3].tolist(),
        coverage_at_fluid_cell=float(data["coverage"][180, 3]),
        donors=donors,
        accepted_weight=accepted_weight,
        water_max_m_s=float(speed[fluid & (phi >= 0.5)].max()),
        air_max_m_s=float(speed[fluid & (phi < 0.5)].max()),
        contour_definition="world ice coverage=0.5; same threshold as production fluid/solid mask",
        node_types=TYPE_LABELS,
        configuration=s.config.to_dict(),
        raw_fields_saved=False,
    )
    package = Path(__file__).resolve().parents[1]
    info["source_sha256"] = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in (
            "simulator.py",
            "config.py",
            "stability_options.py",
            "examples/dambreak_four_ice_melting_2d.py",
        )
    }
    print(
        json.dumps(
            {
                k: info[k]
                for k in (
                    "step",
                    "time_s",
                    "fluid_phi",
                    "mirror",
                    "normal",
                    "crossing",
                    "eta",
                    "ghost_phi",
                )
            }
        ),
        flush=True,
    )
    full_velocity(data, info, out)
    local_velocity(data, info, out)
    mirror_process(data, info, out)
    (out / "annotations_step008544.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    for path in sorted(out.iterdir()):
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
