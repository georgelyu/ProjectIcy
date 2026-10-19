"""Render per-node gradients on the frozen production step-22627 state.

The end-step cache precedes the body/refill geometry update. Preserve that
cache, then refresh the normal production gradient operator on frozen fields.
Save figures and sparse CSV/JSON values only; never serialize full grids.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.benchmarks.render_single_dambreak_step import (
    BOUNDS,
    COLORS,
    LABELS,
    MARK_COLOR,
    classify,
    setup_fonts,
)
from iceflow2d_v3.benchmarks.render_dambreak_step8544 import ice_contours
from iceflow2d_v3.examples.dambreak_single_ice_melting_2d import create_config


def neighbor_difference(data):
    """Reference difference, with center copies at blocked neighbor slots."""
    phi, solid, wall = (data[key] for key in ("phi", "solid", "wall"))
    fluid = (solid == 0) & (wall == 0)
    gradient = np.zeros((*phi.shape, 2), dtype=float)
    for i, j in zip(*np.nonzero(fluid), strict=True):
        for axis in range(2):
            values = []
            for sign in (-1, 1):
                ni, nj = i + (sign if axis == 0 else 0), j + (sign if axis == 1 else 0)
                valid = 0 <= ni < phi.shape[0] and 0 <= nj < phi.shape[1]
                values.append(phi[ni, nj] if valid and fluid[ni, nj] else phi[i, j])
            gradient[i, j, axis] = 0.5 * (values[1] - values[0])
    return gradient


def number(value):
    if value == 0:
        return "0"
    return f"{value:.2e}"


def plot_gradients(data, info, out, bounds, stem, width):
    kinds = classify(data)
    nx, ny = data["phi"].shape
    xa, xb, ya, yb = bounds
    xs = np.arange(int(np.ceil(xa)), int(np.floor(xb)) + 1)
    ys = np.arange(int(np.ceil(ya)), int(np.floor(yb)) + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    fields = (
        ("phase_gradient", "A  求解器相场压缩梯度 ∇φ"),
        ("source_gradient", "B  求解器动量源项 / 压力恢复所用的梯度 ∇φ"),
        ("reference_gradient", "C  流体邻居中心差分：对照值，非求解器实际使用的梯度"),
    )
    cap = max(
        float(np.linalg.norm(data[key][xx, yy], axis=-1).max()) for key, _ in fields
    )
    cap = max(cap, 1e-30)
    height = width * (yb - ya) / (xb - xa) * 3 + 3.5
    fig, axes = plt.subplots(3, 1, figsize=(width, height), layout="constrained")
    for ax, (key, title) in zip(axes, fields, strict=True):
        ax.imshow(
            kinds.T,
            origin="lower",
            extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
            cmap=ListedColormap(COLORS),
            vmin=0,
            vmax=4,
            interpolation="nearest",
            alpha=0.68,
        )
        ice_contours(ax, data)
        ax.axhline(2.5, color="#333333", lw=1.1, zorder=6)
        grad = data[key][xx, yy]
        magnitude = np.linalg.norm(grad, axis=-1)
        active = magnitude > 0
        arrows = grad / cap * 0.38
        ax.quiver(
            xx[active],
            yy[active] + 0.32,
            arrows[..., 0][active],
            arrows[..., 1][active],
            angles="xy",
            scale_units="xy",
            scale=1,
            pivot="mid",
            color="#a60043",
            width=0.0015,
            headwidth=4,
            headlength=5,
            minlength=0,
            zorder=12,
        )
        for i in xs:
            for j in ys:
                gx, gy = data[key][i, j]
                text = f"{number(gx)}\n{number(gy)}"
                solid_node = kinds[i, j] in (0, 4)
                label_size = 17 if gx == 0 and gy == 0 else 10.6 if len(xs) > 16 else 14
                ax.text(
                    i,
                    j - 0.035,
                    text,
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    color="#546170" if solid_node else "#12283c",
                    fontfamily="DejaVu Sans Mono",
                    linespacing=1.12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.64, pad=0.3),
                    zorder=10,
                )
                if data["fresh"][i, j]:
                    ax.add_patch(
                        Rectangle(
                            (i - 0.46, j - 0.46),
                            0.92,
                            0.92,
                            fill=False,
                            edgecolor=MARK_COLOR,
                            lw=1.25,
                            zorder=11,
                        )
                    )
        if xa <= 439 <= xb and ya <= 3 <= yb:
            ax.add_patch(
                Rectangle(
                    (438.51, 2.51),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#742451",
                    lw=2.1,
                    zorder=13,
                )
            )
        ax.set(
            xlim=(xa, xb),
            ylim=(ya, yb),
            xlabel="格点索引 i",
            ylabel="格点索引 j",
            title=title,
            aspect="equal",
        )
        ax.title.set_fontsize(15)
        ax.set_xticks(xs if len(xs) <= 16 else xs[::2])
        ax.set_yticks(ys)
        ax.set_xticks(np.arange(xa, xb + 0.01), minor=True)
        ax.set_yticks(np.arange(ya, yb + 0.01), minor=True)
        ax.grid(which="minor", color="white", lw=0.55)
        ax.tick_params(which="minor", length=0)
    handles = [
        Patch(facecolor=c, edgecolor="#344356", label=t)
        for c, t in zip(COLORS, LABELS, strict=True)
    ]
    handles += [
        Patch(facecolor="none", edgecolor=MARK_COLOR, label="本步新露出的流体点"),
        Patch(facecolor="none", edgecolor="#742451", lw=2, label="后续失稳点 (439,3)"),
        Line2D([], [], color="#a60043", lw=1.5, label="梯度箭头（三幅图统一缩放）"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)
    fig.suptitle(
        f"第 {info['step']:,} 步结束场上的逐格 ∇φ，t={info['time_s']:.9f} s\n"
        "每格上行标注 gₓ，下行标注 gᵧ；单位 LU（每格）。"
        "冰 / 墙标注存储值 0，不参与梯度求解。\n"
        "A、B：冻结流场后调用生产梯度算子重新计算，未推进时间。"
        f"箭头统一比例：{cap:.3g} LU 对应 0.38 格；零梯度无箭头。",
        fontsize=13,
    )
    path = out / f"{stem}_step{info['step']:06d}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/single_dambreak_step22627_gradients"),
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    simulation = IceFlow2D(create_config())
    simulation._thermal_stability_check_enabled = True
    simulation.step(22626)
    previous = simulation.solid_mask.to_numpy().copy()
    simulation.step()
    data = dict(
        phi=simulation.water_phase.to_numpy(),
        solid=simulation.solid_mask.to_numpy(),
        wall=simulation.wall_mask.to_numpy(),
        coverage=simulation._moving_body_fraction.to_numpy(),
        x=np.arange(simulation.nx),
        y=np.arange(simulation.ny),
    )
    data["fresh"] = (previous != 0) & (data["solid"] == 0) & (data["wall"] == 0)
    cached_phase = simulation.phase_gradient_lattice.to_numpy().copy()
    cached_source = simulation.source_gradient_lattice.to_numpy().copy()
    cached_status = simulation.phase_gradient_status.to_numpy().copy()
    pressure = simulation.pressure_lattice.to_numpy().copy()
    velocity = simulation.momentum_velocity_lattice.to_numpy().copy()
    simulation._prepare_phase_gradients()
    data["phase_gradient"] = simulation.phase_gradient_lattice.to_numpy()
    data["source_gradient"] = simulation.source_gradient_lattice.to_numpy()
    status = simulation.phase_gradient_status.to_numpy()
    assert simulation.steps == 22627
    np.testing.assert_array_equal(simulation.water_phase.to_numpy(), data["phi"])
    np.testing.assert_array_equal(simulation.pressure_lattice.to_numpy(), pressure)
    np.testing.assert_array_equal(
        simulation.momentum_velocity_lattice.to_numpy(), velocity
    )
    np.testing.assert_array_equal(simulation.solid_mask.to_numpy(), data["solid"])
    assert np.isclose(data["phi"][439, 3], 0.0004322023596614599, rtol=0, atol=1e-9)
    assert np.array_equal(np.nonzero(data["fresh"][:, 3])[0], np.arange(423, 446))
    data["reference_gradient"] = neighbor_difference(data)
    points = []
    for i in range(420, 448):
        for j in range(2, 9):
            points.append(
                dict(
                    i=i,
                    j=j,
                    phi=float(data["phi"][i, j]),
                    node_type=LABELS[int(classify(data)[i, j])],
                    solid=int(data["solid"][i, j]),
                    wall=int(data["wall"][i, j]),
                    newly_exposed=bool(data["fresh"][i, j]),
                    phase_gradient_lu=data["phase_gradient"][i, j].tolist(),
                    source_gradient_lu=data["source_gradient"][i, j].tolist(),
                    reference_difference_lu=data["reference_gradient"][i, j].tolist(),
                    recomputed_gradient_status=int(status[i, j]),
                    end_step_cached_phase_gradient_lu=cached_phase[i, j].tolist(),
                    end_step_cached_source_gradient_lu=cached_source[i, j].tolist(),
                    end_step_cached_gradient_status=int(cached_status[i, j]),
                )
            )
    csv_path = out / "point_gradients_step022627.csv"
    columns = [
        "i",
        "j",
        "phi",
        "node_type",
        "newly_exposed",
        "phase_gx_lu",
        "phase_gy_lu",
        "source_gx_lu",
        "source_gy_lu",
        "reference_gx_lu",
        "reference_gy_lu",
        "cached_phase_gx_lu",
        "cached_phase_gy_lu",
        "cached_source_gx_lu",
        "cached_source_gy_lu",
        "cached_status",
        "recomputed_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for p in points:
            writer.writerow(
                [
                    p["i"],
                    p["j"],
                    p["phi"],
                    p["node_type"],
                    p["newly_exposed"],
                    *p["phase_gradient_lu"],
                    *p["source_gradient_lu"],
                    *p["reference_difference_lu"],
                    *p["end_step_cached_phase_gradient_lu"],
                    *p["end_step_cached_source_gradient_lu"],
                    p["end_step_cached_gradient_status"],
                    p["recomputed_gradient_status"],
                ]
            )
    info = dict(
        step=simulation.steps,
        time_s=simulation.physical_time_s,
        dx_m=simulation.config.dx,
        gradient_unit="LU, per lattice cell",
        physical_gradient_conversion="gradient_m_inverse = gradient_lu / dx_m",
        gradient_stage="production gradient operator refreshed on frozen end-step-22627 fields; no time advance",
        reference_description="0.5*(neighbor_after-neighbor_before); blocked neighbors use center phi; diagnostic only",
        raw_grid_fields_saved=False,
        local_points=points,
    )
    paths = [
        plot_gradients(data, info, out, BOUNDS, "all_points_grad_phi", 24),
        plot_gradients(
            data, info, out, (429.5, 441.5, 1.5, 4.5), "detail_points_grad_phi", 14
        ),
    ]
    info["outputs"] = [str(path.resolve()) for path in paths] + [
        str(csv_path.resolve())
    ]
    (out / "gradient_annotations_step022627.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "step": info["step"],
                "time_s": info["time_s"],
                "target": next(p for p in points if (p["i"], p["j"]) == (439, 3)),
                "phase_gradient_local_max": float(
                    np.linalg.norm(data["phase_gradient"][420:448, 2:9], axis=-1).max()
                ),
                "source_gradient_local_max": float(
                    np.linalg.norm(data["source_gradient"][420:448, 2:9], axis=-1).max()
                ),
            }
        ),
        flush=True,
    )
    for path in paths:
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
