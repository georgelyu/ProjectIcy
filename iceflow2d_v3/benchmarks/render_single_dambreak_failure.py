"""Render actual phi/gradient fields during the step-22642 failure.

Replay unmodified production kernels. Capture the collision gradient, the
streamed phase before bounding, and the exact state rejected before thermal
coupling. No diagnostic gradient refresh is applied to a captured state.
Only figures and sparse point records are saved; full grids stay in RAM.
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
from iceflow2d_v3.benchmarks.render_dambreak_step8544 import ice_contours
from iceflow2d_v3.benchmarks.render_single_dambreak_gradients import number
from iceflow2d_v3.benchmarks.render_single_dambreak_step import (
    BOUNDS,
    COLORS,
    LABELS,
    MARK_COLOR,
    classify,
    setup_fonts,
)
from iceflow2d_v3.examples.dambreak_single_ice_melting_2d import create_config


STEP = 22642


def capture(s, stage, opening_mask):
    data = dict(
        phi=s.water_phase.to_numpy().copy(),
        solid=s.solid_mask.to_numpy().copy(),
        wall=s.wall_mask.to_numpy().copy(),
        coverage=s._moving_body_fraction.to_numpy().copy(),
        phase_gradient=s.phase_gradient_lattice.to_numpy().copy(),
        source_gradient=s.source_gradient_lattice.to_numpy().copy(),
        gradient_status=s.phase_gradient_status.to_numpy().copy(),
        velocity=s.physical_velocity_lattice.to_numpy().copy(),
        x=np.arange(s.nx),
        y=np.arange(s.ny),
        opening_mask=opening_mask,
    )
    data["stage"] = stage
    return data


def grid_style(ax, data, bounds, *, contours=True):
    xa, xb, ya, yb = bounds
    nx, ny = data["phi"].shape
    ax.imshow(
        classify(data).T,
        origin="lower",
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
        cmap=ListedColormap(COLORS),
        vmin=0,
        vmax=4,
        interpolation="nearest",
        alpha=0.68,
    )
    if contours:
        ice_contours(ax, data)
    ax.axhline(2.5, color="#333333", lw=1.1, zorder=6)
    xs = np.arange(int(np.ceil(xa)), int(np.floor(xb)) + 1)
    ys = np.arange(int(np.ceil(ya)), int(np.floor(yb)) + 1)
    for i in xs:
        for j in ys:
            if data["opening_mask"][i, j]:
                ax.add_patch(
                    Rectangle(
                        (i - 0.46, j - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor=MARK_COLOR,
                        lw=1.15,
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
                lw=2.2,
                zorder=13,
            )
        )
    ax.set(
        xlim=(xa, xb),
        ylim=(ya, yb),
        xlabel="格点索引 i",
        ylabel="格点索引 j",
        aspect="equal",
    )
    ax.set_xticks(xs if len(xs) <= 16 else xs[::2])
    ax.set_yticks(ys)
    ax.set_xticks(np.arange(xa, xb + 0.01), minor=True)
    ax.set_yticks(np.arange(ya, yb + 0.01), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    return xs, ys


def legend(fig):
    handles = [
        Patch(facecolor=c, edgecolor="#344356", label=t)
        for c, t in zip(COLORS, LABELS, strict=True)
    ]
    handles += [
        Patch(
            facecolor="none",
            edgecolor=MARK_COLOR,
            label="第 22,627 步露出的点（历史区域）",
        ),
        Patch(facecolor="none", edgecolor="#742451", lw=2, label="本次失稳点 (439,3)"),
        Line2D([], [], color="#a60043", lw=1.5, label="实际梯度箭头"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9)


def render(data, time_s, out, stem, bounds, width, explanation):
    xa, xb, ya, yb = bounds
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width, width * (yb - ya) / (xb - xa) * 3 + 3.5),
        layout="constrained",
    )
    fields = (
        ("phi", "A  当时的相场 φ（逐格数值）"),
        ("phase_gradient", "B  当时求解器实际相场压缩梯度 ∇φ"),
        ("source_gradient", "C  当时求解器实际动量源项 / 压力恢复梯度 ∇φ"),
    )
    cap = max(
        float(
            np.linalg.norm(
                data[key][
                    int(np.ceil(xa)) : int(np.floor(xb)) + 1,
                    int(np.ceil(ya)) : int(np.floor(yb)) + 1,
                ],
                axis=-1,
            ).max()
        )
        for key in ("phase_gradient", "source_gradient")
    )
    cap = max(cap, 1e-30)
    kinds = classify(data)
    for ax, (key, title) in zip(axes, fields, strict=True):
        xs, ys = grid_style(ax, data, bounds)
        ax.set_title(title, fontsize=15)
        for i in xs:
            for j in ys:
                if key == "phi":
                    solid_node = kinds[i, j] in (0, 4)
                    value = data[key][i, j]
                    text = (
                        "冰"
                        if kinds[i, j] == 4
                        else "墙"
                        if kinds[i, j] == 0
                        else number(value)
                    )
                    size = 13 if len(xs) <= 16 else 10.5
                    color = (
                        "#546170"
                        if solid_node
                        else "#b31536"
                        if value < 0 or value > 1
                        else "#12283c"
                    )
                else:
                    gx, gy = data[key][i, j]
                    text = f"{number(gx)}\n{number(gy)}"
                    size = 17 if gx == 0 and gy == 0 else 14 if len(xs) <= 16 else 10.5
                    color = "#546170" if kinds[i, j] in (0, 4) else "#12283c"
                ax.text(
                    i,
                    j - 0.035,
                    text,
                    ha="center",
                    va="center",
                    fontsize=size,
                    fontfamily="DejaVu Sans Mono"
                    if key != "phi" or kinds[i, j] not in (0, 4)
                    else None,
                    color=color,
                    linespacing=1.12,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.3),
                    zorder=10,
                )
        if key != "phi":
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            gradient = data[key][xx, yy]
            active = np.linalg.norm(gradient, axis=-1) > 0
            arrows = gradient / cap * 0.38
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
    legend(fig)
    fig.suptitle(
        f"失稳发生时：尝试第 {STEP:,} 步，t={time_s:.9f} s\n"
        f"{explanation}\n"
        "梯度上行 gₓ，下行 gᵧ，单位 LU（每格）；零梯度无箭头。"
        "全部使用当步生产字段，未做额外梯度重算。",
        fontsize=13,
    )
    path = out / f"{stem}_step{STEP:06d}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def sequence(states, out):
    xs = np.arange(430, 443)
    fig, axes = plt.subplots(
        len(states), 1, figsize=(15, 2.1 * len(states) + 2), layout="constrained"
    )
    names = {
        "end_22627": "第 22,627 步：刚补点",
        "end_22633": "第 22,633 步：气隙内扰动已增长",
        "end_22634": "第 22,634 步：气隙速度继续放大",
        "end_22638": "第 22,638 步：失稳扩大",
        "end_22641": "第 22,641 步：爆发前一步",
        "macros_unbounded": "第 22,642 步：宏观量恢复后，φ 尚未限幅",
    }
    for ax, data in zip(axes, states, strict=True):
        grid_style(ax, data, (429.5, 442.5, 2.5, 3.5), contours=False)
        for i in xs:
            phi = data["phi"][i, 3]
            speed = np.abs(data["velocity"][i, 3]).sum()
            ax.text(
                i,
                3.06,
                f"φ={phi:.2e}",
                ha="center",
                va="center",
                fontsize=9,
                color="#b31536" if phi < 0 else "#12283c",
                fontfamily="DejaVu Sans Mono",
            )
            ax.text(
                i,
                2.83,
                f"u₁={speed:.2e}",
                ha="center",
                va="center",
                fontsize=9,
                color="#526477",
            )
        ax.set_title(names[data["stage"]], fontsize=13)
        ax.set_yticks([3])
    fig.suptitle(
        "从补点到失稳：同一排气隙的实际 φ 与速度 L1 范数（LU）\n"
        "红框为第 22,627 步露出区域，深紫框为 (439,3)",
        fontsize=14,
    )
    path = out / "opening_to_failure_phi_step022642.png"
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def sparse(data):
    kinds = classify(data)
    return [
        dict(
            i=i,
            j=j,
            phi=float(data["phi"][i, j]),
            node_type=LABELS[int(kinds[i, j])],
            solid=int(data["solid"][i, j]),
            wall=int(data["wall"][i, j]),
            opened_at_step22627=bool(data["opening_mask"][i, j]),
            phase_gradient_lu=data["phase_gradient"][i, j].tolist(),
            source_gradient_lu=data["source_gradient"][i, j].tolist(),
            gradient_status=int(data["gradient_status"][i, j]),
            velocity_lattice=data["velocity"][i, j].tolist(),
            speed_l1_lu=float(np.abs(data["velocity"][i, j]).sum()),
        )
        for i in range(420, 448)
        for j in range(2, 9)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/single_dambreak_failure_step22642"),
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    s = IceFlow2D(create_config())
    s._thermal_stability_check_enabled = True
    s.step(22626)
    previous = s.solid_mask.to_numpy().copy()
    s.step()
    opening = (
        (previous != 0) & (s.solid_mask.to_numpy() == 0) & (s.wall_mask.to_numpy() == 0)
    )
    states = [capture(s, "end_22627", opening)]
    for step in (22633, 22634, 22638, 22641):
        s.step(step - s.steps)
        states.append(capture(s, f"end_{step}", opening))
    s._collide_momentum()
    collision = capture(s, "collision_gradient", opening)
    s._stream()
    s._update_streamed_macroscopic_fields()
    raw = capture(s, "macros_unbounded", opening)
    assert np.isclose(raw["phi"][439, 3], -0.0010235144291073084, rtol=0, atol=1e-9)
    assert np.isclose(
        np.abs(raw["velocity"][439, 3]).sum(), 709.3110175728798, rtol=1e-7
    )
    s._bound_water_phase()
    s._prepare_phase_gradients()
    s._update_pressure()
    pressure = capture(s, "bounded_phase_and_pressure", opening)
    s._advance_body()
    failure = capture(s, "rejected_before_thermal", opening)
    error = None
    try:
        s._advance_thermal()
    except RuntimeError as exc:
        error = str(exc)
    assert error is not None and "at step 22642" in error, error
    assert s.steps == 22641
    time_s = STEP * s.scales.dt_s
    records = states + [collision, raw, pressure, failure]
    info = dict(
        attempted_step=STEP,
        attempted_time_s=time_s,
        last_completed_step=s.steps,
        error=error,
        dx_m=s.config.dx,
        raw_grid_fields_saved=False,
        stages={data["stage"]: sparse(data) for data in records},
        gradient_time_levels={
            "collision_gradient": "computed by production collision on step-22641 phase/geometry; used in attempted step 22642",
            "macros_unbounded": "same collision gradient, before production re-evaluation on newly streamed phi",
            "bounded_phase_and_pressure": "production gradient refreshed after phi bounds, used in pressure recovery",
            "rejected_before_thermal": "actual stored gradient retained from pressure stage; no diagnostic refresh after body update",
        },
    )
    paths = [
        render(
            failure,
            time_s,
            out,
            "failure_phi_actual_gradients_all",
            BOUNDS,
            24,
            "触发热耦合前稳定性检查的实际状态；φ 已按生产流程限幅，步骤尚未完成。",
        ),
        render(
            failure,
            time_s,
            out,
            "failure_phi_actual_gradients_detail",
            (429.5, 441.5, 1.5, 4.5),
            14,
            "触发报错时的实际状态；(439,3) 的速度 L1 范数已达 709.311 LU。",
        ),
        render(
            raw,
            time_s,
            out,
            "failure_unbounded_phi_and_collision_gradients",
            (429.5, 441.5, 1.5, 4.5),
            14,
            "流送 / 宏观量恢复后：φ 尚未限幅；B、C 为本步碰撞实际使用的梯度缓存。",
        ),
        sequence(states + [raw], out),
    ]
    csv_path = out / "failure_point_values_step022642.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stage",
                "i",
                "j",
                "phi",
                "node_type",
                "phase_gx_lu",
                "phase_gy_lu",
                "source_gx_lu",
                "source_gy_lu",
                "gradient_status",
                "ux_lu",
                "uy_lu",
                "speed_l1_lu",
            ]
        )
        for stage, points in info["stages"].items():
            for p in points:
                writer.writerow(
                    [
                        stage,
                        p["i"],
                        p["j"],
                        p["phi"],
                        p["node_type"],
                        *p["phase_gradient_lu"],
                        *p["source_gradient_lu"],
                        p["gradient_status"],
                        *p["velocity_lattice"],
                        p["speed_l1_lu"],
                    ]
                )
    info["outputs"] = [str(path.resolve()) for path in paths] + [
        str(csv_path.resolve())
    ]
    (out / "failure_annotations_step022642.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    for stage in ("macros_unbounded", "rejected_before_thermal"):
        point = next(p for p in info["stages"][stage] if (p["i"], p["j"]) == (439, 3))
        print(json.dumps(dict(stage=stage, target=point)), flush=True)
    for path in paths:
        print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
