"""Render sparse production evidence; no solver or diagnostic recomputation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iceflow2d_v3.benchmarks.render_single_dambreak_step import setup_fonts
from iceflow2d_v3.benchmarks.verify_data_priority_gradient import reconstruct


def main():
    base = Path("iceflow2d_v3/outputs")
    out = base / "four_ice_data_priority_diagnosis"
    trace = json.loads((out / "trace.json").read_text())

    def row(step, stage):
        return next(
            r for r in trace if r["attempted_step"] == step and r["stage"] == stage
        )

    stages = [
        (8520, "body", "8520：开缝前"),
        (8520, "thermal", "8520：侵蚀补点后"),
        (8521, "pressure", "8521：流送/压力后"),
        (8522, "pressure", "8522：流送/压力后"),
        (8523, "macros", "8523：几何更新前"),
        (8523, "body", "8523：几何更新后"),
    ]
    indices = np.arange(149, 157)
    phase = []
    gradient = []
    solid = []
    for step, stage, label in stages:
        r = row(step, stage)
        if "floor_line" in r:
            line = r["floor_line"]
            offset = indices - line["i_start"]
            phase.append(np.asarray(line["phi"])[offset])
            gradient.append(np.asarray(line["source_gradient"])[offset, 0])
            solid.append(np.asarray(line["solid"])[offset])
        else:
            points = {tuple(point["cell"]): point for point in r["points"]}
            phase.append([points[(i, 3)]["phi"] for i in indices])
            gradient.append(
                [points[(i, 3)]["source_gradient_lattice"][0] for i in indices]
            )
            solid.append([points[(i, 3)]["solid"] for i in indices])
    phase = np.array(phase)
    gradient = np.array(gradient)
    solid = np.array(solid) != 0
    setup_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), layout="constrained")
    labels = [v[2] for v in stages]
    for ax, array, title, cmap, vmin, vmax in (
        (axes[0, 0], phase, "地板上方 j=3 的 φ（格内为实际数值）", "Blues", 0, 1),
        (
            axes[0, 1],
            gradient,
            "同位置的实际缓存 ∂φ/∂x（格子单位）",
            "RdBu_r",
            -0.5,
            0.5,
        ),
    ):
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#b3b8be")
        palette.set_under("#e69fd4")
        im = ax.imshow(
            np.ma.masked_where(solid, array),
            cmap=palette,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(indices)), indices)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=title, xlabel="i")
        for rr in range(len(labels)):
            for cc in range(len(indices)):
                text = "冰" if solid[rr, cc] else f"{array[rr, cc]:.2g}"
                color = (
                    "white"
                    if not solid[rr, cc]
                    and (
                        array[rr, cc] > 0.65
                        if ax == axes[0, 0]
                        else abs(array[rr, cc]) > 0.32
                    )
                    else "#111"
                )
                ax.text(cc, rr, text, ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    steps = range(8515, 8533)
    for stage, title, style in (
        ("macros", "宏观量恢复后、几何更新前", "-o"),
        ("body", "几何更新后", "-s"),
    ):
        ys = [row(n, stage)["state"]["speed_l1_max"] for n in steps]
        axes[1, 0].semilogy(list(steps), ys, style, label=title, ms=4)
    axes[1, 0].axhline(16, ls="--", color="crimson", label="现有严重失稳检查阈值 16")
    axes[1, 0].annotate(
        "8523：9.16×10⁴ → 0.120\n异常点被冰体覆盖",
        xy=(8523, 91572.55),
        xytext=(8524.5, 1e4),
        arrowprops={"arrowstyle": "->"},
        fontsize=10,
    )
    axes[1, 0].set(
        title="全域最大 |uₓ|+|uᵧ|：几何更新会掩盖已发散的流体点",
        xlabel="尝试步数",
        ylabel="速度（格子单位，对数坐标）",
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=9, loc="upper left")
    patch = json.loads(
        (
            base / "four_ice_data_priority_opening_controls/smoothing_intervention.json"
        ).read_text()
    )
    extended = json.loads(
        (
            base / "four_ice_data_priority_smoothed_long_control/controls.json"
        ).read_text()
    )[0]
    xs = np.arange(patch["i_range"][0], patch["i_range"][1] + 1)
    axes[1, 1].plot(xs, patch["old_phi"], "o-", label="原始补点后 φ")
    axes[1, 1].plot(
        xs, patch["smoothed_phi"], "s-", label="一次局部平滑，界面宽度 5 格"
    )
    axes[1, 1].set(
        title="匹配状态对照：保持局部 Σφ，保留生产梯度算法",
        xlabel="i，j=3",
        ylabel="φ",
        ylim=(-0.03, 1.05),
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].text(
        0.04,
        0.16,
        f"原状态两步后速度 >1；平滑后继续 {extended['completed_steps']} 步\n后续第 11770 步遇无供体新气穴\n局部 Σφ 差 = {patch['new_sum'] - patch['old_sum']:.2e}\n仅为诊断干预，尚未作为生产补点方案",
        transform=axes[1, 1].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#aaa"},
    )
    fig.suptitle(
        "四冰块 dambreak：新梯度方案下的首次失稳链条\n8520 步半格厚窄缝连通 → 仅一格过渡的密度陡变 → 气相动量/压力放大 → 冰体运动与再次开缝",
        fontsize=15,
    )
    fig.supxlabel(
        "φ < 0 用紫色；灰格为冰。梯度始终取实际缓存：宏观量恢复阶段仍使用碰撞时层，未额外重算。",
        fontsize=10,
    )
    path = out / "root_cause_evidence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    opening = row(8520, "thermal")
    cpu = []
    for point in opening["points"]:
        if point["cell"] not in ([153, 3], [154, 3], [155, 3]):
            continue
        offsets = []
        values = []
        center = point["cell"][0]
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if (di or dj) and point["phase_donor_mask"] >> (
                    (di + 2) * 5 + dj + 2
                ) & 1:
                    assert dj == 0
                    offsets.append([di, dj])
                    values.append(
                        opening["floor_line"]["phi"][center + di - 140] - point["phi"]
                    )
        ref = reconstruct(offsets, values, point["phase_wall_geometry"])
        cpu.append(
            {
                "cell": point["cell"],
                "gpu_gradient": point["source_gradient_lattice"],
                "independent_svd_gradient": ref["gradient"],
                "difference_norm": float(
                    np.linalg.norm(
                        np.array(ref["gradient"]) - point["source_gradient_lattice"]
                    )
                ),
                "boundary_residual": point["phase_gradient_boundary_residual"],
                "rho": point["rho"],
                "phi": point["phi"],
            }
        )
    timeline = []
    for n in range(8520, 8533):
        for stage in ("thermal", "macros", "body"):
            try:
                r = row(n, stage)
            except StopIteration:
                continue
            timeline.append(
                {
                    "attempted_step": n,
                    "stage": stage,
                    "max_velocity_l1_lu": r["state"]["speed_l1_max"],
                    "peak_cell": r["state"]["peak_cell"],
                }
            )
    summary = {
        "production_failure": json.loads(
            (base / "dambreak_four_ice_data_priority_8s/run_failure.json").read_text()
        ),
        "opening_step": 8520,
        "new_cell": [154, 3],
        "opening_phi": 0.5000535845756531,
        "neighbors_phi": [1.0, 0.0001072277664206922],
        "aperture_cells": 0.5001373164122924,
        "gradient_comparison": cpu,
        "timeline": timeline,
        "controls": [
            {k: v for k, v in r.items() if k != "history"}
            for r in json.loads(
                (
                    base / "four_ice_data_priority_opening_controls/controls.json"
                ).read_text()
            )
        ],
        "extended_smoothed_control": {
            key: value for key, value in extended.items() if key != "history"
        },
        "normal_gradient_limitation": "A single fluid row leaves normal curvature unobserved. GPU and independent SVD match the same declared closure; nonzero wall residuals remain. The comparison does not establish a fully resolved physical normal derivative in a half-cell aperture.",
        "normal_gradient_controls": json.loads(
            (base / "four_ice_data_priority_normal_control/controls.json").read_text()
        ),
        "root_cause": "A topology change exposes a half-cell-thick channel joining nearly pure water and air. Local refill creates a one-cell transition. The correct resolved gradient does not make the existing explicit phase/density-source/pressure/boundary operators stable for that underresolved transition. Subsequent geometry masks hide an earlier blow-up and let corrupted rigid impulses seed the reported later failure.",
        "solutions": [
            "Conservatively reconstruct the connected newly opened region and adjacent persistent fluid together, retaining the configured diffuse-interface width and compatible phase/momentum/pressure moments.",
            "Use the same bounded shared density flux in phase and momentum transport; recover source/pressure consistently with that flux.",
            "Represent subcell channels and isolated new pockets with conservative aperture/volume remapping or local refinement; do not borrow fluid state through ice.",
            "Check velocity/finite moments before rigid integration and mask changes as well as after geometry; this is detection, not the numerical fix.",
        ],
        "diagnostic_images": [str(path.resolve())],
        "raw_fields_saved": False,
    }
    (out / "evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
