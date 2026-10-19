"""Render the scalar/sparse evidence; never load or save a full field dump."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    trace = json.loads((out / "sparse_gap_trace.json").read_text())
    controls = json.loads((out / "refill_gradient_controls.json").read_text())
    records = trace["records"]
    opening = next(r for r in records if r["step"] == 8276 and r["stage"] == "body")
    patch = {tuple(p["cell"]): p for p in opening["points"]}
    collision = next(
        r for r in records if r["step"] == 8277 and r["stage"] == "collision"
    )
    p7 = next(p for p in collision["points"] if p["cell"] == [176, 7])
    wall_rows = []
    for nx, ny, x, y in p7["phase_wall_geometry"]:
        if nx * nx + ny * ny > 0.5:
            wall_rows.append([nx, ny, nx * x, nx * y + ny * x, ny * y])
    singular = np.linalg.svd(wall_rows, compute_uv=False)
    controls_compact = []
    for control in controls:
        row = {k: v for k, v in control.items() if k != "history"}
        row["peak_sampled_speed_l1"] = max(
            h["speed_l1_max"] for h in control["history"]
        )
        row["last_sampled_state"] = control["history"][-1]
        controls_compact.append(row)
    root = Path(__file__).resolve().parents[2]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "iceflow2d_v3/simulator.py",
            "iceflow2d_v3/phase_reconstruction.py",
            "iceflow2d_v3/stability_options.py",
            "iceflow2d_v3/config.py",
        )
    }
    evidence = dict(
        failure=trace["failure"],
        opening_step=8276,
        wet_refill_cell=[176, 6],
        dry_refill_cell=[176, 7],
        wet_state={
            key: patch[176, 6][key]
            for key in ("water_phase", "rho", "dynamic_pressure", "physical_velocity")
        },
        dry_state={
            key: patch[176, 7][key]
            for key in ("water_phase", "rho", "dynamic_pressure", "physical_velocity")
        },
        wall_constraint_rows=wall_rows,
        wall_constraint_singular_values=singular.tolist(),
        constraint_rank=int(np.linalg.matrix_rank(wall_rows, tol=1e-6)),
        polynomial_unknowns=5,
        connected_donors_including_center=int(p7["phase_donor_mask"]).bit_count(),
        actual_gradient=p7["source_gradient_lattice"],
        first_phase_failure={
            "step": 8278,
            "source_cell": [176, 7],
            "destination_cell": [176, 8],
            "direction": 2,
            "outgoing_phase_population": next(
                point["phase_stream_buffer"][2]
                for record in records
                if record["step"] == 8278 and record["stage"] == "collision"
                for point in record["points"]
                if point["cell"] == [176, 7]
            ),
            "destination_phi_before_bounding": next(
                point["water_phase"]
                for record in records
                if record["step"] == 8278 and record["stage"] == "macros"
                for point in record["points"]
                if point["cell"] == [176, 8]
            ),
        },
        controls=controls_compact,
        causal_conclusion="New-node refill splits a single connected gap into wet/dry plateaus; over-constrained Neumann reconstruction suppresses resolved tangential gradients. Their interaction destabilizes phase and momentum transport in low-density air.",
        limitations=[
            "Smooth refill is an intentionally nonconservative local diagnostic, not a production fix.",
            "The tangent override ignores small wall inclinations and is hard-coded to this gap; it is not a production boundary condition.",
            "600 steps are only 0.005 s; neither local success nor finite gradients establishes an 8 s solution.",
            "Discrete link-flux consistency and whole-cell geometry/refill conservation remain independent required checks.",
        ],
        source_hashes=source_hashes,
        production_solver_modified_this_diagnosis=False,
        raw_grid_fields_saved=False,
    )
    (out / "diagnosis_summary.json").write_text(json.dumps(evidence, indent=2) + "\n")

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "font.size": 11,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    cmap = plt.get_cmap("viridis")
    for (i, j), point in patch.items():
        if j > 14:
            continue
        body = point["solid_body_id"]
        color = (
            "#8c9bae"
            if body == 0
            else "#dbe6f2"
            if body == 1
            else cmap(np.clip(point["water_phase"] / 0.3, 0, 1))
        )
        ax.add_patch(
            Rectangle(
                (i - 0.5, j - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
            )
        )
        if body < 0:
            value = point["water_phase"]
            label = f"{value:.3f}" if value > 0.001 else "0.00014"
            ax.text(
                i,
                j,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 0.15 else "black",
            )
    ax.add_patch(Rectangle((173.5, 1.5), 4, 1, facecolor="#404040"))
    ax.add_patch(
        Rectangle((175.5, 3.5), 1, 6, fill=False, edgecolor="#e64b35", linewidth=2.2)
    )
    ax.annotate(
        "新暴露列\n红框：第8276步",
        xy=(176.45, 8),
        xytext=(177.9, 9),
        arrowprops=dict(arrowstyle="->", color="#e64b35"),
        color="#b12b20",
        fontsize=10,
    )
    ax.annotate(
        "",
        xy=(176.37, 7.2),
        xytext=(176.37, 5.8),
        arrowprops=dict(arrowstyle="->", color="#e64b35", lw=2),
    )
    ax.text(174, 11.8, "第一块冰", ha="center", color="#2b3b51")
    ax.text(177, 11.8, "第二块冰", ha="center", color="#2b3b51")
    ax.set(
        xlim=(173.5, 179.3),
        ylim=(1.5, 14.7),
        xlabel="格点 i",
        ylabel="格点 j",
        title="A  开缝后的补点状态（标注为相场 φ）",
    )
    ax.set_xticks([174, 175, 176, 177])
    ax.legend(
        handles=[
            Patch(color=cmap(0), label="空气段"),
            Patch(color=cmap(0.98), label="湿润段"),
            Patch(color="#404040", label="地板"),
        ],
        loc="lower right",
        fontsize=9,
    )

    ax = axes[0, 1]
    js = np.arange(3, 18)
    phi = [patch[176, int(j)]["water_phase"] for j in js]
    pressure = [patch[176, int(j)]["dynamic_pressure"] * 1000 for j in js]
    a = ax.plot(phi, js, "o-", color="#2878b5", label="φ")
    ax.set(
        xlabel="相场 φ",
        ylabel="格点 j",
        title="B  同一连通气隙中的突变",
        xlim=(-0.01, 0.34),
        ylim=(2.5, 17.5),
    )
    twin = ax.twiny()
    b = twin.plot(pressure, js, "s--", color="#c64335", label="动态压力")
    twin.set_xlabel(r"动态压力（$\times 10^{-3}$ LU）", color="#c64335")
    ax.axhline(6.5, color="#555555", lw=0.8, ls=":")
    ax.text(
        0.11,
        10.5,
        "6条壁面硬约束，秩为5\n5个重建未知数全部被约束\n尽管有4个供体，∇φ仍为0",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="#cccccc"),
    )
    ax.legend(
        a + b, [line.get_label() for line in a + b], loc="lower right", fontsize=9
    )

    ax = axes[1, 0]
    macros = [r for r in records if r["stage"] == "macros"]
    steps = [r["step"] for r in macros]
    ax.semilogy(
        steps,
        [r["state"]["speed_l1_max"] for r in macros],
        "o-",
        color="#171717",
        label="全域最大值",
    )
    for cell, color in (
        ([176, 7], "#549f3f"),
        ([176, 8], "#e58f23"),
        ([176, 11], "#8052a8"),
    ):
        speed = []
        for r in macros:
            point = next(p for p in r["points"] if p["cell"] == cell)
            speed.append(max(1e-5, sum(abs(v) for v in point["physical_velocity"])))
        ax.semilogy(steps, speed, ".-", color=color, label=str(tuple(cell)))
    ax.axvline(8276, ls="--", color="#999999", lw=1)
    ax.set(
        xlabel="迁移后的步数",
        ylabel=r"$|u_x| + |u_y|$（LU）",
        title="C  原始求解器精确复现：第8287步爆发",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    ax = axes[1, 1]
    labels = {
        "baseline": "原算法",
        "phase_smooth": "仅平滑补点",
        "tangent_gradient": "仅恢复切向梯度",
        "smooth_and_tangent": "同时处理两项",
    }
    colors = {
        "baseline": "#171717",
        "phase_smooth": "#e58f23",
        "tangent_gradient": "#c64335",
        "smooth_and_tangent": "#2878b5",
    }
    for control in controls:
        mode = control["mode"]
        if mode in labels:
            ax.plot(
                [h["step"] - 8276 for h in control["history"]],
                [h["speed_l1_max"] for h in control["history"]],
                "o-" if mode != "smooth_and_tangent" else "-",
                markersize=3,
                color=colors[mode],
                label=labels[mode],
            )
    ax.set_xscale("symlog", linthresh=20)
    ax.set(
        xlabel="从第8276步继续的步数",
        ylabel=r"全域最大 $|u_x| + |u_y|$（LU）",
        title="D  同状态诊断对照（600步 = 0.005秒）",
        ylim=(0, 1.4),
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.suptitle("当前失稳：补点不连续与过度约束的梯度闭合共同触发", fontsize=17)
    fig.savefig(out / "failure_mechanism.png")
    plt.close(fig)
    print(
        json.dumps(
            dict(
                figure=str((out / "failure_mechanism.png").resolve()),
                summary=str((out / "diagnosis_summary.json").resolve()),
            )
        )
    )


if __name__ == "__main__":
    main()
