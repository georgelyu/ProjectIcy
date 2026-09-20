"""Generate reproducible static figures and a Chinese verification report."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
import taichi as ti

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#293547", "text.color": "#17283d",
    "axes.edgecolor": "#8b9aad", "grid.color": "#dde3e9",
    "grid.alpha": 0.7, "figure.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
    "svg.fonttype": "none",
})
COLORS = ["#1468a0", "#dc8936", "#22978e", "#9b62a5"]


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


NAMES = ["moments", "uniform", "diffusion", "manufactured", "layers",
         "advection_diffusion", "stefan_one_phase", "stefan_two_phase",
         "translation", "rotation", "closed", "translating_melt"]
DATA = {name: read(name) for name in NAMES}
if not all(value["passes"] for value in DATA.values()):
    raise RuntimeError("A verification check did not pass; do not issue a success report.")


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def overview():
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6), layout="constrained")
    fig.suptitle("D2Q5 solid-liquid heat model | independent verification", fontsize=19, weight="bold")
    ax = axes[0, 0]
    for name, label, key, color in [
        ("diffusion", "Diffusion", "l2_error", COLORS[0]),
        ("manufactured", "Variable C and k", "temperature_l2_error", COLORS[1]),
        ("advection_diffusion", "Advection + diffusion", "temperature_l2_error", COLORS[2]),
    ]:
        rows = DATA[name]["rows"]
        ax.loglog([r["n"] for r in rows], [r[key] for r in rows], "o-", color=color, label=label)
    nn = np.array([64, 128, 256])
    ax.loglog(nn, 1.4e-5*(64/nn)**2, "--", color="#7d8590", label="Second-order guide")
    ax.set(title="A  Smooth-field convergence", xlabel="Cells per side", ylabel="Temperature RMS error")
    ax.set_xticks(nn, nn)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", linewidth=0.6)

    ax = axes[0, 1]
    archive = np.load(RESULTS / "stefan_two_phase.npz")
    rows = DATA["stefan_two_phase"]["rows"]
    nmax = rows[-1]["n"]
    ax.plot(archive[f"times_{nmax}"], archive[f"front_exact_{nmax}"], "k-", lw=2, label="Stefan exact")
    for n, color in zip([rows[0]["n"], nmax], [COLORS[1], COLORS[0]]):
        ax.plot(archive[f"times_{n}"], archive[f"front_{n}"], "o", color=color,
                markersize=3, markevery=2, label=f"D2Q5, N={n}")
    ax.set(title="B  Two-phase melting front", xlabel="Dimensionless time", ylabel="Melted length / domain length")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.legend(fontsize=8.5)
    ax.grid(True, linewidth=0.6)

    ax = axes[0, 2]
    for name, label, color in [("stefan_one_phase", "Ice initially at melting point", COLORS[0]),
                                ("stefan_two_phase", "Subcooled ice", COLORS[1])]:
        rows = DATA[name]["rows"]
        ax.loglog([r["n"] for r in rows], [100*r["relative_front_error"] for r in rows],
                  "o-", color=color, label=label)
    nn = np.array([r["n"] for r in DATA["stefan_two_phase"]["rows"]])
    ax.set_xticks(nn, nn)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(title="C  Stefan front error", xlabel="Cells along x", ylabel="Relative error (%)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.6)

    ax = axes[1, 0]
    history = np.load(RESULTS / "uniform_heating.npz")["history"]
    ax.axvspan(0, 2, color="#e7eef7", label="Latent-heat interval")
    ax.plot(history[:, 3], history[:, 4], "k--", lw=1.3, label="Exact temperature")
    ax.plot(history[:, 3], history[:, 1], "o", color=COLORS[0], markevery=4, ms=3.5, label="D2Q5 temperature")
    ax.plot(history[:, 3], history[:, 2], "-", color=COLORS[2], lw=1.7, label="Liquid fraction")
    ax.set(title="D  Heating through the melting plateau", xlabel="Enthalpy", ylabel="Temperature / liquid fraction")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linewidth=0.6)

    ax = axes[1, 1]
    archive = np.load(RESULTS / "closed_melting.npz")
    h = archive["history"]
    ax.plot(np.r_[0, h[:, 0] * 0.1 / 80**2], np.r_[0.25, h[:, 1] / 80**2],
            color=COLORS[0], lw=2)
    ax.axhline(720 / 6400, color=COLORS[1], ls="--", label="Energy-predicted final ice")
    ax.set(title="E  Immersed ice in an insulated box", xlabel="Fourier time", ylabel="Solid area / box area")
    ax.text(0.97, 0.94, f'Energy drift: {DATA["closed"]["relative_energy_error"]:.1e}',
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(True, linewidth=0.6)

    ax = axes[1, 2]
    for name, label, color in [("translation", "Translation", COLORS[0]), ("rotation", "Rotation", COLORS[1])]:
        rows = DATA[name]["rows"]
        ax.loglog([r["n"] for r in rows], [100*r["shape_l1_per_solid_volume"] for r in rows],
                  "o-", color=color, label=label)
    ax.set_xticks([64, 128, 256], [64, 128, 256])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(title="F  Moving sharp-interface shape error", xlabel="Cells per side", ylabel="Normalized shape L1 error (%)")
    ax.text(0.97, 0.94, "Prescribed motion; no force solver\nIce-volume error < 1e-7",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, which="both", linewidth=0.6)
    save_figure(fig, "verification_overview")


def stefan_profiles():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), layout="constrained")
    for ax, name, title in zip(axes, ["stefan_one_phase", "stefan_two_phase"],
                               ["One-phase Stefan problem", "Two-phase Stefan problem"]):
        archive = np.load(RESULTS/f"{name}.npz")
        rows = DATA[name]["rows"]
        nmax = rows[-1]["n"]
        ax.plot(archive[f"x_{nmax}"], archive[f"T_exact_{nmax}"], "k-", lw=2, label="Exact")
        for row, color in zip(rows, COLORS):
            n = row["n"]
            ax.plot(archive[f"x_{n}"], archive[f"T_{n}"], "o", ms=3,
                    markevery=max(1, n//64), color=color, label=f"D2Q5 N={n}")
        ax.axvline(rows[-1]["exact_front"], color="#8b9aad", lw=1, ls=":")
        ax.set(title=title, xlabel="x / domain length", ylabel="Temperature relative to melting point",
                xlim=(0, 0.42), ylim=(-0.29, 1.03))
        ax.legend(fontsize=8.5)
        ax.grid(True, linewidth=0.6)
    save_figure(fig, "stefan_profiles")


def closed_snapshots():
    a = np.load(RESULTS/"closed_melting.npz")
    fields = [
        (a["initial_temperature"], a["initial_solid"], "Initial"),
        (a["T_1000"], a["solid_1000"], "Step 1,000"),
        (a["T_5000"], a["solid_5000"], "Step 5,000"),
        (a["T_15000"], a["solid_15000"], "Step 15,000"),
        (a["final_temperature"], a["final_solid"], "Near equilibrium"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13.6, 6.0), layout="constrained")
    fig.suptitle("Immersed ice melting | insulated enclosure | no imposed flow", fontsize=17, weight="bold")
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1)
    x = (np.arange(80)+0.5)/80
    for i, (temp, solid, title) in enumerate(fields):
        im0 = axes[0, i].imshow(temp.T, origin="lower", extent=(0,1,0,1), cmap="coolwarm", norm=norm)
        axes[0, i].contour(x, x, solid.T.astype(float), levels=[0.5], colors="#24394b", linewidths=1)
        axes[0, i].set_title(title, fontsize=10.5)
        im1 = axes[1, i].imshow(solid.T, origin="lower", extent=(0,1,0,1), cmap="Blues", vmin=0, vmax=1)
        axes[1, i].set_xlabel("x / L")
        for ax in axes[:, i]:
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    axes[0,0].set_ylabel("y / L")
    axes[1,0].set_ylabel("y / L")
    fig.colorbar(im0, ax=axes[0,:], shrink=0.85, label="Temperature")
    fig.colorbar(im1, ax=axes[1,:], shrink=0.85, label="Solid fraction")
    save_figure(fig, "closed_melting")


def motion_shapes():
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.1), layout="constrained")
    fig.suptitle("Rigid-motion thermal transport | all material stays at the melting point",
                 fontsize=15, weight="bold")
    for row, name, label in [(0,"translation","One-period translation"),(1,"rotation","90-degree rotation")]:
        archive=np.load(RESULTS/f"{name}.npz")
        n=DATA[name]["rows"][-1]["n"]
        solid,exact=archive[f"solid_{n}"],archive[f"exact_{n}"]
        x=(np.arange(n)+0.5)/n-0.5
        for col, field in enumerate((exact,solid,solid-exact)):
            im=axes[row,col].imshow(field.T,origin="lower",extent=(-.5,.5,-.5,.5),
                                     cmap="Blues" if col<2 else "RdBu_r",vmin=0 if col<2 else -1,vmax=1)
            if col==1:
                axes[row,col].contour(x,x,exact.T,levels=[0.5],colors=COLORS[1],linewidths=1)
            axes[row,col].set_xticks([-.5,0,.5])
            axes[row,col].set_yticks([-.5,0,.5])
            axes[row,col].set_xlabel("x / L")
        axes[row,0].set_ylabel(f"{label}\ny / L")
        axes[row,0].set_title("Exact transformed ice")
        axes[row,1].set_title(f"D2Q5 + transport, N={n}")
        axes[row,2].set_title("Solid-fraction difference")
        fig.colorbar(im,ax=axes[row,2],shrink=.78,label="Difference")
    save_figure(fig,"motion_transport")


def moving_melt():
    a=np.load(RESULTS/"translating_melt.npz")
    n=DATA["translating_melt"]["rows"][-1]["n"]
    target=a[f"target_temperature_{n}"]
    temp=a[f"temperature_{n}"]
    solid=a[f"solid_{n}"]
    target_solid=a[f"target_solid_{n}"]
    x=(np.arange(n)+.5)/n
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),layout="constrained")
    fig.suptitle("Melting while translating | prescribed comoving ice and water velocity",fontsize=15,weight="bold")
    for ax,field,ice,title in zip(axes[:2],[target,temp],[target_solid,solid],
                                  ["Shifted stationary reference","Moving calculation"]):
        im=ax.imshow(field.T,origin="lower",extent=(0,1,0,1),cmap="coolwarm",
                      norm=TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=1))
        ax.contour(x,x,ice.T,levels=[.5],colors="#25374d",linewidths=1)
        ax.set(title=title,xlabel="x / L",ylabel="y / L")
    fig.colorbar(im,ax=axes[:2],shrink=.8,label="Temperature")
    lim=float(np.max(abs(temp-target)))
    im=axes[2].imshow((temp-target).T,origin="lower",extent=(0,1,0,1),cmap="RdBu_r",vmin=-lim,vmax=lim)
    axes[2].set(title="Temperature difference",xlabel="x / L",ylabel="y / L")
    fig.colorbar(im,ax=axes[2],shrink=.8,label="Difference")
    save_figure(fig,"translating_melt")


def table(headers, rows):
    out=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]
    out.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    return "\n".join(out)


def write_csvs():
    for name,value in DATA.items():
        if "rows" in value:
            keys=list(dict.fromkeys(k for row in value["rows"] for k in row))
            with (RESULTS/f"{name}.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=keys)
                writer.writeheader()
                writer.writerows(value["rows"])


def metadata():
    source=Path("/home/georgelyu/下载/Wang_PRE2022_ThermallatticeBoltzmannmodelforliquid-vaporphasechange.pdf")
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"],text=True).strip()
    return {
        "generated_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python_executable":sys.executable,"python_version":platform.python_version(),
        "conda_environment":sys.prefix,"numpy_version":np.__version__,
        "taichi_version":".".join(map(str,ti.__version__)),"matplotlib_version":matplotlib.__version__,
        "gpu":gpu,"gpu_fields":"Taichi f32","host_reference_and_diagnostics":"float64",
        "fast_math":False,"source_pdf":str(source),
        "source_pdf_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_hashes":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                         for name in ["thermal.py","verify.py","make_report.py","derivation_zh.md"]},
        "inputs":"User-supplied PDF and newly derived analytic equations only; no existing repository code.",
        "scope":"Equal-density thermal model and prescribed motion; no force-coupled flow/free-surface solver.",
    }


def build_text(meta):
    d=DATA
    diff,manufactured,adv=d["diffusion"]["rows"],d["manufactured"]["rows"],d["advection_diffusion"]["rows"]
    one,two=d["stefan_one_phase"]["rows"],d["stefan_two_phase"]["rows"]
    trans,rot=d["translation"]["rows"],d["rotation"]["rows"]
    closed=d["closed"]
    moving=d["translating_melt"]["rows"]
    layer=d["layers"]["rows"][-1]
    summary_rows=[
        ["矩条件","CPU 代数检验",f'{d["moments"]["maximum_moment_residual"]:.3e}',"通过"],
        ["均匀加热 / 冷却","8 × 8",f'温度绝对误差 {d["uniform"]["max_temperature_error"]:.3e}',"通过"],
        ["二维扩散",f'{diff[-1]["n"]}²',f'L2={diff[-1]["l2_error"]:.3e}；阶数 {diff[-1]["order"]:.3f}',"通过"],
        ["变热容 / 变导热率瞬态",f'{manufactured[-1]["n"]}²',f'L2={manufactured[-1]["temperature_l2_error"]:.3e}；阶数 {manufactured[-1]["order"]:.3f}',"通过"],
        ["跨材料稳态热流",f'{layer["n"]} × 4',f'左侧热流相对误差 {layer["left_flux_relative_error"]:.3e}',"通过"],
        ["对流–扩散",f'{adv[-1]["n"]}²',f'L2={adv[-1]["temperature_l2_error"]:.3e}；阶数 {adv[-1]["order"]:.3f}',"通过"],
        ["一相 Stefan",f'{one[-1]["n"]} × 4',f'前沿误差 {one[-1]["relative_front_error"]:.3%}',"通过"],
        ["两相 Stefan",f'{two[-1]["n"]} × 4',f'前沿误差 {two[-1]["relative_front_error"]:.3%}',"通过"],
        ["平移输运",f'{trans[-1]["n"]}²',f'冰量相对变化 {trans[-1]["relative_solid_volume_error"]:.3e}',"通过，几何有扩散"],
        ["转动输运",f'{rot[-1]["n"]}²',f'冰量相对变化 {rot[-1]["relative_solid_volume_error"]:.3e}',"通过，几何有扩散"],
        ["封闭容器浸没融化","80²",f'平衡冰量误差 {closed["relative_final_volume_error"]:.3%}',"通过"],
        ["平移中融化",f'{moving[-1]["n"]}²',f'与静止对照的冰量差 {moving[-1]["relative_solid_volume_difference"]:.3%}',"通过坐标变换检验"],
    ]
    lines=[
        "# D2Q5 固液相变模型：数值验证记录",
        "",
        "本报告由 make_report.py 从实际运行的 JSON / NPZ 结果生成。方法和完整推导见 "
        f"[derivation_zh.md]({ROOT}/derivation_zh.md)。所有下述组别的预设检查均通过。",
        "",
        "**结论的范围：** 已验证的是等密度冰水热模型、给定速度的焓输运，以及完全浸没冰块的导热融化。"
        "没有求解受力反馈决定的自由上浮 / 漂移 / 转动，也没有实现水–空气自由液面。"
        "这些耦合的方程与必须补充的验证已在推导报告列明。",
        "",
        "## 运行环境与输入",
        "",
        f'- Python：{meta["python_executable"]}，版本 {meta["python_version"]}（conda icy）。',
        f'- GPU：{meta["gpu"]}；Taichi {meta["taichi_version"]}，CUDA。',
        '- 全部 GPU 状态和计算使用 f32，fast_math=False；参考解和 CPU 汇总使用 float64。',
        '- 输入仅为用户提供的 Wang 等（2022）PDF。没有读取或导入仓库已有代码。',
        '- 物性为明确给定的无量纲测试参数，不是从该论文提取出的真实冰水数据。',
        '- 源文件哈希、PDF SHA-256 和工具版本见 results/provenance.json。',
        "",
        "## 总览",
        "",
        table(["检验","网格","主要结果","判定"],summary_rows),
        "",
        f"![验证总览]({FIGURES}/verification_overview.png)",
        "",
        "“通过”是通过已声明工况和阈值，不能解释为任意参数下的稳定性证明或实验验证。",
        "",
        "## 光滑解与变物性",
        "",
        "温度 L2 使用网格点上的均方根误差，阶数为相邻网格误差比的对数；"
        "时间步按扩散标度随网格加密。扩散系数单独用 Fourier 模态衰减率反演。",
        "",
        table(["N","纯扩散 L2","变物性制造解 L2","对流–扩散 L2"],[
            [a["n"],f'{a["l2_error"]:.6e}',f'{b["temperature_l2_error"]:.6e}',f'{c["temperature_l2_error"]:.6e}']
            for a,b,c in zip(diff,manufactured,adv)]),
        "",
        f'最细网格的热扩散率相对误差为 {diff[-1]["relative_diffusivity_error"]:.6e}。'
        "变物性制造解同时具有空间变化的热容、导热率和非零时间导数，能检测把导热率误用成热扩散率的错误。",
        "",
        f'两层稳态材料取 C=0.5/1、k=0.2/0.05；最细网格温度最大绝对误差为 '
        f'{layer["max_temperature_error"]:.6e}。该稳态解用解析温度初始化，检查的是长时间保持和热流匹配，'
        "不把它当成独立的瞬态收敛实验。其误差已接近 f32 累计舍入量级，不要求加密后继续单调下降。",
        "",
        "## Stefan 相界面和边界热量",
        "",
        "初始时间 t0=0.002，终止时间约为 0.02；初始化对界面单元的显热和潜热分段积分。"
        "右端使用解析温度而不是不相容的有限域远场近似。前沿用液相体积积分定位。",
        "",
    ]
    for label,rows in [("一相：初始冰在熔点",one),("两相：初始冰低于熔点",two)]:
        lines.extend([
            f"### {label}",
            "",
            table(["N","前沿相对误差","温度 L2","热输入对解析值误差","离散能量余额 / 输入热量"],[
                [r["n"],f'{r["relative_front_error"]:.5%}',f'{r["temperature_l2_error"]:.6e}',
                 f'{r["boundary_heat_relative_error_vs_analytic"]:.5%}',f'{r["energy_balance_relative"]:.6e}']
                for r in rows]),
            "",
        ])
    lines.extend([
        "界面问题不显示统一的二阶收敛。相界面落在单元内的位置会让某些单时刻温度误差非单调；"
        "整体前沿、热输入误差随网格加密降低，不能只挑一个较小误差的网格报告。",
        "",
        "一相测试在个别网格存在约 10⁻⁶ 量级的温度下冲；原始 JSON 保留最小、最大温度。"
        "没有通过强行裁剪温度掩盖下冲，整体能量仍按通量更新。",
        "",
        f"![Stefan 温度剖面]({FIGURES}/stefan_profiles.png)",
        "",
        "## 运动：能量正确与几何精确是不同指标",
        "",
        "以下试验全域均在熔点，连续解没有导热或相变，只发生给定的刚体输运。"
        "平移一个周期和转动 90° 后，理论温度仍为零，冰的总量不变。",
        "",
        table(["工况","N","冰量相对变化","形状 L1 / 初始冰量","几何惯量相对误差","质心误差 / 域长"],[
            [name,r["n"],f'{r["relative_solid_volume_error"]:.3e}',
             f'{r["shape_l1_per_solid_volume"]:.3%}',f'{r["relative_geometric_inertia_error"]:.3%}',
             f'{r["centroid_error_over_domain"]:.3e}']
            for name,rows in [("平移",trans),("转动",rot)] for r in rows]),
        "",
        f'最细网格平移 / 转动的形状误差仍分别为 {trans[-1]["shape_l1_per_solid_volume"]:.2%}、'
        f'{rot[-1]["shape_l1_per_solid_volume"]:.2%}。这是保守 TVD 输运对锐界面的数值扩散，'
        "并非冰量损失或额外潜热。"
        "因此当前输运可作为热模型的验证实现，但不能当成长期严格保持刚体边界的最终几何算法；"
        "后续受力耦合宜使用几何重映射或经过验证的相界面输运，并保持焓和质量通量一致。",
        "",
        f"![运动输运的几何比较]({FIGURES}/motion_transport.png)",
        "",
        "## 完全浸没的冰块融化",
        "",
        f'80² 绝热容器内，初始冰面积为 {closed["initial_solid_volume"]:.0f}，'
        f'总能量预测的平衡剩余冰面积为 {closed["predicted_final_solid_volume"]:.0f}。'
        f'计算 {closed["steps"]} 步后剩余 {closed["final_solid_volume"]:.6f}，'
        f'相对差 {closed["relative_final_volume_error"]:.5%}；'
        f'最大残余温差 {closed["max_final_temperature"]:.6e}，能量相对漂移 {closed["relative_energy_error"]:.6e}。',
        "",
        "这里验证的是最终冰量与能量余额，未声称存在这一二维融化形状的闭式解析解。"
        "终止时尚有小的残余温差，因此未将有限时间剩余冰量强行设为理论平衡值。",
        "",
        f"![浸没冰块融化过程]({FIGURES}/closed_melting.png)",
        "",
        "## 平移与融化同时发生",
        "",
        "给冰块及周围水相同的恒定平移速度；周期域中连续解应等于静止解的平移。"
        "这是热模型的 Galilean 一致性检查，不包含相对水流、尾涡、自然对流或自由上浮动力学。",
        "",
        table(["N","冰量相对差","温度 L2 差","液相率 L1 差","能量相对漂移"],[
            [r["n"],f'{r["relative_solid_volume_difference"]:.5%}',f'{r["temperature_l2_error"]:.6e}',
             f'{r["fraction_l1_error"]:.6e}',f'{r["relative_energy_error"]:.6e}'] for r in moving]),
        "",
        f"![平移融化对照]({FIGURES}/translating_melt.png)",
        "",
        "## 复现",
        "",
        "~~~bash",
        f"conda run --no-capture-output -n icy python {ROOT}/verify.py --suite all --grids 64 128 256",
        f"conda run --no-capture-output -n icy python {ROOT}/verify.py --suite stefan --grids 64 128 256 512",
        f"conda run -n icy python {ROOT}/make_report.py",
        "~~~",
        "",
        "第一条运行全部 12 个组别；第二条为 Stefan 锐界面补充 512 网格并更新相应结果。"
        "测试程序不读取本目录外的仓库文件。第三条从结果生成图和本报告，并记录用户提供 PDF 的哈希。",
        "",
        "原始数值同时以 JSON、CSV 和 NPZ 保存。图有 PNG 和可编辑 SVG 两种格式。",
        "",
    ])
    (ROOT/"validation_zh.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    overview()
    stefan_profiles()
    closed_snapshots()
    motion_shapes()
    moving_melt()
    write_csvs()
    meta=metadata()
    (RESULTS/"provenance.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    (RESULTS/"validation_summary.json").write_text(json.dumps({"metadata":meta,"tests":DATA},indent=2,ensure_ascii=False)+"\n")
    build_text(meta)
    print("Wrote five verified-data figures (PNG/SVG), CSV tables, provenance, and validation_zh.md.")


if __name__=="__main__":
    main()
