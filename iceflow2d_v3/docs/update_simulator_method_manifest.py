"""Refresh/check the guide's source hashes and AST method index without CUDA."""

import argparse
import ast
from datetime import date
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DOC = Path(__file__).with_name("simulator_method_zh.md")
MANIFEST = Path(__file__).with_name("simulator_method_source_manifest.json")
SOURCES = (
    "simulator.py",
    "phase_reconstruction.py",
    "config.py",
    "lattice.py",
    "body_state.py",
    "contact.py",
    "stability_options.py",
    "thermal_d2q5.py",
    "viscosity.py",
    "reporting.py",
    "examples/coupled_falling_ice_melting_2d.py",
    "examples/four_falling_ice_melting_2d.py",
    "examples/dambreak_four_ice_melting_2d.py",
    "examples/dambreak_single_ice_melting_2d.py",
)
DESCRIPTIONS = {
    (
        "IceFlow2D",
        "_advance_fluid",
    ): "按配置准备梯度、碰撞、流送、恢复宏观量，并按分支顺序限幅与恢复压力。",
    (
        "IceFlow2D",
        "_refresh_fluid_domain",
    ): "统一提交运动/侵蚀几何，重建新流体；connected_ls 在补点后刷新梯度缓存。",
    (
        "IceFlow2D",
        "_build_hydrostatic_reference",
    ): "冻结满宽池/空气参考，并为局部水柱逐列积分初始超额压力。",
    (
        "IceFlow2D",
        "_initialize_hydrostatic_equilibrium",
    ): "以静止速度初始化分布，局部水柱保留非零初始动态压力。",
    (
        "IceFlow2D",
        "_ice_phase_ghost_legacy",
    ): "保留的标量冰壁镜像插值，未检查供体与中心之间的连通性。",
    (
        "IceFlow2D",
        "_ice_phase_ghost",
    ): "按梯度选项分派标量镜像或 connected_ls 多项式 ghost。",
    (
        "IceFlow2D",
        "_phase_central_gradient",
    ): "读取新分支的源项梯度缓存，或计算 legacy 中心模板。",
    (
        "IceFlow2D",
        "_collide_momentum",
    ): "准备梯度和应力，按选项先碰撞相场并测通量，再碰撞动量。",
    (
        "IceFlow2D",
        "_collide_phase",
    ): "Python 包装：必要时准备梯度，再调用相场碰撞内核。",
    (
        "IceFlow2D",
        "_collide_phase_kernel",
    ): "使用模板或缓存梯度碰撞相场中心矩，并闭合零阶矩。",
    (
        "IceFlow2D",
        "_reduce_wall_contact_ranges",
    ): "从剩余锐界支撑单元归约四面墙的切向接触端点。",
    (
        "IceFlow2D",
        "_wall_contact_impulse",
    ): "按接触点法向速度和转动惯量修正刚体平移与角速度。",
    (
        "IceFlow2D",
        "_project_moving_body_inside_container",
    ): "投影刚体位置并迭代施加含转动的墙接触冲量。",
    (
        "PhaseReconstruction",
        "_initialize_phase_reconstruction",
    ): "仅在 connected_ls 下分配梯度、秩/壁面残差、几何、多项式和双缓冲补点字段。",
    ("PhaseReconstruction", "_phase_donor_allowed"): "解码 5×5 窗口的供体允许位。",
    (
        "PhaseReconstruction",
        "_unclipped_ice_fraction",
    ): "在材料坐标取未经过容器墙裁剪的冰覆盖率。",
    (
        "PhaseReconstruction",
        "_phase_visible_diagonal",
    ): "分段检查对角线上的双线性覆盖率内部最大值，拒绝穿冰通路。",
    (
        "PhaseReconstruction",
        "_phase_ice_geometry",
    ): "估计材料法向和实际阈值交点，不使用动量交点限幅。",
    (
        "PhaseReconstruction",
        "_prepare_phase_geometry",
    ): "准备局部单调连通供体图及八条邻接链的壁面几何。",
    ("PhaseReconstruction", "_phase_fit_basis"): "返回局部二次多项式的五列基函数。",
    (
        "PhaseReconstruction",
        "_phase_minimum_norm_fit",
    ): "双精度列主元 QR、两次正交化和行空间最小范数求解，可返回数据零空间。",
    (
        "PhaseReconstruction",
        "_fit_boundary_phase",
    ): "先拟合连通流体数据及显式曲率闭合，再仅在其零空间内最小化壁面残差。",
    (
        "PhaseReconstruction",
        "_prepare_phase_gradients_kernel",
    ): "内部用原模板，边界用数据优先重构，写入未整体限幅的梯度、秩和壁面残差。",
    (
        "PhaseReconstruction",
        "_prepare_phase_gradients",
    ): "Python 包装：仅 connected_ls 调用梯度准备内核。",
    (
        "PhaseReconstruction",
        "_connected_phase_ghost",
    ): "在指定采样偏移计算并限幅局部多项式值。",
    (
        "PhaseReconstruction",
        "_seed_connected_refill",
    ): "快照持续流体状态并标记待填新节点。",
    (
        "PhaseReconstruction",
        "_propagate_connected_refill",
    ): "只读上轮有效供体，先半径 1 后半径 2 地传播状态。",
    (
        "PhaseReconstruction",
        "_commit_connected_refill",
    ): "统一提交新节点的宏观状态和平衡/可选非平衡分布。",
    (
        "PhaseReconstruction",
        "_refill_connected_nodes",
    ): "调度双缓冲传播与提交，无可达新状态时明确报错。",
}


def methods():
    rows = []
    for filename in ("simulator.py", "phase_reconstruction.py"):
        path = ROOT / "iceflow2d_v3" / filename
        tree = ast.parse(path.read_text())
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates = [(None, top)]
            elif isinstance(top, ast.ClassDef):
                candidates = [
                    (top.name, node)
                    for node in top.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            else:
                continue
            for owner, node in candidates:
                decorators = [ast.unparse(item) for item in node.decorator_list]
                kind = (
                    "kernel"
                    if "ti.kernel" in decorators
                    else "func"
                    if "ti.func" in decorators
                    else "Python"
                )
                rows.append(
                    dict(
                        path=str(path.relative_to(ROOT)),
                        owner=owner,
                        name=node.name,
                        definition_line=node.lineno,
                        decorators=decorators,
                        kind=kind,
                    )
                )
    return rows


def existing_descriptions(body):
    descriptions = {}
    owner = None
    for line in body.splitlines():
        heading = re.match(r"### (\S+) 方法", line)
        if heading:
            owner = heading[1]
        item = re.match(
            r"- \[(.+?)\]\(\.\./(?:simulator|phase_reconstruction)\.py#L\d+\)(?: \[[^]]+\])?：(.*)",
            line,
        )
        if item:
            descriptions[owner, item[1].replace("\\", "")] = item[2]
    return descriptions


def appendix(rows, descriptions):
    lines = [
        "## 附录 A 当前源码的方法索引",
        "",
        "以下行号由 Python AST 从当前源码生成。`kernel` 在 GPU 批量更新字段，`func` 在内核内展开；Python 方法负责调度、包装或主机读取。覆盖 `simulator.py` 的全部顶层函数/类方法，以及继承进 `IceFlow2D` 的 `PhaseReconstruction` 方法。",
        "",
    ]
    last_owner = None
    for row in rows:
        owner, name = row["owner"], row["name"]
        if owner != last_owner:
            lines.extend([f"### {owner} 方法", ""])
            last_owner = owner
        label = name.replace("_", "\\_")
        target = "../" + Path(row["path"]).name
        description = descriptions.get((owner, name))
        if description is None:
            raise ValueError(f"Missing beginner description: {owner}.{name}")
        lines.append(
            f"- [{label}]({target}#L{row['definition_line']}) [{row['kind']}]：{description}"
        )
    return "\n".join(lines) + "\n\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    previous = json.loads(MANIFEST.read_text())
    doc = DOC.read_text()
    # Support both the old title and the generated current title.
    start = doc.index("## 附录 A")
    end = doc.index("## 附录 B")
    descriptions = existing_descriptions(doc[start:end]) | DESCRIPTIONS
    rows = methods()
    sources = [
        dict(
            path="iceflow2d_v3/" + name,
            sha256=hashlib.sha256(
                (ROOT / "iceflow2d_v3" / name).read_bytes()
            ).hexdigest(),
        )
        for name in SOURCES
    ]
    index = appendix(rows, descriptions)
    if args.check:
        assert previous["sources"] == sources, (
            "Source hashes changed; refresh the manifest"
        )
        assert previous["methods"] == rows, (
            "Method definitions changed; refresh the index"
        )
        assert doc[start:end] == index, "Guide method index is stale"
        print(f"Source hashes and {len(rows)} method entries match current code")
        return
    snapshot_date = args.snapshot_date or date.today().isoformat()
    date.fromisoformat(snapshot_date)
    previous.update(
        snapshot_date=snapshot_date,
        sources=sources,
        methods=rows,
        simulator_callable_count=sum(
            row["path"].endswith("/simulator.py") for row in rows
        ),
        reconstruction_callable_count=sum(
            row["owner"] == "PhaseReconstruction" for row in rows
        ),
    )
    updated = doc[:start] + index + doc[end:]
    year, month, day = (int(part) for part in snapshot_date.split("-"))
    updated = re.sub(
        r"代码快照日期：\d+ 年 \d+ 月 \d+ 日。",
        f"代码快照日期：{year} 年 {month} 月 {day} 日。",
        updated,
    )
    DOC.write_text(updated)
    MANIFEST.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
    print(f"Refreshed source hashes and {len(rows)} method entries for {snapshot_date}")


if __name__ == "__main__":
    main()
