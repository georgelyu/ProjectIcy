# W=5 四冰块水量收支核查（2026-09-22）

**结果：W=5 实际跑完 5 s、600,000 步，四块冰的材料质量与固体格点数均归零。液水净增加 3218.447499 格，比 3600 格少 381.552501 格。逐算子账目闭合，最大采样闭合误差 1.080e-10 格。**

原方案表中的 **3338.060569 格来自旧 W=12 算例**。以下数字来自本次 W=5 新仿真，不能直接代替旧 W=12 的 261.939431 格差额。本次没有启用简化水量校正，也没有改变生产求解器。

## 1. 为什么 3600 的推算只在特定条件下成立

若四块 30×30 冰保持不动，周围始终是 phi=1 的纯水，完全融化后每个暴露格点回填 1，增加量就是 4×30×30=3600 格。

本次另用同样 450×150 网格、W=5 和四块 30×30 冰做规定融化的隔离对照：关闭预热，不推进流场或刚体，给足规定热量，先调用材料相变，再刷新融化后的几何。

| 对照 | 仅材料相变增加水量 | 随后几何刷新增加水量 / 格 |
|---|---:|---:|
| 四块冰固定在纯水中 | 0 | 3599.999997139 |
| 四块冰固定在空气中 | 0 | 0.000000000 |

水中结果与 3600 的差仅为 f32 邻域平均的舍入。这个对照直接验证“水下回填近似等体积”的推算，但不是自然热融化模拟。

## 2. W=5 下落融化的完整水量账目

主口径是 **所有非墙、非冰节点的 phi 之和**，不应用 `water_phase_cutoff`。所有数值以 `dx²` 为体积单位；二维物理量是单位面外厚度对应的面积。

| 量 | 格 |
|---|---:|
| 初始液水（2000 步预热后） | 31968.099265547 |
| 5 s 液水 | 35186.546764960 |
| 实际净增加 | 3218.447499413 |
| 初始几何冰体积 | 3600 |
| 初始冰质量折算成液水体积 | 3301.199963298 |
| 守恒要求的末态液水 | 35269.299228845 |
| 实际净增加 − 守恒要求增加 | -82.752463885 |
| 冰水总质量相对误差 | -0.234630% |

各算子在整个 600,000 步中的实际净贡献为：

| 操作 | 对液水的累计贡献 / 格 |
|---|---:|
| 融化退缩后回填新节点 | +2800.245886 |
| 刚体运动中露出旧节点、覆盖新节点 | -3719.547618 |
| 移动冰壁的相场反弹修正 | +4289.312591 |
| phi > 1 截回 1 | -151.973262 |
| phi < 0 截回 0 | +0.000000 |
| 碰撞零阶矩舍入 | +0.381814 |
| 宏观 phi 恢复舍入 | +0.028088 |
| 内部流送净贡献 | +0.000000 |
| 合计 | +3218.447499 |

相场 populations 的内部流送只是把数值从一个流体节点移到另一个，实测净贡献为零。移动冰壁链接额外修改 populations，已单独计入“移动冰壁”一项。直接材料相变不修改 `water_phase`，其后发生的水量变化计入融化几何刷新。

## 3. 从 3600 格出发，差额究竟由什么组成

释放的几何格点必须同时统计运动和融化两个阶段：

```text
融化阶段净释放格点 = 3048
运动阶段净释放格点 = 552
二者合计           = 3600 = 初始固体格点 - 最终固体格点
```

运动阶段的净格点数可以不为零，因为平移、旋转中的二值栅格化会改变固体节点数量。运动阶段释放的流体空间不能全部归到“融化回填”。

把上节账目改写成相对于 3600 的差额：

| 相对几何基准的项 | 格 |
|---|---:|
| 初始几何冰体积 | 3600.000000 |
| 融化回填少于把新格点填满 | -247.754114 |
| 运动启闭节点与移动壁面修正的合计，减去运动净释放格点数 | +17.764973 |
| 相场限幅净变化 | -151.973262 |
| 碰撞、恢复舍入及内部流送残差 | +0.409902 |
| 实际净增加 | 3218.447499 |

```text
3600 - 247.754114 + (+17.764973)
     + (-151.973262) + (+0.409902)
     ≈ 3218.447499 格
```

**融化回填的少补量**：融化阶段释放 3048 格，回填 phi 累计只有 2800.245886 格，平均 phi=0.918716。`_extrapolate_exposed_fluid` 取持续为流体的邻域 phi 平均，并不强制新节点为 1。即使几何上处于水下，只要周围水相 phi 已偏离 1，后续回填仍会继承这个偏差。

平均 phi 接近冰水密度比 0.917，不代表存在相应的回填恒等式。浮力可通过浸没位置间接影响邻域 phi，但本平均值按融化开格次数加权，和静浮时的水下体积比例不同。分段数据、固定密度对照及推导见[回填平均 phi 与密度比的关系](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/research/refill_density_relation_w5_zh.md)。

本次融化暴露节点中，phi≤0.001 的次数为 0，0.001<phi<0.999 的次数为 2590，phi≥0.999 的次数为 458。这是按回填 phi 数值分组；中间组不等于全部位于真实自由液面。

**运动项须与移动壁面项合看**：运动累计露出 34031 次、覆盖 33479 次；对应补入 20326.207430 格水、从流体统计中排除 24045.755048 格水。与此同时，移动壁面修正累计贡献 +4289.312591 格。两个大项彼此抵消，不能把被覆盖的水量单独全算成丢水，或把移动壁面项全算成融水。

**限幅是实测的独立水量变化**：`_bound_water_phase` 将超过 1 的 phi 直接置为 1，累计移除 151.973262 格；负值置零累计增加 0.000000 格。没有把被截掉的部分分配给其他水节点。舍入累计只有 +0.409902 格，量级远小于前述项。

以上是同一次仿真的精确收支分解，不是关闭某项后的独立对照结果。改变限幅或回填还会反馈到流场、运动和传热，不能把删除某一行直接当作修改算法后的预测结果。

![W=5 水量及差额随时间变化](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/water_volume_budget_w5.png)

| 时间 / s | 实际净增加 | 几何释放 | 融化回填少补量 | 运动/壁面相对几何偏差 | 限幅净变化 |
|---|---:|---:|---:|---:|---:|
| 0.00 | -0.000000 | 0 | 0.000000 | +0.000000 | +0.000000 |
| 0.10 | 27.893030 | 10 | 6.994981 | +148.814886 | -123.993969 |
| 0.50 | 1072.816380 | 1040 | 59.744685 | +231.708196 | -139.225386 |
| 1.00 | 1867.980874 | 1938 | 112.012921 | +190.554942 | -148.661667 |
| 2.00 | 2673.876383 | 2843 | 181.354093 | +161.105228 | -149.046730 |
| 3.00 | 3081.623823 | 3312 | 233.374082 | +151.794596 | -149.046730 |
| 4.00 | 3217.368414 | 3599 | 247.754114 | +17.765717 | -151.973262 |
| 5.00 | 3218.447499 | 3600 | 247.754114 | +17.764973 | -151.973262 |

## 4. 运行设置与验证边界

- 主算例 W=5、450×150、四块 30×30 冰、rho_i/rho_w=917/1000、dx=1/3000 m、dt=1/120000 s、预热 2000 步。
- 原 `legacy` 路径在第 12324 步、约 0.1027 s 触发已有流场失稳检查；没有把失败路径当作完整结果。
- 为完成 W=5 的 5 s 核算，使用项目已有 `phase_momentum_coupling=consistent_flux`，其余保持四冰块默认设置。它本身不提供冰融成水的质量源。
- 主算例首次采样到全部材料冰质量为零：4.70 s；固体二值格点已在 4.01 s 的采样中归零，后续仍有低于几何分辨率的材料残冰。采样间隔 0.01 s，算子收支在每一步累计。
- 生产代码文件 SHA-256 与启动时一致；`consistent_flux` 主路径的 128 步带/不带计量对照中，各状态最大差值为 `{"water_phase": 0.0, "phase_populations": 0.0, "momentum_populations": 0.0, "body_mass_lattice": 0.0, "solid_mask": 0.0}`。收支闭合误差 <1e-8 格，几何计数恒等式逐采样严格成立。
- 末态 `final.npz` 独立求和与计量一致；有效流体 phi 范围 [0.000936230, 0.999601305]，末态最大 L1 速度 0.005684371 LU。
- 原始 JSONL 的三个辅助极值字段 `max_l1/phase_min/phase_max` 未采用正确的全局极值归约，因此不作证据；这里的末态极值来自独立 NPZ 检查。加法水量账目与生产稳定性检查不受影响。脚本已修正这三个字段，并在输出内保留实际运行的源文件副本。
- 统计阈值也不是本次差额的来源：主账目包含全部 phi。若用阈值口径，初始会少计 4.580999 格、末态少计 23.574176 格。

**所以，当前水量接近密度比要求是各项收支叠加后的结果，并没有实现 0.917 倍的融水体积转换。W=5 的结果也不应继续引用旧 W=12 的 3338 数字。**

## 5. 复现与文件

在仓库根目录运行（输出目录必须不存在）：

```bash
python -u -m iceflow2d_v3.benchmarks.audit_water_volume_budget --self-check
python -u -m iceflow2d_v3.benchmarks.audit_water_volume_budget \
  --variant legacy --width 5 --end 0.15 --output /tmp/icy_w5_legacy_audit
python -u -m iceflow2d_v3.benchmarks.audit_water_volume_budget \
  --variant consistent_flux --width 5 --end 5 --output /tmp/icy_w5_flux_audit
python -u -m iceflow2d_v3.benchmarks.audit_water_volume_budget \
  --fixed-controls --width 5 --output /tmp/icy_w5_fixed_controls
```

- [计量脚本](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/benchmarks/audit_water_volume_budget.py)
- [完整运行汇总](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/consistent_flux/summary.json)、[配置和源文件摘要](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/consistent_flux/manifest.json)、[本次实际计量源码](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/consistent_flux/audit_source.py)
- [逐采样原始账目](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/consistent_flux/budget.jsonl)、[整理后的 CSV](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/budget.csv)、[独立末态核验](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/analysis.json)
- [规定融化对照](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/fixed_controls_verified/fixed_controls.json)、[默认路径失败记录](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/legacy/summary.json)、[计量不干预状态的对照](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/self_check_consistent_flux.log)
- [图像 PNG](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/water_volume_budget_w5.png)、[可编辑矢量 SVG](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/water_volume_budget_w5_20260922/water_volume_budget_w5.svg)

对应生产代码：[融化材料更新](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:1057)、[几何刷新](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:1562)、[邻域回填](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2199)、[移动壁面相场修正](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2938)、[相场限幅](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3344)。
