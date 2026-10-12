# IceFlow2D v3 双 LBM 冰水质量守恒

冰块从空气中落入热水，继续平移、转动并吸热融化。默认
`water_mass_conservation="conservative_lbm"`：动量保持压力型 D2Q9 MRT LBM，
相场保持 D2Q9 中心矩 LBM；相场 populations 的零阶矩携带实际液水体积。
冰材料格的实际质量损失以同一份数值转移到液水，密度比决定新水体积。
每单位面外厚度满足 `M_ice + rho_water * V_water = constant`。

保留原有锐界格点掩码及 cut-link 边界，流体格点容量 `chi=1`，固体/墙容量为 0。
相场 populations 使用 f64 储存并局部关闭零阶矩，普通碰撞和流送仍由 LBM 完成。
移动壁面的相场修正、节点启闭/回填和相场限幅逐笔记录实际水量变化，以有界转移
关闭收支；融水源按实际材料质量损失生成，并优先放在局部流体格点。
自由液面分配使用事件量、界面格点数和最小容量，不求全域水质量总和。
液体连通性用于隔离无关水滴；非法负值尾部通过最近的可达水体修复，不能跨墙。
显热对流使用相场链路通量，质量转移同时携带显热。

这里的锐界几何仍有格点尺度误差，材料冰体积与二值固体格点数并不逐步相等。
质量守恒口径使用材料冰质量和实际液水量；全融后的水体积由冰水密度比确定。

旧的直接消失和非守恒回填路径保留为 `--water-mass-conservation legacy`，用于
复现历史算例及性能比较。实现、验证、性能数据和复现命令见
[双 LBM 守恒实现与性能报告](research/conservative_lbm_implementation_20261009_zh.md)。

## 运行

需要 NVIDIA CUDA GPU、Taichi 及仓库 `requirements.txt` 中的依赖。

```bash
python -m iceflow2d_v3
```

默认示例采用 `100 × 200` 网格，区域 `0.025 m × 0.050 m`，水深 `0.030 m`，
冰边长 `0.008 m`，初始转角 `5°`，冰底与水面间隙 `0.0015 m`。
水、冰、空气初温分别为 `90 / 0 / 20 °C`，四面墙和气相热界面绝热。
默认物理时长 `5 s`，采样间隔 `0.01 s`，LBM 时间步 `6.25e-6 s`。
默认使用 CM-MRT，并启用相场法向正则化 `1e-3` 和声学松弛率 `0.25`。

短运行：

```bash
python -m iceflow2d_v3 \
  --end-time-s 0.00005 --output-interval-s 0.00005 \
  --no-plot --save-npz --no-progress
```

`--no-plot` 关闭 PNG/GIF，`--save-npz` 保存采样场，`--quiet` 关闭进度和摘要。
完整选项见 `python -m iceflow2d_v3 --help`。

## 多冰块与四冰块算例

```bash
python -m iceflow2d_v3.examples.four_falling_ice_melting_2d
```

该例使用 `450 × 150` 网格，物理区域宽 `0.15 m`、高 `0.05 m`，初始液面
位于 `y=0.025 m`。四块冰均为 `0.01 m × 0.01 m`，无初始倾斜，静止释放。
中心位置为 `(0.03, 0.04)`、`(0.06, 0.04)`、`(0.09, 0.04)`、`(0.12, 0.04) m`；
底面 `y=0.035 m`，即高于液面 `0.01 m`。每块冰占 `30 × 30` 个材料格点。
`dx=1/3000 m`、`dt=1/120000 s`，水/冰/空气初温仍为 `90/0/20 °C`。
默认时长 `5 s`，输出间隔 `0.01 s`；默认目录为
`iceflow2d_v3/outputs/four_falling_ice_melting_2d/`。

四冰块例的数值界面厚度设为 **5 格（约 1.667 mm）**，相场预热 **2000 步**，随后才建立
静水参考场。界面法向仍采用有限差分算法。
本例还默认启用 `--check-stability`，逐步检查空气和水中的 LBM 状态，发生失稳时直接
报告步号、位置、相场和格子速度。其他例可显式传入该选项。

以下历史稳定性结果均来自旧的 `legacy` 水量路径。此前 **12 格界面**的设置已实际完成 **5 s / 600,000 步**，四块冰在最终时刻均融尽；
完整结果和已知的液相体积漂移见[验证记录](benchmarks/four_ice_5s_validation_20260921.md)。
该记录不代表当前 5 格设置的稳定性验证；此前 5 格有限差分方案在约 0.10 s 出现气相失稳。

三项实验性修改现可分别启用，默认均关闭；每次只加表中一项即可复现独立验证：

| 修改 | 命令行选项 | W=5 四冰块独立验证 |
|---|---|---|
| 压力相容的混合递归正则化 | `--momentum-collision-model cm_hrr` | 0.099692 s 失稳 |
| 相场质量通量的完整动量通量修正 | `--phase-momentum-coupling consistent_flux` | 完成 5 s；液相体积增加 10.07%，精度问题仍在 |
| 新暴露节点的非平衡应力回填 | `--exposed-node-refill non_equilibrium` | 0.098658 s 失稳 |

例如单独启用第二项，并将结果写到独立目录：

```bash
python -m iceflow2d_v3.examples.four_falling_ice_melting_2d \
  --phase-momentum-coupling consistent_flux \
  --end-time-s 5 --output-dir iceflow2d_v3/outputs/four_ice_consistent_flux
```

对应 Python 配置字段为 `momentum_collision_model`、`phase_momentum_coupling`、
`exposed_node_refill`，均保存到元数据。`cm_hrr` 的 `hrr_kinetic_blend` /
`--hrr-kinetic-blend` 范围为 `[0,1]`，本次验证使用默认 `0`（梯度重建应力）；
它在有限波长上有额外数值耗散，不能把抑制短波等同于精度提高。
详细推导、针对性测试、误差和复现命令见
[三项独立验证记录](benchmarks/independent_stability_options_20260921.md)。

包含初次入水的短运行：

```bash
python -m iceflow2d_v3.examples.four_falling_ice_melting_2d \
  --end-time-s 0.06 --output-interval-s 0.01 --save-npz
```

所有原示例选项仍可使用，包括 `--thermal-scheme enthalpy_d2q5`、`--no-plot`、
`--output-dir`。`--ice-centers-x-m 0.03 0.06 0.09 0.12` 可修改中心横坐标列表，
冰块尺寸、转角和底面距液面高度由相应原选项设置。

Python 中用 `IceFlowConfig.ice_bodies` 配置不同大小、姿态和温度的冰块。
几何单位为格子，速度单位为格子/LBM 步，角度为弧度，角速度为弧度/LBM 步。
物理坐标除以 `config.dx` 即为格子坐标；下面是在四冰块配置基础上设置两个独立冰块：

```python
from dataclasses import replace
from iceflow2d_v3 import IceBodyConfig, IceFlow2D
from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config

config = replace(create_config(), ice_bodies=(
    IceBodyConfig(size_cells=(30, 30), center_cells=(90, 120), temperature_c=-5),
    IceBodyConfig(size_cells=(24, 24), center_cells=(270, 120), angle_rad=0.1),
))
simulation = IceFlow2D(config)
simulation.step(8)
print(simulation.body_center.to_numpy())  # 多冰块时形状为 (N, 2)，单位为格子
```

冰块编号为配置列表中稳定的零基索引。`body_velocity[b]`、`body_angle[b]`、
`body_mass_lattice[b]` 等访问第 `b` 块冰。省略 `ice_bodies` 时仍使用原单冰块配置，
单冰块的 `[None]` 访问及输出数组维度保持兼容。配置校验会拒绝初始几何重叠和越界。

所有冰块共享一套流体、相场和水温；每块冰分别保存位姿、质量、惯量和材料热状态。
材料网格放在有空列隔离的二维数组中，空列阻断不同冰块间的内部导热，FV 和 D2Q5
使用同一布局。冰水边界热量按冰块编号映射，同一水格点的全部供热请求一起限幅。
流体边界按每条链路识别冰块，分别累加水动力、静水压力和毛细力及力矩。
运动和融化都在全部冰块几何更新后统一重建流体，避免把另一块冰占据的位置填成流体。
一块冰融尽只清除自己的材料和边界；亚网格残冰也逐块暂停运动并继续换热。

冰块间接触使用当前锐界固体格点对应的单元方格；包围盒仅用于筛选，融化后的空洞和
退缩区域不再参与碰撞。采用按质量分配的平移位置修正，以及包含接触点力臂和转动惯量的
等大反向法向冲量。接触无摩擦、无回弹，冰—冰接触面暂设绝热；没有粘连或碎裂。
接触法向和间距只有格子尺度精度，不解析亚网格润滑液膜。容器接触仍沿用原来的
位置约束。多接触与容器约束迭代求解，超出 `body_contact_iterations` 而仍有穿透会报错。

多冰块输出中的 `history.csv` 每个采样时刻按 `body_id` 分行，记录各块冰的运动、质量和
`body_melted_fraction`；公共水量和总融化比例在同一时刻的各行相同。`fields.npz` 中
`body_center_m` 为 `(采样数, 冰块数, 2)`，质量、角度和融化比例为 `(采样数, 冰块数)`；
`solid_body_id` 为 `(采样数, ny, nx)`，`-1` 表示非冰块格点。元数据包含全部初始位置和
各块冰的最终状态，场图标注冰块编号。水量和总冰量仍按整个物理域统计。

温度场默认使用有限体积法（`enthalpy_fv`）。切换到 D2Q5 分布函数模式：

```bash
python -m iceflow2d_v3 --thermal-scheme enthalpy_d2q5
```

传入 `--thermal-scheme enthalpy_fv` 可切回默认模式；Python 配置对应
`ThermalConfig(scheme="enthalpy_d2q5")`。模式在创建模拟器时选择，
`metadata.json` 的 `thermal_scheme` 和 `config.thermal.scheme` 记录实际选项。

D2Q5 参考 `iceflow2d_v2` 的固定 `τ=1` 导热闭合：水的世界网格和冰的材料网格
各维护五个分布函数，碰撞后沿四个轴向流送，公共面的传输量由两侧最小开口率确定。
分布函数之和为单元显热；每个导热子步按材料热扩散率和实际子步长度设置权重，
`Fo = α Δt / dx²`、`w₀ = 1 − 4 Fo`、`w₁…₄ = Fo`。
绝热面反弹，定温墙采用半格距反反弹。对流仍每个 LBM 步单独推进，冰水界面换热、
潜热融化和移动边界重建共用现有逻辑，并同步分布函数的零阶矩。
该固定松弛时间的导热离散与现有有限体积通量等价，因此相同设置下两种模式应接近到舍入误差；
这不是额外可调松弛时间的通用热 LBM 模型。默认模式不分配 D2Q5 分布函数内存。

## step 的三个阶段

旧水量路径的完整解释见[模拟器原理与逐步验证指南](docs/simulator_method_zh.md)，
其中逐步展开 `step()` 的调用链、离散公式和验证入口，并附有 [PDF 版本](docs/simulator_method_zh.pdf)。

`simulator.py` 中的 `step` 只校验步数，并按顺序调用：

1. `_advance_fluid`：动量与相场碰撞，容量有界的相场链路流送、相容显热对流、宏观量和压力恢复。
2. `_advance_body`：积分刚体运动，处理墙面接触，光栅化冰体并重建启闭格点。
3. `_advance_thermal`：累计到热更新间隔后执行导热、界面换热和冰体侵蚀；
   将实际融冰质量和融尽后的剩余显热转移给水，再更新质量、质心、惯量、几何及流体状态。

热更新默认每 8 个 LBM 步执行一次。`step(n)` 的分批方式不改变累计热更新时序。
运动与融化共用 `_refresh_fluid_domain`，不再分成两种填充规则。
每个导热子步结束后立即同步材料质量、几何、流体掩码和水温，再开始下一子步。
冰体的原有光栅覆盖率决定固体掩码、接触范围和 cut-link 交点，不保留原始矩形边缘。
当剩余冰体已没有任何固体格点、无法形成水动力边界时，暂停其刚体平移和转动。
材料质量、连续覆盖率及冰水换热仍保留，继续按潜热融化；不因低于网格分辨率而直接删除质量。
这是亚网格残冰的运动分辨率保护，避免只有重力而没有水动力/壁面接触的不可见残冰穿出容器，
不代表解析了亚网格颗粒的运动。该阶段的材料质心可随侵蚀变化，但参考位姿保持不动。
冰完全融化后，其材料质量、几何覆盖和锐界消失，刚体停止参与运动。

## 热状态与数值约束

冰质量和显热保存在材料坐标网格，水体积和显热保存在世界网格。
热量先把剩余冰升至熔点，再按潜热减少冰质量；实际减少量生成熔点液水。
材料融尽后的多余热量进入液水，默认守恒模式不无记录清零残冰质量。
现有水仍通过导热和冰水界面换热更新温度。

默认液水体积为 `dx² × sum(h_q)`，包括全部相场尾部，热计算阈值不能
删除质量。移动几何转移水时同时转移其显热。相场与热模块共用一份液水体积。
旧的独立外推水温及非守恒热开口重建只在 `legacy` 路径执行。

守恒模式对越界修复的实际质量变化进行配对再分配，不无记录删除水量。
以下 `_bound_water_phase` 行为仅适用于 `legacy`：只修复 `water_phase < 0` 或 `water_phase > 1` 的流体格点，
并同步修正这些格点的相场 populations，使其和等于修复后的相场值。
所有 `[0,1]` 内的相场及 populations 均保持原样，包括界面尾部和纯相端点；
不再执行尾部截断或按 populations 的 L1 范数重建纯相分布。
密度计算已使用 `[0,1]` 限制，修复越界不会改变其密度，因此无需修改动量 populations。
`water_phase_cutoff`（默认 `1e-3`）仅保留为热输运及其 CFL 检查的湿格点阈值。
保留 CFL、Fourier 子步上限及非有限值诊断，以控制显式数值更新的稳定性。
流体仍使用压力–动量 D2Q9 LBM 和 Allen–Cahn 相场，冰壁仍使用 cut-link 动量交换。
融化的法向速度偏移根据实际融化率及固液密度比计算。没有足够受体容量时，
守恒模式报告错误，不丢弃无法分配的水量；封闭无气隙容器还需要相容的体积响应模型。
气相绝热且不求解气相热输运，配置和命令行不再提供未使用的空气比热、导热率选项。

## 分子黏度

动量碰撞的剪切松弛仅使用分子运动黏度：`ν_LU = ν Δt / dx²`、`τ = 0.5 + 3 ν_LU`。
气相、气液界面和冰面附近没有额外剪切黏度下限，也不再叠加 Smagorinsky 涡黏度。
原 `air_interface_relaxation_time`、`smagorinsky_constant` 配置项已删除。

默认 `viscosity_water=None`：按各格点的当前水温计算
[IAPWS SR6-08(2011) 式 (7)](https://iapws.org/technical-guidance/release/LiquidWater.download)
给出的常压液水动力黏度，再用 `ν_water(T) = μ_water(T) / rho_water` 换算。
该关系式适用范围为 −20 至 110 °C，初始水温、熔点及定温边界在创建配置时校验。
水黏度随冷却更新；没有水热容量的相场尾部使用初始水温对应的水黏度。

默认 `viscosity_air=None`：使用
[NASA CAPE 的 Sutherland 关系](https://nasa.github.io/cape-doc/2.3/api/cape/atm.html#cape.atm.SutherlandMKS)
计算 `μ_air(initial_air_temperature_c) / rho_air`。气相不求解热输运，因此使用设定气温。
密度仍采用流动配置中的常数，这与当前 Boussinesq 模型一致；没有另行更新热膨胀密度。
默认密度下，90 °C 水的 `ν≈3.14173e-7 m²/s`，20 °C 空气的 `ν≈1.45065e-5 m²/s`。
界面保持 `ν(φ,T) = φ ν_water(T) + (1−φ) ν_air`，冰面附近同样使用此式。

显式传入 `--kinematic-viscosity-water` 或 `--kinematic-viscosity-air` 时，该相使用指定的
常数运动黏度，单位为 `m²/s`；Python 配置的同名物性字段对应 `viscosity_water/air`。
`metadata.json` 的 `viscosity` 记录实际物性模型和初始数值。
`test_physical_viscosity.py` 从生产碰撞的剪切应力松弛反算黏度，覆盖纯空气、冷热水、
气液界面及冰面邻格，并施加周围速度梯度来检验没有额外涡黏度。

此前含人工黏度的速度稳定性及表面涡诊断属于历史结果，不能作为当前物性设置的稳定性结论。

动量碰撞可通过 `--momentum-collision-model` 选择，默认使用 CM-MRT，仍可切回 RM-MRT：

```bash
python -m iceflow2d_v3 --momentum-collision-model rm_mrt
python -m iceflow2d_v3 --momentum-collision-model cm_mrt
```

Python 配置对应 `IceFlowConfig(momentum_collision_model="cm_mrt")`，选项在创建模拟器时确定，
并保存到 `metadata.json` 的 `config.momentum_collision_model`。
CM-MRT 将现有压力平衡分布和完整源项投影到物理速度对应的中心矩空间中，再施加各矩的松弛率
及对应的半步源项系数。剪切黏度沿用原物性；CM 的二阶迹和 `22` 矩默认使用松弛率 0.25，
`21`、`12` 矩使用 1；RM 保持原来的松弛率。
CM 的零阶和一阶矩共同使用独立松弛率 1，保持半步力的原始动量增量，
并避免将这些模态绑定到接近 2 的剪切松弛率；RM 保持原设置。
两种模式共用平衡分布、静水分解、毛细力、宏观恢复、相场、边界及热耦合。
这提供了只改变动量碰撞模型的对照选项，不代表已证明长期稳定性改善。

底部相场斑点与快速压力振荡由两个独立选项控制。直接运行 `python -m iceflow2d_v3`
已等效于以下组合：

```bash
python -m iceflow2d_v3 \
  --momentum-collision-model cm_mrt \
  --phase-normal-epsilon 1e-3 \
  --cm-acoustic-relaxation-rate 0.25
```

默认使用 `momentum_collision_model="cm_mrt"`、`phase_normal_epsilon=1e-3`、
`cm_acoustic_relaxation_rate=0.25`。Python 配置使用同名字段，实际数值保存到
`metadata.json` 的 `config` 中。两个参数在创建模拟器时确定，可单独关闭或调整。
切换为 `rm_mrt` 且未显式指定声学松弛率时，自动使用其原始值 1，相场正则化仍保持开启。
Python 配置和 CLI 中省略声学松弛率均按碰撞模型解析；显式传入值会接受原有校验。

恢复完整的旧 RM-MRT 设置：

```bash
python -m iceflow2d_v3 --momentum-collision-model rm_mrt --phase-normal-epsilon 0
```

`phase_normal_epsilon` 是非负的格子梯度尺度：启用时压缩法向采用
`grad(phi)/sqrt(|grad(phi)|²+epsilon²)`，并保留防除零保护；极弱的体相梯度不再得到单位
压缩方向。`0` 恢复原法向公式。它可用于两种动量模型，也用于初始化的相场预热；
不更改相场平衡态、mobility、界面宽度、毛细力或局部限幅。

`cm_acoustic_relaxation_rate` 必须在 `(0,2)` 内，同时控制 CM 的二阶迹和 `22` 矩；
值不等于 1 时要求选择 `cm_mrt` 或实验性的 `cm_hrr`；HRR 使用独立压力矩闭合，
具体区别见上述独立验证记录。CM-MRT 的两个矩各自使用匹配的 `(1-s/2)` 源项系数，
`20-02`、`11` 剪切矩仍由物理黏度决定，零阶与一阶矩仍使用共同松弛率 1。
这里不能只调整二阶迹：静止无源极限下，压力恢复包含 `3/5*(M20+M02-M22)`。
测试值 `0.25` 增加数值体积/声学耗散，**不改变物理剪切黏度**；`1` 恢复原碰撞。
这两个推荐数值来自当前网格的诊断，不是所有网格和两相工况的通用最优值。

原因、四组对照及适用限制见
[底部速度诊断](benchmarks/bottom_velocity_root_causes_20260921.md)。
生产代码的复算和回归结果见
[实现验证](benchmarks/stability_options_validation_20260921.md)。
`test_phase_normal.py` 检查纯相、小扰动斑点和界面厚度；`test_momentum_collision.py`
检查压力波衰减与波速、完整源项、平衡态、剪切波黏度和优化开启后的高密度比入水过程。

## 表面张力与冰块受力

表面张力使用切向毛细应力 `K=κ(|grad(phi)|²I−grad(phi)⊗grad(phi))` 的离散散度，
其中 `κ=1.5σW`。应力沿法向的连续积分在平衡 tanh 剖面上为 σ，平直界面法向应力为零。
应力内部保留原有混合梯度，压力源项与应力散度使用中心模板；两者在弯曲界面上的
离散平衡仍有残差。归一化应力和直接替换中心梯度的候选方案未通过所有入水姿态的
稳定性检查，未接入生产代码。

容器墙面及冰面都对相场和应力作镜像延拓，保留沿壁面的变化。冰面按 cut-link
交点与局部法向重建，采用中性润湿条件 `n_ice · grad(phi) = 0`；应力张量也随镜像
变换。相邻流体共用链路应力通量，冰体接收相反的力；力矩包含链路中点力臂及
镜像应力对应的离散修正，使流体与冰体的总力、总力矩闭合。

相场碰撞直接保留守恒的零阶矩，并用静止分布函数闭合重构后的相场量。
相场求和使用双精度累加、单精度存储，减少逐步求和平衡态造成的系统性水量漂移。

默认表面张力为 `0 N/m`（关闭），初始倾角仍为 `5°`，水平平移和转动均自由。
倾斜入水可以产生侧移；本次修复针对穿冰模板丢失切向变化产生的伪力，不强制水平速度为零。
可通过 `--surface-tension 0.072` 显式启用表面张力。

回归覆盖不同厚度的平直界面零力、静止冰面接触线、圆形液滴的
`Δp=σ/R` 和残余速度、冰体与流体的毛细力/力矩平衡、水平入水稳定性，以及零表面张力和纯相极限。
静止基准可运行：

```bash
python -m iceflow2d_v3.benchmarks.check_capillary_equilibrium \
  --case contact --end-time-s 0.1 --output-dir /tmp/ice_contact_check
python -m iceflow2d_v3.benchmarks.check_capillary_equilibrium \
  --case drop --refinement 2 --end-time-s 0.5 --output-dir /tmp/ice_drop_fine_check
```

静止基准不推进冰体与热场；网格细化默认保持物理界面厚度不变。
默认、镜像及水平入水对照可运行：

```bash
python -m iceflow2d_v3.benchmarks.check_capillary_coupling \
  --case default --end-time-s 2 \
  --output-dir iceflow2d_v3/outputs/capillary_check
```

输出目录必须是新目录；其他 `--case` 为 `mirror`、`symmetric`、`no_surface_tension`、
`cold_water`。本次修复依据及验证范围见
[冰面边界修正验证](benchmarks/surface_vortex_fix_validation_20260921.md)；更早的
[表面张力耦合记录](benchmarks/capillary_coupling.md) 保留为历史资料。

## 输出与测试

默认输出目录为 `iceflow2d_v3/outputs/iceflow2d_v3/coupled_falling_ice_melting_2d/`。
`history.csv`、`metadata.json` 和可选的 `fields.npz` 保存冰体运动、剩余质量、融化比例、
当前水量和平均水温，不再输出守恒残差、水量目标或融水注入统计。
默认生成速度、涡量、温度和气液相场的 PNG 序列及 GIF 动画。
温度图与速度图使用相同的水域掩码：只显示 `water_phase >= 0.5` 的流体格点，
冰块、容器墙体和空气区域显示为灰色；温度序列与最终汇总图均只显示水温。
相场输出为 `phase_frames/frame_*.png` 和 `phase_field.gif`，使用 `water_phase`，
色标固定为 `[0, 1]`（0 为空气、1 为水），冰块和容器墙体显示为灰色。
`metadata.json` 记录相场序列路径、帧数和色标；`--no-plot` 同时关闭这些可视化。
图像序列逐帧写盘；只有 `--save-npz` 会在内存中保留所有采样场。

```bash
python -m pytest iceflow2d_v3 --ignore=iceflow2d_v3/outputs -q
python -m ruff check iceflow2d_v3
```

测试覆盖局部相场限幅、热对流稳定性、部分/完全融化不生成液水、冰体完全消失、
水下侵蚀不产生毛细冲击、多层格点温度重建、热子步几何同步、刚体接触及输出。
`test_thermal_d2q5.py` 另覆盖 D2Q5 的解析导热衰减、绝热热量守恒、四面定温通量、
部分开口率、与有限体积法的一致性，以及运动、对流和融化后的分布函数同步。

速度闪烁诊断（每 32 步采样一次实际速度场，输出目录需为新目录）：

```bash
python -m iceflow2d_v3.benchmarks.check_velocity_stability \
  --output-dir iceflow2d_v3/outputs/stability_check
```

`--case cold_water`、`--case no_surface_tension`、`--case fixed_ice` 提供对照。
原因、测量口径和修复结果见 [速度稳定性记录](benchmarks/velocity_stability.md)。

本模型处理初始给定的多个随融化侵蚀的刚体，不自动将融断的材料分裂成新刚体，
也不包含重新冻结、沸腾或蒸发。
`benchmarks/step_profile_*.json` 和 `review_results.json` 是改动前的测量快照；
新版本应通过 `python -m iceflow2d_v3.benchmarks.profile_step --output /tmp/step.json` 重新测量。
