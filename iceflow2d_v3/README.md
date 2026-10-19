# IceFlow2D v3：融化后直接消失的冰块

冰块从空气中落入热水，继续平移、转动并吸热融化。融化只减少材料网格中的冰质量；
融化掉的部分直接消失，不生成液水，也不向流体补入显热、线动量或角动量。
运动或融化露出的格点，都从周围已有流体外推相场、压力和速度。
水下露出的格点延续周围水的状态，空气中的格点延续空气状态；不按融冰量注入液水。
这是局部移动边界重建，不约束重建前后的全局水量。

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
刚体默认 `linear_damping=angular_damping=1.0`，不额外衰减平移和转动速度；
水动力阻力及接触冲量仍参与运动更新。若显式配置额外角阻尼，应按物理衰减率
`gamma_s_inv` 和时间步设置 `angular_damping=exp(-gamma_s_inv * dt_s)`。

短运行：

```bash
python -m iceflow2d_v3 \
  --end-time-s 0.00005 --output-interval-s 0.00005 \
  --no-plot --save-npz --no-progress
```

`--no-plot` 关闭 PNG/GIF，`--save-npz` 保存采样场，`--quiet` 关闭进度和摘要。
完整选项见 `python -m iceflow2d_v3 --help`。

## 热水溃坝冲击四冰块

```bash
python -m iceflow2d_v3.examples.dambreak_four_ice_melting_2d
```

默认计算域为 `150 × 50 mm`，网格 `450 × 150`，与四冰块落水例相同。
沿用 3 格厚的墙层（1 mm）；左侧热水柱的实际宽、高为 `65 × 45 mm`，
范围为 `x=[1,66] mm`、`y=[1,46] mm`，初温 90°C。整个内地板上方初始已有
31 格深的同温水层（约 `10.333 mm`，水面坐标约 `y=11.333 mm`）。
四块 `10 × 10 mm` 冰块初温 0°C，中心依次为 `(85,7)`、`(103,7)`、
`(121,7)`、`(139,7) mm`，冰底离内地板 3 格（1 mm），间隙中充满水。
默认水深按冰的静态吃水估计值加离地间隙向上取整到整格；零转角时使用
`ceil[3 + 30 × (917 − 1.25)/(1000 − 1.25)] = 31` 格，使冰下有可解析的水层。
水和冰初始静止，冰块初始转角为零，空气初温 20°C。
冰块中心距为 18 mm，净距为 8 mm，水柱右缘至第一块冰左缘为 14 mm。
`t=0` 对应瞬时撤闸；域中没有残留闸板。局部水柱使用下游浅水层的静水参考，
参考水压延伸至冰底；水柱相对浅水层的密度差沿各列竖直积分，形成释放后驱动
水平溃流的初始压力差。原满宽水池仍使用原来的水池静水参考。

水的黏度随局部温度按 IAPWS 计算，空气黏度按设定气温使用 Sutherland 定律，
与 `four_falling_ice_melting_2d` 相同，不增加人工剪切黏度。
初始运动黏度分别为 `3.14173258e-7 m²/s`（90°C 水）和
`1.45064520e-5 m²/s`（20°C 空气）；沿用水/空气密度 `1000/1.25 kg/m³`。
冰块可独立平移、转动、接触和融化；融掉的冰直接消失，沿用 v3 模型。
墙接触同时约束接触点的平移与转动法向速度，采用无摩擦、无回弹冲量。
本溃坝例默认使用 5 格弥散界面（约 1.667 mm），保留原物理黏度与时间步。
此前 65 mm 宽水柱、12 格界面的设置已完成 6 s；65 mm 宽、5 格界面的重跑在第 9,168 步（约 0.0764 s）
因气相速度发散退出，未达到 8 s。此前 25 mm 宽水柱、四冰块整体左移 40 mm
的旧 `legacy` 梯度设置在第 8,549 步（约 0.071242 s）出现气相失稳。
入口默认启用 `connected_ls`；其旧版硬约束重构运行在第 8,287 步（约 0.069058 s）
出现气相失稳，仍未达到 8 s；它不能视为已经修复完整溃坝算例。
默认保留逐步稳定性检查，失败时保存
`run_failure.json` 和已生成的部分图片/GIF，不保存原始场数组。
2026-10-14 使用数据优先重构的完整四冰块重跑，在第 `8,532` 步（`0.0711 s`）
于 `(152,3)` 报告速度失稳，仍未达到 8 s；结果保存在
`outputs/dambreak_four_ice_data_priority_8s/`，四类 PNG/GIF 各 8 帧，覆盖 `0–0.07 s`。
逐阶段重放将首次触发追溯到第 `8,520` 步：底部约半格厚的窄缝连通，补点产生
`φ≈1 → 0.50005 → 0.000107` 的未解析过渡。实际切向梯度与独立 SVD 一致，
但后续相场、密度源项、压力及边界离散在该跃迁处失稳；第 `8,523` 步的大速度
又被几何更新后的固体掩码遮蔽，延迟了报错。
保持局部相量的一次平滑干预可继续 `3,249` 步，随后在第 `11,770` 步遇到无连通
供体的新气穴，不能视为完整修复。证据图、梯度复算及匹配状态对照见
`outputs/four_ice_data_priority_diagnosis/evidence_summary.json`。
上述均为历史干底面工况的失稳记录；当前水柱为 `65 × 45 mm`，并增加了底部浅水层。
速度图同时显示空气和水，两相使用不同色图及独立色标：水为 `viridis`、`0–0.8 m/s`，
空气为 `Oranges`、`0–4 m/s`。白色箭头表示水，橙色箭头表示空气；箭头分别按各相速度
上限缩放，并仅限制显示长度，不改变实际速度。虚线标出水面，灰色区域为冰和墙。
涡量图同样显示两相：水为 `RdBu_r`、`±250 s⁻¹`，空气为 `PuOr`、`±5000 s⁻¹`。
两相涡量均按相内的中心差分计算，避免差分跨越气水界面。两个色标在整个序列中固定。
空气色标基于当前短跑的速度/涡量分位数选择，不使用失稳后的发散值。
可用 `--air-velocity-visualization-max-m-s`、`--air-vorticity-visualization-max-s-1` 调整空气色标，
用 `--air-velocity-colormap`、`--air-vorticity-colormap` 调整空气色图。
`--no-velocity-include-air` 或 `--no-vorticity-include-air` 可分别恢复仅显示水的图。

默认目标为 **8 s / 960,000 LBM 步**，每 **0.01 s** 流式生成速度、涡量、温度、
相场 PNG 序列及 GIF；完整运行应有 801 个采样时刻，并生成最终温度/相场图。
默认不保存 NPZ，也不保留采样场数组；`history.csv` 仅记录逐冰块标量诊断，
`metadata.json` 记录参数和结果。默认输出目录为
`iceflow2d_v3/outputs/dambreak_four_ice_melting_2d/`，最终图为
`dambreak_four_ice_melting_2d.png`，序列分别在 `velocity_frames/`、
`vorticity_frames/`、`temperature_frames/`、`phase_frames/`。
可用 `--water-column-width-m`、`--water-column-height-m` 调整水柱实际尺寸。
`--ice-floor-clearance-cells` 调整初始冰底间隙（默认 3 格），自动水深随之变化；
也可用 `--floor-water-depth-m` 指定与网格对齐的底部水深。底部水深从内地板算起，
必须低于左侧水柱高度。复现原干底面场景可同时设置
`--floor-water-depth-m 0 --ice-floor-clearance-cells 0`。

本例启用局部密度梯度源项限幅与补点矩修正，仍使用原 CM-MRT 动量 LBM、
相场 LBM、物理黏度、`W=5`、`dt=1/120000 s` 和 `consistent_flux`。
其他算例的通用配置默认关闭这些选项。参数为：

- `--density-gradient-source-limit 0.05`：仅当
  `|grad(phi)| > [6 phi(1-phi)+0.0005]/W` 时，才检查源项相对于本格及连通
  八邻域最小流体密度的强度。超出预算时，九个方向共用
  `alpha=min(1, Cs*rho_min/[delta_rho*|u|_1*|grad(phi)|_1+eps])`。
  梯度门槛是平衡相场斜率的 1.5 倍加微小底值；这是一项经验判据，
  不是稳定性定理，也不是速度上限。相场本身的梯度和压缩源项不变。
- `--phase-flux-acceleration-limit 0.01`：仅限幅 `consistent_flux` 的数值
  修正力；每个共享面预算为 `0.25*limit*min(rho_left,rho_right)`。
  两端使用相同系数，保持等大反向；重力、浮力及表面张力不受此限幅影响。
  实际力同时用于碰撞和半力恢复。
- `--refill-moment-projection`：新点的相场补齐并重算梯度后，通过固定 D2Q9
  权重修正分布，使一阶矩满足 `rho*u-F/2`，移动方向和满足所指定动态压力
  的恢复公式。保留分布总和及二阶偏应力，不改变补点相量。
  补点使用新几何的物理力；相场数值通量在下一次相场碰撞后重新计算。
- `--gas-state-repair`：对气相占多数的 `phi<0.5` 节点，速度 L1 范数超过
  `0.25` 且单步变化超过 `0.15` 时才检查连通八邻域；其速度还必须超过
  `max(0.25, 2*邻域上一步最大速度+0.05)` 才触发。速度超过 `1.0` 的其他
  相态节点也纳入异常检查；速度超过 `1.0` 且单步变化超过 `0.5` 时直接修复，
  防止已发散的气相扰动扩散到水侧。使用上一步速度不超过 `0.25` 的连通
  邻点平均速度及动态压力；没有可靠参考时取零速度和零动态压力。
  当前有界相量保持不变，重建两套平衡分布并匹配实际半力和当前受限源矩。
  这是局部非守恒修复，不能代替质量/动量误差评估。
  正常高速且无突增/空间异常的空气不受限制。没有额外全域核或 CPU 邻域搜索；
  `gas_repair_count` 和 `gas_repair_peak_count` 记录累计修复次数及单步峰值。

压力恢复与补点投影均调用相同的受限源矩公式，使用该阶段的相场和物理速度。
限幅只改变动量方程的数值源项；它会在触发处引入局部近似，不能据此声称
严格满足连续方程或证明任意工况稳定。计算不含全局求解、邻域排序或逐点迭代；
补点额外核只在实际出现新流体点时启动。
关闭这批选项可用
`--density-gradient-source-limit 0 --phase-flux-acceleration-limit 0 --no-refill-moment-projection --no-gas-state-repair`。

## 热水溃坝冲击单冰块

```bash
python -m iceflow2d_v3.examples.dambreak_single_ice_melting_2d
```

沿用上述四冰块溃坝例的物理、数值和可视化配置，初始水柱同为 `65 × 45 mm`。
单冰块例保留干底面及贴地初始位置，冰块中心为 `(85,6) mm`；可显式指定底部水深和
冰底间隙。默认目标仍为 `8 s / 960,000 LBM 步`，
每 `0.01 s` 输出一次，并保留逐步稳定性检查和失败诊断。
输出目录为 `iceflow2d_v3/outputs/dambreak_single_ice_melting_2d/`，
最终图为 `dambreak_single_ice_melting_2d.png`；场序列目录和 GIF 文件名
与四冰块例相同。
2026-10-13 的旧版硬约束梯度配置运行在第 `22,642` 步（约 `0.188683 s`）因流体速度
超出 D2Q9 稳定性限制退出，未达到 8 s；保留了 `0–0.18 s` 的四类
PNG/GIF（各 19 帧）以及 `run_failure.json`。

2026-10-14 接入数据优先的连通重构后，31 项梯度/耦合/配置检查通过。
单冰块算例完成 `5 s / 600,000` 步，未触发逐步稳定性检查；按用户要求在保存
5 s 的完整输出后主动停止，本次未继续验证到 8 s。
结果位于 `outputs/dambreak_single_ice_data_priority_5s/`，四类 PNG/GIF 各 501 帧，
覆盖 `0–5 s`，完整性和主动停止原因记录在 `run_stop.json`。

2026-10-15 改为固定 3×3 的统一单精度线性梯度，并将溃坝法向正则化显式设为
`0.01` 后，单冰块完整运行 **6 s / 720,000 步**，逐步稳定性检查保持开启。
四类 PNG/GIF 各 121 帧，输出间隔 0.05 s，保存于
`outputs/dambreak_single_ice_compact3x3_6s/`；`run_result.json` 记录完成状态，
`verification.json` 核对源码指纹、所有 PNG、GIF 末帧及 121 个采样状态。
最终水相总量较初始增加约 **6.97%**，该稳定性结果不代表严格相量守恒或精度通过。
当前 3×3 四冰块例尚未完成 8 s 验证；上文四冰块失败记录属于此前算法。

复现本轮单冰块运行（显式保留当时 25 mm 水宽及冰块位置）：

```bash
python -m iceflow2d_v3.examples.dambreak_single_ice_melting_2d \
  --water-column-width-m 0.025 --ice-centers-x-m 0.045 \
  --end-time-s 6 --output-interval-s 0.05 --check-stability \
  --output-dir iceflow2d_v3/outputs/dambreak_single_ice_compact3x3_6s
```

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

此前 **12 格界面**的设置已实际完成 **5 s / 600,000 步**，四块冰在最终时刻均融尽；
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
接触法向和间距只有格子尺度精度，不解析亚网格润滑液膜。容器接触使用位置约束，
并在当前锐界支撑面的接触点施加包含转动惯量的无摩擦、无回弹法向冲量。
多接触与容器约束迭代求解，超出 `body_contact_iterations` 而仍有穿透会报错。

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

面向初学者的完整解释见[模拟器原理与逐步验证指南](docs/simulator_method_zh.md)，
其中按 2026-10-14 的历史源码快照逐步展开 `step()` 的调用链、输入输出、离散公式和验证入口，
并附有 [PDF 版本](docs/simulator_method_zh.pdf)。2026-10-15 的固定 3×3 单梯度实现以本文末节为准；该历史指南中的二次拟合和两套内部梯度不再是当前生产算法。第 3.1 节解释水柱压力，
第 4 节核对每一步的时序，第 5.10 节推导数据优先拟合、单行闭合、零空间壁面修正和诊断字段，第 6.3/6.5 节解释
转动墙接触与分轮补点，第 9 节说明验证范围和尚未实施的方案。

`simulator.py` 中的 `step` 只校验步数，并按顺序调用：

1. `_advance_fluid`：准备相场梯度与毛细应力，动量与相场碰撞、流送、宏观量恢复。
   `consistent_flux` 先碰撞相场、测量通量，再碰撞动量；所有梯度选项均在流送后
   先限幅相场、刷新唯一梯度，再恢复压力。
2. `_advance_body`：消费本轮冲量、积分刚体运动，处理含转动的墙面接触，
   光栅化冰体并重建启闭格点。`connected_ls` 先准备新几何的供体连通关系，
   再通过双缓冲传播回填并刷新梯度；当前回填不是整块守恒重映射。
3. `_advance_thermal`：每步重映射水温并对流水显热；累计到热更新间隔后执行导热、
   界面换热和冰体侵蚀，再更新剩余冰体的质量、质心、惯量和流体掩码。

热更新默认每 8 个 LBM 步执行一次。`step(n)` 的分批方式不改变累计热更新时序。
运动与融化共用 `_refresh_fluid_domain`，不再分成两种填充规则。
每个导热子步结束后立即同步材料质量、几何、流体掩码和水温，再开始下一子步。
冰体的光栅覆盖率统一决定固体掩码、接触范围和 cut-link 交点，不保留原始矩形边缘。
single-node IBB 直接使用水平集插值得到的交点比例，不再将其截到 `[0.05,0.95]`。
其反弹分母为 `1+q`，在链路两端仍有限；保留原有流体端距离下限和分母保护。
当剩余冰体已没有任何固体格点、无法形成水动力边界时，暂停其刚体平移和转动。
材料质量、连续覆盖率及冰水换热仍保留，继续按潜热融化；不因低于网格分辨率而直接删除质量。
这是亚网格残冰的运动分辨率保护，避免只有重力而没有水动力/壁面接触的不可见残冰穿出容器，
不代表解析了亚网格颗粒的运动。该阶段的材料质心可随侵蚀变化，但参考位姿保持不动。
冰完全融化后，其材料质量、几何覆盖和锐界消失，刚体停止参与运动。

## 热状态与数值约束

冰质量和显热保存在材料坐标网格，水体积和显热保存在世界网格。
热量先把剩余冰升至熔点，再按潜热减少冰质量；材料融尽后的多余热量直接丢弃。
低于初始材料单元质量百万分之一的数值残量直接清零，避免留下无限趋近于零的冰格点。
现有水仍通过导热和冰水界面换热更新温度。

热水体积直接取当前有效流体格点的 `dx² × water_phase`。开口变化时，已有水保持局部温度；
新湿格点从最近的湿邻域外推温度，必要时扩大搜索范围以覆盖一次露出的多层格点；
整个区域都没有湿邻域时取熔点。
不再进行全局水量投影、显热修正、融水回退分配或质量/能量残差判断。
因此本版本不要求冰水系统的总质量和总能量闭合。

`_bound_water_phase` 只修复 `water_phase < 0` 或 `water_phase > 1` 的流体格点，
并同步修正这些格点的相场 populations，使其和等于修复后的相场值。
所有 `[0,1]` 内的相场及 populations 均保持原样，包括界面尾部和纯相端点；
不再执行尾部截断或按 populations 的 L1 范数重建纯相分布。
密度计算已使用 `[0,1]` 限制，修复越界不会改变其密度，因此无需修改动量 populations。
`water_phase_cutoff`（默认 `1e-3`）仅保留为热输运及其 CFL 检查的湿格点阈值。
保留 CFL、Fourier 子步上限及非有限值诊断，以控制显式数值更新的稳定性。
流体仍使用压力–动量 D2Q9 LBM 和 Allen–Cahn 相场，冰壁仍使用 cut-link 动量交换。
删除了体积投影遗留的“必须存在自由液面”限制；满水容器也可以模拟冰体消失。
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
所有入口的相场、密度源项、压力恢复与毛细应力均读取同一份固定 3×3 梯度；
内部点退化为 D2Q9 一层中心模板。应力散度仍使用共享链路模板，
弯曲界面的离散平衡仍有残差。归一化应力等历史候选方案未通过所有入水姿态的
稳定性检查，未接入生产代码。

梯度不使用固体相场值或壁面法向硬约束；标量虚拟点采用限幅的线性外推，
不反过来缩放梯度。两种分支的流送边界采用中性润湿条件
`n_ice · grad(phi) = 0`，应力张量仍作镜像变换；新分支限制应力插值供体的连通性。
相邻流体共用链路应力通量，冰体接收相反的力；力矩包含链路中点力臂及
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
温度图与速度图的水相部分使用相同掩码：只显示 `water_phase >= 0.5` 的流体格点，
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

### 固定 3×3 的统一相场梯度（2026-10-15）

所有入口的相场碰撞、毛细应力、密度源项和压力恢复都读取
`phase_gradient_lattice`。内部和边界均使用固定的 3×3 连通流体候选，不读取第二圈。
`source_gradient_lattice` 仅为历史诊断脚本保留的属性别名，返回同一个 Taichi 字段；
它没有独立存储、计算或刷新过程，不存在可独立改变的第二套梯度。

用线性多项式 `phi_i + gx*x + gy*y` 近似局部相场，以 D2Q9 权重求
`M = sum(w*c*c^T)`、`b = sum(w*c*(phi_neighbor-phi_i))`。
二维供体直接用单精度 2×2 显式解求 `M*g=b`；共线供体采用 `g=b/trace(M)`，
保留可观测的切向变化，将缺失的法向导数闭合为零；无供体返回常值闭合。
完整八邻域的结果恰为一层 D2Q9 中心梯度。
该近似允许墙旁曲率误差，不再求 Hessian、双精度 QR、数据零空间或多壁面修正。

候选编码改为九位：`bit=(dx+1)*3+dy+1`，中心为第 4 位。
固体值不参与拟合；对角点需有面连通路径，或通过连续冰覆盖率的链路可见性检查。
`connected_ls` 的补点也通过一层邻居分轮传播，只读取上一轮已完成状态。
若未填节点数不再减少，将剩余新点暂缓激活，不指定其水气相态；有供体的
多层节点正常提交。`solid_mask` 使用 0 表示流体、1 表示真实冰、2 表示待激活。
待激活点不参加流体碰撞、迁移或水侧热输运，对外按静止封闭边界反弹，
不计入冰质量、几何体积、接触支持或刚体受力。它是一项局部几何近似。
每次几何刷新重新将这些点加入候选，但不把它们当作已有流体供体；连通后
正常补点，并执行已启用的矩修正。冰的运动和材料热更新继续进行。
传播次数不设 `nx+ny` 截断：仍有进展时继续传播，避免将长路径误判为无供体。
`refill_pending_events`、`refill_pending_activations`、`refill_pending_max_age_steps`
记录延迟事件、重新激活次数和最长等待步数；长期或大范围待激活需要另行评估。

状态码为 0 非流体、1 完整八邻域、2 单行闭合、3 二维线性拟合、5 孤立常值闭合。
空间秩和拟合秩现在均为两列线性问题的秩（0/1/2），不再包含二次曲率闭合行。
二次系数和壁面残差字段已移除。梯度在碰撞前、相场越界修复后，以及几何更新/
补点后刷新；标量 ghost 单独限幅，梯度不按固体侧外推值整体缩放。
`phase_gradient_scheme` 的历史名称仍接受：`legacy` 保留旧补点和标量镜像兼容路径，
但所有实际梯度消费位置都使用上述同一份缓存。

相同单冰块起始状态的 32 步 CUDA 短窗口测量中，旧/新梯度内核每步总耗时分别约
2.371/0.0166 ms，供体几何准备约 0.161/0.0078 ms，整步约 3.905/1.202 ms。
这是早期状态的短窗口性能记录，不代表全程平均速度；记录位于
`outputs/compact_gradient_3x3_work_20261015/profile_before.json` 与 `profile_after.json`。

统一的一层梯度在原 `phase_normal_epsilon=0.001` 下仍于第 16622 步（0.13852 s）
出现气相尾部速度发散；失败记录保存在该工作目录的 `attempt_01_linear_epsilon1e3/`。
溃坝入口因此显式将法向正则化设为 `0.01`，减少弱梯度区域的相场压缩；
分子黏度、密度、重力、时间步、碰撞松弛率、界面宽度和冰热模型均保持原设置。
固定 3×3 算法不保证整块守恒补点、相场逐链路有界性或所有几何下的稳定性。
