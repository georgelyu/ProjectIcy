# 热水溃坝四冰块：失稳定位与停止修复记录

本文保留停止修复时的历史状态。用户随后重新授权修复并要求保留 single-node；当前修改与验证见
[后续修复记录](dambreak_single_node_10s_validation_20261012.md)。

日期：2026-10-12。用户要求停止修复后，未再修改求解算法；仅对原默认算子的调用进行观测，保存诊断记录。本算例尚未完成 10 s 验证，不能宣称可以稳定运行。

## 场景和当前实现

- 模块：`iceflow2d_v3.examples.dambreak_four_ice_melting_2d`。
- 总区域 150 × 50 mm，450 × 150 格，四面各 3 格墙；dx=1/3000 m，dt=1/120000 s。
- 左侧热水宽 65 mm、高 45 mm，90 °C；尺寸从墙内侧量起，右缘 x=66 mm，水面 y=46 mm。
- 四块 10 × 10 mm、0 °C 的冰，中心为 (80,6)、(100,6)、(120,6)、(140,6) mm，初始贴底静止，允许平移、转动和融化。
- W=5 格，sigma=0，分子黏度，守恒相场，consistent_flux 动量耦合；新例取消刚体人工角阻尼。
- 新例默认 10 s、每 0.01 s 输出，配置生成速度、涡量、温度、相场 PNG/GIF，以及 fields.npz。此处的失败探测为加快定位显式关闭了绘图和整段 NPZ。

实际命令使用 `/home/georgelyu/anaconda3/envs/icy/bin/python`，Taichi 1.7.4，CUDA，RTX 5090。

## 已确认的位置与触发阶段

1. 默认方案在第 5361 步、t=0.044675 s 失稳。重复运行以及插入观测的诊断均在同一步失稳；最坏格点在 (239,3) 与 (240,3) 之间有微小变化。
2. 格点采用零基索引，格心坐标为 `(i+0.5,j+0.5)*dx`，故异常位于 x≈79.83–80.17 mm、y≈1.167 mm，即第一块冰底部与底板之间只有一格厚的气相狭缝。该格上方为冰，下方为墙。
3. 保存的默认重放失败帧中，第一块冰已转动约 4.18°、中心移动至约 x=83.71 mm。异常格的水相比例仅约 3.31e-5，属于气相。
4. 对 `_collide_momentum`、`_update_streamed_macroscopic_fields`、`_update_pressure`、`_synchronize_conservative_momentum`、`_refill_changed_nodes` 和 `_advance_body` 的观测没有更改这些方法的计算内容。
5. 气隙速度突增首次体现在动量流送后的 `_update_streamed_macroscopic_fields`。随后 `_update_pressure` 进一步放大负压；该时刻的回填、相场守恒修正和刚体推进没有直接引起相同的速度跃变。

观测的气隙峰值（速度是欧氏范数，格子单位；物理速度乘 40 m/s）：

| 步号 | 流送后的气隙速度 LU | 随后恢复的压力 LU |
|---|---:|---:|
| 5358 | 0.2577 | -0.0019025 |
| 5359 | 1.0737 | -0.0067592 |
| 5360 | 2.5135 | -0.0348027 |
| 5361 | 253.8852 | -33.2756 |

完整的 4067 条算子观测记录在 `outputs/dambreak_diagnostics/operator_trace/operator_trace.csv`。

## 定位结论及边界

可以将问题定位为**第一块冰运动产生的单格气隙内，动量碰撞/反射/流送与低密度速度恢复、压力恢复之间的数值失稳**。速度恢复计算 `sum(c_q*f_q)/rho(phi)`，气相密度约为水的 1/800，使局部动量误差更敏感；异常负压和速度形成快速放大的反馈。

相关源码入口：

- `simulator.py::_collide_momentum_kernel`：碰撞与边界反射准备。
- `simulator.py::_prepare_momentum_cut_links`：移动冰壁反射及动量交换。
- `simulator.py::_stream`：动量流送。
- `simulator.py::_update_streamed_macroscopic_fields`：按局部相场密度恢复速度，首先观察到气隙速度跃变。
- `simulator.py::_update_pressure`：随后出现快速增长的负压。
- `simulator.py::_project_moving_body_inside_container`：当前容器接触主要作位置投影和质心法向速度约束；狭缝随冰块转动产生。

**尚未证明某一个反射公式是唯一根因。** 更换碰撞正则化、插值反射和声学松弛率均没有消除同类气隙失稳。要进一步区分气隙可解析性、壁面运动/接触一致性、压力模态和质量通量耦合，需要额外的针对性分析；本次按用户要求停止修复，未继续改变这些算法。

失败帧的水相最高速度约 1.12 m/s，而气隙出现约 1.04e4 m/s 的异常数值。相场仍在 [0,1]，冰水总质量相对误差约 8.34e-14。因此当前证据直接指向局部动量/压力失稳，而非可视化失败或总水质量丢失。不能由质量守恒推断速度场正确。

## 停止前的尝试

| 尝试 | 失稳步号 | 失稳时间 s | 位置 |
|---|---:|---:|---|
| 默认 CM-MRT + Tao，声学松弛率 0.25 | 5361 | 0.044675 | 第一块冰底部气隙 |
| CM-HRR，其他默认 | 3729 | 0.031075 | 第一块冰底部气隙 |
| BFL 插值反射 + CM-MRT | 5204 | 0.0433667 | 第一块冰底部气隙 |
| CM-MRT + Tao，声学松弛率 1 | 5389 | 0.0449083 | 第一块冰底部气隙 |

以上全部为失败的短运行，不作为稳定性改进结果。

## 已做的代码改动（未提交，保留工作区供检查）

1. 增加 `initial_water_geometry="dam_break"`，允许局部水柱；旧例仍默认水平水池。
2. 溃坝采用空气静水参考，并按预热后的实际水相逐列积分额外水压，保留挡板移除后的横向压差。水平水池初始化路径未改。
3. 添加热水溃坝四冰块 example、场景标签和对应元数据，复用原有四类场图/动画输出。
4. 增加可选择的 `cut_link_momentum_model="bfl"` 试验路径：碰撞后独立准备插值反射，缺失背后流体节点时使用移动半格反弹。默认仍为 Tao；BFL 探测也失败，不能视为修复。
5. 失败时保存 `failure.json` 与 `failure_state.npz`，用于重放和定位。
6. 添加 CPU 场景/调度测试和 CUDA 压力初始化测试。26 项 CPU 测试通过。CUDA 测试的初始压力、空气参考和零初速度断言通过；随后未经相场预热的锐界面耦合步骤曾因材料映射异常失败，不能记为 CUDA 测试通过。该测试之后调整为只推进流体来检查初始化，用户要求停止后没有再运行该调整版本。

BFL 公式参考：Bouzidi, Firdaouss, Lallemand, *Momentum transfer of a Boltzmann-lattice fluid with boundaries*, Phys. Fluids 13 (2001), [DOI:10.1063/1.1399290](https://doi.org/10.1063/1.1399290)。

## 诊断结果路径

根目录：`/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/dambreak_diagnostics/`。

- `initial_0p5s/run.log`、`history.csv`：首轮默认失败，历史采样止于 0.04 s。
- `baseline_replay/failure.json`、`failure_state.npz`：默认方案重放的失败现场。
- `hrr_0p5s/`、`bfl_0p5s/`、`acoustic1_0p5s/`：各失败候选的日志、短历史、失败现场。
- `operator_trace/diagnose.py`、`operator_trace.csv`、`summary.json`：只观测既有算子的复现脚本及记录。

目前没有完整 10 s 的速度、温度、相场动画结果。
