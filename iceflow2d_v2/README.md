# iceflow2d_v2

按 [实现方案](implementation_plan_zh.md) 新建的 **Taichi 实验性耦合求解器**。已修复细网格长跑暴露的零流体体积格点误分类；73 项完整回归（含原生 CUDA）和故障批次回放通过，12 组短程基准完成。六组正式算例已从初态完成 10 s，累计 1280 万步，各通过 23 项数值检查、零重试。网格、时间步和热块冰量曲线差异仍显著，完整物理精度尚未通过验收。实测结果、已知问题见 [实现状态](validation/implementation_status_zh.md)。LB、热学和守恒状态采用补偿 f32；几何、位姿和容量 Krylov 求解单独使用 f64，多重网格仍为 f32。

保持用户选择的固定密度数值空气：只严格核算冰＋水质量；空气密度仍为 1.25 kg/m³，体积收缩引起的已知空气增量和额外残差分别输出。没有增加气体 EOS，也没有把空气误差抵消到液水量中。

[完整 10 s 报告](outputs/zero_capacity_mask_comparison/report_zh.md) 汇总六组数值结果；[A–H 验收对照表](validation/acceptance_coverage_zh.md) 区分这些证据、实际误差与未验证范围。六组质量/焓相对残差峰值不超过 2.427×10⁻⁹ / 2.227×10⁻⁹。

当前额外的 [黏度匹配粗网格对照](outputs/zero_capacity_mask_matched_nu_100/acceptance/report_zh.md) 从初态完成 10 s，通过 23 项数值检查、零重试；质量/焓残差峰值为 8.726×10⁻¹⁰ / 7.787×10⁻¹⁰，残冰自然消失且最终刚体动量为零。它与细网格具有相同的黏性下限物理尺度；[两档完整对照](outputs/zero_capacity_mask_matched_nu_comparison/report_zh.md) 的冰量差 RMS 仍为初始的 10.260%，不能据此认定空间收敛。这个额外算例不计入正式六组矩阵。

## 运行

从仓库根目录运行。依赖沿用根目录 `requirements.txt`；测试另需 `pytest`。调用库时由调用者执行 `ti.init(..., default_fp=ti.f32, fast_math=False)`。

```bash
python -m pytest -q iceflow2d_v2/tests
python -m iceflow2d_v2.examples.coupled_falling_ice_melting_2d \
  --arch cuda --steps 32 --output iceflow2d_v2/outputs/smoke
```

CPU 可用 `--arch cpu`；测试可用 `ICEFLOW_TEST_ARCH=cpu python -m pytest ...`。

示例默认在终端显示单行进度条，包含完成百分比、已接受的 LBM 步数、物理时间、耗时和 ETA；初始化和图编译期间另有提示。断点续跑从已有步数显示进度，ETA 按本次运行的速度估计。进度条写入标准错误流；`--no-progress` 可切回逐批 JSON 日志，完整诊断始终写入 `diagnostics.csv`。

默认场景是 100×200、10 s、0.01 s 采样配置。当前这一档从 0 s 完成 160 万步，零重试，通过 23 项数值长跑检查及独立检查点审计，见 [当前 10 s 结果](outputs/zero_capacity_mask_100/acceptance/report_zh.md)。末时剩余冰量为初始的 0.3266%，没有人为删除残冰。六组最终状态见 [完成记录](outputs/zero_capacity_mask_comparison/progress.json)；细网格的冰在 6.945–6.950 s 采样区间消失，最终几何质量、惯量和刚体动量归零。`--steps` 用于短程调试；`--end-time` 用于指定物理时间。`--nx 200` 对应 200×400，并保持物理壁厚。`--reference-velocity 8` 将物理时间步减半，所有单位换算和热参数同步重算。

诊断触发时返回退出码 2，并回滚到上一接受批次。主运行入口输出配置与本次源文件哈希、诊断 CSV、`result.json` 和温度场、速度场 PNG，正常运行和失败时均不写任何 NPZ 文件。`--resume` 仍可读取配置、代码哈希和全部字段一致的已有完整重启文件，但不再生成新的磁盘检查点；批次回滚仍使用内存快照。`result.json` 的 `validated_full_simulation` 始终为 false，直至完整验收完成。

与 `iceflow2d` 一样，默认每 **0.01 s 物理时间**直接保存一对图片：`temperature_frames/frame_00000.png` 为水和冰的温度场（°C），`velocity_frames/frame_00000.png` 为水速度大小（m/s）及方向箭头。温度色标固定为初始冰温至初始水温，速度色标默认 0–0.1 m/s，可用 `--velocity-visualization-max-m-s` 调整。`frames.csv` 逐帧记录步数、实际物理时间及图片路径。图片逐帧写盘，只保留当前帧数据。

新运行包含 0 s 初态，续跑包含实际起点，并始终保存终态；不足 0.01 s 的末段也会保存，失败时保存最后接受状态。`--output-interval`（别名 `--output-interval-s`）可调整图片间隔，`--sample-interval` 独立控制诊断批次的最大时间间隔，默认也是 0.01 s。输出时刻逐个映射到不早于目标时间的 LBM 步，避免非整数步长累积偏移。原 `--checkpoint-interval` 已移除。

当前 [100×200 动画](outputs/zero_capacity_mask_comparison/full_10s_100.mp4) 和 [200×400 动画](outputs/zero_capacity_mask_comparison/full_10s_200.mp4) 各使用正式长跑 0–10 s 的 101 帧采样、两倍速播放；帧哈希、完整解码和画面检查均通过，见 [核对记录](outputs/zero_capacity_mask_comparison/artifact_review.json)。另有 0–0.1 s、每 1 ms 实算采样的 [历史入水慢放](outputs/geometric_phase_release_detail_100/release_detail.mp4)，仅用于查看前版入水过程。

每步均执行设备端容量、相界限、有限性和 Mach 检查，默认每至多 1600 步同步到主机验收。`--graph --projection-solver multigrid` 使用 Taichi 1.7.4 图调度和保持原算子的 Galerkin 多重网格。`--profile` 输出实测核耗时。不要把短程 MLUPS 外推成已经完成的 10 s 性能结果。

可选 `--native-graph` 需要 CUDA 12.8/CUPTI 和 g++；它记录已编译的 Taichi 核调用，验证参数布局及驱动操作后建立 CUDA 图。所有网格计算仍由原 Taichi 核执行，记录阶段完整回滚。遇到不支持的驱动调用显式失败；可改用 `--graph`。适配器源码也纳入重启哈希。启用对应回归使用 `ICEFLOW_NATIVE_TESTS=1`。

```bash
python -m iceflow2d_v2.examples.coupled_falling_ice_melting_2d \
  --nx 100 --end-time 10 --native-graph --projection-solver multigrid \
  --pressure-iterations 1200 --output iceflow2d_v2/outputs/run_100
```

上述命令仍执行实验性求解器，完成指定步数与通过完整科学验收分别报告。运行和测试期间不要修改其正在使用的源码；Taichi 在首次调用时读取 Python 函数源码。长测试应从记录哈希的独立副本运行。

已有长跑可用独立后处理脚本检查，不导入求解器或修改数据。下面的检查点审计及旧版 `render_snapshot` / `render_movie` 针对已有 NPZ 数据；新的主运行入口只生成上述图片和文本记录。独立的 `validation/`、`derivation/` 验证工具保留其专用数据输出。第一个算例为轨迹比较的参考；报告保留源文件哈希、每步守卫峰值、空气来源和时间步/热块差异。

报告可通过 `--source-manifest` 校验冻结源码清单，通过 `--process-start-times` 读取实际进程起始时间。完成后还会用 NumPy 从完整检查点独立积分质量和焓，并与最终 CSV 对照；冰完全消失时额外检查几何质量、惯量和刚体动量均为零。接受演化的累计计时不含编译、输出及回滚开销，进程实际耗时另列。两者都不能把共享 GPU 的并发运行当成独占性能。

```bash
python -m iceflow2d_v2.validation.report_full_scene \
  iceflow2d_v2/outputs/run_100 iceflow2d_v2/outputs/run_200 \
  --output iceflow2d_v2/outputs/comparison
python -m iceflow2d_v2.validation.planar_stefan \
  --refinement 1 --end-time .1 --native-graph \
  --output iceflow2d_v2/outputs/planar_stefan
python -m iceflow2d_v2.validation.audit_restart_inventory \
  iceflow2d_v2/outputs/run_100/restart.npz \
  --output iceflow2d_v2/outputs/run_100/independent_inventory.json
```

平面 Stefan 算例保留生产算子，只允许顶面融化；截面流量直接从已接受的输运边通量计算，用来独立检查密度差引起的液侧收缩速度。它使用 100 倍水导热率以缩短验证时间，不能当作原始物理场景。`audit_boundary_instability` 的显式跨版本状态回放仅用于定位缺陷，不可作为生产重启或完整长跑证据。

## 模块和归属

| 文件 | 内容 |
|---|---|
| `config.py` | 独立场景、单位换算、显式拒绝未支持配置 |
| `lattice.py`, `flow.py` | 白名单 D2Q9 原点矩/中心矩、力和梯度、push streaming、SNIBB |
| `polygon.py`, `geometry.py` | 新 PLIC 轮廓、多边形面积/惯量、世界交叠、实际开放面长度；白名单 SDF 查询 |
| `state.py` | 唯一的广延相状态和分布矩恢复 |
| `thermal.py` | 各材料独立 D2Q5、共同物理热量、切割接触面积、导热有界限制 |
| `coupling.py` | 单份接受的融化交换、Stefan 速度、主气区数值空气源及焓账本 |
| `transport.py` | 实际 post-collision 相场链路、成对限制、容量投影、隐式小容量输运 |
| `agglomeration.py`, `connectivity.py` | 按开放面连通的局部切割区域合并、共用广延量权重、保留线/角动量的最小动能混合 |
| `momentum.py`, `rigid.py` | 新共同通量动量修正、固定材料参考点、变质量/惯量、半隐式刚体候选与接触 |
| `phase_momentum.py` | 实际退缩区域的线/角动量积分，共同隐式刚体速度，完全消失时转移全部刚体动量 |
| `subgrid_viscous.py` | 无 SNIBB 链路时，残冰与连通局部流体的守恒、耗散隐式黏性耦合 |
| `diagnostics.py`, `simulation.py` | 一阶时间层、外力账本、守恒/几何/Mach 检查 |
| `multigrid.py`, `reduction.py`, `schedule.py` | GPU 投影加速、分层补偿归约、版本固定的图调度 |
| `cuda_schedule.py`, `cuda_schedule.cpp` | 可选 CUDA 调度适配器，无新的网格演化算法 |
| `checkpoint.py`, `health.py` | 完整重启、失败批次回滚、逐步设备端检查 |

质量、显热、动量的几何转移共用权重。液态潜热由同一水质量携带；D2Q5 的静止分布包含液态参考焓，移动分布只由温度决定。没有逐点裁剪相场来补守恒，也没有全域补水/能量归一化。

刚体位姿通过质心平移和固定材料参考点绕质心转动更新；转动后刷新世界一阶矩。相变动量按实际旧、新多边形之差积分，共同隐式速度同时用于剩余冰和流出材料；完全融化时剩余质量矩阵为零，刚体动量自然归零。面记录舍入差另以 `phase_momentum_roundoff_x/y`、`phase_angular_momentum_roundoff` 输出，不作为外力或全域修正。动量闭合、末期反例与验证范围见 [推导](research/phase_momentum_zh.md)。

## 当前限制

平滑 SDF 的符号不等于精确多边形流体体积。完全被冰覆盖的格点现在强制分类为固体，避免凭空形成气液界面；正体积薄片仍完整保留，SDF 查询和 SNIBB 公式不变，见 [零体积格点反例](research/zero_capacity_mask_zh.md)。

无固壁链路时的新增算子在实际表面上积分黏性阻力，与附近连通流体共同隐式求解，成对保持线/角动量并记录动能损失。它沿用当前有效黏度，不删冰、不添加质量或惯量下限；空冰或已有 SNIBB 链路时不交换动量。该子网格近似的空间阻力精度仍待验证，见 [离散与耗散证明](research/subgrid_viscous_zh.md)。

- 生产热核固定 `tau=1`，水气和冰气绝热；一般 MRT 公式保留在 `lattice.py`。体内导热支持 1/2/4/8 步热块，界面热量、融化质量及几何每步更新；Fourier 衰减已检验，完整 10 s 热块对照已完成但精度未通过。冰气热接触、初始水过冷/再冻结明确拒绝。初始相场执行 500 步无速度预热；守恒基准在预热和局部合并之后测量。
- 新压力连续源已通过源矩及单步线性膨胀场的局部 CE 残差阶检查；闭合容器中的持续相变源与压力场收敛仍未验证。
- 两档完整 10 s 的稳定性守卫通过；附加质量、接触线牵引及长时角动量精度仍未通过完整验收。
- 当前细网格同配置、同初始物理状态的 0–0.4 s 复跑存在差异：冰量差 RMS 为初始冰量的 0.01606%，转角末时差 0.04855 rad。见 [重复性量化](outputs/zero_capacity_mask_repeatability/report_zh.md)；不能把不同网格或热块轨迹的全部差异直接归于参数，也不能用一次复跑给出统计误差界。
- 容量求解失败可回滚整批并增加压力迭代重试；细分几何/相变子步尚未接入。这种线性求解重试不等于自适应物理子步。
- 机械能只输出动能、重力势能及冲量；黏性耗散、表面能、接触损失、数值体积源做功尚未组成完整机械能余额。热焓守恒不等于全系统总能量守恒。无正质量单元的动量舍入残差仍计入全局动量，另以 `zero_mass_momentum_l1_kg_s` 报告，不把它除以人为质量下限计算动能。
- 图执行与 Python 顺序执行有独立一致性测试。稀疏界面列表和完整显存带宽分析尚待完成；常规网格工作已在 Taichi 中，无逐步 `to_numpy()`。

状态格式 2 中空气热量存为 `Ea - Ca*A*Tair_initial`，温度恢复时加回参考温度；输出物理焓和数值空气焓源仍使用同一绝对参考。这避免微小气相消失时损失整个热库存造成假温变。

旧代码参考严格遵循 [白名单](research/reference_scope.md)。生产模块不导入旧包；测试也不运行旧模拟器。[离散约定](research/coupling_discretization_zh.md) 区分公式成立、局部测试通过和完整物理验证。
