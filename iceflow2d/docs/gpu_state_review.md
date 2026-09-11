# GPU 状态与并行实现审查

本次审查覆盖计算内核、格子原语、配置、示例调度、输出与测试。`thermal.py` 已删除，热内核与流体/刚体内核统一放在 `simulator.py`；纯配置与单位换算移入 `config.py`，配置导入不加载 Taichi。热子组件仍明确区分材料网格与世界网格，并引用同一刚体位姿和覆盖率分配。

## 主要修改

- 删除均匀初始质量数组、材料固相率和冰水温度缓存、融化体积缓存、材料插值坐标六数组、旧覆盖率、重复覆盖率/SDF、静态墙数组、半力速度数组及派生诊断标量；世界温度、液相率、焓只在快照时由 CPU 合成。
- 初始材料质量通过 f64 kernel 参数初始化，避免 Taichi 默认 f32 常量先舍入后写入 f64，保证常量初始质量与实际状态一致。
- 两组 populations 使用按方向连续的存储布局；碰撞末尾准备 cut-link 反射并复用原本不再直通流送的槽，删除整张碰撞前非平衡数组。没有在 push streaming 中读取正在被其他线程覆盖的旧分布。
- 冰内导热直接从尚未改变的材料状态读取邻面，移除两张冰面通量数组；界面热请求与接收索引存于每个水线程局部，一次计算、限幅后原子散射。
- 开口测量使用 f64 warp 部分统计和第二级归约；其他热点标量先在 warp 内合并。遍历补齐到 128 线程块，补齐项用和/极值的恒等元，不访问越界状态。f64 shuffle 按两个 32 位字搬运，不降精度。
- 水量投影直接回传同一次归约的值与导数；熔化动量支撑合法性在 GPU 判定，减少逐标量往返。刚体活动判定移入 GPU 积分。
- 热对流 CFL 改为四个有效面**总流出速率**。例如中心速度为零、四邻格各向外 0.8 LU 时，中心总流出速率为 1.6；仅使用最大单元中心速度会漏掉这个正性限制。
- 输出只压缩写入一次 NPZ；快照批量生成派生场，并复用已归约的热总量计算平均水温。删除未使用的 `imageio` 依赖及线性浮力模型中无效的二次密度参数。
- 浮点配置保存验证后的数值；非法步数在修改状态前拒绝。`cd` 改为准确的 `smagorinsky_constant`。

## 保留状态的判据

廉价且读取时间层一致的量都按需计算。必须保留的情况包括独立守恒量、不可恢复的历史、跨 kernel 的只读输入、会被覆盖的旧时间层、非局部归约，以及经测量值得复用的材料到世界光栅。数学上可写成差的表达式，如果会破坏近均匀状态下的数值精度，也不能视为等价的替代。

局部寄存器中的平衡分布、矩、插值权重、热请求与 warp 部分和只活在当前线程/归约阶段，不保存为第二份持久物理状态。碰撞使用 `raw_moment_20` 等名称明确矩阶，格子原语采用 `lattice_direction`、`momentum_equilibrium` 等名称。

`_DerivedField` 是只读计算视图，不是 Taichi field，不分配持久显存。多个 Python 属性引用同一个 field 也只算一次分配。下表枚举默认 100×200 网格的全部唯一持久分配；只统计元素有效载荷，不含 Taichi 内存池、分配对齐、JIT、驱动或运行时临时区。

## 唯一 GPU field 清单

| 名称 | 形状 / 类型 | 有效字节 | 保留理由 |
|---|---|---:|---|
| `flow.momentum_populations` | 100×200×9 / f32 | 720000 | 演化分布含非平衡模态；密度、相场或速度不足以重建。 |
| `flow.momentum_stream_buffer` | 100×200×9 / f32 | 720000 | 碰撞后、流送前时间层；流送写当前分布时必须保持输入只读。相场缓冲的两个失效方向平面还供投影掩膜复用。 |
| `flow.phase_populations` | 100×200×9 / f32 | 720000 | 演化分布含非平衡模态；密度、相场或速度不足以重建。 |
| `flow.phase_stream_buffer` | 100×200×9 / f32 | 720000 | 碰撞后、流送前时间层；流送写当前分布时必须保持输入只读。相场缓冲的两个失效方向平面还供投影掩膜复用。 |
| `flow.water_phase` | 100×200 / f32 | 80000 | 冻结的相场供邻域梯度和投影读取；冰内另保存覆盖前的水相储备，不能从冰内分布求和恢复。 |
| `flow.momentum_velocity_lattice` | 100×200 / f32 | 160000 | 冻结的动量速度供邻域应变率、碰撞、refill 和密度投影使用；这些阶段会写 populations，不能边改分布边重新读取邻居的宏观速度。 |
| `flow.fluid_acceleration_lattice` | 100×200 / f32 | 160000 | 由相场二阶邻域、重力和速度梯度计算，跨流送保留半力修正所需旧时间层。 |
| `flow.pressure_lattice` | 100×200 / f32 | 80000 | 碰撞/启闭节点需要既定时间层的压力；冰内保存动态压力储备，不能由当前冰内 populations 恢复。 |
| `flow.reference_pressure_by_row` | 200 / f32 | 800 | 参考密度的垂直前缀积分；逐格重算需非局部求和，保留一维剖面。 |
| `flow.reference_density_by_row` | 200 / f32 | 800 | 初始化冻结的行平均密度，独立于随后演化的相场。 |
| `flow.solid_mask` | 100×200 / i8 | 20000 | 已提交的流体拓扑；接触候选光栅及热子步更新覆盖率时，此掩膜仍必须冻结。 |
| `flow.previous_solid_mask` | 100×200 / i8 | 20000 | 上一已提交拓扑，确定启闭单元和合法旧流体供体，防止并行 refill 互读新值。 |
| `flow.body_velocity` | 标量 / f32 | 8 | 刚体独立动力学/位姿状态，无法由其他当前状态推导。 |
| `flow.body_angle` | 标量 / f32 | 4 | 刚体独立动力学/位姿状态，无法由其他当前状态推导。 |
| `flow.body_angular_velocity` | 标量 / f32 | 4 | 刚体独立动力学/位姿状态，无法由其他当前状态推导。 |
| `flow.body_mass_lattice` | 标量 / f64 | 8 | 经过解析度阈值和单调性检查后接受的质量性质；材料网格求和非局部，且候选归约不一定被接受。 |
| `flow.body_inertia_lattice` | 标量 / f64 | 8 | 经过解析度阈值和单调性检查后接受的质量性质；材料网格求和非局部，且候选归约不一定被接受。 |
| `flow.body_local_center_of_mass` | 标量 / f64 | 16 | 经过解析度阈值和单调性检查后接受的质量性质；材料网格求和非局部，且候选归约不一定被接受。 |
| `flow.body_reference_origin` | 标量 / f32 | 8 | 刚体独立动力学/位姿状态，无法由其他当前状态推导。 |
| `flow.cumulative_melted_momentum_lattice` | 标量 / f64 | 16 | 历次侵蚀中刚体实际损失的动量历史；当前剩余质量/速度不能重建历史。 |
| `flow.cumulative_fluid_melt_carrier_momentum_lattice` | 标量 / f64 | 16 | 拓扑/refill/密度投影带来的累计流体动量变化，用于区分临时流体变化和后续修正。 |
| `flow.cumulative_fluid_melt_correction_momentum_lattice` | 标量 / f64 | 16 | 实际写入 populations 的累计修正动量，与 carrier 历史独立；两者相加才是总注入量。 |
| `flow.cumulative_melted_angular_momentum_lattice` | 标量 / f64 | 8 | 历次侵蚀中刚体实际损失的动量历史；当前剩余质量/速度不能重建历史。 |
| `flow.cumulative_fluid_melt_carrier_angular_momentum_lattice` | 标量 / f64 | 8 | 拓扑/refill/密度投影带来的累计流体动量变化，用于区分临时流体变化和后续修正。 |
| `flow.cumulative_fluid_melt_correction_angular_momentum_lattice` | 标量 / f64 | 8 | 实际写入 populations 的累计修正动量，与 carrier 历史独立；两者相加才是总注入量。 |
| `flow._body_reduced_mass` | 标量 / f64 | 8 | 当前材料质量、一次矩、原点二次矩的候选归约；接受新值前必须同时保留旧刚体质量性质。 |
| `flow._body_reduced_first_moment` | 标量 / f64 | 16 | 当前材料质量、一次矩、原点二次矩的候选归约；接受新值前必须同时保留旧刚体质量性质。 |
| `flow._body_reduced_inertia_origin` | 标量 / f64 | 8 | 当前材料质量、一次矩、原点二次矩的候选归约；接受新值前必须同时保留旧刚体质量性质。 |
| `flow._thermal_fluid_momentum_before` | 标量 / f64 | 16 | 热更新开始前的全域流体动量快照。 |
| `flow._thermal_fluid_momentum_provisional` | 标量 / f64 | 16 | 热/refill/相场投影后、动量修正前的独立时间层快照。 |
| `flow._thermal_fluid_momentum_after` | 标量 / f64 | 16 | 每次实际修正后的独立归约，测量 f32 写回误差并驱动余量修正。 |
| `flow._thermal_body_momentum_before` | 标量 / f64 | 16 | 慢热开始时的累计刚体损失基线；与结束时历史相减得到本次实际损失。 |
| `flow._thermal_fluid_angular_momentum_before` | 标量 / f64 | 8 | 热更新开始前的全域流体动量快照。 |
| `flow._thermal_fluid_angular_momentum_provisional` | 标量 / f64 | 8 | 热/refill/相场投影后、动量修正前的独立时间层快照。 |
| `flow._thermal_fluid_angular_momentum_after` | 标量 / f64 | 8 | 每次实际修正后的独立归约，测量 f32 写回误差并驱动余量修正。 |
| `flow._thermal_body_angular_momentum_before` | 标量 / f64 | 8 | 慢热开始时的累计刚体损失基线；与结束时历史相减得到本次实际损失。 |
| `flow._melt_momentum_wet_weight` | 标量 / f64 | 8 | 全湿支撑权重总和，需全域归约。 |
| `flow._melt_momentum_wet_target` | 标量 / i32 | 4 | 两个不同湿单元的编码极值，用于构造闭合角动量的局部力偶。 |
| `flow._melt_momentum_wet_target_max` | 标量 / i32 | 4 | 两个不同湿单元的编码极值，用于构造闭合角动量的局部力偶。 |
| `flow._melt_momentum_weight_first_moment` | 标量 / f64 | 16 | 全湿支撑一次、二次空间矩，需全域归约；归一化质心及极惯量已改为派生视图。 |
| `flow._melt_momentum_weight_second_moment` | 标量 / f64 | 8 | 全湿支撑一次、二次空间矩，需全域归约；归一化质心及极惯量已改为派生视图。 |
| `flow._body_contact_support_extrema` | 4 / f32 | 16 | 未裁墙候选锐界的四个空间极值，需几何归约。 |
| `flow._body_contact_projection_changed` | 标量 / i8 | 1 | 本次接触是否更改位置的事务结果；已提交位姿不能恢复该历史判定。 |
| `flow.hydrodynamic_impulse` | 标量 / f64 | 16 | 所有 cut-link 的合冲量/力矩，碰撞边界准备累加，刚体积分消费并清零。 |
| `flow.hydrodynamic_torque` | 标量 / f64 | 8 | 所有 cut-link 的合冲量/力矩，碰撞边界准备累加，刚体积分消费并清零。 |
| `flow.water_volume_current` | 标量 / f64 | 8 | 当前裁剪/候选投影的全域体积归约；不是单格水相可直接变换的量。 |
| `flow.water_projection_derivative` | 标量 / f64 | 8 | 当前试探 logit 位移的全域导数/自由集权重，供求根和末位闭合复用。 |
| `flow._moving_body_fraction` | 100×200 / f32 | 80000 | 唯一材料到世界的覆盖率光栅，与 thermal.world_body_solid_fraction 是同一对象；复用旋转和四点材料采样结果，避免在接触、锐界和多个热面重复采样。 |
| `flow.maximum_water_outflow_rate_lattice` | 标量 / f32 | 4 | 合法水单元各面总流出速率的全域最大值，控制正性 CFL。 |
| `flow.maximum_fluid_speed_l1` | 标量 / f32 | 4 | 可选稳定性检查的独立全域最大速度/非法状态归约。 |
| `flow.thermal_state_invalid` | 标量 / i32 | 4 | 可选稳定性检查的独立全域最大速度/非法状态归约。 |
| `flow.phase_change_initial_water_volume` | 标量 / f64 | 8 | 冻结的初始水量，当前相场/热水体积不能还原。 |
| `thermal.body_solid_mass` | 32×32 / f64 | 8192 | 材料网格两项独立守恒状态：剩余质量和相对熔点显热。 |
| `thermal.body_sensible_energy` | 32×32 / f64 | 8192 | 材料网格两项独立守恒状态：剩余质量和相对熔点显热。 |
| `thermal._body_heat_delta` | 32×32 / f64 | 8192 | 导热及多个水线程散射的热量累加缓冲；相变在下一 kernel 消费，不能原地更新冰温。 |
| `thermal._body_melt_target` | 32×32 / i32 | 4096 | 本次热交换实际关联水单元的编码，跨相变保留以匹配融水注入位置。 |
| `thermal._body_melt_mass_step` | 32×32 / f64 | 8192 | 刚完成的相变产生的质量与剩余显热；新的材料状态已不包含这些脱离量。 |
| `thermal._body_melt_sensible_step` | 32×32 / f64 | 8192 | 刚完成的相变产生的质量与剩余显热；新的材料状态已不包含这些脱离量。 |
| `thermal.water_volume_m2` | 100×200 / f64 | 160000 | 世界网格独立的广延水量和显热；对流、熔水注入与开口同步之间不能用瞬时相场替代水量。 |
| `thermal.water_sensible_energy` | 100×200 / f64 | 160000 | 世界网格独立的广延水量和显热；对流、熔水注入与开口同步之间不能用瞬时相场替代水量。 |
| `thermal._water_energy_flux_x` | 101×200 / f64 | 161600 | 冻结旧水状态计算的共享面通量。两侧更新会覆盖迎风水量与显热，届时不能安全地从一个面通量和正在改变的源状态推导另一个。 |
| `thermal._water_energy_flux_y` | 100×201 / f64 | 160800 | 冻结旧水状态计算的共享面通量。两侧更新会覆盖迎风水量与显热，届时不能安全地从一个面通量和正在改变的源状态推导另一个。 |
| `thermal._water_volume_flux_x` | 101×200 / f64 | 161600 | 冻结旧水状态计算的共享面通量。两侧更新会覆盖迎风水量与显热，届时不能安全地从一个面通量和正在改变的源状态推导另一个。 |
| `thermal._water_volume_flux_y` | 100×201 / f64 | 160800 | 冻结旧水状态计算的共享面通量。两侧更新会覆盖迎风水量与显热，届时不能安全地从一个面通量和正在改变的源状态推导另一个。 |
| `thermal._phase_target_energy` | 100×200 / f64 | 160000 | 开口重构的新显热缓冲；保留旧水能量供邻域外推读取，避免新旧重构混读。 |
| `thermal._aperture_reduction_partials` | 628×6 / f64 | 30144 | 每个 CUDA warp 的六项独立部分统计，供第二级归约；用小型临时数组减少全域标量原子竞争，无完整状态副本。 |
| `thermal._phase_volume_before` | 标量 / f64 | 8 | 最近一次开口同步前的独立总量快照。 |
| `thermal._phase_energy_before` | 标量 / f64 | 8 | 最近一次开口同步前的独立总量快照。 |
| `thermal._phase_base_capacity` | 标量 / f64 | 8 | 分别归约相场加权开口和全湿格容量；一个不能确定另一个。 |
| `thermal._phase_full_capacity` | 标量 / f64 | 8 | 分别归约相场加权开口和全湿格容量；一个不能确定另一个。 |
| `thermal._phase_specific_energy_min` | 标量 / f64 | 8 | 同步前湿水比显热的两个独立极值，限制新温度范围。 |
| `thermal._phase_specific_energy_max` | 标量 / f64 | 8 | 同步前湿水比显热的两个独立极值，限制新温度范围。 |
| `thermal._phase_reconstructed_energy` | 标量 / f64 | 8 | 新目标显热的全域和，用于求实际需要的总能量修正。 |
| `thermal._phase_cooling_capacity` | 标量 / f64 | 8 | 逐格非负差值的稳定求和。用全域大数相减会在近均匀温度下损失修正所需有效位，因此不替换成不等价的抵消表达式。 |
| `thermal._phase_heating_capacity` | 标量 / f64 | 8 | 逐格非负差值的稳定求和。用全域大数相减会在近均匀温度下损失修正所需有效位，因此不替换成不等价的抵消表达式。 |
| `thermal.aperture_energy_correction_abs_j_m` | 标量 / f64 | 8 | 每次非局部能量校正绝对值的累计历史，不能由当前总焓误差恢复。 |
| `thermal._phase_volume_after` | 标量 / f64 | 8 | 最近一次同步后实际总量归约；与同步前快照共同构成守恒审计。 |
| `thermal._phase_energy_after` | 标量 / f64 | 8 | 最近一次同步后实际总量归约；与同步前快照共同构成守恒审计。 |
| `thermal._melt_injection_eligible` | 100×200 / i8 | 20000 | 散射前冻结的注入候选集；避免一个线程新增水量改变另一个线程的目标选择。 |
| `thermal.boundary_power` | 标量 / f64 | 8 | 当前导热步的净外边界功率，跨通量计算和能量更新保留。 |
| `thermal.boundary_heat_input` | 标量 / f64 | 8 | 累计外边界输入热量，当前功率不能恢复其时间积分。 |
| `thermal._unassigned_melt_mass` | 标量 / f64 | 8 | 失去局部接收格的本步融水质量/显热池；注入后清空，体积直接由质量求得。 |
| `thermal._unassigned_melt_energy` | 标量 / f64 | 8 | 失去局部接收格的本步融水质量/显热池；注入后清空，体积直接由质量求得。 |
| `thermal._melt_fallback_free_weight` | 标量 / f64 | 8 | 回退接收集合的空余容量和现有水量两个独立归约。 |
| `thermal._melt_fallback_wet_weight` | 标量 / f64 | 8 | 回退接收集合的空余容量和现有水量两个独立归约。 |
| `thermal._interval_body_melt_mass` | 标量 / f64 | 8 | 本热更新内实际脱离冰与实际进入水的两个独立和；相减审计注入守恒，不能预设相等。 |
| `thermal._interval_water_melt_mass` | 标量 / f64 | 8 | 本热更新内实际脱离冰与实际进入水的两个独立和；相减审计注入守恒，不能预设相等。 |
| `thermal._solid_body_mass_sum` | 标量 / f64 | 8 | 输出时才执行的材料/水独立广延量全域归约，避免每步进行诊断。 |
| `thermal._water_mass_sum` | 标量 / f64 | 8 | 输出时才执行的材料/水独立广延量全域归约，避免每步进行诊断。 |
| `thermal._body_sensible_sum` | 标量 / f64 | 8 | 输出时才执行的材料/水独立广延量全域归约，避免每步进行诊断。 |
| `thermal._water_sensible_sum` | 标量 / f64 | 8 | 输出时才执行的材料/水独立广延量全域归约，避免每步进行诊断。 |

## 验证与实测

CPU 配置/输出测试 27 项、CUDA 数值与状态测试 22 项。覆盖绝热能量与质量守恒、受限热量融化、局部温度重构、相场有界投影、cut-link 反射、刚体质量/接触、融水线/角动量闭合、面流出 CFL、派生视图与 GPU 读值一致性。

CUDA Compute Sanitizer `memcheck` 检查整个 CUDA 测试集。Taichi 会用 `cuPointerGetAttribute` 探测 NumPy 主机指针，预期返回 `CUDA_ERROR_INVALID_VALUE`；已核对未经筛选的报告全部是这类探测。正式检查用 `--report-api-errors no` 排除这些 API 返回，保留设备内存越界和未对齐访问检测。该检查不是全局内存数据竞争的形式化证明；跨 kernel 的只读/写入边界也在上表逐项说明。

默认物理工况另外运行 4800 LBM 步至 0.03 s，经历入水并开始融化；CSV、metadata、54 个 NPZ 数组、三组各 4 帧 PNG/GIF 均生成。所有输出数组有限，温度在 0–90 °C；总质量残差约 1.3e-13 kg/m，总热能残差约 5.1e-8 J/m。未把这段短程验证当作完整 3 s 工况的长期稳定性或网格收敛证明。

同一 RTX 5090、Taichi 1.7.4，预热 32 步后分三批各测 128 步，在 CPU 计时前后同步 CUDA；不计初始化、JIT 和输出。原始摘要保存在 [review_results.json](../../benchmarks/review_results.json)。

| 网格 | 原每步中位数 | 修改后 | 耗时减少 | field 有效载荷：原 → 新 |
|---|---:|---:|---:|---:|
| 100×200 | 2.056 ms | 1.582 ms | 23.1% | 7,556,255 → 4,702,165 B |
| 200×400 | 3.424 ms | 2.292 ms | 33.1% | 30,211,903 → 18,793,589 B |

计数为 148 → 91 个唯一持久 field。小型 warp 部分统计缓冲是新增的独立归约工作区，已计入以上数字。耗时是当前机器上仿真初段的结果，会随分辨率、入水阶段、子步数和 GPU 负载改变。

## Python API 变更

`IceFlow2D` 的包级导入和 `step()` 保留。配置从 `iceflow2d.config` 或包级导入；原 `iceflow2d.thermal` 模块路径已删除。主要字段变更如下，CSV/NPZ 中已有的输出键保留。

| 原名称 | 当前名称 |
|---|---|
| `simulation.cfg` | `simulation.config` |
| `f` / `f_post` | `momentum_populations` / `momentum_stream_buffer` |
| `h` / `h_post` | `phase_populations` / `phase_stream_buffer` |
| `phi` / `u` / `p` | `water_phase` / `momentum_velocity_lattice` / `pressure_lattice` |
| `fluid_force` | `fluid_acceleration_lattice`（原来存的就是加速度） |
| `wall` / `solid` / `solid_prev` | `wall_mask` / `solid_mask` / `previous_solid_mask` |
| `sdf` / `thermal_advection_velocity` | `body_signed_distance_m` / `physical_velocity_lattice`（计算视图） |
| `IceFlowConfig.cd` | `IceFlowConfig.smagorinsky_constant` |

材料固相率和温度不再接受 `from_numpy()` 写入，应更新 `body_solid_mass`、`body_sensible_energy` 或世界水广延量。输出场优先调用 `sample_thermal_fields()` 一次取齐；单个输出视图的 `to_numpy()` 仍可用于主机读取。静水参考二维视图的实际存储是两条一维剖面。

## 复现命令

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s iceflow2d -p 'test_*.py' -v
compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 99 \
  python -m unittest discover -s iceflow2d -p 'test_*.py' -v
python benchmarks/benchmark_solver.py --output /tmp/iceflow-benchmark.json
python benchmarks/benchmark_solver.py --resolution-x 200 --resolution-y 400
```
