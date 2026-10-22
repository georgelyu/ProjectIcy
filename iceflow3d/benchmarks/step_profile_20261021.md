# IceFlow3D step profiling — 2026-10-21

本报告记录切换默认值之前的 **CM-HRR** 测量。当前默认已改为 `cm_mrt`，
新测量和对照见 [CM-MRT profiling 报告](step_profile_cm_mrt_20261021.md)。
下方历史复现命令对应当时源码；在当前源码上重测请使用新报告中的显式模型及检查点选项。

本次测量针对当前工作树中的 `IceFlow3D.step`。**450 网格主要受 GPU 核执行限制；冰运动后，几何更新中的全局水量投影成为最大单项热点。** 小网格更受归约和主机调度影响，不能直接用小网格热点指导默认大网格优化。

生产求解器及数值配置未修改。新增可复现 profiling 入口、两个隔离实验和本报告。原始数据在 `iceflow3d/outputs/profiling_20261021/`，精简汇总见 [JSON](step_profile_20261021_summary.json)。

![Profile overview](../outputs/profiling_20261021/step_profile.png)

## 实测基线

| 场景 | 网格 | 未插桩 step 中位数 | 三次测量范围 | CUDA 核累计时间 |
| --- | --- | ---: | ---: | ---: |
| 初始阶段，默认 5 格界面 | 150×50×30 | **5.167 ms** | 5.160–5.184 ms | 3.835 ms |
| 初始阶段，默认 5 格界面 | 450×150×90 | **45.999 ms** | 45.675–46.106 ms | 43.368 ms |
| 四冰块运动，0.1 s 完整检查点，5 格界面 | 450×150×90 | **57.734 ms** | 57.599–58.380 ms | 55.182 ms |
| 冰全部融化，5 s 完整检查点，10 格界面 | 450×150×90 | **46.968 ms** | 46.925–47.243 ms | 未单独测量 |

默认 450 网格对应 6,075,000 个格点、物理时间步 1/120000 s。57.734 ms/step 对应约 17.3 step/s、105.2 MLUPS。若整个过程维持该短窗口速度，5 秒物理时间的纯 step 约需 9.62 小时；这只是换算，不是整场运行预测，状态演化和输出开销会改变实际总时长。

GPU 时间与墙钟基线来自独立运行，不相加。Nsight 的同一 16 步窗口进一步测得 **57.922 ms/step 墙钟、55.957 ms/step GPU 核时间**，GPU 核执行占约 **96.6%**，支持“大网格优先优化设备端”的结论。

## 如何测量

- RTX 3090 24 GiB，驱动 570.211.01；Intel i9-14900K；Python 3.13.9、Taichi 1.7.4、NumPy 1.26.4。使用现有 `/home/georgelyu/anaconda3/bin/python`。仓库 `.venv` 没有 Taichi。
- 使用完整四冰块、CM-HRR、热输运、融化、接触、相量守恒及逐步稳定性检查。热传导周期保持每 8 个 LBM 步一次。
- 新建算例保留 2000 步相场预热；随后推进 32 个耦合步。再预执行整个测量窗口，使窗口内条件分支完成 JIT，然后恢复完整内存检查点。
- 每次测量均从同一完整状态恢复，包括所有设备字段、刚体数组和 Python 步数/热累计状态。每组重复 3 次、每次 128 步，交替测量原始 step 和带同步的分层计时。
- 大网格初始窗口为 step 32→160；运动窗口为 12032→12160，约 0.100267–0.101333 s；融尽窗口为 600032→600160。小网格窗口为 step 32→160，0.0008–0.004 s。
- 初始化、相场预热、JIT、检查点传输、诊断和写文件均在计时区间外；每个区间末尾 `ti.sync()`。
- 分层计时相对原始 step 的中位数开销：小网格 10.87%，大网格初始 0.95%，运动态 1.60%，融尽态 1.02%。因此吞吐使用未插桩结果，设备占比使用独立 CUDA event 结果。
- 两个历史检查点与当前求解器源码 SHA-256 完全匹配；各组结束再次校验源码未变化。5 s 检查点采用 10 格界面，应与其他场景区分，不能当成严格单因素对照。
- 所有被测窗口都通过原有逐步保护；采样末态相量相对漂移绝对值不超过 5.1×10^-8。这里不把短窗口试验当作长期数值验收。

CUDA 事件计时口径依据 [Taichi profiler 文档](https://docs.taichi-lang.org/docs/master/profiler)，运行时记录接口按本机 1.7.4 实现读取。旧 `profile_solver.py` 同时开启 cProfile 和 GPU profiler、相场预热为零，本报告没有把它的墙钟值混入正式基线。

## 运动场景的耗时分解

下表以 0.1 s 检查点为主要依据；GPU 占比的分母为 55.182 ms。

| 核/核组 | ms/step | GPU 占比 | 实际工作 |
| --- | ---: | ---: | --- |
| `_project_phase_volume` | **8.814** | **15.97%** | 每步平均 1.094 次，全域相量校正并修改 27 个 h 和 27 个 f 分布 |
| `_collide_momentum` | 6.027 | 10.92% | D3Q27 CM-HRR 碰撞 |
| `_forces` | 5.985 | 10.85% | 每步平均 2.125 次，包括几何更新后的无通量版本 |
| `_stream` | 5.704 | 10.34% | 两组 D3Q27 分布的迁移/反射 |
| `_prepare_boundaries` | 5.097 | 9.24% | 全域遍历，识别冰邻接链并计算反射与反力 |
| `_gradients` | 4.287 | 7.77% | 每步平均 **3.125 次**全域梯度/毛细应力计算 |
| `_measure_outflow` | 4.178 | 7.57% | 六面热对流出流率和全局最大值 |
| `_measure_phase_balance` | 2.557 | 4.63% | 每步平均 **2.219 次**水量/界面容量统计 |
| `_collide_phase` | 2.349 | 4.26% | 相场碰撞 |
| `_commit_water_targets` | 1.557 | 2.82% | 每步平均 2.125 次水热场重映射及能量记账 |
| `_recover_phase` | 1.396 | 2.53% | 相场恢复、限幅和双精度累计记账 |

`launches_per_step` 是 CUDA task 次数，不等于 Python 方法调用次数；例如 `_measure_phase_balance` 每次调用含初始化和归约两个 task。

分层墙钟计时中，普通 `_refresh_geometry` 为 **17.54 ms/step**，含投影、统计、补点、梯度、力和水热场同步；热传导路径还会每 8 步额外调用一次几何更新。该 17.54 ms 已包含对应子核，不能再与上表相加。热对流阶段 7.73 ms，刚体推进约 1.03 ms。

起始阶段投影只约每 128 步发生一次；运动态接近每步一次。与初始态相比，运动态墙钟增加约 11.74 ms，主要来自投影及其额外统计/补点。这解释了为何只测前几步会漏掉最重要的热点。

源码定位：`simulator.py:1122` step；`:1028` 几何刷新；`:999` 水量守恒循环；`:967` 全域分布投影；`:737` 冰边界；`:356` 梯度；`thermal.py:187` 出流统计；`rigid.py:136` 刚体状态上传。

## 根因证据

### 1. 局部冰边界工作被实现为全域大核

0.1 s 检查点有 **5,267,284** 个流体格点，其中仅 **23,550（0.447%）** 在 D3Q27 邻域内邻接冰。`_prepare_boundaries` 却为全部流体格点加载 27 项 `f_post`，再检查 27 个方向。

5 s 检查点的冰格点、冰邻接流体格点均为零，该核仍执行，并在分层计时中消耗 **4.955 ms/step**。这直接确认了全域扫描及无冰状态的冗余成本。

### 2. 全局水量约束带来额外的 D3Q27 读写

几何重映射后，通过全域测量、CPU 读取误差/容量、全域分布校正、再次测量来维持水库存。运动态仅投影与统计就用掉 **11.37 ms/step、20.61% GPU 时间**。不能直接关闭相量守恒获得速度；应该保持校正语义，减少它的扫描、同步以及分布读写。

### 3. 梯度重复计算可消除一遍

上一耦合步末次 `_refresh_geometry` 已更新相场梯度和毛细应力。之后热对流只改水热字段；若发生融化，传导路径也会再次刷新几何和梯度。因此正常连续 step 的下一步入口仍可复用该结果。

隔离原型在不改生产源码的情况下跳过每步入口的重复梯度，保留其余调用。三组 128 步的中位数为 **58.217 → 56.985 ms/step**，减少 **1.231 ms（2.12%）**，速度比 **1.022×**。

测量起点缓存梯度及毛细应力与重新计算结果逐位相同。128 步后，原型与基线相场最大差为 2.98×10^-7、速度最大差为 3.73×10^-8 LU，与同状态基线重跑的差异相同量级；固体 ID、刚体中心相同。并非所有浮点字段逐位相同，完整误差见 `gradient_reuse.json`。正式实现应使用有效性标志，并覆盖初始化、加载检查点及外部改变相场/几何的情况。

### 4. 碰撞核存在明确的寄存器压力

Nsight 时间线记录 `_collide_momentum` **213 registers/thread**，`_collide_phase` 167，冰边界 101；block size 均为 128。主要热点的 `localMemoryPerThread` 为零，本次没有观察到 local memory 分配，不能将热点直接归因于 register spilling。

按计算能力 8.6 的 64K 寄存器/SM、48 warp/SM 上限推算，213 寄存器、128 线程的动量核最多容纳两个这样的 block，即寄存器资源允许的理论 occupancy 上限约 **16.7%**。这是由 launch 参数推导的上限，并非测得的实际 occupancy。[NVIDIA Ampere 调优指南](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)

### 5. 主机同步很多，但复制带宽不是大网格主瓶颈

Nsight 的 16 步窗口记录 946 次 kernel launch（59.125/step）、708 次 `cuStreamSynchronize`（44.25/step）。双向复制 606 次，但总数据量仅 **25,176 bytes**，设备复制累计仅 **0.509 ms/16 步**。

`cuStreamSynchronize` 占 CUDA API 时间 96.2%，其中大量时间是在等待上述 GPU 核完成。不能据此认定“同步本身占 96% 可全部优化掉”。同窗口墙钟减去 GPU 核累计仅约 **1.96 ms/step**，优先重写 CPU 接触或减少传输字节无法解释并消除 50 多 ms 的设备工作。

小网格则不同：水量统计、热重映射记账、相场恢复合计占 GPU 时间 **51.6%**；刚体推进约 0.91 ms，约为未插桩 step 的 18%（分层计时有扰动）。小网格优化应更关注归约、调用次数和数据上传打包。

## 加速方案与实施顺序

| 优先级 | 方案 | 可争取的耗时预算及证据 | 实施要点 |
| --- | --- | --- | --- |
| P0 | 复用有效的梯度/毛细应力 | **已测 2.12% step 降时** | 保留 recover phase、补点和相量投影后的必要更新；用有效性标志处理状态失效 |
| P0 | 冰边界候选列表/局部 tile；无活动冰时跳过冰边界路径 | 当前 **5.10 ms/step**，融尽态仍 4.96 ms；完全消除该项的理想 step 降时约 8.8%，实际要扣除列表维护成本 | 候选必须覆盖全部 26 个邻接方向；几何变化时更新；反射必须在 streaming 覆写前完成，继续保证反力/力矩和移动边界一致 |
| P1 | 优化全局水量投影与统计 | 活动态合计 **11.37 ms/step**；若该组耗时减半，整体理论降时约 **9.8%** | 将归约并入已有遍历或按块汇总；减少独立标量回读；研究把对 h/f 的平衡增量融合进下一次碰撞，避免额外读写 54 个分布。必须保留非平衡部分、局部速度/动态压力、守恒容差和记账，重新验证补点/重启时序 |
| P1 | 复用已有速度上界，减少独立 CFL 全域扫描 | 当前 **4.18 ms/step**，可争取约 7% 的预算，尚未实现验证 | 对当前流体速度 L1 上界 M，可保守证明六面出流率 ≤3.5M；若 3.5M≤0.5，热对流必为一次子步，可跳过精确出流扫描。必须保证 M 对几何更新/补点后的场仍有效，否则回退原扫描，保留有限性和子步检查 |
| P1 | 减少碰撞核活跃临时量，调 block size；优化 streaming 与宏观量恢复的数据流 | 动量碰撞 6.03 ms、streaming 5.70 ms；寄存器证据明确，但速度收益未验证 | 尝试 64/128/256 线程并实测；按默认 HRR 所需模态简化生成代码、缩短 27 项临时向量寿命。可研究 pull streaming/宏观量局部累积；避免跨线程读写竞争，压力恢复仍需全域相场更新后的梯度 |
| P2 | 打包刚体上传/回读、归并标量诊断 | 大网格收益较小；小网格有价值 | 当前每次 upload 分五次 from_numpy；将力/力矩、质量属性、姿态分别打包，减少小 kernel 和同步，保持 CPU 接触算法 |
| P2 | 几何不变/无冰时增量刷新；只对露出格点计算补点所需力 | 无通量力每步平均 1.125 次，当前约 **2.7 ms/step**；需按状态区别评估 | 将相量守恒与冰几何工作拆分；无冰不等于可以关闭全局水库存约束；保留相场变化后梯度和毛细力的正确时序 |

这些预算有重叠，**不能直接相加，也不是已实现的加速比**。建议第一阶段先完成缓存复用和局部边界处理，再优化水量投影/CFL；在保持数值约束的前提下，以整体耗时降低 15%–25% 作为需要实测验收的目标。若动量碰撞单项提高到 2×，按其当前占比，整个 step 也只约快 5.5%，不能期待单改 collision 就翻倍。

## 已排除的简单改法

对 6,075,000 个合成格点的 FP64 相量/容量归约做了独立微基准：显式 `atomic_add` 与 `+=` 在稳定重复中均约 **1.095 ms**；4096 路条带局部和加第二遍归约约 **1.028 ms**，只改善约 6%；末段改串行反而约 **1.147 ms**。

所以“把 atomic_add 换成 +=”没有实测收益，朴素两遍归约也不足以支持大幅加速承诺。三种组织方式在该合成输入上与 NumPy FP64 结果相同，尚不能据此代替真实流场验收。Taichi 1.7.4 的 CUDA 快速 warp reduction 映射只列出 i32/f32，FP64 需要额外设计与测量，但这也不证明当前耗时全部来自原子争用。[对应版本实现](https://github.com/taichi-dev/taichi/blob/v1.7.4/taichi/codegen/cuda/codegen_cuda.cpp#L338-L370)

## 测量边界

本机 GPU 同时承担桌面显示，没有其他计算任务参与这些测量；未锁 GPU 时钟，报告保留三次重复范围。Nsight Systems 已取得时间线和寄存器资源数据。驱动 `RmProfilingAdminOnly=1` 限制硬件性能计数器，因此没有采集 Nsight Compute 的实际 occupancy、DRAM 带宽及 stall 指标；本报告不声称已证明某个核达到带宽/算力上限。进一步调优按 [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 对这些假设做验证。

## 复现与交付物

在仓库根目录执行；默认保持原有物理设置。替换 `wall` 为 `kernels` 可得到独立 GPU 计时。

```bash
python iceflow3d/benchmarks/profile_step.py \
  --resolution-x 450 --mode wall --steps 128 --repeats 3 \
  --output iceflow3d/outputs/profiling_repeat/early_450_wall.json

python iceflow3d/benchmarks/profile_step.py \
  --mode kernels --steps 128 --repeats 3 \
  --checkpoint iceflow3d/outputs/dambreak_four_ice_blender_450_5s/stability_trials/original_0p1_checkpoint.npz \
  --output iceflow3d/outputs/profiling_repeat/active_450_kernels.json

python iceflow3d/benchmarks/probe_gradient_reuse.py \
  --checkpoint iceflow3d/outputs/dambreak_four_ice_blender_450_5s/stability_trials/original_0p1_checkpoint.npz \
  --output iceflow3d/outputs/profiling_repeat/gradient_reuse.json

python iceflow3d/benchmarks/probe_reduction.py \
  --output iceflow3d/outputs/profiling_repeat/reduction.json

nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=iceflow3d/outputs/profiling_repeat/active_nsys \
  python iceflow3d/benchmarks/profile_step.py --mode capture --steps 16 \
  --checkpoint iceflow3d/outputs/dambreak_four_ice_blender_450_5s/stability_trials/original_0p1_checkpoint.npz \
  --output iceflow3d/outputs/profiling_repeat/active_capture.json
```

- [完整 step profiler](profile_step.py)：未插桩/嵌套同步计时、CUDA event、独立 cProfile、Nsight capture 模式。
- [梯度复用隔离实验](probe_gradient_reuse.py)：同状态 A/B 和基线重跑误差对照。
- [归约微基准](probe_reduction.py)：显式原子、增量赋值、两遍及串行末段对照。
- [精简数据](step_profile_20261021_summary.json)：全部基线、核组耗时、分层调用、检查点状态和原型误差。
- 原始 `*_wall.json`、`*_kernels.json`、`*.log`、`active_450_nsys.nsys-rep`、SQLite、Nsight CSV、图表及源文件摘要保存在 `../outputs/profiling_20261021/`。

验证范围：运行上述真实 CUDA 测量、检查场景及原型数值差异，新增脚本通过 Ruff 和 Python 语法检查；生产求解器的源码摘要保持不变。
