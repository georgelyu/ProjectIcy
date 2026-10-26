# IceFlow3D

三维气水两相、自由运动和接触的冰块、吸热融化。当前求解器采用 **D3Q27 HOME 动量 + D3Q7 AA 原地相场**；两套碰撞均为 central-moment MRT，不含 HRR。冰融化后消失，不生成液水；这一既有模型语义保持不变。

HOME 用两组各 10 个浮点矩替代两组 27 方向动量分布。相场只有一组 7 方向分布，以奇偶步寻址完成碰撞与迁移。新模型来源、浮力和精度检查、全部 FP64 使用位置见 [实现与验证报告](research/home_implementation_20261022_zh.md)。

## 保留的参数与模型

- 界面宽度 `W=5`，mobility `0.1`，法向正则化 `0.01`。
- 水使用原温度相关动力黏度公式，空气使用原 Sutherland 公式；混合动力黏度线性插值后除以密度。没有黏度下限或人工增黏。
- 时间步按原比例换算：默认 450×150×90 网格 `dt=1/120000 s`；150×50×30 网格 `dt=25 μs`。没有子步改变流体 dt。
- 动量剪切采用动力学中心矩；体积模态沿用 `cm_acoustic_relaxation_rate=0.6`。论文中的体积松弛率 1 是实现公式的特例。
- 相场和动量继续使用同一空间网格；没有采用论文的动量降分辨率方案。
- 保留静水参考压力、表面张力、热浮力、六自由度刚体与热/融化耦合。HOME 边界受力先把速度型分布乘回局部密度，再加入参考压力。
- 全局水量投影和气相状态修复关闭。数值保护报错，不通过改 W、dt、物理黏度或限速隐藏失稳。

三维生产 GPU 路径使用 FP32，包括热状态、热通量、刚体受力与质量/惯量归约。CPU 上的姿态、接触求解及诊断归约保留 NumPy FP64；它们不在 GPU 热点内。

## HOME 并行优化与完整 step 计时（2026-10-22）

本轮基线是优化开始时的 **HOME + D3Q7 AA** 源码，已保存在
`outputs/home_optimization_450_20261022/r00_source/`；不是旧全分布 D3Q27。
参照 `Home_ThreePhase` 修正了三阶 `xyz` 重建系数：该项六个排列需要系数 27，
原实现误用了 13.5。独立三阶 Hermite 求和回归覆盖了这个错误。

RTX 3090、450×150×90。每个样本从相同的第 31 步状态恢复，计时第 32 步，
包含热对流、完整的导热/换热/融化更新、刚体推进、几何刷新和稳定性检查。
热累计从 7 归零，导热推进 `8*dt`；没有把耗时除以 8。初始化、JIT、状态恢复和输出不计入。

| 无 profiler 的完整 step 墙时中位数 | 优化前 | 优化后 | 耗时减少 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 默认四冰块 dam-break | 26.84 ms | 14.14 ms | 47.3% | 1.90× |
| 倾斜、平移及旋转的冰块 | 29.01 ms | 14.90 ms | 48.7% | 1.95× |

优化后各取 9 个独立样本；基线默认工况 7 个、运动工况 5 个。
另开进程用 CUDA event 测得默认工况 GPU 时间总和 **24.74 → 12.61 ms**：

| GPU 部分（同一完整 step） | 优化前 | 优化后 |
| --- | ---: | ---: |
| HOME 迁移/重建/动量恢复 | 11.281 ms | 2.988 ms |
| 力计算及几何分类 | 5.013 ms | 2.273 ms |
| 主相场梯度 | 1.103 ms | 0.690 ms |
| 动量 CM 碰撞 | 1.171 ms | 1.144 ms |
| D3Q7 AA 碰撞 | 0.899 ms | 0.903 ms |
| 几何刷新、布局修复及局部补点 | 2.037 ms | 1.590 ms |
| 宏观量恢复及稳定性扫描 | 0.767 ms | 0.621 ms |
| 热模块 GPU 核 | 2.303 ms | 2.293 ms |

GPU 时间与墙时来自不同测量，不相加。逐阶段同步测量仅用于归因，原始记录也保留在 JSON 中。

主要改动：按完整流体、固定水槽和运动冰面拆分 HOME 与力计算；常规方向在编译时展开；
直接复用 HOME 归一化压力并从三个相场梯度分量计算表面应力作用；仅在变化区域初始化和传播补点；
批量传输刚体数据；合并相场/压力恢复和稳定性扫描；让主梯度与补点梯度使用相同的分支算法。
共享内存瓦片、边界压缩列表、力与碰撞强行融合等未改善实测时间，已撤回。

唯一有意改变的空间近似是变密度黏性力中 `grad(u)` 改为六面中心差分，
与参考仓库一致；碰撞中的动力学剪切矩、真实黏度、W=5、dt 和完整 26 链接压力修正均保留。
热方程、热物性和热更新间隔未改动。

优化前后两工况各额外推进 256 个耦合步并清空末尾热累计，均通过稳定性检查，
无待补点格子。共同流体的中间截面速度相对 L2 差异分别为 **0.0101% / 0.0452%**，
相场最大差分别为 **3.78e-4 / 2.20e-4**；冰块最大位置差为 **5.12e-5 格**。
这些是推进至 0.002392 s 的短程对照，不是多秒 dam-break 稳定性结论。

独立物理验证通过：纯水静水浮力误差约 **0.000920%**，W=5 自由液面部分浸没浮力误差
**0.5385%**，热浮力误差 **0.0000383%**；7 组 8,000 步剪切波相对物理黏度最大偏差
**0.556%**（包含空间离散误差）。当前源码实际编译的 **69 个 PTX 模块中 `.f64` 出现次数为 0**；
CPU 姿态/接触、初始参考压力构造及诊断归约仍保留 FP64。

本轮 **53/53 项回归通过**，覆盖独立 HOME/CM 矩推导、D3Q7 两种 AA 布局、
力及梯度专用路径与通用路径对照、热量守恒、刚体接触、检查点、移动几何和多层补点。
GPU 用例使用 CUDA；配置、接触约束和输出等主机测试也包含在总数中。
计时、物理验证和 PTX 审计的核心源码哈希一致。

证据：[汇总及数值差异](outputs/home_optimization_450_20261022/summary.json)、
[性能图](outputs/home_optimization_450_20261022/performance.png)、
[迭代记录](outputs/home_optimization_450_20261022/iterations.json)、
[源码与参数哈希](outputs/home_optimization_450_20261022/manifest.json)、
[浮力及黏度](outputs/home_optimization_450_20261022/final_physics.json)、
[GPU 精度审计](outputs/home_optimization_450_20261022/final_gpu_ptx.json)、
[HOME 专项测试](outputs/home_optimization_450_20261022/final_home_tests.log)、
[其余回归记录](outputs/home_optimization_450_20261022/regression_queue.json)。

```bash
python iceflow3d/benchmarks/profile_complete_step.py --mode wall --repeats 9 --output /tmp/home-wall.json
python iceflow3d/benchmarks/profile_complete_step.py --mode kernels --repeats 5 --output /tmp/home-kernels.json
python iceflow3d/benchmarks/profile_complete_step.py --case moving --mode wall --repeats 9 --validation-steps 256 --output /tmp/home-moving.json
```

## 运行

新增独立算例：[干冰落入绝热水缸](dryice/README.md)。入口为
`python -m iceflow3d.examples.dry_ice_in_hot_water_3d`，使用纯 CO₂ 开放气域、
内部固定水缸与吸热升华。它的配置、边界和热源放在 `dryice/` 中；
下列默认入口继续运行原 dambreak，不切换物性或边界。

依赖 Python、NumPy 和 Taichi。当前机器使用 `/home/georgelyu/anaconda3/bin/python`。

```bash
python -m iceflow3d --resolution-x 150 --steps 4000 --output-dir /tmp/home-demo
```

默认 450×150×90 算例、四块冰、物理尺寸和原有输出选项仍通过
`iceflow3d/examples/dambreak_four_ice_melting_3d.py` 配置。`--momentum-collision-model` 只接受 `cm_mrt`；HRR 参数已移除。

完整检查点版本已升级为 **2**，保存 HOME 两组矩和相场 AA 奇偶状态。原分布函数求解器的版本 1 检查点不能直接加载；必须使用旧源码继续旧运行，或从新模型初始状态启动。可视化快照仍不是重启文件。

## 验证命令

```bash
ICEFLOW3D_TEST_BACKEND=cuda python -m unittest discover -s iceflow3d/tests -v
python iceflow3d/benchmarks/validate_home.py --steps 8000 --output /tmp/home-physics.json
python iceflow3d/benchmarks/check_phase_transport.py --steps 4000 --output /tmp/home-phase.json
python iceflow3d/benchmarks/audit_gpu_kernels.py --output /tmp/home-ptx.json
python iceflow3d/benchmarks/audit_gpu_precision.py --steps 2000 --output /tmp/home-precision.json
```

精度对照脚本在临时源码副本中把本次转换的热模块和刚体归约恢复为 FP64；流动精度另以 NumPy FP64 线性算子的剪切波特征模态验证。PTX 检查会编译流动、相场、热、融化和移动几何补点路径，检查实际生成的 `.f64` 指令。

`check_mrt_modes.py` 现在转到 HOME 黏度验证。`profile_step.py`、`profile_complete_step.py` 和完整耦合诊断脚本支持新模型。历史 HRR、二维对齐、D3Q27 全分布编码及旧界面分布探针属于旧模型实验，需使用迁移前提交 `7889ec4fea33cefca4404559e81a5d54f30b1d63` 复现。

旧求解器的长跑、视频、性能数字与命令保留在 [历史说明](research/legacy_d3q27_readme_20261022.md)，不作为 HOME 的验证结果。二维目录保持独立，未切换到 HOME。

## 冻结旧版镜像梯度算例的渲染交付（2026-10-22）

本次使用启动时冻结的 D3Q27 全分布 CM-MRT 与统一壁面镜像梯度源码，
不包含随后主线的 HOME 迁移。450×150×90、5 格界面、真实水/空气黏度、
`dt=8.333333 μs`，水量投影及气相修复关闭。原计划 5 s，用户要求提前停止；
实际中断于 2.024625 s，交付截至最近完整检查点的 **0–2.00 s / 240000 步**。
期间未触发数值保护，运行过程中未改求解器或参数。最大采样水相体积偏差为
3.44396%，完整检查点和原中断记录均保留，不据此声称五秒稳定。

每 0.02 s 保存，共 **101 张 1920×1080、16 位 RGB PNG**；视频为
50 fps / 101 帧 / 2.02 s（包含两个端点）。仿真实际墙时 6920.206 s，
含初始化、输出和中断尾段；渲染 540.991 s，平均 5.356 s/帧，编码 1.697 s。
已核验所有 PNG、完整视频解码，并检查代表原图、概览及全部解码帧的连续图表。

[交付与验证记录](outputs/dambreak_450_5s_mirror_blender_1080p_20261022/delivery.json) ·
[全部图片](outputs/dambreak_450_5s_mirror_blender_1080p_20261022/render/frames/) ·
[视频](outputs/dambreak_450_5s_mirror_blender_1080p_20261022/render/dambreak_four_ice_450_2s_1080p_50fps.mp4)
