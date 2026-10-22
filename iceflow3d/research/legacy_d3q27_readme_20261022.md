# 原 D3Q27 分布函数求解器说明（历史存档）

对应迁移前提交 `7889ec4fea33cefca4404559e81a5d54f30b1d63`。本页的模型、性能与长跑结果不代表 HOME 版本；复现实验应检出该提交。

# IceFlow3D

`iceflow2d_v3` 的三维扩展：气水两相、自由运动和接触的冰块、吸热融化。
保留 v3 的“融化后消失”语义：冰质量减少，**不生成液水，不注入融水动量或显热**。
运动或融化露出的流体格点从连通的已有流体重建。原二维目录不变。

当前默认动量碰撞为 **`cm_mrt`**，可用 `--momentum-collision-model cm_hrr`
选择 HRR 版本。下述既有 8 s 粗网格和 5 s 大网格验收采用 `cm_hrr`，
不作为新默认 `cm_mrt` 的长时稳定性验收。
默认值仅影响新建算例；完整检查点恢复时仍使用保存的碰撞模型。
切换后的 step 性能、CM-HRR 对照及后续加速方案见
[CM-MRT profiling 报告](../benchmarks/step_profile_cm_mrt_20261021.md)。

2026-10-22 起，相场法向正则化 `phase_normal_epsilon` 默认改为 **`0.01`**，
与二维四冰块溃坝示例一致，以抑制微小体相梯度造成的亏损聚集。
更强的正则化也会削弱界面尾部的压缩，可能增加残余混合相分数。
此前采用 `1e-8` 的运行结果不作为这个新默认值的完整耦合验收；
检查点恢复仍使用保存的配置。历史平面界面对照脚本显式保留两种旧测试参数。

当前按要求注释了全局水相总量归约、界面投影及其分布/动量补偿，
`conserve_phase_volume` 默认为 `False`；求解器拒绝显式启用已注释的路径。
相场方程自身的局部碰撞闭合、水热场同步及稳定性检查继续执行。
诊断仍记录水量漂移；此前开启全局投影时的验收不代表此配置的验收。
旧版 CM-MRT 曾在约 0.03455 s 触发保护；关闭投影的 CM-HRR 对照完成
1 s，水量漂移 +1.588%。当时记录见
[停用水量守恒修正的验证记录](../benchmarks/phase_conservation_off_150_1s_20261021.md)。
随后修正 CM-MRT 四阶中心矩与剪切模态的耦合，并独立闭合压力模态。
**2026-10-21，修正物性插值前的 150×50×30、CM-MRT 从零完成 4 s / 160000 步**，
水和空气黏度公式、5 格界面宽度、25 μs 时间步均保持原值。
累计峰值 L1 速度 0.173576 LU，低于 0.5 的保护阈值；全局水量投影
继续关闭，末态水量漂移 **+3.3885%**。本次只输出 z=15 mm 的二维截面，
每 0.02 s 一帧，共 201 帧，无三维场或检查点输出。结果与核验见
[4 s 截面运行记录](../outputs/dambreak_150_4s_zmid_20261021/metadata.json) 和
[核验结果](../outputs/dambreak_150_4s_zmid_20261021/verification.json)。

随后修正了两相物性与单精度误差：动力黏度线性插值后再除以混合密度；
该轮验证曾把法向正则化从 `0.01` 改为 `1e-8`，恢复界面尾部的相分离；
D3Q27 权重直接使用精确分数，并用双精度累计和闭合局部强迫动量。
**界面宽度仍为 5 格，水/空气的物性公式、mobility 和 CM-MRT 剪切机制不变。**
纯相及均匀混合物的 CUDA 剪切波黏度误差均小于 0.7%，完整 CUDA 回归
31 项通过。定向检查和溃坝复跑结果见
[物性与纯相保持修正记录](../benchmarks/constitutive_fix_20261021.md)。
修正后同一粗网格从零完成 4 s / 160000 步，累计峰值 L1 速度
0.369476 LU，末态水量漂移 +2.5703%。
[新运行核验](../outputs/dambreak_150_4s_corrected_20261021/verification.json)；
[新截面视频](../outputs/dambreak_150_4s_corrected_20261021/dambreak_z_mid.mp4)。

这是实验性三维版本。已修复移动冰边界、三维高阶矩、密度源项和几何补点水量
以及晚期接触方面的问题；**150×50×30 粗网格已通过 8 s、320000 步验收**，
其中接触修复后从 7.5 s 完整检查点续算。该次验收按要求取消了默认大网格
8 s 长跑；后续另行开展的 450×150×90、5 s Blender 制作不能视为已通过，
应以对应运行目录的完成和核验记录为准。
完整长时验收的配置、结果和限制见
[稳定性排查报告](../benchmarks/stability_8s_report.md)。稳定运行与网格收敛、
实验精度验证是不同的结论。

## 完整 thermal step 性能（2026-10-21）

在 RTX 3090 上，以 **450×150×90** 测量第 31→32 步：热更新累计量
从 7 变为 0，完整执行热对流、导热、冰水换热、融化、刚体/几何更新及检查。
热更新间隔保持 8 步，实际求解 `8Δt = 66.667 μs`；没有把热耗时除以 8。
每次从相同内存状态恢复，使用相同的计时前 GPU 预热，取 9 次墙钟中位数。
初始化、JIT、状态恢复、诊断与输出不计入下表。

| 工况 | 优化前 | 优化后 | 加速 |
| --- | ---: | ---: | ---: |
| 标准初始溃坝 | 54.79 ms | 31.47 ms | 1.74× |
| 平移、旋转冰体，含新暴露格点 | 57.42 ms | 33.79 ms | 1.70× |

迭代中保留的改动：

- 几何变化后的梯度只更新受影响包围盒，平衡态力只在新格点计算。
  当前镜像插值使用保守的六格依赖范围，包含标签不变的亚格点运动/融化。
- 冰边界反射、冰水换热先判断实际邻接关系，跳过不相关格点。
- pull 迁移融合相场/速度恢复，压力仍等待新相场梯度；累加保持双精度。
- 全流体邻域使用解析的 `A=I/3`，墙面/切割格点保留连通性与广义逆求解。
- `3 max‖u‖₁` 已足以保证单个热对流子步时省去 CFL 扫描，否则仍精确扫描。
- 导热后相场和流固格点都未变化时跳过恒等热重映射，去除接触前的重复上传。
- 材料列内先以双精度归约质量和惯量，再合并到全局；该项约 1.61→0.076 ms。

逐内核 CUDA 事件总时间约 48.66→26.92 ms；事件测试的早期轮次没有后来增加的
GPU 预热，因此正式加速比使用上面的同方法墙钟对照。最终主要 GPU 开销为
迁移/恢复 5.17 ms、动量碰撞 3.52 ms、力计算 2.95 ms。继续大幅降低这些开销
需要进一步重构分布函数布局或专用内核。无收益的零值原子操作筛选、成对张力
改写和按冰体分拆归约均已撤回。

CM-MRT、真实水/空气黏度、5 格界面、热参数、全局水量投影关闭状态均保持不变。
256 步移动冰体对照的截面速度相对 L2 差异为 0.0071%，相场为 0.0372%，
最大温差约 `6.14e-6 °C`，`phi=0.5` 分类和固体编号相同。这是短程数值一致性
检查，不是新的 4 s 细网格稳定性验收。35 项 CUDA 回归通过，包含融合迁移、
局部补点、CFL 分支和非均匀材料归约的新检查。只保存二维截面、标量结果和源码副本。

[完整结果与核验](../outputs/step_optimization_450_20261021/summary.json)；
[性能图](../outputs/step_optimization_450_20261021/performance.png)；
[CUDA 回归日志](../outputs/step_optimization_450_20261021/cuda_regressions.log)。

```bash
python iceflow3d/benchmarks/profile_complete_step.py --mode wall --repeats 9 \
  --output /tmp/iceflow3d_complete_step.json
python iceflow3d/benchmarks/profile_complete_step.py --mode kernels --case moving \
  --validation-steps 256 --output /tmp/iceflow3d_moving_step.json
```

## 运行四冰块溃坝

2026-10-22 已接入跨 step 的相场梯度/应力缓存，以及冰体旋转包围盒内的
局部冰水换热。当时的几何补点使用一格邻域，现已按镜像依赖范围扩大，
同时覆盖新露出、新覆盖格点和连续覆盖率变化；检查点恢复自动使缓存失效。
直接修改相场、覆盖率或固体掩码的外部调用者应调用
`sim.invalidate_gradient_cache()`，下一个 step 会重建缓存。
四块旋转、平移冰体的 450×150×90 同状态完整热 step 对照，中位数由
31.675 ms 降至 29.747 ms（减少 6.1%）；256 步后相场、速度、压力和固体编号
逐位一致，最大温差约 `1.99e-13 °C`。38 项 CUDA 回归通过。
这些是短程对照，不代替新的六秒长跑验收。
[细网格对照记录](../outputs/dambreak_450_6s_blender_1080p_20261022/validation/integration.json)。

使用仓库的 Python 环境和 `requirements.txt`，不增加依赖。默认使用 CUDA，
也可用 `--backend cpu` 在 CPU 上运行小网格验证。

```bash
python -m iceflow3d
# 同一个入口：
python -m iceflow3d.examples.dambreak_four_ice
python -m iceflow3d.examples.dambreak_four_ice_melting_3d
python iceflow3d/examples/dambreak_four_ice_melting_3d.py
```

默认目标为 8 秒，每 0.01 秒采样一次。450×150×90 网格共有 607.5 万格，
仅四组 D3Q27 分布就需要约 2.44 GiB；还需宏观量、材料网格和热场的显存。
完整运行需要 960,000 步。建议先用下面的短运行验证环境；短运行通过不代表
完整 8 秒的稳定性或网格收敛性通过。

```bash
python -m iceflow3d \
  --resolution-x 150 --phase-warmup-steps 20 \
  --steps 80 --output-interval-s 0.001 \
  --save-vtk --save-npz --no-progress \
  --output-dir iceflow3d/outputs/smoke
```

可观察到水流推动冰块、转动及融化的 CM-HRR 短演示：

```bash
python -m iceflow3d.examples.dambreak_four_ice \
  --resolution-x 150 --end-time-s 0.08 --output-interval-s 0.01 \
  --momentum-collision-model cm_hrr \
  --save-vtk --save-npz \
  --output-dir iceflow3d/outputs/dambreak_four_ice_demo
```

`--resolution-y`、`--resolution-z` 默认按立方格距自动推导；若显式指定，
必须满足 `dx=dy=dz`。物理水柱尺寸和冰边长需与网格对齐。粗网格会改变界面
相对水槽与冰块的分辨率，不能直接等同于默认网格。

`--no-plot` 关闭图像，`--save-vtk` 保存 ParaView 体数据，
`--save-npz` 按采样时刻分别保存三维数组，`--quiet` 关闭进度摘要。
已有输出目录默认拒绝覆盖；显式使用 `--overwrite` 可替换本例生成的文件。
完整选项见 `python -m iceflow3d --help`。

只保存 **z 中心截面**的粗网格 4 s 运行：

```bash
python -m iceflow3d.examples.dambreak_midplane \
  --resolution-x 150 --end-time-s 4 --output-interval-s 0.02 \
  --output-dir iceflow3d/outputs/dambreak_150_4s_midplane
python -m iceflow3d.postprocessing.midplane_fields \
  iceflow3d/outputs/dambreak_150_4s_midplane
```

此入口保存二维截面 `slices/*.npz`、诊断表和配置记录，不写三维场或检查点。
后处理生成速度、温度、相场三联图 `frames/*.png`、`final.png` 和
`dambreak_z_mid.mp4`。偶数层网格在相邻格心之间插值到精确的 z 中心面；
固体编号不做数值平均。速度颜色表示三维速率，箭头表示面内方向，
全程使用固定色标（速率在 0.05 m/s 以上采用对数色标）。
温度图显示水和冰的温度；空气显示给定初温，因为当前模型不求解气相能量方程。

## 默认几何与物理量

坐标采用 **x 向下游，y 向上，z 为水槽深度**；六面均为封闭无滑移绝热墙。

| 参数 | 默认值 |
| --- | --- |
| 水槽 / 网格 | 150×50×30 mm / 450×150×90 |
| 格距 / 时间步 | 1/3000 m / 1/120000 s |
| 墙厚 | 3 格，即 1 mm |
| 左侧热水柱 | 65×45 mm，横跨内水槽深度 |
| 水柱范围 | x=[1,66] mm，y=[1,46] mm |
| 下游浅水深度 | 31 格，即约 10.333 mm，从内地板算起 |
| 四冰块 | 10×10×10 mm 立方体 |
| 初始中心 x | 85、103、121、139 mm |
| 初始中心 y / z | 7 / 15 mm |
| 冰底间隙 | 3 格，即 1 mm |
| 水 / 冰 / 空气初温 | 90 / 0 / 20 °C |
| 水 / 冰 / 空气密度 | 1000 / 917 / 1.25 kg/m³ |
| 弥散界面宽度 / 相场预热 | 5 格 / 2000 步 |
| 初始线速度 / 角速度 | 零 |

`t=0` 为瞬时撤闸，不保留闸板。浅水深度默认按静态吃水加冰底间隙向上取整。
`--floor-water-depth-m 0 --ice-floor-clearance-cells 0` 可设置干底面。
`--domain-depth-m` 调整水槽深度，`--ice-centers-z-m` 设置四块冰的深度位置，
`--ice-angle-degrees RX RY RZ` 设置三轴初始转角。全部物理广延量使用真实
三维体积、质量与能量，不是二维的单位厚度量。

## 求解方法与范围

- **D3Q27 压力–动量 CM-MRT / CM-HRR**：`rho(phi)` 与压力独立，平衡分布零阶矩为零。
  采用三个可分离 3×3 中心矩变换；剪切模态保持水温相关的分子黏度，
  迹及偶数高阶压力模态使用 **0.6** 松弛率；可用
  `--cm-acoustic-relaxation-rate` 显式设置。水黏度沿用 v3 的 IAPWS 关联式，
  空气使用 Sutherland 关系，无额外人工剪切黏度。
  当前两相区域按相场线性混合**动力黏度**，再除以混合密度得到运动黏度。
  此前 450 网格五秒结果使用旧的运动黏度线性混合；后续复核发现其液相主体
  `phi≈0.979`，旧混合系数约为同温纯水的 1.87 倍（选定末态内部区域的
  倍率中位数）。该历史结果不能作为当前物性实现的流动衰减验证，
  详见 [黏度复核](../benchmarks/blender_450_5s_report.md)。
  三维溃坝默认使用 **CM-MRT**，在中心矩空间按不同模态松弛；四阶独立
  模态先扣除其二阶偏应力成分，该成分仍按原分子黏度对应的剪切率松弛。
  两种碰撞模型均独立闭合原始压力模态，保留低阶动量矩。
  可选 `cm_hrr` 使用 v3 已有 HRR 思路的三维扩展：用速度梯度重建非平衡偏应力，
  补齐三维递推四阶中心矩，并独立闭合 D3Q27 压力模态。两种模式的平衡分布
  均采用张量积形式；HRR 模式默认补偿对角三阶矩的三次速度缺陷。
  冰面梯度采用系数有界、对线性速度精确的虚拟点，避免小截断距离放大误差。
  在 `cm_hrr` 下，`--hrr-kinetic-blend 0` 表示完全采用该重建，
  `1` 表示保留动力学偏应力。它在有限波长上带来数值耗散，不能等同于
  原始 CM-MRT 的网格误差；物理黏度与时间步不变。
  `--momentum-collision-model cm_hrr` 可显式启用该重建；默认 `cm_mrt`
  保留未经 HRR 重建的中心矩碰撞，HRR 专属选项在此模式下不参与碰撞。
- **D3Q27 保守 Allen–Cahn 相场**：三维一阶中心矩给出迁移率，逐格碰撞
  精确闭合相场零阶矩；相量越界修正的累计体积单独记录。
  统一使用带壁面镜像取值的 D3Q27 梯度，假设中性润湿（90° 接触角）。
  相场、毛细应力、密度源项、压力恢复和补点平衡态共用一个梯度缓存。
- **相场与动量耦合**：测量碰撞前后平均相场通量，沿六个共享面计算
  `-div(u ⊗ J)`，两端使用相同通量。统一梯度不支持单独限制密度源项，
  历史配置字段 `density_gradient_source_limit` 仅接受 0；默认不截断质量通量，
  保持不可压缩关系与质量/动量输运一致性；补点包含完整半步源项。
  D3Q27 的静止权重为 8/27，压力恢复系数为 **9/19**，
  不是 D2Q9 的 3/5。
- **静水与毛细力**：用下游浅水层建立冻结静水参考，沿每个 x–z 柱积分
  水柱相对密度得到释放压力。表面张力是三维对称切向应力散度；冰表面接受
  对应反力。移动边界按截断格子链反射及动量交换求力和三分量力矩。
- **三维刚体**：材料网格归约质心、质量和完整 3×3 惯性张量；四元数积分
  三轴转动，保留陀螺项。接触使用剩余材料体素的有向包围盒与 15 轴 SAT，
  施加无摩擦、无回弹的法向冲量，计入转动有效质量。迭代预算用尽时检查
  修正后的实际残差；融化残体的墙面接触若收敛过慢，联合求解支撑点冲量，
  再核验全部约束，保持原有非穿透容差。每步仅将少量刚体状态
  传回 CPU 进行接触和积分，流体和热场仍在 Taichi 设备上。
- **热与融化**：世界网格上的水体积与显热采用三个方向的共享面有限体积
  输运，材料网格上的冰做三维导热。水冰界面用三线性权重分配等大反向热量；
  冰先升温，再消耗潜热并移除材料。导热按三维 Fourier 条件子步化，
  对流按实际六面出流量控制 CFL。最终不足一次热更新周期的部分会补齐。
- **露出格点**：从连通流体分层传播相场、动态压力和三维速度；无供体的新
  孤立空穴暂时封闭，后续几何更新会重试。不会按融冰质量填入指定水量。
  原有按 `phi*(1-phi)` 权重恢复初始总水量的全局修正现已注释，
  包括全域总量统计以及相应的 h/f 分布补偿。局部补点会产生可测的总水量漂移。

本实现提供三维核心路径，未逐项复制 v3 的可选 RM-MRT、D2Q5 热求解器
等实验分支。三维接触采用包围盒近似：融化形成凹形残体时可能提前接触，
没有裂解、再冻结、亚格距润滑膜或流体网格自适应。

当前不施加全局水量约束，也不声称局部重映射严格守恒动量和热量。
`history.csv` 记录 `phase_volume_projection_m3`（当前从零运行时应为零）、
`phase_projection_momentum_*_n_s`、`phase_clip_volume_m3`、
`thermal_remap_energy_j`、`discarded_melt_heat_j`、`gas_repair_count`
和 `pending_refill_cells`。默认关闭气相异常状态修复。
微小残冰删除的能量缺口以负值记入
`discarded_melt_heat_j`。默认保持逐步稳定性检查；失败会保存 `run_failure.json`
和此前已经完成的输出。

## 输出

默认目录：`iceflow3d/outputs/dambreak_four_ice_melting_3d/`。

- `history.csv`：每块冰的位置、速度、三轴角速度、四元数和质量，以及水量和热量诊断。
- `metadata.json`：完整配置、单位映射、采样清单、运行是否完成。
- `frames/frame_*.png`、`dambreak_four_ice_melting_3d.gif`：三维水冰体素表面
  与中间深度的温度、速度截面。绘图仅对三维表面降采样，不影响求解数据。
  温度截面把 `phi<0.5` 的气侧显示为设定气温；原始体数据仍保留水热场。
- `dambreak_four_ice_melting_3d.png`：完成运行的最后一帧。
- `vtk/frame_*.vti`、`fields.pvd`：可选完整三维场，用 ParaView 打开 `fields.pvd`。
- `fields/frame_*.npz`：可选原始数组，顺序 `(x,y,z)`，速度为 `(x,y,z,3)`；
  包含时间、格距、刚体质心和四元数。可视化快照不是完整重启文件。

完整重启使用 `iceflow3d.checkpoint.save_checkpoint` / `load_checkpoint`，
包括所有分布、材料状态、待更新热量和刚体状态。诊断入口
`benchmarks/diagnose_stability.py` 支持 `--checkpoint-interval-s`、
`--restart` 和冻结源码的 `--source-root`，默认校验恢复时的源码及字段。

PNG、GIF、VTK、NPZ 都按帧写盘，内存不随帧数累积。VTK 写入使用真实米制
单元坐标，速度为 m/s，压力为 Pa，温度为 °C。`solid_body_id` 中 `-1` 表示
流体，`-2` 表示容器，`-3` 表示暂缓重建的空穴，`0..3` 表示各冰块。

## Python 与测试

```python
from iceflow3d import IceFlow3D
from iceflow3d.examples.dambreak_four_ice_melting_3d import create_config, parse_args

config = create_config(parse_args(["--resolution-x", "150", "--phase-warmup-steps", "20"]))
sim = IceFlow3D(config)
sim.step(80)
sim.flush_thermal()
fields = sim.snapshot()
print(sim.diagnostics())
```

```bash
# 含 CPU 数值核的全部检查
python -m unittest discover -s iceflow3d/tests -v

# 在 CUDA 上复核数值核
ICEFLOW3D_TEST_BACKEND=cuda python -m unittest discover \
  -s iceflow3d/tests -p 'test_numerics.py' -v
```

测试覆盖 D3Q27 二阶/四阶各向同性、中心矩往返、压力/动量平衡矩及源项、
壁面镜像、斜冰面插值、梯度缓存失效、惯性张量、接触冲量的动量守恒、三向有限体积输运、
水冰热交换、消失融化、整块冰移除后的多层补点及 VTK 数组顺序。
首次运行需编译 Taichi 核，明显慢于后续运行。

## 统一壁面镜像梯度（2026-10-22）

实现位于 `gradients.py`。在格子单位下，所有相场/密度相关位置使用
`g = 3 Σ(q≠rest) w_q c_q [φ_ext(I+c_q) − φ(I)]`。
物理相场梯度为 `g/dx`，物理密度梯度为 `(ρ_water−ρ_air)g/dx`。
密度仍由同一个有界相场线性插值得到，不增加独立密度演化或密度梯度场。

- 流体邻居直接取值。容器墙位于流体格点与第一层墙格点中心的中点；
  逐坐标镜像索引，棱角同时镜像多个坐标，因此保留斜格子链的切向偏移。
  例如下壁 `φ_ext(i+1,b−1,k) = φ(i+1,b,k)`，不再统一复制中心值。
- 冰面沿用移动边界的覆盖率阈值和截断比例 `η`（范围 `[0.05,0.95]`）。
  用链两端覆盖率的中心差分插值得到法向 `n`，界面点为 `B=I+ηc`，
  镜像点为 `M=J−2[(J−B)·n]n`。法向退化时用链方向作为近似法向。
  这一覆盖率导数仅描述几何，不是另一套相场/密度梯度。
- 在 `M` 做三线性插值；只用流体角点，丢弃固体角点后重新归一化正权重。
  保持常值和局部取值范围。斜面缺角的插值不保证线性精确；极小未解析空腔
  若没有有效插值支撑，则退回起点流体值。该边界近似仍需长时算例评估。
- 每次相场更新后计算一次全域梯度/毛细应力；下一次碰撞直接复用。
  冰体移动或融化后仅刷新变化包围盒及六格依赖范围，即使标签没有翻转。
  压力恢复、动量碰撞、初始化和补点不再单独扫描邻居计算密度梯度。
  热场/几何更新后的局部刷新与新时间层的全域刷新属于必要更新。

原来的连通最小二乘相场梯度、独立的中心复制密度梯度及密度专属限幅
已移除。速度梯度（可选 HRR）和几何覆盖率导数是不同物理量，仍保留。
新增的几何刷新包围盒只是临时字段，不写入完整检查点；旧检查点仍需显式
允许源码变更，加载后会重建梯度。黏度、5 格界面、CM-MRT、时间步配置、
相场法向正则化和 donor 动量通量均未在本次修改中调整。
历史冻结版本及已完成的 4.30 秒渲染保持原样，不作为该梯度的长时验收。

本次 CPU/CUDA 完整回归均为 **46/46 通过**，含 6 项新增镜像/共享源项/
缓存测试。局部与全域刷新的相场梯度逐位一致；CPU 应力乘积允许两倍
float32 机器精度的舍入差，CUDA 仍要求逐位一致。
RTX 3090 上，450×150×90、四块旋转移动冰体、完整热更新的单步基准，
5 次恢复同状态测量的中位数为 **31.579 → 30.719 ms**（减少约 2.7%）。
初始化、预热、JIT 和状态拷贝不计时。两个版本的配置和流程相同，各自
独立预热；跨版本边界离散已变，不能将该对照解释为轨迹逐位一致性证明。
新版本继续 256 步并完成剩余热更新，峰值 L1 格子速度 0.031870，
未触发稳定性保护。此段只到物理 0.002392 秒，不是长时 dambreak 验收。
详见[统一梯度验证记录](../outputs/unified_mirror_gradient_20261022/verification.json)。

## 稳定性验证

本机 NVIDIA RTX 3090、Taichi 1.7.4、Python 3.13.9。流体、热和重启修复后的
完整测试集在 CPU 和 CUDA 上各 **25 项通过**，包含源项零阶/一阶矩、
小间隙边界、相场预热通量保留、整块冰消失及完整检查点恢复。
随后纯 NumPy 接触修复通过全部 **16 项配置/刚体/输出测试**（含 4 项新增），
并通过 500 组墙面联合冲量的约束与动能检查。

粗网格 150×50×30 已完成 **8 s / 320000 步**，全程每步最大 L1 速度
**0.208615 LU < 0.5 LU**，最大采样相量相对误差 **1.17×10^-7**，
气相修复次数为零。末态四块残冰均在容器内，穿透为零。
结果保存在 [最终验收目录](../outputs/stability_8s/final_150_completed/)；
[核验记录](../outputs/stability_8s/final_150_completed/verification.json) 明确包含
0–7.5 s 前段与接触修复后的完整状态续算。默认分辨率仍为 450×150×90，
本次按用户要求取消其完整 8 s 验证，不能将粗网格结果视为大网格通过。

长跑实验、失败场和冻结源码保存在 `outputs/stability_8s/`。
[排查报告](../benchmarks/stability_8s_report.md) 区分每个修复的证据、未奏效的
尝试、完整运行验收与仍存在的建模限制。失败及主动取消的实验均保留原状态。

以下为既有 CM-HRR 验收的历史诊断命令。复现其完整条件须使用当时开启
全局水量投影的冻结源码；当前关闭投影的实验命令见上方最新验证记录。

```bash
python iceflow3d/benchmarks/diagnose_stability.py \
  --resolution-x 150 --end-time-s 8 --sample-steps 2000 \
  --momentum-collision-model cm_hrr \
  --snapshot-interval-s 0.1 --checkpoint-interval-s 0.5 \
  --no-plot --output-dir iceflow3d/outputs/validation_8s

python iceflow3d/benchmarks/verify_stability_run.py \
  iceflow3d/outputs/validation_8s --resolution-x 150 --render
```

诊断入口生成 `diagnostic.json`、`timeline.jsonl`、原始体数据及两份滚动检查点；
复核脚本核验完成状态、每步速度峰值、体积、热量记账、原始场和源码摘要，
再生成 `verification.json`、`history.csv`，可选 PNG/GIF。
替换 `--resolution-x 450` 可用默认网格，耗时和存储量会显著增加。

## Blender 制作流程

`rendering/` 提供仿真、全分辨率表面缓存、本机 Blender Cycles 渲染和视频
合成的独立入口。此流程保存几何所需的数据，不生成速度、温度或压力着色。
完成状态必须同时满足仿真末时刻和最终影片核验；小样图不代表完整制作完成。
此前的 5 秒制作采用 10 格界面从零重跑，失败与对照证据见
[5 秒 Blender 制作记录](../benchmarks/blender_450_5s_report.md)。
该设置已完成 **450×150×90、5 s / 600000 步**，251 帧原始数据和最终
完整检查点均已通过核验；实测仿真总耗时 **8 小时 29 分 52.9 秒**，包含
初始化和保存，不含 Blender。2026-10-21 10:54（北京时间）按用户要求停止
正式渲染并暂停自动续作，保留 14 张 4K 图片（0–0.26 s），未生成完整影片。

```bash
python iceflow3d/rendering/simulate.py \
  --run-dir /absolute/run/simulation \
  --resolution-x 450 --end-time-s 6 --output-interval-s 0.02 \
  --interface-width 5 --momentum-collision-model cm_mrt --geometry-only

python iceflow3d/rendering/mesh_cache.py \
  --simulation /absolute/run/simulation --output /absolute/run/meshes --watch

/usr/local/blender-5.0.1/blender -b --factory-startup --python-exit-code 1 \
  --python iceflow3d/rendering/blender_scene.py -- \
  --cache /absolute/run/meshes --output /absolute/run/render \
  --device OPTIX --width 1920 --height 1080 --samples 64 \
  --noise-threshold 0.02 --min-samples 16 --volume-step-rate 1.5 --all
```

原始帧包含水相体场、固体编号、每块冰的材料占比、刚体姿态、
实际时间和单位尺度；滚动完整检查点另外保存全部分布、热场及待处理状态。
默认另存速度；`--geometry-only` 可省去当前无运动模糊镜头不使用的速度体场
和表面速度插值。此选项不改变几何、材质或完整检查点，也不能用于速度截面图。
原始帧不能代替检查点重启。表面取 `phi=0.5`，只做小于一个格距的滤波，
不降低网格分辨率，不添加虚构的飞溅。为避免水下冰块周围出现空气壳，
渲染水面在冰内部延拓；此处理仅作用于显示几何。

Blender 使用米制和右手坐标映射 `(x-Lx/2, Lz/2-z, y)`，水和冰使用折射、
吸收/散射材质，水槽使用玻璃。默认输出 1920×1080、128 samples、16 位 RGB PNG，
支持 OIDN GPU 降噪、自适应采样和体积步长选项。帧数从缓存清单读取；六秒
每 0.02 秒采样包含两个端点，共 301 帧，50 fps 视频容器时长为 6.02 s。
再经明确的 sRGB 到 BT.709 转换合成 H.264 视频。
体积积分默认使用 `--volume-integrator ray_marching --volume-step-rate 1.0`，
最多 4096 步，以避开 Blender 5.0.1 在毫米级残冰上构建 OpenVDB SDF 时的
固定行列式容差崩溃；仍保留冰的异质散射。故障证据、画面回归及步长近似
说明见制作记录。此前已停止的后台制作使用 `production_plan.json` 所记录的独立
渲染修订副本，原冻结仿真源码和脚本不覆盖。

本次制作单独保存于 `outputs/dambreak_450_6s_blender_1080p_20261022/`，
使用 CM-MRT、5 格界面、原水/空气黏度、关闭全局水量投影。
2026-10-22 10:30（北京时间），用户要求停止仿真并开始后续制作，原六秒
目标因此结束。停止时实际推进至 4.3550875 秒；为保留可完整核验的末态，
最终采用最近的 **4.30 秒 / 1032000 步完整检查点**，输出物理 0–4.30 秒的
**216 帧**，每 0.02 秒一帧，50 fps 视频时长为 **4.32 秒**。
额外的 4.32、4.34 秒两帧及原中断记录保存在 `user_stop/`，没有删除或作为
最终完整末态使用。目录名称保留历史 `6s`，不得据此宣称完成六秒。
正式仿真会话累计墙时 25352.527 秒（含中断尾段及重算），没有再次启动仿真。
`user_stop/finalization.json` 记录截取依据、完整检查点摘要和保留位置；
4.30 秒完整仿真核验已通过：峰值 L1 格子速度 0.336984，核算能量最大残差
`1.613e-5 J`，未投影水相体积最大偏差 2.6582%。本次 216 张 1920×1080、
16 位 RGB PNG 和 50 fps H.264 视频均已完成，视频完整解码无错误。
已查看本次概览图、原生分辨率代表帧及视频全部 216 帧的连续缩略图，
未见缺帧、黑帧、镜头跳变或明显整体亮度闪变；这属于逐帧检查，不是实时播放验收。
人工检查记录见 `quality_checks/final_review.json`，机器与人工结果汇总于
[交付清单](../outputs/dambreak_450_6s_blender_1080p_20261022/delivery.json)。
[全部 PNG 图片](../outputs/dambreak_450_6s_blender_1080p_20261022/render/frames/)；
[4.30 秒算例视频](../outputs/dambreak_450_6s_blender_1080p_20261022/render/dambreak_four_ice_450_4.3s_1080p_50fps.mp4)；
[Blender 工程](../outputs/dambreak_450_6s_blender_1080p_20261022/render/dambreak_four_ice.blend)。
正式仿真累计 **7 小时 2 分 32.5 秒**；独立早期失败尝试 **25 分 19.9 秒**，
故障重放 **39 分 24.0 秒**，三者去重合计 **8 小时 7 分 16.3 秒**。
这些数值不包含单元测试、独立通道流验证、人工排查间隔及后处理。
正式渲染共 **18 分 12.6 秒**，平均 **5.058 秒/帧**，范围 4.710–6.132 秒/帧；
视频编码 **3.377 秒**。完整时间记录见 `simulation_effort.json`、
`simulation/timing.json`、`render/render.json` 和 `delivery.json`。
正式渲染选择 64 samples、噪声阈值 0.02、体积步长 1.5、OIDN GPU 降噪；
镜头、灯光、折射材质和全分辨率表面不变。同为 1080P 的既有几何对照中，
512 samples 旧配置为 41.24–79.33 s/帧，128 samples 为 8.34–10.39 s/帧，
64 samples 为 4.52–5.62 s/帧；这是预演测试，正式制作耗时以最终清单为准。
[画质与性能选型记录](../outputs/dambreak_450_6s_blender_1080p_20261022/quality_checks/selection.json)。
新长跑在第 15963 步发现四冰体/墙面接触的 32 次迭代不足：残余穿透约
`1.015e-6` 格。捕获同一接触状态后，48、64、128 次上限均收敛到相同结果；
默认上限因此改为 128，仍提前退出，穿透和闭合速度容差保持 `1e-7`。
17 项配置/刚体测试通过，包含新捕获状态与动能检查。制作从 0.1 秒完整
检查点使用 `--body-contact-iterations 128 --allow-source-change` 显式续算；
该改动只增加求解迭代预算，不改物理参数。旧源码和失败记录保留。
随后在约 0.138 秒出现下游冰体附近的流体失稳；优化前全域梯度/换热对照
在同一步、同一格点复现，未归因于梯度缓存。局部迎风动量通量试验仍失稳，
没有接入生产。`simulate.py --time-step-scale 0.5` 支持从零进行半时间步验证，
物理步长为 4.167 μs，六秒需 1440000 步；水/空气黏度公式、网格、界面宽度
和速度保护阈值保持原值。该半时间步算例在第 28935 步、约 0.120558 秒
同样触发速度保护，总墙时 799.128 秒，未解决失稳。这次失败已归档到
`attempts/time_step_half/`，完整检查点和失败场均保留。不同力项、边界和输运
离散的晚期检查点对照记录于
[诊断证据](../outputs/dambreak_450_6s_blender_1080p_20261022/stability_trials/findings.json)，
这些早期失败的诊断变体未合入主求解器。
当时密度源项单独使用反射格子模板的修订完成半时间步 0.1→0.15 秒重放；
该冻结版本的相场/毛细力仍使用最小二乘梯度。两相通道流 2000 步的解析解速度
相对 L2 误差从 0.3862% 降到 0.2215%。冻结版本
`source_revisions/density_source_boundary/` 已通过从零到 0.30 秒的验证：
72000 步、16 帧，峰值 L1 格子速度 0.331559，核算能量最大残差
`4.801e-6 J`，该段总墙时 2050.355 秒。无水量投影时水相体积最大偏差
0.7184%，需要在最终报告保留此项记录。相同时间步的 0.12 秒对照中，
水相归一化 L1 差 1.295%、冰体位置最大差 0.637 mm。
边界修订已同步到主求解器，并从完整 0.30 秒检查点续算；随后在第
176791 步（约 0.73663 秒）于内部稀薄相场格点 `(26,81,46)` 再次触发
速度保护，两个正式会话累计墙时 4701.727 秒，尚未完成六秒。
最后完整检查点在 0.70 秒，现场保留于 `incident_records/bulk_176791/`，
原配置重放在同一步、同格点复现；该位置相场动量通量项比重力与
毛细力合计大约高三个数量级。冻结候选
`source_revisions/donor_phase_momentum/` 将这一通量的面速度改用守恒
迎风插值，物理黏度不变，但相场扩散通量与速度差并存处增加数值耗散。
独立能量恒等式检查、40 项 CPU 回归和 CUDA 动量/能量检查通过；
该新故障点的 0.735→0.80 秒重放通过，峰值 L1 格子速度 0.339210，
墙时 416.989 秒。此前在旧冰面边界上失败的迎风试验使用不同基线，
不属于当前已修正边界的新故障验收。
候选已通过完整耦合 0.70→0.82 秒验证，该段墙时 742.772 秒；
从相同的 0.70 秒完整检查点出发，0.72 秒水相归一化 L1 差为 1.381%，
水相二值区域交并比 98.671%，冰体位置最大差 0.0830 mm，冰体质量
最大相对差 0.00806%。至 0.82 秒共 196800 步、42 帧，连续轨迹峰值
L1 格子速度 0.336984，核算能量最大残差 `1.144e-5 J`，未投影水相
体积最大偏差 1.8961%。验收见 `validation/donor_phase_momentum_acceptance.json`，
完整检查点保留在 `validation/checkpoint_donor_0p82.npz`。
2026-10-22 04:59，验收程序已将该修正及新增动量/能量测试合入主源码，
并从完整 0.82 秒状态继续六秒制作。该数值方案从物理 0.70 秒生效，
时间步、物理模型、状态字段及分布函数定义相容，没有重缩放状态。
旧失败轨迹完整保留在 `incident_records/bulk_176791/production_archive/`；
其会话时间已包含在新轨迹的累计墙时中，不可再次累加。
状态见 `donor_continuation.json`、`simulation/simulation.json` 和 `pipeline.json`。
短段验收不能代替完整六秒验收，也不能证明长期轨迹不变。
此前边界修订的 CPU 39 项回归中其余 38 项通过，换热逐位对照的差异在原版本也可复现：
换热量相对误差约 `3e-7`、总能量守恒残差约 `1e-16 J`，没有遗漏换热格点。
测试现对 CPU 的单精度几何中间计算使用相应舍入容差，CUDA 仍要求水能量
逐位相等；修订后的单项复测通过，求解器换热代码不变。详细验收记录见
`validation/density_source_boundary_acceptance.json`。
`production_plan.json`、`pipeline.json` 与 `delivery.json` 记录本次用户提前结束后的
4.30 秒交付已完成；旧几何预演图不属于新算例结果，历史六秒目标没有完成。

`simulate.py` 的 `timing.json` 分项记录初始化、计算、诊断、保存原始帧和
检查点的实际墙钟时间；仿真总时间包含这些开销，不包含表面提取和 Blender。
`finish_pipeline.py` 仅在完整仿真核验通过后渲染、编码并生成 `delivery.json`；
失败仿真会阻止最终制作。该协调器用于有进程记录的后台制作目录，单独的
三个入口也可手动运行。长跑前应冻结所用求解器及脚本，保留源码摘要。

### 已保存结果的 z 中间截面速度图

450 网格五秒结果可直接通过 `postprocessing/velocity_slice.py` 生成截面图，
不启动求解器或 Blender。该入口处理此制作的全部 251 帧，输出 1920×960 PNG、
50 fps 视频、九帧概览以及用于复核的二维截面 NPZ。

```bash
python iceflow3d/postprocessing/velocity_slice.py \
  --simulation iceflow3d/outputs/dambreak_four_ice_blender_450_5s/simulation \
  --output iceflow3d/outputs/dambreak_four_ice_blender_450_5s/velocity_z_mid
```

截面位于几何中面 `z=15 mm`，由零起始索引 44、45 两层的格心数据线性插值；
插值涉及任意固体格点时将该位置遮罩，不对固体编号插值。颜色为水和空气中
三维速度的欧氏模长，单位 m/s；所有帧使用相同的线性色标，范围覆盖全部
保存截面的最大值。箭头表示 `(u_x,u_y)`，每 12 格采样，长度比例固定；
小于全局色标上限 1% 的截面内速度不画箭头，但仍保留速度颜色。
白色轮廓为水相场 `phi=0.5`，浅灰为冰、深灰为壁面。

输出目录中的 `frames/` 保存逐帧图片，`velocity_z_mid_50fps.mp4` 为动画，
`overview.jpg` 为概览；`visualization.json` 记录插值规则、色标、源数据摘要、
输出核验和后处理耗时。251 帧包含 0、5 秒端点，视频时长为 5.02 秒。

### 已保存结果的 z 中间截面相场图

先完成上面的速度截面提取，再运行以下入口。它复用同一套 `z=15 mm`
二维截面缓存，并核对缓存与正式仿真清单的源数据摘要。

```bash
python iceflow3d/postprocessing/phase_slice.py \
  --root iceflow3d/outputs/dambreak_four_ice_blender_450_5s
```

输出位于制作目录的 `phase_z_mid/`：`frames/` 包含全部 251 张
1920×960 PNG，`phase_z_mid_50fps.mp4` 为 50 fps、5.02 秒动画，
`overview.jpg` 为九时刻概览。色标固定为 0–1：0 表示空气、1 表示水，
浅灰表示冰、深灰表示壁面，白色轮廓表示 `phi=0.5`。相场只做几何中面
所需的层间插值，不做平滑、截断或逐帧归一化；后期液相未达到 1 的现象
保留在图中，每帧另标出流体格点的相场最大值。

`visualization.json` 记录截面定义、全部帧的摘要、视频帧数和解码核验。
按用户要求只生成相场，不生成温度图，也不重跑仿真。
