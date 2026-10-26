# 干冰落入热水：独立的开放水缸算例

当前为**研究性实现**：几何、物性约束、产气源和运行路径已验证，
移动边界下的整体 CO₂/热量守恒尚未达到定量计算要求。实测偏差见文末，
不能把稳定运行等同于物理精度通过。

```bash
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --end-time-s 1 --output-dir outputs/dryice
```

默认使用 CUDA。也支持 `--backend cpu`。首次运行包含 Taichi JIT 编译时间。
结果目录包含参数及源码哈希 `metadata.json`、诊断序列 `history.csv`、中心截面 PNG；
`--save-npz` 另外保存完整三维场，`--no-plot` 关闭图像。
已经包含一次运行的目录会被拒绝覆盖。失败时写入 `run_failure.json` 并抛出错误。
NPZ 是可视化数据，当前新算例没有重启接口；原 dambreak 的检查点与入口保持原样。
CSV 的 `ice_0_x_m/y_m/z_m` 沿用共享求解器的数组原点；加上 metadata 的
`array_origin_m` 后得到以物理计算域底角为原点的坐标。

## 高频中心截面输出与双相可视化

```bash
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --resolution 216 --end-time-s 6 --output-interval-s 0.01 --save-slices --no-plot --output-dir outputs/dryice-216-6s
python -m iceflow3d.examples.render_dry_ice_slices outputs/dryice-216-6s
```

后处理追加 `--gas-speed-max-m-s 1.5` 可将气相速度色标固定为 0–1.5 m/s，
超过上限的值使用色标最高颜色；原始速度数据、箭头缩放和液相色标保持原值。

该配置推进 720,000 步，计划保存 601 帧（含 t=0 和 t=6 s）。
`slices/slice_XXXXX.npz` 只从 GPU 导出 216×216 截面，不保存整个三维体数据。
数值数组按 `(y,x)` 排列；`velocity_m_s` 按 `(y,x,3)` 排列，最后一维是
`ux,uy,uz`，单位 m/s。温度为 °C；`water_phase` 为液体相场；
`solid_body_id` 的 −1 表示流体、−2 表示绝热缸壁、非负值表示干冰。
绝热缸壁没有温度自由度，`temperature_c` 对应位置为 NaN。
`water_temperature_c` 另外保留弥散界面中的液体分量温度。
每帧还包含实际步数、时刻、dx、x/y 坐标和 z 坐标，不是检查点。

偶数网格的几何中面位于两层中心之间。这里沿用现有 3D 图的 `nz//2` 层，
不跨相界面插值；默认实际 z=36.166667 mm，目标几何中面 z=36 mm。
两者分别写入 `z_m` 和 `requested_z_m`，坐标原点是物理域底角。

后处理生成 `visualization/frames/frame_XXXXX.png`、`central_z_fields.mp4`、
`final.png` 和缸体放大图 `final_tank_zoom.png`。速度幅值含三个分量，箭头为
平面内的 ux/uy，按各相全序列最大速度分别缩放。沿用 `iceflow2d_v3` 的双相图层方式：
速度采用液体 `viridis` / 气体 `Oranges`；温度采用液体 `inferno` / 气体 `Blues`，
各有独立色标，整段序列的色标范围固定。气相温度为模型规定的恒温 20°C，
不是计算得到的 CO₂ 热输运。相场使用 `Blues`，固体另行着色。
默认视频为 30 fps；每帧对应 0.01 s，播放速度为物理速度的 0.3 倍。
后处理可单独对已完成的部分帧运行；`summary.json` 同时记录仿真状态、原始极值、
逐帧时刻、源码及输入数据哈希，部分序列不能当作完整 6 s 结果。

## 几何、物性和单位

竖直方向为 y，物理坐标原点在计算域底部角点。默认物理计算域为 72×72×72 mm，
有效网格 216³，存储数组额外包含一层边界 halo，为 218³。

| 项目 | 默认值 |
| --- | --- |
| 水缸外尺寸 x×y×z | 36×24×36 mm |
| 水缸位置 | x/z 居中，底面接触计算域地面 |
| 外底面积 / 计算域底面积 | 1/4 |
| 缸高 / 计算域高度 | 1/3 |
| 缸壁与缸底厚度 | 1 mm，固定不动，绝热 |
| 初始水深 / 缸沿余高 | 16 mm / 7 mm |
| 干冰 | 8 mm 立方体，初始底面高于水面 5 mm |
| 水 / CO₂ / 干冰初温 | 60°C / 20°C / −78.5°C |
| 网格间距 | 1/3 mm |
| 数值界面宽度 | **W=5 格**，约 1.667 mm |
| 流体时间步 | **dt=1/120000 s≈8.333 μs**，与默认 dambreak 相同 |

地面无滑移；水缸是缸底和四块侧壁组成的内部固体，没有顶盖。
计算域其余五面连接纯 CO₂ 压力库，允许气体双向交换。
侧面的压力包含气体静水头，顶部压力为 101325 Pa。
输出压力是相对顶部环境压力的表压，不包含 101325 Pa 常数。
外部边界只处理流体；干冰若越过开放计算域边界会明确报错。

水使用现有温度相关动力黏度关联式，碰撞使用当地水温；
CO₂ 使用 [Fenghour、Wakeham、Vesovic (1998)](https://srd.nist.gov/jpcrdreprint/1.556013.pdf)
的背景黏度关联式，按等温气体库状态计算，不使用空气的 Sutherland 公式。
默认约为水 **4.66×10⁻⁴ Pa·s**、CO₂ **1.47×10⁻⁵ Pa·s**。
没有黏度下限、增黏、速度裁剪或气相状态修复。
dt 仍按 `dx * 0.1 / 4 m/s` 换算；修改分辨率时 W 保持 5 格。

新算例的相场法向正则化量为 `1e-6`。原值 `0.01` 会削弱稀薄相场尾部的
压缩项，使液相扩散到缸外并驱动气体伪流动。这一独立设置保持 W=5 与
mobility=0.1；原 dambreak 的正则化量保持 0.01。

气体密度采用大气压下理想气体近似（20°C 时约 1.82953 kg/m³），
干冰密度 1560 kg/m³。水密度 1000 kg/m³、表面张力 0.072 N/m、
水热容与导热系数沿用现有模型的常数近似。
默认边长按网格取整，实际配置写入 metadata；低分辨率会改变实际壁厚和干冰尺寸。

## 热源与适用范围

这是采用气膜闭合的升华模型。水温由有限体积热输运求解，固定缸壁和地面热通量为零。
气体使用等温库近似，不单独求解 CO₂ 温度场。表面热流采用串联热阻：

`q'' = (T_donor - T_s) / [(δ_film + δ_resolved_gas)/k_CO2 + dx/(2 k_water)]`

默认亚网格气膜厚度 0.1 mm，可用 `--film-thickness-m` 设置；
已解析的气体间隙另计热阻。沿表面法线的搜索被固定固壁截断，不能穿过水缸传热。
没有找到水时采用静止气体导热近似，由气体库提供热量，并单独记录这部分热量。
干冰先升温到 −78.5°C，再升华；升华潜热为 573 kJ/kg。
由于新气体按 20°C 状态加入，热预算还包含将其从升华温度加热到气体库温度的能量。

固体质量减少，并将相同质量和随体速度对应的动量分配到邻近可连通流体格点。
气体产生对应 HOME 压力方程中的体积膨胀源
`dm * (1/rho_CO2 - 1/rho_dry_ice)`；水相不获得升华质量。
相场继续使用 D3Q7 AA，开放面的未知入射分布在两种奇偶布局中分别正确处理。
干冰用原六自由度刚体组件运动，并与缸底、四壁和地面接触。

气膜厚度尚未通过实验标定，模型没有水冻结、CO₂ 溶解、水蒸气凝结白雾、
表面微观成核或干冰破裂。液水达到冻结温度以下时会停止并提示模型缺项。
W=5 意味着不能把小于数个网格的气泡当成解析结果；默认网格用于毫米级较大气泡，
气泡尺寸分布与升华速率仍需网格、气膜和外边界距离的收敛及实验验证。

## 分项运行与诊断

```bash
# 保持干冰固定、关闭升华：检查静水、固壁和开放气域
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --fixed-body --no-sublimation --end-time-s 0.05 --output-dir outputs/dryice-static

# 从完全浸没的固定干冰开始：隔离产气与热源
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --initial-position submerged --fixed-body --end-time-s 0.1 --output-dir outputs/dryice-source

# 放在缸底，仍允许运动：检查产气、浮力和接触
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --initial-position bottom --end-time-s 0.1 --output-dir outputs/dryice-bottom

# 网格加密，物理尺寸不变；W 仍为 5，dt=6.25 μs
python -m iceflow3d.examples.dry_ice_in_hot_water_3d --resolution 288 --end-time-s 0.1 --output-dir outputs/dryice-refined

ICEFLOW3D_TEST_BACKEND=cuda python -m unittest iceflow3d.tests.test_dryice -v
```

低分辨率选项仅供程序检查。尤其 24³/36³ 网格的弥散界面接近水深，
会与缸沿相互作用，不能用于该物理场景的气泡结论。

CSV 同时保存固体质量、累计升华量、气体库存及净流出、水的净流出、
水供热、气体库供热、升华焓、开放面流出的水热量以及残差。
气体库存按定密度和气相体积分数计算；低马赫压力方程的瞬态可压缩误差和
移动固体的离散几何变化也会反映在整体 CO₂ 残差中。
沿用的热状态重映射产生的能量变化单独记录为 `thermal_remap_energy_j`；
`thermal_budget_residual_j` 扣除这一已报告数值项，不代表重映射本身物理守恒。
应同时检查残差、重映射量和 `phase_clip_volume_m3`，不能只看程序是否运行完毕。

水的热状态保存相对初始水温的能量，干冰保存相对升华温度的显热；
因此 `water_energy_j` 在冷却时为负。这样可在 FP32 中分辨微小冷却，
避免从巨大的背景显热反复扣减小热流。材料质量和源积分使用补偿求和，
仅升华核使用严格浮点运算；共享流动求解器的编译配置会恢复。
`water_mass_residual_kg` 使用完整相场积分，热网格的 `water_volume_m3`
仍沿用原求解器的微小液相截断，两者应区分。
开放面的体积通量直接累计实际穿过边界的 HOME 分布；同时输出
`co2_pressure_storage_kg` 与加入瞬态压力存储后的 `co2_continuity_residual_kg`。
图中气体显示气体库温度；NPZ 的 `water_temperature_c` 另外保留弥散界面中的液体分量温度。

## 已验证内容

新增 15 项测试覆盖物性参考表、几何比例、W/dt、浸没体下方静水头、
AA 两种奇偶布局、五面开放边界、壁面绝热、升华热/质量/动量分配、
微小质量扣减、固定气源的离散体积收支、冻结模型范围、缸底接触及截面导出。
原有 53 项回归全部通过，原默认 450×150×90 dambreak 另完成 256 步试跑。
原求解器及 dambreak 的 Python 源文件没有改动。

完整运行会同时受气膜闭合、移动固体的离散补点和热状态重映射影响。
通过分项测试和稳定性检查不等于通过整体守恒或网格收敛验证。
尤其需要把整体 CO₂ 残差与**本次升华质量**比较，而不只与大得多的背景气体库存比较。
该例目前用于研究耦合机制；不能据此宣称已经定量预测了气泡尺寸分布或升华速率。

2026-10-22，RTX 3090，最终源码的独立运行结果：

| 工况 | 推进时间 / 步数 | 水质量收支残差 / 初始水质量 | CO₂ 连续性残差 / 升华量 |
| --- | --- | --- | --- |
| 144³ 静水、固定干冰、关闭升华 | 0.05 s / 4,000 | −0.0000261% | 无升华 |
| 216³ 自由落水 | 0.12 s / 14,400 | +0.150% | −30.9% |
| 144³ 自由落水 | 0.50 s / 40,000 | +0.127% | −38.1% |

三次均完成，落水工况形成缸底附近产气区。静水最大 L1 速度约 0.00228 m/s，
10%–90% 相场厚度约 5.65 格（W=5 的连续 tanh 轮廓对应 5.49 格）。
升华源的固体质量账本残差小于 5×10⁻¹⁷ kg，但这不能抵消上表的整体 CO₂ 偏差。
216³ / 144³ 落水的热重映射项分别约 −0.125 / −3.182 J，
对应水与气体库提供的总热量约 0.390 / 1.847 J；后一个重映射量超过实际源热量，
不能忽略。扣除已记录重映射项后的热预算残差分别约 −2.56×10⁻⁵ / 0.0466 J。
因此下一项数值工作应是移动边界的守恒补点及相场一致的焓输运，
而不是用增黏、改变 W 或缩小流体 dt 掩盖误差。

本机原始参数、逐帧诊断、图像、测试日志和源码哈希保存在
[`outputs/dry_ice_validation`](../outputs/dry_ice_validation/validation.json)。
上述不同分辨率的运行时长不同，不构成网格收敛对比；也未验证默认 1 s 全时段。

## 2026-10-23：216³ 算例的性能优化

本轮只修改独立 `dryice` 扩展；共享求解器、原 dambreak 入口和其默认参数的
Python 源文件与优化前逐字节一致。保留 W=5、dt=1/120000 s、水/CO₂ 物理黏度、
热更新频率、升华源、刚体接触和稳定性检查。没有引入新的物理近似。

RTX 3090、Taichi 1.7.4、CUDA FP32 下，对完整耦合推进步反复 profiling。
每次从同一个完整状态恢复；预热 63 步后测 3×32 步，覆盖每 8 步一次的导热更新。
初始化、JIT、状态复制、诊断输出不进入计时。下表是 **profiler 开启时** 的中位步耗时，
独立关闭 profiler 的 7×128 步结果另存于下方的证据目录。

| 不开启 profiler 的工况 | 原实现 ms/步 | 优化后 ms/步 | 提速 |
| --- | ---: | ---: | ---: |
| 从缸底初态推进 | 42.956 | 19.703 | 2.18× |
| 带平移和转动的干冰 | 42.432 | 19.100 | 2.22× |

| 保留的阶段 | ms/步 |
| --- | ---: |
| 原实现 | 47.567 |
| 热量收支只对非零项原子累加 | 29.594 |
| 利用全流体分类专门处理内部 D3Q27 迁移 | 27.347 |
| 分离固定/开放边界与移动干冰迁移 | 25.770 |
| 恒温区黏度常量折叠、空产气源和旧状态写入优化 | 23.936 |
| 固定缸壁镜像几何缓存 | 23.290 |
| 跳过恒为零的热通量及无通量更新 | 21.780 |
| 复用已更新的边界分类，去掉空扫描 | 21.061 |
| 只枚举五个开放面的相场入射格点 | 20.869 |

其中热通量应用核从约 19.53 ms/步降到约 0.16 ms/步（按完整导热周期摊销）。
恒温区仍使用原混合黏度公式，包括稀薄液相尾部；不将小相场值直接截成纯气体。
旧速度只在移动壁面反射需要的格点保存。固定壁面缓存只保存静态几何，
对角镜像仍读取当前流体/固体标签。热通量活动掩码由不可变的面通量构建，
不在原位更新水体积时读取邻格体积来决定是否跳过，以免引入读写竞态。

试过但未保留的方案包括第三阶系数预存、完整 27 分布预存、部分展开的内部迁移、
动态循环的固定边界迁移，以及紧凑遍历 halo。它们增加了寄存器、内存流量或不连续访问，
实测没有收益或变慢。最终仍以内部迁移、受力和相场核为主要开销；
继续大幅提速需要更深的内核融合、数据布局或 CPU/GPU 刚体耦合改造。

新增 6 项数值回归，连同原 15 项干冰测试，共 **21 项 CUDA 测试通过**。
覆盖迁移与原通用核的对照、固定镜像面/边/角、移动几何分类缓存、开放面唯一枚举、
面通量散度与热量预算、微小液相尾部的黏度。原默认 450×150×90 dambreak
另完成 256 步试跑。最初启动的整个旧测试套件因反复 JIT 耗时中止，日志保留；
本轮没有声称重新跑完原有全部 53 项测试。

另以完全相同配置运行 **216³、0.2 s、24,000 步、21 个截面**，与先前 6 s 运行的
同一时段比较。入水/下沉/触底轨迹与累计升华量保持接近；触底后的局部气泡与瞬时
速度并不逐点相同。因此还运行了仅跳过零值诊断累加的独立对照，用于判断当前浮点
实现本身的重复运行差异。所有差异、守恒诊断和源文件哈希均保留在证据目录，
这些检查不构成物理精度或整体守恒验证。优化后的版本没有重新运行完整 6 s。
缸底及运动初态还分别进行了 256 步的全三维对照：固体标签完全一致，速度相对 L2
差异分别约 0.0052% 和 0.0046%，相场约 0.0109% 和 0.0144%。

两次对照都与原 6 s 运行的前 0.2 s 比较：

| 对照版本 | 全时段最大位置差 mm | 末态累计升华量差 | 末态截面相场 RMS 差 | 末态速度 RMS 差 m/s |
| --- | ---: | ---: | ---: | ---: |
| 仅跳过零值诊断累加 | 0.480 | 0.206% | 0.03163 | 0.02944 |
| 全部保留的优化 | 0.237 | 0.389% | 0.02300 | 0.02833 |

速度比较包含气相的三个分量，在两次运行共同的流体格点计算。其末态逐点相对 L2
差异分别约 120% 与 115%，不能称为完整流场的逐点保持。仅改变诊断操作的对照也有
相近差异，说明当前浮点实现的重复运行敏感性不可忽略；这一个对照并不构成统计置信区间。
完整优化版本的液体温度末态截面 RMS 差约 0.051°C，CO₂ 仍为规定的恒温 20°C。
本轮证据支持被检查时段的宏观落水/触底轨迹与升华总量保持接近，不支持 6 s
气泡细节或瞬时速度的逐点一致声明。

复现当前版本的独立计时和 CUDA 核计时：

```bash
python iceflow3d/benchmarks/profile_dryice.py --mode wall --steps 128 --repeats 7 --output outputs/dryice-wall.json
python iceflow3d/benchmarks/profile_dryice.py --mode kernels --validation-steps 256 --output outputs/dryice-kernels.json
python iceflow3d/benchmarks/profile_dryice.py --case moving --mode wall --steps 128 --repeats 7 --validation-steps 256 --output outputs/dryice-moving.json
```

`--source-root` 可指向包含冻结 `iceflow3d` 包的目录，以相同测量脚本运行旧实现。
`compare_dryice_profiles.py` 对照完整三维短程输出，`compare_dryice_runs.py` 对照逐帧物理时间、
轨迹和中心截面。本机证据目录为
[`outputs/dryice_optimization_216_20261023`](../outputs/dryice_optimization_216_20261023)，
包含各轮源码/测量、被撤销方案、最终独立计时、测试日志、动态对照与性能图。

## 完整三维科学数据与 Blender 制作

`rendering/dryice_export.py` 调用此算例的原配置及 solver，不改变 W、dt、黏度或
原 dambreak 流程。216³、6 s、每 0.01 s 保存一次对应 720000 步、601 帧（含两端）：

```bash
python -m iceflow3d.rendering.dryice_export --resolution 216 --end-time-s 6 --output-interval-s 0.01 --output-dir outputs/dryice-film/simulation
python -m iceflow3d.rendering.dryice_mesh --simulation outputs/dryice-film/simulation --output outputs/dryice-film/meshes --watch
python -m iceflow3d.rendering.dryice_portable --simulation outputs/dryice-film/simulation --output outputs/dryice-film/paraview --watch
```

每帧 HDF5 保存 `velocity_m_s[...,3]`、`pressure_pa`、`temperature_c`、
`water_temperature_c`、`water_phase`、`ice_fraction`、`solid_body_id`、
`gas_mass_source_kg_m3_s`。数组顺序为 **z,y,x**，速度分量仍为求解器的 x,y,z，
**y 为竖直方向**；去除一格 halo 后形状为 216³。首个物理采样点为
`(dx/2,dx/2,dx/2)`。表压包括静水压，零点为顶部开放边界的环境压力；固体内
压力及速度不是流体解，使用 `solid_body_id == -1` 掩码。气相温度仍为规定的
20°C，液体分量温度另行保存。材料占比、质量、能量和刚体姿态保存在 `body/`。
这些文件是可视化快照，**不是完整续算检查点**。

原始输出使用快速无损 LZF。`dryice_portable` 同步生成通用 GZIP HDF5，逐数组
核对一致后再发布文件；科学软件应打开 `paraview/fields.xdmf`。原始文件保留不变。
与 XDMF 一起移动整个 `paraview` 目录，避免破坏相对路径。另存中心 z 平面的 NPZ。

表面缓存由全分辨率 φ=0.5 等值面和干冰材料占比=0.5 等值面生成。干冰采用
完整三维材料网格及刚体变换，不依赖单个截面是否与干冰相交。仅显示几何做
σ=0.35 格的高斯滤波；水面在固体内部使用最近流体值延拓，不人为增加空气壳。
没有生成仿真之外的气泡、泡沫或烟雾。Blender 坐标映射为
`(x-Lx/2, Lz/2-z, y)`，单位为米。水表面同时保留实际速度、温度和压力顶点属性。

`dryice_blender.py` 使用 Cycles/OptiX，生成透明水体主视图和三场科学剖切视图。
科学视图的固定色标为水速度 0–0.5 m/s、气体速度 0–1.5 m/s、温度 0–60°C、
表压 −250–250 Pa；超出色标仅使显示饱和，不裁剪 HDF5 数值。透明缸壳是显示
材质，热边界仍是绝热固体；科学视图隐藏缸体前半部分以显示剖切面。

`dryice_pipeline.py` 等待已有计算结束，核对完整体场、时间步和源码摘要，再
渲染及合成。它不会另行启动计算。`dryice_finish.py` 保留 16 位无文字原图，
生成带单位/色标的 PNG，并以明确的 sRGB→BT.709 转换编码 25 fps H.264。
601 帧视频容器时长为 24.04 s，对应 4 倍慢放及末帧停留一帧。失败的计算只
交付已保存部分，不能标记为完整 6 s；数值数据完整性检查不等于物理模型定量验证。

2026-10-23 制作目录为
[`outputs/dry_ice_216_6s_blender_20261023`](../outputs/dry_ice_216_6s_blender_20261023)。
实时状态见 `simulation/simulation.json` 与 `pipeline.json`。完整成片完成后位于
`render/beauty/` 和 `render/science/`；各自有 `frames/frame_000000.png` 起的逐帧图
及 `dry_ice_beauty_25fps.mp4` / `dry_ice_science_25fps.mp4`。
