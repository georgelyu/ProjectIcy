# IceFlow2D（无热版本）

IceFlow2D 是独立于 `mixture2d` 的二维空气–水–冰耦合示例。两相流求解器是
`mixture2d` 纯流体路径的最小分支：D2Q9 中心矩 LBM 求解速度/压力，并用另一组
D2Q9 分布函数演化守恒相场。原目录中的源码不需要、也不会被修改。

当前冰块是一个保持面积的二维刚体，具有平移与转动自由度。它由有符号距离场
光栅化成尖锐、不渗透的移动边界；流体格点在切割链路上采用 unified 插值反弹
（也可选 halfway 反弹），动量交换给出流体对冰的力和力矩。冰受到重力、流体
动量交换以及按邻近水/空气密度计算的静水浮力，容器墙使用恢复系数和摩擦处理
刚体接触。移动边界新释放的流体格点由邻域重建，相场采用无通量边界；移动光栅
造成的水体积残差由严格的界面熵投影消除，不再向全域水相或空气相分摊。

无融化时，目标水量在相场预热前固定。每次投影只选择与 `phi=0.5` 等值线连通的
弥散界面带，并求一个全局 logit 位移，使离散水量恢复到目标值。该映射保持
`phi=0/1` 为严格不动点、保证 `0 <= phi <= 1`，并用
`h_eq(phi_new,u)-h_eq(phi_old,u)` 更新相场 populations，从而保留非平衡部分。
界面分辨阈值外的数值尾部会先回到精确 `0/1`，其质量差同样由真实界面承担，
避免 roundoff 或单边 clipping 长期累积成空气斑点或深水气孔。
`volume_correction_rate` 已删除；投影是完整约束，不能用欠松弛代替守恒。容量不足、
没有真实界面或所需界面位移超过 `volume_projection_max_shift` 时，求解器会明确报错，
而不会退回到污染纯相的全域修正。

代码还用 OBB 与单位格的精确多边形相交计算 `solid_fraction`，输出刚体面积、GCL、
swept-water 以及 refill 前后水量账本。这些量用于揭示二值 moving-mask 的几何误差；
当前 LBM populations 仍存放在二值 active nodes 上，因此它们是 cut-cell 几何审计层，
不是完整的 ALE/cut-cell 通量离散，不能据此宣称物理 cut-cell 体积已经严格守恒。

底板采用四角点接触冲量约束和广义坐标位置修正；冲量同时更新平移与角速度，
因此底板能够提供支撑力矩，而不会再用旋转 AABB 把冰块无功抬高。切向冲量使用
库仑约束 `|J_t| <= mu J_n`，默认 `bottom_wall_friction=0.03`，代表光滑或湿润
底面的低摩擦数值基线：接触时存在摩擦，但水柱冲击足够强时冰块仍会滑动。这个
系数需要按具体底面标定，不是通用材料常数。侧墙与顶墙继续使用 `wall_friction`。

当前水动力的动态部分由逐链动量交换成对耦合；静水部分采用按同一高度两侧
相场采样得到的 Archimedes 排水浮力，以补足速度型 LBM 归一化分布中缺失的
绝对静水压力。该近似只把合浮力施加在质心，不再使用瞬态相场延拓的一阶矩产生
人工浮力矩；冰块转动来自 cut-link 水动力矩和底板接触角冲量。这一修正能精确
通过水平水面的浸没面积测试，但它没有向流体施加单独的等反向体力，因此本版本
是稳定的机械耦合基线，不应宣称全系统离散动量严格守恒。后续可用守恒 cut-cell
压力牵引替换该修正。

刚体平移默认使用 `linear_damping=1.0`，即不再每个 LBM 步人为衰减冰块动量；
实际水动力阻力由 cut-link 动量交换产生。`force_relaxation` 仍对逐链合力做短时
低通滤波，平移和转动速度上限继续作为低 Mach 数稳定保护。若为特殊高冲击工况
显式设置 `linear_damping<1`，应把它视为数值稳定器，而不是物理阻力模型。

这套实现**不使用 MPM**：没有形变和碎裂需求时，单刚体状态比弹性 MPM 更直接，
也不会引入多孔介质体积分数。因此这里没有 `epsilon`，固体内部不是
`epsilon=0` 的多孔 LBM 单元，而是被排除在流体域之外的尖锐边界。本阶段也没有
温度、焓、传热、结冰或融化；后续加入相变时需显式更新刚体几何、质量、质心和
转动惯量，并守恒地向水相与能量方程交换质量和潜热。

## 默认算例

默认参数与以下旧示例一致：

```bash
python mixture2d/examples/dam_break_2d.py --mode coupled --material ice
```

即分辨率为 `600 x 300`，左下水柱宽为区域的 `25%`、高为 `2/3`，中央冰块为
`90 x 90` 格，底边位于 `y=3`，冰密度为 `917 kg/m^3`，相场预热 `500` 步。
IceFlow2D 中冰为尖锐刚体，而不是旧示例中的弹性 MPM 粒子块。

运行新实现：

```bash
python iceflow2d/examples/dam_break_2d.py --mode coupled --material ice --progress
```

默认执行 120 帧、每帧 100 步，并把 PNG 写入
`outputs/iceflow2d/coupled_ice/frame_XXXXX.png`。常用选项与旧示例保持同一风格：

```bash
python iceflow2d/examples/dam_break_2d.py \
  --frames 20 --steps-per-frame 50 \
  --resolution-x 600 --resolution-y 300 \
  --output-dir outputs/my_ice_run \
  --save-npz --no-progress
```

`--save-npz` 同时保存刚体状态和需要诊断的场；`--show-gui` 显示 Taichi 窗口。
求解器要求 Taichi CUDA 后端。参数、缩放量、当前步数与耦合方式写入输出目录的
`metadata.json`。

原有的 `python -m iceflow2d` 入口仍然可用，并转发到同一个示例脚本。

## 冰块自由落水算例

`ice_fall_2d.py` 初始化一个横跨容器内部宽度的水平水池，并把刚体冰块放在水面
上方的空气区。冰块从静止开始受重力下落，撞击自由界面后由同一套 cut-link
动量交换、显式静水浮力和刚体接触算法继续演化。默认冰块有 `5°` 小倾角，便于
观察入水时的非对称水动力和转动。

```bash
python iceflow2d/examples/ice_fall_2d.py --progress
```

默认 `600 x 300` 布局为：水面 `y=120`，冰块 `60 x 60` cells，冰块最低角点
位于 `y=180`，所以真实空气间隙为 `60` cells。间隙按 `resolution-y` 的 `20%`
缩放，并按旋转 OBB 的最低角点计算；改变倾角不会悄悄缩短指定落差。脚本还要求
冰块与弥散水气界面至少间隔默认的 10-cell 界面带。

按默认格子重力估算，冰角约在第 `3513` 步进入弥散界面、在第 `3797` 步到达
`phi=0.5` 名义水面，后者对应 `frame_00037.png`。撞击速度约为
`0.0316 cells/step`，低于默认刚体限速；这只是用于定位输出的自由落体估算，
不是数值解断言。默认 120 帧适合观察入水、下潜和初次回升，不代表已达到稳定漂浮。

常用自定义参数：

```bash
python iceflow2d/examples/ice_fall_2d.py \
  --water-level-fraction 0.40 \
  --drop-height-fraction 0.20 \
  --ice-width-fraction 0.10 \
  --ice-height-fraction 0.20 \
  --ice-angle-degrees 5 \
  --frames 120 --steps-per-frame 100 \
  --output-dir outputs/iceflow2d/my_ice_fall \
  --save-npz --progress
```

快速检查可以缩小网格；所有几何比例和默认落差会同步缩放：

```bash
python iceflow2d/examples/ice_fall_2d.py \
  --resolution-x 120 --resolution-y 60 \
  --frames 2 --steps-per-frame 5 \
  --phase-warmup-steps 5 --no-progress
```

该小网格命令只用于检查入口和输出链路；固定 5-cell 界面宽度下，12-cell 冰块
不足以作为定量物理工况。入水峰值载荷还会受到二值 moving boundary、力低通和
全局水量投影影响，因此本例应作为定性耦合演示，而不是冲击力标定基准。

也可以用 `--drop-height-cells` 指定绝对格子间隙，或用带符号的
`--initial-horizontal-speed`、`--initial-vertical-speed` 设置初速度。二者合速度
不能超过求解器默认的 `0.08 cells/step` 低 Mach 上限。

相场 warm-up 只平滑水气界面，并不建立静水压力；默认落差给水池留出了冰块入水
前的流体调整时间。若把落差和运行时间大幅缩短，应把初始压力瞬态纳入结果解释。
