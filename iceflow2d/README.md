# IceFlow2D（机械与固定冰热相变）

IceFlow2D 是一个二维空气–水–冰耦合求解器。流体使用 D2Q9
pressure--momentum LBM 求解动量与动态压力，并用另一组 D2Q9 分布函数演化守恒相场。
机械算例仍可使用原来的移动刚体；设置 `IceFlowConfig.thermal` 后，则启用固定冰首版
热耦合：同一网格上的守恒有限体积焓方程负责导热和速度对流，液相率反向更新冰边界。

当前冰块是一个保持面积的二维刚体，具有平移与转动自由度。它由有符号距离场
光栅化成尖锐、不渗透的移动边界；流体格点在切割链路上采用 unified 插值反弹
（也可选 halfway 反弹），动量交换给出流体对冰的力和力矩。冰受到重力、流体
动量交换，容器墙使用恢复系数和摩擦处理刚体接触；不再另外叠加显式阿基米德
浮力。移动边界新释放的流体格点由邻域重建，相场采用无通量边界；移动光栅
造成的水体积残差由严格的界面熵投影消除，不再向全域水相或空气相分摊。

无融化时，目标水量在相场预热前固定。每次投影只选择与 `phi=0.5` 等值线连通的
弥散界面带，并求一个全局 logit 位移，使离散水量恢复到目标值。该映射保持
`phi=0/1` 为严格不动点、保证 `0 <= phi <= 1`，并用
`h_eq(phi_new,u)-h_eq(phi_old,u)` 更新相场 populations，从而保留非平衡部分。
界面分辨阈值外的数值尾部会先回到精确 `0/1`，其质量差同样由真实界面承担，
避免 roundoff 或单边 clipping 长期累积成空气斑点或深水气孔。
投影是完整约束，不能用欠松弛代替守恒。容量不足、没有真实界面或所需界面位移超过
`volume_projection_max_shift` 时，求解器会明确报错，
而不会退回到污染纯相的全域修正。

底板采用四角点接触冲量约束和广义坐标位置修正；冲量同时更新平移与角速度，
因此底板能够提供支撑力矩，而不会再用旋转 AABB 把冰块无功抬高。切向冲量使用
库仑约束 `|J_t| <= mu J_n`，默认 `bottom_wall_friction=0.03`，代表光滑或湿润
底面的低摩擦数值基线：接触时存在摩擦，但水柱冲击足够强时冰块仍会滑动。这个
系数需要按具体底面标定，不是通用材料常数。侧墙与顶墙继续使用 `wall_friction`。

当前水动力完全来自逐链动量交换，不按排水体积另加浮力。pressure--momentum
populations 的一阶矩为 `rho*u`，二阶矩为 `rho*u*u+p_dyn*I`；材料密度仍由相场给出。
启用单一选项 `well_balanced_hydrostatics` 时，求解器会整体启用 well-balanced
静水方案：在相场 warm-up 和水量投影后冻结 `rho_ref/p_H`，让 populations 只携带
动态压力 `p_dyn=p-p_H`，把 bulk 重力源改写为 `(rho-rho_ref)g`，并在移动冰体的
cut links 上补回与同一 `p_H` 匹配的静水压力冲量。这些步骤共同保证静水压力只计算
一次，不能拆成相互独立的开关。

cut-link 静水压力冲量按
[Guo et al. (JCP 540, 114293, 2025)](https://doi.org/10.1016/j.jcp.2025.114293)
式 (23)，在每条 cut-link 由相邻 SDF 值割线插值得到的边界点 `x_w` 构造静水平衡参考。
最终水动力是动态项与该平衡参考项的代数和，因此阿基米德力属于
well-balanced cut-link 动量交换，不是刚体方程中另加的一项。当前只使用原生 D2Q9 links，
没有实现论文式 (27) 的 sub-grid 积分。
满水域使用常密度解析参考；横跨活动区域的水平水池则在相场 warm-up 和水量投影后，
先对每一行的 `rho(phi)` 求参考密度，再从初始自由液面沿重力方向积分
`p_H=integral rho_ref*g*dx`。这样 `p_H` 穿过弥散水气界面仍连续，不会产生
`rho(x) g*(x-x_ref)` 在变密度闭合表面上的 gauge 误差。两相都显式求解，故
`rho_link` 已包含水/空气权重，不再额外乘论文单相 VOF 路径中的 `epsilon`。

刚体平移默认使用 `linear_damping=1.0`，即不再每个 LBM 步人为衰减冰块动量；
实际水动力阻力由 cut-link 动量交换产生。刚体当前直接使用未滤波的逐链合力，
平移和转动速度上限继续作为低 Mach 数稳定保护。若为特殊高冲击工况
显式设置 `linear_damping<1`，应把它视为数值稳定器，而不是物理阻力模型。

这套实现**不使用 MPM**：没有形变和碎裂需求时，单刚体状态比弹性 MPM 更直接，
也不会引入多孔介质体积分数。因此这里没有 `epsilon`，固体内部不是
`epsilon=0` 的多孔 LBM 单元，而是被排除在流体域之外的尖锐边界。热耦合首版只允许
冰的位姿固定，暂不更新刚体质心、惯量或运动；固相边界会随液相率变化，融化节点
会重建 LBM populations，结冰节点则从流体域移除。

## LBM–焓相变耦合算例

```bash
python iceflow2d/examples/coupled_fixed_ice_melting_2d.py
```

默认物理尺度为 `2.5 x 3.5 mm`、网格 `80 x 112`，因此
`dx=31.25 μm`。水面位于 `2.75 mm`，固定方冰大小为 `1 x 1 mm`，中心位于
`(1.25, 1.50) mm`。冰初温为 `0 °C`，水和左/右/底热边界为 `60 °C`，顶壁与
水气热界面绝热。总物理时间 `0.30 s`，对应约 `35,560` 个 LBM 步；升高水温使
毫米级冰在短时间内即可出现多格点融深。Taichi 参考运行的最终融化面积约为
`49.0%`，等效均匀融深约 `0.143 mm`（约 4.6 格），且中心仍保持固态。

冰和水分别使用 `rho_i=917 kg/m³` 与 `rho_w=1000 kg/m³`。连续固相体积由
`V_s=sum(1-lambda)` 计算；相变后的物理总液态水体积为

```text
V_water,total = V_water,0 + (rho_i / rho_w) * (V_s,0 - V_s)
```

所以一个单位冰体积完全融化只生成 `0.917` 单位水体积；其余 `0.083` 体积表现为
自由液面下降。由于 LBM 的活动域按 `lambda=0.5` 整格切换，还会跟踪尖锐掩膜体积
`V_g`，实际投影目标使用

```text
V_water,active = V_water,0
               + (1-rho_i/rho_w) * (V_s-V_s,0)
               - (V_g-V_g,0)
```

这样整格释放或冻结时，目标水量同步跳变一格，不会把几何离散误差误投影到自由面；
当尖锐体积等于连续固相体积时，它严格退化为上面的总液态水公式。反向结冰时同一
公式自动给出体积膨胀。焓反演的潜热区使用
`rho_i L`，液态显热使用水的真实密度和热容。热场采用 f64 焓与面通量，并按
Fourier/Courant 条件自动子步进；LBM 速度进入迎风显热通量，非零重力工况还可通过
Boussinesq 项把温度反馈到动量方程。

默认验证把机械重力和表面张力设为零，以隔离导热、相界面更新和密度收缩水量修正。
当前限制是：冰体不能移动，水气热界面必须绝热且需要保留空气层供体积投影使用；
移动冰与移动热材料掩膜的守恒重映射属于后续版本。输出目录
`outputs/iceflow2d/coupled_fixed_ice_melting_2d/` 包含 `history.csv`、`fields.npz`、
`metadata.json` 和最终温度/液相率图。

这里的“没有相变”特指移动刚体水动力求解器。仓库现已加入一个独立的 CPU 二维
固定冰相变验证，作为后续二维热耦合的基础：冰不平移，固相区域可因相变缩小；
求解器使用守恒有限体积焓法，热量从四个方向进入。

```bash
python iceflow2d/examples/stefan_melting_2d.py
```

默认尺度为 `30 x 30 mm` 水域、`120 x 120` 网格（`dx=dy=0.25 mm`），中央
放置 `12 x 12 mm` 方冰。冰初始温度和熔点均为 `0 °C`，水体及四壁保持
`20 °C`，模拟 `180 s`。最终约 `44.3%` 的初始固相面积融化，等效均匀融深约
`1.52 mm`（超过 6 个网格），中心仍保持固态，能够清楚显示二维角部加速融化。
算例采用冰水等密度近似以隔离潜热与传热，不包含相变体积变化、流动或自然对流。
验证项包括四壁输入热量与全域焓变化守恒、D4 旋转/镜像对称、固相面积单调下降和
`80/120/160` 网格加密趋势。输出目录
`outputs/iceflow2d/stefan_melting_2d/` 包含历史 CSV、元数据、二维场 NPZ 和汇总图。

## 默认算例

默认分辨率为 `600 x 300`，左下水柱宽为区域的 `25%`、高为 `2/3`，中央冰块为
`90 x 90` 格，底边位于 `y=3`，冰密度为 `917 kg/m^3`，相场预热 `500` 步。
冰采用尖锐、不渗透的刚体表示。

运行默认算例：

```bash
python -m iceflow2d --progress
```

默认执行 120 帧、每帧 100 步，并把 PNG 写入
`outputs/iceflow2d/coupled_ice/frame_XXXXX.png`。常用选项：

```bash
python iceflow2d/examples/dam_break_2d.py \
  --frames 20 --steps-per-frame 50 \
  --resolution-x 600 --resolution-y 300 \
  --output-dir outputs/my_ice_run \
  --no-progress
```

`--show-gui` 显示 Taichi 窗口。求解器要求 Taichi CUDA 后端。

原有的 `python -m iceflow2d` 入口仍然可用，并转发到同一个示例脚本。

## 冰块自由落水算例

`ice_fall_2d.py` 初始化一个横跨容器内部宽度的水平水池，并把刚体冰块放在水面
上方的空气区。冰块从静止开始受重力下落，撞击自由界面后由同一套 cut-link
动量交换和刚体接触算法继续演化，不再另加显式静水浮力。默认冰块有 `5°` 小倾角，便于
观察入水时的非对称水动力和转动。

```bash
python iceflow2d/examples/ice_fall_2d.py --progress
```

默认 `300 x 600` 布局为：水面 `y=420`，冰块 `30 x 30` cells，冰块最低角点
位于 `y=438`，所以真实空气间隙为 `18` cells。间隙按 `resolution-y` 的 `3%`
缩放，并按旋转 OBB 的最低角点计算；改变倾角不会悄悄缩短指定落差。脚本还要求
冰块与弥散水气界面至少间隔默认的 10-cell 界面带。

按默认格子重力估算，冰角约在第 `1960` 步进入弥散界面、在第 `2940` 步到达
`phi=0.5` 名义水面，后者约对应 `frame_00028.png`。撞击速度约为
`0.01225 cells/step`，低于默认刚体限速；这只是用于定位输出的自由落体估算，
不是数值解断言。默认 500 帧用于观察入水、下潜和回升，不代表已达到稳定漂浮。

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
  --progress
```

快速检查可以缩小网格；所有几何比例和默认落差会同步缩放：

```bash
python iceflow2d/examples/ice_fall_2d.py \
  --resolution-x 120 --resolution-y 60 \
  --drop-height-cells 10 \
  --frames 2 --steps-per-frame 5 \
  --phase-warmup-steps 5 --no-progress
```

该小网格命令只用于检查入口和输出链路；固定 5-cell 界面宽度下，`12 x 3` 冰块
不足以作为定量物理工况。入水峰值载荷还会受到二值 moving boundary、原生
cut-link 几何和全局水量投影影响，因此本例应作为定性耦合演示，而不是冲击力标定基准。

也可以用 `--drop-height-cells` 指定绝对格子间隙，或用带符号的
`--initial-horizontal-speed`、`--initial-vertical-speed` 设置初速度。二者合速度
不能超过求解器默认的 `0.08 cells/step` 低 Mach 上限。

自由落水算例当前默认关闭 `well_balanced_hydrostatics`。若在
`create_ice_fall_config` 中启用它，相场 warm-up 和水量投影之后，算例会冻结初始
水平池的 `rho_ref/p_H`，并令动态压力残差为零。水动力 populations 采用 Liang
等人的 pressure--momentum（P--rho*u）
两相模型：材料密度仍由
`rho=rho_air+(rho_water-rho_air)phi` 给出，population 的一阶矩为 `rho*u`，二阶矩为
`rho*u*u+p_dyn*I`，而零阶矩不是材料密度。这样不会把 800:1 的物性密度跳跃误解释为
`p=c_s^2 rho` 的理想气体压力跳跃。

该 well-balanced 方案把静水参考从 population 中分裂出去：bulk source 使用
`(rho-rho_ref)g`，cut-link traction 只补回 `p_H`；因此静水压力恰好计算一次。动态压力则由 population 的
二阶矩直接传入 MEM，不再在 bulk 中显式计算 `-grad(p_dyn)/rho`。

碰撞采用 Liang 平衡态和源项的 MRT 扩展：剪切矩保持物理 `tau=0.5+3nu`，bulk 与
非水动力 ghost 矩独立松弛；界面处按论文插值运动黏度，而不是人为抬高液相黏度。
启用该选项时，默认 500 步相场 warm-up 是静水平衡参考构造的一部分，不应在定量运行中缩短。
