# IceFlow2D v3 模拟器原理与逐步验证指南

> **版本说明（2026-10-15）**：本指南正文及配套 PDF 保留 2026-10-14 的二次重构快照。当前生产代码已按新要求改为固定 3×3 候选和单精度线性拟合，所有相场、毛细应力、密度源项及压力恢复共用一个梯度字段。当前算法及验证记录见 [README 的统一梯度说明](../README.md)。下文两套内部梯度、5×5 二次拟合、QR 和壁面零空间修正均属于历史实现，源码指纹也对应该历史快照。

`iceflow2d_v3/simulator.py` 模拟二维容器中的空气、水和自由运动的冰块。水与空气用格子玻尔兹曼方法推进，气液界面用相场表示，冰块按刚体运动，热量先使冰升温，再使冰质量减少。**v3 的融化产物直接消失**：它不生成融水，也不把消失材料的动量和显热补入流体。这一假设决定了哪些检查可以要求守恒，哪些检查只能要求局部离散关系正确。

本指南面向第一次接触这类程序的读者。公式以当前工作区的实现为准；带“连续形式”说明的方程用于理解离散算法的目标，具体数值更新仍以随后给出的离散公式为准。阅读顺序建议：先读第 1 至 4 节，再沿第 5 至 7 节对照 `step()`，最后使用第 9 节的检查方法。末尾附录提供完整的方法定位索引。

代码快照日期：2026 年 10 月 14 日。具体源码指纹见同目录 `simulator_method_source_manifest.json`；工作区文件变动后，应重新核对方法位置与公式。

本次同步覆盖数据优先的 `connected_ls` 梯度重构及其单/四冰块溃坝验证。下表把当前生产算法与尚未实施的后续方案分开；旧版硬约束重构仅作为历史对照。

| 内容 | 当前实现 | 阅读位置 |
|---|---|---|
| 局部水柱的初始压力 | 空气静水参考加上逐列水相超额密度积分；满宽水池保留原参考 | 3.1 |
| 容器墙接触 | 位置投影加接触点的平移/转动法向冲量 | 6.3 |
| `connected_ls` 相场梯度 | 连通流体数据优先的二次拟合；单行显式闭合；仅在数据零空间内最小化壁面残差 | 5.10 |
| 相场梯度缓存的时序 | 碰撞前、流送后、每次运动/侵蚀几何与补点提交后刷新 | 4、5.8、6.5 |
| 梯度与相场有界性 | 梯度系数不按固体侧极值整体缩放；标量 ghost 单独限幅，流送后的相场仍用原越界修复 | 5.8、5.10 |
| `connected_ls` 新格点补点 | 双缓冲分轮传播，只读上一轮已完成的连通供体 | 6.5 |
| `consistent_flux` | 已存在的相场质量通量选项继续使用；仍是四个轴向面上的动量修正 | 8.3 |
| 整块守恒重映射、孤立气穴的亚网格状态、逐 D2Q9 链路限流及动量闭合 | 尚未实施；数据优先梯度本身不提供这些保证 | 9.6 |

`connected_ls` 是本次溃坝入口的默认值，并非所有入口的默认值。三项开关彼此独立：

| 入口 | 相场梯度 `phase_gradient_scheme` | 相场/动量耦合 `phase_momentum_coupling` | 新节点分布 `exposed_node_refill` |
|---|---|---|---|
| 直接构造 `IceFlowConfig()` | `legacy` | `legacy` | `equilibrium` |
| 单冰块和四冰块落水 CLI | `legacy` | `consistent_flux` | `equilibrium` |
| 溃坝 CLI / `create_config()` | `connected_ls` | `consistent_flux` | `equilibrium` |

溃坝默认目标仍为 8 s。数据优先重构下，**单冰块已按用户要求完成并停止于 5 s / 600000 步；四冰块在第 8532 步、0.0711 s 失稳，尚未完成 8 s**。旧版硬约束四冰块的第 8287 步失败属于历史结果。当前验证的具体范围及后续无供体气穴见第 9.5 节。

## 1 从一个格点开始理解模拟器

### 1.1 网格保存什么

将容器划分为许多正方形格子，每格边长为 $\Delta x$。世界网格的索引是 $(i,j)$，$i$ 向右增长，$j$ 向上增长；格心的格子坐标是

$$
\boldsymbol x_{ij}=(i+\tfrac12,\ j+\tfrac12).
$$

每一个流体格点保存水比例 $\phi$、速度、压力，以及两组各 9 个数的“分布函数”。这些数是恢复宏观量和搬运信息的数值变量，不必把它们想成真实的粒子数。

- `water_phase` 即 $\phi$：$\phi=1$ 表示水，$\phi=0$ 表示空气，介于二者之间表示气液过渡层。
- `phase_populations` 记为 $h_q$：其和恢复水比例。
- `momentum_populations` 记为 $f_q$：其一阶矩恢复动量，压力由另一组合恢复。
- `solid_mask` 标记当前冰占据的格点；`wall_mask` 标记容器墙。两者都是几何约束。

**水比例和冰比例是两种不同的量。** `water_phase` 区分水和空气；材料冰比例决定冰还剩多少以及冰壁在哪里。冰块内部保存的 `water_phase` 是被覆盖前的环境状态，不参与当前流体演化，也不能算进当前水量。

本指南统一用 $f_q$ 表示动量分布、$h_q$ 表示相场分布；部分诊断脚本把动量分布称为 `g`，它仍对应同一个 `momentum_populations` 字段。第 8.5 节的热学 $g_q$ 是另一套 D2Q5 能量分布。

### 1.2 为什么要保存两套网格

世界网格固定在容器里，水和空气在它上面运动。冰的材料网格跟随冰块平移和转动：同一个材料格始终代表同一小块冰，直到它融化消失。这避免了每次移动冰块都重新搬运冰的质量与热量。

设材料参考原点为 $\boldsymbol O$，转角为 $\theta$，旋转矩阵为

$$
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}.
$$

材料坐标 $\boldsymbol\xi$ 与世界坐标的关系是

$$
\boldsymbol x=\boldsymbol O+R\boldsymbol\xi,
\qquad
\boldsymbol\xi=R^T(\boldsymbol x-\boldsymbol O).
$$

材料网格中的质量、热量是权威状态；世界网格中的冰覆盖率只是将材料状态按当前姿态插值后的几何表示。多冰块时，各块材料矩形放在同一张二维数组中，相邻矩形间留一列空格。空格的冰比例为零，因此不会通过数组邻接关系把一块冰的热传给另一块冰。

对应代码：`MovingBodyThermal2D.__init__`、`_body_local_coordinates`、`_body_grid_coordinates`、`_sample_body_fraction`。

### 1.3 先掌握六个术语

| 术语 | 在本程序中的含义 |
|---|---|
| 碰撞 collision | 在同一个格点内，按局部压力、速度和物性重新组合分布函数 |
| 流送 streaming | 将碰撞后的数沿格子方向传给邻格 |
| 矩 moment | 分布函数乘上方向分量的幂再求和，如 $\sum_q c_{qx}f_q$ |
| 显热 sensible heat | 相对于熔点的升温或降温能量 |
| 潜热 latent heat | 冰已到熔点后，融掉单位质量需要的能量 |
| 光栅化 rasterization | 将当前材料形状和姿态映射成世界网格上的覆盖率与固体掩码 |

## 2 符号与单位

### 2.1 格子单位不是国际单位

流体和刚体力学主要使用格子单位，记为下标 $L$。一个 LBM 步的格子时间长度为 1，一个格子边长也为 1；物理长度和时间分别为 $\Delta x$ 和 $\Delta t$。代码采用

$$
\Delta t=\frac{u_{\mathrm{ref},L}\Delta x}{u_{\mathrm{ref}}},
\qquad u_{\mathrm{ref},L}=0.1.
$$

这个参考速度用于选择单位换算，不是每格流体的真实速度。主要换算为

$$
\boldsymbol x_{\mathrm{phys}}=\Delta x\boldsymbol x_L,
\quad
\boldsymbol u_{\mathrm{phys}}=\frac{\Delta x}{\Delta t}\boldsymbol u_L,
\quad
\omega_{\mathrm{phys}}=\frac{\omega_L}{\Delta t},
$$

$$
\boldsymbol g_L=\boldsymbol g_{\mathrm{phys}}\frac{\Delta t^2}{\Delta x},
\quad
\nu_L=\nu_{\mathrm{phys}}\frac{\Delta t}{\Delta x^2},
\quad
p_{\mathrm{phys}}=\rho_w\left(\frac{\Delta x}{\Delta t}\right)^2p_L,
$$

$$
\sigma_L=\frac{\sigma_{\mathrm{phys}}\Delta t^2}{\rho_w\Delta x^3}.
$$

流体密度以物理水密度为参考，因此 $\rho_{w,L}=1$，$\rho_{a,L}=\rho_a/\rho_w$。下文流体、碰撞、刚体公式省略 $L$ 时均使用格子单位；热学公式使用物理单位，并明确写出 $\Delta x$ 和子步时长。

溃坝例中，$\Delta x=1/3000\ \mathrm m$、$u_{\mathrm{ref}}=4\ \mathrm{m/s}$，所以

$$
\Delta t=\frac1{120000}\ \mathrm s,\qquad
1\ \mathrm{LU\ velocity}=40\ \mathrm{m/s},\qquad
0.1\ \mathrm{LU\ velocity}=4\ \mathrm{m/s}.
$$

界面厚度 $W=5$ 对应约 1.667 mm。物理黏度并未因溃坝例而增加：初始 90°C 水的 $\nu_w\approx3.14173258\times10^{-7}\ \mathrm{m^2/s}$，20°C 空气的 $\nu_a\approx1.45064520\times10^{-5}\ \mathrm{m^2/s}$。因此 $\tau_w\approx0.500070689$、$\tau_a\approx0.503263952$。它们是剪切松弛时间，不是声学松弛率 0.25；显示色标上限也不参与这些换算或碰撞。

### 2.2 二维质量与热量按单位厚度计算

这是二维模型，垂直于画面的厚度取 1 m。面积 $\Delta x^2$ 对应每单位厚度的体积，因此：

- `water_volume_m2` 的单位为 $\mathrm{m^2}$，表示水截面积，也可理解为每米厚度的体积。
- `body_solid_mass` 的单位为 $\mathrm{kg/m}$。
- `body_sensible_energy`、`water_sensible_energy` 的单位为 $\mathrm{J/m}$。
- 内部热通量的单位为 $\mathrm{W/m}$。

设物理冰质量为 $M'$，物理转动惯量为 $I'$，均按单位厚度计，则

$$
M_L=\frac{M'}{\rho_w\Delta x^2},
\qquad
I_L=\frac{I'}{\rho_w\Delta x^4}.
$$

不要直接把材料网格中的物理质量代入格子刚体方程。

### 2.3 关键字段对照

| 代码字段 | 数学含义 |
|---|---|
| `water_phase` | 水比例 $\phi$ |
| `momentum_velocity_lattice` | 未加半步力修正的速度 $\boldsymbol v$ |
| `fluid_acceleration_lattice` | 此步碰撞时保存的格子加速度 $\boldsymbol a$ |
| `physical_velocity_lattice` | 流体有效速度 $\boldsymbol u=\boldsymbol v+\boldsymbol a/2$ |
| `pressure_lattice` | 总压力 $p=p_H+p_d$ |
| `reference_pressure_by_row` | 初始化后冻结的静水压力 $p_H(j)$ |
| `reference_density_by_row` | 初始化后冻结的参考密度 $\rho_H(j)$ |
| `phase_gradient_lattice` | `connected_ls` 中供相场碰撞和毛细应力使用的梯度 $\boldsymbol G$ |
| `source_gradient_lattice` | `connected_ls` 中供密度源项和压力恢复使用的梯度 $\boldsymbol D$ |
| `phase_wall_geometry[...,q]` | 第 $q+1$ 条邻接链路的冰/墙法向及交点相对位移 |
| `phase_donor_mask` | 5×5 局部窗口中可用供体的 25 位标志 |
| `phase_reconstruction_coefficients` | 边界局部多项式的五个系数 |
| `phase_gradient_status` | 内部模板、单行闭合、满秩/不完整二次或孤立常值闭合的状态码 |
| `phase_gradient_spatial_rank` | 流体供体偏移张成的空间维数；内部模板记 2 |
| `phase_gradient_data_rank` | 加入显式曲率闭合后的拟合矩阵秩；内部模板记 5 |
| `phase_gradient_boundary_residual` | 当前多项式在八条壁面链路上的最大法向导数残差 |
| `refill_state` / `refill_next_state` | 两轮补点之间的 $(\hat\phi,p_d,u_x,u_y)$ 状态缓冲 |
| `stability_velocity` | 本轮相场碰撞前缓存的有效速度，供通量/HRR 使用 |
| `phase_mass_flux` | `consistent_flux` 测得的格点非对流质量通量 $\boldsymbol j$ |
| `body_reference_origin` | 材料参考原点 $\boldsymbol O$ |
| `body_local_center_of_mass` | 材料坐标中的剩余冰质心 $\boldsymbol\xi_c$ |
| `body_center` | 世界质心 $\boldsymbol C=\boldsymbol O+R\boldsymbol\xi_c$ |
| `body_velocity` | 世界质心速度 $\boldsymbol V$ |
| `hydrodynamic_impulse` | 一个格子时间步的边界冲量累加 $\boldsymbol J$ |
| `hydrodynamic_torque` | 一个格子时间步的角冲量累加 $J_\theta$ |

`physical_velocity_lattice` 中的 “physical” 表示已经作力修正的有效速度；字段本身仍是格子单位。转成 m/s 仍需乘 $\Delta x/\Delta t$。固体和墙处，这个视图返回零；查询冰壁速度应使用 `_body_velocity_at`。

字段 `body_center`、温度、壁面和静水参考场等由 `_DerivedField` 即时计算。它们通常不是另一份需要同步的 GPU 数组。单冰块可用 `[None]`，多冰块必须按稳定的零基 `body_id` 访问；`BodyField.array()` 总是保留冰块维度。

上述重建字段只在选择 `connected_ls` 时分配；`phase_mass_flux` 只在选择 `consistent_flux` 时分配。两个名称相似的梯度在边界附近相同，在完整流体邻域内使用不同模板，见第 5.10 节。

## 3 初始化建立什么状态

`IceFlow2D.__init__` 不仅分配内存，还建立能直接推进的初始状态。顺序如下。

1. 校验配置并初始化 Taichi CUDA。当前实现要求自由运动冰块、`body_ale` 材料热网格、`unified` cut-link 边界、静水平衡拆分和线性热浮力。
2. 建立单位换算，分配 $f_q,h_q$、碰撞缓冲区、压力、相场、几何以及各块冰的刚体状态。
3. 初始化各块冰的速度、转角、参考原点、质量和惯量；材料格点先填入初始冰质量。
4. 光栅化初始几何、处理接触、建立固体掩码。此时冰覆盖率已经来自材料网格。
5. 初始化气液指示场和分布函数。流体初速度为零，冰格点保存环境相场并显示刚体速度。
6. 若 `phase_warmup_steps > 0`，只推进相场，得到平滑的初始气液界面。这些预热迭代不计入 `simulation.steps`，也不推进热量或刚体位置。
7. 恢复宏观场并限幅相场，然后初始化水显热和材料冰显热。这样热状态对应的是预热结束后的水比例。
8. 从材料质量重新计算质心和惯量；按当前初始相场建立冻结的静水参考场及初始压力；将流体分布函数设为静止的对应压力平衡态。满宽水池的动态压力为零，局部水柱的动态压力一般非零。

`connected_ls` 在初始掩码建立后准备供体连通关系和壁面几何。每次相场预热迭代先准备梯度，再碰撞、流送、恢复相场、修复越界。预热结束后才初始化热状态和水柱压力，避免使用尚未平滑的初始相场积分压力。

### 3.1 静水参考压力如何产生

**满宽水池分支。** 条件是 `water_width == nx - boundary_cells`。初始化时对每一行的有效流体格点求平均水比例 $\bar\phi_j$，得到

$$
\rho_H(j)=\rho_a+(\rho_w-\rho_a)\bar\phi_j.
$$

没有有效流体格点的行使用初始水面指示值。相邻行的参考压力满足梯形积分

$$
p_H(j+1)-p_H(j)=\frac{g_y}{2}\,[\rho_H(j)+\rho_H(j+1)].
$$

代码在 `water_height` 对应的初始液面处设置零压基准，并对上下第一层采用半格距离。参考密度和压力以后不随液面移动而重算。流体分布函数只携带 $p_d=p-p_H$，而冰的边界受力会补回 $p_H$ 的牵引。这让初始静水负载从演化变量中拆出，同时保留静水对冰的作用。

**局部水柱分支，本次新增。** 溃坝的右侧初始为空气，不能把水柱与空气的行平均密度作为全域参考，否则干侧也会承受不合适的水柱参考负载。代码改为 $\rho_H(j)=\rho_a$，仍按上述半格/梯形规则积分空气参考压 $p_H(j)$。

设 $A_{ij}=1$ 表示有效流体、其余为零，水相相对于空气的超额密度为

$$
\delta\rho_{ij}=A_{ij}(\rho_w-\rho_a)\hat\phi_{ij}.
$$

从域顶向下对每一列求和，水柱的初始动态压力为

$$
p_{d,ij}^{0}=-g_y\left[\sum_{k=j}^{n_y-1}\delta\rho_{ik}
-\frac12\delta\rho_{ij}\right],\qquad p_{ij}^{0}=p_H(j)+p_{d,ij}^{0}.
$$

减去半个本格密度，是从格心而非格底积分的半格修正。$g_y<0$，所以水柱下部的超额压力为正。初始化动量分布携带这份非零 $p_d^0$；撤闸后它的水平梯度驱动溃流。这里没有模拟闸板撤离，也没有每步重新积分水柱压力：$p_H$ 冻结，$p_d$ 随 LBM 演化。

验证入口：`_build_hydrostatic_reference`、`_initialize_hydrostatic_equilibrium`，以及 `test_dambreak.py` 中 `DamBreakCudaTests` 的 `test_initial_column_hydrostatic_pressure`。满宽池检查 $p-p_H=0$；水柱检查 $\rho_H=\rho_a$、逐列积分、干侧超额压力接近零，以及两种分支的初始流体速度均为零。不要对溃坝强制要求所有初始动态压力为零。

### 3.2 溃坝几何与算法参数的区别

当前入口设置 150×50 mm、450×150 网格，内墙层为 3 格。水柱范围 $x=[1,66]$ mm、$y=[1,46]$ mm；四块 10×10 mm 冰的中心为 $(85,6)$、$(103,6)$、$(121,6)$、$(139,6)$ mm。这些是初始条件。将水柱宽度、冰块位置或输出时长改为其他值，不会改写碰撞公式。

`phase_gradient_scheme="connected_ls"` 改变梯度和补点算法；`phase_momentum_coupling="consistent_flux"` 改变相场碰撞时机及动量源项；二者是不同的数值选项。速度图的水色标 0–0.8 m/s、空气色标 0–4 m/s，以及涡量色标仅属于输出设置，不截断求解器中的实际速度。

## 4 step 的完整调用顺序

`step(n)` 中的 $n$ 是 **LBM 步数**，不是秒，也不是热子步数。它接受非负 Python 整数或 NumPy 整数，拒绝布尔值、浮点数和负数。`step(0)` 不改变状态。对一次迭代，源码调用关系为

```text
step
  _advance_fluid
    _collide_momentum
      _prepare_phase_gradients           [connected_ls：准备本轮梯度缓存]
      _prepare_capillary_stress
      [cm_hrr 或 consistent_flux] _cache_stability_velocity
      [consistent_flux] _collide_phase(gradients_prepared=True)
        _collide_phase_kernel
      [consistent_flux] _measure_phase_mass_flux
      _collide_momentum_kernel
        _prepare_momentum_cut_links
    [phase_momentum_coupling == legacy] _collide_phase(gradients_prepared=True)
    _stream
    _update_streamed_macroscopic_fields
    [phase_gradient_scheme == connected_ls]
      _bound_water_phase
      _prepare_phase_gradients           [用流送后的相场重建梯度]
      _update_pressure
    [phase_gradient_scheme == legacy]
      _update_pressure
      _bound_water_phase
  _advance_body
    _integrate_rigid_ice
    _refresh_fluid_domain(resolve_contact=True)
      _solve_contact_and_rasterize
      _update_solid_mask
      [connected_ls] _prepare_phase_geometry
      [connected_ls] _refill_connected_nodes
        _seed_connected_refill
        重复：_propagate_connected_refill -> 交换两轮缓冲
        _commit_connected_refill
      [connected_ls] _prepare_phase_gradients [用新几何和补点后的相场刷新]
      [legacy] _refill_changed_nodes
  _advance_thermal
    _advance_moving_thermal_fast
      _measure_water_outflow_rate
      [可选] 热对流前的状态诊断
      thermal.advance_fast
        重映射水量和显热
        对流子步：面通量 -> 更新 -> 重映射
    pending += 1
    [pending >= K] _advance_moving_thermal_slow(pending)
      thermal.advance_slow
        每个导热子步
          冰内导热与水内导热，或 D2Q5 导热
          冰水界面换热
          冰升温及潜热融化
          [D2Q5] 同步能量矩
          _refresh_eroded_body
            重算质量、质心、惯量
            更新几何并重建流体
            同步水量及显热
      pending = 0
  steps += 1
```

这里 $K=\texttt{update\_interval\_lbm\_steps}$。`IceFlowConfig()` 自带的热配置取 $K=8$；单独构造 `ThermalConfig()` 的默认值是 $K=1$。实际行为取决于传入的配置，不能把 8 写成算法固定常数。

同样要区分两个独立的 `legacy`：`phase_gradient_scheme` 控制如何求梯度及补点，`phase_momentum_coupling` 控制是否先碰撞相场并施加质量通量修正。一个选择 `legacy` 不会自动关闭另一个。实际默认值见开头的入口表；复现实验应把两项都写清楚。

### 4.1 先用一张表核对每个操作的输入输出

| 操作 | 为什么需要它 | 读取什么 | 更新什么 | 核对位置 |
|---|---|---|---|---|
| 准备几何/连通关系 | 避免重建跨冰借值 | 当前掩码、覆盖率、材料姿态 | 供体掩码、交点、法向 | 初始化及每次 `_refresh_fluid_domain` |
| 准备相场梯度 | 给界面压缩、密度源项及毛细应力共同提供空间信息 | 当前 $\phi$、最近一次已提交的几何 | 两套梯度、多项式系数、秩及壁面残差 | 5.10 |
| 缓存速度并碰撞相场 | 测量实际相场碰撞产生的质量通量 | $h_q,\phi,\boldsymbol v,\boldsymbol a$ | $h_q^*$、缓存速度、$\boldsymbol j$ | 5.5、8.3 |
| 碰撞动量及准备边界 | 产生黏性与压力响应，并求冰的负载 | $f_q,p_d,\rho,\boldsymbol G,\boldsymbol D,\boldsymbol j$ | $f_q^{prepared}$、本轮加速度、冰冲量 | 5.3、5.4 |
| 流送 | 搬运分布函数到邻格或反弹槽 | 两个碰撞缓冲区、当前几何 | $f_q,h_q$ | 5.6 |
| 宏观恢复 | 把分布函数还原成可用场 | 流送后的 $f_q,h_q$ | $\phi,\boldsymbol v$ | 5.7 |
| 限幅、重建梯度、恢复压力 | 使压力使用本轮新相场对应的边界梯度 | 新 $\phi,f_q,\boldsymbol v$ 及已保存加速度 | 合法相场、新梯度、总压力 | 5.8，顺序依选项 |
| 刚体积分、接触、补点 | 冰消费负载并改变流体域 | 本轮冰冲量、材料质量、当前流体 | 冰位姿/速度、新掩码、露出节点状态 | 6 |
| 快热过程 | 按新几何搬运水温 | 新流体域、有效速度、水显热 | 热水面积与水显热 | 7.2–7.4 |
| 慢热过程 | 导热、供热并侵蚀冰 | 累计时间、两套热网格 | 冰质量/显热及回调后的新几何 | 7.5–7.10 |
| 增加 `steps` | 只给完成整轮的步计时 | 前面操作均成功 | 完成步数与流动物理时间 | `step()` 末尾 |

表中的“准备几何”在刚体/侵蚀提交几何时执行，下一次碰撞准备梯度时复用它；几何没有变化时不重复求整个连通图。

### 4.2 每一步用的是哪个时刻的状态

设进入一轮时 `steps = n`。流体使用此时的冰几何与热状态，累加冰壁负载；刚体立即消费本轮负载，移动冰块并重建流体；随后热对流使用更新后的几何，以及流送后恢复的速度和已保存的加速度。若本轮触发慢热更新，则用本轮可见的流体与姿态处理累计时间段 $K\Delta t$。

这是按顺序分裂的显式耦合，并非同一时间步中反复迭代直至流动、刚体、热场同时收敛。新温度和新融化几何对动量碰撞的反馈出现在下一 LBM 步。

还要区分两次速度：相场碰撞及缓存使用进入本轮时的 $\boldsymbol u_h=\boldsymbol v^n+\boldsymbol a^{old}/2$；动量内核先组装新加速度，再用 $\boldsymbol u_f=\boldsymbol v^n+\boldsymbol a^{new}/2$。流送后得到 $\boldsymbol v^{new}$，此时有效速度为 $\boldsymbol v^{new}+\boldsymbol a^{new}/2$。源码没有在这三者之间作非线性迭代。

`connected_ls` 在流送后的相场修复之后刷新梯度并恢复压力；每次 `_refresh_fluid_domain` 又依次提交几何、供体关系和补点，最后刷新梯度。**正常公开 `step()` 返回时，梯度缓存对应最后一次几何/相场提交。** 若调试代码直接改写 `water_phase` 或掩码，需要分别刷新几何和梯度；这是调试写入的责任。记录中间阶段时不要额外重算：宏观量恢复后、准备新梯度前的缓存仍属于碰撞时层，正是该阶段真实使用的状态。

若某个子阶段抛出异常，`steps += 1` 不会执行，但此前已更新的数组也不会自动回滚。因此失败报告中的“尝试第 $n+1$ 步”和 `simulation.steps=n` 可以同时成立；复算必须恢复完整同状态，而不能仅修改步号。

### 4.3 两个时间和三个计数不要混淆

$$
t_{\mathrm{flow}}=\texttt{simulation.steps}\,\Delta t,
\qquad
t_{\mathrm{slow}}=\texttt{simulation.thermal.time\_s}.
$$

从初始化开始仅通过公开 `step()` 推进时，一轮完成后有

$$
t_{\mathrm{flow}}-t_{\mathrm{slow}}
=\texttt{\_thermal\_lbm\_steps\_pending}\,\Delta t.
$$

水显热对流每步都执行，因此即使 `thermal.time_s` 暂时落后，也不表示热字段完全停留在旧时刻。它只记录已经完成的慢过程时间。`thermal.steps` 记录导热子步数，而非 LBM 步数。

例：$K=8$，先 `step(3)` 再 `step(5)`，慢过程在总第 8 步触发，与直接 `step(8)` 的调度一致。`step(3)` 结束时不会自动补做剩余 3 步的导热。

## 5 第一阶段推进流体与气液界面

下面按功能解释公式，碰撞先后取决于耦合选项。默认溃坝的 `consistent_flux` 顺序是：准备梯度和应力（5.2）→相场碰撞（5.5）→质量通量测量（8.3）→动量碰撞及冰壁处理（5.3、5.4）→流送（5.6）→宏观恢复（5.7）→分支对应的压力/相场处理（5.8）。选择耦合 `legacy` 时，相场碰撞移到动量碰撞之后。两条路径每步都只碰撞相场一次。

### 5.1 D2Q9 的九个方向

D2 表示二维，Q9 表示每格 9 个方向。方向和权重为

$$
\begin{aligned}
\boldsymbol c_0&=(0,0),\\
\boldsymbol c_{1,2,3,4}&=(1,0),(0,1),(-1,0),(0,-1),\\
\boldsymbol c_{5,6,7,8}&=(1,1),(-1,1),(-1,-1),(1,-1),
\end{aligned}
$$

$$
w_0=\frac49,\qquad w_{1\ldots4}=\frac19,\qquad w_{5\ldots8}=\frac1{36},\qquad c_s^2=\frac13.
$$

反方向编号是 $(0,3,4,1,2,7,8,5,6)$。$c_s$ 是离散模型的格子声速，不是物理水中声速。源码定义在 `lattice.py`。

### 5.2 准备梯度、界面应力与流体力的公式

`_collide_momentum` 先调用 `_prepare_phase_gradients`，再调用 `_prepare_capillary_stress`。前者在 `legacy` 下不做额外计算，在 `connected_ls` 下建立本轮相场与密度源项的梯度缓存。界面附近的 $\phi$ 有空间梯度，梯度大的区域形成表面张力。程序使用切向界面应力

$$
K=\kappa\left(|\boldsymbol G|^2I-\boldsymbol G\boldsymbol G^T\right),
\qquad \kappa=\frac32\sigma W,
$$

$$
(K_{xx},K_{xy},K_{yy})
=\kappa(G_y^2,-G_xG_y,G_x^2).
$$

$W=\texttt{interface\_width}$，单位为格子。`legacy` 以及 `connected_ls` 的完整流体邻域内，$\boldsymbol G$ 的实际离散式是

$$
\boldsymbol G_i=\sum_{q=1}^{8}w_q\boldsymbol c_q
\left[2(\phi_{i+q}-\phi_i)+\frac12(4\phi_{i+q}-\phi_{i+2q}-3\phi_i)\right].
$$

这里 $i+q$ 简写世界格点 $\boldsymbol x_i+\boldsymbol c_q$。`legacy` 对墙和冰挡住的邻点使用第 5.9 节的镜像延拓；`connected_ls` 在边界邻域使用第 5.10 节拟合得到的梯度，不执行原标量镜像插值。随后 `_capillary_force_density` 按共享链路计算

$$
\boldsymbol F_{\sigma,i}
=\sum_{q=1}^{8}3w_q(K_i+K_{i+q}^{\mathrm{ext}})\boldsymbol c_q.
$$

有效流体邻点的 $K^{\mathrm{ext}}$ 是邻格应力；边界使用反射延拓。在流体内部，每条链路两端的贡献等大反向。若 $\sigma=0$，这组应力与力均为零。水平平界面的梯度只有 $y$ 分量，因而 $K_{yy}=0$，这避免对平界面制造法向毛细驱动。

流体密度由水比例决定：

$$
\hat\phi=\min(1,\max(0,\phi)),\qquad
\rho=\rho_a+(\rho_w-\rho_a)\hat\phi.
$$

`_fluid_force_and_gradient` 组装单位格子面积的力密度，再除以密度得到加速度：

$$
b(\phi)=\min(1,\max(0,2\hat\phi-1)),
$$

$$
\boldsymbol F_g=(\rho-\rho_H)\boldsymbol g
-b(\phi)\rho_w\beta_T(T_w-T_{\mathrm{ref}})\boldsymbol g,
$$

$$
\boldsymbol a=\frac{\boldsymbol F_g+\boldsymbol F_\sigma}{\rho}.
$$

第一项是相对于冻结静水参考的密度负载；第二项是线性热浮力。密度 $\rho$ 本身仍由相场混合，温度只通过该附加力产生浮力效应。$b$ 使空气温度不会错误地驱动气液过渡层的气侧。热浮力中的 $T_w$ 来自上一次可见的水显热状态。

同一函数还返回用于密度源项的中心梯度

$$
\nabla_c\phi_i=3\sum_{q=1}^{8}w_q\boldsymbol c_q(\phi_{i+q}-\phi_i).
$$

记该密度源项梯度为 $\boldsymbol D$。它与上述两层模板的 $\boldsymbol G$ 在完整流体邻域内不同；`connected_ls` 在边界邻域令二者等于同一个拟合多项式的中心梯度。

**验证。** 纯相且无重力扰动时，毛细应力应为零；水平平界面不应产生明显毛细加速度；冰壁接触处的毛细反作用力和力矩应与流体贡献匹配。

### 5.3 动量碰撞：按局部物性更新分布

碰撞不是让冰块彼此相撞。这里的碰撞是调整同一格点的 $f_q$，让它既输运动量，又给出指定的黏度和压力行为。

先定义

$$
\boldsymbol v=\frac{\sum_q\boldsymbol c_qf_q}{\rho},
\qquad \boldsymbol u=\boldsymbol v+\frac12\boldsymbol a,
\qquad p_d=p-p_H.
$$

$\boldsymbol u$ 的半步力修正与源项、压力恢复配套。碰撞使用的平衡分布是

$$
f_q^{\mathrm{eq}}=3(w_q-\delta_{q0})p_d
+w_q\rho\left[3\boldsymbol c_q\!\cdot\!\boldsymbol u
+\frac92(\boldsymbol c_q\!\cdot\!\boldsymbol u)^2
-\frac32|\boldsymbol u|^2\right].
$$

$\delta_{q0}$ 在 $q=0$ 时为 1，其余方向为 0。由方向的对称性可得到

$$
\sum_qf_q^{\mathrm{eq}}=0,\quad
\sum_q\boldsymbol c_qf_q^{\mathrm{eq}}=\rho\boldsymbol u,\quad
\sum_q\boldsymbol c_q\boldsymbol c_q^Tf_q^{\mathrm{eq}}
=p_dI+\rho\boldsymbol u\boldsymbol u^T.
$$

所以本程序不能用 $\rho=\sum_qf_q$，也不能用普通等温 LBM 的 $p=c_s^2\rho$。平衡态零阶矩为零，并不要求实际非平衡分布的和每时刻都为零。$f_q$ 为负也不能单独作为失稳依据。

碰撞中的方向源项为

$$
S_q=3w_q\left[\boldsymbol c_q\cdot\boldsymbol F
+(\rho_w-\rho_a)(\boldsymbol c_q\cdot\boldsymbol u)
(\boldsymbol c_q\cdot\nabla_c\phi)\right],
\qquad\boldsymbol F=\rho\boldsymbol a.
$$

第二项处理相场对应的密度梯度，不等同于附加重力。

#### 黏度如何进入碰撞

$$
\nu(\phi,T)=\hat\phi\nu_w(T)+(1-\hat\phi)\nu_a,
\qquad \tau=\frac12+3\nu_L,
\qquad s_\nu=\frac1\tau.
$$

温度相关水黏度和气黏度在第 8.1 节列出。当前代码只使用分子黏度，没有额外的界面黏度下限、冰面黏度或 Smagorinsky 涡黏度。

#### 把九个分布函数变成九个矩

令矩参考速度为 $\boldsymbol U$，则

$$
m_{ab}=\sum_q(c_{qx}-U_x)^a(c_{qy}-U_y)^bf_q,
\qquad a,b\in\{0,1,2\}.
$$

九个矩在代码中的顺序为 00、10、01、20、02、11、21、12、22。

`rm_mrt` 取 $\boldsymbol U=0$，是原始矩；`cm_mrt` 和 `cm_hrr` 取 $\boldsymbol U=\boldsymbol u$，是中心矩。平衡矩 $m^{\mathrm{eq}}$ 与源矩 $S_m$ 也由同一套投影计算。每一类矩按

$$
m^*=m-s(m-m^{\mathrm{eq}})+(1-\tfrac12s)S_m
$$

更新，再由 `reconstruct_central_moment_population` 逆变换回 $f_q^*$。星号表示“碰撞后、流送前”。不同矩使用不同松弛率，这就是 MRT。

| 矩或矩组合 | `rm_mrt` | `cm_mrt` |
|---|---|---|
| $m_{00},m_{10},m_{01}$ | $s_\nu$ | 1 |
| $m_{20}-m_{02},m_{11}$，即偏应力 | $s_\nu$ | $s_\nu$ |
| $m_{20}+m_{02}$，即迹 | 1 | 声学松弛率 $s_a$ |
| $m_{21},m_{12}$ | 1 | 1 |
| $m_{22}$ | 1 | 声学松弛率 $s_a$ |

代码默认使用 `cm_mrt`，默认 $s_a=0.25$。中心矩低阶三个矩统一取 1，以保持逆平移后的原始动量增量与力一致。迹与四阶矩同步使用声学率，因为它们共同影响压力模式。修改 $s_a$ 不等于修改分子剪切黏度。`cm_hrr` 的额外处理见第 8.2 节。

**验证。** 将生产碰撞与独立矩阵形式比较；在施加源项时检查原始动量增量；由剪切应力衰减或剪切波反算 $\nu$，比只检查配置中的 $\tau$ 更有说服力。

### 5.4 动量碰撞末尾：准备冰壁反弹与刚体负载

动量碰撞的最后调用 `_prepare_momentum_cut_links`。若方向 $q$ 的邻格是冰，那么流体格心到邻格格心的线段穿过冰表面，称为 cut-link。

设某块冰的世界覆盖率为 $F_b$，锐界阈值为 $\eta=\texttt{solid\_liquid\_threshold}$。程序定义

$$
d_b=(\eta-F_b)\Delta x,
\quad
\lambda=\operatorname{clip}_{[0.05,0.95]}
\left(\frac{d_f}{d_f-d_s+10^{-12}}\right),
\quad d_f=\max(d_b(\boldsymbol x_f),10^{-6}\Delta x).
$$

$d_s$ 是冰邻格的水平集值。$d_b$ 有长度单位，但**不是到冰表面的真实欧氏距离**；它是覆盖率阈值的水平集，只用来插值边界位置。$\lambda$ 的上下限避免交点太靠近格心产生极端插值。

冰壁交点与速度为

$$
\boldsymbol x_B=\boldsymbol x_f+\lambda\boldsymbol c_q,
\qquad
\boldsymbol u_B=\boldsymbol V+\omega\,\mathcal P(\boldsymbol x_B-\boldsymbol C),
\qquad \mathcal P(x,y)=(-y,x).
$$

设 $\bar q$ 是 $q$ 的反向，旧分布非平衡部分为

$$
f_q^{\mathrm{neq}}=f_q-f_q^{\mathrm{eq}}(p_d,\rho,\boldsymbol u).
$$

实际写入反弹槽的值为

$$
f_{\bar q}^{B}
=\frac{f_{\bar q}^{\mathrm{eq}}(p_d,\rho,\boldsymbol u_B)
+f_q^{\mathrm{neq}}+\lambda f_{\bar q}^{*}}{1+\lambda}.
$$

这里用的是旧的 $f_q$ 和尚未被边界改写的 $f_q^*$。程序在流送前完成该计算，以免并行流送覆盖仍需要读取的旧分布。反弹值暂存于本格的出射槽 $q$，实际流送时才写入本格的反向槽 $\bar q$。

链路给冰的冲量包括动态部分和冻结静水压力牵引：

$$
\boldsymbol J_q=(\boldsymbol c_q-\boldsymbol u_B)f_q^*
-(-\boldsymbol c_q-\boldsymbol u_B)f_{\bar q}^{B}
+6w_qp_{H,B}\boldsymbol c_q,
$$

$$
p_{H,B}=(1-\lambda)p_H(\boldsymbol x_f)+\lambda p_H(\boldsymbol x_s),
\qquad
J_{\theta,q}=(\boldsymbol x_B-\boldsymbol C)\times\boldsymbol J_q.
$$

二维叉积 $\boldsymbol r\times\boldsymbol J=r_xJ_y-r_yJ_x$ 是一个标量。当前格子时间步长为 1，因此冲量在刚体更新中不再额外乘物理 $\Delta t$。

若存在表面张力，还要向冰补入阻塞链路毛细力的反作用：

$$
\boldsymbol J_{\sigma,q}=-3w_q(K_f+K_g)\boldsymbol c_q.
$$

$K_g$ 是镜像应力。它的力臂用链路中点 $\boldsymbol x_f+\boldsymbol c_q/2$，并加上角冲量闭合项

$$
J_{\theta,\sigma,q}
=(\boldsymbol x_f+\tfrac12\boldsymbol c_q-\boldsymbol C)\times\boldsymbol J_{\sigma,q}
-\frac32w_q\boldsymbol c_q\times[(K_f-K_g)\boldsymbol c_q].
$$

这与动量反弹交点的力臂不同。删掉补项会改变接触处的角动量平衡。所有链路按 `solid_body_id` 累加到对应冰块；冲量与角冲量采用 f64 原子加法，减轻大量正负贡献相消时的舍入误差。

**验证。** 对单条链路手算反弹值、静水牵引和力矩，再比较生产代码；多冰块检查同一水格点指向不同冰块的链路各归其主。

### 5.5 相场碰撞：输运并维持气液界面

`_collide_phase` 决定水比例如何随流动搬运，同时使气液过渡层维持有限厚度。用于理解的连续形式是守恒 Allen–Cahn 输运

$$
\partial_t\phi+\nabla\cdot(\phi\boldsymbol u)
=\nabla\cdot\left[M(\nabla\phi-\Lambda\boldsymbol n)\right],
\qquad \Lambda=\frac{4\phi(1-\phi)}{W}.
$$

$M=\texttt{mobility}$ 是格子迁移率：梯度项平滑界面，$\Lambda\boldsymbol n$ 压缩界面，二者共同形成过渡层。该连续式描述离散方法的目标，并不表示代码直接对右侧做有限差分。

`_collide_phase` 现在是一个 Python 包装方法，实际 GPU 碰撞在 `_collide_phase_kernel` 中执行。单独调用包装方法会先准备梯度；主循环已经准备过时传入 `gradients_prepared=True`，避免重复计算。`legacy` 内核使用第 5.2 节的两层模板；`connected_ls` 直接读取 `phase_gradient_lattice`，再求法向。默认有正则化：

$$
\boldsymbol n=\frac{\boldsymbol G}{\sqrt{|\boldsymbol G|^2+\epsilon_n^2}+10^{-14}},
\qquad \epsilon_n=\texttt{phase\_normal\_epsilon}.
$$

若 $\epsilon_n=0$，分母改为 $|\boldsymbol G|+10^{-14}$。正则化让极小梯度的法向长度小于 1，避免均匀区的微小数值噪声获得完整压缩强度；它也使实际离散稳态不必严格等于未正则化的解析剖面。

平衡相场分布和方向源项为

$$
h_q^{\mathrm{eq}}=w_q\phi\left[1+3\boldsymbol c_q\cdot\boldsymbol u
+\frac92(\boldsymbol c_q\cdot\boldsymbol u)^2-\frac32|\boldsymbol u|^2\right],
\qquad S_{h,q}=w_q\Lambda\boldsymbol c_q\cdot\boldsymbol n.
$$

它们满足 $\sum_qh_q^{\mathrm{eq}}=\phi$、$\sum_qS_{h,q}=0$。转到以 $\boldsymbol u$ 为参考的中心矩后，零阶矩直接设为 $\phi$；一阶矩用

$$
s_\phi=\frac1{3M+1/2},\qquad
m_{10,h}^*=m_{10,h}-s_\phi(m_{10,h}-m_{10,h}^{\mathrm{eq}})
+(1-\tfrac12s_\phi)S_{10,h}
$$

更新，$m_{01,h}$ 同理。其余高阶矩使用松弛率 1，即 $m_h^*=m_h^{\mathrm{eq}}+S_h/2$。逆变换后再次用静止方向闭合

$$
h_0^*=\phi-\sum_{q=1}^{8}h_q^*.
$$

求和用 f64，再写回 f32。该修正保持碰撞前后的局部相场总量，并避免每步重复求和带来的偏置累积；它不保证移动边界更新后的全局水量。

在默认 `legacy` 耦合中，相场在动量碰撞后碰撞，使用本轮更新的加速度。若选择 `consistent_flux`，相场碰撞移入 `_collide_momentum`，在本轮动量碰撞前只执行一次；详情见第 8.3 节。

### 5.6 流送与反弹

`_stream` 采用 push 流送。对流体到流体链路，

$$
f_q(\boldsymbol x+\boldsymbol c_q,t+1)=f_q^{\mathrm{prepared}}(\boldsymbol x,t),
\qquad
h_q(\boldsymbol x+\boldsymbol c_q,t+1)=h_q^*(\boldsymbol x,t).
$$

`prepared` 在普通链路上就是 $f_q^*$，在冰链路上已经是第 5.4 节的反弹值。邻格为墙、冰或域外时，不写入邻格，而写入当前格的反方向。

容器墙处相场直接反弹。冰壁移动时，相场反弹带有速度修正：

$$
h_{\bar q}^{B}=h_q^*-6w_q\phi\boldsymbol c_q\cdot\boldsymbol u_B.
$$

因此“碰撞守恒相场”不能直接推导成“移动冰壁和格点启闭也严格守恒水量”。整个推进还包含壁面修正、格点覆盖和露出，以及局部限幅。

缓冲区把碰撞写出的数据和流送要覆盖的旧分布分开。每个有效目的槽由唯一来源写入，避免边流送边读取已被邻线程改写的数。

当前 `_stream` 以目的格点是否为有效流体判断链路是否流送，尚未复用 `connected_ls` 的连续对角可见性检查。后者只限制重建供体；不能把“供体连通”理解成已经统一了所有 D2Q9 输运链路。普通流体链路上的净相场迁移量可诊断为 $H_{ij}=h_q^*(i)-h_{\bar q}^*(j)$，但生产代码目前没有存储或限幅这组逐链路通量。

### 5.7 恢复相场与速度

`_update_streamed_macroscopic_fields` 对流送后的分布求和：

$$
\phi^{\mathrm{new}}=\sum_qh_q^{\mathrm{new}},
\qquad
\rho^{\mathrm{new}}=\rho(\operatorname{clip}(\phi^{\mathrm{new}},0,1)),
$$

$$
\boldsymbol v^{\mathrm{new}}=\frac{\sum_q\boldsymbol c_qf_q^{\mathrm{new}}}
{\max(\rho^{\mathrm{new}},10^{-12})}.
$$

此时保存的 $\boldsymbol a$ 仍是本轮碰撞算出的值，供随后压力和热输运的半步修正使用。固体格点显示冰的刚体速度；墙速度为零，墙相场为零。冰内部的环境相场不在这里清除。

### 5.8 修复相场、更新梯度并恢复压力

本节的操作顺序依 `phase_gradient_scheme` 分支而不同。`legacy` 仍先恢复压力、再修复相场；`connected_ls` 先修复相场、再准备新梯度、最后恢复压力。这样压力使用的是已修复的新相场对应的边界重建，而不会沿用碰撞前的梯度缓存。

`_update_pressure` 不通过求和全部 $f_q$ 获得密度，而按

$$
\nabla\rho=(\rho_w-\rho_a)\nabla_c\phi,
\quad s_0=-\frac32w_0|\boldsymbol u|^2,
$$

$$
p_d=\frac35\left[\sum_{q=1}^{8}f_q
+\frac12\boldsymbol u\cdot\nabla\rho+\rho s_0\right],
\qquad p=p_H+p_d
$$

恢复压力。速度是本轮流送后的 $\boldsymbol v+\boldsymbol a/2$。源码中的 `0.6` 就是 $3/5$。墙压力设为零，冰格点的压力槽保留已有值。

`_bound_water_phase` 只修复有效流体格中 $\phi<0$ 或 $\phi>1$ 的值：

$$
\phi_b=\operatorname{clip}(\phi,0,1),\qquad
\tilde h_q=h_q+h_q^{\mathrm{eq}}(\phi_b,\boldsymbol v)-h_q^{\mathrm{eq}}(\phi,\boldsymbol v),
$$

$$
\tilde h_0\leftarrow\tilde h_0+\phi_b-\sum_q\tilde h_q.
$$

注意此处使用的是 $\boldsymbol v$，不是额外重算的半步速度。所有区间内的相场和分布保持原样，包括很小的水相尾部；不做尾部截断。由于密度计算已限幅，这次修复不改变密度，因而不改动 $f_q$。

`water_phase_cutoff` 只决定哪些格点参加热输运及其 CFL 检查，不是相场限幅阈值。局部限幅会改变相场总量，不能将它当作全局水量修正。

### 5.9 边界邻点如何保持中性润湿

本节先描述保留的 `legacy` 标量算法，再说明当前两种选项共用的应力延拓。`connected_ls` 的梯度算法在下一节展开。

梯度模板会访问墙或冰内部。`_phase_sample_inside_walls` 将采样索引跨半格墙镜像，例如左墙内边界为 $i=\ell$ 时，$i<\ell$ 的索引映射为 $2\ell-1-i$。镜像表达零法向导数，同时保留切向变化。

冰处先由覆盖率梯度求局部法向 $\boldsymbol n_B$，沿 cut-link 求交点 $\boldsymbol x_B$，再将采样点镜像为

$$
\boldsymbol x'=\boldsymbol x-2[(\boldsymbol x-\boldsymbol x_B)\cdot\boldsymbol n_B]\boldsymbol n_B.
$$

`legacy` 的 `_ice_phase_ghost_legacy` 对镜像点周围的有效流体双线性插值，插值的是相对于当前格的差值，以保持纯相常数场。其供体只检查“该点是流体”，未检查它与中心格点是否隔着冰。镜像点靠近地板时，双线性插值的域外格点还会先经过地板索引反射；这可能让原本位于冰另一侧的流体成为供体，是历史第 8544 步问题的来源。

应力不是标量，`_ice_capillary_stress` 使用

$$
Q=I-2\boldsymbol n_B\boldsymbol n_B^T,\qquad K_g=QK_{\mathrm{mirror}}Q^T.
$$

选择 `connected_ls` 后，应力镜像使用未裁剪材料覆盖率估计的法向和实际阈值交点，镜像插值及固定墙反射的供体还受 `phase_donor_mask` 约束。标量 ghost 则返回下一节的局部多项式值。相场碰撞与毛细应力梯度直接读缓存，不再依靠这个标量 ghost 求梯度。

这里的中性润湿边界没有额外指定亲水或疏水接触角参数。把所有被挡采样改成中心值，会丢失切向信息；连通拟合的目标是保留有数据支持的变化，但当前过度约束仍可能将梯度锁成零，见下一节。

### 5.10 数据优先 connected_ls：连通供体、零空间壁面修正与诊断

源码位于 `phase_reconstruction.py`，通过 `IceFlow2D(PhaseReconstruction, StabilityOptions)` 继承。`connected_ls` 可理解为“只读局部连通流体的最小二乘拟合”。它没有改动完整流体区域的梯度模板、相场迁移率或动量剪切黏度。

先解释四个术语：**最小二乘**使供体预测误差的平方和尽量小；**秩**是数据能区分的独立系数方向数；**零空间**是改变系数而不改变现有数据预测的方向；**Neumann 条件**规定垂直于壁面的导数，本例以零导数为目标。当前实现先让流体数据确定可观测变化，再用壁面条件补充剩余信息。

#### 先准备哪些供体可以使用

`_prepare_phase_geometry` 在每个流体格点周围建立 5×5 窗口。偏移 $(d_i,d_j)$ 均在 $[-2,2]$ 内，以从中心出发的单调路径寻找供体：优先通过上下左右流体格；没有面连通路径时，允许沿合法的 D2Q9 对角链连接。路径不能穿固体，也不绕到窗口外。

供体标志编码为

$$
b(d_i,d_j)=5(d_i+2)+(d_j+2),\qquad
\mathrm{allowed}(d_i,d_j)=
\left[(\mathrm{mask}\gg b)\ \&\ 1\right]\ne0.
$$

中心自身也占一位，但不会当作拟合数据。因此“mask 中有 5 位”通常意味着中心加 4 个其他供体。

对角两端都是流体，仍可能在中间穿过冰。`_phase_visible_diagonal` 把线段写成 $\boldsymbol x(t)=(1-t)\boldsymbol x_0+t\boldsymbol x_1$，映射到每块冰的材料坐标，并在材料插值格线交叉处切分。双线性材料覆盖率沿每小段为二次函数

$$
F(t)=a t^2+b t+c.
$$

逐段检查端点和位于段内的极大点；任一活跃冰块的最大覆盖率达到 $\eta-10^{-6}$，就拒绝该对角供体通路。这里检查的是本代码的双线性材料覆盖率，不是解析多边形的精确距离。

#### 再准备壁面交点和法向

冰表面的几何闭合为覆盖率阈值 $F_b=\eta$。`_phase_ice_geometry` 从材料插值得到的未裁剪覆盖率，在流体与固体端点分别做 $x/y$ 两侧差分并相加，形成向量 $\boldsymbol d$；取

$$
\boldsymbol n_B=\frac{\boldsymbol d}{|\boldsymbol d|},\qquad
\zeta=\operatorname{clip}_{[0,1]}
\frac{\eta-F_b(\boldsymbol x_f)}{F_b(\boldsymbol x_s)-F_b(\boldsymbol x_f)},\qquad
\boldsymbol r_B=\zeta\boldsymbol c_q.
$$

零覆盖率梯度时用链路方向作法向；分母太小时交点比例退回 0.5。固定墙交点在真实半格墙面上，法向沿坐标轴。相场几何的 $\zeta$ 不使用动量 cut-link 的 $[0.05,0.95]$ 限制；二者用途不同，核对代码时不要混用。

#### 完整流体邻域保留原模板

若八个方向的第一、第二层采样点都在域内且为流体，状态标记为 `1`。此时 $\boldsymbol G$ 使用第 5.2 节的两层模板，$\boldsymbol D$ 使用中心模板；直接读取这些邻点，不作拟合。若第二层有固体而第一层没有壁面交点，也会进入拟合分支，但该点可以没有壁面残差行。内部点的空间秩/拟合秩字段固定记为 2/5，表示使用完整模板，并非实际执行了秩分解。

#### 先拟合连通流体数据

以当前格点为局部原点，固定常数项为实际相场 $\phi_i$，使用二次多项式

$$
\widehat\phi(x,y)=\phi_i+g_x\,x+g_y\,y
+\frac12H_{xx}x^2+H_{xy}xy+\frac12H_{yy}y^2,
$$

$$
\boldsymbol\beta=(g_x,g_y,H_{xx},H_{xy},H_{yy})^T,
\qquad B(x,y)=(x,y,x^2/2,xy,y^2/2).
$$

五个系数中，前两个为格心梯度，其余三个组成对称 Hessian $H$。对允许的非中心流体供体 $\boldsymbol r_k=(x_k,y_k)$，建立

$$
A_k=\frac{B(x_k,y_k)}{|\boldsymbol r_k|},\qquad
b_k=\frac{\phi_k-\phi_i}{|\boldsymbol r_k|}.
$$

最小化 $\|A\boldsymbol\beta-b\|_2^2$，等价于残差平方按距离平方倒数加权。只有连通流体点进入数据；非流体点不设为 $\phi=0$，也不作为数据。拟合函数自身不裁剪 $\phi_i$ 或 $\phi_k$，正常推进在第 5.8 节的规定时机修复相场。拟合、刷新梯度都不会修改相场字段。

#### 单行供体：保留切向信息，显式声明缺失曲率

若所有供体偏移共线，其空间秩为 1，记单位切向为 $\boldsymbol t=(t_x,t_y)$，垂向为 $\boldsymbol m=(-t_y,t_x)$。代码按 $s_k=\boldsymbol t\cdot\boldsymbol r_k$ 选择紧致供体：两侧都有供体时，各保留最近一点；只有一侧时，保留距离不超过最近距离两倍的供体。在当前 5×5 共线整数窗口中，后一种情形至多保留两个点。

单行数据无法区分完整法向曲率，代码增加明确的模型闭合

$$
\boldsymbol t^TH\boldsymbol m=0,
$$

其系数行为

$$
L_{tn}=(0,0,t_xm_x,t_xm_y+t_ym_x,t_ym_y).
$$

仅有一个供体时，再加 $\boldsymbol t^TH\boldsymbol t=0$，即

$$
L_{tt}=(0,0,t_x^2,2t_xt_y,t_y^2),
$$

以得到单侧一阶切向估计。两侧各一个供体时，格心加两侧点构成紧致二次差分；等距水平行的切向梯度为 $(\phi_{i+1}-\phi_{i-1})/2$。垂向信息仍依赖闭合和后续壁面条件，不能把单行拟合称为完整二维梯度的精确解。

将实际添加的闭合行组成 $L$，右端为零；没有闭合时只保留原数据：

$$
\widetilde A=\begin{pmatrix}A\\L\end{pmatrix},\qquad
\widetilde b=\begin{pmatrix}b\\0\end{pmatrix},\qquad
\boldsymbol\beta_0=\widetilde A^\dagger\widetilde b.
$$

这里 $\dagger$ 表示按数值秩求最小范数最小二乘解。二维供体若不足以确定全部五个系数，当前实现保留最小范数解和未确定方向，不再退回旧版线性分支。

#### 壁面条件只修正数据零空间

对于八个即时邻接方向中存在的墙/冰交点 $\boldsymbol r_B=(x_B,y_B)$，零法向导数的目标行为

$$
C_B=(n_x,n_y,n_x\,x_B,n_x\,y_B+n_y\,x_B,n_y\,y_B).
$$

将各行组成 $C$。先求 $Z$，其列是 $\ker\widetilde A$ 的正交基，再只允许

$$
\boldsymbol\beta=\boldsymbol\beta_0+Zz,
\qquad \min_z\|C(\boldsymbol\beta_0+Zz)\|_2^2.
$$

因此实际修正为

$$
z=-(CZ)^\dagger C\boldsymbol\beta_0,\qquad
\boldsymbol\beta=\boldsymbol\beta_0-Z(CZ)^\dagger C\boldsymbol\beta_0.
$$

没有供体时跳过壁面修正，返回孤立常值闭合；数据满秩时 $Z$ 没有可调方向，壁面条件不能覆盖数据解。其他情形的关键关系为

$$
\widetilde A\boldsymbol\beta=\widetilde A\boldsymbol\beta_0,
\qquad A\boldsymbol\beta=A\boldsymbol\beta_0.
$$

**保持的是数据拟合的预测，不是强行让每个供体残差为零。** 当二次多项式本就不能精确表示所有数据时，原有最小二乘残差仍存在；壁面修正不会进一步破坏该数据拟合。壁面行按链路等权使用，没有按连续表面分组，也没有大惩罚权重。

数据和壁面条件不兼容时，$C\boldsymbol\beta$ 可以非零。零法向导数是待最小化目标，当前实现不保证每条壁面链路都精确满足它；这避免把近似平行壁面的几何误差变成锁死流体切向变化的硬约束。

#### GPU 上如何求最小范数解

`_phase_minimum_norm_fit` 使用 f64 的列主元 modified Gram-Schmidt QR，并在列消元和行空间求解中进行两次正交化。它在行空间求最小范数系数，必要时再构造正交零空间；不形成正规方程，也不直接求近奇异矩阵的逆。数据矩阵预留 26 行：24 个非中心供体槽和最多两条曲率闭合，未使用槽置零。

设初始最大列范数为 $\ell_{max}$，列残差范数需满足

$$
\|a_{\mathrm{res}}\|>\tau\max(1,\ell_{max})
$$

才计入秩。数据阶段取 $\tau=10^{-10}$，壁面零空间阶段取 $10^{-8}$；零空间正交基筛选阈值为 $10^{-10}$。它们是 QR 的数值秩截断，不是旧版判断壁面夹角的 $10^{-3}$ 和硬约束独立性的 $10^{-6}$。最终系数和梯度写回 f32。

`_fit_boundary_phase(i,j,quadratic)` 保留历史接口参数 `quadratic`，当前函数不再以它切换线性/二次分支；实际准备内核统一请求上述数据优先二次重构。CPU 参考脚本用 SVD 实现同一数学目标，用于独立核对，生产内核不执行 CPU SVD。

#### 梯度、标量 ghost 和相场限幅分开处理

边界附近直接保存未整体缩放的系数，并令

$$
\boldsymbol G_i=\boldsymbol D_i=(g_x,g_y).
$$

物理梯度为格子梯度除以 $\Delta x$。完整流体邻域仍保留两套不同的原模板。旧版 `_bounded_phase_polynomial_scale` 已删除；当前不会为了让固体侧 $[-2,2]^2$ 上的多项式值处处位于 $[0,1]$ 而把梯度整体压小或压成零。

标量虚拟点只对返回值单独限幅：

$$
\phi_{ghost}(x,y)=\operatorname{clip}_{[0,1]}
[\phi_i+B(x,y)\boldsymbol\beta].
$$

当前相场碰撞、密度源项和毛细应力直接读梯度缓存，不依靠该限幅 ghost 重新差分。ghost 的有界性不保证 LBM 流送后的相场有界；原相场越界修复仍按第 5.8 节执行，逐链路限流尚未实现。

#### 状态和质量字段怎样读取

| `phase_gradient_status` | 当前含义 |
|---|---|
| 0 | 墙/冰等非流体，未重建 |
| 1 | 完整流体邻域，使用原模板 |
| 2 | 单行供体及显式曲率闭合，不代表完整二维信息 |
| 3 | 加入闭合后的数据矩阵秩为 5 的二次拟合 |
| 4 | 二维供体但二次拟合秩不足，保留未确定方向 |
| 5 | 无非中心供体，返回孤立常值闭合 |

`phase_gradient_spatial_rank` 为 0、1 或 2；边界点的 `phase_gradient_data_rank` 是 $\widetilde A$ 的数值秩，**包含显式曲率闭合行**，不是纯流体数据的秩。内部模板固定写 2/5，非流体写 0/0。状态 5 现在表示孤立点；有供体的均匀相场也可给出零梯度，但不会因此成为状态 5。

`phase_gradient_boundary_residual` 保存

$$
R_B=\max_{q=1,\ldots,8}|C_q\boldsymbol\beta|.
$$

没有壁面行、内部模板和非流体的该字段记零，不能单凭零残差推断梯度准确。应同时检查空间秩、拟合秩、数据分辨率和壁面残差；尤其单行供体的完整法向信息依赖闭合。

#### 旧问题修正与当前验证边界

旧版先把壁面条件设为硬约束，再在其零空间拟合流体数据；近似平行壁面可能耗尽梯度自由度，且整个多项式的有界缩放还可能抹掉密度源项需要的梯度。上述数据优先顺序及取消整体缩放已替换这两条路径。旧版状态 5 可以来自硬约束锁死；阅读历史第 8277 步等记录时，不能套用当前状态码解释。

验证入口为 `test_phase_reconstruction.py`，当前 12 项测试涵盖仿射切向场、微倾斜单行缝隙、独立 SVD 对照、二次角点、纯相/均匀混合物、隔冰供体拒绝、孤立点闭合、固体侧多项式极值不抹掉梯度、多轮补点、对角可见性和非零表面张力的平界面。数学复算一致只说明实现符合所选重构及闭合，不证明欠分辨气隙中的完整物理梯度、全程稳定性或守恒已得到保证。单/四冰块实测见第 9.5 节。

## 6 第二阶段推进刚体并重建流体域

### 6.1 步骤一消费本轮流体负载

`_integrate_rigid_ice` 读取刚刚由动量碰撞累加的冲量。设冰质量为 $M$，转动惯量为 $I$，线速度、角速度为 $\boldsymbol V,\omega$，阻尼系数为 $d_v,d_\omega$，则

$$
\boldsymbol V^{+}=d_v\left(\boldsymbol V+\frac{\boldsymbol J+M\boldsymbol g}{M}\right),
\qquad
\omega^{+}=d_\omega\left(\omega+\frac{J_\theta}{I}\right),
$$

$$
\boldsymbol C^{+}=\boldsymbol C+\boldsymbol V^{+},
\qquad \theta^{+}=\theta+\omega^{+},
\qquad
\boldsymbol O^{+}=\boldsymbol C^{+}-R(\theta^{+})\boldsymbol\xi_c.
$$

先更新速度，再用更新后的速度推进位置，是一种一阶的半隐式 Euler 刚体积分。上述 $M\boldsymbol g$ 已是一个格子时间步的重力冲量。默认配置 $d_v=d_\omega=1$，不额外衰减刚体速度；水动力及接触仍会改变平移与转动。若显式启用额外角阻尼，应按物理衰减率 $\gamma$（单位 $\mathrm{s^{-1}}$）设 $d_\omega=\exp(-\gamma\Delta t_{\mathrm{SI}})$，以保持不同时间步下的衰减时间尺度一致。此时角阻尼会额外耗散旋转。

每次积分后将 `hydrodynamic_impulse` 和 `hydrodynamic_torque` 清零，使下一轮重新累加，而不会重复施加上一轮负载。

积分只对同时满足以下条件的冰块执行：剩余惯量足以判为活跃，而且当前仍存在锐界固体格点，即有可解析的水动力边界。若还有材料质量却没有固体格点，则固定参考姿态，将线速度和角速度置零，继续保留热状态及连续覆盖率换热。这样避免不可见的亚网格残冰只受重力而穿出容器。

**验证。** 临时给一块冰指定已知 $\boldsymbol J,J_\theta$，按公式计算一步变化；检查积分后累加器为零。对亚网格残冰检查材料质量仍存在、姿态不动且仍能吸热。

### 6.2 步骤二将剩余材料投影到世界网格

`_calculate_moving_body_fraction` 对每块冰 $b$ 计算覆盖率 $F_b(i,j)$。设材料格中心的插值坐标为 $(a,b)$，则

$$
F_b(i,j)=\operatorname{clip}_{[0,1]}
\sum_{r,s\in\{0,1\}}w_x(r)w_y(s)f^{\mathrm{ice}}_{a_0+r,b_0+s},
$$

$a_0=\lfloor a\rfloor$、$b_0=\lfloor b\rfloor$；$w_x(0)=1-(a-a_0)$、$w_x(1)=a-a_0$，$w_y$ 同理；材料矩形外贡献为零。材料格的剩余冰比例为

$$
f^{\mathrm{ice}}=\frac{\operatorname{clip}(m,0,m_0)}{m_0},
\qquad m_0=\rho_{\mathrm{ice}}\Delta x^2.
$$

多冰块的合并覆盖率取 $F=\max_bF_b$，不是各块覆盖率相加。接触处理完成后，`_commit_moving_body_raster` 才把墙内覆盖率清零。这样墙面约束可以看到尚未裁掉的穿墙部分。

### 6.3 步骤三处理容器墙与冰块接触

`_solve_contact_and_rasterize(resolve_contact=True)` 在提交新几何前处理接触。墙面支撑范围由当前覆盖率达到阈值的世界格点对应的单位方格求出；已融掉的格点不再参与。它不使用一直不变的初始矩形。

墙面约束先根据剩余锐界支撑范围，把质心投影回允许区间。这保证位置没有越墙，但单独截断质心速度还不够：即使 $V_y=0$，旋转的冰角仍可能向地板运动。本次新增 `_reduce_wall_contact_ranges` 和 `_wall_contact_impulse`，在支撑面上选择朝墙运动最强的接触点，并同时调整平移和转动。

设指向容器内部的单位法向为 $\boldsymbol n$，接触点相对质心的力臂为 $\boldsymbol r$，$\ell=\boldsymbol r\times\boldsymbol n$。接触点法向速度为

$$
v_n=\left(\boldsymbol V+\omega\mathcal P\boldsymbol r\right)\cdot\boldsymbol n
=\boldsymbol V\cdot\boldsymbol n+\omega\ell.
$$

当 $v_n<0$ 时，代码计算无回弹、无摩擦冲量

$$
J_n=\frac{-v_n}{1/M+\ell^2/I},\qquad
\boldsymbol V' =\boldsymbol V+\frac{J_n}{M}\boldsymbol n,\qquad
\omega'=\omega+\frac{J_n\ell}{I}.
$$

代回可得当前接触点 $v_n'=0$。分母包含转动惯量，墙通过偏离质心的接触点也能改变角速度。底/顶/左/右墙的法向分别为 $(0,1),(0,-1),(1,0),(-1,0)$。

具体选点不是解析多边形接触：从当前锐界支撑单元取得切向端点区间 $[s_{min},s_{max}]$，令 $k=n_y-n_x$，$\ell=ks$。若 $k\omega>0$ 选择 $s_{min}$，若 $k\omega<0$ 选择 $s_{max}$，否则选择中点。一次墙投影内对四面墙重复四轮冲量处理；外层再与冰—冰接触交替迭代。纯冲量更新不需要重写材料参考原点，只有位置投影改变姿态时才重新光栅化。

多冰块接触由 `contact.py` 的 `BodyContacts` 执行。先用包围盒缩小搜索范围，再寻找实际锐界单元的重叠；从四个轴向中选择需要平移分离距离最小的方向。法向 $\boldsymbol n$ 指向冰块 A 的分离方向，接触点为重叠格心的平均值。接触点相对质心的力臂为 $\boldsymbol r_A,\boldsymbol r_B$，相对法向速度为

$$
v_n=\left[\boldsymbol V_A+\omega_A\mathcal P\boldsymbol r_A
-\boldsymbol V_B-\omega_B\mathcal P\boldsymbol r_B\right]\cdot\boldsymbol n.
$$

当 $v_n<0$，即两块冰正在相互靠近，施加无回弹的法向冲量

$$
J_n=\frac{-v_n}{1/M_A+1/M_B
+(\boldsymbol r_A\times\boldsymbol n)^2/I_A
+(\boldsymbol r_B\times\boldsymbol n)^2/I_B}.
$$

A 增加 $J_n\boldsymbol n/M_A$，B 减少 $J_n\boldsymbol n/M_B$；角速度按对应力臂和惯量更新。没有切向摩擦冲量。穿透深度为 $\delta$ 时，位置修正是

$$
\Delta\boldsymbol O_A=(\delta+10^{-4})\frac{M_B}{M_A+M_B}\boldsymbol n,
\qquad
\Delta\boldsymbol O_B=-(\delta+10^{-4})\frac{M_A}{M_A+M_B}\boldsymbol n.
$$

两块冰的总质量加权位置保持不变。墙约束和冰块对接触反复交替，每次位置改变都重新光栅化；超过 `body_contact_iterations` 且仍有冰块重叠时抛出异常。接触几何只有格子尺度精度，没有亚网格润滑膜、连续多边形精确接触、碎裂或粘连。

融化回调使用 `resolve_contact=False`：按既有姿态刷新覆盖率，不再次积分运动或求接触冲量；墙支撑缓存会重置，下一次刚体阶段再重新建立。

### 6.4 步骤四提交固体掩码并识别格点启闭

`_update_solid_mask` 先备份旧 `solid_mask` 和旧 `solid_body_id`，再根据各块当前覆盖率建立

$$
S_{ij}=1\quad\Longleftrightarrow\quad
\text{非墙格点且有活跃冰块满足 }F_b(i,j)\ge\eta.
$$

非冰格点的 `solid_body_id=-1`。程序按所有冰块更新后的联合几何判断启闭，避免把被另一块冰占据的位置回填为流体。持续流体格点不需要重建。

几何体积由 `phase_change_geometry_volume_cells()` 返回，统计锐界固体格点的数量。

材料体积由 `phase_change_solid_volume_cells()` 返回，来自材料质量除以冰密度。双线性插值和阈值光栅化会使二者不相等。融化比例应以材料质量为依据，而不是仅数冰格点。

### 6.5 步骤五给新露出的格点重建环境流体

两种梯度选项采用不同的补点实现。共同点是：检查所有冰块联合更新后的所有者变化，新变为冰的格点更新刚体速度并清零流体加速度；新变为流体的格点重建环境状态。它们都没有按融掉的冰量生成液水，也没有全局水量投影。

**保留的 `legacy` 分支。** `_refill_changed_nodes` 对新变为流体的格点使用 `_extrapolate_exposed_fluid`。

外推寻找最近一个存在“持续流体”的方形环：邻点必须在本轮前后都是流体，不能使用本轮同批新填的格点。该环的状态做等权平均：

$$
(\phi,p_d,v_x,v_y)_{\mathrm{new}}
=\frac1N\sum_{k\in\mathcal N}(\hat\phi,p-p_H,v_x,v_y)_k.
$$

找不到邻居时保留本格已有的限幅环境相场、动态压力残量和旧所属冰块在该点的当前刚体速度。通常第一环即可；整块冰消失可能露出多层格点，所以搜索半径可扩大到整个域尺度。

获得环境状态后，设置

$$
p_{\mathrm{new}}=p_{d,\mathrm{new}}+p_H(j),\qquad
f_q=f_q^{\mathrm{eq}}(p_{d,\mathrm{new}},\rho(\phi_{\mathrm{new}}),\boldsymbol v_{\mathrm{new}}),
\qquad h_q=h_q^{\mathrm{eq}}(\phi_{\mathrm{new}},\boldsymbol v_{\mathrm{new}}),
$$

并将加速度置零。外推的是动态压力，再加当前位置的静水参考值，避免把冰原位置的静水压随冰一起搬运。若开启 `non_equilibrium`，还补入邻域应力，第 8.4 节说明其公式。

水下露出的格点因邻域是水而恢复为水，空气中露出的格点因邻域是空气而恢复为空气。该过程没有读取“这一轮融掉多少冰”来决定回填量；运动和融化共用同一规则。

**当前 `connected_ls` 分支。** `_refresh_fluid_domain` 在更新掩码后先调用 `_prepare_phase_geometry`，再调用 `_refill_connected_nodes`，补点完成后调用 `_prepare_phase_gradients`。补点逐轮传播已有流体状态，有供体的点正常提交，剩余孤立分量暂缓激活：

1. `_seed_connected_refill`：持续流体状态写入缓冲并标为有效，新暴露节点标为待填。状态是 $(\hat\phi,p_d,u_x,u_y)$，这里的 $\boldsymbol u$ 是有效速度，而 `legacy` 平均的是未加半力的 $\boldsymbol v$。
2. `_propagate_connected_refill`：待填节点只找固定 3×3 内的连通有效供体。只读上一轮缓冲，结果写入下一轮缓冲。上一轮有效的点原样复制，不再修改。
3. `_refill_connected_nodes`：每轮检查未填节点数。若未填数量不再减少，调用 `_defer_unresolved_refill`，将剩余点的 `solid_mask` 标为 2，停止本轮传播；仍有进展时不按轮数上限提前截断。
4. `_commit_connected_refill`：对已获得状态的新点一次性写回相场、动态压力加当前位置的参考压、平衡 $f_q/h_q$ 和速度；加速度清零，并保留可选非平衡应力回填。若启用补点矩投影，重算梯度后再匹配源项和半力对应的矩。

设第 $r$ 轮状态/有效标志为 $z_i^{(r)},v_i^{(r)}$，$z=(\hat\phi,p_d,u_x,u_y)$。待填点选取一层连通的有效供体集 $\mathcal N_i^{(r)}$，则

$$
z_i^{(r+1)}=\frac1{|\mathcal N_i^{(r)}|}\sum_{k\in\mathcal N_i^{(r)}}z_k^{(r)},\qquad
v_i^{(r+1)}=1.
$$

已有有效点满足 $z_i^{(r+1)}=z_i^{(r)}$。每轮全体完成后才交换缓冲，因此不会在同一 GPU 内核里读取另一个线程刚填的状态。可以跨多轮填完一次露出的多层节点，但每次读取仍局限于 3×3 连通窗口。

无可达供体的新点暂不参加流体碰撞、迁移和水侧热输运，对外使用静止封闭边界；不指定其相态，也不将其计入真实冰的质量、几何体积、接触支持或刚体受力。掩码 0/1/2 分别表示正常流体、真实冰和待激活点。下一次几何刷新按真实冰覆盖率恢复候选掩码，旧待激活状态保存在 `previous_solid_mask` 中，防止未初始化点被当作供体。连通后这些点重新补点，包含上一轮与本轮 `solid_body_id` 均为 -1 的情况。物理冰的运动和材料热更新持续推进。

这是针对少量孤立格点的几何近似，会暂时少解析一部分流体体积，并不保证封闭腔压力或严格相变质量守恒。实现复用原有掩码和 3×3 双缓冲传播；额外维护首次待激活步数及事件、激活次数和最长等待时间，用于检查近似是否仍局限于少量节点。

#### 当前补点为何仍可能制造不连续

双缓冲解决的是并行读取顺序和跨冰供体问题，不等于整块连续或守恒重建。一旦得到一个有效供体，格点就固定下来；最近环有供体时，也不会再考虑远一环的另一端状态。两端不同状态向中间传播，仍可能在接合处产生台阶。

数据优先重构下，四冰块在第 8520 步慢热侵蚀后打开 `(154,3)`。冰面与地板的交点间距约为 0.50014 格；沿缝隙的相邻点为：

| 格点 | 补点后相场 | 混合密度（LU） | 实际切向梯度（LU） |
|---|---|---|---|
| `(153,3)` | 1 | 1 | −0.24782 |
| `(154,3)` | 0.500054 | 0.500678 | −0.49995 |
| `(155,3)` | 0.000107 | 0.001357 | −0.24997 |

新点恰好由湿侧和干侧状态等权平均，只有一个中间格承担过渡。梯度的独立 SVD 重算与 GPU 结果相符，但气侧密度很低、过渡欠分辨，现有分布函数/源项/压力更新仍会失稳。单项分布为负并非充分判据；应检查相场总和越界、净动量及几何更新前后的速度，具体过程见第 9.5 节。

旧版硬约束记录中，第 8276 步的 `(176,4)` 至 `(176,6)` 为湿侧 0.29335，`(176,7)` 至 `(176,9)` 为空气侧 0.000139，随后又有切向梯度被锁零的问题。这是旧算法的历史证据，不能作为当前第 8520 步实际梯度仍为零的依据。

本算法不从关闭节点收集水量、也不从持续流体扣除新填节点的水量，独立平均速度也不等于守恒动量重映射。上一轮讨论的“连通区域统一补点、水量/动量预算、共享通量限界”尚未实现。

**验证。** `test_phase_reconstruction.py` 检查多层传播只读上轮状态、拒绝真正隔离的供体；`test_thermal_coupling.py` 检查水下/空气中的几何消失与热状态衔接。另记录补点前后的有效流体水量 $\Delta x^2\sum\phi$，并明确当前代码不保证该差值为零。持续流体的 $f_q/h_q$ 不会被当前补点直接改写。

## 7 第三阶段推进热量与融化

### 7.1 热状态以质量和能量为基础

设熔点为 $T_m$、比热为 $c_w,c_i$、导热率为 $k_w,k_i$、潜热为 $L$。世界格点存水面积 $V$、水显热 $E_w$；材料格点存剩余冰质量 $m$、冰显热 $E_i$：

$$
E_w=\rho_w c_w V(T_w-T_m),
\qquad E_i=mc_i(T_i-T_m).
$$

有水或冰时温度按反函数读取：

$$
T_w=T_m+\frac{E_w}{\rho_wc_wV},
\qquad
T_i=\min\left(T_m,T_m+\frac{E_i}{mc_i}\right).
$$

没有水的格点温度视图显示设定的空气温度，但程序并不求解空气热输运。无剩余冰时冰温视图取熔点。常量物性默认值为 $c_w=4186$、$c_i=2100\ \mathrm{J/(kg\,K)}$，$k_w=0.60$、$k_i=2.20\ \mathrm{W/(m\,K)}$，$L=334000\ \mathrm{J/kg}$，$T_m=0\ ^\circ\mathrm C$，实际值取配置。

### 7.2 快过程先决定对流子步长度

每个 LBM 步都执行 `_advance_moving_thermal_fast`。先用 `_measure_water_outflow_rate` 对有热水体积的流体格，求其各个有效湿邻面向外速度之和：

$$
r_i=\sum_{f\in\text{四个有效湿邻面}}
\max(0,\boldsymbol u_f\cdot\boldsymbol n_f),
\quad \boldsymbol u_f=\frac{\boldsymbol u_i+\boldsymbol u_k}{2},
\qquad r_{\max}=\max_ir_i.
$$

速度使用格子单位。这是为了限制一格在一个迎风子步中流出的总比例，比只用本格 $|u_x|+|u_y|$ 更能处理面速度发散的情形。

允许的物理对流步长和所需子步数为

$$
\delta t_{\mathrm{adv,max}}=C_{\max}\frac{\Delta t}{r_{\max}},
\qquad
N_{\mathrm{adv}}=\max\left(1,\left\lceil\frac{\Delta t_{\mathrm{fast}}}
{\delta t_{\mathrm{adv,max}}}-10^{-14}\right\rceil\right),
$$

$\Delta t_{\mathrm{fast}}=\Delta t$，$C_{\max}=\texttt{max\_courant\_number}$ 默认 0.5；关闭对流或速度为零时步长上限取无穷。每个实际子步长是 $\delta t=\Delta t_{\mathrm{fast}}/N_{\mathrm{adv}}$。所需数量超过 `max_substeps_per_update` 时抛出异常，而不是悄悄限制子步数并继续不稳定计算。

可选的完整状态诊断由 `_thermal_stability_check_enabled` 控制，默认关闭；启用后检查流体速度、相场和相场分布是否有限，以及 $|u_x|+|u_y|$ 是否超过代码的异常阈值 16。**16 只用于发现严重失稳，不能理解为可接受的低 Mach 速度上限。** 即使完整诊断关闭，对流的面出流率约束仍执行。

### 7.3 快过程重映射水面积和显热

`synchronize_water_aperture` 将热水面积匹配到当前相场和冰掩码：

$$
V_{ij}^{\mathrm{target}}=
\begin{cases}
\Delta x^2\hat\phi_{ij},&\text{非墙、非冰且 }\hat\phi_{ij}>\phi_{\mathrm{cut}},\\
0,&\text{其余情况}.
\end{cases}
$$

这不是 $\Delta x^2\phi(1-F)$：当前实现用锐界固体掩码排除冰，用相场确定流体格中的水面积。小于湿阈值的相场尾部没有热水面积。

已有水的格点保持原有单位面积显热 $e=E_w/V$，因此保持原温度：

$$
E_w^{\mathrm{target}}=V^{\mathrm{target}}\frac{E_w^{\mathrm{old}}}{V^{\mathrm{old}}}.
$$

新湿格点没有原水面积时，从最近一个有旧水体积的方形环按水面积和距离加权：

$$
e_{\mathrm{new}}=\frac{\sum_{k\in\mathcal N}E_{w,k}^{\mathrm{old}}/r_k^2}
{\sum_{k\in\mathcal N}V_k^{\mathrm{old}}/r_k^2},
\qquad E_w^{\mathrm{target}}=V^{\mathrm{target}}e_{\mathrm{new}}.
$$

这里 $r_k^2=(\Delta i)^2+(\Delta j)^2$。该热外推与第 6.5 节的流体等权外推不同。它读取旧热状态，允许使用刚被冰覆盖但尚有旧水面积的格点；若整个域都找不到旧湿邻域，则 $e=0$，新水温为熔点。

`_build_phase_aperture_targets` 先计算目标能量，`_apply_phase_aperture_transfer` 再统一提交，以防并行更新中读到已经改写的邻格温度。该局部重映射既不投影全局水量，也不修正全局显热；覆盖或新增水面积会改变总量。

### 7.4 快过程用迎风通量输运水显热

对左格 L、右格 R 之间的 $x$ 面，速度先转成物理单位：

$$
u_f=\frac12(u_{x,L}+u_{x,R})\frac{\Delta x}{\Delta t}.
$$

若 $u_f\ge0$，上游格为 L，否则为 R。设其水开口比例 $a_{\mathrm{up}}=\operatorname{clip}(V_{\mathrm{up}}/\Delta x^2,0,1)$，则

$$
Q_{V,f}=u_fa_{\mathrm{up}}\Delta x,
\qquad Q_{E,f}=Q_{V,f}\frac{E_{w,\mathrm{up}}}{\max(V_{\mathrm{up}},10^{-30})}.
$$

$y$ 面同理。只有面两侧均为有效湿流体时才有对流通量；墙、冰和干侧面通量为零。按右减左、上减下的面差更新

$$
V^{*}_{ij}=V_{ij}-\delta t\left(Q_{V,i+1/2,j}-Q_{V,i-1/2,j}
+Q_{V,i,j+1/2}-Q_{V,i,j-1/2}\right),
$$

$$
E^{*}_{w,ij}=E_{w,ij}-\delta t\left(Q_{E,i+1/2,j}-Q_{E,i-1/2,j}
+Q_{E,i,j+1/2}-Q_{E,i,j-1/2}\right).
$$

原代码的面数组采用整数面下标，所以公式右面 $i+1/2$ 对应数组下标 `i+1`。通量已含面长 $\Delta x$，不再把差值除以面积：这里更新的是总面积与总显热，而不是每单位体积的密度。

每个对流子步之后再次执行第 7.3 节的重映射，把 $V^*$ 复位到当前 LBM 相场对应的目标面积，并按输运后温度重建 $E_w$。因此热子步的水面积是温度输运的辅助量，最终水面积由相场决定。单独的共享面通量更新在闭域内可相消，后续重映射则不保证全局面积和能量守恒。

**验证。** 初始均匀水温时，水量与显热按同一上游比例输运，更新和重映射后应保持温度；更强的验证给定温度梯度，检查能量沿速度方向输运以及出流率对子步数量的限制。

### 7.5 慢过程计算导热子步

当累计 LBM 步数达到 $K$ 时，`advance_slow` 处理物理时段

$$
\Delta t_{\mathrm{slow}}=\texttt{pending}\,\Delta t.
$$

导热率与热扩散率的关系是

$$
\alpha_w=\frac{k_w}{\rho_wc_w},\qquad
\alpha_i=\frac{k_i}{\rho_ic_i},\qquad
\alpha_{\max}=\max(\alpha_w,\alpha_i).
$$

Fourier 条件给出

$$
\delta t_{\mathrm{diff,max}}=Fo_{\max}\frac{\Delta x^2}{\alpha_{\max}},
\quad
N_{\mathrm{diff}}=\max\left(1,\left\lceil\frac{\Delta t_{\mathrm{slow}}}
{\delta t_{\mathrm{diff,max}}}-10^{-14}\right\rceil\right),
\quad \delta t=\frac{\Delta t_{\mathrm{slow}}}{N_{\mathrm{diff}}}.
$$

`max_fourier_number` 默认 0.15。这个条件控制显式体内导热；界面换热另有可用水显热限幅。子步超过配置上限同样报错。慢过程内部不重新流送动量，也不执行水显热对流。

### 7.6 每个慢子步先处理冰内与水内导热

默认 `enthalpy_fv` 使用有限体积法。冰材料格点 $r$ 与轴向邻格 $s$ 的有效接触比例为

$$
a_{rs}=\min(f_r^{\mathrm{ice}},f_s^{\mathrm{ice}}).
$$

其流出热功率和热量增量为

$$
Q_{r\to s}=k_i a_{rs}(T_r-T_s),
\qquad \Delta E_{i,r}^{\mathrm{cond}}=-\delta t\sum_sQ_{r\to s}.
$$

按二维单位厚度，面长和格间距离均为 $\Delta x$，两者相消，所以公式里没有额外 $\Delta x$。计算阶段材料质量和显热不改变，每格可以独立计算自己的通量和，不会读到邻线程修改后的温度。结果写入 `_body_heat_delta`，稍后和界面热量一起提交。

水的内部导热面取 $a_f=\min(V_L,V_R)/\Delta x^2$，有效湿邻格之间

$$
Q_{E,f}^{\mathrm{cond}}=-k_wa_f(T_R-T_L).
$$

仍按面通量差更新 $E_w$，但不改 $V$。水与空气之间绝热，水与冰的换热留到下一步骤，冰与冰接触也没有独立的界面热通量。

绝热墙面 $Q=0$。定温墙距第一层流体格心只有半格距离，指向水内部的热量为

$$
Q_{\mathrm{wall}\to w}=2k_wa_w(T_{\mathrm{wall}}-T_w).
$$

系数 2 来自半格距离；各方向面数组的正负号不同，但“水温低于墙温时向水供热”的物理符号相同。代码没有为冰—容器接触额外建立同样的定温墙换热。

若选择 `enthalpy_d2q5`，上述体内导热由五分布函数方案实现，冰水界面与潜热处理仍共用下面步骤；见第 8.5 节。

### 7.7 每个慢子步再处理冰水界面换热

`_exchange_interface_heat` 以每个湿世界格点为出发点，检查该格及四个轴向邻格对每块冰的覆盖率。包括本格是为了让未达到锐界阈值的残冰覆盖仍能换热。每个覆盖采样点再映射到最多 4 个材料格；因此每个水格对每块冰最多有 $5\times4=20$ 个热量请求。

冰水导热率取调和平均

$$
k_{wi}=\frac{2k_wk_i}{k_w+k_i}.
$$

设采样点冰覆盖率为 $F_b$、水开口比例为 $a_w$，则接触比例和材料分配比例为

$$
a_{\mathrm{contact}}=\min\left(a_w,\min(1,F_b/\eta)\right),
\qquad
\chi_r=\frac{w_xw_yf_r^{\mathrm{ice}}}{F_b}.
$$

单个材料格的请求热量为

$$
q_r^{\mathrm{req}}=\delta t\,k_{wi}\,a_{\mathrm{contact}}\,\chi_r
\max(0,T_w-T_{i,r}).
$$

注意是 $\max(0,\cdot)$：当前界面函数只让水向冰供热，没有反向冷却水到熔点以下或重新结冰的对称机制。这一界面接触与材料分配是格子尺度闭合，并非精确切割多边形上的表面积积分。

同一水格的所有冰块和所有材料请求必须一起限幅，防止被重复取走显热。令

$$
Q^{\mathrm{req}}=\sum_rq_r^{\mathrm{req}},\qquad
\gamma=\min\left(1,\frac{\max(0,E_w)}{Q^{\mathrm{req}}}\right)
$$

且零请求时取 $\gamma=0$。实际转移 $q_r=\gamma q_r^{\mathrm{req}}$，然后

$$
E_w\leftarrow E_w-\sum_rq_r,
\qquad \Delta E_{i,r}\leftarrow\Delta E_{i,r}+q_r.
$$

水线程只修改自身 $E_w$，材料格收到的多个请求用 f64 原子加法累加。该单次换热操作的水显热损失和材料热增量相等；这不等同于整个 v3 冰水系统总能量守恒。

**验证。** 对同一个水格同时接触两块对称冰，检查供热只限幅一次、两块分配对称，且 $\sum q_r\le\max(0,E_w^{\mathrm{before}})$。

### 7.8 每个慢子步将热量变成升温或质量消失

`_apply_body_heat_and_phase_change` 将冰内导热和界面热量一次合并，设

$$
\widetilde E_i=E_i+\Delta E_i.
$$

若 $\widetilde E_i\le0$，只更新显热 $E_i'=\widetilde E_i$，质量保持不变；若 $\widetilde E_i>0$，超过熔点所需的能量用于融化：

$$
m'=\max\left(0,m-\frac{\widetilde E_i}{L}\right),\qquad E_i'=0.
$$

此外，当 $m'\le10^{-6}m_0$ 时把 $m'$ 和 $E_i'$ 一起置零。这是清除数值残量，不是所有亚网格残冰都立即删除。若供热超过整格融化所需潜热，超出的热量直接丢弃，不返还给水或生成融水显热。

例：1 kg 冰从 $-10\ ^\circ\mathrm C$ 加热到熔点需要 $21000$ J；若再供 $167000$ J，最终剩 0.5 kg 冰，温度为熔点。按单位厚度和格点实际质量同比缩放即可核对源码。该函数只改材料质量和显热，不直接改水相场或流体分布。

### 7.9 每个慢子步立即重算剩余冰的质量属性

融化改变的不仅是冰形状，还有质量、质心和惯量。回调 `_refresh_eroded_body` 首先调用 `_update_moving_body_mass_properties`。

对同一冰块的材料格 $r$，其格子质量和材料格心坐标为 $m_{r,L}$、$\boldsymbol\xi_r$。代码归约

$$
M=\sum_rm_{r,L},\qquad
\boldsymbol\xi_c=\frac{\sum_rm_{r,L}\boldsymbol\xi_r}{M},
$$

$$
I_O=\sum_rm_{r,L}\left(|\boldsymbol\xi_r|^2+\frac{f_r^{\mathrm{ice}}}{6}\right),
\qquad I=\max(0,I_O-M|\boldsymbol\xi_c|^2).
$$

$f^{\mathrm{ice}}/6$ 是实现给出的材料格内惯量闭合；满格时为单位正方形的质心惯量项 $1/6$。不能把部分材料格理解为仍拥有完整单位正方形惯量，也不能直接用锐界世界格点归约替代它。初始完整矩形可恢复 $I=M(N_x^2+N_y^2)/12$。

活跃判据是质量归约时 $M>10^{-12}$，且 $I>10^{-14}\max(I_0,1)$。不满足惯量条件时储存惯量设零，速度和角速度归零。判据作用于格子单位。

材料参考原点和转角保持不动，材料质心从 $\boldsymbol\xi_c^{\mathrm{old}}$ 移到 $\boldsymbol\xi_c^{\mathrm{new}}$。为使存活材料继续保持原刚体速度场，质心速度应同步变为

$$
\Delta\boldsymbol C=R(\boldsymbol\xi_c^{\mathrm{new}}-\boldsymbol\xi_c^{\mathrm{old}}),
\qquad
\boldsymbol V^{\mathrm{new}}=\boldsymbol V^{\mathrm{old}}+\omega\mathcal P\Delta\boldsymbol C.
$$

这是重选质心的速度修正，不是额外施加融水冲量。代码没有强制保持融化前后的 $M\boldsymbol V$ 或 $I\omega$。

### 7.10 每个慢子步结束就刷新几何和热水区域

质量归约之后，回调依次：

1. 按新材料比例光栅化冰体，更新固体掩码。
2. 用同一套环境外推恢复新暴露流体格点。
3. 按新相场与掩码重映射热水面积和显热；D2Q5 模式同步分布函数能量矩。

**这一回调在每个导热子步都执行，不是整次慢更新结束才执行。** 下一子步必须同时看到退缩后的冰壁、剩余质量、新流体格和重建水温，才能在新边界换热。慢更新全部结束后，`thermal.steps += N_diff`、`thermal.time_s += Δt_slow`，外层将 pending 清零，最后 `step()` 才增加 LBM 计数。

## 8 可选分支与输出接口

### 8.1 温度相关黏度

若 `viscosity_water` 给出数值，水使用该常量运动黏度。若为 `None`，水动力黏度按 `viscosity.py` 和 `lattice.py` 中的系数计算。令 $\vartheta=(T_w+273.15)/300$，则代码实现

$$
\mu_w(T_w)=10^{-6}\left(280.68\vartheta^{-1.9}
+511.45\vartheta^{-7.7}+61.131\vartheta^{-19.6}
+0.45903\vartheta^{-40}\right),\qquad \nu_w=\mu_w/\rho_w.
$$

温度以摄氏度输入，$\mu$ 单位为 Pa·s，$\nu$ 单位为 $\mathrm{m^2/s}$。源码将其标注为 IAPWS 常压液水关系，并校验初始水温、熔点和定温墙温在 $[-20,110]\ ^\circ\mathrm C$ 内。运行时有热水体积的格点使用当前水温；没有热水体积的相场尾部使用初始水温的水黏度。

若 `viscosity_air=None`，以固定设定气温 $T_{a,K}$ 计算

$$
\mu_a=1.716\times10^{-5}
\left(\frac{T_{a,K}}{273.15}\right)^{3/2}
\frac{273.15+110.33333}{T_{a,K}+110.33333},
\qquad \nu_a=\mu_a/\rho_a.
$$

气相没有热输运，所以该气黏度不会因水温变化而更新。上述公式是本仓库实际使用的物性闭合，不代表模拟器同时求解了温度相关密度或全套真实流体状态方程。

### 8.2 cm_hrr 重建偏应力并保留压力模式

该选项在 `cm_mrt` 基础上用缓存有效速度的梯度重建偏应力。速度梯度按

$$
A_i=3\sum_{q=1}^{8}w_q\boldsymbol u_{i+q}^{\mathrm{ext}}\boldsymbol c_q^T
$$

计算。固定墙用 $-\boldsymbol u_i$ 反射；冰边界用实际交点比例延拓

$$
\boldsymbol u_g=\boldsymbol u_i+
\frac{\boldsymbol u_B-\boldsymbol u_i}{\max(\lambda,10^{-3})}.
$$

用于二阶偏应力的重建量为

$$
D_{\mathrm{rec}}=-\frac{2\rho\tau}{3}(A_{xx}-A_{yy}),
\qquad H_{\mathrm{rec}}=-\frac{\rho\tau}{3}(A_{xy}+A_{yx}).
$$

设 $b=\texttt{hrr\_kinetic\_blend}$，以 $D=m_{20}-m_{02}$ 为例，碰撞前替换为

$$
D_{\mathrm{blend}}=D^{\mathrm{eq}}+b(D-D^{\mathrm{eq}})
+(1-b)(D_{\mathrm{rec}}-S_D/2).
$$

$m_{11}$ 同理。然后按分子剪切松弛率更新。三阶中心非平衡矩去掉，而四阶矩需独立保持压力组合：

$$
P=m_{22}^{\mathrm{raw}}-(m_{20}^{\mathrm{raw}}+m_{02}^{\mathrm{raw}})
=\sum_q[c_{qx}^2c_{qy}^2-|\boldsymbol c_q|^2]f_q.
$$

$P$ 按声学率更新；在逆平移到原始矩后要求 $m_{22}^{\mathrm{raw},*}=\mathrm{trace}^{\mathrm{raw},*}+P^*$，据此闭合四阶中心矩。这个独立压力模式不能替换成质量分布的 Maxwell 平衡矩。

该选项可能增加有限波长的数值耗散。剪切率仍来自分子黏度，并不能推出所有波长的衰减都和基础碰撞完全一致，也不能仅凭运行不发散判断精度提高。

### 8.3 consistent_flux 让相场质量通量参与动量输运

这是会改变 `step()` 实际顺序的选项。`_collide_momentum` 先准备梯度及界面应力，缓存进入本轮时的有效速度 $\boldsymbol u^{\mathrm{cache}}$，再碰撞一次相场、测量实际碰撞产生的非对流质量通量，最后碰撞动量。外层 `_advance_fluid` 随后跳过第二次相场碰撞。该选项在添加溃坝前已经存在，本次仍复用原通量公式，只让它读取当前选定的梯度/几何。

质量通量由真实的旧分布和碰撞后分布测量：

$$
\boldsymbol j_i=(\rho_w-\rho_a)
\left[\sum_q\boldsymbol c_q\frac{h_q+h_q^*}{2}
-\phi_i\boldsymbol u_i^{\mathrm{cache}}\right].
$$

用于理解的连续近似为 $\boldsymbol j\approx-(\rho_w-\rho_a)M(\nabla\phi-\Lambda\boldsymbol n)$，但生产代码使用上式，没有另行换一套梯度或法向估计通量。

该非对流质量通量携带动量，因而增加力密度

$$
\boldsymbol F_j=-\nabla\cdot(\boldsymbol u^{\mathrm{cache}}\otimes\boldsymbol j)
=-\boldsymbol u^{\mathrm{cache}}\nabla\cdot\boldsymbol j
-(\boldsymbol j\cdot\nabla)\boldsymbol u^{\mathrm{cache}}.
$$

代码在四个轴向共享面上，用邻格通量与速度的算术平均离散完整的散度。轴 $a\in\{x,y\}$，方向 $s\in\{-1,1\}$，若邻格 $k=i+s\boldsymbol e_a$ 是流体，则

$$
\bar j_a=\frac{j_{a,i}+j_{a,k}}2,\qquad
\bar{\boldsymbol u}=\frac{\boldsymbol u_i^{cache}+\boldsymbol u_k^{cache}}2,\qquad
\boldsymbol F_{j,i}\mathrel{-}=s\,\bar j_a\bar{\boldsymbol u}.
$$

冰和墙面不传这份非对流质量通量。将 $\boldsymbol F_j/\rho$ 加到动量加速度，并同时进入源项与半步速度、压力恢复。这里计算的是力密度；直接把它当成加速度会漏掉除以本格混合密度的操作。

这会改变气液界面附近的动量耦合；它不把第 6.5 和 7.3 节的局部回填、热水重映射变成全局质量投影，因此不保证 v3 水量或冰水总质量闭合。

名称 `consistent_flux` 表示这份连续方程层面的完整通量修正，不能据此断言逐链路代数收支已经与相场 LBM 完全一致。相场实际沿八条 D2Q9 链路迁移，当前修正沿四个轴向面计算；生产代码还没有上一轮提出的逐链路共享 $H_{ij}$、限流系数或限流后重新计算动量修正。

### 8.4 non_equilibrium 给新露出格点补应力

默认只补平衡分布；该选项还从持续流体邻域计算非平衡二阶应力

$$
\Pi_k^{\mathrm{neq}}=\sum_q\boldsymbol c_q\boldsymbol c_q^T
\left[f_{q,k}-f_{q,k}^{\mathrm{eq}}(p_{d,k},\rho_k,\boldsymbol v_k)\right],
\qquad
\Pi_{\mathrm{new}}=\rho_{\mathrm{new}}\frac1N\sum_{k\in\mathcal N}
\frac{\Pi_k^{\mathrm{neq}}}{\rho_k}.
$$

先按密度归一化，防止把水侧应力量级直接拷贝到低密度空气。设该应力分量为 $(\pi_{xx},\pi_{xy},\pi_{yy})$，分布增量的原始矩设为

$$
\delta m_{00}=\delta m_{10}=\delta m_{01}=0,
\quad (\delta m_{20},\delta m_{11},\delta m_{02})=(\pi_{xx},\pi_{xy},\pi_{yy}),
$$

$$
\delta m_{21}=v_y\pi_{xx}+2v_x\pi_{xy},\quad
\delta m_{12}=v_x\pi_{yy}+2v_y\pi_{xy},\quad
\delta m_{22}=\pi_{xx}+\pi_{yy}.
$$

逆变换得到 $\delta f_q$ 并加到回填平衡分布。如此保持总和与一阶动量不变，且

$$
\sum_{q\ne0}\delta f_q
=\delta m_{20}+\delta m_{02}-\delta m_{22}=0,
$$

因而不改变压力恢复所用的非静止分布和。该选项不改变回填相场与水面积。

本次 `connected_ls` 分支把供体搜索限制到半径 1、2 且要求 `phase_donor_mask` 允许、前后都是流体；`legacy` 仍使用原来的扩大方形环搜索。应力不会从同批新填格点直接继承。溃坝当前默认 `equilibrium`，因此默认运行没有开启这一应力补项。

### 8.5 enthalpy_d2q5 用五个分布实现体内导热

D2Q5 使用静止、右、上、左、下五个方向。此处的分布记为 $g_q$，不要与动量 $f_q$ 混淆；它们承载单位厚度的显热，满足 $\sum_{q=0}^{4}g_q=E$。

本实现固定热松弛时间为 1，故碰撞直接设为

$$
g^*=(E-4B,\ B,\ B,\ B,\ B),\qquad B=k\delta t\,a(T-T_m).
$$

水取 $a=V/\Delta x^2$，冰取 $a=f^{\mathrm{ice}}$。由于 $E=\rho c\Delta x^2a(T-T_m)$，可写成

$$
Fo=\frac{\alpha\delta t}{\Delta x^2},\quad
w_0^{\mathrm{th}}=1-4Fo,\quad w_{1\ldots4}^{\mathrm{th}}=Fo,
\quad g_q^*=w_q^{\mathrm{th}}E.
$$

两侧开口为 $a_L,a_R$ 时，只在公共部分 $a_s=\min(a_L,a_R)$ 交换。对一侧格点的面能量增量为

$$
\Delta E_f=\frac{a_s}{a_R}g_{\mathrm{in}}^*
-\frac{a_s}{a_L}g_{\mathrm{out}}^*
=k\delta t\,a_s(T_R-T_L).
$$

最后一个等式说明它与有限体积导热面通量等价。无开口部分反弹；定温墙采用半格距离反反弹，得到第 7.6 节的系数 2。这里没有自由可调的热松弛时间，也不在热分布中另行执行水对流。

水量重映射、水对流、界面换热和融化仍修改权威显热状态。`synchronize()` 通过

$$
g_0\leftarrow E-\sum_{q=1}^{4}g_q
$$

同步零阶矩；材料消失时清零其所有热分布。D2Q5 与 FV 在相同配置下应得到接近舍入误差的体内导热结果。默认 FV 模式不分配这些热分布数组。

### 8.6 采样字段不能代替权威材料状态

`sample_thermal_fields()` 调用 `sample_world_fields()`，在 CPU 上临时构造供输出使用的温度、体积焓、液相比例和材料标记。其水侧输出体积焓是

$$
H_{w,\mathrm{out}}=\frac{\rho_wLV+E_w}{\Delta x^2}.
$$

这里加入液水潜热基准仅用于输出焓表示，并不表示融冰曾作为液水源加入。达到冰覆盖阈值的输出格点按对应材料格显示 $E_i/\Delta x^2$，液相比例设为零；墙温设为熔点，其他输出为零。部分亚阈值残冰不一定显示为锐界冰格，因此计算剩余冰量应读 `body_solid_mass` 或 `body_mass_lattice`。

输出接口构造的数组不是推进过程中的另一套独立热状态。内部 `.to_numpy()` 的空间轴为 `(nx, ny)`；示例保存或绘图可能转置为 `(ny, nx)`，核对坐标时应先检查数组形状。

## 9 如何逐项验证代码

### 9.1 先验证局部关系再验证整体行为

以下每项都是可检查的关系，但作用范围不同。

| 检查对象 | 应检查的关系 | 适用范围 |
|---|---|---|
| 相场宏观量 | $\phi\approx\sum_qh_q$ | 当前有效流体格点 |
| 混合密度 | $\rho=\rho_a+(\rho_w-\rho_a)\hat\phi$ | 密度来自相场 |
| 有效速度 | $\boldsymbol u=\sum_q\boldsymbol c_qf_q/\rho+\boldsymbol a/2$ | 当前有效流体格点 |
| 热水面积 | $V=\Delta x^2\hat\phi$ 或零 | 每次热重映射结束后 |
| 冰质量 | $M_L=\sum_rm_r/(\rho_w\Delta x^2)$ | 融化回调归约后 |
| 冰内显热 | $E_i\le0$、$m\ge0$ | 潜热处理后 |
| 冰水换热 | 水的显热损失等于材料热增量 | 单次界面交换，不包括重映射 |
| 子步时间 | $N\delta t=\Delta t_{\mathrm{update}}$ | 对流或导热的一次调用 |
| 慢热调度 | 流动时间减慢热时间等于 pending 对应时间 | 仅经公开 `step` 推进 |
| D2Q5 能量矩 | $\sum_qg_q\approx E$ | 热分布同步后 |

f32 分布的求和不应要求位级完全相等；按量纲和量级设合理的绝对或相对容差。f64 原子归约在 GPU 上仍可能因加法顺序略有不同；此外，部分内核常量先按默认 f32 精度形成，再转换成 f64，因此质量换算也不宜要求 $10^{-10}$ 级的相对误差。本快照的示例质量换算相对差约为 $4.75\times10^{-8}$。配置分批时序相同，不自动等于跨运行、跨硬件的逐位相同结果。

### 9.2 一段可直接运行的状态与时序检查

在仓库根目录把下面代码保存为 Python 文件后运行。需要本项目依赖和 CUDA GPU。该小算例让冰完全浸水，重力和表面张力为零，用来检查存储关系、热更新节奏和分批调用；它不验证冰入水的长时间精度。

```python
import numpy as np
from iceflow2d_v3 import IceBodyConfig, IceFlow2D, IceFlowConfig
from iceflow2d_v3.config import ThermalConfig

cfg = IceFlowConfig(
    resolution=(32, 48), dx=5e-4, reference_velocity=4.0,
    boundary_cells=2,
    water_width_fraction=30 / 32,
    water_height_fraction=24 / 48,
    ice_bodies=(IceBodyConfig(
        size_cells=(8, 8), center_cells=(16, 16),
    ),),
    gravity=(0.0, 0.0), sigma=0.0, phase_warmup_steps=0,
    phase_gradient_scheme="legacy",
    phase_momentum_coupling="legacy",
    thermal=ThermalConfig(
        initial_water_temperature_c=90.0,
        initial_ice_temperature_c=0.0,
        buoyancy_reference_temperature_c=90.0,
        update_interval_lbm_steps=8,
    ),
)
s = IceFlow2D(cfg)
initial_mass = s.thermal.body_solid_mass.to_numpy().sum()
s.step(3)
assert s.steps == 3 and s._thermal_lbm_steps_pending == 3
assert s.thermal.time_s == 0.0
s.step(5)
assert s.steps == 8 and s._thermal_lbm_steps_pending == 0
assert np.isclose(s.physical_time_s, 8 * s.scales.dt_s)
assert np.isclose(s.thermal.time_s, s.physical_time_s)

active = ((s.wall_mask.to_numpy() == 0)
          & (s.solid_mask.to_numpy() == 0))
phi = s.water_phase.to_numpy()
h = s.phase_populations.to_numpy()
f = s.momentum_populations.to_numpy()
assert np.all((phi[active] >= 0) & (phi[active] <= 1))
np.testing.assert_allclose(
    h.astype(np.float64).sum(axis=-1)[active], phi[active],
    atol=2e-6, rtol=2e-6,
)

c = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=np.float64)
rho = cfg.rho_air / cfg.rho_water + (
    1 - cfg.rho_air / cfg.rho_water
) * np.clip(phi, 0, 1)
raw_velocity = np.einsum("ijq,qd->ijd", f, c) / rho[..., None]
np.testing.assert_allclose(
    raw_velocity[active],
    s.momentum_velocity_lattice.to_numpy()[active],
    atol=2e-6, rtol=2e-6,
)
np.testing.assert_allclose(
    s.physical_velocity_lattice.to_numpy()[active],
    (s.momentum_velocity_lattice.to_numpy()
     + 0.5 * s.fluid_acceleration_lattice.to_numpy())[active],
    atol=2e-6, rtol=2e-6,
)

expected_volume = np.where(
    active & (phi > cfg.water_phase_cutoff),
    cfg.dx**2 * np.clip(phi.astype(np.float64), 0, 1), 0.0,
)
np.testing.assert_allclose(
    s.thermal.water_volume_m2.to_numpy(), expected_volume,
    atol=1e-15, rtol=1e-6,
)
material_mass = s.thermal.body_solid_mass.to_numpy().sum()
np.testing.assert_allclose(
    s.body_mass_lattice.array().sum(),
    material_mass / (cfg.rho_water * cfg.dx**2), rtol=1e-7,
)
assert 0 <= material_mass <= initial_mass
assert np.all(s.thermal.body_sensible_energy.to_numpy() <= 0)
for name in (
    "momentum_populations", "phase_populations", "pressure_lattice",
    "physical_velocity_lattice", "temperature",
):
    assert np.isfinite(getattr(s, name).to_numpy()).all(), name

whole = IceFlow2D(cfg)
whole.step(8)
for name in ("water_phase", "body_mass_lattice", "body_center"):
    np.testing.assert_allclose(
        getattr(s, name).to_numpy(), getattr(whole, name).to_numpy(),
        atol=2e-6, rtol=2e-6,
    )
print("状态关系、热更新时序和分批调用检查通过")
```

### 9.3 针对每一类公式运行现有测试

现有测试中有直接调用生产内核的检查，也有完整小算例。按学习顺序使用这些入口：

- `test_momentum_collision.py`：独立矩阵碰撞对照、平衡态、剪切波黏度、压力波；理解第 5.3 节。
- `test_iceflow.py`：单链路反弹、物理质量与惯量、墙约束、短耦合推进；理解第 5.4 与 6.1 节。
- `test_phase_normal.py`、`test_phase_bounds.py`：法向正则化、均匀相场和局部越界修复；理解第 5.5 与 5.8 节。
- `test_capillary_coupling.py`：平界面毛细力、Laplace 压差、冰上的反作用力和力矩；理解第 5.2 与 5.9 节。
- `test_thermal_coupling.py`：水下和空气中消失、材料质量、热重映射、逐子步刷新、累计时序；理解第 6.5 与 7 节。
- `test_thermal_d2q5.py`：余弦温度衰减、FV 等价性、定温墙、热分布能量矩；理解第 8.5 节。
- `test_multibody.py`：各块负载独立、共享水热量限幅、接触冲量、逐块消失；理解第 6.3 与 7.7 节。
- `test_physical_viscosity.py`：从实际碰撞反算冷热水、空气、界面和冰邻格的黏度；理解第 8.1 节。
- `test_stability_options.py`：HRR 压力模式、质量通量力、应力回填；理解第 8.2 至 8.4 节。
- `test_dambreak.py`：局部水柱的初始压力与自由运动回归；理解第 3.1 节，长回归当前仍有失败。
- `test_phase_reconstruction.py`：本次增加的供体连通性、受约束梯度、连续多项式有界限制和分轮补点；理解第 5.10 与 6.5 节。
- `tests/test_dambreak_four_ice_melting_example.py`、`tests/test_velocity_air_visualization.py`：几何/物性默认值与输出口径；输出检查不会验证求解器稳定性。

可先运行短的核心检查，再按需要运行完整文件。例如：

```bash
python -m pytest -q \
  iceflow2d_v3/test_iceflow.py \
  iceflow2d_v3/test_phase_bounds.py \
  iceflow2d_v3/test_thermal_coupling.py \
  iceflow2d_v3/test_thermal_d2q5.py \
  iceflow2d_v3/test_stability_options.py \
  iceflow2d_v3/test_physical_viscosity.py
```

### 9.4 哪些守恒关系不能强制要求

**局部碰撞和共享面通量的守恒，不等于整个带移动边界的 v3 模型守恒。** 水量漂移可以按操作逐段记录：流送前后、相场限幅前后、刚体掩码切换与回填前后、融化回填前后，分别求有效流体域的 $\Delta x^2\sum\phi$。热水面积还排除低于湿阈值的尾部，统计口径应固定。

全局变化有明确来源：

1. 冰格点变为流体时按环境外推生成新状态，流体被冰覆盖时从当前有效域移除；没有全局水量投影。
2. 相场限幅直接改变局部 $\phi$；移动壁面的相场反弹还带速度修正。
3. 热重映射按局部温度和目标面积重建显热；没有全局能量补偿。
4. 融化删除材料质量，完全融尽后的多余热量丢弃；不把材料作为水、显热或动量源加入。
5. 显式启用的线性和角阻尼（默认均为 1）、无回弹接触、墙速度截断会耗散机械能；刚体材料退缩也不强制保持原冰总动量。

因此不能要求“水质量加冰质量不变”或“冰水总能量不变”作为当前代码的必过断言。若研究目标需要真实冰水转化的质量、能量和动量闭合，需要更改模型及移动边界重建，而不是只调整检查容差。

### 9.5 当前数据优先重构及历史版本的核对记录

验证必须区分重构正确性、短时推进、全程稳定性和模型守恒。下表中的长算例及对照已经在此前任务中实际运行；本轮文档同步不重复执行 5 s / 8 s 仿真。

| 数据优先版本的验证 | 已记录结果 | 范围与证据 |
|---|---|---|
| 梯度、耦合和配置回归 | 31 项通过 | 单冰块结果目录中的 `regression.log` 和源码指纹 |
| 当前梯度测试 | 12 项通过 | 本轮文档同步重新执行，覆盖第 5.10 节的主要关系；结果见 manifest |
| 单冰块溃坝 | 完成 5 s / 600000 步后按用户要求停止 | 未触发严重失稳检查；每类 PNG/GIF 501 帧，未继续验证 8 s |
| 四冰块 8 s 目标运行 | 第 8532 步、0.0711 s 失败 | `(152,3)` 的 L1 速度约 1009.54 LU；最后完整输出 0.07 s，每类 8 帧 |
| 四冰块开缝后的局部干预 | 保持局部相量、按 W=5 平滑后继续 3249 步 | 随后第 11770 步 `(236,6)` 新流体无可达供体；不代表全程修复 |

单冰块结果为 `outputs/dambreak_single_ice_data_priority_5s/`，主动停止及 GIF 完整性记录在 `run_stop.json`。四冰块失败结果为 `outputs/dambreak_four_ice_data_priority_8s/`，目标和配置记录在 `run_manifest.json`，失败原因在 `run_failure.json`。这些相对路径均以 `iceflow2d_v3/` 为基准。四冰块逐阶段记录、独立 SVD 对照、残差与干预汇总见 `outputs/four_ice_data_priority_diagnosis/evidence_summary.json`。

#### 四冰块为何仍会失稳

首次触发早于最终报错：第 8520 步侵蚀打开第 6.5 节的半格窄缝后，过渡变为 1 → 0.50005 → 0.000107。第 8521 步气侧 `(156,3)` 的密度约 0.001355，净 x 动量却达到约 0.0004205，恢复速度约 0.31035；压力恢复中的密度梯度项贡献约 −0.002332，主导了压力由约 −0.0000120 到 −0.002151 的变化。第 8522 步相场总和约 −0.002026、L1 速度约 10.35 LU；第 8523 步几何更新前速度约 91572.6 LU，更新后异常点被冰体掩码覆盖，直到第 8532 步再次触发已有检查。

把局部 φ 总量保持不变并按 W=5 展宽剖面，可避开这次立即爆发；单独重置压力、重置分布、固定开缝后的冰体，或将局部法向梯度置零，都仍在两步后出现大速度。这些对照支持“未解析补点过渡触发当前相场/密度源项/压力/边界离散的不稳定”，没有证明某一条源项是唯一原因，也没有给出通用梯度阈值。单行法向仍是模型闭合，SVD 一致不等于物理法向信息已被解析。

复现和读取时注意：局部相场限幅只修复 φ 与相场分布，不同步消除动量分布中的错误；已有 `check_stability` 在热对流前检查，位于刚体几何更新之后。计划中的“几何更新前检查”尚未接入生产调度。

#### 历史记录的适用范围

2026 年 10 月 8 日快照的 50 项通过/1 项配置预期失败，以及 10 月 13 日硬约束 `connected_ls` 的 10 项边界测试、17 项边界/毛细回归、73 项通过/1 项宽水柱回归失败，都属于各自当时的源码版本。旧四冰块在第 8287 步失稳；旧单冰块在第 22642 步失稳。不要把这些数字或旧测试中“整体多项式限幅”的断言当成当前版本的验收条件。历史日志及本轮文档检查分别保存在 manifest 的历史验证字段与 `documentation_validation` 中。

### 9.6 当前已实施的修正与后续方案

已进入生产代码的是第 5.10 节的数据优先重构、单行显式曲率闭合、数据零空间内的壁面残差最小化、取消固体侧多项式极值对梯度的整体限幅，以及每次几何/补点提交后的缓存刷新。原先讨论的“壁面不能覆盖已观测切向变化”已通过零空间顺序实现；它不是按连续表面分组的软约束方案。

以下仍属于后续设计：

1. 对新开、关闭和相邻持续流体组成的区域做有界、守恒重映射，结合移动边界的流体体积预算保持配置界面宽度，并兼容相场、动量和压力状态。
2. 提取实际 D2Q9 共享链路相场/密度通量，在同一限流结果上闭合动量输运和压力恢复；当前四轴面 `consistent_flux` 不等于已完成该代数闭合。
3. 对亚网格窄缝和孤立气穴保存守恒状态，或采用体积分数/开口面积及局部加密；不能跨冰借值或无条件置空气。v3 当前融化产物消失的模型假设仍需保留，不能凭空增加融水源。
4. 在刚体积分和掩码更新前检查有限值、速度与分布矩，避免已异常流体被覆盖后继续影响刚体；这是故障检测，不能替代数值修复。

一次局部平滑是区分触发因素的诊断干预，其局部 φ 总量不变，不代表已经完成移动边界的全局质量、动量或热量守恒重映射。当前生产代码没有这些预算与逐链路限流。

### 9.7 用小算例验证本次增加的调用分支

先运行不需要推进长仿真的配置检查，再运行已有局部 CUDA 测试：

```bash
python -m pytest -q \
  iceflow2d_v3/tests/test_config_validation.py \
  iceflow2d_v3/tests/test_dambreak_four_ice_melting_example.py \
  iceflow2d_v3/tests/test_velocity_air_visualization.py

python -m pytest -q iceflow2d_v3/test_phase_reconstruction.py
python -m pytest -q iceflow2d_v3/test_dambreak.py -k initial_column_hydrostatic_pressure
```

其中新几何/可视化配置检查不证明流动稳定。后两条需要 CUDA，分别验证边界公式和初始水柱压力。宽水柱历史长回归曾失败；本轮文档没有重跑该长回归，不应把历史结果当成本轮通过证明。

第 9.2 节代码明确选择两个 `legacy` 默认分支。要同时覆盖新旧调度，可把其 `cfg` 对应的短状态关系检查扩展到下面四组；每组新建模拟器，避免修改已经编译的 Taichi 配置：

```python
from dataclasses import replace

for gradient_scheme in ("legacy", "connected_ls"):
    for coupling in ("legacy", "consistent_flux"):
        case = replace(cfg, phase_gradient_scheme=gradient_scheme,
                       phase_momentum_coupling=coupling)
        probe = IceFlow2D(case)
        probe.step(3)
        assert probe._thermal_lbm_steps_pending == 3
        probe.step(5)
        assert probe.steps == 8
        assert probe._thermal_lbm_steps_pending == 0
        assert np.isclose(probe.thermal.time_s, probe.physical_time_s)

        fluid = ((probe.wall_mask.to_numpy() == 0)
                 & (probe.solid_mask.to_numpy() == 0))
        phi = probe.water_phase.to_numpy()
        np.testing.assert_allclose(
            probe.phase_populations.to_numpy().astype(np.float64)
                 .sum(axis=-1)[fluid], phi[fluid],
            atol=2e-6, rtol=2e-6)
        assert np.all((phi[fluid] >= 0) & (phi[fluid] <= 1))

        if gradient_scheme == "connected_ls":
            # 正常 step 已在最终几何/补点提交后刷新梯度；直接检查缓存。
            cached = {}
            for name in ("phase_gradient_lattice", "source_gradient_lattice",
                         "phase_reconstruction_coefficients",
                         "phase_gradient_boundary_residual"):
                cached[name] = getattr(probe, name).to_numpy().copy()
                assert np.isfinite(cached[name]).all()
            # 仅为验证缓存时层，再重算一次，正常推进无需手动调用。
            probe._prepare_phase_gradients()
            for name, values in cached.items():
                np.testing.assert_allclose(
                    getattr(probe, name).to_numpy(), values,
                    atol=2e-7, rtol=2e-6)
print("四组短状态关系与热调度检查通过")
```

这些检查只读内存中的数组，不写 NPZ 或场文件。验证更长的自由运动时，应逐阶段记录标量/少量格点，区分碰撞、流送、相场修复、压力、刚体补点、侵蚀补点首次异常的位置。

## 10 快速定位一个异常

观察某个错误时，先定位它第一次出现的阶段，能比直接改参数更快找到原因。

| 现象 | 优先检查 |
|---|---|
| 密度由分布和计算后异常 | 是否错误地把 $\sum f_q$ 当成密度 |
| 水静止时冰不受浮力 | cut-link 是否补入静水参考压力牵引 |
| 气液界面附近加速度很大 | 当前梯度分支、供体连通性、混合密度、质量通量力以及力密度除以密度 |
| 新露出的水下格点变空气 | 连通窗口、分轮传播、是否存在持续湿供体 |
| `connected_ls` 梯度或壁面残差可疑 | 供体空间秩、加入闭合后的数据秩、相场分辨率、单行法向闭合与缓存时层；不要按旧硬约束解释状态 5 |
| 新气隙两端状态形成台阶 | 是否一旦标为有效就停止更新，是否有整块重建 |
| 补点报告 isolated 节点 | 实际对角通路是否可见；不要跨冰借供体或无条件填空气 |
| 冰质心不越墙，冰角仍穿墙 | 接触点速度 $\boldsymbol V\cdot\boldsymbol n+\omega\ell$ 及转动冲量 |
| 冰融化后仍有矩形障碍 | 掩码、接触支撑与 cut-link 是否来自当前覆盖率 |
| 热更新分批运行后不一致 | pending 是否跨 `step()` 调用保存 |
| 同一水格给多块冰供热过多 | 全部请求是否在同一限幅中处理 |
| 剩余冰看不见却持续下落 | 是否检查存在可解析固体边界 |
| 质心随融化变化后速度跳变 | 是否按 $\omega\mathcal P\Delta\boldsymbol C$ 同步质心速度 |
| 热子步数突然很大 | 先检查面出流率、有效速度、有限值与格子单位 |
| D2Q5 温度与显热不一致 | 是否同步 $\sum g_q=E$，以及清除消失格点的分布 |

初学者最有效的阅读路线是：从 `step` 展开三个阶段，在每个阶段开始和结束取一次快照，按第 9.1 节的关系核对，再选择相应的单内核测试。不要一次打印整个域的全部分布；先选一个格点、一条链路或一块材料格，手算该处的更新。

## 附录 A 当前源码的方法索引

以下行号由 Python AST 从当前源码生成。`kernel` 在 GPU 批量更新字段，`func` 在内核内展开；Python 方法负责调度、包装或主机读取。覆盖 `simulator.py` 的全部顶层函数/类方法，以及继承进 `IceFlow2D` 的 `PhaseReconstruction` 方法。

- [ensure\_taichi\_cuda](../simulator.py#L52) [Python]：建立或校验 CUDA 运行时，拒绝 CPU 回退。
### _DerivedField 方法

- [\_\_init\_\_](../simulator.py#L79) [Python]：校验输入、分配状态并建立初始条件。
- [\_\_getitem\_\_](../simulator.py#L87) [Python]：在设备或主机端读取字段，并处理单冰块索引。
- [\_\_setitem\_\_](../simulator.py#L100) [Python]：向可写底层状态提交主机修改。
- [to\_numpy](../simulator.py#L105) [Python]：在主机端构造当前派生值的 NumPy 副本。
### MovingBodyThermal2D 方法

- [\_\_init\_\_](../simulator.py#L129) [Python]：校验输入、分配状态并建立初始条件。
- [maximum\_advection\_time\_step\_s](../simulator.py#L281) [Python]：由面出流率和 Courant 上限给出最大对流步长。
- [\_validate\_substep\_count](../simulator.py#L291) [Python]：检查热子步数是否超过配置上限。
- [required\_advection\_substeps](../simulator.py#L316) [Python]：计算快对流过程所需的子步数。
- [required\_diffusion\_substeps](../simulator.py#L340) [Python]：计算慢导热过程所需的子步数。
- [initialize](../simulator.py#L352) [Python]：初始化材料质量、显热和水热状态。
- [advance\_fast](../simulator.py#L368) [Python]：每 LBM 步重映射热水区域并执行迎风对流。
- [advance\_slow](../simulator.py#L397) [Python]：执行导热、界面换热、潜热融化和逐子步回调。
- [synchronize\_water\_aperture](../simulator.py#L441) [Python]：使热水面积匹配相场和掩码，局部重建显热。
- [\_initialize\_body\_state](../simulator.py#L459) [kernel]：为有效材料格设置初始冰质量和冰显热。
- [\_initialize\_water\_state](../simulator.py#L473) [kernel]：为当前湿流体格设置水面积和水显热。
- [\_build\_phase\_aperture\_targets](../simulator.py#L502) [kernel]：冻结旧热状态并计算重映射后的目标显热。
- [\_apply\_phase\_aperture\_transfer](../simulator.py#L548) [kernel]：统一提交重映射的水面积和显热。
- [\_accumulate\_body\_conduction](../simulator.py#L561) [kernel]：按材料轴向邻接累加冰内导热增量。
- [\_compute\_water\_conduction\_flux\_x](../simulator.py#L588) [kernel]：计算水 x 面导热及左右墙热通量。
- [\_compute\_water\_advection\_flux\_x](../simulator.py#L686) [kernel]：计算水 x 面的上游面积与显热通量。
- [\_compute\_water\_conduction\_flux\_y](../simulator.py#L737) [kernel]：计算水 y 面导热及上下墙热通量。
- [\_compute\_water\_advection\_flux\_y](../simulator.py#L835) [kernel]：计算水 y 面的上游面积与显热通量。
- [\_update\_water\_advection](../simulator.py#L886) [kernel]：按面通量差更新水面积和显热。
- [\_update\_water\_conduction](../simulator.py#L905) [kernel]：按导热通量差只更新水显热。
- [\_exchange\_interface\_heat](../simulator.py#L917) [kernel]：将一个水格的全部冰水供热请求统一限幅并分发。
- [\_apply\_body\_heat\_and\_phase\_change](../simulator.py#L1058) [kernel]：先处理冰显热，再按潜热删除融化材料。
- [\_body\_fraction\_at](../simulator.py#L1075) [func]：从材料质量推导剩余冰比例。
- [\_body\_temperature\_at](../simulator.py#L1085) [func]：从材料质量和冰显热推导冰温。
- [\_water\_temperature\_at](../simulator.py#L1095) [func]：从水面积和显热推导水温，干格显示设定气温。
- [\_body\_grid\_coordinates](../simulator.py#L1105) [func]：把世界格心映射到材料数组的插值坐标。
- [\_water\_aperture\_target](../simulator.py#L1134) [func]：从湿阈值、相场和掩码给出目标热水面积。
- [\_body\_fraction\_numpy](../simulator.py#L1152) [Python]：构造材料冰比例的主机数组。
- [\_body\_temperature\_numpy](../simulator.py#L1161) [Python]：构造材料冰温的主机数组。
- [\_water\_temperature\_numpy](../simulator.py#L1174) [Python]：构造水温的主机数组。
- [sample\_world\_fields](../simulator.py#L1186) [Python]：将权威热状态映射为世界网格输出数组。
### IceFlow2D 方法

- [\_\_init\_\_](../simulator.py#L1270) [Python]：校验输入、分配状态并建立初始条件。
- [step](../simulator.py#L1539) [Python]：校验步数，循环推进流体、刚体、热过程和 LBM 计数。
- [\_advance\_fluid](../simulator.py#L1553) [Python]：按配置准备梯度、碰撞、流送、恢复宏观量，并按分支顺序限幅与恢复压力。
- [\_advance\_body](../simulator.py#L1567) [Python]：积分刚体并按新姿态刷新流体域。
- [\_refresh\_fluid\_domain](../simulator.py#L1571) [Python]：统一提交运动/侵蚀几何，重建新流体；connected_ls 在补点后刷新梯度缓存。
- [\_advance\_thermal](../simulator.py#L1582) [Python]：推进快热过程，累计到间隔后触发慢过程。
- [physical\_time\_s](../simulator.py#L1593) [Python]：由 LBM 总步数换算物理流动时间。
- [\_advance\_moving\_thermal\_fast](../simulator.py#L1596) [Python]：收集出流率，调用热对流与水域重映射。
- [\_advance\_moving\_thermal\_slow](../simulator.py#L1612) [Python]：把累计 LBM 时段交给慢导热与融化。
- [\_refresh\_eroded\_body](../simulator.py#L1624) [Python]：每次材料侵蚀后同步质量属性、几何、流体和热水。
- [\_is\_fluid\_cell](../simulator.py#L1635) [func]：判断格点既不是墙也不是冰。
- [\_measure\_water\_outflow\_rate](../simulator.py#L1639) [kernel]：归约湿流体格的最大面出流速度之和。
- [\_collect\_thermal\_stability\_metrics](../simulator.py#L1675) [kernel]：收集完整流体速度和非有限状态指标。
- [\_check\_thermal\_advection\_stability](../simulator.py#L1704) [Python]：执行可选的热对流前完整状态检查。
- [\_validate\_thermal\_advection\_velocity](../simulator.py#L1716) [Python]：对非有限状态或极端格子速度给出定位异常。
- [\_body\_velocity\_at](../simulator.py#L1784) [func]：从质心速度和角速度求冰上任意点的速度。
- [\_cell\_point](../simulator.py#L1790) [func]：返回世界格心的格子坐标。
- [\_phase\_sample\_inside\_walls](../simulator.py#L1794) [func]：将跨容器墙的模板索引作半格镜像。
- [\_ice\_mirror\_sample](../simulator.py#L1815) [func]：按冰表面法向和 cut-link 交点反射采样位置。
- [\_ice\_phase\_ghost\_legacy](../simulator.py#L1853) [func]：保留的标量冰壁镜像插值，未检查供体与中心之间的连通性。
- [\_ice\_phase\_ghost](../simulator.py#L1874) [func]：按梯度选项分派标量镜像或 connected_ls 多项式 ghost。
- [\_phase\_neighbor](../simulator.py#L1883) [func]：按距离读取流体、墙镜像或冰镜像邻相场。
- [\_body\_local\_coordinates](../simulator.py#L1911) [func]：将世界坐标逆旋转到材料参考坐标。
- [\_nearest\_body\_index](../simulator.py#L1927) [func]：从材料坐标计算最近所属材料格索引。
- [\_sample\_body\_fraction](../simulator.py#L1944) [func]：双线性插值当前剩余材料冰比例。
- [\_calculate\_moving\_body\_fraction](../simulator.py#L1976) [kernel]：计算各块世界覆盖率及其最大值合并场。
- [\_commit\_moving\_body\_raster](../simulator.py#L1988) [kernel]：接触约束完成后裁掉墙内覆盖率。
- [\_initialize\_body](../simulator.py#L1996) [Python]：初始化各块冰的刚体状态和负载累加器。
- [\_initialize\_geometry](../simulator.py#L2018) [kernel]：建立空的初始固体及旧固体掩码。
- [\_initialize\_fluid](../simulator.py#L2025) [kernel]：设置初始水空气指示场、环境储备和分布。
- [\_build\_hydrostatic\_reference](../simulator.py#L2057) [Python]：冻结满宽池/空气参考，并为局部水柱逐列积分初始超额压力。
- [\_initialize\_hydrostatic\_equilibrium](../simulator.py#L2116) [kernel]：以静止速度初始化分布，局部水柱保留非零初始动态压力。
- [\_initialize\_moving\_body\_geometry](../simulator.py#L2155) [Python]：提交初始固体所有者并同步旧掩码。
- [\_update\_solid\_mask](../simulator.py#L2163) [kernel]：按当前阈值覆盖率提交所有冰块的锐界掩码。
- [\_update\_moving\_body\_mass\_properties](../simulator.py#L2180) [Python]：调度材料质量属性归约和质心速度同步。
- [\_reduce\_moving\_body\_mass\_properties](../simulator.py#L2187) [kernel]：逐材料格归约质量、一阶质量矩和原点惯量。
- [\_apply\_moving\_body\_mass\_properties](../simulator.py#L2217) [kernel]：更新质心、惯量、活跃状态和剩余材料速度。
- [\_extrapolate\_exposed\_fluid](../simulator.py#L2255) [func]：在最近持续流体邻域外推相场、动态压力和速度。
- [\_refill\_changed\_nodes](../simulator.py#L2301) [kernel]：处理所有者改变的格点并恢复新暴露流体分布。
- [\_prepare\_capillary\_stress](../simulator.py#L2353) [kernel]：按相场梯度建立切向毛细应力张量。
- [\_phase\_central\_gradient](../simulator.py#L2399) [func]：读取新分支的源项梯度缓存，或计算 legacy 中心模板。
- [\_capillary\_traction](../simulator.py#L2417) [func]：计算对称应力张量乘链路方向。
- [\_ice\_capillary\_stress](../simulator.py#L2426) [func]：镜像插值并反射冰邻点的应力张量。
- [\_capillary\_force\_density](../simulator.py#L2464) [func]：按共享链路牵引求毛细力密度。
- [\_fluid\_kinematic\_viscosity\_lattice](../simulator.py#L2518) [func]：计算相场和水温决定的分子格子黏度。
- [\_fluid\_force\_and\_gradient](../simulator.py#L2534) [func]：组装重力残量、热浮力、毛细加速度和中心梯度。
- [\_collide\_momentum](../simulator.py#L2577) [Python]：准备梯度和应力，按选项先碰撞相场并测通量，再碰撞动量。
- [\_collide\_momentum\_kernel](../simulator.py#L2594) [kernel]：投影矩、处理源项与松弛、逆变换并准备冰链路。
- [\_collide\_phase](../simulator.py#L2857) [Python]：Python 包装：必要时准备梯度，再调用相场碰撞内核。
- [\_collide\_phase\_kernel](../simulator.py#L2863) [kernel]：使用模板或缓存梯度碰撞相场中心矩，并闭合零阶矩。
- [\_stream](../simulator.py#L3027) [kernel]：同时 push 流送动量和相场，并处理阻塞链路。
- [\_stream\_phase\_only](../simulator.py#L3069) [kernel]：在初始化相场预热时只流送相场分布。
- [\_update\_streamed\_macroscopic\_fields](../simulator.py#L3105) [kernel]：恢复相场与未修正速度，并处理墙和冰速度。
- [\_update\_streamed\_phase](../simulator.py#L3137) [kernel]：在预热中仅恢复流送后的相场。
- [\_warm\_start\_phase](../simulator.py#L3148) [Python]：初始化时重复相场碰撞、流送和恢复。
- [\_update\_pressure](../simulator.py#L3157) [kernel]：从非静止动量分布恢复动态压力，再加静水压力。
- [\_prepare\_moving\_body\_contact\_support](../simulator.py#L3188) [Python]：供外部调用的覆盖率与墙支撑重建包装。
- [\_prepare\_moving\_body\_contact\_support\_kernel](../simulator.py#L3202) [kernel]：重置或归约各块当前墙接触支撑。
- [\_reset\_moving\_body\_contact\_support](../simulator.py#L3219) [func]：初始化支撑极值归约缓存。
- [\_reduce\_moving\_body\_contact\_extrema](../simulator.py#L3226) [func]：归约当前锐界单元方格的世界支撑极值。
- [\_finalize\_moving\_body\_contact\_support](../simulator.py#L3253) [func]：将无锐界接触几何的支撑清零。
- [\_reduce\_wall\_contact\_ranges](../simulator.py#L3260) [func]：从剩余锐界支撑单元归约四面墙的切向接触端点。
- [\_wall\_contact\_impulse](../simulator.py#L3295) [func]：按接触点法向速度和转动惯量修正刚体平移与角速度。
- [\_integrate\_rigid\_ice](../simulator.py#L3326) [kernel]：消费负载推进刚体；无解析边界的残冰固定姿态。
- [\_project\_moving\_body\_inside\_container](../simulator.py#L3377) [kernel]：投影刚体位置并迭代施加含转动的墙接触冲量。
- [\_solve\_contact\_and\_rasterize](../simulator.py#L3465) [Python]：迭代墙与冰块对接触，提交共同接受的几何。
- [\_reduce\_phase\_change\_geometry\_volume](../simulator.py#L3509) [kernel]：统计锐界固体格点数量。
- [phase\_change\_solid\_volume\_cells](../simulator.py#L3516) [Python]：返回由剩余材料质量换算的冰体积格数。
- [phase\_change\_geometry\_volume\_cells](../simulator.py#L3519) [Python]：返回阈值几何占据的固体格数。
- [\_bound\_water\_phase](../simulator.py#L3523) [kernel]：只修复越界流体相场，并同步其相场分布。
- [sample\_thermal\_fields](../simulator.py#L3548) [Python]：调度供输出用的世界热字段采样。
- [\_wall\_at](../simulator.py#L3552) [func]：在设备端按域边界计算静态墙标记。
- [\_wall\_numpy](../simulator.py#L3562) [Python]：在主机端构造静态墙标记数组。
- [\_signed\_distance\_at](../simulator.py#L3573) [func]：返回覆盖率阈值水平集，非真实欧氏距离。
- [\_signed\_distance\_numpy](../simulator.py#L3581) [Python]：在主机端构造相同的水平集数组。
- [\_reference\_pressure\_at](../simulator.py#L3591) [func]：读取对应行的冻结静水参考压力。
- [\_reference\_density\_at](../simulator.py#L3595) [func]：读取对应行的冻结静水参考密度。
- [\_physical\_velocity\_at](../simulator.py#L3599) [func]：在有效流体格添加半步力修正，其他格返回零。
- [\_physical\_velocity\_numpy](../simulator.py#L3608) [Python]：构造相同的有效流体速度主机数组。
- [\_read\_body\_initial\_mass\_lattice](../simulator.py#L3619) [func]：读取指定冰块的初始格子质量。
- [\_read\_body\_active](../simulator.py#L3624) [func]：由正的剩余惯量给出冰块活跃标记。
- [\_read\_cumulative\_melted\_mass\_lattice](../simulator.py#L3629) [func]：返回初始质量减剩余质量的融化量。
- [\_read\_phase\_change\_initial\_solid\_volume](../simulator.py#L3634) [func]：将全部初始冰质量换算为体积格数。
- [\_read\_phase\_change\_current\_solid\_volume](../simulator.py#L3643) [func]：将全部剩余冰质量换算为体积格数。
- [\_read\_\_body\_contact\_geometry\_active](../simulator.py#L3652) [func]：从支撑极值判断是否有接触几何。
- [\_body\_center\_at](../simulator.py#L3661) [func]：由参考原点、转角和材料质心求世界质心。
- [\_body\_center\_numpy](../simulator.py#L3671) [Python]：构造各块世界质心的主机数组。
- [\_set\_body\_center\_host](../simulator.py#L3684) [Python]：将主机设置的世界质心转换为材料参考原点。
- [\_body\_level\_at](../simulator.py#L3700) [func]：返回指定冰块的覆盖率阈值水平集。
- [\_cut\_link\_fraction](../simulator.py#L3707) [func]：求流体到冰链路的阈值交点比例并限幅。
- [\_prepare\_momentum\_cut\_links](../simulator.py#L3724) [func]：准备冰壁反弹并按所属冰块累加全部边界负载。
- [\_allocate\_populations](../simulator.py#L3840) [Python]：分配按 q 主序存储但以 i、j、q 访问的 GPU 分布。
### PhaseReconstruction 方法

- [\_initialize\_phase\_reconstruction](../phase_reconstruction.py#L23) [Python]：仅在 connected_ls 下分配梯度、秩/壁面残差、几何、多项式和双缓冲补点字段。
- [\_phase\_donor\_allowed](../phase_reconstruction.py#L47) [func]：解码 5×5 窗口的供体允许位。
- [\_unclipped\_ice\_fraction](../phase_reconstruction.py#L55) [func]：在材料坐标取未经过容器墙裁剪的冰覆盖率。
- [\_phase\_visible\_diagonal](../phase_reconstruction.py#L61) [func]：分段检查对角线上的双线性覆盖率内部最大值，拒绝穿冰通路。
- [\_phase\_ice\_geometry](../phase_reconstruction.py#L111) [func]：估计材料法向和实际阈值交点，不使用动量交点限幅。
- [\_prepare\_phase\_geometry](../phase_reconstruction.py#L144) [kernel]：准备局部单调连通供体图及八条邻接链的壁面几何。
- [\_phase\_fit\_basis](../phase_reconstruction.py#L225) [func]：返回局部二次多项式的五列基函数。
- [\_phase\_minimum\_norm\_fit](../phase_reconstruction.py#L229) [func]：双精度列主元 QR、两次正交化和行空间最小范数求解，可返回数据零空间。
- [\_fit\_boundary\_phase](../phase_reconstruction.py#L316) [func]：先拟合连通流体数据及显式曲率闭合，再仅在其零空间内最小化壁面残差。
- [\_prepare\_phase\_gradients\_kernel](../phase_reconstruction.py#L410) [kernel]：内部用原模板，边界用数据优先重构，写入未整体限幅的梯度、秩和壁面残差。
- [\_prepare\_phase\_gradients](../phase_reconstruction.py#L460) [Python]：Python 包装：仅 connected_ls 调用梯度准备内核。
- [\_connected\_phase\_ghost](../phase_reconstruction.py#L465) [func]：在指定采样偏移计算并限幅局部多项式值。
- [\_seed\_connected\_refill](../phase_reconstruction.py#L475) [kernel]：快照持续流体状态并标记待填新节点。
- [\_propagate\_connected\_refill](../phase_reconstruction.py#L501) [kernel]：只读上轮有效供体，先半径 1 后半径 2 地传播状态。
- [\_commit\_connected\_refill](../phase_reconstruction.py#L529) [kernel]：统一提交新节点的宏观状态和平衡/可选非平衡分布。
- [\_refill\_connected\_nodes](../phase_reconstruction.py#L369) [Python]：调度双缓冲传播与提交，无可达新状态时暂缓激活。

## 附录 B 关联模块与源码快照

- [config.py](../config.py)：输入配置、物性、单位换算和合法性校验。
- [lattice.py](../lattice.py)：D2Q9 方向、平衡分布、密度混合与矩逆变换。
- [body_state.py](../body_state.py)：统一的多冰块字段与单冰块兼容索引。
- [contact.py](../contact.py)：冰块间检测、位置修正和法向冲量。
- [stability_options.py](../stability_options.py)：HRR、质量通量动量耦合及应力回填。
- [phase_reconstruction.py](../phase_reconstruction.py)：连通供体、几何法向、数据优先多项式梯度、质量诊断和分轮补点。
- [thermal_d2q5.py](../thermal_d2q5.py)：固定松弛时间的五分布导热和能量矩同步。
- [viscosity.py](../viscosity.py)：分子黏度系数和主机物性计算。
- [溃坝入口](../examples/dambreak_four_ice_melting_2d.py)：场景初始条件、默认算法选项及目标时长。
- [reporting.py](../reporting.py)：物理单位输出、两相不同色标和流式图像；显示限幅不参与求解。
- [source manifest](simulator_method_source_manifest.json)：当前文件 SHA256、方法行号和验证范围。

本指南有同目录 PDF 版本。Markdown 是维护源，修改公式后可使用 Pandoc 和 XeLaTeX 重新生成；中文字体与版式参数保存在 `simulator_method_pdf_header.tex`，生成命令见 `build_simulator_method_pdf.sh`。

在仓库根目录可独立验证文档是否仍对应当前源码，无需 CUDA：

```bash
python iceflow2d_v3/docs/update_simulator_method_manifest.py --check
```

更改源码、核对并更新正文公式后，再更新指纹/方法行号和 PDF：

```bash
python iceflow2d_v3/docs/update_simulator_method_manifest.py --snapshot-date 2026-10-14
bash iceflow2d_v3/docs/build_simulator_method_pdf.sh
```

日期应改为实际核对日期。脚本从 AST 生成方法索引并比较源码 SHA256，不会自动证明正文公式正确；公式与算法选项仍需人工核对以及第 9 节的代码验证。
