# IceFlow2D v3 模拟器原理与逐步验证指南

`iceflow2d_v3/simulator.py` 模拟二维容器中的空气、水和自由运动的冰块。水与空气用格子玻尔兹曼方法推进，气液界面用相场表示，冰块按刚体运动，热量先使冰升温，再使冰质量减少。**v3 的融化产物直接消失**：它不生成融水，也不把消失材料的动量和显热补入流体。这一假设决定了哪些检查可以要求守恒，哪些检查只能要求局部离散关系正确。

本指南面向第一次接触这类程序的读者。公式以当前工作区的实现为准；带“连续形式”说明的方程用于理解离散算法的目标，具体数值更新仍以随后给出的离散公式为准。阅读顺序建议：先读第 1 至 4 节，再沿第 5 至 7 节对照 `step()`，最后使用第 9 节的检查方法。末尾附录提供完整的方法定位索引。

代码快照日期：2026 年 10 月 8 日。具体源码指纹见同目录 `simulator_method_source_manifest.json`；工作区文件变动后，应重新核对方法位置与公式。

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
| `body_reference_origin` | 材料参考原点 $\boldsymbol O$ |
| `body_local_center_of_mass` | 材料坐标中的剩余冰质心 $\boldsymbol\xi_c$ |
| `body_center` | 世界质心 $\boldsymbol C=\boldsymbol O+R\boldsymbol\xi_c$ |
| `body_velocity` | 世界质心速度 $\boldsymbol V$ |
| `hydrodynamic_impulse` | 一个格子时间步的边界冲量累加 $\boldsymbol J$ |
| `hydrodynamic_torque` | 一个格子时间步的角冲量累加 $J_\theta$ |

`physical_velocity_lattice` 中的 “physical” 表示已经作力修正的有效速度；字段本身仍是格子单位。转成 m/s 仍需乘 $\Delta x/\Delta t$。固体和墙处，这个视图返回零；查询冰壁速度应使用 `_body_velocity_at`。

字段 `body_center`、温度、壁面和静水参考场等由 `_DerivedField` 即时计算。它们通常不是另一份需要同步的 GPU 数组。单冰块可用 `[None]`，多冰块必须按稳定的零基 `body_id` 访问；`BodyField.array()` 总是保留冰块维度。

## 3 初始化建立什么状态

`IceFlow2D.__init__` 不仅分配内存，还建立能直接推进的初始状态。顺序如下。

1. 校验配置并初始化 Taichi CUDA。当前实现要求自由运动冰块、`body_ale` 材料热网格、`unified` cut-link 边界、静水平衡拆分和线性热浮力。
2. 建立单位换算，分配 $f_q,h_q$、碰撞缓冲区、压力、相场、几何以及各块冰的刚体状态。
3. 初始化各块冰的速度、转角、参考原点、质量和惯量；材料格点先填入初始冰质量。
4. 光栅化初始几何、处理接触、建立固体掩码。此时冰覆盖率已经来自材料网格。
5. 初始化气液指示场和分布函数。流体初速度为零，冰格点保存环境相场并显示刚体速度。
6. 若 `phase_warmup_steps > 0`，只推进相场，得到平滑的初始气液界面。这些预热迭代不计入 `simulation.steps`，也不推进热量或刚体位置。
7. 恢复宏观场并限幅相场，然后初始化水显热和材料冰显热。这样热状态对应的是预热结束后的水比例。
8. 从材料质量重新计算质心和惯量；按当前初始相场建立冻结的静水参考场；将流体分布函数设为静止、零动态压力的平衡态。

### 3.1 静水参考压力如何产生

初始化时对每一行的有效流体格点求平均水比例 $\bar\phi_j$，得到

$$
\rho_H(j)=\rho_a+(\rho_w-\rho_a)\bar\phi_j.
$$

没有有效流体格点的行使用初始水面指示值。相邻行的参考压力满足梯形积分

$$
p_H(j+1)-p_H(j)=\frac{g_y}{2}\,[\rho_H(j)+\rho_H(j+1)].
$$

代码在 `water_height` 对应的初始液面处设置零压基准，并对上下第一层采用半格距离。参考密度和压力以后不随液面移动而重算。流体分布函数只携带 $p_d=p-p_H$，而冰的边界受力会补回 $p_H$ 的牵引。这让初始静水负载从演化变量中拆出，同时保留静水对冰的作用。

验证入口：`_build_hydrostatic_reference`、`_initialize_hydrostatic_equilibrium`。检查 $\rho_H$、$p_H$ 沿 $x$ 方向相同，初始化流体速度为零，且 $p-p_H=0$。

## 4 step 的完整调用顺序

`step(n)` 中的 $n$ 是 **LBM 步数**，不是秒，也不是热子步数。它接受非负 Python 整数或 NumPy 整数，拒绝布尔值、浮点数和负数。`step(0)` 不改变状态。对一次迭代，源码调用关系为

```text
step
  _advance_fluid
    _collide_momentum
      _prepare_capillary_stress
      [可选] 缓存速度，先碰撞相场并测量质量通量
      _collide_momentum_kernel
        _prepare_momentum_cut_links
    [legacy] _collide_phase
    _stream
    _update_streamed_macroscopic_fields
    _update_pressure
    _bound_water_phase
  _advance_body
    _integrate_rigid_ice
    _refresh_fluid_domain(resolve_contact=True)
      _solve_contact_and_rasterize
      _update_solid_mask
      _refill_changed_nodes
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

同样要区分算法配置与示例入口：当前 `IceFlowConfig.phase_momentum_coupling` 默认是 `legacy`，但当前工作区的单冰块命令行示例默认是 `consistent_flux`，四冰块入口也复用该解析器。运行示例时应按后者理解调用顺序，或显式传入 `--phase-momentum-coupling legacy`。这一区别也是第 9.5 节配置测试失败的原因。

### 4.1 每一步用的是哪个时刻的状态

设进入一轮时 `steps = n`。流体使用此时的冰几何与热状态，累加冰壁负载；刚体立即消费本轮负载，移动冰块并重建流体；随后热对流使用更新后的几何，以及流送后恢复的速度和已保存的加速度。若本轮触发慢热更新，则用本轮可见的流体与姿态处理累计时间段 $K\Delta t$。

这是按顺序分裂的显式耦合，并非同一时间步中反复迭代直至流动、刚体、热场同时收敛。新温度和新融化几何对动量碰撞的反馈出现在下一 LBM 步。

### 4.2 两个时间和三个计数不要混淆

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

### 5.2 步骤一计算界面应力与流体加速度

`_collide_momentum` 首先调用 `_prepare_capillary_stress`。界面附近的 $\phi$ 有空间梯度，梯度大的区域形成表面张力。程序使用切向界面应力

$$
K=\kappa\left(|\boldsymbol G|^2I-\boldsymbol G\boldsymbol G^T\right),
\qquad \kappa=\frac32\sigma W,
$$

$$
(K_{xx},K_{xy},K_{yy})
=\kappa(G_y^2,-G_xG_y,G_x^2).
$$

$W=\texttt{interface\_width}$，单位为格子。$\boldsymbol G$ 的实际离散式是

$$
\boldsymbol G_i=\sum_{q=1}^{8}w_q\boldsymbol c_q
\left[2(\phi_{i+q}-\phi_i)+\frac12(4\phi_{i+q}-\phi_{i+2q}-3\phi_i)\right].
$$

这里 $i+q$ 简写世界格点 $\boldsymbol x_i+\boldsymbol c_q$。墙和冰挡住的邻点使用第 5.9 节的镜像延拓，不直接读固体内的相场。随后 `_capillary_force_density` 按共享链路计算

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

这与上述两层模板的 $\boldsymbol G$ 是不同的离散梯度。

**验证。** 纯相且无重力扰动时，毛细应力应为零；水平平界面不应产生明显毛细加速度；冰壁接触处的毛细反作用力和力矩应与流体贡献匹配。

### 5.3 步骤二按局部物性碰撞动量分布

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

### 5.4 步骤三准备冰壁反弹与冰的水动力负载

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

### 5.5 步骤四碰撞相场分布

`_collide_phase` 决定水比例如何随流动搬运，同时使气液过渡层维持有限厚度。用于理解的连续形式是守恒 Allen–Cahn 输运

$$
\partial_t\phi+\nabla\cdot(\phi\boldsymbol u)
=\nabla\cdot\left[M(\nabla\phi-\Lambda\boldsymbol n)\right],
\qquad \Lambda=\frac{4\phi(1-\phi)}{W}.
$$

$M=\texttt{mobility}$ 是格子迁移率：梯度项平滑界面，$\Lambda\boldsymbol n$ 压缩界面，二者共同形成过渡层。该连续式描述离散方法的目标，并不表示代码直接对右侧做有限差分。

实际代码先用与第 5.2 节 $\boldsymbol G$ 相同的两层模板求梯度，再求法向。默认有正则化：

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

### 5.6 步骤五流送与反弹

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

### 5.7 步骤六恢复相场与速度

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

### 5.8 步骤七恢复总压力并修复相场越界

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

随后 `_bound_water_phase` 只修复有效流体格中 $\phi<0$ 或 $\phi>1$ 的值：

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

梯度模板会访问墙或冰内部。`_phase_sample_inside_walls` 将采样索引跨半格墙镜像，例如左墙内边界为 $i=\ell$ 时，$i<\ell$ 的索引映射为 $2\ell-1-i$。镜像表达零法向导数，同时保留切向变化。

冰处先由覆盖率梯度求局部法向 $\boldsymbol n_B$，沿 cut-link 求交点 $\boldsymbol x_B$，再将采样点镜像为

$$
\boldsymbol x'=\boldsymbol x-2[(\boldsymbol x-\boldsymbol x_B)\cdot\boldsymbol n_B]\boldsymbol n_B.
$$

`_ice_phase_ghost` 对镜像点周围的有效流体双线性插值，插值的是相对于当前格的差值，以保持纯相常数场。应力不是标量，`_ice_capillary_stress` 使用

$$
Q=I-2\boldsymbol n_B\boldsymbol n_B^T,\qquad K_g=QK_{\mathrm{mirror}}Q^T.
$$

把被挡住的所有斜向采样都替换为当前格值，会抹掉切向变化；当前实现专门保留这部分变化。这里的中性润湿边界没有额外指定亲水或疏水接触角参数。

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

先更新速度，再用更新后的速度推进位置，是一种一阶的半隐式 Euler 刚体积分。上述 $M\boldsymbol g$ 已是一个格子时间步的重力冲量。默认配置 $d_v=1$，$d_\omega=0.999$；角阻尼会耗散旋转，不能期待无外力矩时角动量严格不变。

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

墙面约束把质心投影回允许区间，且只去掉朝墙外的线速度。例如左墙接触时 $V_x\leftarrow\max(V_x,0)$，右墙接触时 $V_x\leftarrow\min(V_x,0)$。该墙面处理是位置投影和速度截断，没有另行计算墙面接触点的转动冲量。

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

`_refill_changed_nodes` 检查冰块所有者是否改变。新变为冰的格点只更新刚体速度并清零流体加速度；新变为流体的格点使用 `_extrapolate_exposed_fluid`。

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

**验证。** 分别让水下冰和空气中冰消失，检查新暴露区域延续周围相态；对持续流体格点检查重建函数没有直接改动其分布函数。

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

这是会改变 `step()` 实际顺序的选项。`_collide_momentum` 先准备界面应力，缓存进入本轮时的有效速度 $\boldsymbol u^{\mathrm{cache}}$，再碰撞一次相场、测量实际碰撞产生的非对流质量通量，最后碰撞动量。外层 `_advance_fluid` 随后跳过第二次相场碰撞。

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

代码在四个轴向共享面上，用邻格通量与速度的算术平均离散完整的散度；冰和墙面不传非对流质量通量。将 $\boldsymbol F_j/\rho$ 加到动量加速度，并同时进入源项与半步速度、压力恢复。

这会改变气液界面附近的动量耦合；它不把第 6.5 和 7.3 节的局部回填、热水重映射变成全局质量投影，因此不保证 v3 水量或冰水总质量闭合。

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
5. 线性和角阻尼、无回弹接触、墙速度截断会耗散机械能；刚体材料退缩也不强制保持原冰总动量。

因此不能要求“水质量加冰质量不变”或“冰水总能量不变”作为当前代码的必过断言。若研究目标需要真实冰水转化的质量、能量和动量闭合，需要更改模型及移动边界重建，而不是只调整检查容差。

### 9.5 本快照的实际核对记录

2026 年 10 月 8 日在当前工作区运行了第 9.3 节列出的六个测试文件，并额外运行：原始矩与中心矩独立碰撞对照、平界面零毛细加速度、冰毛细反作用力和力矩、共享水热量限幅、接触冲量平衡、逐块亚网格残冰冻结，以及非法步数校验。结果为 **50 项通过、1 项失败**；没有跳过 GPU 测试。

失败项是 `StabilityConfigurationTests.test_options_default_off_and_are_independent`：它假定示例入口默认 `phase_momentum_coupling="legacy"`，而当前工作区的示例解析器实际默认 `"consistent_flux"`，显式选择同一值自然不会产生它期待的配置差异。该失败揭示的是当前入口默认值与测试预期不一致；本指南按实际代码描述两种配置，不改动求解器或测试来消除该差异。

这些检查支持对应的局部数值关系和短过程，不构成所有参数、长时间稳定性或连续方程精度的证明。尤其是“运行未发散”与“水量误差足够小”应分别验证。

第 9.2 节的完整示例也已在 CUDA 上执行通过，检查了状态关系、8 步慢热调度以及 `step(3)` 加 `step(5)` 与 `step(8)` 的结果一致性。

## 10 快速定位一个异常

观察某个错误时，先定位它第一次出现的阶段，能比直接改参数更快找到原因。

| 现象 | 优先检查 |
|---|---|
| 密度由分布和计算后异常 | 是否错误地把 $\sum f_q$ 当成密度 |
| 水静止时冰不受浮力 | cut-link 是否补入静水参考压力牵引 |
| 气液界面附近加速度很大 | 梯度模板、镜像邻点、混合密度与力密度除以密度 |
| 新露出的水下格点变空气 | 是否只搜索一层或读了同批新填格点 |
| 冰融化后仍有矩形障碍 | 掩码、接触支撑与 cut-link 是否来自当前覆盖率 |
| 热更新分批运行后不一致 | pending 是否跨 `step()` 调用保存 |
| 同一水格给多块冰供热过多 | 全部请求是否在同一限幅中处理 |
| 剩余冰看不见却持续下落 | 是否检查存在可解析固体边界 |
| 质心随融化变化后速度跳变 | 是否按 $\omega\mathcal P\Delta\boldsymbol C$ 同步质心速度 |
| 热子步数突然很大 | 先检查面出流率、有效速度、有限值与格子单位 |
| D2Q5 温度与显热不一致 | 是否同步 $\sum g_q=E$，以及清除消失格点的分布 |

初学者最有效的阅读路线是：从 `step` 展开三个阶段，在每个阶段开始和结束取一次快照，按第 9.1 节的关系核对，再选择相应的单内核测试。不要一次打印整个域的全部分布；先选一个格点、一条链路或一块材料格，手算该处的更新。

## 附录 A simulator.py 完整方法索引
以下定位对应本快照的函数定义行。`@ti.kernel` 在 GPU 上批量更新字段；`@ti.func` 在内核内展开为设备计算；无这类标记的方法负责 Python 调度、主机读取或包装。下列索引覆盖该文件全部顶层函数与类方法，包括兼容视图和输出方法。
- [ensure\_taichi\_cuda](../simulator.py#L51)：建立或校验 CUDA 运行时，拒绝 CPU 回退。

### _DerivedField 方法

- [\_\_init\_\_](../simulator.py#L78) [Python]：校验输入、分配状态并建立初始条件。
- [\_\_getitem\_\_](../simulator.py#L86) [Python]：在设备或主机端读取字段，并处理单冰块索引。
- [\_\_setitem\_\_](../simulator.py#L99) [Python]：向可写底层状态提交主机修改。
- [to\_numpy](../simulator.py#L104) [Python]：在主机端构造当前派生值的 NumPy 副本。

### MovingBodyThermal2D 方法

- [\_\_init\_\_](../simulator.py#L128) [Python]：校验输入、分配状态并建立初始条件。
- [maximum\_advection\_time\_step\_s](../simulator.py#L280) [Python]：由面出流率和 Courant 上限给出最大对流步长。
- [\_validate\_substep\_count](../simulator.py#L290) [Python]：检查热子步数是否超过配置上限。
- [required\_advection\_substeps](../simulator.py#L315) [Python]：计算快对流过程所需的子步数。
- [required\_diffusion\_substeps](../simulator.py#L339) [Python]：计算慢导热过程所需的子步数。
- [initialize](../simulator.py#L351) [Python]：初始化材料质量、显热和水热状态。
- [advance\_fast](../simulator.py#L367) [Python]：每 LBM 步重映射热水区域并执行迎风对流。
- [advance\_slow](../simulator.py#L396) [Python]：执行导热、界面换热、潜热融化和逐子步回调。
- [synchronize\_water\_aperture](../simulator.py#L440) [Python]：使热水面积匹配相场和掩码，局部重建显热。
- [\_initialize\_body\_state](../simulator.py#L458) [kernel]：为有效材料格设置初始冰质量和冰显热。
- [\_initialize\_water\_state](../simulator.py#L472) [kernel]：为当前湿流体格设置水面积和水显热。
- [\_build\_phase\_aperture\_targets](../simulator.py#L501) [kernel]：冻结旧热状态并计算重映射后的目标显热。
- [\_apply\_phase\_aperture\_transfer](../simulator.py#L547) [kernel]：统一提交重映射的水面积和显热。
- [\_accumulate\_body\_conduction](../simulator.py#L560) [kernel]：按材料轴向邻接累加冰内导热增量。
- [\_compute\_water\_conduction\_flux\_x](../simulator.py#L587) [kernel]：计算水 x 面导热及左右墙热通量。
- [\_compute\_water\_advection\_flux\_x](../simulator.py#L685) [kernel]：计算水 x 面的上游面积与显热通量。
- [\_compute\_water\_conduction\_flux\_y](../simulator.py#L736) [kernel]：计算水 y 面导热及上下墙热通量。
- [\_compute\_water\_advection\_flux\_y](../simulator.py#L834) [kernel]：计算水 y 面的上游面积与显热通量。
- [\_update\_water\_advection](../simulator.py#L885) [kernel]：按面通量差更新水面积和显热。
- [\_update\_water\_conduction](../simulator.py#L904) [kernel]：按导热通量差只更新水显热。
- [\_exchange\_interface\_heat](../simulator.py#L916) [kernel]：将一个水格的全部冰水供热请求统一限幅并分发。
- [\_apply\_body\_heat\_and\_phase\_change](../simulator.py#L1057) [kernel]：先处理冰显热，再按潜热删除融化材料。
- [\_body\_fraction\_at](../simulator.py#L1074) [func]：从材料质量推导剩余冰比例。
- [\_body\_temperature\_at](../simulator.py#L1084) [func]：从材料质量和冰显热推导冰温。
- [\_water\_temperature\_at](../simulator.py#L1094) [func]：从水面积和显热推导水温，干格显示设定气温。
- [\_body\_grid\_coordinates](../simulator.py#L1104) [func]：把世界格心映射到材料数组的插值坐标。
- [\_water\_aperture\_target](../simulator.py#L1133) [func]：从湿阈值、相场和掩码给出目标热水面积。
- [\_body\_fraction\_numpy](../simulator.py#L1151) [Python]：构造材料冰比例的主机数组。
- [\_body\_temperature\_numpy](../simulator.py#L1160) [Python]：构造材料冰温的主机数组。
- [\_water\_temperature\_numpy](../simulator.py#L1173) [Python]：构造水温的主机数组。
- [sample\_world\_fields](../simulator.py#L1185) [Python]：将权威热状态映射为世界网格输出数组。

### IceFlow2D 方法

- [\_\_init\_\_](../simulator.py#L1269) [Python]：校验输入、分配状态并建立初始条件。
- [step](../simulator.py#L1535) [Python]：校验步数，循环推进流体、刚体、热过程和 LBM 计数。
- [\_advance\_fluid](../simulator.py#L1549) [Python]：推进两组分布并恢复相场、速度和压力。
- [\_advance\_body](../simulator.py#L1558) [Python]：积分刚体并按新姿态刷新流体域。
- [\_refresh\_fluid\_domain](../simulator.py#L1562) [Python]：统一处理光栅化、固体掩码提交和环境回填。
- [\_advance\_thermal](../simulator.py#L1568) [Python]：推进快热过程，累计到间隔后触发慢过程。
- [physical\_time\_s](../simulator.py#L1579) [Python]：由 LBM 总步数换算物理流动时间。
- [\_advance\_moving\_thermal\_fast](../simulator.py#L1582) [Python]：收集出流率，调用热对流与水域重映射。
- [\_advance\_moving\_thermal\_slow](../simulator.py#L1598) [Python]：把累计 LBM 时段交给慢导热与融化。
- [\_refresh\_eroded\_body](../simulator.py#L1610) [Python]：每次材料侵蚀后同步质量属性、几何、流体和热水。
- [\_is\_fluid\_cell](../simulator.py#L1621) [func]：判断格点既不是墙也不是冰。
- [\_measure\_water\_outflow\_rate](../simulator.py#L1625) [kernel]：归约湿流体格的最大面出流速度之和。
- [\_collect\_thermal\_stability\_metrics](../simulator.py#L1661) [kernel]：收集完整流体速度和非有限状态指标。
- [\_check\_thermal\_advection\_stability](../simulator.py#L1690) [Python]：执行可选的热对流前完整状态检查。
- [\_validate\_thermal\_advection\_velocity](../simulator.py#L1702) [Python]：对非有限状态或极端格子速度给出定位异常。
- [\_body\_velocity\_at](../simulator.py#L1770) [func]：从质心速度和角速度求冰上任意点的速度。
- [\_cell\_point](../simulator.py#L1776) [func]：返回世界格心的格子坐标。
- [\_phase\_sample\_inside\_walls](../simulator.py#L1780) [func]：将跨容器墙的模板索引作半格镜像。
- [\_ice\_mirror\_sample](../simulator.py#L1801) [func]：按冰表面法向和 cut-link 交点反射采样位置。
- [\_ice\_phase\_ghost](../simulator.py#L1833) [func]：在冰表面镜像点插值得到相场虚拟邻值。
- [\_phase\_neighbor](../simulator.py#L1854) [func]：按距离读取流体、墙镜像或冰镜像邻相场。
- [\_body\_local\_coordinates](../simulator.py#L1882) [func]：将世界坐标逆旋转到材料参考坐标。
- [\_nearest\_body\_index](../simulator.py#L1898) [func]：从材料坐标计算最近所属材料格索引。
- [\_sample\_body\_fraction](../simulator.py#L1915) [func]：双线性插值当前剩余材料冰比例。
- [\_calculate\_moving\_body\_fraction](../simulator.py#L1947) [kernel]：计算各块世界覆盖率及其最大值合并场。
- [\_commit\_moving\_body\_raster](../simulator.py#L1959) [kernel]：接触约束完成后裁掉墙内覆盖率。
- [\_initialize\_body](../simulator.py#L1967) [Python]：初始化各块冰的刚体状态和负载累加器。
- [\_initialize\_geometry](../simulator.py#L1989) [kernel]：建立空的初始固体及旧固体掩码。
- [\_initialize\_fluid](../simulator.py#L1996) [kernel]：设置初始水空气指示场、环境储备和分布。
- [\_build\_hydrostatic\_reference](../simulator.py#L2028) [Python]：按初始行平均密度积分并冻结静水参考场。
- [\_initialize\_hydrostatic\_equilibrium](../simulator.py#L2067) [kernel]：建立零动态压力的静止流体平衡态。
- [\_initialize\_moving\_body\_geometry](../simulator.py#L2101) [Python]：提交初始固体所有者并同步旧掩码。
- [\_update\_solid\_mask](../simulator.py#L2107) [kernel]：按当前阈值覆盖率提交所有冰块的锐界掩码。
- [\_update\_moving\_body\_mass\_properties](../simulator.py#L2124) [Python]：调度材料质量属性归约和质心速度同步。
- [\_reduce\_moving\_body\_mass\_properties](../simulator.py#L2131) [kernel]：逐材料格归约质量、一阶质量矩和原点惯量。
- [\_apply\_moving\_body\_mass\_properties](../simulator.py#L2161) [kernel]：更新质心、惯量、活跃状态和剩余材料速度。
- [\_extrapolate\_exposed\_fluid](../simulator.py#L2199) [func]：在最近持续流体邻域外推相场、动态压力和速度。
- [\_refill\_changed\_nodes](../simulator.py#L2245) [kernel]：处理所有者改变的格点并恢复新暴露流体分布。
- [\_prepare\_capillary\_stress](../simulator.py#L2297) [kernel]：按相场梯度建立切向毛细应力张量。
- [\_phase\_central\_gradient](../simulator.py#L2336) [func]：计算密度源项和压力恢复用的一层中心梯度。
- [\_capillary\_traction](../simulator.py#L2348) [func]：计算对称应力张量乘链路方向。
- [\_ice\_capillary\_stress](../simulator.py#L2357) [func]：镜像插值并反射冰邻点的应力张量。
- [\_capillary\_force\_density](../simulator.py#L2390) [func]：按共享链路牵引求毛细力密度。
- [\_fluid\_kinematic\_viscosity\_lattice](../simulator.py#L2438) [func]：计算相场和水温决定的分子格子黏度。
- [\_fluid\_force\_and\_gradient](../simulator.py#L2454) [func]：组装重力残量、热浮力、毛细加速度和中心梯度。
- [\_collide\_momentum](../simulator.py#L2497) [Python]：准备应力及可选通量耦合，再调度动量内核。
- [\_collide\_momentum\_kernel](../simulator.py#L2513) [kernel]：投影矩、处理源项与松弛、逆变换并准备冰链路。
- [\_collide\_phase](../simulator.py#L2777) [kernel]：以中心矩推进相场分布并闭合局部零阶矩。
- [\_stream](../simulator.py#L2938) [kernel]：同时 push 流送动量和相场，并处理阻塞链路。
- [\_stream\_phase\_only](../simulator.py#L2980) [kernel]：在初始化相场预热时只流送相场分布。
- [\_update\_streamed\_macroscopic\_fields](../simulator.py#L3016) [kernel]：恢复相场与未修正速度，并处理墙和冰速度。
- [\_update\_streamed\_phase](../simulator.py#L3048) [kernel]：在预热中仅恢复流送后的相场。
- [\_warm\_start\_phase](../simulator.py#L3059) [Python]：初始化时重复相场碰撞、流送和恢复。
- [\_update\_pressure](../simulator.py#L3066) [kernel]：从非静止动量分布恢复动态压力，再加静水压力。
- [\_prepare\_moving\_body\_contact\_support](../simulator.py#L3097) [Python]：供外部调用的覆盖率与墙支撑重建包装。
- [\_prepare\_moving\_body\_contact\_support\_kernel](../simulator.py#L3111) [kernel]：重置或归约各块当前墙接触支撑。
- [\_reset\_moving\_body\_contact\_support](../simulator.py#L3123) [func]：初始化支撑极值归约缓存。
- [\_reduce\_moving\_body\_contact\_extrema](../simulator.py#L3130) [func]：归约当前锐界单元方格的世界支撑极值。
- [\_finalize\_moving\_body\_contact\_support](../simulator.py#L3157) [func]：将无锐界接触几何的支撑清零。
- [\_integrate\_rigid\_ice](../simulator.py#L3164) [kernel]：消费负载推进刚体；无解析边界的残冰固定姿态。
- [\_project\_moving\_body\_inside\_container](../simulator.py#L3215) [kernel]：修正穿墙位置并截断朝墙外的线速度。
- [\_solve\_contact\_and\_rasterize](../simulator.py#L3286) [Python]：迭代墙与冰块对接触，提交共同接受的几何。
- [\_reduce\_phase\_change\_geometry\_volume](../simulator.py#L3330) [kernel]：统计锐界固体格点数量。
- [phase\_change\_solid\_volume\_cells](../simulator.py#L3337) [Python]：返回由剩余材料质量换算的冰体积格数。
- [phase\_change\_geometry\_volume\_cells](../simulator.py#L3340) [Python]：返回阈值几何占据的固体格数。
- [\_bound\_water\_phase](../simulator.py#L3344) [kernel]：只修复越界流体相场，并同步其相场分布。
- [sample\_thermal\_fields](../simulator.py#L3369) [Python]：调度供输出用的世界热字段采样。
- [\_wall\_at](../simulator.py#L3373) [func]：在设备端按域边界计算静态墙标记。
- [\_wall\_numpy](../simulator.py#L3383) [Python]：在主机端构造静态墙标记数组。
- [\_signed\_distance\_at](../simulator.py#L3394) [func]：返回覆盖率阈值水平集，非真实欧氏距离。
- [\_signed\_distance\_numpy](../simulator.py#L3402) [Python]：在主机端构造相同的水平集数组。
- [\_reference\_pressure\_at](../simulator.py#L3412) [func]：读取对应行的冻结静水参考压力。
- [\_reference\_density\_at](../simulator.py#L3416) [func]：读取对应行的冻结静水参考密度。
- [\_physical\_velocity\_at](../simulator.py#L3420) [func]：在有效流体格添加半步力修正，其他格返回零。
- [\_physical\_velocity\_numpy](../simulator.py#L3429) [Python]：构造相同的有效流体速度主机数组。
- [\_read\_body\_initial\_mass\_lattice](../simulator.py#L3440) [func]：读取指定冰块的初始格子质量。
- [\_read\_body\_active](../simulator.py#L3445) [func]：由正的剩余惯量给出冰块活跃标记。
- [\_read\_cumulative\_melted\_mass\_lattice](../simulator.py#L3450) [func]：返回初始质量减剩余质量的融化量。
- [\_read\_phase\_change\_initial\_solid\_volume](../simulator.py#L3455) [func]：将全部初始冰质量换算为体积格数。
- [\_read\_phase\_change\_current\_solid\_volume](../simulator.py#L3464) [func]：将全部剩余冰质量换算为体积格数。
- [\_read\_\_body\_contact\_geometry\_active](../simulator.py#L3473) [func]：从支撑极值判断是否有接触几何。
- [\_body\_center\_at](../simulator.py#L3482) [func]：由参考原点、转角和材料质心求世界质心。
- [\_body\_center\_numpy](../simulator.py#L3492) [Python]：构造各块世界质心的主机数组。
- [\_set\_body\_center\_host](../simulator.py#L3505) [Python]：将主机设置的世界质心转换为材料参考原点。
- [\_body\_level\_at](../simulator.py#L3521) [func]：返回指定冰块的覆盖率阈值水平集。
- [\_cut\_link\_fraction](../simulator.py#L3528) [func]：求流体到冰链路的阈值交点比例并限幅。
- [\_prepare\_momentum\_cut\_links](../simulator.py#L3545) [func]：准备冰壁反弹并按所属冰块累加全部边界负载。
- [\_allocate\_populations](../simulator.py#L3661) [Python]：分配按 q 主序存储但以 i、j、q 访问的 GPU 分布。

## 附录 B 关联模块与源码快照

- [config.py](../config.py)：输入配置、物性、单位换算和合法性校验。
- [lattice.py](../lattice.py)：D2Q9 方向、平衡分布、密度混合与矩逆变换。
- [body_state.py](../body_state.py)：统一的多冰块字段与单冰块兼容索引。
- [contact.py](../contact.py)：冰块间检测、位置修正和法向冲量。
- [stability_options.py](../stability_options.py)：HRR、质量通量动量耦合及应力回填。
- [thermal_d2q5.py](../thermal_d2q5.py)：固定松弛时间的五分布导热和能量矩同步。
- [viscosity.py](../viscosity.py)：分子黏度系数和主机物性计算。
- [source manifest](simulator_method_source_manifest.json)：当前文件 SHA256、方法行号和验证范围。

本指南有同目录 PDF 版本。Markdown 是维护源，修改公式后可使用 Pandoc 和 XeLaTeX 重新生成；中文字体与版式参数保存在 `simulator_method_pdf_header.tex`，生成命令见 `build_simulator_method_pdf.sh`。
