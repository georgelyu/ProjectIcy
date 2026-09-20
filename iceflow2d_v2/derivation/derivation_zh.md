# 从 Wang 等（2022）的 D2Q5 温度模型到运动冰块的融化

## 1. 结论、依据和验证范围

**可以使用 D2Q5 处理冰水固液相变。可迁移的是温度扩散的离散结构，不能直接迁移气液相变的热力学闭合。** 对纯物质，温度在熔点处不能确定液相率，因此温度场之外必须保留潜热状态。本工作给出两种明确区分的形式：

1. 沿原文式（21）扩展的温度分布模型：仍有 $T=\sum_i g_i$，但源项必须包含正确的融化质量源及运动输运。
2. 实际实现、数值验证的守恒焓 D2Q5 模型：$\sum_i g_i=H$，移动方向的平衡分布仍仅依赖温度，潜热存入静止方向。它继承原文低阶矩的思路，但不是原文式（21）的原样实现。

本次独立编写 thermal.py、verify.py；没有读取、导入或参考仓库已有代码。唯一文献输入是用户提供的：

Lei Wang, Jiangxu Huang, Kun He, *Thermal lattice Boltzmann model for liquid-vapor phase change*, Physical Review E **106**, 055308 (2022), DOI: 10.1103/PhysRevE.106.055308。

下文的守恒律、固液闭合和解析基准由方程直接推导。没有检索其他论文或使用外部基准代码。测试使用无量纲参数，它们不是该论文给出的真实冰水物性。

| 内容 | 本次处理 |
|---|---|
| 单相、变热容、变导热率的热传导 | 推导并验证 |
| 纯物质熔点平台、潜热、不同固液比热 | 推导并验证 |
| 一相 / 两相 Stefan 问题 | 与自行推导的解析解比较、网格加密 |
| 完全浸没冰块的导热融化 | 二维闭合热系统验证 |
| 冰块平移、转动引起的焓输运 | 给定速度的守恒和几何输运验证 |
| 运动中融化 | 给定冰水共同平移速度的 Galilean 一致性验证 |
| 流体受力反馈决定冰块上浮、漂移、角速度 | 给出耦合方程；本次代码未求解该力学系统 |
| 水–空气自由液面及部分露出水面的冰块 | 给出接入条件；本次未实现、未验证三物态耦合 |

这里“给定运动的验证”不是“自由漂浮受力耦合验证”。目前结果可以验证热模型及其运动输运，不能用来宣称已预测真实冰块的上浮速度、姿态或露出水面的比例。

## 2. 原文究竟提供了什么

原文式（14）、（20）为

$$
C_v\partial_t T=\nabla\cdot(k\nabla T)+Q_{\rm lv},\qquad
C_v=\rho c_v,
$$
$$
Q_{\rm lv}=-C_v\mathbf u\cdot\nabla T
-T\left(\frac{\partial p_{\rm EOS}}{\partial T}\right)_\rho
\nabla\cdot\mathbf u .
\tag{1}
$$

论文通过气液 EOS、密度场与温度场的耦合发生蒸发 / 沸腾。式（1）中的压力相关项不是可以直接改名为“融化潜热”的常数热源。

原文的关键设计是：

- 式（21）：把热容保留在时间储存项前，而不是将正确算子 $C_v^{-1}\nabla\cdot(k\nabla T)$ 改成 $\nabla\cdot[(k/C_v)\nabla T]$。
- 式（22）、（26）：使用 $g_i^{eq}=\widehat w_iT$ 和 D2Q5，不在平衡分布中放入速度的二次项。
- 式（24）：在该特定时间离散中用 $S_i=\widehat w_i C_v\Delta t\,\partial_t^2T/2$ 修正时间偏差。
- 式（36）、（49）、（50）：一阶非平衡矩给出局部温度梯度。
- 式（38）、（52）、（53）：恢复的是导热率 $k$，其与松弛时间的关系不应再误用局部热扩散率 $k/C_v$。

例如

$$
\nabla\cdot\left(\frac{k}{C}\nabla T\right)
=\frac{1}{C}\nabla\cdot(k\nabla T)
-\frac{k}{C^2}\nabla C\cdot\nabla T .
\tag{2}
$$

当 $C$ 跨冰水界面变化时，末项不能凭空出现。改变一个普通温度 LBM 的局部扩散率，未必就是正确的变物性传热。

原文验证了液滴蒸发和沸腾，并没有验证冰的融化、固体刚体运动或水–空气自由液面的融冰问题。原文式（54）的固壁润湿条件也不是融化界面的运动方程。

## 3. 先建立固液相变的守恒方程

### 3.1 一般的两速度、两密度形式

以 $f$ 表示液相体积分数，$1-f$ 表示固相体积分数；$\mathbf v_s$ 是冰的材料速度，$\mathbf u_l$ 是液水的速度。令 $\dot m>0$ 表示单位总体积、单位时间由冰转成水的质量，单位为质量 / 体积 / 时间。两相密度在各自相内取常数，但不必相等：

$$
\partial_t[\rho_s(1-f)]+\nabla\cdot[\rho_s(1-f)\mathbf v_s]=-\dot m,
\tag{3a}
$$
$$
\partial_t(\rho_l f)+\nabla\cdot(\rho_l f\mathbf u_l)=\dot m.
\tag{3b}
$$

取 $\theta=T-T_m$，以熔点处固体焓为零点：

$$
h_s=c_s\theta,\qquad h_l=c_l\theta+L,\qquad L>0.
$$

这里 $c_s,c_l$ 是质量比热；低速、近常压的固液模型采用显热焓 / 比热闭合，不直接照搬原文的气液 $c_v$ 和 EOS。

单位总体积的能量和焓通量为

$$
\mathcal H=\rho_s(1-f)h_s+\rho_lf h_l,
\qquad
\mathbf J_H=\rho_s(1-f)h_s\mathbf v_s+\rho_l f h_l\mathbf u_l.
\tag{4}
$$

忽略黏性耗散、显著压缩功等效应，守恒方程为

$$
\boxed{\partial_t\mathcal H+\nabla\cdot\mathbf J_H
=\nabla\cdot(k\nabla T)+Q_{\rm ext}.}
\tag{5}
$$

将式（3）代入式（5），不需要求任何热容的空间梯度，即得温度形式

$$
\boxed{C\partial_t\theta
=\nabla\cdot(k\nabla\theta)
-\mathbf A\cdot\nabla\theta
-\dot m\,[L+(c_l-c_s)\theta]+Q_{\rm ext},}
\tag{6}
$$
$$
C=\rho_s(1-f)c_s+\rho_l f c_l,\qquad
\mathbf A=\rho_s(1-f)c_s\mathbf v_s+\rho_l f c_l\mathbf u_l.
$$

这就是应该放入 Wang 型温度求解器的固液方程。相变项中出现的是质量转化率 $\dot m$，不是任意选择的液相率时间差。

### 3.2 等密度、共同速度的验证模型

本次代码使用 $\rho_s=\rho_l=\rho_0$。用同一个连续速度场 $\mathbf v$ 表示固体中的刚体速度和液体中的流体速度，并假定
$\nabla\cdot\mathbf v=0$。测试中的速度直接给定，不来自流体受力求解。

此时
$$
\dot m=\rho_0D_tf,\qquad D_t=\partial_t+\mathbf v\cdot\nabla.
$$

取体积热容 $C_s=\rho_0c_s,\ C_l=\rho_0c_l$，体积潜热 $\mathcal L=\rho_0L$，则

$$
\mathcal H=C(f)\theta+\mathcal Lf,\qquad
C(f)=(1-f)C_s+fC_l,
\tag{7}
$$
$$
\boxed{C(f)D_t\theta
=\nabla\cdot(k\nabla\theta)
-[\mathcal L+(C_l-C_s)\theta]D_tf+Q_{\rm ext}.}
\tag{8}
$$

**对自由运动冰块，潜热项必须含物质导数 $D_tf$。** 固定网格上冰块即使完全不融化，平移或转动也通常有 $\partial_t f\ne0$，但正确结果是 $D_tf=0$。仅使用 $-\mathcal L\partial_t f$ 会把材料运动误当成熔化 / 凝固。

若固定冰体且界面上没有体积输运，才可以相应退化为局部时间导数。若 $C_s\ne C_l$，有限厚度的混合状态中还应保留 $(C_l-C_s)\theta D_tf$；严格纯物质界面上 $\theta=0$，该显热差项在界面处消失。

式（8）的守恒形式是

$$
\boxed{\partial_t\mathcal H+\nabla\cdot(\mathbf v\mathcal H)
=\nabla\cdot(k\nabla\theta)+Q_{\rm ext}.}
\tag{9}
$$

### 3.3 Stefan 界面条件由守恒律得到

设法向 $\mathbf n$ 从冰指向水，界面速度为 $\mathbf V_\Gamma$，熔化质量通量 $j>0$：

$$
j=\rho_s(\mathbf v_s-\mathbf V_\Gamma)\cdot\mathbf n
=\rho_l(\mathbf u_l-\mathbf V_\Gamma)\cdot\mathbf n .
\tag{10}
$$

对式（5）在界面两侧的薄层积分，厚度趋于零，忽略界面储能，并取 $T=T_m$，得到

$$
\boxed{jL=(\mathbf q_s-\mathbf q_l)\cdot\mathbf n
=k_l\partial_nT_l-k_s\partial_nT_s,\qquad
\mathbf q=-k\nabla T.}
\tag{11}
$$

因此，焓方程的弱解包含正确的 Stefan 热流跳跃条件。实现中不需要再额外加一次同样的潜热源，否则会重复计算潜热。

等密度只是前期热模型的受控简化。真实冰水密度不同，式（3）相加后不能无条件设体积平均速度散度为零：

$$
\nabla\cdot[(1-f)\mathbf v_s+f\mathbf u_l]
=\dot m\left(\frac1{\rho_l}-\frac1{\rho_s}\right).
\tag{12}
$$

未来计算自由液面位移时，这个体积变化不能忽略后仍宣称水量严格正确。

## 4. 沿原文直接扩展的温度 D2Q5

先无量纲化；否则 $(C-1)$ 中的 “1” 不能和有量纲热容相减。选参考体积热容 $C_r$、温差 $\Delta T_r$、网格间距 $\Delta x$、时间步长 $\Delta t$：

$$
\Theta=\frac{T-T_m}{\Delta T_r},\quad
C^*=\frac C{C_r},\quad
K=\frac{k\Delta t}{C_r\Delta x^2},\quad
H=\frac{\mathcal H}{C_r\Delta T_r},\quad
\ell=\frac{\rho_0L}{C_r\Delta T_r}.
\tag{13}
$$

以下格子单位为 $\Delta x=\Delta t=c=1$；保留 $\delta t$ 的公式用于展示时间步因素。

格子速度及权重沿原文取

$$
\mathbf e_0=(0,0),\
\mathbf e_1=(1,0),\
\mathbf e_2=(0,1),\
\mathbf e_3=(-1,0),\
\mathbf e_4=(0,-1),
\tag{14}
$$
$$
w_0=1-\varpi,\quad w_{1,2,3,4}=\frac{\varpi}{4},\quad
c_T^2=\frac{\varpi}{2},\quad 0<\varpi<1.
\tag{15}
$$

把温度形式写成
$C^*\partial_t\Theta=\nabla\cdot(K\nabla\Theta)+Q_{\rm sl}$。
在等密度共同速度模型中，

$$
Q_{\rm sl}
=-C^*\mathbf v\cdot\nabla\Theta
-[\ell+(C_l^*-C_s^*)\Theta](\partial_t f+\mathbf v\cdot\nabla f)
+Q_{\rm ext}^*.
\tag{16}
$$

一般情况应使用式（6）按式（13）无量纲化后的源项，而不是式（16）。

沿原文式（21），形式上的固液扩展为

$$
\begin{aligned}
C^*g_i(\mathbf x+\mathbf e_i\delta t,t+\delta t)
={}&g_i(\mathbf x,t)
+(C^*-1)g_i(\mathbf x+\mathbf e_i\delta t,t)\\
&-\tau^{-1}(g_i-w_i\Theta)
+\delta t\,w_iQ_{\rm sl}
+\frac{\delta t^2}{2}w_iC^*\partial_t^2\Theta ,
\end{aligned}
\tag{17}
$$
$$
\Theta=\sum_i g_i .
$$

式（17）中与原文一致，未标位置的系数属于该方程展开点 $(\mathbf x,t)$。变热容情况下不能在 pull 实现中随意把它换成另一格点的系数，还声称是同一个离散式。

令 $g_i=g_i^{(0)}+\epsilon g_i^{(1)}+\epsilon^2g_i^{(2)}$，
$\partial_t=\epsilon\partial_{t_1}+\epsilon^2\partial_{t_2}$，
$\nabla=\epsilon\nabla_1$，且源项与原文采用相容标度。沿原文式（29）至（38）的取矩步骤：

$$
g_i^{(0)}=w_i\Theta,\qquad
C^*\partial_{t_1}\Theta=Q_{\rm sl}^{(1)},
\tag{18}
$$
$$
\sum_i\mathbf e_i g_i^{(1)}
=-\tau\delta t\,c_T^2\nabla_1\Theta,
\tag{19}
$$
$$
C^*\partial_{t_2}\Theta
=\nabla_1\cdot[(\tau-\tfrac12)c_T^2\delta t\nabla_1\Theta].
\tag{20}
$$

式（17）的时间修正抵消相应的 $C^*\delta t\,\partial_{t_1}^2\Theta/2$。于是

$$
K=(\tau-\tfrac12)c_T^2\delta t.
\tag{21}
$$

这说明**直接温度扩展在连续极限上成立**。但它还不是完整的可执行固液算法：未知 $f^{n+1}$ 出现在相变源中，必须和局部焓约束一起求解；简单滞后 $\partial_t f$、熔化后强行重置温度，或忽略原文时间修正，都没有被上述推导证明。相界面处还必须检查离散能量守恒与收敛，不能只靠光滑场的 Chapman–Enskog 展开。

本次没有把式（17）作为数值验证对象。下面给出一个潜热闭合直接、离散能量可核验的实现形式。

## 5. 实际验证的守恒焓 D2Q5

### 5.1 局部状态反演

下文 $C_s,C_l$ 都指已经除以 $C_r$ 的无量纲体积热容，省略星号。对于纯物质、零熔化温区：

$$
(\Theta,f)=
\begin{cases}
(H/C_s,0),&H<0,\\
(0,H/\ell),&0\le H\le\ell,\\
((H-\ell)/C_l,1),&H>\ell.
\end{cases}
\tag{22}
$$

这是焓和相态的物理反演，而不是对温度方程结果作非守恒裁剪。温度等于熔点时，可以继续吸热并增加 $f$；$f=1$ 后才继续升温。冷却时顺序相反。

为验证单相传热，可以令 $\ell=0,\ C_s=C_l=C$，于是 $H=C\Theta$。代码中特别处理了 $\ell=0$ 分支，不会除零。

### 5.2 平衡分布和矩

取

$$
\boxed{
g_0^{eq}=H-\varpi\Theta,\qquad
g_{1,2,3,4}^{eq}=\frac{\varpi}{4}\Theta .
}
\tag{23}
$$

与原文的联系很直接：仍是四个移动方向的线性温度平衡分布，只把额外的显热储存及潜热放入静止方向。

$$
\sum_i g_i^{eq}=H,\qquad
\sum_i\mathbf e_i g_i^{eq}=0,\qquad
\sum_i e_{i\alpha}e_{i\beta}g_i^{eq}=c_T^2\Theta\delta_{\alpha\beta}.
\tag{24}
$$

**不能把五个方向全设成 $w_iH$**。这样二阶矩变成 $c_T^2H I$，恢复的会是扩散焓的方程；熔点平台上虽然温度梯度为零，液相率梯度仍可能产生虚假的潜热扩散。

采用原文式（40）的矩阵

$$
M=
\begin{pmatrix}
1&1&1&1&1\\
0&1&0&-1&0\\
0&0&1&0&-1\\
0&1&1&1&1\\
0&1&-1&1&-1
\end{pmatrix},
\qquad
\mathbf m^{eq}=(H,0,0,\varpi\Theta,0)^T.
\tag{25}
$$

纯导热步骤为

$$
\mathbf m^\star=\mathbf m-
\operatorname{diag}(s_0,s_q,s_q,1,1)(\mathbf m-\mathbf m^{eq}),
\quad
g_i(\mathbf x+\mathbf e_i,t+1)=g_i^\star(\mathbf x,t).
\tag{26}
$$

$m_0=H$ 守恒，因此 $s_0$ 不影响这一碰撞步骤。令

$$
\boxed{s_q^{-1}=\tau_q=\frac12+\frac{K(f)}{c_T^2}},\qquad
K(f)=(1-f)K_s+fK_l.
\tag{27}
$$

其中相变格点的导热率插值是本次选择的数值闭合，不是由论文规定的界面微观定律。锐界面极限通过加密网格检验。

若记 $m_1=g_1-g_3,\ m_2=g_2-g_4$，碰撞后的四个移动分布可以直接写成

$$
\begin{aligned}
g_1^\star&=\varpi\Theta/4+(1-s_q)m_1/2,&
g_3^\star&=\varpi\Theta/4-(1-s_q)m_1/2,\\
g_2^\star&=\varpi\Theta/4+(1-s_q)m_2/2,&
g_4^\star&=\varpi\Theta/4-(1-s_q)m_2/2,\\
g_0^\star&=H-\varpi\Theta.
\end{aligned}
\tag{28}
$$

### 5.3 Chapman–Enskog 恢复能量方程

先只看无对流、无外源的导热子步。采用扩散标度
$\partial_t=\epsilon^2\partial_{t_2},\ \nabla=\epsilon\nabla_1$。

零阶给出式（23）。一阶方程为

$$
\mathbf e_i\cdot\nabla_1 g_i^{(0)}
=-\delta t^{-1}(M^{-1}SM)_{ij}g_j^{(1)}.
$$

对速度取一阶矩：

$$
\mathbf j^{(1)}=\sum_i\mathbf e_i g_i^{(1)}
=-\delta t\,c_T^2s_q^{-1}\nabla_1\Theta.
\tag{29}
$$

二阶方程取零阶矩：

$$
\partial_{t_2}H+\nabla_1\cdot\mathbf j^{(1)}
+\frac{\delta t}{2}c_T^2\nabla_1^2\Theta=0.
\tag{30}
$$

将式（29）代入，得到

$$
\boxed{\partial_t H
=\nabla\cdot\left[
c_T^2\delta t(s_q^{-1}-\tfrac12)\nabla\Theta
\right]
=\nabla\cdot(K\nabla\Theta).}
\tag{31}
$$

即使 $K$ 随空间变化，系数也留在散度里面；热容通过 $H\leftrightarrow\Theta,f$ 进入储能关系。恢复的是
$C\partial_t\Theta=\nabla\cdot(K\nabla\Theta)$，不是式（2）中的错误算子。

这里没有在格子碰撞中塞入原文一阶对流源，采用的也是扩散子步标度，因此**不应该机械追加原文的 $S_i$**。对流在单独的守恒步骤中完成。上述推导不等于对任意时间分裂和任意界面参数的无条件精度证明；实际分裂格式另做对流–扩散收敛验证。

在光滑区域仍有局部通量关系

$$
\nabla\Theta\simeq-\frac{s_q}{c_T^2\delta t}
\begin{pmatrix}m_1\\m_2\end{pmatrix},
\qquad
\mathbf q\simeq(1-s_q/2)
\begin{pmatrix}m_1\\m_2\end{pmatrix}.
\tag{32}
$$

D2Q5 的二阶各向同性足以恢复式（31）。它没有完整的四阶各向同性，例如
$\sum_iw_ie_{ix}^2e_{iy}^2=0$，所以不应声称高阶截断误差也完全各向同性。

### 5.4 潜热守恒和移动输运

每个导热子步按输入与输出的移动分布之差更新：

$$
H_{\mathbf x}^{n+1}=H_{\mathbf x}^{n}
+\sum_{i=1}^4
\left[g_i^\star(\mathbf x-\mathbf e_i)-g_i^\star(\mathbf x)\right].
\tag{33}
$$

它在精确算术下与式（26）的零阶矩完全等价。域内相邻格点的通量成对抵消；周期或绝热边界下全域 $\sum H$ 守恒。之后由式（22）反演 $\Theta,f$，并令
$g_0=H-\sum_{i=1}^4g_i$。

本实现保留一个独立的 $H$ 和补偿量，避免每一步从 $g_0+\sum g_{\rm moving}$ 反复相减损失小热增量。这不是另加一种潜热模型。

运动时求解

$$
\partial_tH=-\nabla\cdot(\mathbf vH)
\tag{34}
$$

的守恒有限体积子步：面速度、MC 限制的分片线性重构、迎风通量和 SSPRK2。一个完整时间步为

$$
\mathcal A_{\delta t/2}\ \mathcal D_{\delta t}\ \mathcal A_{\delta t/2},
\tag{35}
$$

其中 $\mathcal A$ 为焓输运，$\mathcal D$ 为 D2Q5 导热。每次输运后保持四个移动分布不变，只调整静止分布的储能并重新反演温度。

选择这一实现的理由是能明确检查全域能量守恒与潜热输运。它的代价是对流需要面通量和邻居数据，不能声称保留了原文所有对流源项的局部计算优势。对于不连续的冰水边界，限制器还会引入几何扩散；验证报告单独列出形状误差，不用“体积守恒”冒充“刚体形状完全不变”。

在一般两速度 / 三物态场景中，式（34）的通量必须改为式（4）或式（52）的相质量通量构造，不能继续无条件用单个 $\mathbf vH$。

### 5.5 边界与浮点设置

边界均放在格点外半个格距。周期边界直接迁移。对绝热边界：

$$
g_{\rm in}=g_{\rm out}.
\tag{36}
$$

对给定温度 $\Theta_b$ 的半格点边界：

$$
g_{\rm in}=2w_i\Theta_b-g_{\rm out}.
\tag{37}
$$

只有移动分布参与式（37），因此不能把 $\Theta_b$ 替换成含潜热的 $H_b$。离散边界输入热量为每条边界链的 $g_{\rm in}-g_{\rm out}$；这与体积能量变化独立累计后比较。

GPU 上分布、温度、液相率、热容、热导率、补偿量、面速度均为 Taichi f32。CPU 上解析解和汇总诊断使用 NumPy float64，不是把 GPU 状态改为 f64。Taichi 初始化必须设置 fast_math=False，以防编译器重排补偿求和表达式。初次均匀加热检查发现快速数学优化下约 $3.3\times10^{-5}$ 的累计温度误差；关闭该优化后降至约 $1.2\times10^{-7}$，无须使用 GPU f64。

测试采用 $\varpi=0.4<\min(C_s,C_l)$ 并确保 $K>0,\ \tau_q>1/2$。正的储能系数和正导热率是合理的参数约束，但不是对所有 MRT 参数、所有非线性界面的普遍稳定性证明。

## 6. 正确性验证及其独立依据

具体结果、误差定义、运行环境和图见 validation_zh.md；原始结果在 results 中。验收同时覆盖方程、解析解、能量余额、网格加密和运动输运，不以单张融化图片作为正确性证据。

### 6.1 均匀加热 / 冷却

周期域内无梯度，给定恒定体积热源 $Q_0$，解析解为

$$
H(t)=H(0)+Q_0t.
\tag{38}
$$

代入式（22）即得到先升温、在熔点吸收潜热、完全融化后再升温的分段解。随后施加反向热源，检查能否回到初始焓。此测试直接检查潜热符号、平台宽度、比热分支与储能实现。

### 6.2 二维光滑扩散和带平移的扩散

取常数 $C,K$，周期方形域边长为 1：

$$
\Theta(x,y,t)=A\sin[2\pi(x-Ut)]
\cos[2\pi(y-Vt)]\exp(-8\pi^2\alpha t),\qquad
\alpha=K/C.
\tag{39}
$$

$U=V=0$ 验证二维导热和扩散系数；非零速度验证对流及分裂误差。网格加密时保持物理 Péclet 数不变，并按 $\delta t\propto\delta x^2$ 缩放。报告中的约二阶是这个加密路径的总误差阶，不是独立证明任意固定网格上的二阶时间精度。

### 6.3 变热容、变导热率制造解

取
$$
C=0.8+0.2\sin(2\pi x)\cos(2\pi y),\quad
k=1+0.3\cos(2\pi x),
\quad
\Theta=0.2e^{-t}\sin(2\pi x)\sin(2\pi y).
$$

按
$$
Q=C\partial_t\Theta-\nabla\cdot(k\nabla\Theta)
\tag{40}
$$
逐项解析计算源。代码中的源系数为
$$
Q=e^{-t}\left\{
[-C+8\pi^2k]\,0.2\sin(2\pi x)\sin(2\pi y)
+0.24\pi^2\sin(2\pi x)\cos(2\pi x)\sin(2\pi y)
\right\}.
$$

该测试能区分式（31）与错误的 $\nabla\cdot[(k/C)\nabla\Theta]$，而不只检查常物性问题。

### 6.4 两层材料的稳态热阻

长度 $a,b$、导热率 $k_1,k_2$ 的两层材料满足
$$
q=\frac{\Theta_L-\Theta_R}{a/k_1+b/k_2}.
\tag{41}
$$

温度在两层分别线性变化，界面温度连续，热流相同；改变热容不改变该稳态解。本测试以解析稳态初始化，经过长时间迭代检查该解是否被保持，并拟合两侧斜率核验热流。它检验的是稳态保持与通量匹配；瞬态变物性正确性由式（40）另外检验。

### 6.5 Stefan 解析解

左侧热壁为 $T_h>T_m$，右侧初始冰温为 $T_i\le T_m$，液区 $0<x<s(t)$，固区 $x>s(t)$。设
$$
\alpha_l=k_l/C_l,\quad \alpha_s=k_s/C_s,\quad
s(t)=2\lambda\sqrt{\alpha_l t},\quad
\eta=\lambda\sqrt{\alpha_l/\alpha_s}.
\tag{42}
$$

将扩散方程的相似变量 $x/(2\sqrt{\alpha t})$ 代入，解常微分方程并满足边界温度，得
$$
T_l=T_h-(T_h-T_m)\frac{\operatorname{erf}[x/(2\sqrt{\alpha_l t})]}
{\operatorname{erf}(\lambda)},
\tag{43}
$$
$$
T_s=T_i+(T_m-T_i)\frac{\operatorname{erfc}[x/(2\sqrt{\alpha_s t})]}
{\operatorname{erfc}(\eta)}.
\tag{44}
$$

将界面两侧梯度代入式（11）：
$$
\frac{k_l(T_h-T_m)e^{-\lambda^2}}
{\sqrt{\pi\alpha_l}\operatorname{erf}\lambda}
-
\frac{k_s(T_m-T_i)e^{-\eta^2}}
{\sqrt{\pi\alpha_s}\operatorname{erfc}\eta}
=\mathcal L\lambda\sqrt{\alpha_l}.
\tag{45}
$$

代码用自行编写的二分法求根。若 $T_i=T_m$，第二项为零，退化为一相 Stefan 问题：

$$
\sqrt{\pi}\lambda e^{\lambda^2}\operatorname{erf}\lambda
=\mathrm{Ste},\qquad
\mathrm{Ste}=\frac{C_l(T_h-T_m)}{\mathcal L}.
\tag{46}
$$

本次 $C_s=0.5,C_l=1,k_s=2,k_l=1,\mathcal L=2,T_m=0,T_h=1$，分别取 $T_i=0$ 和 $-0.25$。在非零时间 $t_0=0.002$ 用解析解初始化，以免把 $t=0$ 的奇异热流误差混入主要比较。含界面的单元分别积分固液显热和潜热，而不是按单元中心粗暴判相。

有限计算域右端施加解析解在同一位置和时间的温度，不把半无限域的远场温度错误当成有限域精确边界。界面位置采用
$$
s_{\rm num}=\Delta x\sum_x\overline f(x)
\tag{47}
$$
计算液体体积对应的熔化长度，避免把任意 $f=0.5$ 等值线定位误差混入体积指标。另比较温度、随时间的前沿和解析总能量增量。

该离散界面没有光滑解的二阶精度保证。前沿通常接近一阶，且单元内相界面位置会使某些单时刻温度误差非单调；报告保留这些结果。

### 6.6 冰块运动与浸没融化

三个性质分别核验：

1. **仅平移 / 转动，无热驱动：** 全域 $T=T_m$，冰中 $H=0$，水中 $H=\ell$。解析解为刚体变换后的同一相态，冰量和温度不变。检查一个周期平移和 $90^\circ$ 转动后的体积、质心、几何转动惯量和形状误差。
2. **闭合绝热域的冷冰融化：** $80^2$ 网格中 $40^2$ 冰块初温 $-0.5$，水初温 1，$C_s=0.5,C_l=1,\ell=5$。初始总焓为 28400；平衡仍有冰、最终 $T=0$，因此理论剩余冰面积为 $6400-28400/5=720$。
3. **平移中融化：** 周期域中冰与周围水共同平移，同时导热融化。连续方程预言结果只是静止融化解的平移。把两次计算在同一时刻对齐，检查差异随网格加密减小，同时检查能量。

第三项的对照是同一已验证导热离散的静止解，属于坐标变换一致性验证，不应称为另一个独立 Stefan 解析解。它也没有相对冰水流动、尾涡或受力反馈。

## 7. 自由漂浮的力学如何与热模型衔接

D2Q5 只负责能量方程，不能单独生成压力、黏性力、漂移速度或角速度。原文也另用了 D2Q9 流体求解器。

对二维刚体冰域 $\Omega_s(t)$，材料速度可以表示为
$$
\mathbf v_s(\mathbf x,t)=\mathbf V(t)+\omega(t)\mathbf e_z\times
[\mathbf x-\mathbf R(t)].
\tag{48}
$$

熔化使质量、质心和惯量变化：
$$
M_s=\int_{\Omega_s}\rho_s\,dV,\qquad
\mathbf R=\frac{1}{M_s}\int_{\Omega_s}\rho_s\mathbf x\,dV,\qquad
I_s=\int_{\Omega_s}\rho_s|\mathbf x-\mathbf R|^2\,dV.
$$

要从体积分数积分这些量，至少应保持输运的质量一致性；仅用像素是否超过 0.5 判断冰体，容易在融化和转动中引入跳变。

若界面脱离冰体的质量按冰侧速度携带动量，使用与界面动量跳跃相容的净牵引力 $\mathbf F_\Gamma$，固体控制体满足
$$
\dot M_s=-\int_\Gamma j\,dA,\qquad
\dot{\mathbf P}_s=\mathbf F_\Gamma+M_s\mathbf g
-\int_\Gamma j\mathbf v_s\,dA.
\tag{49}
$$

关于固定惯性原点 $O$ 的角动量满足
$$
\dot{\mathbf L}_{s,O}=\boldsymbol\tau_{\Gamma,O}
 +(\mathbf R-O)\times M_s\mathbf g
-\int_\Gamma j(\mathbf x-O)\times\mathbf v_s\,dA.
\tag{50}
$$

这里的牵引力必须包括所采用界面模型要求的应力 / 表面力；若直接使用水侧应力且冰水法向速度不同，还必须按界面动量跳跃处理质量转移的动量通量，不能默认它等于冰侧牵引力。式（49）也不能简化成“更新变小后的质量，同时还照旧积累全部旧动量”。

一个容易忽略的几何问题是：令 $\mathbf V=\mathbf P_s/M_s$，则非对称融化时
$$
\boxed{\dot{\mathbf R}
=\mathbf V-\frac1{M_s}\int_\Gamma j(\mathbf x-\mathbf R)\,dA.}
\tag{51}
$$

剩余冰体的质心运动不完全等于刚体材料平移速度。可用固定材料参考点来参数化刚体，或显式处理式（51）；不能同时把材料参考点、剩余质量质心和固定初始几何中心混为一谈。

流体侧还需要质量连续方程、动量方程、相变质量通量对应的法向速度条件、黏性 / 无滑移条件，以及与冰体等大反向的作用力。若加入浮力，真实水的密度–温度关系、冰水密度差、实际长度尺度和重力必须作为输入指定；本次无量纲热验证并没有提供这些物性数据或上浮动力学校准。

## 8. 后续接入水–空气自由液面

用户后续的“气液两相流里融冰”从热学和材料守恒上包含三个物态。定义
$$
\chi_s+\chi_l+\chi_g=1,
$$
分别表示冰、液水、空气体积分数。需要至少两个独立的相态变量；不能用一个液相率 $f$ 同时表示“水还是空气”和“冰还是水”。

一种方便的变量选择是总凝聚水占比 $\alpha=\chi_s+\chi_l$，以及其中液态份额 $f=\chi_l/\alpha$。在 $\alpha>0$ 处，
$\chi_l=\alpha f,\ \chi_s=\alpha(1-f),\ \chi_g=1-\alpha$。
但 $\alpha$ 的更新也必须满足真实密度和相变体积变化，不能只任意 advect 一个颜色场。

能量与通量应扩展为
$$
\mathcal H=\chi_s\rho_sh_s+\chi_l\rho_lh_l+\chi_g\rho_gh_g,
$$
$$
\boxed{\mathbf J_H
=\chi_s\rho_sh_s\mathbf v_s+
\chi_l\rho_lh_l\mathbf u_l+
\chi_g\rho_gh_g\mathbf u_g.}
\tag{52}
$$

各相的质量通量已经由界面 / 流体算法决定后，温度输运必须使用相同的面质量通量：
$$
J_{H,\rm face}=F_{m,s}h_{s,\rm face}
+F_{m,l}h_{l,\rm face}+F_{m,g}h_{g,\rm face}.
\tag{53}
$$

这可以避免气液密度差下“质量走了一条通量，显热和潜热却走另一条通量”的不一致。D2Q5 的导热矩条件不变，但其宏观状态反演和对流通量必须升级为这个三物态闭合。

需要重新验证的内容包括：

- 冰水密度差对应的式（12）及自由液面体积变化；融化水不能凭空创建 / 丢失质量。
- 水–空气界面的表面张力、气液压力、冰体受力和接触线条件。若冰体部分出水，浮力和水线不能从当前给定速度热测试推断。
- 空气的低体积热容和跨界面导热。当前 $\varpi=0.4$ 是冰水测试参数，不能直接用于任意气液热容比；应重新选择无量纲尺度、权重、时间步或热子循环，并做高物性比的解析 / 守恒测试。
- 若忽略空气传热，需要明确以热边界近似；若包含空气传热，则其比热、导热率和界面温度 / 热流条件要明确给出。
- 如果未来还模拟水蒸发，熔化潜热与汽化热力学是两套相变机制。不能不加区分地叠加原文 EOS 相变项和焓中的潜热，造成重复计热。

因此，现有 D2Q5 热核有明确的接入路径，但本次验证结论不延伸为“水–空气–冰全耦合模拟已经正确”。

## 9. 实现建议

先保留本目录作为独立的方程验证基线。接入后续流体代码时，应复用同一套单元能量、面质量 / 焓通量和冰体质量统计，并逐级通过：

1. 受力驱动的浸没刚体平移 / 转动与流固动量交换验证；
2. 受力运动与融化同时存在的质量、能量、动量收支；
3. 最后加入水–空气自由液面、高物性比和部分露出水面的冰块。

网格加密、通量余额和解析基准仍应保留；真实上浮速度、融化时间和最终姿态的定量可信度需要这些新耦合的独立证据。
