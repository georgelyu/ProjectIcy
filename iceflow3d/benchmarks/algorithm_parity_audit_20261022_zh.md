**IceFlow3D 与 iceflow2d_v3 算法一致性审计 — 2026-10-22**

结论：当前 `iceflow3d` 不满足“仅扩展维度，完整保留 iceflow2d_v3 的算法”的要求。存在可选算法缺失、主路径离散/边界/耦合算法修改，以及默认物理参数变化。部分修改是有意的稳定性或精度改进；本报告判断的是是否保留二维基线，不据此断言哪版物理上更正确。

基线是本次工作区的实际源码，包含尚未纳入 Git 的三维目录，不用旧报告或 README 替代实现。检查覆盖配置、格子平衡态、相场/动量碰撞与恢复、梯度和通量、固体边界、刚体/接触、移动补点、热输运及融化。输出、渲染和检查点的新增能力不作为算法缺失。没有修改两版求解器。

复现文件：[审计脚本](/home/georgelyu/Repos/ProjectIcy/iceflow3d/benchmarks/audit_2d_parity.py)，[CUDA 数值结果及核心源码 SHA-256](/home/georgelyu/Repos/ProjectIcy/iceflow3d/benchmarks/algorithm_parity_20261022.json)。脚本使用新的小网格实例，对相应算子分别输入受控状态；不读取或改写用户仿真输出。可平面退化的测试使用不随 z 变化的输入，并在远离 z 墙的格点比较；这不是把有限深度、有前后墙的完整三维水槽当成二维水槽。

**已经确认的主路径差异**

1. **混合黏度的本构公式被替换。** 二维使用 `nu = phi*nu_water + (1-phi)*nu_air`；三维先线性混合动力黏度，再除以混合密度。纯水/纯空气的关联式一致，不代表混合区一致。90°C 水、20°C 空气、`phi=0.5` 时，实际设备函数给出二维 `7.410313e-6 m²/s`、三维 `3.318915e-7 m²/s`，相差 **22.3275 倍**。此外，二维没有已输运水温的弥散尾部沿用初始水温计算水黏度，三维直接调用干格点返回空气初温的 `_water_temperature`。因此差异不限于插值表达式。

   代码：[二维黏度](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2530)，[三维黏度](/home/georgelyu/Repos/ProjectIcy/iceflow3d/lattice.py:50)，[三维调用](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:617)，[干格点温度](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:72)。

2. **动量平衡态从二阶多项式改为张量积形式，平面退化也不相同。** 二维速度项是 `3(c·u)+4.5(c·u)²-1.5u²`；三维是三个一维因子的乘积，包含额外混合高阶速度项。对 `u=(0.08,0.06,0)`、`rho=0.7`、`p=0.012`，把三维平衡分布沿 `c_z` 求和后，与二维分布的最大差为 `1.424630e-4`；`M21` 分别为 `0.0140000` 和 `0.0142688`。低阶质量、动量和应力矩相同，不能推出整个碰撞算法相同。

   代码：[二维平衡态](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/lattice.py:76)，[三维平衡态](/home/georgelyu/Repos/ProjectIcy/iceflow3d/lattice.py:64)。

3. **CM-MRT/CM-HRR 的闭合及时间层有重新设计。** 三维四阶中心矩先分离二阶偏应力成分，压力相关偶数高阶模态使用声学松弛率；两种碰撞模式均通过独立的原始压力模态补偿，再作强迫一阶矩的浮点闭合。二维只在 HRR 分支显式闭合独立原始压力模态，CM-MRT 没有同样的后处理。三维 HRR 还新增默认启用的三次速度缺陷修正。

   另外，二维在本次动量碰撞中用新算出的加速度构造 `j/rho + a_new/2`；三维使用上次恢复后保存的 `velocity_lattice`，本次 `_forces` 后没有等价更新这一步半力速度。力随相场或温度变化时，平衡态和中心矩参考速度的时间层不同。

   三维新增自由度及新矩本身是扩维所需；其具体闭合选择需要明确的二维退化验证，不能仅凭模型名相同认定一致。上条已经给出了平衡态不一致的直接反例。代码：[二维碰撞与速度](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2606)，[二维 HRR 压力闭合](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2822)，[三维高阶闭合](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:677)，[三维压力/动量补偿](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:737)。

4. **密度源项不再使用相场和毛细力共用的梯度。** 二维 `_phase_central_gradient` 返回同一个连通最小二乘梯度，供源项与压力恢复使用。三维对相场/毛细力保留最小二乘梯度，却在 `_limited_gradient` 中另算缺少固体方向贡献的 `3*sum(w*c*delta_phi)`。即使限幅参数为零，这个差异也存在。底墙第一排、法向斜率 0.01 的线性场，二维源梯度为 0.01，三维相场梯度也为 0.01，而三维源梯度为 **0.005**。

   代码：[二维统一梯度](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2425)，[二维力/梯度](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2546)，[三维独立源梯度](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:442)，[三维压力恢复](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:972)。

5. **一致质量通量对应的动量通量由中心格式改为迎风格式。** 两版都可从相场碰撞测量非对流质量通量 `J`。二维 `consistent_flux` 的面速度是两侧速度的算术平均；三维按面质量通量方向选供体速度。三维因此增加了速度跳变的数值耗散，并且不提供二维 `legacy` 耦合选择。受控局部场中，二维 x 力为 `-1.000000e-4`，三维为 `-8.000001e-5` LU；两版的源项限幅均关闭。

   代码：[二维中心通量](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/stability_options.py:185)，[三维迎风通量](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:582)。三维已有测试也显式检验这种新增耗散：[通量能量测试](/home/georgelyu/Repos/ProjectIcy/iceflow3d/tests/test_phase_flux_energy.py:9)。

6. **冰面的中性润湿毛细应力镜像没有移植。** 二维把采样点反射到流体侧，并使用 `Kghost=R*Kmirror*Rᵀ`；冰体反力矩还有与镜像应力相配套的修正。三维仅对容器墙反射张量，冰邻点直接返回本格应力。对于法向沿 x 的平冰面、给定 `Kxy=1`，实际二维镜像值为 **-1**，三维为 **+1**。这改变了冰面接触线附近的力与力矩闭合。

   代码：[二维冰面应力](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2438)，[二维配套力矩](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3870)，[三维邻点应力](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:520)，[三维冰体受力](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:885)。

7. **HRR 冰面速度虚拟点的重建公式不同。** 二维用 `u_i+(u_b-u_i)/eta`；三维若反方向连通，用 `(2*u_b+(eta-1)*u_back)/(1+eta)`，否则用 `2*u_b-u_i`。两者可对特定线性场一致，但对一般场不同。半格静止冰面、`u_i=0.04`、`u_back=0.1` 时，二维虚拟速度为 -0.04，三维为 -0.0333333。三维的有界系数是额外算法选择。

   代码：[二维 HRR 虚拟点](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/stability_options.py:124)，[三维 HRR 虚拟点](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:779)。

8. **相场越界后的分布修正不同。** 二维把 `h_eq(phi_bounded,u)-h_eq(phi_old,u)` 加到所有方向；三维把差值全部加到静止分布。前者同步改变相场一阶及更高矩，后者保留它们。`phi=1.1`、`u=(0.07,-0.03)` 时，两版都裁剪到 1，但相场一阶矩分别约为 `(0.07,-0.03)` 与 `(0.077,-0.033)`。因此后续输运和所测 `J` 都可能不同。

   代码：[二维裁剪](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3572)，[三维裁剪](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:953)，[三维独立恢复路径](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:989)。

9. **连通供体判定省去了二维的对角几何可见性检查。** 二维在两条轴向路径都被挡住时，还检查实际材料插值覆盖是否阻断对角线；三维只接受 2×2×2 邻域内的面连通路径。二维 `_phase_visible_diagonal` 的对应三维几何判定不存在。这改变了狭缝和角点处的梯度供体、补点传播和待补空穴判定。

   代码：[二维实际覆盖判定](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/phase_reconstruction.py:76)，[二维使用位置](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/phase_reconstruction.py:160)，[三维面连通判定](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:183)。注意：二维当前主梯度内核本身共用最小二乘实现，不能把 `legacy` 配置名误读成另一个仍在执行的主梯度公式。

10. **移动补点的平均权重、可选分布和半步修正都变了。** 二维连通补点对有效供体等权平均；三维使用格子权重。八个平面供体中只有一个轴向供体的相场为 1、其余为 0，且沿 z 延拓时，二维得到 **0.125**，三维得到 **0.200000**。二维还提供 `non_equilibrium` 的密度归一化应力外推，三维只有平衡态补点。

   二维可选 `_project_refill_moments` 是对已有分布作规定的一阶/压力矩提升；三维总是重建平衡态，并减去完整半步方向源项。即使目标宏观压力/速度一致，分布和非平衡应力也不必一致。三维不能通过一个开关恢复二维这两类补点选择。

   代码：[二维传播](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/phase_reconstruction.py:269)，[二维非平衡补点](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/phase_reconstruction.py:352)，[二维矩提升](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/phase_reconstruction.py:389)，[三维传播](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1096)，[三维重建](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1133)，[三维半步源项](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:353)。

11. **刚体积分及流固耦合顺序改变。** 二维 `step` 先预测并约束位移，以实际接受的位移/转角提供本步壁面速度；流体迁移后提交姿态，再用本步水动力冲量更新刚体速度。三维先让流体使用旧刚体速度完成边界交换，再更新速度、移动姿态、处理接触。它不仅是从一个角度换成四元数。

   从静止开始施加相同的 `delta_v_x=0.02` 冲量、关闭重力且没有接触时，二维本步位移为 **0**，三维为 **0.02 格**，末速度相同。接触存在时，三维事后的接触约束也没有反馈给本步已经执行的流体壁面速度。

   代码：[二维 step](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:1559)，[二维实际壁速](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/body_motion.py:518)，[二维提交/加冲量](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/body_motion.py:550)，[三维 step](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1330)，[三维积分](/home/georgelyu/Repos/ProjectIcy/iceflow3d/rigid.py:172)。

12. **融化后的接触形状、残体运动及接触回退策略改变。** 二维材料包围盒追踪 `solid_liquid_threshold` 等值线，连续接触预测之后还有实际栅格重叠检查、缩小本步位移和保持旧姿态的回退。三维用所有正质量体素的完整外包盒做 SAT，增加了墙面接触集合冲量求解，但没有二维的栅格复核/位移回退链。

   均匀融化到 75% 的 8×8 横截面，二维接触边界是 `±3.833333`，三维仍为 `±4`。融化到 25% 时，二维没有阈值等值线，三维仍保留完整体素包围盒。二维没有已解析固体边界的残体会保持姿态并继续换热；三维只要质量非零仍会进行刚体积分和体素包围盒接触。提前接触、残体运动及接触失败时的行为都不同。三维还在刚体平动与转动估计行程超过半格时直接报错，二维主步采用的是上述预测和回退流程。

   代码：[二维阈值边界](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/body_motion.py:54)，[二维接触复核/回退](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/body_motion.py:575)，[二维残体规则](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3365)，[三维完整体素边界](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1273)，[三维接触](/home/georgelyu/Repos/ProjectIcy/iceflow3d/rigid.py:206)。

13. **部分融化材料的格内惯量假设改变。** 二维单材料格的惯量项为 `mass*(r²+fraction/6)`；三维张量的格内项固定为 `mass/6`。两版在完整材料格上符合相应维数的惯量，但融化后采用了不同的格内形状假设。上述 75% 材料的 `Izz/mass` 分别约为 10.625 与 10.666667 格²；25% 时为 10.541667 与 10.666667。扩维需要明确如何推广格内收缩模型，不能只把固定 `1/6` 当成原公式的张量版。

   代码：[二维惯量归约](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2233)，[三维惯量归约](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1279)。

14. **新湿格点的温度重建从逐圈搜索改成一层搜索。** 二维扩展搜索到最近有水的整圈，以 `1/distance²` 加权水能量和体积；三维只查询 3×3×3 邻域，直接按体积平均，没有供体就给零显热，即熔点温度。整层材料移除或较大新暴露区域会触发这个差异。在熔点为 0°C、最近 80°C 热水供体相距两格的控制案例中，二维重建为 **80°C**，三维为 **0°C**。

   代码：[二维温度外推](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:503)，[三维温度外推](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:99)。

15. **一个世界采样点与多个冰体换热的能力被缩减。** 二维保留每个冰体的覆盖率，热交换遍历全部冰体。三维栅格化只保存最大覆盖率及其唯一 `coverage_body_id`，热交换每个采样点只处理这个冰体。不同冰体的插值支撑域可以在窄水隙相交而不发生固体穿透，此时二维可同时向两块冰分配热量，三维会省略非最大覆盖冰体的这部分热交换。守恒地把已算热量一正一负分配，并不能证明没有漏掉另一块冰的接触。

   代码：[二维多冰覆盖](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2012)，[二维遍历各冰体换热](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:954)，[三维只选最大覆盖](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:263)，[三维唯一换热对象](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:329)。此项由执行路径直接确认，未单独做双冰体端到端定量测试。

16. **相场初始化和预热收尾不同。** 二维从 0/1 指示函数预热，构造静水平衡时把相场分布重新设为 `h_eq`；三维直接从 tanh 弥散界面开始，静水平衡初始化保留预热后的 `h`。预热为零的相同平面水面两侧，二维相场为 1/0，三维为约 0.598688/0.401312。即使预热使宏观相场接近，保留还是清除非平衡相场分布也改变第一步的相场通量。

   代码：[二维初始相场](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2061)，[二维重设 h](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2187)，[三维 tanh 初态](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:281)，[三维保留 h](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:341)。

17. **移动冰面相场反射的位置和预热运动处理不同。** 二维正式流体步的相场反射把 `fraction` 固定为 0.5；三维使用覆盖率计算的 cut-link `eta`。冰体有角速度时，同一链上不同采样点的壁速不同，因此反射量会改变。二维预热 `_stream_phase_only` 还包含移动冰面的速度修正；三维预热直接 `_stream(True)`，未调用 `_prepare_boundaries`，没有对应修正。初始静止冰体不会暴露后一个差异，但非零初速/角速度配置会。

   代码：[二维正式步反射](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3038)，[二维预热反射](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:3075)，[三维预热顺序](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:140)，[三维正式步反射](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:868)。

**缺失或不能按二维方式选择的能力**

| 二维当前支持 | 三维当前情况 | 对照位置 |
| --- | --- | --- |
| `rm_mrt` 动量碰撞 | 配置直接拒绝，只允许 `cm_mrt` / `cm_hrr` | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/config.py:501)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/config.py:296) |
| `enthalpy_d2q5` 格子热传导，与 FV 二选一 | 只有 FV，没有对应三维格子热传导分支 | [二维调用](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:425)、[二维实现](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/thermal_d2q5.py:1)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:234) |
| 各墙绝热/Dirichlet 温度边界 | 只实现绝热墙 | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/config.py:86)、[三维通量](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:134) |
| `phase_momentum_coupling=legacy/consistent_flux` | 始终执行修改后的 consistent flux | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2594)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1337) |
| 平衡态/非平衡应力补点、可选 `refill_moment_projection` | 固定执行另一套平衡重建与半步源项修正 | 见第 10 项 |
| legacy 扩圈补点与 connected 分层补点 | 只保留改过权重/连通性的分层补点 | [二维切换](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:1602)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1221) |
| 指定常数水/气运动黏度 | 只能使用关联式 | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/config.py:824)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:617) |
| 独立 `buoyancy_reference_temperature_c` | 固定以初始水温为参考 | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2563)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:553) |
| 可配置 `water_phase_cutoff` | 热水体积阈值硬编码为 `1e-3` | [二维](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/config.py:477)、[三维](/home/georgelyu/Repos/ProjectIcy/iceflow3d/thermal.py:65) |

移植 D2Q5 分支需要设计对应的三维格子，不能把二维五方向原样用于三维；但直接删除这一可选方法同样不符合“所有算法保留”的要求。

**默认配置也未对齐**

以下比较的是两个“四冰块溃坝”入口实际构造的配置，避免把库默认值与示例默认值混用。脚本已调用两版配置工厂验证。

| 参数 | iceflow2d_v3 四冰块示例 | iceflow3d 四冰块示例 |
| --- | --- | --- |
| 动量碰撞 | `cm_mrt` | `cm_mrt` |
| 表面张力 N/m | **0** | **0.072** |
| 相场法向 epsilon | **0.01** | **1e-8** |
| 声学松弛率 | **0.25** | **0.6** |
| 密度源限幅 | **0.05** | **0** |
| 质量通量加速度限幅 | **0.01** | **0** |
| 气相异常修复 | **开启** | **关闭** |
| 补点矩修正 | 开启二维矩提升 | 固定三维重建，公式不同 |
| 接触迭代上限 | 32 | 128 |
| 最大 Fourier 数 | 0.15 | 0.10 |
| 热子步预算 | 64 | 128 |
| 热更新间隔 / 相场预热 | 8 步 / 2000 步 | 8 步 / 2000 步 |

库级默认值另有差别：二维库 `phase_normal_epsilon=1e-3`、`phase_momentum_coupling=legacy`、热更新间隔为 1；不能用它们代替上表的二维溃坝配置。三维有些参数在 Python 配置类中存在，但示例 CLI 没有对应开关，仍不能只用相同命令行选项复现二维配置。

配置来源：[二维示例覆盖](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/examples/dambreak_four_ice_melting_2d.py:55)、[二维通用参数](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/examples/coupled_falling_ice_melting_2d.py:123)、[三维配置](/home/georgelyu/Repos/ProjectIcy/iceflow3d/config.py:175)、[三维示例工厂](/home/georgelyu/Repos/ProjectIcy/iceflow3d/examples/dambreak_four_ice_melting_3d.py:209)。

**保留的结构与合理扩维内容**

| 内容 | 审计判断 |
| --- | --- |
| D2Q9 → D3Q27、两分量 → 三分量、3×3 → 3×3×3 邻域 | 合理扩维；不能据此豁免前述公式/闭合变化 |
| 密度由相场确定、压力独立、压力分布的零阶平衡矩为零 | 保留基本框架 |
| 压力恢复系数 `3/5` → `9/19` | 来自静止权重变化：`1/[3*(1-w0)]`，属于合理格子差异 |
| 保守 Allen–Cahn 主碰撞、迁移率对应一阶中心矩松弛 | 主要结构保留；相同参数的局部平面碰撞对照最大分布差为 `1.862645e-8`（容差 `2e-7`）；裁剪/初始化/边界差异见上文 |
| 水/气纯相黏度关联式，热容、导热率和潜热 | 公式/常数保留；混合方法不同 |
| 冻结静水参考、动态压力残差和匹配的刚体压力牵引 | 基本结构保留；两版压力存储含义及参考零点不同，不能直接逐值比较总压力与动态压力 |
| Tao cut-link 动量反射、动量交换与反力 | 主公式保留；壁速时间层、平衡态和毛细边界不同 |
| 世界水热场 + 材料冰热场，先升温再消耗潜热 | 保留 |
| 融冰消失、不生成液水、不注入融水动量和显热 | 保留；不能把这个模型选择列为三维漏项 |
| 共享面 FV 热对流/导热、CFL/Fourier 子步 | 保留主要结构并增加 z 方向；温度补点、边界选择和多冰换热不同 |
| 平面转角 → 四元数，惯量标量 → 完整张量，加入陀螺项 | 合理扩维；部分融化格内惯量假设与积分顺序另计 |
| 无供体的新空穴暂封闭、之后重试 | 两版连通补点都保留；供体集合不同 |

特别核查：**当前二维主路径也没有全局水量投影**。三维把自己的历史全局投影注释掉，不应单独认定为“从 iceflow2d_v3 漏移植了全局水量守恒算法”。二维热孔隙重映射也明确不做全局体积或能量补偿。两版都有局部相场零阶矩闭合；这与全局水量严格守恒是不同的性质。[二维说明及执行](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:442)，[三维被禁用的历史投影](/home/georgelyu/Repos/ProjectIcy/iceflow3d/simulator.py:1146)。

另外，稳定性守卫也没有逐项照搬：二维可选检查包含逐方向相场分布的非有限值扫描，三维常规检查主要扫描宏观速度、压力和相场，并在几何更新前执行；三维还有热场检查。两版都具备检查机制，但检查量、时间点和默认设置不同。三维的梯度缓存、融合迁移、局部热交换包围盒和 FP64 归约主要属于实现优化，应通过自身一致性检查验收，不能用它们解释前述本构和边界差异。

**复现与验收含义**

从仓库根目录运行：

```bash
python iceflow3d/benchmarks/audit_2d_parity.py --output /tmp/iceflow_parity.json
ICEFLOW3D_TEST_BACKEND=cuda python -m unittest discover -s iceflow3d/tests -v
```

实际使用已有 Anaconda Python 3.13.9、Taichi 1.7.4、RTX 3090 CUDA；仓库 `.venv` 没有 Taichi，因此没有使用它，也没有安装或改变依赖。

本次已完成 **13 组局部算子/状态对照**，包括一个相场碰撞一致性控制项和多组不等价反例。额外启动的三维全量回归仅完成缓存恢复、覆盖/露出梯度缓存、局部换热三项，均通过；随后在检查点往返测试仍执行期间停止附加运行，**全量回归未完成**，不能记为整套通过。[回归执行记录](/home/georgelyu/Repos/ProjectIcy/iceflow3d/benchmarks/algorithm_parity_regressions_20261022.log)。

局部反例证明不等价，并不量化所有差异在完整溃坝中的累计误差。三维已有单元测试主要验证自身的守恒、稳定性、接触、缓存与检查点契约；其中有些还明确验证新算法，不能用“这些测试通过”替代“与二维算法一致”。本次没有为此再做数秒物理时间的完整溃坝长跑，因为已有局部反例足以否定纯扩维。

达到目标需要按二维当前源码逐项恢复上述公式、边界、时序和选择项，并对必须新增的三维矩/旋转项单独规定扩维规则。应先固定二维库与目标示例的基线配置，再增加共享算子的平面退化对照，以及移动冰面、融化残体、接触和换热的对应检查。仅调整默认参数不能完成这个目标。
