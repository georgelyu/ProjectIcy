# 融水转移方法调研

检索日期：2026-09-17。本次调研的目标是把固体损失的质量、显热/潜热和动量，以相容的离散通量交给流体，并处理移动边界的几何体积变化。它不是寻找一个可以原样粘贴进现有 LBM 的“注水函数”。

## 1. 已核对的主要研究

### Huang、Lin、Ardekani，JCP 2022

*A consistent and conservative Phase-Field model for thermo-gas-liquid-solid flows including liquid-solid phase change*，JCP 449, 110795。

- [DOI](https://doi.org/10.1016/j.jcp.2021.110795)
- [作者开放全文](https://arxiv.org/html/2102.06863v3)
- 本地存档：huang_2022.html、huang_2022.txt。

已读全文中第 2.1.4–2.1.6、3、4.1 节。与本任务最相关的结论：

1. 气、水、冰必须有明确且互相相容的体积分数，不能让两套模型分别决定“有多少液体”。
2. 密度不同的固液相变必须伴随体积变化；给整个三物态系统强制散度为零通常会破坏质量守恒。
3. 动量通量必须使用与质量方程相同的质量通量，能量输运也必须包含相场扩散/压缩所对应的物性输运。
4. 等温平衡与 Galilean 不变性是必要的验证项目。

原文式（16）明确将热容通量与相变体积通量纳入温度方程。其固体边界采用惩罚方式，默认固体速度为零；因此不能用该论文宣称“SDF + 自由刚体 + single-node 反弹”已经得到验证。

也应区分连续守恒与实际离散误差：原文第 3 节讨论了离散质量残差对动量的一致性修正，第 4.1.4 节的相变材料质量误差小于约 0.004%。本项目仍须检查自己的逐面离散余额。

**采用：** 三物态守恒关系、质量–动量–能量通量一致性和验证原则。

**不采用：** 替换指定气液碰撞核、Brinkman 固体惩罚和原文的固液相场演化方程。

### Weber、Mukherjee、Class、Brandt，JCP 2026

*A consistent and scalable framework suitable for boiling flows using the conservative diffuse interface method*，JCP 551, 114680。

- [DOI](https://doi.org/10.1016/j.jcp.2026.114680)
- [作者机构公开记录与全文](https://publikationen.bibliothek.kit.edu/1000190025)
- 本地全文：weber_2026.pdf、weber_2026.txt。

已核对第 2 节的质量、动量、能量方程，以及第 4.4 节的能量检验。

原文式（8）表明，相场正则化通量会进入实际质量通量；式（15）–（16）进一步讨论其动量贡献。遗漏这些项可能错误预测相变引起的压力跳跃。论文也强调局部相变时间尺度、对流、黏性、导热、表面张力和重力约束。

**一个直接影响选型的限制：** 原文第 2.3 节说明，它为了避免物性与温度采用不同离散产生的输运不一致，选择了非守恒温度方程；第 4.4 节报告最大约 **0.4%** 的总焓误差。因此这篇最新方法不能作为“其温度实现天然严格守恒”的依据。

**采用：** 相场通量对动量和能量必须一致，以及相变压力/速度跳跃必须单独验证的原则。

**不采用：** 其非守恒温度实现、汽化动力学闭合和 FFT 压力求解器。冰的相变率仍由 D2Q5 导热与焓/Stefan 约束给出。

### Berger、Giuliani，SISC 2024

*A New Provably Stable Weighted State Redistribution Algorithm*，SIAM Journal on Scientific Computing 46(5), A2848–A2873。

- [出版方原始摘要](https://doi.org/10.1137/23M1597484)
- 方法前身：*A state redistribution algorithm for finite volume schemes on cut cell meshes*，JCP 428 (2021), 109820。

已核对出版方摘要、发表信息和相关方法说明。加权状态重分配用于处理切割单元过小造成的时间步限制，原文在其分析覆盖的情况下证明单调性、TVD 和 GKS 稳定性，并解释了预合并的作用。

**采用方向：** 为新守恒几何更新设计局部、体积加权、共用权重的重分配，防止极小切割体积导致过大温度/速度。

**适用边界：** 这不是 LBM 稳定性证明。不能把对特定有限体积方程的结论直接外推到本项目的整个碰撞、反弹和相变耦合系统。实际使用前需针对所用守恒量与 cut-cell 更新完成离散推导和单独基准。

## 2. 近期检索到的其他研究及证据等级

| 研究 | 核对程度 | 在本方案中的用途 |
|---|---|---|
| Poblador-Ibanez 等，JCP 2025，*A momentum balance correction to the non-conservative one-fluid formulation in boiling flows using volume-of-fluid*，[DOI](https://doi.org/10.1016/j.jcp.2024.113704) | Crossref 元数据；Weber 2026 的讨论和引用 | 补充说明相变动量/压力问题仍是当前研究重点；未据此抄写实现公式 |
| Dupuy 等，Computers & Fluids 2025，*Time integration of cut-cell surfaces for geometric conservation, applied to scalar transport with moving interfaces*，[DOI](https://doi.org/10.1016/j.compfluid.2025.106627) | 出版元数据和原始标题已核实，未获得并审读全文 | 几何时间积分方向的后续参考；本方案 GCL 等式直接由体积守恒推导 |
| Song、Tian、Pan，2026-09-03 预印本，*A conservative coupling method of sharp-interface and multi-species model for compressible reacting gas-liquid flows with phase change*，[原文](https://arxiv.org/html/2609.03509v1) | 已读摘要和界面耦合相关论述 | 说明最新锐界面工作继续采用界面通量交换；压缩波、化学反应和汽化模型不适合直接移植到冰水问题 |
| Zhan 等，JCP 2025，*A thermodynamically consistent and conservative diffuse-interface model for gas/liquid-liquid-solid flows*，[DOI](https://doi.org/10.1016/j.jcp.2025.113949) | 仅核实元数据 | 不以标题代替对方法的审读，不作为本次实现公式依据 |

上述列表刻意区分了全文、摘要和元数据。没有把未读全文的研究当成已验证的算法依据。

## 3. 选型结论

推荐采用：**界面通量决定融化质量，几何守恒决定体积变化，同一组相质量通量输运动量和焓。**

核心转移单位是同一次界面交换产生的：

$$
(\Delta m,\ \Delta\mathbf P,\ \Delta H,\ \Delta V_s,\ \Delta V_l).
$$

这些量不能由各模块各自估算；也不能仅在“一个固体格点变成流体格点”时补水。对于本场景：

$$
\Delta V_s=\frac{\Delta m}{917},\qquad
\Delta V_l=\frac{\Delta m}{1000}=0.917\,\Delta V_s.
$$

用户已明确选择保留固定密度数值空气，只要求冰＋水质量严格守恒。因而本方案采用上述研究的界面通量和一致性原则，但不宣称整个封闭气液固系统质量守恒。空气体积变化产生的模型质量误差及数值源携带的焓另外记录；不能混入融水或用修改水量抵消。

必须同时处理：

- 刚体移动扫过网格，但没有融化的几何重映射；
- 真正的固液转移及其显热、潜热；
- 不同密度造成的体积收缩与压力/速度响应；
- 相场扩散、压缩和迁移的实际离散通量；
- 新出现、被覆盖及极小流体体积中的守恒状态；
- 融化后刚体质量、质心、惯量和动量的变化。

这是一套基于近期研究原则、针对指定 D2Q9 与反弹内核重新推导的组合方案。**尚未实现和验证的新组合不能预先宣称达到某种全局最优的 state of the art，也不能承诺任意参数下无条件稳定。** 应以逐步守恒余额、解析跳跃条件、加密收敛和长时间稳定试验验收。
