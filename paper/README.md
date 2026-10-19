# IceFlow2D v3 方法论文

英文 ACM TOG 双栏方法稿，正文依据 2026-10-16 当前工作区的 `iceflow2d_v3` 源码编写。
篇幅上限为 **4 页，包含标题、两张方法图和参考文献**。保留类文件的默认字号、页边距和行距。

## 编译

在本目录使用本机 LaTeX：

```bash
cd /home/georgelyu/Repos/ProjectIcy/paper
make
```

`make` 调用 `latexmk`、`pdflatex` 和 BibTeX，生成 `main.pdf`，并通过 `pdfinfo`
检查总页数。中间文件放在 `build/`。可用 `make check` 单独检查页数；
`make clean` 清理 LaTeX 中间文件。直接编辑 `main.tex` 后也可运行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
```

## 文件

- `main.tex`：标题、ACM TOG 格式设置、正文与文献入口。
- `sections/method.tex`：方法部分；源码注释标明对应实现。
- `references.bib`：收录 14 项书目；当前正文引用 13 项，包括 11 篇论文及 2 项物性技术资料。
- `main.pdf`：本机编译的四页以内预览。
- `figures/`：两张矢量方法图（PDF/SVG）、解析例子数据与梯度核验证结果。
- `scripts/build_method_figures.py`：可复现的绘图源文件，使用 NumPy 和 Matplotlib。
- `scripts/verify_gradient_figure.py`：调用当前 Taichi 连通筛选／梯度核核验图中例子。
- `acmart.cls`、`ACM-Reference-Format.bst`：从 `/home/georgelyu/下载/acmart-primary`
  原样复制的模板组件（acmart 2.20，2026-08-16）。
- `vendor/acmart/`：原始类源文件、安装文件和许可证，保留模板来源和分发条件。
- `source_manifest.json`：本次撰写所核对源码的 SHA-256 指纹及配置范围。
- `citation_audit.json`：引用核对记录，列出文献、仓库证据、引用位置及未纳入的理由。

## 引用核对（2026-10-16）

以下记录对应 2026-10-16 的引用核对。2026-10-19 扩写质量—动量推导前，正文已移除
`Coreixas2019` 的引用；本次保留这一状态及其书目条目，当前编译的参考文献为 13 项。

核对范围包括 `iceflow2d_v3` 源码注释、方法文档、`benchmarks/` 与 `research/`，
并追溯 `iceflow2d/docs/main.tex`、v2 热模型推导和调研记录。
判断依据是本文所描述的当前组合；历史调研中出现某篇论文，不等于当前实现采用了它。
原稿的 4 项引用保留，补入 8 篇论文和 2 项技术资料。所有条目均在相应正文中引用，
没有通过 `\nocite{*}` 堆入参考文献。

| 引用 | 正文位置与作用 | 本次处理 |
| --- | --- | --- |
| Geier et al. 2015 | §1.2，守恒 Allen–Cahn 相场 | 保留 |
| Liang et al. 2018 | §1.2，压力—动量分布、源项与宏观恢复 | 保留 |
| Tao et al. 2018 | §1.4，单节点曲边界反弹 | 保留 |
| Wen et al. 2014 | §1.4，相对速度动量交换 | 保留 |
| Sato & Koshimura 2023 | §1.2，抑制体相法向污染的动机；本文使用简单光滑正则化 | 新增 |
| Hajabdollahi et al. 2021 | §1.2，压力型中心矩碰撞的设计依据；未声称复现其整套模型 | 新增，使用 2019 预印本的正式发表版本 |
| Coreixas et al. 2019 | §1.2，碰撞基底与平衡态必须配套转换的理论背景 | 新增 |
| Qi et al. 2025 | §1.2，体积／剪切耗散分离；联合 trace/22 松弛由本项目推导 | 新增 |
| Mirjalili & Mani 2021 | §1.2，相场质量通量必须进入动量输运的原则 | 新增，使用 2019 预印本的正式发表版本 |
| Zhan et al. 2022 | §1.2，完整非平流动量通量散度修正 | 新增，使用 2021 预印本的正式发表版本 |
| Zhang et al. 2020 | §1.4，接触线直接毛细载荷；共享应力通量是本项目实现 | 新增 |
| IAPWS 2011 | §1.5，水黏度关联式 (7)，与代码的四组系数核对 | 新增，技术发布文件 |
| NASA CAPE 文档 2025 | §1.5，Sutherland 空气黏度公式与 SI 常数 | 新增，软件技术文档 |
| Hu et al. 2019 | §1.6，Taichi 实现平台 | 新增，软件归属引用 |

新增论文的作者、年份、卷期／文章号与 DOI 已对照出版社、作者预印本或出版社提交的
Crossref 元数据核对。物性来源对照 IAPWS 原始发布文件和 NASA 版本化文档。
文中明确保留当前分裂通量离散不保证精确动能守恒的边界，没有借引用扩大守恒或稳定性结论。

未纳入当前正文的文献及原因详见 `citation_audit.json`：例如 Wang 2022 对应
未展开的 `enthalpy_d2q5`，Wissocq 2020 对应未选用的 `cm_hrr`；Huang 2024 是尚未接入的
融水守恒提案。v2 的守恒融化、切割单元重分配和隐式耦合文献不归为 v3 当前算法来源。
若以后改用这些分支或扩展相关工作，应同时更新正文和引用，不能仅将条目加入 BibTeX。

## 模板与投稿

主文件使用 `acmtog,anonymous,review`，对应
[SIGGRAPH Technical Papers 官方格式](https://www.siggraph.org/preparing-your-content/author-instructions/)，
并使用作者—年份引用。额外的 `nonacm` 仅用于工作稿，隐藏尚未取得的出版元数据；
正式投稿时移除它，填入真实 `\acmSubmissionID`，再依投稿要求设置元数据。
正式录用稿应按 ACM 提供的信息填写版权、DOI、卷期和文章号。

本文件是可继续扩展的**方法部分草稿**，没有编造作者、单位、实验结果或发表信息。
扩展全文时，在 `main.tex` 中加入摘要、引言、相关工作、实验与结论，并重新设置篇幅预算。

## 本文描述的实现范围

正文集中描述当前溃坝入口采用的组合：`cm_mrt`、`connected_ls`、
`consistent_flux`、`enthalpy_fv`，以及可选的密度源项限幅、共享面通量限幅、
补点矩投影和异常状态修复。它不表示这些选项都是 `IceFlowConfig` 的默认值：
基础配置的 `phase_gradient_scheme` 与 `phase_momentum_coupling` 仍设为 `legacy`；
前者保留兼容选项名，当前各流体算子共同使用的梯度核已统一为紧凑线性拟合。
落水入口默认开启 `consistent_flux`，溃坝入口进一步选择 `connected_ls` 的边界补点路径和局部保护。

特别核对了以下当前行为：

- 连通梯度是固定 `3×3` 邻域的单精度线性拟合；旧文档中的 `5×5` 二次拟合、
  QR 与壁面零空间修正不属于当前实现。
- 冰体先预测接触与允许位移，边界速度由允许位移得到；在旧几何上推进流体后才提交新位姿。
- 新露出节点沿连通供体逐层补点；无供体区域暂时关闭，不凭空指定气相或水相。
- 融化材料直接消失，不生成融水；材料残量和多余热量的处理均已写入正文。
- 相场截断、补点、热状态重映射及异常状态修复不保证全局守恒。
- 源项限幅为经验稳定化措施；正文不把历史算例完成记录当作当前方法的稳定性证明。

`enthalpy_d2q5`、`rm_mrt`、`cm_hrr`、旧版补点和非平衡补点等替代分支未展开，
以保持方法叙述集中且满足四页限制。正文中的 `f_q` 是代码的动量分布，
`g_q` 是代码的相场分布；论文统一使用物理含义命名，未照搬代码中所有临时变量名。

异常状态修复的具体门槛见 `stability_options.py::_gas_velocity_outlier`：
气相占优或速度 L1 范数超过 1 的节点，须同时超过 0.25 的速度和 0.15 的单步变化；
随后检查是否超过 `max(0.25, 2*邻域前一步最大速度+0.05)`，或是否同时超过
1 的速度和 0.5 的变化。修复供体来自前一步速度不超过 0.25 的连通邻居；
没有可靠供体时采用零速度、零动态压力。所有门槛均为格子单位。

## 源码对应

| 正文 | 实现位置 |
| --- | --- |
| 状态、材料栅格与质量属性 | `simulator.py::MovingBodyThermal2D`、`_sample_body_fraction`、`_reduce_moving_body_mass_properties` |
| 相场、压力分布与碰撞 | `lattice.py`、`simulator.py::_collide_phase_kernel`、`_collide_momentum_kernel`、`_update_pressure` |
| 共用梯度、连通补点与矩投影 | `phase_reconstruction.py` |
| 相场质量通量与局部保护 | `stability_options.py` |
| 移动边界与接触预测 | `simulator.py::_prepare_momentum_cut_links`、`body_motion.py`、`contact.py` |
| 热输运、换热与融化 | `simulator.py::MovingBodyThermal2D` |
| 时间推进与每次侵蚀后的几何刷新 | `simulator.py::step`、`_advance_thermal`、`_refresh_eroded_body` |

所有路径均相对于仓库的 `iceflow2d_v3/`。源码与历史说明存在差异时，以本次指纹对应的源码为准。

## 质量—动量耦合的推导

Sec. 1.2 从界面扩散／压缩引起的额外质量输运出发，由相场方程、密度关系和
不可压缩条件得到连续质量通量 `J_c`，再说明该通量携带的动量及其散度修正。
随后定义由碰撞前后相场分布的一阶矩估计的离散通量 `J`，解释共享面差分与显式力耦合。
`J_c` 与 `J` 分别表示连续模型和离散估计；文中保留两者的区别、限幅影响，
以及当前实现不继承逐链路严格一致性和离散动能守恒的说明。

## Sec. 1.3 的矢量图与验证

图 1 展示连通供体筛选、按方向秩求解梯度，以及新露出流体的同步逐层补点。
图 2 用局部解析线性场区分两个作用：缺失邻居时的方向归一化，以及排除隔着冰层的流体样本。
圆内数字是相场值；灰色区域是冰，蓝色为接受的供体，橙色虚线为被阻断的对角连接。

两个例子均令查询点 `phi=0.4`、同侧线性场的真实梯度为 `(0.12, 0.08)`。
例 (a) 排除东、东北和东南方向；相同供体上的 `3*b` 对照给出
`(0.0600, 0.0667)`，相对误差约 42.6%。例 (b) 排除东、北与被冰层阻隔的东北样本；
东北流体值独立设为 0.9。不检查连通性的加权线性拟合给出
`(0.1575, 0.1175)`，相对误差约 36.8%。连通拟合在两例中均为秩二，解析误差为零。
这些是局部算子对照，不是整场仿真的精度或收敛率实验；共线供体只能恢复沿其方向的梯度。

图中的斜冰层由旋转 45 度的材料网格单列占据率的线性插值得到，阈值为 0.5。
核验脚本继承未经修改的 `PhaseReconstruction`，在 Taichi CPU、float32 下验证供体掩码、
梯度与秩一闭合；两个秩二例子的梯度绝对误差均小于 `3e-8`。
独立的解析矩阵、对照数值和源码指纹保存在 `figures/*.json`。

```bash
make figures         # 有需要时重新生成矢量图；需 NumPy、Matplotlib
make verify-figures  # 额外检查当前梯度实现；需本项目的 Taichi 环境
make                 # 编译正文并检查四页上限
```

两张 PDF 已作为可移植插图保留，SVG 可继续编辑；常规 LaTeX 编译不要求 Taichi。

### 单栏排版与制图规格

两张图均按 **89 mm × 79.25 mm** 的单栏原稿尺寸绘制，在 ACM TOG 文中通过
`\includegraphics[width=\columnwidth]` 等比排版；当前实际栏宽约 **85.45 mm**。
图 1 和图 2 并列在第 2 页，各自使用独立的单栏 `figure` 环境，图文对应 Sec. 1.3。

分图标号、上方示意图中心和下方面板标题在两图之间统一对齐。文字使用
Arial 度量兼容的 Liberation Sans 无衬线字体：普通标签 6.3–7 pt，粗体分图字母
8 pt；嵌入论文后普通标签约 6.0–6.7 pt。全部标注为黑色；颜色用于几何与数据，
并由实线／虚线、叉号、空心／实心点和菱形补充区分。结构线宽为 0.4–0.95 pt。
PDF 嵌入 TrueType 字体，SVG 保留可编辑文字，原始解析例子及核验结果未变。

本次排版依据 [Nature 当前 figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
中的单栏尺寸、可编辑矢量、字体和可访问性要求，以及
[Nature artwork guide](https://www.nature.com/documents/nature-final-artwork.pdf)
中的分图标号规格（核对日期：2026-10-16）。这是制图规范参考；正文继续使用 ACM TOG 模板。

## Sec. 1.6 的局部保护（2026-10-19）

本节开头直接说明热子步的动机：显式热更新过大可能移走超过单元存量的水，
或造成温度过冲，因此单独细分热更新，保留流体时间步。
对流每个流体步更新，导热与侵蚀可以累积多个流体步；分别按流出速率和
最大热扩散率直接计算子步数，不再逐步推导稳定性不等式。
所用流出比例与 Fourier 数上限为 0.5 和 0.15，所需子步超过 64 时停止。

随后说明窄缝下兼顾效率与稳定性的动机，仅将密度梯度源项限幅、共享面动量通量
限幅分为两项，每项保留作用对象与公式依据。用户注释掉的异常速度修复和相场
区间裁剪源码保持原样，热子步不再单列。补点矩投影另作为恢复公式的一致性修正说明。

密度梯度的参考斜率来自 Hajabdollahi et al. 2021 式 (2)–(3) 的 tanh 平衡剖面；
1.5 倍门槛和小底值明确标为经验选择。共享面系数来自四个面的加速度预算。
本次保留 Sec. 1.6 之前的当前源文、宏、插图、代码和引用键集合，保留现有公式标签，
本机编译后仍为 4 页，含两张图和 13 项参考文献。
