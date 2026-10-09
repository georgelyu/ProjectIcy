**冰块两侧约 45° 速度细条纹：定位与修复方向（2026-09-21）**

已复现用户指定输出在 0.74 s 开始明显的斜带。证据支持：**接触区的局部毛细力激发了当前低黏度压力型 D2Q9 离散的方向性误差，形成沿格子对角线的细条纹。** 这种条纹含有明确的数值成分；不能据此把冰块两侧的大尺度绕流、热对流或全部毛细流都判为非物理。

本次新增诊断脚本、独立实验及本报告，没有修改生产求解器。生成用户图片的原运行未保存 `fields.npz`；本次用其 `metadata.json` 对应的相同配置从初态复算到 0.8 s。在 0.2、0.4、0.6、0.74、0.8 s，冰体质量与用户 `history.csv` 相差约 10⁻¹² 格子质量，质心相差不超过约 2×10⁻⁹ m（保存时的浮点转换量级），支持复现了该运行。

![同一状态关闭不同驱动后的右侧斜带](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/diagonal_velocity_20260921/diagnosis_summary.png)

**观察对象与绘图排查。** 用户路径为 [velocity_frames](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/iceflow2d_v3/coupled_falling_ice_melting_2d/velocity_frames)。45° 首先是细带的空间走向，并不意味着其中的速度总朝下。例如 0.74 s 的右侧点 `(17.125, 30.625) mm`，实际速度为 `(-0.02168, +0.01513) m/s`，指向左上方接触区。绘图使用相同的横纵箭头缩放、`angles="xy"`、`scale_units="xy"` 和等比例坐标，没有发现把其他方向画成 45° 的问题。[箭头缩放](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/reporting.py:414)、[速度绘制](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/reporting.py:638)。

**触发源：接触区毛细力。** 从同一个 0.74 s 完整检查点续算 0.04 s，分别仅关闭热浮力或表面张力。为减少大涡旋背景的影响，在右侧固定窗口 `16<x<20 mm, 29.5<y<31.8 mm` 中，计算速度沿左上方向 `(-1,+1)/√2` 的分量。斜带取 `x+y=48 mm` 的一格宽条带，再减去两侧 `x+y=47、49 mm` 平行条带的均值，得到“斜带突出量”。各条带均只统计有效水格点。

| 0.74→0.78 s 对照 | 窗口内最大水速，m/s | 斜带突出量，mm/s |
| --- | ---: | ---: |
| 原计算 | 0.08124 | 24.64 |
| 仅关闭热浮力 | 0.08047 | 24.62 |
| 仅关闭表面张力 | 0.02945 | 0.916 |

关闭热浮力对该指标影响约 0.08%；关闭表面张力后降低约 96.3%。这是固定窗口、短时控制试验的指标，不是全场数值误差比例。关闭表面张力会改变物理问题，不能作为正式修复。

进一步固定 0.74 s 的冰体几何、相场和温度，将冰体速度、流体速度及动态压力清零，只推进动量：保留表面张力时，40 ms 后仍生成斜带，突出量 21.89 mm/s；去掉表面张力时为 −0.300 mm/s，细带基本消失。这说明条纹的再生成不需要持续运动、融化、节点 refill 或相场输运。已逐数组检查两组实验的相场、固体掩码、温度、质心和质量保持不变。

这里冻结的是入水过程中的非平衡界面；它不应被当作理论上必须静止的平衡接触线。该实验隔离驱动来源，不能单独证明所有产生的流速都是伪流。斜带尖锐形状的数值证据来自下面的均匀水试验。

0.74 s 右接触区 `(15.875,32.125) mm` 的毛细力除以密度约为 90.81 m/s²；这只是毛细项，**不是扣除压力梯度后的净加速度**。沿细带向水内延伸后，该项迅速变小，例如 `(17.625,30.125) mm` 约为 1.22×10⁻⁵ m/s²。条纹不等于整条路径上都有很大的局部毛细力，而是动量离散对局部受力的空间响应。

**为何偏向 45°：均匀水中也存在的离散响应。** 对生产代码实际采用的压力—动量平衡态、压力恢复及 CM 松弛矩阵，在静止纯水附近线性化。使用周期 `96×96` 网格、相同的 `ν_LU=3.14173×10⁻⁵`、声学松弛率 0.25，在一个格点施加很小的恒定竖向力，并扣除全域均值力。没有冰体边界、气液界面、热场或重力。

3200 步后出现沿 ±45° 的 X 形细条纹。再直接运行生产 `_collide_momentum_kernel`，仅把边界改为周期并规定同一外力；实际速度场与独立线性结果的相对 L2 差为 **0.1377%**。外力幅值为 `10⁻⁷ LU`，最大动量速度约 `1.835×10⁻⁵ LU`，处于小扰动范围。

相同力、黏度与周期网格的不可压缩非定常 Stokes 傅里叶参考没有同样的 X 形条纹。点源本身欠解析，参考也存在频谱截断纹理；把两种计算中的源项都展宽为相同的高斯分布后，LBM 的额外对角线响应仍可辨认。源宽从 0、1 增至 2 格时，距源 5–20 格环带内相对 Stokes 参考的速度差分别约为 111%、52.5%、12.9%。这些差异同时包含有限时间可压缩响应与空间离散误差，不能全部命名为单一的高阶模态误差；但它们明确显示了源项解析度和离散方向性的影响。

![同一受力的 LBM 与 Stokes 线性响应](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/diagonal_velocity_20260921/linear_vs_stokes.png)

默认 90 °C 纯水中，`τ_s=0.5+3ν_LU≈0.5000943`、`s_s≈1.999623`，非常接近过松弛极限。当前高阶 `21/12` 矩按速率 1 直接松弛到平衡态及半步源项；保持正确的长波剪切黏度，并不保证几个格子尺度的受力响应各向同性。独立试验也显示，单把声学松弛率从 0.25 恢复为 1，仍有明显对角条纹。因此它与之前底部声学驻波的诊断不能混为一谈。

对应实现位置：

- [分子黏度与剪切松弛](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2341)；[高阶矩更新](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2492)。
- [毛细应力](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2121) 内部使用两层混合梯度；[毛细力](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2214) 使用链路应力散度；[压力恢复](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2824) 使用中心密度梯度。曲面及阶梯接触区尚未证明这些离散满足静止力平衡。
- [移动节点重建](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/simulator.py:2080) 仍可能影响动态误差幅值，但不是本轮条纹出现的必要条件，不宜把它直接写成主要根因。

**修复优先级与已排除的捷径。**

1. **优先联合修正动量与力的高波数响应，保持物理剪切黏度。** 以本次均匀水局部受力试验作为独立验收，评估适配压力型分布的非平衡正则化、递归/混合应力重建或高阶矩闭合。必须同时保留压力恢复关系、完整密度梯度源项及各矩的 `(1-s/2)` 力系数；不能直接搬用质量型 LBM 的中心矩公式。除长波剪切衰减外，要检查不同方向、不同波长的强迫响应与耗散。
2. **配套改善接触区压力—毛细力平衡。** 在相同冰面几何和润湿条件下，使压力梯度、相场梯度、毛细应力散度的离散尽量相容，减少局部高频力残差，并继续保证冰体接收相反力和对应力矩。目标是同时通过平界面、旋转冰面接触线、静滴 Laplace 压差和当前动态入水回归。现有直立平面接触线近静止，只覆盖了一个简单构形。
3. **增加接触区解析度并做配套网格验证。** 保持物理几何、黏度、表面张力、物理界面厚度和 mobility；检查斜带幅值、宽度及冰体轨迹是否收敛。不能仅把 `W` 改大后认为问题解决，也不能只平滑输出图片。本次高斯源试验只支持“源项解析度值得改进”，没有验证可直接用于生产的力过滤器。

本次已经试过一个独立候选：将 `s21=s12` 从 1 改成 0.01，并使用匹配的 0.995 源项系数。均匀水中的条纹显著减弱，长波剪切衰减仍趋近原分子黏度；但在同一冻结接触区试验中，斜带突出量仅从 21.89 降到 12.71 mm/s，而全水域峰值从 **0.1066 升到 0.2035 m/s**。它只改变了误差形态，**本轮不接受为生产修复**。也不建议通过关闭 σ、提高物理黏度、清零某个速度分量或滤波图片处理该问题。

上述方向与低黏度 LBM 正则化中的各向异性研究一致，但论文并不直接验证本代码；正则化本身也可能给欠解析波引入错误耗散，必须实测。[Wissocq、Coreixas、Boussuge，PRE 102, 053305 (2020)](https://arxiv.org/abs/2006.07353)。空间非均匀外力还会产生与碰撞参数有关的离散误差，说明只改碰撞或只改力而不联合检查并不充分。[Silva，Computers & Fluids 203, 104537 (2020)](https://doi.org/10.1016/j.compfluid.2020.104537)。这两篇文献支持修复审查方向，不替代本轮控制试验。

**复现及范围。** [诊断脚本](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/benchmarks/diagnose_diagonal_velocity.py) 可从初态运行，或从完整检查点做短时控制。输出目录必须不存在：

```bash
python -m iceflow2d_v3.benchmarks.diagnose_diagonal_velocity \
  --output /tmp/ice_diagonal_baseline --end 0.8 --interval 0.02

python -m iceflow2d_v3.benchmarks.diagnose_diagonal_velocity \
  --restart /tmp/ice_diagonal_baseline/checkpoint_0118400 \
  --output /tmp/ice_diagonal_no_sigma --case no_sigma \
  --end 0.78 --interval 0.002 --checkpoint-times

python -m iceflow2d_v3.benchmarks.diagnose_diagonal_velocity \
  --restart /tmp/ice_diagonal_baseline/checkpoint_0118400 \
  --output /tmp/ice_diagonal_frozen --frozen --reset-flow \
  --end 0.78 --interval 0.002 --checkpoint-times
```

全部数据保存在 [diagonal_velocity_20260921](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/diagonal_velocity_20260921)。`baseline` 为从初态的生产复算；`restart_*` 为同状态续算；`frozen_*` 为冻结状态并清零流动的驱动对照；`candidate_frozen` 为未采纳候选。`linear_response.py`、`linear_reference.py`、`verify_linear.py` 分别计算独立格子响应、Stokes 参考和生产内核复核；按此顺序运行即可重建对应数据。`control_metrics.json` 和 `frozen_state_checks.json` 保存控制指标和冻结检查，`linear_verification.json` 保存生产内核验证结果。

生产 `simulator.py` SHA-256 在诊断期间保持为 `d2c636e99740de300c61f2b5aff182c6fa86cd4f454da88f654dba3a31328286`。本轮验证覆盖原算例至 0.8 s、同状态后续 40 ms、均匀水线性响应以及一个未采纳候选；没有完成正式修复、修复后的完整 5 s 动态验证或定量网格收敛。剩余任务已经缩小到可独立验收的动量/受力空间离散问题。
