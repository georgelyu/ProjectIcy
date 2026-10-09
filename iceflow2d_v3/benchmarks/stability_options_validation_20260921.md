**相场法向正则化与 CM-MRT 声学松弛：实现验证（2026-09-21）**

两项优化已加入生产代码，分别由 `phase_normal_epsilon` 和 `cm_acoustic_relaxation_rate`
控制。后续默认设置已调整为 `cm_mrt`、`1e-3`、`0.25`，直接运行即可使用下面的组合；
下文记录的是实现验证时的实验结果：

```bash
python -m iceflow2d_v3 \
  --momentum-collision-model cm_mrt \
  --phase-normal-epsilon 1e-3 \
  --cm-acoustic-relaxation-rate 0.25
```

求解器仅修改 `_collide_phase` 和 `_collide_momentum_kernel`。
相场参数正则化压缩通量法向，保留防除零保护和原限幅逻辑；CM 参数联合松弛 trace、22
矩，并使用对应的 `(1-s/2)` 源项系数。剪切松弛率、平衡分布、力的定义、边界、热耦合及
物性关系没有改动。声学参数非默认值要求 CM-MRT；相场参数独立于动量模型。
实际参数已纳入 `config.to_dict()` 和输出 `metadata.json`。

**验证结果**

- `python -m pytest iceflow2d_v3 --ignore=iceflow2d_v3/outputs -q`：84 项通过。
- Ruff 检查和 `git diff --check` 通过。
- 实际 CLI 启用两项优化，运行 8 步，生成 history、metadata、NPZ，元数据正确记录选项。
- 完整源项矩阵对照覆盖 RM、原 CM、CM 松弛率 0.25 和 0.6；压力平衡态不动点通过。
- 压力波测试确认联合松弛增强衰减且保留波速；剪切波及各相、不同温度、冰面邻格的应力
  检查确认物理剪切黏度保持不变。
- 相场测试覆盖纯相端点、极小 epsilon 下的有限值、微小体相斑点抑制、平面界面厚度与相量。
- 关闭优化后，RM-MRT 和 CM-MRT 在 0、4096、8192 步的 12 项状态数组与修改前逐位一致。
  冰体质量的 f64 并行归约最大差异为 1.82×10⁻¹²（格子质量），相对量级约 10⁻¹⁵。

正式代码同时做了恢复诊断初始检查点、以及从正常预热初始化开始的两种完整复算。
物理黏度、网格、时间步、W 和 mobility 均保持原值。

| 运行方式 | t=1.07 s 底部最低 φ | 底部最大速度，mm/s | 压力波动 RMS，Pa | 速度波动 RMS，mm/s |
|---|---:|---:|---:|---:|
| 修改前 CM-MRT 诊断基线 | 0.948556 | 28.39 | 134.97 | 3.171 |
| 优化开启，恢复相同初始检查点 | 0.999940 | 2.36 | 12.51 | 0.283 |
| 优化开启，正常预热初始化 | 0.999930 | 2.65 | 13.69 | 0.321 |

瞬时指标取 x∈(2,23) mm、y∈(2,13) mm；时变指标在同一区域每隔 4 格采样，时间窗
1.0700–1.1212 s、间隔 50 μs，减去各点时间均值后计算 RMS；速度使用二维向量 RMS。
正式实现的完整轨迹不与先前诊断副本逐位相同，因此这里使用本次实测值，不沿用诊断表格
中的优化结果。开启优化也会影响冰体轨迹，速度更小不能单独作为所有物理预测更准确的证明。

验证覆盖到 1.1212 s；更长时间、网格收敛和曲面/移动边界精度仍应针对目标算例检查。
这里的声学耗散确实增加，但物理剪切黏度未提高。

测试日志、状态对照和复算脚本保存在
[实现验证目录](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/outputs/stability_options_20260921)。
原始机理和四组诊断见
[底部速度异常报告](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v3/benchmarks/bottom_velocity_root_causes_20260921.md)。
