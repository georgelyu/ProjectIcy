# 零流体体积与 LB 格点分类一致性

前版 `subgrid_viscous_200` 最后接受 6.575 s，下一批次出现 NaN/Inf。同源码逐步回放把首个容量越界定位到第 2,104,165 步（6.575515625 s）；提高到 1200 次容量迭代后仍在后续两步失稳。

[现场记录](../outputs/subgrid_viscous_200_failure_replay/zero_capacity_node.json) 显示格点 (129,253) 的精确流体体积为零，但平滑 SDF 为 +0.01479095，原分类仍把它当成流体。宏观恢复把该空格点的相分数设为零；下一次碰撞把它视作液水中的气体，表面张力加速度从约 10⁻⁶ 跃增到 2.43 格子单位，污染边界受力，再引发容量和速度连锁越界。此时冰量仍为初始的约 1.54%，无固壁链路的子网格黏性算子没有启用。

修复只补充新几何与旧局部查询间的一致性约束：非容器壁格点若精确流体体积等于零，则归类为固体。体积仍由原多边形积分给出，SDF 查询和 cut-link/SNIBB 公式未修改；任何正体积薄片均保留，不引入体积阈值，不裁剪质量、热量或动量，也不修改固定密度空气近似。

新增回归用四个面积各为 1/4 的材料三角形，在旋转 45° 后精确拼成一个完整世界格点。其中心的平滑 SDF 为正，但实际流体体积为零，因此必须为固体。平移 10⁻⁸ 格后重新打开的薄片必须保留为正体积流体。

- [负对照](../outputs/zero_capacity_mask_before_test.log)：旧实现明确失败。
- [几何回归](../outputs/zero_capacity_mask_geometry_tests.log)：10 项全部通过。
- [修复回放](../outputs/zero_capacity_mask_replay/result.json)：6.575–6.58 s 原失败批次通过，质量/焓残差 3.63×10⁻¹¹ / 7.22×10⁻¹⁰。
- [末期诊断续跑](../outputs/zero_capacity_mask_diagnostic_extension_200/diagnostic_provenance.json)：从修复回放的 6.58 s 状态完成到 10 s、零重试。残冰在 7.145–7.150 s 采样区间消失，子网格黏性算子启用 26,311 步；[7.4 s 独立库存](../outputs/zero_capacity_mask_comparison/diagnostic_post_extinction_inventory.json) 与 [10 s 最终库存](../outputs/zero_capacity_mask_diagnostic_extension_200/independent_inventory.json) 各九项检查通过，几何质量、惯量和刚体动量均为零。最终质量/焓残差为 1.94×10⁻¹⁰ / 8.63×10⁻¹⁰；这仍是跨版本初态的诊断续跑，不是从零正式验收。
- [独立库存积分](../outputs/zero_capacity_mask_replay/independent_inventory.json)：九项检查通过。
- [完整回归](../outputs/zero_capacity_mask_full_tests.log)：73 项通过（877.31 s，含原生 CUDA）。
- [当前冻结源码](../outputs/zero_capacity_mask_frozen_source/source_manifest.json)：26 个核心文件中仅 `geometry.py` 相对前版改变；[六组正式轨迹](../outputs/zero_capacity_mask_comparison/report_zh.md) 均从初态完成 10 s、各通过 23 项数值检查且零重试。细网格在 6.945–6.950 s 采样区间完全融化，最终独立库存、几何质量/惯量及刚体动量检查通过；网格、时间步和热块精度仍未通过完整验收。

回放显式加载旧版已接受状态，仅用于定位与验证缺陷，不可拼接成当前源码的完整 10 s 轨迹。精确零体积分类也不意味着平滑 SDF 代理的距离精度、SNIBB 与切割体积的空间误差或 A–H 物理验收已经通过。
