# 物性插值、纯相保持和 CM-MRT 精度修正

用户约束：150×50×30 粗网格，CM-MRT，界面宽度仍为 5 格；水和空气的
物性公式不变。全局水量投影仍关闭。时间步仍为 25 μs。本次不输出三维数据场。

## 修正

1. **先插值动力黏度，再除以局部密度**：
   `mu = phi*mu_water(T) + (1-phi)*mu_air`，
   `nu = mu/(phi*rho_water + (1-phi)*rho_air)`。
   原实现线性插值 `nu`，使有效 `mu=rho*nu` 超过两相端点。
   90°C 水和 20°C 空气在 `phi=0.5` 时，原值为 3.70979 mPa·s；
   修正值为 0.166153 mPa·s，位于两相之间。纯水仍为 0.314173 mPa·s，
   纯空气仍为 0.0181331 mPa·s。
2. **法向量正则化只用于避免除零**：默认 `phase_normal_epsilon` 从
   `0.01` 改为 `1e-8`。原值在界面尾部与梯度同量级，显著削弱压缩通量，
   造成体相混合。没有修改 W、mobility，没有把相场按阈值强行改成 0/1。
3. **单精度格子权重直接取精确分数**：D3Q27 权重直接写为
   `8/27, 2/27, 1/54, 1/216`，避免三个已舍入因子相乘。
   原 f32 权重的 `3*sum(w*c_y^2)=1.0000001080334187`；
   修正后为 `1.0000000074505806`。这个很小的逐步动量偏差足以污染
   低分子黏度下的剪切衰减。CM-MRT 仍使用分布函数的动力学剪切矩，未改为 HRR。
4. **局部强迫动量使用双精度闭合**：仅修正权重后，混合密度下仍会因为
   `rho*u`、平衡分布和宏观求和各自的 f32 舍入产生系统偏差。
   用 f64 累计碰撞前后的一阶矩缺陷，通过三个反对称轴向分布对补回。
   这保持零阶矩、压力和二阶原始矩不变；宏观动量、移动分布和也以 f64 累计。
   分布数组和宏观场仍为 f32；没有新增全域归约或水量投影。

## 定向验证

CUDA 周期剪切波，波长 24 格，振幅 1e-4 LU，8000 步，拟合实际速度的
Fourier 振幅衰减率。以下是相对物性公式的黏度误差，而非输入松弛率误差：

| 流体 | 修正前 | 修正后 |
| --- | ---: | ---: |
| 水 90°C | −19.565% | +0.491% |
| 水 20°C | −5.609% | +0.570% |
| 水 0°C | −3.036% | +0.596% |
| 空气 20°C | +0.226% | +0.569% |

[最终剪切波数据](../outputs/constitutive_fix_20261021/shear_momentum_closure.json)。
均匀混合物 `phi=0.1,0.5,0.99` 的黏度衰减误差分别为
`+0.583%,+0.676%,+0.695%`。只修权重的中间版本在 phi=0.99 时仍有
−6.972% 误差，见[中间版本失败记录](../outputs/constitutive_fix_20261021/shear_with_mixtures.json)。
该波长本身有约 0.57% 的离散色散误差，不把单个波长的通过视为网格收敛。
最终源码还以 1e-3 LU 振幅重复了全部 7 种物性检查，最大误差为
0.699997%，见[第二振幅验证](../outputs/constitutive_fix_20261021/shear_amplitude_1e3.json)。
两个振幅合计 14 个案例均通过，最后一次检查的源文件哈希与 4 秒运行一致。

周期平面界面在静止和 `u_x=0.025` LU 两种条件下演化 8000 步。
体相统计取距解析界面超过 12 格的区域：

| 指标 | 原正则化 0.01 | 新正则化 1e-8 |
| --- | ---: | ---: |
| 静止气相的 phi 中位数 | 0.00709196 | 3.10531e-7 |
| 平移气相的 phi 中位数 | 0.00702761 | 2.87082e-7 |
| 静止水相的 phi 中位数 | 0.99290472 | 0.99999946 |
| 平移水相的 phi 中位数 | 0.99297222 | 0.99999893 |
| 静止界面拟合宽度 | 5.0683 格 | 5.0566 格 |
| 平移界面拟合宽度 | 5.0690 格 | 5.0568 格 |

新设置的最大体相误差低于 3.7e-5，水量相对漂移低于 3.2e-5。
宽度按 `logit(phi)=4*d/W` 在 0.1<phi<0.9 的区域拟合。
数据见[相场输运检查](../outputs/constitutive_fix_20261021/phase_transport.json)，
图见[物性与相场对照](../outputs/constitutive_fix_20261021/constitutive_checks.png)。

CUDA 完整回归 **31 项通过**，覆盖碰撞强迫动量、动力学剪切矩、压力闭合、
热量输运、移动冰与几何补点、接触和检查点恢复。日志见
[tests_final_cuda.log](../outputs/constitutive_fix_20261021/tests_final_cuda.log)。

复现命令：

```bash
python iceflow3d/benchmarks/check_mrt_modes.py --output shear.json
python iceflow3d/benchmarks/check_phase_transport.py --output phase.json
ICEFLOW3D_TEST_BACKEND=cuda python -m unittest discover -s iceflow3d/tests -v
```

## 4 秒耦合复跑

从零完成 **160000 步 / 4.0 s**，计算墙钟时间 954.41 s。
150×50×30、CM-MRT、dx=1 mm、dt=25 μs、W=5、mobility=0.1，
水/空气物性公式、几何、初始温度、边界和重力均不变。
全局水量投影与气相修复均关闭，未增加速度限幅或湍流黏度。

- 累计最大 L1 速度：0.369476 LU，低于 0.5 保护阈值；末态 0.0426085 LU。
- 201 个输出时刻的宏观场，以及末态四组分布函数，均为有限值。
- 温度范围 0–90°C；计入已记录的热场重映射后的最大能量残差
  3.05e-6 J。这个收支指标不意味着重映射本身没有改变热量。
- 末态冰质量 3.01271e-5 kg；接触残差最大 3.98e-12 LU。
- 水量漂移 **+2.57034%**；原运行是 +3.38847%。全局投影保持关闭，
  这里没有宣称水量已守恒，也没有通过投影隐藏漂移。
- 仅输出 z=15 mm 二维截面，每 0.02 s 一帧，共 201 帧；无三维场或检查点。

末态体相对照如下。统计来自 z 中心截面，取距 `phi=0.5` 界面和固体至少
6 格的流体区域；它们不是全体积统计。黏度根据插值后的截面 phi/T 计算，
不是三维黏度的严格截面平均。

| 4 s 指标 | 原运行 | 修正后 |
| --- | ---: | ---: |
| 水区 phi 中位数 | 0.99247783 | 0.99994153 |
| 气区 phi 中位数 | 0.00754127 | 0.00008876 |
| 气区密度中位数，kg/m³ | 8.78184 | 1.33865 |
| 水区 nu / 同温度纯水 nu 中位数 | 1.328491 | 1.000003 |

气区密度仍未处处达到 1.25 kg/m³；5 格弥散界面及其尾部仍保留，
没有把剩余小相分数强行截为零。此次通过证明上述物性/精度缺陷得到修正
并完成耦合稳定性检查，不代表毫米级气泡、薄液膜或真实溅射已经得到充分分辨。

结果：

- [运行记录及源文件哈希](../outputs/dambreak_150_4s_corrected_20261021/metadata.json)
- [全程核验](../outputs/dambreak_150_4s_corrected_20261021/verification.json)
- [速度、温度、相场截面视频](../outputs/dambreak_150_4s_corrected_20261021/dambreak_z_mid.mp4)
- [相场前后对照](../outputs/constitutive_fix_20261021/comparison/phase_comparison.png)
- [体相物性前后对照](../outputs/constitutive_fix_20261021/comparison/bulk_comparison.png)
- [逐时刻对照数据](../outputs/constitutive_fix_20261021/comparison/comparison.json)

复现运行：

```bash
python -m iceflow3d.examples.dambreak_midplane \
  --resolution-x 150 --end-time-s 4 --output-interval-s 0.02 \
  --output-dir iceflow3d/outputs/dambreak_150_4s_corrected_new
python -m iceflow3d.postprocessing.midplane_fields \
  iceflow3d/outputs/dambreak_150_4s_corrected_new
```
