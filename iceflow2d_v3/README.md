# IceFlow2D v3：融化后直接消失的冰块

冰块从空气中落入热水，继续平移、转动并吸热融化。融化只减少材料网格中的冰质量；
融化掉的部分直接消失，不生成液水，也不向流体补入显热、线动量或角动量。
融化释放的流体格点初始化为空气，已有水随后可以通过流动进入这些位置。

## 运行

需要 NVIDIA CUDA GPU、Taichi 及仓库 `requirements.txt` 中的依赖。

```bash
python -m iceflow2d_v3
```

默认示例采用 `100 × 200` 网格，区域 `0.025 m × 0.050 m`，水深 `0.030 m`，
冰边长 `0.008 m`，初始转角 `5°`，冰底与水面间隙 `0.0015 m`。
水、冰、空气初温分别为 `90 / 0 / 20 °C`，四面墙和气相热界面绝热。
默认物理时长 `10 s`，采样间隔 `0.01 s`，LBM 时间步 `6.25e-6 s`。

短运行：

```bash
python -m iceflow2d_v3 \
  --end-time-s 0.00005 --output-interval-s 0.00005 \
  --no-plot --save-npz --no-progress
```

`--no-plot` 关闭 PNG/GIF，`--save-npz` 保存采样场，`--quiet` 关闭进度和摘要。
完整选项见 `python -m iceflow2d_v3 --help`。

## step 的三个阶段

`simulator.py` 中的 `step` 只校验步数，并按顺序调用：

1. `_advance_fluid`：动量与相场碰撞、流送、宏观量和压力恢复、局部相场限幅。
2. `_advance_body`：积分刚体运动，处理墙面接触，光栅化冰体并重建启闭格点。
3. `_advance_thermal`：每步重映射水温并对流水显热；累计到热更新间隔后执行导热、
   界面换热和冰体侵蚀，再更新剩余冰体的质量、质心、惯量和流体掩码。

热更新默认每 8 个 LBM 步执行一次。`step(n)` 的分批方式不改变累计热更新时序。
普通运动释放的格点从旧流体邻域重建；融化释放的格点液水相场置零。
冰完全融化后，其材料质量、几何覆盖和锐界消失，刚体停止参与运动。

## 热状态与数值约束

冰质量和显热保存在材料坐标网格，水体积和显热保存在世界网格。
热量先把剩余冰升至熔点，再按潜热减少冰质量；材料融尽后的多余热量直接丢弃。
低于初始材料单元质量百万分之一的数值残量直接清零，避免留下无限趋近于零的冰格点。
现有水仍通过导热和冰水界面换热更新温度。

热水体积直接取当前有效流体格点的 `dx² × water_phase`。开口变化时，已有水保持局部温度；
新湿格点从最近的湿邻域外推温度，搜索半径最多三格；没有湿邻域时取熔点。
不再进行全局水量投影、显热修正、融水回退分配或质量/能量残差判断。
因此本版本不要求冰水系统的总质量和总能量闭合。

保留 `water_phase_cutoff`（默认 `1e-3`）用于去除相场数值尾部，并同步调整对应 populations。
保留 CFL、Fourier 子步上限及非有限值诊断，以控制显式数值更新的稳定性。
流体仍使用压力–动量 D2Q9 LBM 和 Allen–Cahn 相场，冰壁仍使用 cut-link 动量交换。

## 输出与测试

默认输出目录为 `iceflow2d_v3/outputs/iceflow2d_v3/coupled_falling_ice_melting_2d/`。
`history.csv`、`metadata.json` 和可选的 `fields.npz` 保存冰体运动、剩余质量、融化比例、
当前水量和平均水温，不再输出守恒残差、水量目标或融水注入统计。
图像序列逐帧写盘；只有 `--save-npz` 会在内存中保留所有采样场。

```bash
python -m pytest iceflow2d_v3 -q
python -m ruff check iceflow2d_v3
```

测试覆盖局部相场限幅、热对流稳定性、部分/完全融化不生成液水、冰体完全消失、
多速率时序、刚体接触以及 CSV/NPZ/metadata 输出。

本模型只处理一个随融化侵蚀的刚体，不包含碎裂、重新冻结、沸腾或蒸发。
算法文档源码集中在 `code_algorithm_zh/`；`docs/main.tex` 复用同一份文档。
`benchmarks/step_profile_*.json` 和 `review_results.json` 是改动前的测量快照；
新版本应通过 `python -m iceflow2d_v3.benchmarks.profile_step --output /tmp/step.json` 重新测量。
