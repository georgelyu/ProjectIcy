# D2Q5 冰水融化：独立推导与验证

本目录仅依据用户提供的 Wang 等（PRE 106, 055308, 2022）论文及自行推导的守恒律 / 解析解建立，没有读取或复用仓库已有代码。

阅读顺序：

1. [完整推导](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/derivation_zh.md)：原文温度模型的固液扩展、守恒焓 D2Q5、运动潜热输运、Stefan 条件，以及后续自由液面的接入方程。
2. [实测验证报告](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/validation_zh.md)：收敛曲线、误差表、融化图、运动输运局限。
3. [热核](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/thermal.py)、[验证程序](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/verify.py)、[报告生成程序](/home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/make_report.py)。

## 已实现的模型

温度决定移动分布的热通量，焓决定显热和潜热储存。熔点处液相率由局部焓直接求出；没有温度裁剪丢失能量，也没有有限熔化温区参数。

所有 GPU 变量为 Taichi f32。使用补偿求和及 fast_math=False 处理小热增量，不依赖 GPU f64。CPU 解析解和结果汇总使用 float64。

运动输运使用保守 MC-MUSCL / SSPRK2，与 D2Q5 导热作对称分裂；这是独立推导的实现，不是原文式（21）的原样复现。原文形式本身的固液扩展和它还需补充的闭合在推导报告中另行列明。

## 物理范围

代码验证等密度冰水热模型、给定运动，以及完全浸没冰块的导热融化。它不是受力决定运动的刚体–流体模拟器，不包含水–空气自由液面。

平移 / 转动测试验证了潜热不会因运动被错误生成或消耗，但 TVD 输运会使锐界面变宽。报告同时列出体积、形状和几何转动惯量误差。

未来气液两相流中的融冰需要冰、水、空气三个物态的质量与焓闭合；推导已列出密度变化、面质量通量、刚体质量 / 惯量和自由液面的衔接要求。

## 运行

~~~bash
conda run --no-capture-output -n icy python /home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/verify.py --suite all --grids 64 128 256
conda run --no-capture-output -n icy python /home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/verify.py --suite stefan --grids 64 128 256 512
conda run -n icy python /home/georgelyu/Repos/ProjectIcy/iceflow2d_v2/make_report.py
~~~

依赖为 icy 环境内已有的 numpy、matplotlib、taichi；无需 scipy、sympy 或仓库其他模块。CPU 可用 --arch cpu 运行，但当前报告数据来自 CUDA。

程序中的数组轴顺序为 (x, y)，图像绘制时转置。热核单位为格子单位；无量纲及物理量换算见推导式（13）、（21）、（27）。

## 产物

- results：实际运行的 JSON、CSV、NPZ、日志与来源哈希。
- figures：五组验证图，均有 PNG 和 SVG。
- reference：从用户 PDF 提取的文字及核对公式所用页面图片。

参数表、验证阈值和解析解全部随程序保留。运行结果不是对真实冰水物性、上浮速度或三物态耦合的实验验证。
