# v3 step 性能测量

当前 `step` 分为 `_advance_fluid`、`_advance_body` 和 `_advance_thermal`。
计时脚本直接包装这三个实际入口，不复制执行流程。

```bash
python -m iceflow2d_v3.benchmarks.profile_step --output /tmp/step-wall.json
python -m iceflow2d_v3.benchmarks.profile_step --mode kernels --output /tmp/step-gpu.json
```

脚本预热 JIT，在同一状态检查点重复测量，并记录源码 SHA-256。
分阶段计时会增加同步开销，因此同时提供完整 step 的测量结果。

`benchmarks/step_profile_*.json` 是删除融水和守恒校正之前的历史数据，不能用于解释当前代码耗时。
