import os
import numpy as np
import pytest
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot, fields


@pytest.mark.skipif(os.environ.get('ICEFLOW_NATIVE_TESTS') != '1', reason='optional CUDA 12.8/CUPTI adapter')
def test_native_graph_records_without_advancing_and_matches_taichi():
    from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
    sim = Simulation(Config(nx=25, ny=50, warmup_steps=0, projection_solver='multigrid'))
    sim.enable_graph(recovery=True)
    before = Snapshot(sim)
    native = CudaKernelSchedule(sim)
    try:
        assert sim.steps == 0
        for name, field in fields(sim).items():
            np.testing.assert_array_equal(field.to_numpy(), before.arrays[name], err_msg=name)
        sim.step(12)
        expected = Snapshot(sim)
        before.restore(sim)
        sim.schedule = native
        sim.step(12)
        for name, field in fields(sim).items():
            if name.endswith('.parent') or name == 'merge.members': continue
            np.testing.assert_allclose(field.to_numpy(), expected.arrays[name], atol=2e-7, rtol=2e-5, err_msg=name)
        sim.validate()
    finally:
        native.close()
