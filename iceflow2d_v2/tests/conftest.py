import os
import pytest
import taichi as ti


@pytest.fixture(scope='session', autouse=True)
def taichi_runtime():
    arch = ti.cuda if os.environ.get('ICEFLOW_TEST_ARCH', 'cuda').lower() == 'cuda' else ti.cpu
    ti.init(arch=arch, default_fp=ti.f32, fast_math=False, random_seed=1729)
    yield
    ti.reset()
