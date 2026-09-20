"""Optional CUDA graph launch adapter; all numerical kernels remain Taichi.

The adapter validates the observed 32-byte kernel ABI and rejects unrecognized
driver operations. Discovery/recording execute against a complete rollback
snapshot and never count as accepted evolution.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import taichi as ti

_active = False


def load_library():
    source = Path(__file__).with_suffix('.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = Path(tempfile.gettempdir())/'iceflow2d_v2_cuda'/digest
    library = cache/'schedule.so'
    if not library.exists():
        compiler = shutil.which('g++')
        cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-12.8'))/'targets/x86_64-linux'
        if not compiler or not (cuda/'include/cupti.h').exists():
            raise RuntimeError('Native graph execution requires g++ and the CUDA 12.8 toolkit with CUPTI; use --graph otherwise.')
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache/f'schedule-{os.getpid()}.so'
        command = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', str(source),
                   '-I'+str(cuda/'include'), '-L'+str(cuda/'lib'),
                   '-Wl,-rpath,'+str(cuda/'lib'), '-lcupti', '-lcuda', '-o', str(temporary)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(library)
    loaded = ctypes.CDLL(str(library))
    for name in ('begin_probe', 'end_probe', 'record_begin', 'destroy_graph'):
        getattr(loaded, name).argtypes = []
        getattr(loaded, name).restype = ctypes.c_int
    for name in ('record_end', 'launch_graph'):
        getattr(loaded, name).argtypes = [ctypes.c_int]
        getattr(loaded, name).restype = ctypes.c_int
    return loaded, digest


class CudaKernelSchedule:
    def __init__(self, simulation):
        global _active
        from .checkpoint import Snapshot
        if ti.__version__ != (1, 7, 4) or ti.lang.impl.current_cfg().arch != ti.cuda:
            raise RuntimeError('Native schedule requires Taichi 1.7.4 on CUDA.')
        if _active: raise RuntimeError('Only one native schedule may be active in a process.')
        if simulation.schedule is None: raise RuntimeError('Build the Taichi schedule first.')
        self.original = simulation.schedule
        self.library, digest = load_library()
        self.metadata = {'kind': 'cuda_graph', 'taichi_version': '1.7.4', 'adapter_cpp_sha256': digest}
        self.closed = False
        before = Snapshot(simulation)
        try:
            self._check(self.library.begin_probe(), 'discover')
            self.original.run(1); ti.sync()
            self._check(self.library.end_probe(), 'validate kernel layouts')
            self._check(self.library.record_begin(), 'record')
            self.original.run(1); ti.sync()
            self._check(self.library.record_end(1), 'build CUDA graph')
        finally:
            before.restore(simulation)
        _active = True

    @staticmethod
    def _check(status, operation):
        if status: raise RuntimeError(f'CUDA graph {operation} failed with status {status}.')

    def run(self, count):
        if self.closed or count < 0: raise RuntimeError('Invalid native schedule/count.')
        self._check(self.library.launch_graph(count), 'launch')

    def close(self):
        global _active
        if not self.closed:
            self._check(self.library.destroy_graph(), 'destroy')
            self.closed = True
            _active = False
