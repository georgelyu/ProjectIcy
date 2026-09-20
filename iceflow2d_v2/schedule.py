"""Capture the exact Python kernel order as a Taichi 1.7.4 device graph.

Taichi's public Python graph helper cannot instantiate class/template kernels.
This small version-pinned adapter supplies their already compiled kernels to
the same graph builder. No numerical kernels or state updates are replaced.
"""
import taichi as ti


class KernelSchedule:
    def __init__(self, simulation):
        if ti.__version__ != (1, 7, 4):
            raise RuntimeError('The class-kernel graph adapter requires Taichi 1.7.4.')
        objects, seen = [], set()
        def visit(obj):
            if id(obj) in seen: return
            seen.add(id(obj))
            if isinstance(obj, (tuple, list)):
                for child in obj: visit(child)
                return
            if type(obj).__module__.startswith('iceflow2d_v2') and hasattr(obj, '__dict__'):
                if getattr(type(obj), '_data_oriented', False): objects.append(obj)
                for child in vars(obj).values(): visit(child)
        for obj in vars(simulation).values(): visit(obj)
        builder = ti.graph.GraphBuilder()
        sequence = builder.create_sequential()
        saved = []
        compiled = {}
        self.dispatches = 0
        try:
            for obj in objects:
                for name, member in type(obj).__dict__.items():
                    if not getattr(member, '_is_wrapped_kernel', False): continue
                    original = getattr(obj, name)
                    def capture(*args, _method=original):
                        primal, owner = _method._primal, _method._kernel_owner
                        key = (primal, owner, *args)
                        if key not in compiled:
                            specialization = primal.ensure_compiled(owner, *args)
                            compiled[key] = primal.compiled_kernels[specialization]
                        sequence.seq_.dispatch(compiled[key], [])
                        self.dispatches += 1
                    saved.append((obj, name))
                    setattr(obj, name, capture)
            simulation._advance()
        finally:
            for obj, name in saved: delattr(obj, name)
        builder.append(sequence)
        self.graph = builder.compile()

    def run(self, count):
        for _ in range(count): self.graph.run({})
