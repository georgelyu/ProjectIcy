"""Body-indexed device storage with the legacy one-body scalar view."""

import numpy as np
import taichi as ti


class BodyField:
    """Every allocation has a leading body axis; kernels index it explicitly.

    For a single body, host ``[None]`` and ``to_numpy`` keep the old scalar
    contract. ``array()`` always includes the body axis, including for N=1.
    """

    def __init__(self, dtype, count, *, components=None, tail=()):
        self.count = count
        self.tail = tail
        shape = (count, *tail)
        self.field = (
            ti.field(dtype, shape=shape)
            if components is None
            else ti.Vector.field(components, dtype, shape=shape)
        )

    def _index(self, index):
        if index is None:
            if self.count != 1:
                raise IndexError("multiple bodies require a body_id index")
            return 0
        if self.count == 1 and self.tail and not isinstance(index, tuple):
            return (0, index)
        return index

    def __getitem__(self, index):
        index = self._index(index)
        if ti.lang.impl.inside_kernel():
            # Return the underlying field lvalue, not a copied @ti.func result:
            # assignments and atomic reductions must reach device storage.
            indices = index if isinstance(index, tuple) else (index,)
            builder = ti.lang.impl.get_runtime().compiling_callable.ast_builder()
            return ti.lang.impl.subscript(builder, self.field, *indices)
        return self.field[index]

    def __setitem__(self, index, value):
        self.field[self._index(index)] = value

    def fill(self, value):
        self.field.fill(value)

    def array(self):
        return self.field.to_numpy()

    def to_numpy(self):
        values = self.array()
        return values[0] if self.count == 1 else values

    def from_numpy(self, values):
        values = np.asarray(values)
        if self.count == 1 and values.shape == self.array().shape[1:]:
            values = values[None]
        self.field.from_numpy(np.ascontiguousarray(values))
