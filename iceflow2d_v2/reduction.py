"""Coalesced, compensated device reductions (no host scalar reads)."""
import taichi as ti
from .lattice import add_compensated


@ti.data_oriented
class DeviceSum:
    def __init__(self, count, width, precision=ti.f32):
        self.count, self.width, self.precision = count, width, precision
        self.first = ti.Vector.field(width, precision, (count+31)//32)
        self.second = ti.Vector.field(width, precision, ((count+31)//32+31)//32)

    @ti.kernel
    def collect(self, read: ti.template()):
        for part in self.first:
            value, comp = ti.Vector.zero(self.precision, self.width), ti.Vector.zero(self.precision, self.width)
            for offset in range(32):
                index = part+offset*self.first.shape[0]
                if index < self.count:
                    value, comp = add_compensated(value, comp, read(index), self.precision)
            self.first[part] = value-comp

    @ti.kernel
    def group(self):
        for part in self.second:
            value, comp = ti.Vector.zero(self.precision, self.width), ti.Vector.zero(self.precision, self.width)
            for offset in range(32):
                index = part+offset*self.second.shape[0]
                if index < self.first.shape[0]:
                    value, comp = add_compensated(value, comp, self.first[index], self.precision)
            self.second[part] = value-comp

    @ti.kernel
    def finish(self, write: ti.template()):
        value, comp = ti.Vector.zero(self.precision, self.width), ti.Vector.zero(self.precision, self.width)
        ti.loop_config(serialize=True)
        for index in range(self.second.shape[0]):
            value, comp = add_compensated(value, comp, self.second[index], self.precision)
        write(value-comp)

    def run(self, read, write):
        self.collect(read); self.group(); self.finish(write)
