"""Aggregation multigrid for the same symmetric open-link Laplacian.

Coarse weights are P^T A P for piecewise-constant aggregates, not a second
geometric model. A final PCG polish always evaluates the original operator.
"""
import taichi as ti
from .lattice import direction


@ti.data_oriented
class Level:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.k = ti.Vector.field(4, ti.f32, (nx, ny))
        self.diag = ti.field(ti.f32, (nx, ny))
        self.b = ti.field(ti.f32, (nx, ny))
        self.x = ti.field(ti.f32, (nx, ny))
        self.tmp = ti.field(ti.f32, (nx, ny))

    @ti.func
    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    @ti.func
    def apply(self, i, j):
        result = 0.0
        for k in ti.static(range(4)):
            e = direction(ti.static((1, 2, 5, 6)[k]))
            ni, nj = i+e.x, j+e.y
            if self.inside(ni, nj): result += self.k[i, j][k]*(self.x[i, j]-self.x[ni, nj])
            ni, nj = i-e.x, j-e.y
            if self.inside(ni, nj): result += self.k[ni, nj][k]*(self.x[i, j]-self.x[ni, nj])
        return result

    @ti.kernel
    def load(self, weights: ti.template(), rhs: ti.template()):
        for i, j in self.x:
            self.k[i, j], self.b[i, j], self.x[i, j] = weights[i, j], rhs[i, j], 0.0

    @ti.kernel
    def coarsen(self, fine: ti.template()):
        for i, j in self.k:
            weights = ti.Vector.zero(ti.f32, 4)
            for k in ti.static(range(4)):
                e = direction(ti.static((1, 2, 5, 6)[k]))
                ni, nj = i+e.x, j+e.y
                if self.inside(ni, nj):
                    for a, b in ti.static(ti.ndrange(2, 2)):
                        x, y = 2*i+a, 2*j+b
                        if fine.inside(x, y):
                            for q in ti.static(range(4)):
                                f = direction(ti.static((1, 2, 5, 6)[q]))
                                xx, yy = x+f.x, y+f.y
                                if fine.inside(xx, yy) and xx//2 == ni and yy//2 == nj:
                                    weights[k] += fine.k[x, y][q]
                                xx, yy = x-f.x, y-f.y
                                if fine.inside(xx, yy) and xx//2 == ni and yy//2 == nj:
                                    weights[k] += fine.k[xx, yy][q]
            self.k[i, j] = weights

    @ti.kernel
    def diagonal(self):
        for i, j in self.diag:
            value = self.k[i, j].sum()
            for k in ti.static(range(4)):
                e = direction(ti.static((1, 2, 5, 6)[k]))
                ni, nj = i-e.x, j-e.y
                if self.inside(ni, nj): value += self.k[ni, nj][k]
            self.diag[i, j] = value

    @ti.kernel
    def smooth(self):
        for i, j in self.x:
            value = 0.0
            if self.diag[i, j] > 1e-20:
                value = self.x[i, j]+.7*(self.b[i, j]-self.apply(i, j))/self.diag[i, j]
            self.tmp[i, j] = value

    @ti.kernel
    def accept(self):
        for i, j in self.x: self.x[i, j] = self.tmp[i, j]

    @ti.kernel
    def restrict(self, fine: ti.template()):
        for i, j in self.b:
            value = 0.0
            for a, b in ti.static(ti.ndrange(2, 2)):
                x, y = 2*i+a, 2*j+b
                if fine.inside(x, y): value += fine.b[x, y]-fine.apply(x, y)
            self.b[i, j], self.x[i, j] = value, 0.0

    @ti.kernel
    def prolong(self, coarse: ti.template()):
        for i, j in self.x:
            if self.diag[i, j] > 1e-20: self.x[i, j] += coarse.x[i//2, j//2]

    @ti.kernel
    def coarse_cuda(self):
        # One block, uniform barriers, <=128 cells. This avoids launching one
        # kernel for every sweep on a nearly empty coarsest grid.
        ti.loop_config(block_dim=128)
        for node in range(128):
            shared = ti.simt.block.SharedArray((128,), ti.f32)
            i, j = node//self.ny, node%self.ny
            value, diagonal, rhs = 0.0, 0.0, 0.0
            weights = ti.Vector.zero(ti.f32, 8)
            neighbours = ti.Vector.zero(ti.i32, 8)
            if node < self.nx*self.ny:
                value, diagonal, rhs = self.x[i, j], self.diag[i, j], self.b[i, j]
                for k in ti.static(range(4)):
                    e = direction(ti.static((1, 2, 5, 6)[k]))
                    ni, nj = i+e.x, j+e.y
                    if self.inside(ni, nj):
                        weights[2*k], neighbours[2*k] = self.k[i, j][k], ni*self.ny+nj
                    ni, nj = i-e.x, j-e.y
                    if self.inside(ni, nj):
                        weights[2*k+1], neighbours[2*k+1] = self.k[ni, nj][k], ni*self.ny+nj
            shared[node] = value
            ti.simt.block.sync()
            for _ in range(96):
                total = rhs
                for k in ti.static(range(8)): total += weights[k]*shared[neighbours[k]]
                value = 0.0
                if diagonal > 1e-20: value = .25*shared[node]+.75*total/diagonal
                ti.simt.block.sync()
                shared[node] = value
                ti.simt.block.sync()
            if node < self.nx*self.ny: self.x[i, j] = value


class Multigrid:
    def __init__(self, nx, ny, cycles=6):
        self.levels = [Level(nx, ny)]
        self.cycles = cycles
        self.cuda = ti.lang.impl.current_cfg().arch == ti.cuda
        while nx*ny > 128:
            nx, ny = (nx+1)//2, (ny+1)//2
            self.levels.append(Level(nx, ny))

    def solve(self, weights, rhs):
        self.levels[0].load(weights, rhs)
        self.levels[0].diagonal()
        for fine, coarse in zip(self.levels, self.levels[1:]):
            coarse.coarsen(fine); coarse.diagonal()
        for _ in range(self.cycles):
            for fine, coarse in zip(self.levels, self.levels[1:]):
                for _ in range(2): fine.smooth(); fine.accept()
                coarse.restrict(fine)
            if self.cuda: self.levels[-1].coarse_cuda()
            else:
                for _ in range(96): self.levels[-1].smooth(); self.levels[-1].accept()
            for fine, coarse in reversed(list(zip(self.levels, self.levels[1:]))):
                fine.prolong(coarse)
                for _ in range(2): fine.smooth(); fine.accept()
