"""Monotone GPU union-find for the actual open-face connection graph.

Each union lowers a root index using atomic_min. No bounded-distance flood
or assumption about the body's separation from a wall is required.
"""
import taichi as ti
from .lattice import direction


@ti.data_oriented
class Components:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.parent = ti.field(ti.i32, nx*ny)
        self.minimum = ti.field(ti.i32, ())
        self.mark = ti.field(ti.i32, nx*ny)

    @ti.func
    def root(self, node):
        while self.parent[node] != node:
            node = self.parent[node]
        return node

    @ti.func
    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        while a != b:
            lo, hi = ti.min(a, b), ti.max(a, b)
            previous = ti.atomic_min(self.parent[hi], lo)
            a, b = self.root(lo), self.root(previous)

    @ti.kernel
    def seed(self, active: ti.template(), threshold: ti.template()):
        self.minimum[None] = self.nx*self.ny
        for i, j in active:
            node = i*self.ny+j
            self.parent[node] = -1
            self.mark[node] = 0
            if active[i, j] > threshold:
                self.parent[node] = node
                ti.atomic_min(self.minimum[None], node)

    @ti.kernel
    def connect(self, weights: ti.template(), diagonals: ti.template()):
        for i, j in weights:
            node = i*self.ny+j
            if self.parent[node] >= 0:
                for k in ti.static(range(4 if diagonals else 2)):
                    q = ti.static((1, 2, 5, 6)[k])
                    e = direction(q)
                    ni, nj = i+e.x, j+e.y
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        other = ni*self.ny+nj
                        if weights[i, j][k] > 0.0 and self.parent[other] >= 0:
                            self.union(node, other)

    @ti.kernel
    def labels(self, target: ti.template()):
        for i, j in target:
            node = i*self.ny+j
            label = -1
            if self.parent[node] >= 0:
                root = self.root(node)
                self.parent[node] = root
                label = root+1
                if root == self.minimum[None]: label = 0
            target[i, j] = label

    @ti.kernel
    def roots(self, target: ti.template()):
        for i, j in target:
            node = i*self.ny+j
            root = -1
            if self.parent[node] >= 0: root = self.root(node)
            target[i, j] = root

    @ti.kernel
    def mark_top(self, top: ti.template()):
        for i in range(self.nx):
            node = i*self.ny+top
            if self.parent[node] >= 0:
                ti.atomic_max(self.mark[self.root(node)], 1)

    @ti.kernel
    def connected_to_top(self, target: ti.template()):
        for i, j in target:
            node = i*self.ny+j
            connected = 0
            if self.parent[node] >= 0:
                connected = self.mark[self.root(node)]
            target[i, j] = connected
