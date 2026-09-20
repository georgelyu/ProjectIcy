"""Local cut-cell pre-merging for the acoustic LB update.

Implicit scalar transport alone does not remove the acoustic CFL restriction
of a small cut cell. Merge each cut fluid node with an open, full cardinal
neighbour, and use identical extensive mixing weights for every phase/heat.
A local momentum dipole preserves angular momentum about a fixed origin.
"""
import taichi as ti
from .lattice import direction, cross, add_compensated, weight


@ti.data_oriented
class PairMerge:
    def __init__(self, cfg, geo, state, flow):
        self.cfg, self.geo, self.state, self.flow = cfg, geo, state, flow
        shape = (cfg.nx, cfg.ny)
        self.owner = ti.Vector.field(2, ti.i32, shape)
        self.transfer = ti.Vector.field(6, ti.f32, shape)
        self.pressure = ti.field(ti.f32, shape)
        self.activity = ti.field(ti.f32, shape)

    @ti.kernel
    def choose(self):
        for i, j in self.owner:
            target, best = ti.Vector([i, j]), 0.0
            volume = self.geo.capacity[i, j]
            if 0.0 < volume < 0.999999:
                for q in range(1, 5):
                    e = direction(q); ni, nj = i+e.x, j+e.y
                    if self.geo.inside(ni, nj) and not self.geo.wall(ni, nj) and self.geo.capacity[ni, nj] >= 0.999999:
                        face = 0.0
                        if e.x != 0: face = self.geo.face_open[ti.min(i, ni), j].x
                        else: face = self.geo.face_open[i, ti.min(j, nj)].y
                        # Merge along the outward interface normal. Choosing a
                        # tangential full neighbour leaves the normal acoustic
                        # mode untouched: a two-cell angular-momentum constraint
                        # preserves velocity differences perpendicular to its arm.
                        outward = ti.max(0.0, self.geo.sdf[ni, nj]-self.geo.sdf[i, j])
                        score = face*(outward+1e-6)
                        if score > best: target, best = ti.Vector([ni, nj]), score
            self.owner[i, j] = target
            self.transfer[i, j] = ti.Vector.zero(ti.f32, 6)
            self.activity[i, j] = 0.0
            self.pressure[i, j] = self.flow.pressure[i, j]

    @ti.kernel
    def proposals(self):
        for i, j in self.owner:
            owner = self.owner[i, j]
            if owner.x == i and owner.y == j:
                total = self.state.value(i, j)
                comp = ti.Vector.zero(ti.f32, 6)
                volume = self.geo.capacity[i, j]
                pressure = volume*self.flow.pressure[i, j]
                for q in range(1, 5):
                    e = direction(q); ni, nj = i+e.x, j+e.y
                    if self.geo.inside(ni, nj):
                        target = self.owner[ni, nj]
                        if target.x == i and target.y == j:
                            total, comp = add_compensated(total, comp, self.state.value(ni, nj))
                            volume += self.geo.capacity[ni, nj]
                            pressure += self.geo.capacity[ni, nj]*self.flow.pressure[ni, nj]
                if volume > self.geo.capacity[i, j]:
                    common = total/volume
                    p = pressure/volume
                    torque = 0.0
                    for q in range(1, 5):
                        e = direction(q); ni, nj = i+e.x, j+e.y
                        if self.geo.inside(ni, nj):
                            target = self.owner[ni, nj]
                            if target.x == i and target.y == j:
                                delta = self.geo.capacity[ni, nj]*common-self.state.value(ni, nj)
                                self.transfer[ni, nj] = delta
                                self.pressure[ni, nj] = p
                                torque += cross(ti.cast(e, ti.f32), ti.Vector([delta[4], delta[5]]))
                    # Spread the angular correction by the same volume weights.
                    # Choosing a single microscopic leaf would create huge u=P/M.
                    leaf_volume = volume-self.geo.capacity[i, j]
                    for q in range(1, 5):
                        e = direction(q); ni, nj = i+e.x, j+e.y
                        if self.geo.inside(ni, nj):
                            target = self.owner[ni, nj]
                            if target.x == i and target.y == j:
                                arm = ti.cast(e, ti.f32)
                                correction = -torque*self.geo.capacity[ni, nj]/leaf_volume*ti.Vector([-arm.y, arm.x])
                                self.transfer[ni, nj][4] += correction.x
                                self.transfer[ni, nj][5] += correction.y
                    self.pressure[i, j] = p

    @ti.kernel
    def commit(self):
        for i, j in self.owner:
            target = self.owner[i, j]
            delta = self.transfer[i, j]
            if target.x == i and target.y == j:
                delta = ti.Vector.zero(ti.f32, 6)
                for q in range(1, 5):
                    e = direction(q); ni, nj = i+e.x, j+e.y
                    if self.geo.inside(ni, nj):
                        owner = self.owner[ni, nj]
                        if owner.x == i and owner.y == j: delta -= self.transfer[ni, nj]
            self.state.add(i, j, delta)
            self.activity[i, j] = ti.abs(delta[0])
            dp = self.pressure[i, j]-self.flow.pressure[i, j]
            for q in range(9):
                change = 3*weight(q)*dp
                if q == 0: change -= 3*dp
                self.flow.f[q, i, j] += change
            self.flow.pressure[i, j] = self.pressure[i, j]

    def step(self):
        self.choose(); self.proposals(); self.commit()


from .agglomeration import PatchMerge as CutCellMerge
