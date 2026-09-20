"""Connected local patches with minimum-energy, angular-conservative mixing.

Two collinear cells cannot damp a transverse acoustic mode while preserving
angular momentum. Small cut components therefore join a connected fuller patch;
each accepted patch retains its rigid translation/rotation and removes only
its unresolved internal velocity variation. All phase/heat weights coincide.
"""
import taichi as ti
from .lattice import add_compensated, cross, weight, direction
from .connectivity import Components


@ti.data_oriented
class PatchMerge:
    def __init__(self, cfg, geo, state, flow):
        self.cfg, self.geo, self.state, self.flow = cfg, geo, state, flow
        shape = (cfg.nx, cfg.ny)
        self.component_volume = ti.field(ti.f32, shape)
        self.component_cut = ti.field(ti.i32, shape)
        self.active = ti.field(ti.i32, shape)
        self.edges = ti.Vector.field(2, ti.f32, shape)
        self.components = Components(*shape)
        self.root = ti.field(ti.i32, shape)
        self.leader = ti.Vector.field(2, ti.i32, shape)
        self.owner = ti.Vector.field(2, ti.i32, shape)
        self.common = ti.Vector.field(6, ti.f32, shape)
        self.spin = ti.Vector.field(3, ti.f32, shape)
        self.mean_pressure = ti.field(ti.f32, shape)
        self.transfer = ti.Vector.field(6, ti.f32, shape)
        self.activity = ti.field(ti.f32, shape)
        self.member_count = ti.field(ti.i32, cfg.nx*cfg.ny)
        self.member_offset = ti.field(ti.i32, cfg.nx*cfg.ny)
        self.member_cursor = ti.field(ti.i32, cfg.nx*cfg.ny)
        self.members = ti.field(ti.i32, cfg.nx*cfg.ny)
        self.attachment = ti.field(ti.i32, cfg.nx*cfg.ny)
        self.orphan_attachment = ti.field(ti.i32, cfg.nx*cfg.ny)
        groups = (cfg.nx*cfg.ny+255)//256
        self.group_count = ti.field(ti.i32, groups)
        self.group_offset = ti.field(ti.i32, groups)

    @ti.kernel
    def prepare_components(self):
        for i, j in self.active:
            self.active[i, j] = ti.cast(self.geo.capacity[i, j] > 0.0, ti.i32)
            e = ti.Vector([0.0, 0.0])
            for axis in ti.static(range(2)):
                ni, nj = i+1-axis, j+axis
                if self.geo.inside(ni, nj):
                    if i//3 == ni//3 and j//3 == nj//3:
                        e[axis] = self.geo.face_open[i, j][axis]
            self.edges[i, j] = e

    @ti.kernel
    def measure_components(self):
        for i, j in self.root:
            if self.root[i, j] == i*self.cfg.ny+j:
                volume, cut = 0.0, 0
                for a, b in ti.ndrange(3, 3):
                    ni, nj = 3*(i//3)+a, 3*(j//3)+b
                    if self.geo.inside(ni, nj) and self.root[ni, nj] == self.root[i, j]:
                        v = self.geo.capacity[ni, nj]
                        volume += v
                        if 0.0 < v < .999999: cut = 1
                self.component_volume[i, j], self.component_cut[i, j] = volume, cut

    @ti.kernel
    def join_small_components(self):
        # A tile can contain a large fluid patch AND a separate thin corner.
        # Assess each connected component, then attach the small one across an
        # actual open face. First connect the entire unresolved film, including
        # volume valleys that cannot be traversed by an uphill-only rule.
        # The next two kernels attach each film to just one fuller patch.
        for i, j in self.root:
            if self.root[i, j] == i*self.cfg.ny+j:
                if self.component_cut[i, j] and self.attachment[i*self.cfg.ny+j] == self.cfg.nx*self.cfg.ny:
                    for a, b in ti.ndrange(3, 3):
                        x, y = 3*(i//3)+a, 3*(j//3)+b
                        if self.geo.inside(x, y) and self.root[x, y] == self.root[i, j]:
                            for q in range(1, 5):
                                e = direction(q)
                                nx, ny = x+e.x, y+e.y
                                if self.geo.inside(nx, ny) and self.root[nx, ny] >= 0:
                                    other = self.root[nx, ny]
                                    axis = 0
                                    if e.x == 0: axis = 1
                                    opening = self.geo.face_open[ti.min(x, nx), ti.min(y, ny)][axis]
                                    if opening > 0.0 and self.component_cut[other//self.cfg.ny, other%self.cfg.ny] and self.attachment[other] == self.cfg.nx*self.cfg.ny:
                                        self.components.union(i*self.cfg.ny+j, other)

    @ti.kernel
    def choose_full_receivers(self):
        for node in self.attachment: self.attachment[node] = self.cfg.nx*self.cfg.ny
        for i, j in self.root:
            if self.root[i, j] == i*self.cfg.ny+j and self.component_cut[i, j]:
                best, target = 0.0, -1
                for a, b in ti.ndrange(3, 3):
                    x, y = 3*(i//3)+a, 3*(j//3)+b
                    if self.geo.inside(x, y) and self.root[x, y] == self.root[i, j]:
                        for q in range(1, 5):
                            e = direction(q)
                            nx, ny = x+e.x, y+e.y
                            if self.geo.inside(nx, ny) and self.root[nx, ny] >= 0:
                                other = self.root[nx, ny]
                                volume = self.component_volume[other//self.cfg.ny, other%self.cfg.ny]
                                cut = self.component_cut[other//self.cfg.ny, other%self.cfg.ny]
                                axis = 0
                                if e.x == 0: axis = 1
                                opening = self.geo.face_open[ti.min(x, nx), ti.min(y, ny)][axis]
                                # An uncut tile can still be a one-cell-wide
                                # strip beside a container wall or solid edge.
                                # Attaching a thin cut cell to that collinear
                                # strip cannot damp transverse acoustic spin
                                # while preserving angular momentum. At least
                                # four *uncut* cells in a 3x3 tile guarantees a
                                # two-dimensional receiver. A moderately cut
                                # tile is not a stable anchor for another cut
                                # tile: their angular mode can retain a large
                                # boundary impulse with too little inertia.
                                if cut == 0 and volume >= 4 and opening > best:
                                    best, target = opening, other
                if target >= 0:
                    self.attachment[i*self.cfg.ny+j] = target

    @ti.kernel
    def choose_orphan_receivers(self):
        for node in self.orphan_attachment: self.orphan_attachment[node] = self.cfg.nx*self.cfg.ny
        for i, j in self.root:
            node = i*self.cfg.ny+j
            if self.root[i, j] == node and self.component_cut[i, j] and self.attachment[node] == self.cfg.nx*self.cfg.ny:
                best, target = 0.0, -1
                for a, b in ti.ndrange(3, 3):
                    x, y = 3*(i//3)+a, 3*(j//3)+b
                    if self.geo.inside(x, y) and self.root[x, y] == node:
                        for q in range(1, 5):
                            e = direction(q)
                            nx, ny = x+e.x, y+e.y
                            if self.geo.inside(nx, ny) and self.root[nx, ny] >= 0:
                                other = self.root[nx, ny]
                                axis = 0
                                if e.x == 0: axis = 1
                                opening = self.geo.face_open[ti.min(x, nx), ti.min(y, ny)][axis]
                                if self.attachment[other] < self.cfg.nx*self.cfg.ny and opening > best:
                                    best, target = opening, other
                if target >= 0:
                    root = self.components.root(node)
                    ti.atomic_min(self.orphan_attachment[root], target)

    @ti.kernel
    def attach_full_receivers(self):
        for node in self.attachment:
            if self.attachment[node] < self.cfg.nx*self.cfg.ny:
                self.components.union(node, self.attachment[node])
        for node in self.orphan_attachment:
            if self.orphan_attachment[node] < self.cfg.nx*self.cfg.ny:
                self.components.union(node, self.orphan_attachment[node])

    @ti.kernel
    def mark_active_components(self):
        for i, j in self.root:
            if self.root[i, j] >= 0 and 0.0 < self.geo.capacity[i, j] < .999999:
                ti.atomic_max(self.components.mark[self.root[i, j]], 1)
        for i, j in self.active:
            active = 0
            if self.root[i, j] >= 0: active = self.components.mark[self.root[i, j]]
            self.active[i, j] = active

    @ti.kernel
    def count_members(self):
        for node in self.member_count: self.member_count[node] = 0
        for i, j in self.root:
            if self.active[i, j]: ti.atomic_add(self.member_count[self.root[i, j]], 1)

    @ti.kernel
    def prefix_groups(self):
        for group in self.group_count:
            total = 0
            for k in range(256):
                node = 256*group+k
                if node < self.cfg.nx*self.cfg.ny:
                    self.member_offset[node] = total
                    total += self.member_count[node]
            self.group_count[group] = total

    @ti.kernel
    def prefix_bases(self):
        total = 0
        ti.loop_config(serialize=True)
        for group in range(self.group_count.shape[0]):
            self.group_offset[group] = total
            total += self.group_count[group]

    @ti.kernel
    def list_members(self):
        for node in self.member_cursor: self.member_cursor[node] = 0
        for i, j in self.root:
            if self.active[i, j]:
                root = self.root[i, j]
                index = self.member_offset[root]+self.group_offset[root//256]+ti.atomic_add(self.member_cursor[root], 1)
                self.members[index] = i*self.cfg.ny+j

    @ti.func
    def member_start(self, root):
        return self.member_offset[root]+self.group_offset[root//256]

    @ti.kernel
    def choose_leaders(self):
        for i, j in self.leader:
            if self.active[i, j] and self.root[i, j] == i*self.cfg.ny+j:
                best, leader = ti.cast(-1.0, ti.f64), ti.Vector([i, j])
                root = self.root[i, j]
                start = self.member_start(root)
                for k in range(self.member_count[root]):
                    node = self.members[start+k]
                    ni, nj = node//self.cfg.ny, node%self.cfg.ny
                    capacity = self.geo.capacity[ni, nj]
                    if capacity > best or (capacity == best and node < leader.x*self.cfg.ny+leader.y):
                        leader, best = ti.Vector([ni, nj]), capacity
                self.leader[i, j] = leader

    @ti.kernel
    def assign(self):
        for i, j in self.owner:
            owner = ti.Vector([i, j])
            if self.active[i, j] and self.root[i, j] >= 0:
                root = self.root[i, j]
                owner = self.leader[root//self.cfg.ny, root%self.cfg.ny]
            self.owner[i, j] = owner

    @ti.kernel
    def gather(self):
        for i, j in self.owner:
            if self.active[i, j] and (self.owner[i, j] == ti.Vector([i, j])).all():
                # [V, V*x, V*y, V*r^2, p*V, angular about the leader]
                moments, mc = ti.Vector.zero(ti.f64, 6), ti.Vector.zero(ti.f64, 6)
                total, comp = ti.Vector.zero(ti.f64, 6), ti.Vector.zero(ti.f64, 6)
                root = self.root[i, j]
                start = self.member_start(root)
                for k in range(self.member_count[root]):
                    node = self.members[start+k]
                    ni, nj = node//self.cfg.ny, node%self.cfg.ny
                    di, dj = ni-i, nj-j
                    q = ti.cast(self.state.q[ni, nj], ti.f64)-ti.cast(self.state.comp[ni, nj], ti.f64)
                    volume = self.geo.capacity[ni, nj]
                    total, comp = add_compensated(total, comp, q, ti.f64)
                    increment = ti.Vector([volume, volume*di, volume*dj, volume*(di*di+dj*dj),
                                           volume*self.flow.pressure[ni, nj], di*q[5]-dj*q[4]])
                    moments, mc = add_compensated(moments, mc, increment, ti.f64)
                total, moments = total-comp, moments-mc
                v = ti.max(moments[0], 1e-30)
                center = ti.Vector([moments[1], moments[2]])/v
                density = (total[0]+ti.static(self.cfg.rho_a_lu)*total[1])/v
                inertia = density*(moments[3]-v*center.norm_sqr())
                angular = moments[5]-cross(center, ti.Vector([total[4], total[5]]))
                omega = 0.0
                if inertia > 1e-20: omega = angular/inertia
                self.common[i, j] = total/v
                self.spin[i, j] = ti.Vector([center.x, center.y, omega])
                self.mean_pressure[i, j] = moments[4]/v

    @ti.kernel
    def propose(self):
        for i, j in self.owner:
            delta = ti.Vector.zero(ti.f32, 6)
            target = self.owner[i, j]
            if self.active[i, j] and (target != ti.Vector([i, j])).any():
                common, spin = self.common[target.x, target.y], self.spin[target.x, target.y]
                accepted = self.geo.capacity[i, j]*common
                r = ti.Vector([i-target.x-spin.x, j-target.y-spin.y])
                mass = accepted[0]+ti.static(self.cfg.rho_a_lu)*accepted[1]
                accepted[4] -= mass*spin.z*r.y
                accepted[5] += mass*spin.z*r.x
                delta = accepted-self.state.value(i, j)
            self.transfer[i, j] = delta

    @ti.kernel
    def commit(self):
        for i, j in self.owner:
            target = self.owner[i, j]
            delta = self.transfer[i, j]
            self.state.heat_position[i, j] = ti.Vector([i+.5, j+.5])
            if self.active[i, j]:
                spin = self.spin[target.x, target.y]
                self.state.heat_position[i, j] = ti.Vector([target.x+.5+spin.x, target.y+.5+spin.y])
                if (target == ti.Vector([i, j])).all():
                    total, comp = ti.Vector.zero(ti.f32, 6), ti.Vector.zero(ti.f32, 6)
                    root = self.root[i, j]
                    start = self.member_start(root)
                    for k in range(self.member_count[root]):
                        node = self.members[start+k]
                        ni, nj = node//self.cfg.ny, node%self.cfg.ny
                        total, comp = add_compensated(total, comp, self.transfer[ni, nj])
                    delta = -(total-comp)
                pressure = self.mean_pressure[target.x, target.y]
                dp = pressure-self.flow.pressure[i, j]
                for q in range(9):
                    change = 3*weight(q)*dp
                    if q == 0: change -= 3*dp
                    self.flow.f[q, i, j] += change
                self.flow.pressure[i, j] = pressure
            self.state.add(i, j, delta)
            self.activity[i, j] = ti.abs(delta[0])

    def step(self):
        self.prepare_components()
        self.components.seed(self.active, 0.0); self.components.connect(self.edges, False)
        self.components.roots(self.root); self.measure_components()
        self.choose_full_receivers(); self.join_small_components()
        self.choose_orphan_receivers(); self.attach_full_receivers()
        self.components.roots(self.root); self.mark_active_components()
        self.count_members(); self.prefix_groups(); self.prefix_bases(); self.list_members()
        self.choose_leaders(); self.assign()
        self.gather(); self.propose(); self.commit()
