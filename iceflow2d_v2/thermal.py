"""Material-separated D2Q5 conduction with paired extensive cut-cell fluxes.

The present cut-cell closure deliberately fixes tau=1. Generic MRT algebra
lives in lattice.thermal_post; changing tau needs a new flux validation.
"""
import taichi as ti
from .lattice import direction, opposite, thermal_post, add_compensated
from .polygon import segment_interval


@ti.data_oriented
class Thermal:
    def __init__(self, config, geometry, state):
        self.cfg, self.geo, self.state = config, geometry, state
        self.gw = ti.Vector.field(5, ti.f32, (config.nx, config.ny))
        self.ga = ti.Vector.field(5, ti.f32, (config.nx, config.ny))
        self.gs = ti.Vector.field(5, ti.f32, (config.nb, config.nb))
        self.pw = ti.Vector.field(5, ti.f32, (config.nx, config.ny))
        self.pa = ti.Vector.field(5, ti.f32, (config.nx, config.ny))
        self.ps = ti.Vector.field(5, ti.f32, (config.nb, config.nb))
        self.flux_w = ti.Vector.field(2, ti.f32, (config.nx, config.ny))
        self.flux_a = ti.Vector.field(2, ti.f32, (config.nx, config.ny))
        self.flux_s = ti.Vector.field(2, ti.f32, (config.nb, config.nb))
        self.limited = ti.field(ti.f32, (config.nx, config.ny))
        self.block_position = ti.field(ti.i32, ())
        self.reconstruct()

    @ti.kernel
    def begin(self):
        self.block_position[None] = (self.block_position[None]+1) % self.cfg.coupling_steps

    @ti.kernel
    def reconstruct(self):
        for i, j in self.gw:
            value = self.state.value(i, j)
            tw, ta = self.state.temperature_w[i, j], self.state.temperature_a[i, j]
            hw, ha = 0.0, 0.0
            if value[0] > 1e-20: hw = value[2]/value[0]+ti.static(self.cfg.latent_lu)
            if value[1] > 1e-20: ha = value[3]/(ti.static(self.cfg.air_capacity_lu)*value[1])
            self.gw[i, j] = thermal_post(hw, tw, self.gw[i, j], ti.static(self.cfg.thermal_weight('water')), 1.0)
            self.ga[i, j] = thermal_post(ha, ta-ti.static(self.cfg.air_temperature/self.cfg.temperature_scale), self.ga[i, j], ti.static(self.cfg.thermal_weight('air')), 1.0)
        for a, b in self.gs:
            heat = self.geo.ice_heat[a, b]-self.geo.ice_heat_comp[a, b]
            fraction = self.geo.material_fraction(a, b)
            ts = 0.0
            if fraction > 1e-20: ts = heat/(ti.static(self.cfg.ice_capacity_lu)*fraction)
            self.geo.ice_temperature[a, b] = ts
            self.gs[a, b] = thermal_post(ts, ts, self.gs[a, b], ti.static(self.cfg.thermal_weight('ice')), 1.0)

    @ti.kernel
    def collide(self):
        for i, j in self.gw:
            self.pw[i, j] = thermal_post(self.gw[i, j].sum(), self.state.temperature_w[i, j], self.gw[i, j], ti.static(self.cfg.thermal_weight('water')), 1.0)
            self.pa[i, j] = thermal_post(self.ga[i, j].sum(), self.state.temperature_a[i, j]-ti.static(self.cfg.air_temperature/self.cfg.temperature_scale), self.ga[i, j], ti.static(self.cfg.thermal_weight('air')), 1.0)
        for a, b in self.gs:
            self.ps[a, b] = thermal_post(self.gs[a, b].sum(), self.geo.ice_temperature[a, b], self.gs[a, b], ti.static(self.cfg.thermal_weight('ice')), 1.0)

    @ti.func
    def capacity_limit(self, candidate, left_capacity, right_capacity, left_t, right_t):
        # At most 1/4 of the two-cell equilibration heat per face. Each cell
        # has <=4 faces, so positivity survives a tiny cut cell without /V blow-up.
        reduced = left_capacity*right_capacity/ti.max(left_capacity+right_capacity, 1e-30)
        bound = 0.25*reduced*ti.abs(left_t-right_t)
        return ti.math.sign(candidate)*ti.min(ti.abs(candidate), bound)

    @ti.kernel
    def build_fluxes(self):
        for i, j in self.gw:
            fw, fa, limited = ti.Vector.zero(ti.f32, 2), ti.Vector.zero(ti.f32, 2), 0.0
            for axis in ti.static(range(2)):
                ni, nj = i+1-axis, j+axis
                if self.block_position[None] == 0 and self.geo.inside(ni, nj) and not self.geo.wall(i, j) and not self.geo.wall(ni, nj):
                    left, right = self.state.value(i, j), self.state.value(ni, nj)
                    q, oq = axis+1, axis+3
                    aperture = self.geo.face_open[i, j][axis]
                    for material in ti.static(range(2)):
                        lc, rc = left[material], right[material]
                        capacity = 1.0
                        tl, tr = self.state.temperature_w[i, j], self.state.temperature_w[ni, nj]
                        candidate = self.pw[i, j][q]-self.pw[ni, nj][oq]
                        if ti.static(material == 1):
                            capacity = self.cfg.air_capacity_lu
                            tl, tr = self.state.temperature_a[i, j], self.state.temperature_a[ni, nj]
                            candidate = self.pa[i, j][q]-self.pa[ni, nj][oq]
                        fraction = 0.0
                        if lc > 0.0 and rc > 0.0:
                            fraction = ti.min(lc/ti.max(self.geo.capacity[i, j], 1e-30), rc/ti.max(self.geo.capacity[ni, nj], 1e-30))
                        separation = ti.abs((self.state.heat_position[ni, nj]-self.state.heat_position[i, j])[axis])
                        candidate *= aperture*fraction*capacity/ti.max(.25, separation)
                        value = self.capacity_limit(candidate, capacity*lc, capacity*rc, tl, tr)
                        limited += ti.abs(value-candidate)
                        if ti.static(material == 0): fw[axis] = value
                        else: fa[axis] = value
            self.flux_w[i, j], self.flux_a[i, j], self.limited[i, j] = fw, fa, limited
        for a, b in self.gs:
            flux = ti.Vector.zero(ti.f32, 2)
            for axis in ti.static(range(2)):
                na, nb = a+1-axis, b+axis
                if self.block_position[None] == 0 and na < self.cfg.nb and nb < self.cfg.nb:
                    left, right = self.geo.material_fraction(a, b), self.geo.material_fraction(na, nb)
                    cap = ti.static(self.cfg.ice_capacity_lu)
                    start, end = ti.Vector([.5, -.5]), ti.Vector([.5, .5])
                    shift = ti.Vector([1.0, 0.0])
                    if ti.static(axis == 1):
                        start, shift = ti.Vector([-.5, .5]), ti.Vector([0.0, 1.0])
                    first = segment_interval(self.geo.poly[a, b], self.geo.count[a, b], start, end)
                    second = segment_interval(self.geo.poly[na, nb], self.geo.count[na, nb], start-shift, end-shift)
                    contact = ti.max(0.0, ti.min(first.y, second.y)-ti.max(first.x, second.x))
                    candidate = contact*cap*(self.ps[a, b][axis+1]-self.ps[na, nb][axis+3])
                    flux[axis] = self.capacity_limit(candidate, cap*left, cap*right,
                                                      self.geo.ice_temperature[a, b], self.geo.ice_temperature[na, nb])
            self.flux_s[a, b] = flux

    @ti.kernel
    def commit(self):
        for i, j in self.gw:
            dw, da = -self.flux_w[i, j].sum(), -self.flux_a[i, j].sum()
            if i > 0: dw, da = dw+self.flux_w[i-1, j].x, da+self.flux_a[i-1, j].x
            if j > 0: dw, da = dw+self.flux_w[i, j-1].y, da+self.flux_a[i, j-1].y
            self.state.add(i, j, ti.Vector([0.0, 0.0, dw, da, 0.0, 0.0]))
        for a, b in self.gs:
            delta = -self.flux_s[a, b].sum()
            if a > 0: delta += self.flux_s[a-1, b].x
            if b > 0: delta += self.flux_s[a, b-1].y
            self.geo.ice_heat[a, b], self.geo.ice_heat_comp[a, b] = add_compensated(self.geo.ice_heat[a, b], self.geo.ice_heat_comp[a, b], delta)

    def step(self):
        # Only bulk conduction is blocked. Interfacial heat/mass exchange,
        # geometry, enthalpy advection and temperatures update every fluid step.
        self.begin()
        self.reconstruct()
        self.collide()
        self.build_fluxes()
        self.commit()
        self.state.temperatures()
        self.reconstruct()
