"""Independent D2Q5 enthalpy/temperature solver derived from moment constraints.

No ProjectIcy modules are imported. GPU state and arithmetic use Taichi f32.
The resting population stores latent/sensible energy; moving populations
transport temperature. Advection is a conservative, split MUSCL-SSPRK2 step.
See derivation_zh.md for the relationship to Wang et al. (2022).
"""

from dataclasses import dataclass

import numpy as np
import taichi as ti


PERIODIC, ADIABATIC, DIRICHLET = 0, 1, 2


@dataclass
class ThermalParameters:
    c_s: float = 0.5
    c_l: float = 1.0
    latent: float = 5.0
    k_s: float = 0.15
    k_l: float = 0.05
    weight: float = 0.4


@ti.data_oriented
class D2Q5:
    """Cell-centred lattice, dx=dt=1; boundaries lie half a link outside cells.

    Array axes are (x,y). Heat capacity and conductivity are dimensionless;
    k is conductivity divided by the reference volumetric heat capacity.
    bc order is left, right, bottom, top. All populations have energy units.
    """

    def __init__(self, nx, ny, params=None, bc=(PERIODIC,) * 4):
        self.nx, self.ny = int(nx), int(ny)
        self.params = params or ThermalParameters()
        p = self.params
        if not (0.0 < p.weight < min(p.c_s, p.c_l, 1.0)):
            raise ValueError("Use 0 < weight < min(C_s, C_l, 1) for positive storage.")
        if min(p.k_s, p.k_l) <= 0 or p.latent < 0:
            raise ValueError("Conductivities must be positive; latent heat nonnegative.")
        self.w = float(p.weight)
        self.cs2 = self.w / 2.0
        self.bc = tuple(int(b) for b in bc)
        if (self.bc[0] == PERIODIC) != (self.bc[1] == PERIODIC):
            raise ValueError("Both x boundaries must be periodic together.")
        if (self.bc[2] == PERIODIC) != (self.bc[3] == PERIODIC):
            raise ValueError("Both y boundaries must be periodic together.")
        shape = (self.nx, self.ny)
        self.h = ti.field(ti.f32, shape)
        self.h_comp = ti.field(ti.f32, shape)
        self.temperature = ti.field(ti.f32, shape)
        self.liquid = ti.field(ti.f32, shape)
        self.c_s = ti.field(ti.f32, shape)
        self.c_l = ti.field(ti.f32, shape)
        self.latent = ti.field(ti.f32, shape)
        self.k_s = ti.field(ti.f32, shape)
        self.k_l = ti.field(ti.f32, shape)
        self.g = ti.Vector.field(5, ti.f32, shape)
        self.post = ti.Vector.field(5, ti.f32, shape)
        self.boundary_energy = ti.field(ti.f32, shape)
        self.boundary_comp = ti.field(ti.f32, shape)
        self.h_stage = ti.field(ti.f32, shape)
        self.rhs_first = ti.field(ti.f32, shape)
        self.fx = ti.field(ti.f32, (self.nx + 1, self.ny))
        self.fy = ti.field(ti.f32, (self.nx, self.ny + 1))
        self.ux_face = ti.field(ti.f32, (self.nx + 1, self.ny))
        self.uy_face = ti.field(ti.f32, (self.nx, self.ny + 1))
        for field, value in ((self.c_s, p.c_s), (self.c_l, p.c_l),
                             (self.latent, p.latent), (self.k_s, p.k_s),
                             (self.k_l, p.k_l)):
            field.fill(value)

    @ti.func
    def state(self, enthalpy, cs, cl, latent):
        temp, fraction = 0.0, 0.0
        if enthalpy < 0.0:
            temp = enthalpy / cs
        elif enthalpy > latent:
            temp, fraction = (enthalpy - latent) / cl, 1.0
        elif latent > 0.0:
            fraction = enthalpy / latent
        else:
            fraction = 1.0
        return temp, fraction

    @ti.func
    def add_energy(self, i, j, increment):
        # Compensated f32 addition avoids losing small heat increments near H=L.
        # Initialize Taichi with fast_math=False to preserve these operations.
        corrected = increment - self.h_comp[i, j]
        total = self.h[i, j] + corrected
        self.h_comp[i, j] = (total - self.h[i, j]) - corrected
        self.h[i, j] = total

    @ti.func
    def update_state(self, i, j):
        temp, frac = self.state(self.h[i, j], self.c_s[i, j],
                                self.c_l[i, j], self.latent[i, j])
        self.temperature[i, j], self.liquid[i, j] = temp, frac

    @ti.kernel
    def initialize_populations(self):
        for i, j in self.h:
            self.h_comp[i, j] = 0.0
            self.boundary_energy[i, j], self.boundary_comp[i, j] = 0.0, 0.0
            self.update_state(i, j)
            moving = self.w * self.temperature[i, j] / 4.0
            self.g[i, j] = ti.Vector([self.h[i, j] - 4.0 * moving,
                                      moving, moving, moving, moving])

    def initialize(self, enthalpy):
        value = np.broadcast_to(np.asarray(enthalpy, dtype=np.float32),
                                (self.nx, self.ny)).copy()
        self.h.from_numpy(value)
        self.initialize_populations()

    @ti.kernel
    def collide(self):
        for i, j in self.h:
            g = self.g[i, j]
            fraction = self.liquid[i, j]
            conductivity = ((1.0 - fraction) * self.k_s[i, j]
                            + fraction * self.k_l[i, j])
            relaxation = 1.0 / (0.5 + conductivity / self.cs2)
            mx = (1.0 - relaxation) * (g[1] - g[3])
            my = (1.0 - relaxation) * (g[2] - g[4])
            m3 = self.w * self.temperature[i, j]
            # s_3=s_4=1; conserved m_0 is H, not temperature.
            self.post[i, j] = ti.Vector([
                self.h[i, j] - m3,
                0.25 * m3 + 0.5 * mx,
                0.25 * m3 + 0.5 * my,
                0.25 * m3 - 0.5 * mx,
                0.25 * m3 - 0.5 * my,
            ])

    @ti.kernel
    def stream(self, left: ti.f32, right: ti.f32,
               bottom: ti.f32, top: ti.f32):
        for i, j in self.h:
            incoming = ti.Vector.zero(ti.f32, 5)
            increment, boundary = 0.0, 0.0
            for q in ti.static(range(1, 5)):
                ex = ti.static((0, 1, 0, -1, 0)[q])
                ey = ti.static((0, 0, 1, 0, -1)[q])
                opposite = ti.static((0, 3, 4, 1, 2)[q])
                x, y = i - ex, j - ey
                bc_type, wall_t, outside = -1, 0.0, False
                if x < 0:
                    bc_type, wall_t, outside = self.bc[0], left, True
                elif x >= self.nx:
                    bc_type, wall_t, outside = self.bc[1], right, True
                elif y < 0:
                    bc_type, wall_t, outside = self.bc[2], bottom, True
                elif y >= self.ny:
                    bc_type, wall_t, outside = self.bc[3], top, True
                value = 0.0
                if not outside or bc_type == PERIODIC:
                    value = self.post[(x + self.nx) % self.nx,
                                      (y + self.ny) % self.ny][q]
                else:
                    outgoing = self.post[i, j][opposite]
                    value = outgoing
                    if bc_type == DIRICHLET:
                        value = 0.5 * self.w * wall_t - outgoing
                    boundary += value - outgoing
                incoming[q] = value
                increment += value - self.post[i, j][q]
            self.add_energy(i, j, increment)
            incoming[0] = self.h[i, j] - (incoming[1] + incoming[2]
                                                   + incoming[3] + incoming[4])
            self.g[i, j] = incoming
            adjusted = boundary - self.boundary_comp[i, j]
            new_boundary = self.boundary_energy[i, j] + adjusted
            self.boundary_comp[i, j] = (new_boundary - self.boundary_energy[i, j]) - adjusted
            self.boundary_energy[i, j] = new_boundary
            self.update_state(i, j)

    def diffuse(self, walls=(0.0, 0.0, 0.0, 0.0)):
        self.collide()
        self.stream(*map(float, walls))

    @ti.kernel
    def add_uniform_source(self, increment: ti.f32):
        for i, j in self.h:
            self.add_energy(i, j, increment)
            self.g[i, j][0] = self.h[i, j] - (self.g[i, j][1] + self.g[i, j][2]
                                             + self.g[i, j][3] + self.g[i, j][4])
            self.update_state(i, j)

    @ti.kernel
    def add_source_field(self, source: ti.template(), factor: ti.f32):
        for i, j in self.h:
            self.add_energy(i, j, factor * source[i, j])
            self.g[i, j][0] = self.h[i, j] - (self.g[i, j][1] + self.g[i, j][2]
                                             + self.g[i, j][3] + self.g[i, j][4])
            self.update_state(i, j)

    @ti.kernel
    def set_rigid_velocity(self, vx: ti.f32, vy: ti.f32, omega: ti.f32,
                           cx: ti.f32, cy: ti.f32):
        for i, j in self.ux_face:
            self.ux_face[i, j] = vx - omega * (j + 0.5 - cy)
        for i, j in self.uy_face:
            self.uy_face[i, j] = vy + omega * (i + 0.5 - cx)

    @ti.kernel
    def set_velocity_from_cells(self, velocity: ti.template()):
        for i, j in self.ux_face:
            value = 0.5 * (velocity[(i - 1 + self.nx) % self.nx, j][0]
                           + velocity[i % self.nx, j][0])
            if ti.static(self.bc[0] != PERIODIC):
                if i == 0 or i == self.nx:
                    value = 0.0
            self.ux_face[i, j] = value
        for i, j in self.uy_face:
            value = 0.5 * (velocity[i, (j - 1 + self.ny) % self.ny][1]
                           + velocity[i, j % self.ny][1])
            if ti.static(self.bc[2] != PERIODIC):
                if j == 0 or j == self.ny:
                    value = 0.0
            self.uy_face[i, j] = value

    @ti.func
    def sample(self, field: ti.template(), i, j):
        x, y = i, j
        if ti.static(self.bc[0] == PERIODIC):
            x = (x + self.nx) % self.nx
        else:
            x = ti.max(0, ti.min(self.nx - 1, x))
        if ti.static(self.bc[2] == PERIODIC):
            y = (y + self.ny) % self.ny
        else:
            y = ti.max(0, ti.min(self.ny - 1, y))
        return field[x, y]

    @ti.func
    def mc_slope(self, a, b, c):
        dl, dr = b - a, c - b
        slope = 0.0
        if dl * dr > 0.0:
            slope = ti.min(2.0 * ti.abs(dl), 2.0 * ti.abs(dr),
                           0.5 * ti.abs(dl + dr))
            if dl < 0.0:
                slope = -slope
        return slope

    @ti.kernel
    def advective_fluxes(self, field: ti.template()):
        for i, j in self.fx:
            hm = self.sample(field, i - 1, j)
            hp = self.sample(field, i, j)
            sm = self.mc_slope(self.sample(field, i - 2, j), hm, hp)
            sp = self.mc_slope(hm, hp, self.sample(field, i + 1, j))
            vel = self.ux_face[i, j]
            transported = hp - 0.5 * sp
            if vel >= 0.0:
                transported = hm + 0.5 * sm
            value = vel * transported
            if ti.static(self.bc[0] != PERIODIC):
                if i == 0 or i == self.nx:
                    value = 0.0
            self.fx[i, j] = value
        for i, j in self.fy:
            hm = self.sample(field, i, j - 1)
            hp = self.sample(field, i, j)
            sm = self.mc_slope(self.sample(field, i, j - 2), hm, hp)
            sp = self.mc_slope(hm, hp, self.sample(field, i, j + 1))
            vel = self.uy_face[i, j]
            transported = hp - 0.5 * sp
            if vel >= 0.0:
                transported = hm + 0.5 * sm
            value = vel * transported
            if ti.static(self.bc[2] != PERIODIC):
                if j == 0 or j == self.ny:
                    value = 0.0
            self.fy[i, j] = value

    @ti.kernel
    def advective_stage_one(self, dt: ti.f32):
        for i, j in self.h:
            rhs = -(self.fx[i + 1, j] - self.fx[i, j]
                    + self.fy[i, j + 1] - self.fy[i, j])
            self.rhs_first[i, j] = rhs
            self.h_stage[i, j] = self.h[i, j] + dt * rhs

    @ti.kernel
    def advective_stage_two(self, dt: ti.f32):
        for i, j in self.h:
            rhs = -(self.fx[i + 1, j] - self.fx[i, j]
                    + self.fy[i, j + 1] - self.fy[i, j])
            self.add_energy(i, j, 0.5 * dt * (self.rhs_first[i, j] + rhs))
            self.g[i, j][0] = self.h[i, j] - (self.g[i, j][1] + self.g[i, j][2]
                                             + self.g[i, j][3] + self.g[i, j][4])
            self.update_state(i, j)

    def advect(self, dt=1.0):
        self.advective_fluxes(self.h)
        self.advective_stage_one(float(dt))
        self.advective_fluxes(self.h_stage)
        self.advective_stage_two(float(dt))

    def step(self, walls=(0.0, 0.0, 0.0, 0.0), advect=False):
        if advect:
            self.advect(0.5)
        self.diffuse(walls)
        if advect:
            self.advect(0.5)

    def arrays(self):
        return {"enthalpy": self.h.to_numpy(), "temperature": self.temperature.to_numpy(),
                "liquid_fraction": self.liquid.to_numpy()}

    def energy(self):
        return float(np.sum(self.h.to_numpy(), dtype=np.float64))

    def boundary_heat(self):
        return float(np.sum(self.boundary_energy.to_numpy(), dtype=np.float64))


def enthalpy_from_state(temperature, liquid_fraction, c_s, c_l, latent):
    """Physical initialization; mixed cells are isothermal at T_m=0."""
    t, f = np.asarray(temperature), np.asarray(liquid_fraction)
    return ((1.0 - f) * c_s + f * c_l) * t + latent * f
