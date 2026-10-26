"""Specialized kernels for the open-tank example; shared HOME is unchanged."""

# ruff: noqa: E741 -- lattice indices
import taichi as ti

from ..home import HomeMomentum, pack, reconstruct
from ..lattice import Q, direction, mixture_kinematic_viscosity, weight


@ti.data_oriented
class OpenTankFlow:
    _mixture_kinematic_viscosity = HomeMomentum._kinematic_viscosity

    def _forces(self, with_flux, moment_pressure=False, classified=False):
        if not classified:
            self._classify_home()
        self._forces_bulk(with_flux, moment_pressure)
        # This geometry uses flags -1, 0 and 2. The shared rectangular-tank
        # kernel (flag 1) is empty, so do not scan the whole grid for it.
        self._forces_boundary(with_flux, moment_pressure)

    @ti.func
    def _kinematic_viscosity(self, I):
        # The existing thermal model uses the prescribed reservoir temperature
        # wherever its water volume is zero, including tiny diffuse tails.
        # Expose that constant to the compiler so the four water-viscosity
        # powers are folded, while retaining the exact phase mixing law.
        nu = 0.0
        if self.water_volume[I] > 1e-30:
            nu = self._mixture_kinematic_viscosity(I)
        else:
            nu = (
                mixture_kinematic_viscosity(
                    self.water_phase[I],
                    ti.cast(self.config.thermal.initial_air_temperature_c, ti.f32),
                    self.config.rho_water,
                    self.config.rho_air,
                    self.mu_air,
                )
                * self.scales.dt_s
                / self.scales.dx_m**2
            )
        return nu

    @ti.kernel
    def _stream_home_bulk(self):
        ti.loop_config(block_dim=128)
        for I in ti.grouped(self.water_phase):
            flag = self._home_boundary[I]
            if flag == 0:
                # Classification guarantees that all 27 donors are fluid.
                # Avoid rereading their masks and carrying a bounce-back path
                # in the dense bulk kernel. Reconstruction and moments retain
                # the same D3Q27 arithmetic as the shared implementation.
                p = 0.0
                j = ti.Vector.zero(ti.f32, 3)
                S = ti.Matrix.zero(ti.f32, 3, 3)
                for q in ti.static(range(Q)):
                    c = direction(q)
                    cf = ti.cast(c, ti.f32)
                    value = reconstruct(q, self.home_post[I - c])
                    p += value
                    j += cf * value
                    S += (
                        cf.outer_product(cf) - ti.Matrix.identity(ti.f32, 3) / 3
                    ) * value
                self.home[I] = pack(p, j, S)
                self.velocity_lattice[I] = j + 0.5 * self.home_acceleration[I]
            elif flag < 0:
                self.velocity_lattice[I] = ti.Vector.zero(ti.f32, 3)
                if self.solid_body_id[I] >= 0:
                    self.velocity_lattice[I] = self._body_velocity_at(
                        ti.cast(I, ti.f32) + 0.5, self.solid_body_id[I]
                    )

    def _stream_home_boundary(self):
        self._stream_home_fixed_open()
        self._stream_home_moving()

    @ti.kernel
    def _stream_home_fixed_open(self):
        ti.loop_config(block_dim=128)
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 2 and self._home_moving_neighbor[I] == 0:
                p = 0.0
                j = ti.Vector.zero(ti.f32, 3)
                S = ti.Matrix.zero(ti.f32, 3, 3)
                m = self.home_post[I]
                u = self.velocity_lattice[I]
                for q in ti.static(range(Q)):
                    c = direction(q)
                    cf = ti.cast(c, ti.f32)
                    J = I - c
                    value = 0.0
                    if self._fluid(J):
                        value = reconstruct(q, self.home_post[J])
                    else:
                        raw = reconstruct(26 - q, m)
                        value = raw
                        if self._open(J):
                            # The odd first/third Hermite terms cancel in
                            # eq(q,0,u)+eq(-q,0,u). This is the same pressure
                            # anti-bounce-back as the generic boundary path.
                            value = (
                                9 * weight(q) * (cf.dot(u) ** 2 - u.dot(u) / 3) - raw
                            )
                            self.open_volume_flux_cell[I] += raw - value
                    p += value
                    j += cf * value
                    S += (
                        cf.outer_product(cf) - ti.Matrix.identity(ti.f32, 3) / 3
                    ) * value
                self.home[I] = pack(p, j, S)
                self.velocity_lattice[I] = j + 0.5 * self.home_acceleration[I]

    @ti.kernel
    def _stream_home_moving(self):
        for I in ti.grouped(self.water_phase):
            if self._home_boundary[I] == 2 and self._home_moving_neighbor[I] != 0:
                self.old_velocity[I] = self.velocity_lattice[I]
                # Retain the full mirror geometry and momentum exchange at
                # moving dry ice, including any mixed fixed/open links.
                p = 0.0
                j = ti.Vector.zero(ti.f32, 3)
                S = ti.Matrix.zero(ti.f32, 3, 3)
                for q in range(Q):
                    c = direction(q)
                    cf = ti.cast(c, ti.f32)
                    J = I - c
                    value = 0.0
                    if self._fluid(J):
                        value = reconstruct(q, self.home_post[J])
                    else:
                        value = self._home_reflection(I, 26 - q)
                    p += value
                    j += cf * value
                    S += (
                        cf.outer_product(cf) - ti.Matrix.identity(ti.f32, 3) / 3
                    ) * value
                self.home[I] = pack(p, j, S)
                self.velocity_lattice[I] = j + 0.5 * self.home_acceleration[I]
