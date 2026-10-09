"""D2Q5 bulk conduction for the moving-body thermal solver.

As in iceflow2d_v2.thermal, tau is fixed at one and advection is split from
conduction. With Fo = alpha * dt / dx**2, the equilibrium weights are
(1 - 4*Fo, Fo, Fo, Fo, Fo); equivalently varpi = 4*Fo in v2's thermal_post.
Each material has its own diffusivity and the actual diffusion substep sets
its weights. The existing Fourier bound keeps 4*Fo <= 2/3.

Populations carry extensive sensible energy (J per metre of depth). Their
zeroth moment is the water/body energy; the moving moments carry T - T_m.
Shared faces transmit only the minimum of the two material apertures and
reflect the remainder. This is v2's paired population flux closure expressed
as a pull stream, and reduces to ordinary D2Q5 streaming in full cells.
Walls use bounce-back (adiabatic) or anti-bounce-back (Dirichlet). Ice/water
interfaces are reflected here and exchanged by the parent's contact solver.

ALE remapping, advection and melting remain authoritative extensive-energy
updates. Synchronization puts their energy increments in the resting
population and clears cells whose material disappeared. Latent heat remains
in the parent's mass-removal law, consistent with v3's disappearing melt.
"""

import taichi as ti

from .config import _ADIABATIC, _DIRICHLET

_DIRECTIONS = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
_OPPOSITES = (0, 3, 4, 1, 2)


@ti.func
def _equilibrium(energy, moving):
    # tau=1: both flux moments and the two higher moments fully relax.
    return ti.Vector([energy - 4.0 * moving, moving, moving, moving, moving])


@ti.func
def _with_energy(populations, energy):
    populations[0] = energy - (
        populations[1] + populations[2] + populations[3] + populations[4]
    )
    return populations


@ti.func
def _face_transfer(outgoing, incoming, aperture, neighbor_aperture):
    shared = ti.min(aperture, neighbor_aperture)
    return shared / neighbor_aperture * incoming - shared / aperture * outgoing


@ti.data_oriented
class D2Q5Thermal2D:
    """Optional distribution storage; allocated only for enthalpy_d2q5."""

    def __init__(self, thermal):
        self.thermal = thermal
        world_shape = (thermal.nx, thermal.ny)
        body_shape = (thermal.body_nx, thermal.body_ny)
        self.water_populations = ti.Vector.field(5, ti.f64, shape=world_shape)
        self.body_populations = ti.Vector.field(5, ti.f64, shape=body_shape)
        self._water_post = ti.Vector.field(5, ti.f64, shape=world_shape)
        self._body_post = ti.Vector.field(5, ti.f64, shape=body_shape)
        # A pull from q=1 sees the left wall, q=2 the bottom, etc.
        self._boundary_kinds = (
            _ADIABATIC,
            thermal._left_kind,
            thermal._bottom_kind,
            thermal._right_kind,
            thermal._top_kind,
        )
        self._boundary_temperatures = (
            0.0,
            thermal._left_value,
            thermal._bottom_value,
            thermal._right_value,
            thermal._top_value,
        )

    def initialize(self):
        self.water_populations.fill(0.0)
        self.body_populations.fill(0.0)
        self.synchronize()

    @ti.func
    def _water_aperture(self, i, j):
        return ti.min(1.0, self.thermal.water_volume_m2[i, j] / self.thermal._cell_area)

    @ti.func
    def _water_active(
        self,
        i,
        j,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        return (
            wall_mask[i, j] == 0
            and solid_mask[i, j] == 0
            and water_phase[i, j] > self.thermal._water_phase_cutoff
            and self.thermal.water_volume_m2[i, j] > 1.0e-30
        )

    @ti.kernel
    def synchronize(self):
        for i, j in self.water_populations:
            populations = ti.Vector.zero(ti.f64, 5)
            if self.thermal.water_volume_m2[i, j] > 1.0e-30:
                populations = _with_energy(
                    self.water_populations[i, j],
                    self.thermal.water_sensible_energy[i, j],
                )
            self.water_populations[i, j] = populations
        for i, j in self.body_populations:
            populations = ti.Vector.zero(ti.f64, 5)
            if self.thermal.body_solid_mass[i, j] > 1.0e-30:
                populations = _with_energy(
                    self.body_populations[i, j],
                    self.thermal.body_sensible_energy[i, j],
                )
            self.body_populations[i, j] = populations

    @ti.kernel
    def collide(self, time_step_s: ti.f64):
        # Reconstruct m0 from the extensive state, including split sources.
        # At tau=1 the old non-equilibrium flux has zero post-collision weight.
        for i, j in self.water_populations:
            populations = ti.Vector.zero(ti.f64, 5)
            if self.thermal.water_volume_m2[i, j] > 1.0e-30:
                moving = (
                    self.thermal._water_conductivity
                    * time_step_s
                    * self._water_aperture(i, j)
                    * (
                        self.thermal._water_temperature_at(i, j)
                        - self.thermal._melting_temperature_c
                    )
                )
                populations = _equilibrium(
                    self.thermal.water_sensible_energy[i, j], moving
                )
            self._water_post[i, j] = populations
        for i, j in self.body_populations:
            populations = ti.Vector.zero(ti.f64, 5)
            if self.thermal.body_solid_mass[i, j] > 1.0e-30:
                moving = (
                    self.thermal._ice_conductivity
                    * time_step_s
                    * ti.cast(self.thermal._body_fraction_at(i, j), ti.f64)
                    * (
                        self.thermal._body_temperature_at(i, j)
                        - self.thermal._melting_temperature_c
                    )
                )
                populations = _equilibrium(
                    self.thermal.body_sensible_energy[i, j], moving
                )
            self._body_post[i, j] = populations

    @ti.kernel
    def stream(
        self,
        time_step_s: ti.f64,
        water_phase: ti.template(),
        wall_mask: ti.template(),
        solid_mask: ti.template(),
    ):
        for i, j in self.water_populations:
            incoming = ti.Vector.zero(ti.f64, 5)
            if self._water_active(i, j, water_phase, wall_mask, solid_mask):
                aperture = self._water_aperture(i, j)
                energy_delta = ti.cast(0.0, ti.f64)
                for q in ti.static(range(1, 5)):
                    ni = i - ti.static(_DIRECTIONS[q][0])
                    nj = j - ti.static(_DIRECTIONS[q][1])
                    outgoing = self._water_post[i, j][ti.static(_OPPOSITES[q])]
                    transfer = ti.cast(0.0, ti.f64)
                    boundary = True
                    if 0 <= ni < self.thermal.nx and 0 <= nj < self.thermal.ny:
                        boundary = wall_mask[ni, nj] != 0
                        if self._water_active(
                            ni, nj, water_phase, wall_mask, solid_mask
                        ):
                            transfer = _face_transfer(
                                outgoing,
                                self._water_post[ni, nj][q],
                                aperture,
                                self._water_aperture(ni, nj),
                            )
                    if ti.static(self._boundary_kinds[q] == _DIRICHLET):
                        if boundary:
                            wall_population = (
                                self.thermal._water_conductivity
                                * time_step_s
                                * aperture
                                * (
                                    ti.static(self._boundary_temperatures[q])
                                    - self.thermal._melting_temperature_c
                                )
                            )
                            transfer = 2.0 * (wall_population - outgoing)
                    incoming[q] = outgoing + transfer
                    energy_delta += transfer
                energy = self.thermal.water_sensible_energy[i, j] + energy_delta
                self.thermal.water_sensible_energy[i, j] = energy
                incoming = _with_energy(incoming, energy)
            self.water_populations[i, j] = incoming

        for i, j in self.body_populations:
            incoming = ti.Vector.zero(ti.f64, 5)
            energy_delta = ti.cast(0.0, ti.f64)
            aperture = ti.cast(self.thermal._body_fraction_at(i, j), ti.f64)
            if aperture > 0.0:
                for q in ti.static(range(1, 5)):
                    ni = i - ti.static(_DIRECTIONS[q][0])
                    nj = j - ti.static(_DIRECTIONS[q][1])
                    outgoing = self._body_post[i, j][ti.static(_OPPOSITES[q])]
                    transfer = ti.cast(0.0, ti.f64)
                    if (
                        0 <= ni < self.thermal.body_nx
                        and 0 <= nj < self.thermal.body_ny
                    ):
                        neighbor_aperture = ti.cast(
                            self.thermal._body_fraction_at(ni, nj), ti.f64
                        )
                        if neighbor_aperture > 0.0:
                            transfer = _face_transfer(
                                outgoing,
                                self._body_post[ni, nj][q],
                                aperture,
                                neighbor_aperture,
                            )
                    incoming[q] = outgoing + transfer
                    energy_delta += transfer
                incoming = _with_energy(
                    incoming, self.thermal.body_sensible_energy[i, j] + energy_delta
                )
            self.body_populations[i, j] = incoming
            # Parent adds contact heat before its one phase-change commit.
            self.thermal._body_heat_delta[i, j] = energy_delta
