"""Opt-in static internal walls and five pressure openings for HOME/D3Q7 AA.

One exterior halo is populated from the interior. Incoming HOME populations
use pressure anti-bounce-back; velocity is not set to zero.
Phase openings replace only the reflected
incoming AA slot, AFTER collision, so both parities retain their ownership.
"""
# ruff: noqa: E741 -- lattice indices

import numpy as np
import taichi as ti
from ..home import HomeMomentum, equilibrium, reconstruct, pack
from ..lattice import direction
from ..phase import InPlacePhase, phase_direction, phase_equilibrium7
from ..simulator import IceFlow3D


@ti.data_oriented
class OpenTankBoundaries:
    _moving_wall_reflection = HomeMomentum._home_reflection
    _closed_stress_neighbor_velocity = IceFlow3D._stress_neighbor_velocity

    @ti.func
    def _stress_neighbor_velocity(self, I, c):
        value = self.velocity_lattice[I]
        if not self._open(I + c):
            value = self._closed_stress_neighbor_velocity(I, c)
        return value

    @ti.func
    def _fixed_solid(self, I):
        wall = I.y < 1
        lo = ti.Vector(ti.static(self.config.tank_lower_cells))
        hi = ti.Vector(ti.static(self.config.tank_upper_cells))
        t = ti.static(self.config.tank_wall_cells)
        p = ti.cast(I, ti.f32) + 0.5
        if (p >= lo).all() and (p < hi).all():
            wall = (
                p.y < lo.y + t
                or p.x < lo.x + t
                or p.x >= hi.x - t
                or p.z < lo.z + t
                or p.z >= hi.z - t
            )
        return wall

    @ti.func
    def _open(self, I):
        return I.y >= 1 and (
            I.x < 1
            or I.x >= self.nx - 1
            or I.y >= self.ny - 1
            or I.z < 1
            or I.z >= self.nz - 1
        )

    @ti.kernel
    def _rasterize(self):
        self._gradient_geometry_lower[None] = ti.Vector(self.shape)
        self._gradient_geometry_upper[None] = ti.Vector([-1, -1, -1])
        for I in ti.grouped(self.water_phase):
            self.previous_body_id[I] = self.solid_body_id[I]
            fraction, owner, new_id = 0.0, -1, -1
            if self._fixed_solid(I):
                new_id = -2
            elif self._open(I):
                new_id = -4
            else:
                for b in range(self.body_count):
                    f = self._fraction(ti.cast(I, ti.f32) + 0.5, b)
                    if f > fraction:
                        fraction, owner = f, b
                if fraction >= self.config.thermal.solid_liquid_threshold:
                    new_id = owner
            if fraction != self.coverage[I] or new_id != self.solid_body_id[I]:
                ti.atomic_min(self._gradient_geometry_lower[None], I)
                ti.atomic_max(self._gradient_geometry_upper[None], I)
            self.coverage[I], self.coverage_body_id[I] = fraction, owner
            self.solid_body_id[I] = new_id

    @ti.kernel
    def _initialize_phase(self):
        for I in ti.grouped(self.water_phase):
            # The tank never moves. Cache the six axis reflections once;
            # diagonal mirrors still check the current fluid/solid mask.
            faces = 0
            for a, side in ti.static(ti.ndrange(3, 2)):
                e = ti.Vector.zero(ti.i32, 3)
                e[a] = 2 * side - 1
                if self._fixed_solid(I + e):
                    faces |= 1 << (2 * a + side)
            self._fixed_mirror_faces[I] = ti.cast(faces, ti.u8)
            phi = 0.0
            lo, hi = (
                ti.Vector(ti.static(self.config.tank_lower_cells)),
                ti.Vector(ti.static(self.config.tank_upper_cells)),
            )
            t = ti.static(self.config.tank_wall_cells)
            p = ti.cast(I, ti.f32) + 0.5
            if (
                self._fluid(I)
                and lo.x + t <= p.x < hi.x - t
                and lo.z + t <= p.z < hi.z - t
                and p.y >= lo.y + t
            ):
                phi = 0.5 + 0.5 * ti.tanh(
                    2 * (self.config.water_surface_cells - p.y) / 5.0
                )
            self.water_phase[I] = phi
            for q in ti.static(range(7)):
                self.h[I, q] = phase_equilibrium7(q, phi, ti.Vector.zero(ti.f32, 3))

    def _build_reference(self):
        # Ambient hydrostatics everywhere; only the water inside the tank has
        # excess head. Gauge zero is the ambient pressure at the top opening.
        rho = np.full(self.ny, self.rho_air, np.float32)
        y = np.arange(self.ny) + 0.5
        pref = -self.gravity[1] * self.rho_air * (self.ny - 1 - y)
        ids, phi = self.solid_body_id.to_numpy(), self.water_phase.to_numpy()
        excess = np.where(ids == -1, phi * (1 - self.rho_air), 0.0)
        # A submerged solid interrupts a vertical fluid column, but does not
        # remove the hydrostatic head imposed by the connected free surface.
        # Continue the surrounding water profile through immersed solid cells.
        lo, hi, t = (
            self.config.tank_lower_cells,
            self.config.tank_upper_cells,
            self.config.tank_wall_cells,
        )
        inner = (
            (np.arange(self.nx)[:, None, None] + 0.5 >= lo[0] + t)
            & (np.arange(self.nx)[:, None, None] + 0.5 < hi[0] - t)
            & (np.arange(self.nz)[None, None, :] + 0.5 >= lo[2] + t)
            & (np.arange(self.nz)[None, None, :] + 0.5 < hi[2] - t)
        )
        wet_columns = inner & (ids == -1)
        profile = np.sum(
            np.where(wet_columns, phi, 0), axis=(0, 2), dtype=float
        ) / np.maximum(wet_columns.sum(axis=(0, 2)), 1)
        immersed = (ids >= 0) & (y[None, :, None] < self.config.water_surface_cells)
        excess = np.where(immersed, (1 - self.rho_air) * profile[None, :, None], excess)
        dynamic = -self.gravity[1] * (
            np.cumsum(excess[:, ::-1, :], axis=1)[:, ::-1, :] - 0.5 * excess
        )
        self.reference_density.from_numpy(rho)
        self.reference_pressure.from_numpy(pref.astype(np.float32))
        self.pressure_lattice.from_numpy(dynamic.astype(np.float32))

    @ti.func
    def _tank_mirror_index(self, J):
        K = ti.max(1, ti.min(ti.Vector(self.shape) - 2, J))
        if J.y < 1:
            K.y = 1 - J.y
        return K

    @ti.func
    def _home_boundary_at(self, I):
        flag = -1
        moving = False
        if self._fluid(I):
            flag = 0
            for a, b, c in ti.static(ti.ndrange(3, 3, 3)):
                J = I + ti.Vector([a - 1, b - 1, c - 1])
                if not self._fluid(J):
                    flag = 2
                    if self._inside(J):
                        if self.solid_body_id[J] >= 0 or self.solid_body_id[J] == -3:
                            moving = True
        self._home_moving_neighbor[I] = ti.cast(moving, ti.i8)
        return flag

    @ti.func
    def _fixed_mirror(self, I, c):
        K = I + c
        R = ti.Matrix.identity(ti.f32, 3)
        faces = ti.cast(self._fixed_mirror_faces[I], ti.i32)
        for a in ti.static(range(3)):
            bit = 2 * a + ti.cast(c[a] > 0, ti.i32)
            if c[a] != 0 and (faces & (1 << bit)) != 0:
                K[a], R[a, a] = I[a], -1.0
        if not self._fluid(K):
            K = I
        return K, R

    @ti.func
    def _wall_phase_sample(self, I, c):
        J = I + c
        value = self.water_phase[I]
        if self._inside(J):
            if self._open(J):
                value = self.water_phase[J]
            elif self.solid_body_id[J] == -2:
                K, R = self._fixed_mirror(I, c)
                value = self.water_phase[K]
            elif self.solid_body_id[J] >= 0:
                value = self._fluid_phase_interpolate(
                    self._ice_mirror_point(I, J), value
                )
        return value

    @ti.func
    def _capillary_neighbor(self, I, c):
        J = I + c
        stress = self.capillary_stress[I]
        if self._fluid(J):
            stress = self.capillary_stress[J]
        elif self._inside(J) and self.solid_body_id[J] == -2:
            K, R = self._fixed_mirror(I, c)
            stress = R @ self.capillary_stress[K] @ R
        return stress

    @ti.func
    def _home_reflection(self, I, outgoing):
        J = I + direction(outgoing)
        value = 0.0
        if self._open(J):
            incoming = 26 - outgoing
            m = self.home_post[I]
            u = self.velocity_lattice[I]
            # Pressure anti-bounce-back: the even part prescribes reservoir
            # pressure, while the odd part transports normal velocity. Copying
            # the incoming non-equilibrium component back onto itself leaves
            # an undamped feedback mode at the low gas viscosity.
            value = (
                equilibrium(incoming, 0.0, u)
                + equilibrium(outgoing, 0.0, u)
                - reconstruct(outgoing, m)
            )
            # Count the actual crossing populations, at the same time/location
            # as streaming. Each boundary thread owns its cell accumulator.
            self.open_volume_flux_cell[I] += reconstruct(outgoing, m) - value
        else:
            value = self._moving_wall_reflection(I, outgoing)
        return value

    def _open_face_count(self, inset):
        x, y, z = self.nx - 2 * inset, self.ny - 1 - inset, self.nz - 2 * inset
        return 2 * y * (x + z - 2) + (x - 2) * (z - 2)

    @ti.func
    def _open_face_index(self, k, inset: ti.template()):
        # Five faces, with each edge/corner owned exactly once. inset=0 is
        # the halo; inset=1 is the adjacent physical layer. There is no bottom
        # opening, so both enumerations start at y=1.
        x = ti.static(self.nx - 2 * inset)
        y = ti.static(self.ny - 1 - inset)
        z = ti.static(self.nz - 2 * inset)
        nx_face, nz_face = y * z, (x - 2) * y
        I = ti.Vector.zero(ti.i32, 3)
        if k < 2 * nx_face:
            side, local = k // nx_face, k % nx_face
            I = ti.Vector([inset + side * (x - 1), 1 + local // z, inset + local % z])
        elif k < 2 * (nx_face + nz_face):
            local = k - 2 * nx_face
            side, local = local // nz_face, local % nz_face
            I = ti.Vector(
                [inset + 1 + local // y, 1 + local % y, inset + side * (z - 1)]
            )
        else:
            local = k - 2 * (nx_face + nz_face)
            I = ti.Vector(
                [inset + 1 + local // (z - 2), y, inset + 1 + local % (z - 2)]
            )
        return I

    @ti.kernel
    def _fill_open_halo(self):
        # Dense traversal coalesces the halo's vector stores better than the
        # compact face enumeration on CUDA; retain the measured faster path.
        for I in ti.grouped(self.water_phase):
            if self._open(I):
                J = ti.max(1, ti.min(ti.Vector(self.shape) - 2, I))
                u = self.velocity_lattice[J]
                normal = ti.cast(I - J, ti.f32)
                phi = self.water_phase[J]
                if u.dot(normal) < 0:
                    phi = 0.0
                self.water_phase[I] = phi
                self.velocity_lattice[I] = u
                self.pressure_lattice[I] = 0.0
                self.home[I] = pack(0.0, u, u.outer_product(u))

    def _collide_phase(self, moving_walls=True):
        InPlacePhase._collide_phase(self, moving_walls)
        self._phase_open_incoming()

    @ti.kernel
    def _phase_open_incoming(self):
        for k in range(ti.static(self._open_face_count(1))):
            I = self._open_face_index(k, 1)
            if self._fluid(I):
                phi, u = self.water_phase[I], self.velocity_lattice[I]
                for q in ti.static(range(1, 7)):
                    c = phase_direction(q)
                    if self._open(I - c):
                        # Missing incoming slots are local in BOTH AA parities.
                        # Preserve the outgoing non-equilibrium part.
                        opposite = ti.static(q + 1 if q % 2 else q - 1)
                        boundary_phi = phi
                        if u.dot(ti.cast(c, ti.f32)) > 0:
                            boundary_phi = 0.0
                        old = self.h[I, q]
                        value = (
                            old
                            - phase_equilibrium7(opposite, phi, u)
                            + phase_equilibrium7(q, boundary_phi, u)
                        )
                        self.h[I, q] = value
                        ti.atomic_add(self.water_outflow_lu[None], old - value)
