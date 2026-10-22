"""D3Q7 central-moment Allen--Cahn transport in one AA-pattern buffer.

Parity 0: incoming populations are local. Parity 1: direction q is in
(x-cq, opposite(q)), except a solid link reflects to the local q slot.
Each collision writes precisely the seven addresses it read, permuted by
opposite(q). Thus no other thread can overwrite a live input. Geometry
changes permute only changed link pairs before exposed cells are refilled.
"""

# ruff: noqa: E741
import taichi as ti

from .lattice import density

PHASE_DIRECTIONS = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
PHASE_OPPOSITE = (0, 2, 1, 4, 3, 6, 5)


@ti.func
def phase_direction(q: ti.template()):
    return ti.Vector(ti.static(PHASE_DIRECTIONS[q]))


@ti.func
def phase_equilibrium7(q: ti.template(), phi, u):
    w = ti.static(0.25 if q == 0 else 0.125)
    return w * phi * (1 + 4 * phase_direction(q).dot(u))


@ti.func
def phase_collision(h, phi, u, normal, mobility, width):
    """CM-MRT diag(1,s,s,s,1,1,1), paper (11)--(14), ds^2=1/4.

    The second central moments relax to those of the first-order h_eq.
    Recovering raw second moments adds 2*u*delta(first moment), which
    distinguishes this operator from a raw-moment/BGK collision.
    """
    first = ti.Vector([h[1] - h[2], h[3] - h[4], h[5] - h[6]])
    source = phi * (1 - phi) / width * normal
    rate = 1 / (0.5 + 4 * mobility)
    delta = (1 - rate) * (first - phi * u) + (1 - 0.5 * rate) * source
    j = phi * u + delta
    second = ti.Vector([phi * 0.25, phi * 0.25, phi * 0.25]) + 2 * u * (
        delta - 0.5 * source
    )
    post = ti.Vector.zero(ti.f32, 7)
    for a in ti.static(range(3)):
        post[2 * a + 1] = 0.5 * (second[a] + j[a])
        post[2 * a + 2] = 0.5 * (second[a] - j[a])
    post[0] = phi - (second.x + second.y + second.z)
    flux = 0.5 * (first + j) - phi * u
    return post, flux


@ti.data_oriented
class InPlacePhase:
    @ti.func
    def _phase_neighbor(self, I, c):
        return I + c

    @ti.func
    def _phase_address(self, I, q: ti.template()):
        J, slot = I, ti.cast(q, ti.i32)
        if self.phase_parity[None] != 0:
            neighbor = self._phase_neighbor(I, -phase_direction(q))
            if self._fluid(neighbor):
                J = neighbor
                slot = ti.cast(ti.static(PHASE_OPPOSITE[q]), ti.i32)
        return J, slot

    @ti.func
    def _read_phase(self, I, q: ti.template()):
        J, slot = self._phase_address(I, q)
        return self.h[J, slot]

    @ti.func
    def _set_phase_equilibrium(self, I, phi, u):
        for q in ti.static(range(7)):
            J, slot = self._phase_address(I, q)
            self.h[J, slot] = phase_equilibrium7(q, phi, u)

    def _collide_phase(self, moving_walls=True):
        self._collide_phase_kernel(moving_walls)

    @ti.kernel
    def _collide_phase_kernel(self, moving_walls: ti.template()):
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                phi, u = self.water_phase[I], self.velocity_lattice[I]
                grad = self.phase_gradient_lattice[I]
                normal = grad / ti.sqrt(
                    grad.dot(grad) + self.config.phase_normal_epsilon**2 + 1e-28
                )
                incoming = ti.Vector.zero(ti.f32, 7)
                for q in ti.static(range(7)):
                    incoming[q] = self._read_phase(I, q)
                post, flux = phase_collision(
                    incoming,
                    phi,
                    u,
                    normal,
                    self.config.mobility,
                    self.config.interface_width,
                )
                self.phase_mass_flux[I] = (1 - self.rho_air) * flux
                for q in ti.static(range(7)):
                    value = post[q]
                    neighbor = self._phase_neighbor(I, phase_direction(q))
                    if ti.static(q != 0 and moving_walls):
                        if self._inside(neighbor) and self.solid_body_id[neighbor] >= 0:
                            eta = self._boundary_link_fraction(I, neighbor)
                            point = ti.cast(I, ti.f32) + 0.5 + eta * phase_direction(q)
                            ub = self._body_velocity_at(
                                point, self.solid_body_id[neighbor]
                            )
                            value -= phi * phase_direction(q).dot(ub)  # 2*w/ds^2=1
                    J, slot = self._phase_address(I, ti.static(PHASE_OPPOSITE[q]))
                    self.h[J, slot] = value
        self.phase_parity[None] = 1 - self.phase_parity[None]

    def _recover_phase(self, recover_flow=False):
        self._recover_phase_kernel(recover_flow)

    @ti.kernel
    def _recover_phase_kernel(self, recover_flow: ti.template()):
        if ti.static(recover_flow and self.config.check_stability):
            self.invalid_index[None] = self.nx * self.ny * self.nz
            self.maximum_speed[None] = 0.0
        for I in ti.grouped(self.water_phase):
            if self._fluid(I):
                # Pair opposite directions first; seven FP32 values only.
                total = self.h[I, 0]
                for a in ti.static(range(3)):
                    total += self._read_phase(I, 2 * a + 1) + self._read_phase(
                        I, 2 * a + 2
                    )
                bounded = total
                if not ti.math.isnan(total) and not ti.math.isinf(total):
                    bounded = ti.min(1.0, ti.max(0.0, total))
                self.h[I, 0] += bounded - total
                self.water_phase[I] = bounded
                ti.atomic_add(self.phase_clip_volume[None], bounded - total)
                if ti.static(recover_flow):
                    pressure = self.home[I][0] * density(bounded, self.rho_air) / 3
                    self.pressure_lattice[I] = pressure
                    if ti.static(self.config.check_stability):
                        self._scan_stability_at(I, bounded, pressure)
            else:
                self.water_phase[I] = 0.0
        if ti.static(recover_flow and self.config.check_stability):
            self.peak_speed[None] = ti.max(
                self.peak_speed[None], self.maximum_speed[None]
            )

    @ti.kernel
    def _repair_phase_layout(self):
        # In odd layout each fluid-fluid link swaps two addresses. A mask
        # change toggles just that swap; existing logical populations survive
        # exactly, and subsequent refill owns only newly fluid addresses.
        for I in ti.grouped(self.water_phase):
            if self.phase_parity[None] != 0:
                for a in ti.static(range(3)):
                    q = ti.static(2 * a + 1)
                    J = I + phase_direction(q)
                    if self._inside(J):
                        before = (
                            self.previous_body_id[I] == -1
                            and self.previous_body_id[J] == -1
                        )
                        after = (
                            self.solid_body_id[I] == -1 and self.solid_body_id[J] == -1
                        )
                        if before != after:
                            value = self.h[I, ti.static(PHASE_OPPOSITE[q])]
                            self.h[I, ti.static(PHASE_OPPOSITE[q])] = self.h[J, q]
                            self.h[J, q] = value

    @ti.kernel
    def export_phase_populations(self, out: ti.types.ndarray()):
        """Canonical diagnostic export; not used by stepping or checkpoints."""
        for I in ti.grouped(self.water_phase):
            for q in ti.static(range(7)):
                out[I.x, I.y, I.z, q] = self._read_phase(I, q)
