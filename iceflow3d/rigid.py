"""Six-DOF rigid motion and frictionless, inelastic OBB contact.

The OBB encloses the *remaining* material voxels. It is conservative for an
irregular melted remnant: concavities and subcell lubrication are unresolved.
Only the few body states cross the host/device boundary, never the fluid grid.
"""

from itertools import combinations, product
import numpy as np

SIGNS = np.array(list(product((-1.0, 1.0), repeat=3)))


def rotation_matrix(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array(
        [
            w * v - x * i - y * j - z * k,
            w * i + x * v + y * k - z * j,
            w * j - x * k + y * v + z * i,
            w * k + x * j - y * i + z * v,
        ]
    )


def advance_quaternion(q, omega, dt=1.0):
    angle = np.linalg.norm(omega) * dt
    scale = 0.5 * dt if angle < 1e-12 else np.sin(0.5 * angle) / np.linalg.norm(omega)
    result = quaternion_multiply(np.r_[np.cos(0.5 * angle), scale * omega], q)
    return result / np.linalg.norm(result)


def separating_axis(ca, ra, ha, cb, rb, hb):
    """Return the minimum translation normal (b -> a) and depth, or None."""
    delta = ca - cb
    if np.any(abs(delta) > abs(ra) @ ha + abs(rb) @ hb + 1e-9):
        return None
    candidates = [*ra.T, *rb.T, *(np.cross(a, b) for a in ra.T for b in rb.T)]
    best = (None, np.inf)
    for axis in candidates:
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        n = axis / length
        depth = np.dot(ha, abs(ra.T @ n)) + np.dot(hb, abs(rb.T @ n)) - abs(delta @ n)
        if depth < -1e-9:
            return None
        if depth < best[1]:
            best = (n if delta @ n >= 0 else -n, max(0.0, depth))
    return best


def support(center, rotation, half, direction):
    projected = rotation.T @ direction
    # On a face return its center, rather than an arbitrary corner.
    signs = np.where(abs(projected) < 1e-9, 0.0, np.sign(projected))
    return center + rotation @ (half * signs)


def apply_contact_impulse(
    va, wa, ma, ia, ra, normal, vb=None, wb=None, mb=None, ib=None, rb=None
):
    """Modify velocities in place. Infinite-mass b denotes a stationary wall."""
    relative = va + np.cross(wa, ra)
    effective = 1 / ma + normal @ np.cross(ia @ np.cross(ra, normal), ra)
    if vb is not None:
        relative -= vb + np.cross(wb, rb)
        effective += 1 / mb + normal @ np.cross(ib @ np.cross(rb, normal), rb)
    closing = relative @ normal
    if closing < 0:
        impulse = -closing / effective * normal
        va += impulse / ma
        wa += ia @ np.cross(ra, impulse)
        if vb is not None:
            vb -= impulse / mb
            wb -= ib @ np.cross(rb, impulse)
    return max(0.0, -closing)


def apply_wall_contact_manifold(velocity, omega, mass, inverse_inertia, arms, normal):
    """Project all wall support velocities with nonnegative, inelastic impulses.

    A planar normal-velocity Jacobian has rank at most three (translation along
    the normal and two rotations). Enumerating independent active sets avoids
    slow pointwise iteration for a low-inertia melted remnant with a large OBB.
    """
    angular = np.cross(arms, normal)
    speed = (velocity + np.cross(omega, arms)) @ normal
    tolerance = 1e-12 * max(1.0, float(np.max(abs(speed))))
    if speed.min() >= -tolerance:
        return
    effective = 1 / mass + angular @ inverse_inertia @ angular.T
    for count in range(1, min(3, len(arms)) + 1):
        for subset in combinations(range(len(arms)), count):
            indices = np.array(subset)
            block = effective[np.ix_(indices, indices)]
            impulses = np.linalg.lstsq(block, -speed[indices], rcond=1e-12)[0]
            if np.any(impulses < 0):
                continue
            result = speed + effective[:, indices] @ impulses
            if np.any(result < -tolerance) or np.any(abs(result[indices]) > tolerance):
                continue
            velocity += normal * impulses.sum() / mass
            omega += inverse_inertia @ (angular[indices].T @ impulses)
            return
    raise RuntimeError("wall contact manifold has no feasible nonnegative impulse")


class RigidBodies:
    def __init__(self, simulation):
        self.s = simulation
        bodies = simulation.config.bodies
        self.origin = np.array([b.center_cells for b in bodies], dtype=float)
        self.q = np.array([b.quaternion_wxyz for b in bodies], dtype=float)
        self.velocity = np.array([b.velocity_lattice for b in bodies], dtype=float)
        self.omega = np.array([b.angular_velocity_lattice for b in bodies], dtype=float)
        self.com = np.zeros((len(bodies), 3))
        self.mass = np.zeros(len(bodies))
        self.inertia = np.zeros((len(bodies), 3, 3))
        self.lower = -0.5 * np.array([b.size_cells for b in bodies], dtype=float)
        self.upper = -self.lower
        self.upload()

    def upload(self):
        s = self.s
        rotations = np.array([rotation_matrix(q) for q in self.q])
        # One transfer for the small body state. Host pose/contact arithmetic
        # stays FP64; the device receives the same once-rounded FP32 values.
        state = np.concatenate(
            (
                self.origin,
                rotations.reshape(s.body_count, 9),
                self.origin + np.einsum("bij,bj->bi", rotations, self.com),
                self.velocity,
                self.omega,
            ),
            axis=1,
        ).astype(np.float32)
        s._upload_rigid_state(state)

    def update_mass(self, *, upload=True):
        s = self.s
        s._reduce_mass()
        state = np.empty((s.body_count, 19), np.float32)
        s._download_rigid_state(state, True)
        state = state.astype(float)
        mass, first = state[:, 0].copy(), state[:, 1:4].copy()
        second = state[:, 4:13].reshape(s.body_count, 3, 3)
        for b in range(s.body_count):
            if mass[b] > 1e-15:
                com = first[b] / mass[b]
                shift = rotation_matrix(self.q[b]) @ (com - self.com[b])
                self.velocity[b] += np.cross(self.omega[b], shift)
                self.com[b] = com
                self.inertia[b] = second[b] - mass[b] * (
                    np.dot(com, com) * np.eye(3) - np.outer(com, com)
                )
            else:
                self.velocity[b] = self.omega[b] = 0.0
        self.mass = mass
        self.lower = state[:, 13:16].copy()
        self.upper = state[:, 16:19].copy()
        if upload:
            self.upload()

    def advance(self):
        s = self.s
        state = np.empty((s.body_count, 6), np.float32)
        s._download_rigid_state(state, False)
        forces, torques = state[:, :3].astype(float), state[:, 3:].astype(float)
        gravity = np.array(s.config.gravity) * s.scales.dt_s**2 / s.scales.dx_m
        for b in range(s.body_count):
            if self.mass[b] <= 1e-15:
                continue
            r = rotation_matrix(self.q[b])
            inertia = r @ self.inertia[b] @ r.T
            center = self.origin[b] + r @ self.com[b]
            self.velocity[b] += forces[b] / self.mass[b] + gravity
            self.omega[b] += np.linalg.solve(
                inertia, torques[b] - np.cross(self.omega[b], inertia @ self.omega[b])
            )
            self.velocity[b] *= s.config.linear_damping
            self.omega[b] *= s.config.angular_damping
            # Prevent contact tunnelling; fail rather than silently cap motion.
            travel = (
                np.linalg.norm(self.velocity[b])
                + np.linalg.norm(self.omega[b])
                * np.linalg.norm(self.upper[b] - self.lower[b])
                / 2
            )
            if not np.isfinite(travel) or travel > 0.5:
                raise RuntimeError(
                    f"rigid displacement exceeds half a cell at step {s.steps + 1}, body {b}"
                )
            center += self.velocity[b]
            self.q[b] = advance_quaternion(self.q[b], self.omega[b])
            self.origin[b] = center - rotation_matrix(self.q[b]) @ self.com[b]
        self.resolve_contacts()
        self.upload()

    def resolve_contacts(self):
        s = self.s
        low = np.full(3, s.config.boundary_cells, dtype=float)
        high = np.array(s.shape) - low
        active = [
            b
            for b in range(s.body_count)
            if self.mass[b] > 1e-15 and np.all(self.upper[b] > self.lower[b])
        ]
        for _ in range(s.config.body_contact_iterations):
            maximum = 0.0
            closing_speed = 0.0
            rotations = [rotation_matrix(q) for q in self.q]
            inverses = {
                b: np.linalg.inv(rotations[b] @ self.inertia[b] @ rotations[b].T)
                for b in active
            }
            for b in active:
                r = rotations[b]
                verts = (
                    self.origin[b]
                    + (
                        self.lower[b]
                        + (SIGNS + 1) / 2 * (self.upper[b] - self.lower[b])
                    )
                    @ r.T
                )
                for axis, side in product(range(3), (-1, 1)):
                    extreme = (
                        verts[:, axis].min() if side == 1 else verts[:, axis].max()
                    )
                    depth = low[axis] - extreme if side == 1 else extreme - high[axis]
                    if depth >= -1e-8:
                        n = np.eye(3)[axis] * side
                        correction = max(0.0, depth) * n
                        self.origin[b] += correction
                        verts += correction
                        maximum = max(maximum, depth)
                        # All support vertices matter for a rotating cube.
                        target = (
                            verts[:, axis].min() if side == 1 else verts[:, axis].max()
                        )
                        for p in verts[abs(verts[:, axis] - target) < 1e-6]:
                            arm = p - (self.origin[b] + r @ self.com[b])
                            closing_speed = max(
                                closing_speed,
                                apply_contact_impulse(
                                    self.velocity[b],
                                    self.omega[b],
                                    self.mass[b],
                                    inverses[b],
                                    arm,
                                    n,
                                ),
                            )
            for ai, a in enumerate(active):
                for b in active[ai + 1 :]:
                    ra, rb = rotations[a], rotations[b]
                    ca = self.origin[a] + ra @ ((self.lower[a] + self.upper[a]) / 2)
                    cb = self.origin[b] + rb @ ((self.lower[b] + self.upper[b]) / 2)
                    ha, hb = (
                        (self.upper[a] - self.lower[a]) / 2,
                        (self.upper[b] - self.lower[b]) / 2,
                    )
                    hit = separating_axis(ca, ra, ha, cb, rb, hb)
                    if hit is None:
                        continue
                    n, depth = hit
                    point = 0.5 * (support(ca, ra, ha, -n) + support(cb, rb, hb, n))
                    closing_speed = max(
                        closing_speed,
                        apply_contact_impulse(
                            self.velocity[a],
                            self.omega[a],
                            self.mass[a],
                            inverses[a],
                            point - self.origin[a] - ra @ self.com[a],
                            n,
                            self.velocity[b],
                            self.omega[b],
                            self.mass[b],
                            inverses[b],
                            point - self.origin[b] - rb @ self.com[b],
                        ),
                    )
                    total = self.mass[a] + self.mass[b]
                    self.origin[a] += (depth + 1e-9) * n * self.mass[b] / total
                    self.origin[b] -= (depth + 1e-9) * n * self.mass[a] / total
                    maximum = max(maximum, depth)
            if maximum < 1e-7 and closing_speed < 1e-7:
                return
        # The sweep records violations *before* applying its impulses. On the
        # final sweep those can exceed the tolerance even though the resulting
        # simultaneous constraints already satisfy it. Check the actual state;
        # do not add impulses, relax the tolerance, or accept residual overlap.
        maximum, closing_speed = self.contact_residuals()
        if maximum < 1e-7 and closing_speed < 1e-7:
            return
        if maximum < 1e-7:
            # A nearly flat, low-inertia remnant can converge too slowly under
            # point impulses. Solve each wall's support set simultaneously.
            # Recheck all constraints afterwards, including other walls/pairs.
            for b in active:
                r = rotations[b]
                center = self.origin[b] + r @ self.com[b]
                verts = (
                    self.origin[b]
                    + (
                        self.lower[b]
                        + (SIGNS + 1) / 2 * (self.upper[b] - self.lower[b])
                    )
                    @ r.T
                )
                for axis, side in product(range(3), (-1, 1)):
                    extreme = (
                        verts[:, axis].min() if side == 1 else verts[:, axis].max()
                    )
                    depth = low[axis] - extreme if side == 1 else extreme - high[axis]
                    if depth >= -1e-8:
                        points = verts[abs(verts[:, axis] - extreme) < 1e-6]
                        apply_wall_contact_manifold(
                            self.velocity[b],
                            self.omega[b],
                            self.mass[b],
                            inverses[b],
                            points - center,
                            np.eye(3)[axis] * side,
                        )
            maximum, closing_speed = self.contact_residuals()
            if maximum < 1e-7 and closing_speed < 1e-7:
                return
        raise RuntimeError(
            f"contact projection did not converge at step {s.steps + 1}: "
            f"penetration={maximum:.9g} cells, closing_speed={closing_speed:.9g} LU"
        )

    def contact_residuals(self):
        """Return current maximum penetration and closing speed without mutation."""
        s = self.s
        low = np.full(3, s.config.boundary_cells, dtype=float)
        high = np.array(s.shape) - low
        active = [
            b
            for b in range(s.body_count)
            if self.mass[b] > 1e-15 and np.all(self.upper[b] > self.lower[b])
        ]
        rotations = [rotation_matrix(q) for q in self.q]
        maximum, closing_speed = 0.0, 0.0
        for b in active:
            r = rotations[b]
            verts = (
                self.origin[b]
                + (self.lower[b] + (SIGNS + 1) / 2 * (self.upper[b] - self.lower[b]))
                @ r.T
            )
            center = self.origin[b] + r @ self.com[b]
            for axis, side in product(range(3), (-1, 1)):
                extreme = verts[:, axis].min() if side == 1 else verts[:, axis].max()
                depth = low[axis] - extreme if side == 1 else extreme - high[axis]
                maximum = max(maximum, depth)
                if depth >= -1e-8:
                    points = verts[abs(verts[:, axis] - extreme) < 1e-6]
                    speeds = self.velocity[b] + np.cross(self.omega[b], points - center)
                    closing_speed = max(
                        closing_speed, float((-side * speeds[:, axis]).max())
                    )
        for ai, a in enumerate(active):
            for b in active[ai + 1 :]:
                ra, rb = rotations[a], rotations[b]
                ca = self.origin[a] + ra @ ((self.lower[a] + self.upper[a]) / 2)
                cb = self.origin[b] + rb @ ((self.lower[b] + self.upper[b]) / 2)
                ha, hb = (
                    (self.upper[a] - self.lower[a]) / 2,
                    (self.upper[b] - self.lower[b]) / 2,
                )
                hit = separating_axis(ca, ra, ha, cb, rb, hb)
                if hit is not None:
                    n, depth = hit
                    point = 0.5 * (support(ca, ra, ha, -n) + support(cb, rb, hb, n))
                    relative = (
                        self.velocity[a]
                        + np.cross(
                            self.omega[a], point - self.origin[a] - ra @ self.com[a]
                        )
                        - self.velocity[b]
                        - np.cross(
                            self.omega[b], point - self.origin[b] - rb @ self.com[b]
                        )
                    )
                    maximum = max(maximum, depth)
                    closing_speed = max(closing_speed, float(-relative @ n))
        return maximum, closing_speed
