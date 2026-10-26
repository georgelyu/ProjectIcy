"""An open hollow tank is five fixed boxes, never a solid bounding box."""

import numpy as np
from ..rigid import (
    RigidBodies,
    SIGNS,
    rotation_matrix,
    separating_axis,
    apply_wall_contact_manifold,
)


def tank_boxes(config):
    lo, hi = np.array(config.tank_lower_cells), np.array(config.tank_upper_cells)
    t = config.tank_wall_cells
    result = [(lo, np.array([hi[0], lo[1] + t, hi[2]]))]
    for a in (0, 2):
        low_hi, high_lo = hi.copy(), lo.copy()
        low_hi[a], high_lo[a] = lo[a] + t, hi[a] - t
        result.extend([(lo.copy(), low_hi), (high_lo, hi.copy())])
    return [(0.5 * (a + b), 0.5 * (b - a)) for a, b in result]


class OpenTankBodies(RigidBodies):
    def advance(self):
        if self.s.config.body_motion_enabled:
            super().advance()

    def resolve_contacts(self):
        for _ in range(self.s.config.body_contact_iterations):
            worst = 0.0
            closing = 0.0
            for b in range(len(self.mass)):
                if self.mass[b] <= 1e-15:
                    continue
                r = rotation_matrix(self.q[b])
                inertia_inv = np.linalg.inv(r @ self.inertia[b] @ r.T)
                for cb, hb in tank_boxes(self.s.config):
                    ca = self.origin[b] + r @ ((self.lower[b] + self.upper[b]) / 2)
                    ha = (self.upper[b] - self.lower[b]) / 2
                    hit = separating_axis(ca, r, ha, cb, np.eye(3), hb)
                    if hit is None:
                        continue
                    normal, depth = hit
                    worst = max(worst, depth)
                    self.origin[b] += depth * normal
                    vertices = (
                        self.origin[b]
                        + (
                            self.lower[b]
                            + (SIGNS + 1) / 2 * (self.upper[b] - self.lower[b])
                        )
                        @ r.T
                    )
                    projection = vertices @ normal
                    points = vertices[projection < projection.min() + 1e-7]
                    center = self.origin[b] + r @ self.com[b]
                    closing = max(
                        closing,
                        float(
                            np.maximum(
                                0,
                                -(
                                    self.velocity[b]
                                    + np.cross(self.omega[b], points - center)
                                )
                                @ normal,
                            ).max()
                        ),
                    )
                    apply_wall_contact_manifold(
                        self.velocity[b],
                        self.omega[b],
                        self.mass[b],
                        inertia_inv,
                        points - center,
                        normal,
                    )
                vertices = (
                    self.origin[b]
                    + (
                        self.lower[b]
                        + (SIGNS + 1) / 2 * (self.upper[b] - self.lower[b])
                    )
                    @ r.T
                )
                depth = 1.0 - vertices[:, 1].min()
                if depth >= -1e-9:
                    worst = max(worst, depth)
                    self.origin[b, 1] += max(0, depth)
                    vertices[:, 1] += max(0, depth)
                    points = vertices[vertices[:, 1] < vertices[:, 1].min() + 1e-7]
                    center = self.origin[b] + r @ self.com[b]
                    apply_wall_contact_manifold(
                        self.velocity[b],
                        self.omega[b],
                        self.mass[b],
                        inertia_inv,
                        points - center,
                        np.array([0.0, 1.0, 0.0]),
                    )
                # Exiting bodies require a separate rigid-body outflow model.
                if (
                    np.any(vertices[:, (0, 2)] < 1)
                    or np.any(vertices[:, (0, 2)] > np.array(self.s.shape)[[0, 2]] - 1)
                    or vertices[:, 1].max() > self.s.ny - 1
                ):
                    raise RuntimeError("dry ice reached an open domain boundary")
            if worst < 1e-7 and closing < 1e-7:
                return
        raise RuntimeError("dry-ice/tank contact did not converge")
