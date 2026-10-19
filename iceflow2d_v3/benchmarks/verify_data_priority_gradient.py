"""CPU reference for data-priority connected phase reconstruction.

Fit connected fluid samples first. Boundary derivatives may alter only the
null space of that fit. Single-row stencils preserve their resolved tangent;
unresolved cross curvature uses a declared zero-cross-curvature closure.
No phase-bound limiter is applied to the source gradient. Production files
are unchanged. Full grids and matched snapshots stay in RAM.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as four
from iceflow2d_v3.examples import dambreak_single_ice_melting_2d as single


def basis(x):
    x = np.asarray(x, dtype=float)
    a, b = x[:, 0], x[:, 1]
    return np.column_stack((a, b, 0.5 * a * a, a * b, 0.5 * b * b))


def decompose(a, rtol=1e-10):
    u, s, v = np.linalg.svd(a, full_matrices=True)
    rank = int(np.count_nonzero(s > rtol * max(1.0, float(s.max()) if s.size else 0.0)))
    inverse = (
        (v[:rank].T / s[:rank]) @ u[:, :rank].T
        if rank
        else np.zeros((a.shape[1], a.shape[0]))
    )
    return inverse, v[rank:].T, rank


def reconstruct(offsets, delta_phi, walls, *, prefer_nearest=True):
    offsets = np.asarray(offsets, dtype=float).reshape(-1, 2)
    values = np.asarray(delta_phi, dtype=float)
    walls = np.asarray(walls, dtype=float).reshape(-1, 4)
    if not len(offsets):
        return dict(
            gradient=[0.0, 0.0],
            coefficients=[0.0] * 5,
            spatial_rank=0,
            data_rank=0,
            boundary_residual=0.0,
            data_change=0.0,
            closure="unresolved isolated node; zero derivative is a closure",
        )
    _, _, spatial_rank = decompose(offsets)
    closure = "quadratic fluid fit"
    extra = []
    if spatial_rank == 1:
        _, _, v = np.linalg.svd(offsets, full_matrices=False)
        tangent = v[0]
        normal = np.array([-tangent[1], tangent[0]])
        if prefer_nearest:
            coordinate = offsets @ tangent
            negative = np.flatnonzero(coordinate < 0)
            positive = np.flatnonzero(coordinate > 0)
            if len(negative) and len(positive):
                keep = [
                    negative[np.argmin(np.abs(coordinate[negative]))],
                    positive[np.argmin(np.abs(coordinate[positive]))],
                ]
            else:
                keep = np.argsort(np.abs(coordinate))[:2].tolist()
            offsets, values = offsets[keep], values[keep]
        tx, ty = tangent
        nx, ny = normal
        # This is a model closure for information absent from a single row,
        # not a second normal constraint on the center gradient.
        extra.append([0, 0, tx * nx, tx * ny + ty * nx, ty * ny])
        closure = "single-row tangent fit; unresolved cross curvature set to zero"
        if len(offsets) == 1:
            extra.append([0, 0, tx * tx, 2 * tx * ty, ty * ty])
            closure = "one-sided first-order tangent; unresolved curvature closure"
    weights = 1 / np.linalg.norm(offsets, axis=1)
    data_matrix = basis(offsets) * weights[:, None]
    rhs = values * weights
    matrix = (
        np.vstack((data_matrix, np.asarray(extra).reshape(-1, 5)))
        if extra
        else data_matrix
    )
    rhs_full = np.r_[rhs, np.zeros(len(extra))]
    inverse, null, data_rank = decompose(matrix)
    initial = inverse @ rhs_full
    wall_matrix = np.array(
        [
            [nx, ny, nx * x, nx * y + ny * x, ny * y]
            for nx, ny, x, y in walls
            if nx * nx + ny * ny > 0.5
        ]
    ).reshape(-1, 5)
    result = initial.copy()
    if len(wall_matrix) and null.shape[1]:
        wall_inverse, _, _ = decompose(wall_matrix @ null, rtol=1e-8)
        result -= null @ (wall_inverse @ (wall_matrix @ initial))
    if spatial_rank == 2 and data_rank < 5:
        closure += "; insufficient quadratic rank, unresolved components retained"
    change = float(np.linalg.norm(data_matrix @ (result - initial), ord=np.inf))
    assert change < 1e-9, change
    return dict(
        gradient=result[:2].tolist(),
        coefficients=result.tolist(),
        spatial_rank=spatial_rank,
        data_rank=data_rank,
        boundary_residual=float(np.linalg.norm(wall_matrix @ result, ord=np.inf))
        if len(wall_matrix)
        else 0.0,
        data_change=change,
        closure=closure,
    )


def from_grid(phi, mask, geometry, cell):
    i, j = cell
    offsets = []
    values = []
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if (di or dj) and (int(mask[i, j]) >> ((di + 2) * 5 + dj + 2)) & 1:
                offsets.append([di, dj])
                values.append(float(phi[i + di, j + dj]) - float(phi[i, j]))
    return reconstruct(offsets, values, geometry[i, j])


def manufactured():
    checks = []

    def check(name, points, coefficients, walls):
        values = basis(points) @ np.asarray(coefficients)
        result = reconstruct(points, values, walls)
        error = float(np.linalg.norm(np.array(result["gradient"]) - coefficients[:2]))
        assert error < 1e-9, (name, error, result)
        checks.append(
            dict(
                name=name,
                expected_gradient=coefficients[:2],
                gradient_error=error,
                **result,
            )
        )

    half = np.array([[i, j] for i in range(-2, 3) for j in range(3) if i or j], float)
    full = np.array(
        [[i, j] for i in range(-2, 3) for j in range(-2, 3) if i or j], float
    )
    floor = [[0, 1, 0, -0.5]]
    for value in (0.0, 0.37, 1.0):
        check(f"uniform_phi_{value}_at_wall", half, [0, 0, 0, 0, 0], floor)
    check("flat_wall_affine_tangent", half, [0.005, 0, 0, 0, 0], floor)
    eps = 3e-4
    upper = 0.02
    gx = 0.005
    hyy = eps * gx / (upper + 0.5)
    check(
        "slightly_tilted_single_row",
        [[-1, 0], [1, 0]],
        [gx, 0.5 * hyy, 0, 0, hyy],
        [[0, 1, 0, -0.5], [-eps, 1, 0, upper]],
    )
    corner = np.array([[i, j] for i in range(3) for j in range(3) if i or j], float)
    check(
        "two_wall_quadratic_corner",
        corner,
        [0.001, 0.002, 0.002, 0, 0.004],
        [[1, 0, -0.5, 0], [0, 1, 0, -0.5]],
    )
    check("bulk_quadratic", full, [0.007, -0.002, 0.003, 0.0004, 0.001], [])
    check("one_sided_affine_tangent", [[1, 0]], [0.005, 0, 0, 0, 0], floor)
    # A real separated fluid pocket: build its connectivity without examining
    # phi, then change the value across the intervening solid cell.
    fluid = np.zeros((11, 11), dtype=bool)
    fluid[5:8, 5] = True
    fluid[3, 5] = True
    reached = {(5, 5)}
    for distance in range(1, 5):
        for i in range(3, 8):
            for j in range(3, 8):
                if fluid[i, j] and abs(i - 5) + abs(j - 5) == distance:
                    if any(
                        (i + di, j + dj) in reached
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    ):
                        reached.add((i, j))
    mask = np.zeros((11, 11), dtype=np.int32)
    for i, j in reached:
        mask[5, 5] |= 1 << ((i - 5 + 2) * 5 + j - 5 + 2)
    assert (3, 5) not in reached and (7, 5) in reached
    geometry = np.zeros((11, 11, 8, 4))
    geometry[5, 5, 1] = [0, 1, 0, 0.5]
    geometry[5, 5, 3] = [0, 1, 0, -0.5]
    for excluded_value in (0.0, 0.9):
        phase = np.full((11, 11), 1e-6)
        phase[3, 5] = excluded_value
        reconstructed = from_grid(phase, mask, geometry, (5, 5))
        error = float(np.linalg.norm(reconstructed["gradient"]))
        assert error == 0, (excluded_value, reconstructed)
        checks.append(
            dict(
                name=f"opposite_side_excluded_phi_{excluded_value}",
                expected_gradient=[0, 0],
                gradient_error=error,
                disconnected_value=excluded_value,
                **reconstructed,
            )
        )
    return checks


def point_records(s, cells):
    s._prepare_phase_geometry()
    s._prepare_phase_gradients()
    phi = s.water_phase.to_numpy()
    mask = s.phase_donor_mask.to_numpy()
    geometry = s.phase_wall_geometry.to_numpy()
    gradient = s.source_gradient_lattice.to_numpy()
    return [
        dict(
            cell=list(cell),
            phi=float(phi[cell]),
            existing_gradient=gradient[cell].tolist(),
            connected_donor_count=int(mask[cell]).bit_count() - 1,
            proposed=from_grid(phi, mask, geometry, cell),
        )
        for cell in cells
    ]


def install_local_reference(s, cells):
    original = s._prepare_phase_gradients

    def prepare():
        original()
        phi = s.water_phase.to_numpy()
        mask = s.phase_donor_mask.to_numpy()
        geometry = s.phase_wall_geometry.to_numpy()
        solid = s.solid_mask.to_numpy()
        wall = s.wall_mask.to_numpy()
        source = s.source_gradient_lattice.to_numpy()
        phase = s.phase_gradient_lattice.to_numpy()
        for cell in cells:
            if solid[cell] == 0 and wall[cell] == 0:
                gradient = from_grid(phi, mask, geometry, cell)["gradient"]
                source[cell] = gradient
                phase[cell] = gradient
        s.source_gradient_lattice.from_numpy(source)
        s.phase_gradient_lattice.from_numpy(phase)

    s._prepare_phase_gradients = prepare


def replay(s, cells, steps, *, frozen=False):
    install_local_reference(s, cells)
    s._thermal_stability_check_enabled = True
    history = [state_summary(s)]
    start = s.steps
    error = None
    began = time.monotonic()
    try:
        for k in range(steps):
            if frozen:
                s._advance_fluid()
                s.steps += 1
            else:
                s.step()
            if k % 20 == 0 or k == steps - 1:
                history.append(state_summary(s))
                if history[-1]["speed_l1_max"] > 1:
                    break
    except RuntimeError as exc:
        error = str(exc)
    return dict(
        start_step=start,
        completed_steps=s.steps - start,
        requested_steps=steps,
        frozen_geometry=frozen,
        error=error,
        history=history,
        last=state_summary(s),
        elapsed_s=time.monotonic() - began,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow2d_v3/outputs/data_priority_gradient_proposal"),
    )
    parser.add_argument("--replay-steps", type=int, default=240)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = dict(
        manufactured_checks=manufactured(),
        production_solver_modified=False,
        raw_fields_saved=False,
        cases=[],
        limitations=[
            "Single-row cross curvature is a declared model closure, not observable from the samples.",
            "Nonzero boundary residual means the supplied field and the local polynomial cannot satisfy all boundary equations; it must be reported, never resolved by wiping the data-supported gradient.",
            "Replay overrides selected local gradient fields only; it is not a complete production implementation or an 8 s stability result.",
        ],
    )
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    cases = [
        (
            "single_connected",
            single.create_config(),
            22627,
            [(i, 3) for i in range(416, 447)],
            [(433, 3), (434, 3), (435, 3), (437, 3), (439, 3)],
        ),
        (
            "four_connected",
            four.create_config(),
            8276,
            [(i, j) for i in range(175, 178) for j in range(3, 34)],
            [(176, 6), (176, 7), (176, 8), (176, 11)],
        ),
        (
            "four_legacy_ghost",
            replace(four.create_config(), phase_gradient_scheme="legacy"),
            8544,
            [(i, j) for i in range(178, 183) for j in range(3, 8)],
            [(180, 3), (181, 3)],
        ),
    ]
    for name, config, checkpoint, cells, probes in cases:
        print(json.dumps(dict(event="begin", case=name)), flush=True)
        s = IceFlow2D(config)
        s._thermal_stability_check_enabled = True
        s.step(checkpoint)
        old_gradient = None
        if name == "four_legacy_ghost":
            # Use the native legacy source gradient before changing only the
            # inspection instance. The full original snapshot stays in RAM.
            from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import PointProbe

            probe = PointProbe(config)
            snapshot = (device_snapshot(s), host_snapshot(s))
            restore(probe, snapshot)
            old_gradient = [
                probe.point_rows()[0][key] for key in ("grad_phi_x", "grad_phi_y")
            ]
            target = IceFlow2D(
                replace(
                    config, phase_gradient_scheme="connected_ls", phase_warmup_steps=0
                )
            )
            restore(target, snapshot)
            s = target
            s.body_velocity.fill(0)
            s.body_angular_velocity.fill(0)
        records = point_records(s, probes)
        trial = replay(s, cells, args.replay_steps, frozen=name == "four_legacy_ghost")
        case = dict(
            name=name,
            checkpoint=checkpoint,
            points=records,
            legacy_original_gradient=old_gradient,
            replay=trial,
        )
        result["cases"].append(case)
        (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    name=name,
                    points=records,
                    replay={k: v for k, v in trial.items() if k != "history"},
                )
            ),
            flush=True,
        )
    print((out / "verification.json").resolve(), flush=True)


if __name__ == "__main__":
    main()
