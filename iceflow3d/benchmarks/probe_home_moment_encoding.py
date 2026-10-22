"""Read-only solver probes for the HOME feasibility study.

Runs isolated current collision kernels on CPU, not a HOME implementation or
a coupled-flow validation. Production solver files are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig  # noqa: E402
from iceflow3d.lattice import DIRECTIONS, REST, WEIGHTS  # noqa: E402

C = np.asarray(DIRECTIONS, dtype=np.float64)
W = np.asarray(WEIGHTS, dtype=np.float64)
POWERS = np.array(list(np.ndindex(3, 3, 3)))


def phase_post(h, phi, u, gradient, *, reduced=False):
    """FP64 algebra corresponding to the present phase collision."""
    normal = gradient / np.sqrt(gradient @ gradient + 0.01**2 + 1e-28)
    cu = C @ u
    eq = W * phi * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (u @ u))
    src = W * 4 * phi * (1 - phi) / 5 * (C @ normal)
    matrix = np.prod((C[None] - u) ** POWERS[:, None], axis=2)
    moments = np.zeros(27)
    if reduced:
        # Only the four incoming moments are read from h here.
        zeroth, first = h.sum(), C.T @ h
        residual = first - u * zeroth - (C - u).T @ eq
        residual += 0.5 * (C - u).T @ src
    else:
        residual = (matrix @ (h - eq + 0.5 * src))[[9, 3, 1]]
    moments[[9, 3, 1]] = (1 - 1 / (0.5 + 3 * 0.1)) * residual
    post = eq + 0.5 * src + np.linalg.solve(matrix, moments)
    post[REST] = phi - np.delete(post, REST).sum()
    flux = (1 - 1.25 / 1000) * (0.5 * (C.T @ h + C.T @ post) - phi * u)
    return post, flux


def algebra_probe():
    rng = np.random.default_rng(20261022)
    errors, flux_errors = [], []
    for _ in range(128):
        phi = rng.uniform(0, 1)
        u = rng.uniform(-0.08, 0.08, 3)
        gradient = rng.uniform(-0.2, 0.2, 3)
        h = W * phi + rng.normal(0, 0.003, 27)
        h[REST] += phi - h.sum()
        a, fa = phase_post(h, phi, u, gradient)
        b, fb = phase_post(h, phi, u, gradient, reduced=True)
        errors.append(np.max(abs(a - b)))
        flux_errors.append(np.max(abs(fa - fb)))
    assert max(errors) < 1e-13
    return dict(
        samples=128,
        max_population_error=float(max(errors)),
        max_phase_mass_flux_error=float(max(flux_errors)),
        scope="FP64 local algebra; boundary, streaming and time-layer storage excluded",
    )


def kernel_probe():
    config = IceFlowConfig(
        nx=12,
        ny=10,
        nz=8,
        domain_size_m=(0.012, 0.010, 0.008),
        boundary_cells=1,
        water_width=5,
        water_height=7,
        floor_water_depth_cells=2,
        ice_bodies=(IceBodyConfig((8, 4, 4), (2, 2, 2)),),
        phase_warmup_steps=0,
        momentum_collision_model="cm_mrt",
        backend="cpu",
    )
    sim = IceFlow3D(config)
    sim.solid_body_id.fill(-1)
    sim.force_density.fill(0)
    sim.phase_gradient_lattice.fill(0)
    sim.velocity_lattice.fill(0)
    sim.pressure_lattice.fill(0)
    sim.water_phase.fill(1)
    sim.f.fill(0)
    sim._collide_momentum()
    base_post = sim.f_post.to_numpy()[5, 5, 4].astype(np.float64)
    matrix = np.prod(C[None] ** POWERS[:, None], axis=2)
    perturb_moments = np.zeros(27)
    perturb_moments[24] = 1 / 4096  # 220
    perturb_moments[20] = -1 / 4096  # 202
    delta = np.linalg.solve(matrix, perturb_moments).astype(np.float32)
    field = np.broadcast_to(delta, sim.shape + (27,)).copy()
    sim.f.from_numpy(field)
    sim._collide_momentum()
    post_delta = sim.f_post.to_numpy()[5, 5, 4].astype(np.float64) - base_post
    expected = (1 - config.cm_acoustic_relaxation_rate) * delta
    low = matrix[POWERS.sum(axis=1) <= 2] @ delta
    moving = np.delete(delta.astype(np.float64), REST).sum()
    np.testing.assert_allclose(post_delta, expected, atol=1e-10, rtol=1e-5)
    assert np.max(abs(low)) < 1e-15 and abs(moving) < 1e-15
    assert np.max(abs(post_delta)) > 1e-6
    momentum = dict(
        backend="actual Taichi CPU CM-MRT collision kernel",
        acoustic_relaxation_rate=config.cm_acoustic_relaxation_rate,
        input_220_delta=float(perturb_moments[24]),
        input_202_delta=float(perturb_moments[20]),
        max_low_order_moment_delta=float(np.max(abs(low))),
        moving_population_sum_delta=float(moving),
        max_post_population_delta=float(np.max(abs(post_delta))),
        max_error_vs_retained_ghost_prediction=float(
            np.max(abs(post_delta - expected))
        ),
        conclusion="Identical pressure, velocity and second moments do not determine the current collision output.",
    )

    # Full incoming distributions versus a representation constructed solely
    # from their four moments. Deliberately erase every other incoming mode.
    rng = np.random.default_rng(723)
    phi = rng.uniform(0.05, 0.95, sim.shape).astype(np.float32)
    u = rng.uniform(-0.06, 0.06, sim.shape + (3,)).astype(np.float32)
    gradient = rng.uniform(-0.2, 0.2, sim.shape + (3,)).astype(np.float32)
    h = (phi[..., None] * W + rng.normal(0, 0.002, sim.shape + (27,))).astype(
        np.float32
    )
    h[..., REST] += phi - h.sum(axis=-1, dtype=np.float64)
    zeroth = h.sum(axis=-1, dtype=np.float64)
    first = np.einsum("...q,qa->...a", h.astype(np.float64), C)
    compact = np.zeros_like(h)
    compact[..., REST] = zeroth
    for axis in range(3):
        cp = np.zeros(3)
        cp[axis] = 1
        pos = int(np.flatnonzero(np.all(C == cp, axis=1))[0])
        neg = int(np.flatnonzero(np.all(C == -cp, axis=1))[0])
        compact[..., pos] = 0.5 * first[..., axis]
        compact[..., neg] = -0.5 * first[..., axis]
    sim.water_phase.from_numpy(phi)
    sim.velocity_lattice.from_numpy(u)
    sim.phase_gradient_lattice.from_numpy(gradient)
    sim.h.from_numpy(h)
    sim._collide_phase()
    post_a = sim.h_post.to_numpy().astype(np.float64)
    flux_a = sim.phase_mass_flux.to_numpy().astype(np.float64)
    sim.h.from_numpy(compact)
    sim._collide_phase()
    post_b = sim.h_post.to_numpy().astype(np.float64)
    flux_b = sim.phase_mass_flux.to_numpy().astype(np.float64)
    err, ferr = np.max(abs(post_a - post_b)), np.max(abs(flux_a - flux_b))
    assert err < 2e-7 and ferr < 2e-7
    phase = dict(
        backend="actual Taichi CPU phase collision kernel",
        cells=int(np.prod(sim.shape)),
        max_post_population_error=float(err),
        max_phase_mass_flux_error=float(ferr),
        max_reencoded_zeroth_rounding=float(
            np.max(abs(compact.sum(axis=-1, dtype=np.float64) - zeroth))
        ),
        scope="One local collision; FP32 order/rounding changes allowed; no full HOME stepping",
    )
    return momentum, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [
        "iceflow3d/simulator.py",
        "iceflow3d/lattice.py",
        "iceflow3d/config.py",
        "iceflow2d_v3/simulator.py",
        "iceflow2d_v3/config.py",
    ]
    manifest = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}
    algebra = algebra_probe()
    momentum, phase = kernel_probe()
    for p, digest in manifest.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    cells = 450 * 150 * 90
    schemes = {
        "current_2x27_2x27": 108,
        "home_flow_only_2x10_2x27": 74,
        "phase_d3q7_only_2x27_2x7": 68,
        "home_and_d3q7_double_buffers": 34,
        "home_and_phase_four_moments_ideal_lower_budget": 28,
    }
    result = dict(
        study_date="2026-10-22",
        source_sha256=manifest,
        phase_algebra=algebra,
        momentum_counterexample=momentum,
        phase_kernel=phase,
        memory_model=dict(
            cells=cells,
            bytes_per_scalar=4,
            scope="Population/moment state only, same grid, explicit double buffers; auxiliary and boundary snapshots excluded",
            schemes={
                k: dict(
                    floats_per_cell=v,
                    gib=cells * v * 4 / 2**30,
                    reduction_vs_current=1 - v / 108,
                )
                for k, v in schemes.items()
            },
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
