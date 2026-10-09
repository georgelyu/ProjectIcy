"""Independent Fourier symbol and production-kernel check of pressure HRR.

Includes streaming and pressure recovery. The HRR velocity derivative uses
the Fourier symbol of its D2Q9 stencil; a local-only collision Jacobian would
incorrectly omit that derivative.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from iceflow2d_v3.benchmarks.check_interface_translation import TranslationProbe
from iceflow2d_v3.test_iceflow import _falling_config

C = np.array(
    [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=float,
)
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def equilibrium(u, p=0.0):
    cu = C @ u
    result = W * (3 * p + 3 * cu + 4.5 * cu**2 - 1.5 * (u @ u))
    result[0] -= 3 * p
    return result


def basis(u):
    x, y = (C - u).T
    return np.array(
        [
            np.ones(9),
            x,
            y,
            x * x + y * y,
            x * x - y * y,
            x * y,
            x * x * y,
            x * y * y,
            x * x * y * y,
        ]
    )


def collide(f, nu, model, gradient=None, acoustic=0.25, blend=0.0):
    u = f @ C
    p = 0.6 * (f[1:].sum() - 2 / 3 * (u @ u))
    b = basis(u)
    eq = equilibrium(u, p)
    m, meq = b @ f, b @ eq
    tau = 0.5 + 3 * nu
    rates = np.array([1, 1, 1, acoustic, 1 / tau, 1 / tau, 1, 1, acoustic])
    if model == "cm_hrr":
        stress = (
            -2
            * tau
            / 3
            * np.array(
                [
                    gradient[0, 0] - gradient[1, 1],
                    0.5 * (gradient[0, 1] + gradient[1, 0]),
                ]
            )
        )
        m[4:6] = meq[4:6] + blend * (m[4:6] - meq[4:6]) + (1 - blend) * stress
    post = np.linalg.solve(b, m - rates * (m - meq))
    if model == "cm_hrr":
        # Replace only raw fourth order. Dense inversion independently enforces
        # the pressure-moment constraint, without using the device inverse shift.
        raw = basis(np.zeros(2))
        rp = raw @ post
        pre, req = raw @ f, raw @ eq
        rp[8] = (
            rp[3] + (1 - acoustic) * (pre[8] - pre[3]) + acoustic * (req[8] - req[3])
        )
        post = np.linalg.solve(raw, rp)
    return post


def amplification(nu, u, k, model, blend=0.0):
    f0 = equilibrium(u)
    # Complex perturbations carry the spatial Fourier derivative explicitly.
    grad_symbol = np.sum((3 * W * np.exp(1j * (C @ k)))[:, None] * C, axis=0)
    jac = np.zeros((9, 9), dtype=complex)
    epsilon = 1e-6
    for q in range(9):
        df = np.eye(9, dtype=complex)[q]
        grad = np.outer(df @ C, grad_symbol)
        plus = collide(f0 + epsilon * df, nu, model, epsilon * grad, blend=blend)
        minus = collide(f0 - epsilon * df, nu, model, -epsilon * grad, blend=blend)
        jac[:, q] = (plus - minus) / (2 * epsilon)
    return np.exp(-1j * (C @ k))[:, None] * jac


def verify_mode(model, speed):
    nu = 1.4506451972115707e-5 * 0.1 / ((1 / 3000) * 4)
    k = np.array([-13, -13]) * np.pi / 32
    u0 = np.full(2, speed / np.sqrt(2))
    f0 = equilibrium(u0)
    eig, vec = np.linalg.eig(amplification(nu, u0, k, model))
    index = np.argmax(abs(eig))
    evec = vec[:, index]
    x, y = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    phase = k[0] * x + k[1] * y
    s = TranslationProbe(
        _falling_config(
            resolution=(64, 64),
            water_width_fraction=62 / 64,
            dx=1 / 3000,
            phase_warmup_steps=0,
            sigma=0,
            gravity=(0, 0),
            rho_air=1000,
            viscosity_air=1.4506451972115707e-5,
            momentum_collision_model=model,
        )
    )
    s.water_phase.fill(0)
    s.phase_populations.fill(0)
    s.fluid_acceleration_lattice.fill(0)
    s.reference_pressure_by_row.fill(0)
    s.momentum_populations.from_numpy(
        np.real(f0 + 1e-5 * np.exp(1j * phase)[..., None] * evec).astype(np.float32)
    )
    s._update_streamed_macroscopic_fields()
    s._update_pressure()
    projector = np.linalg.inv(vec)[index]
    rows = []
    # Strongly damped modes reach f32 roundoff quickly; fit their first 16 steps.
    fit_steps = 16 if abs(eig[index]) < 0.9 else 80
    for step in range(fit_steps + 1):
        pop = s.momentum_populations.to_numpy().astype(float)
        fourier = 2 * np.mean((pop - f0) * np.exp(-1j * phase)[..., None], axis=(0, 1))
        rows.append({"step": step, "amplitude": float(abs(projector @ fourier))})
        if step < fit_steps:
            s._advance_fluid()
    fitted = np.exp(
        np.polyfit(
            [r["step"] for r in rows], np.log([r["amplitude"] for r in rows]), 1
        )[0]
    )
    return {
        "model": model,
        "speed": speed,
        "k": k.tolist(),
        "predicted_radius": float(abs(eig[index])),
        "measured_radius": float(fitted),
        "samples": rows,
    }


def shear_viscosity(model):
    s = TranslationProbe(
        _falling_config(
            resolution=(32, 64),
            phase_warmup_steps=0,
            sigma=0,
            gravity=(0, 0),
            viscosity_water=2e-5,
            momentum_collision_model=model,
        )
    )
    k = 2 * np.pi / s.ny
    wave = np.sin(k * np.arange(s.ny))
    u = np.zeros((s.nx, s.ny, 2))
    u[..., 0] = 0.01 * wave
    cu = np.einsum("ija,qa->ijq", u, C)
    f = W * (3 * cu + 4.5 * cu**2 - 1.5 * np.sum(u * u, axis=-1)[..., None])
    s.water_phase.fill(1)
    s.phase_populations.from_numpy((f + W).astype(np.float32))
    s.momentum_populations.from_numpy(f.astype(np.float32))
    s.reference_pressure_by_row.fill(0)
    s.fluid_acceleration_lattice.fill(0)
    s._update_streamed_macroscopic_fields()
    s._update_pressure()
    rows = []
    for step in range(2049):
        if step % 256 == 0 and step:
            amp = 2 * np.mean(s.momentum_velocity_lattice.to_numpy()[..., 0] * wave)
            rows.append((step, float(amp)))
        if step < 2048:
            # Freeze pure phase for this viscosity measurement.
            s._collide_momentum()
            s.stream_momentum()
            s._update_streamed_macroscopic_fields()
            s.water_phase.fill(1)
            s._update_pressure()
    measured = (
        -np.polyfit(np.array(rows)[:, 0], np.log(np.array(rows)[:, 1]), 1)[0] / k**2
    )
    return {
        "model": model,
        "expected_nu": s._water_viscosity_lattice,
        "measured_nu": float(measured),
        "relative_error": float(measured / s._water_viscosity_lattice - 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    modes = [
        verify_mode(model, speed)
        for model in ("cm_mrt", "cm_hrr")
        for speed in (0.05, 0.2)
    ]
    shear = [shear_viscosity(model) for model in ("cm_mrt", "cm_hrr")]
    result = {"modes": modes, "shear": shear}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
