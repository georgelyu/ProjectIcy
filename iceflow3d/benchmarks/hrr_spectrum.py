"""Fourier symbol of the homogeneous, unit-acoustic-rate CM-HRR update.

The symbol includes collision, D3Q27 streaming and pressure recovery. It is
not a stability proof for variable-density flow or moving boundaries.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from iceflow3d.lattice import DIRECTIONS, WEIGHTS, REST, W0


def symbol(velocity, viscosity, wavevectors, recursive=False, product=False):
    c, w = np.asarray(DIRECTIONS), np.asarray(WEIGHTS)
    u = np.asarray(velocity)
    k = np.atleast_2d(wavevectors)
    basis = []
    for axis in range(3):
        v = u[axis]
        basis.append(
            np.array(
                [
                    [0.5 * v * (v - 1), v - 0.5, 0.5],
                    [1 - v * v, -2 * v, -1],
                    [0.5 * v * (v + 1), v + 0.5, 0.5],
                ]
            )
        )
    inverse = np.einsum("ia,jb,kc->ijkabc", *basis).reshape(27, 27)
    jacobian = np.empty((27, 4))
    jacobian[:, 0] = 3 * w
    jacobian[REST, 0] -= 3
    jacobian[:, 1:] = w[:, None] * (3 * c + 9 * (c @ u)[:, None] * c - 3 * u)
    if product:
        factors = 1 + 3 * c * u + 1.5 * (3 * c * c - 1) * u * u
        for a in range(3):
            other = [b for b in range(3) if b != a]
            jacobian[:, a + 1] = (
                w
                * factors[:, other].prod(axis=1)
                * (3 * c[:, a] + 3 * (3 * c[:, a] ** 2 - 1) * u[a])
            )
    derivative = 3 * (np.exp(1j * (k @ c.T)) * w) @ c
    tau = 0.5 + 3 * viscosity
    population = np.broadcast_to(jacobian, (len(k), 27, 4)).astype(complex).copy()
    lift = (
        inverse[:, 26]
        if np.all(u == 0)
        else np.prod(np.where(c == 0, -1.0, 0.5), axis=1)
    )
    for a in range(3):
        stress = (
            -tau
            / 3
            * (
                np.einsum("i,nj->nij", np.eye(3)[a], derivative)
                + np.einsum("ni,j->nij", derivative, np.eye(3)[a])
            )
        )
        stress -= np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
        moment = np.zeros((len(k), 27), complex)
        for axis, index in enumerate((18, 6, 2)):
            moment[:, index] = stress[:, axis, axis]
        for i, j, index in ((0, 1, 12), (0, 2, 10), (1, 2, 4)):
            moment[:, index] = stress[:, i, j]
        if recursive:
            for i, j, index in ((0, 1, 24), (0, 2, 20), (1, 2, 8)):
                moment[:, index] = (stress[:, i, i] + stress[:, j, j]) / 3
            for i, j, index in ((1, 2, 22), (0, 2, 16), (0, 1, 14)):
                moment[:, index] = stress[:, i, j] / 3
        perturbation = (1 - 1 / tau) * moment @ inverse.T
        perturbation -= (perturbation.sum(axis=1) - perturbation[:, REST])[
            :, None
        ] * lift
        population[:, :, a + 1] += perturbation
    recovery = np.empty((4, 27))
    recovery[1:] = c.T
    recovery[0] = (1 - 3 * W0 * (c @ u)) / (3 * (1 - W0))
    recovery[0, REST] -= 1 / (3 * (1 - W0))
    if product:
        recovery[0] = (1 + c @ jacobian[REST, 1:]) / (3 * (1 - W0))
        recovery[0, REST] -= 1 / (3 * (1 - W0))
    return np.einsum("iq,nq,nqj->nij", recovery, np.exp(-1j * (k @ c.T)), population)


def population_symbol(
    velocity,
    viscosity,
    wavevectors,
    bulk=1.0,
    recursive=True,
    product=True,
    cubic_correction=False,
):
    """Full population symbol, including memory of acoustic moments."""
    c, w = np.asarray(DIRECTIONS), np.asarray(WEIGHTS)
    u = np.asarray(velocity)
    k = np.atleast_2d(wavevectors)
    powers = np.array(list(np.ndindex(3, 3, 3)))
    transform = np.prod((c[None] - u) ** powers[:, None], axis=-1)
    inverse = np.linalg.inv(transform)
    eq = np.empty((27, 4))
    eq[:, 0] = 3 * w
    eq[REST, 0] -= 3
    eq[:, 1:] = w[:, None] * (3 * c + 9 * (c @ u)[:, None] * c - 3 * u)
    if product:
        factors = 1 + 3 * c * u + 1.5 * (3 * c * c - 1) * u * u
        for a in range(3):
            eq[:, a + 1] = (
                w
                * factors[:, [b for b in range(3) if b != a]].prod(axis=1)
                * (3 * c[:, a] + 3 * (3 * c[:, a] ** 2 - 1) * u[a])
            )
    recovery = np.vstack([(1 + c @ eq[REST, 1:]) / (3 * (1 - W0)), c.T])
    recovery[0, REST] -= 1 / (3 * (1 - W0))
    eq_pop = eq @ recovery
    kinetic = transform @ (np.eye(27) - eq_pop)
    derivative = 3 * (np.exp(1j * (k @ c.T)) * w) @ c
    tau = 0.5 + 3 * viscosity
    moment = np.zeros((len(k), 27, 27), complex)
    trace = (kinetic[18] + kinetic[6] + kinetic[2]) / 3
    for index in (18, 6, 2):
        moment[:, index] = (1 - bulk) * trace
    for index in (24, 20, 8, 26):
        moment[:, index] += (1 - bulk) * kinetic[index]
    for a in range(3):
        stress = (
            -tau
            / 3
            * (
                np.einsum("i,nj->nij", np.eye(3)[a], derivative)
                + np.einsum("ni,j->nij", derivative, np.eye(3)[a])
            )
        )
        stress -= np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
        shear = np.zeros((len(k), 27), complex)
        for axis, index in enumerate((18, 6, 2)):
            shear[:, index] = (1 - 1 / tau) * stress[:, axis, axis]
        for i, j, index in ((0, 1, 12), (0, 2, 10), (1, 2, 4)):
            shear[:, index] = (1 - 1 / tau) * stress[:, i, j]
        if recursive:
            for i, j, index in ((0, 1, 24), (0, 2, 20), (1, 2, 8)):
                shear[:, index] = (
                    (bulk - 1 / tau) * (stress[:, i, i] + stress[:, j, j]) / 3
                )
            for i, j, index in ((1, 2, 22), (0, 2, 16), (0, 1, 14)):
                shear[:, index] = (1 - 1 / tau) * stress[:, i, j] / 3
        if cubic_correction:
            correction = -1.5 * u[a] ** 2 * derivative[:, a]
            shear[:, (18, 6, 2)[a]] += correction
            for axes, index in (((0, 1), 24), ((0, 2), 20), ((1, 2), 8)):
                if a in axes:
                    shear[:, index] += correction / 3
            shear[:, 26] += correction / 9
        moment += shear[:, :, None] * recovery[a + 1]
    post = eq_pop + inverse @ moment
    moving = np.ones(27)
    moving[REST] = 0
    target = (1 - bulk) * moving + bulk * (moving @ eq_pop)
    difference = target - np.einsum("q,nqj->nj", moving, post)
    lift = np.prod(np.where(c == 0, -1.0, 0.5), axis=1)
    post += lift[None, :, None] * difference[:, None]
    return np.exp(-1j * (k @ c.T))[:, :, None] * post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wave = np.linspace(-np.pi, np.pi, 25)
    k = np.array(np.meshgrid(wave, wave, wave)).reshape(3, -1).T
    result = []
    for nu in (0.0000236, 0.001088):
        for speed in (0, 0.02, 0.05, 0.1, 0.2):
            velocity = np.full(3, speed / np.sqrt(3))
            radii = np.abs(np.linalg.eigvals(symbol(velocity, nu, k))).max(axis=1)
            index = radii.argmax()
            result.append(
                {
                    "viscosity_lattice": nu,
                    "speed_lattice": speed,
                    "max_amplification": float(radii[index]),
                    "wavevector": k[index].tolist(),
                }
            )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
