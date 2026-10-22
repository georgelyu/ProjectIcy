"""Homogeneous CM-MRT Fourier analysis in an independent moment basis.

This linear analysis does not establish multiphase or boundary stability.
The viscosity check uses the long-wave shear eigenvalue, not a fitted input
relaxation parameter. Run with ``python -m iceflow3d.benchmarks.mrt_spectrum``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from iceflow3d.lattice import DIRECTIONS, REST, WEIGHTS
from iceflow3d.viscosity import (
    air_dynamic_viscosity_pa_s,
    water_dynamic_viscosity_pa_s,
)


def population_symbol(velocity, viscosity, wavevectors, bulk=0.6, corrected=True):
    c, w = np.asarray(DIRECTIONS), np.asarray(WEIGHTS)
    u = np.asarray(velocity)
    k = np.atleast_2d(wavevectors)
    powers = np.array(list(np.ndindex(3, 3, 3)))
    central = np.prod((c[None] - u) ** powers[:, None], axis=-1)
    basis = central.copy()
    trace = (central[18] + central[6] + central[2]) / 3
    basis[18] = trace
    basis[6] = central[18] - central[6]
    basis[2] = central[18] - central[2]
    rates = np.ones(27)
    rates[[6, 2, 12, 10, 4]] = 1 / (0.5 + 3 * viscosity)
    rates[[18, 24, 20, 8, 26]] = bulk
    if corrected:
        # Independent fourth-order coordinates remove the second-order
        # deviatoric contribution. The raw moving-population sum is the
        # pressure coordinate and replaces the sixth-order monomial.
        for i, j, index in ((18, 6, 24), (18, 2, 20), (6, 2, 8)):
            basis[index] -= (central[i] + central[j] - 2 * trace) / 3
        for i, index in ((4, 22), (10, 16), (12, 14)):
            basis[index] -= central[i] / 3
        basis[26] = 1
        basis[26, REST] = 0
    eq = np.empty((27, 4))
    eq[:, 0] = 3 * w
    eq[REST, 0] -= 3
    factors = 1 + 3 * c * u + 1.5 * (3 * c * c - 1) * u * u
    for a in range(3):
        eq[:, a + 1] = (
            w
            * factors[:, [b for b in range(3) if b != a]].prod(axis=1)
            * (3 * c[:, a] + 3 * (3 * c[:, a] ** 2 - 1) * u[a])
        )
    recovery = np.vstack([(1 + c @ eq[REST, 1:]) / (3 * (1 - w[REST])), c.T])
    recovery[0, REST] -= 1 / (3 * (1 - w[REST]))
    equilibrium = eq @ recovery
    relaxation = np.linalg.solve(basis, rates[:, None] * basis)
    collision = np.eye(27) - relaxation @ (np.eye(27) - equilibrium)
    return np.exp(-1j * (k @ c.T))[:, :, None] * collision


def shear_mode(viscosity, cells, bulk=0.6):
    k = 2 * np.pi / cells
    matrix = population_symbol(np.zeros(3), viscosity, [[k, 0, 0]], bulk)[0]
    values, vectors = np.linalg.eig(matrix)
    cy = np.asarray(DIRECTIONS)[:, 1]
    selected = np.argmax(abs(cy @ vectors) / (1e-12 + abs(1 - values)))
    vector = vectors[:, selected]
    vector /= cy @ vector
    return values[selected], vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wave = np.linspace(-np.pi, np.pi, 13)
    k = np.array(np.meshgrid(wave, wave, wave)).reshape(3, -1).T
    spectra, viscosities = [], []
    for corrected in (False, True):
        for speed in (0, 0.02, 0.05, 0.1):
            nu = water_dynamic_viscosity_pa_s(90) / 1000 * 25
            values = population_symbol(
                np.full(3, speed / np.sqrt(3)), nu, k, corrected=corrected
            )
            radii = abs(np.linalg.eigvals(values)).max(axis=1)
            i = radii.argmax()
            spectra.append(
                dict(
                    corrected=corrected,
                    speed=speed,
                    radius=float(radii[i]),
                    wavevector=k[i].tolist(),
                )
            )
    for phase, temperature, mu, density in (
        ("water", 90, water_dynamic_viscosity_pa_s(90), 1000),
        ("water", 20, water_dynamic_viscosity_pa_s(20), 1000),
        ("water", 0, water_dynamic_viscosity_pa_s(0), 1000),
        ("air", 20, air_dynamic_viscosity_pa_s(20), 1.25),
    ):
        nu = mu / density * 25  # dx=1 mm, dt=25 us, unchanged dam-break scales.
        for cells in (24, 48, 96):
            value, _ = shear_mode(nu, cells)
            measured = -np.log(abs(value)) / (2 * np.pi / cells) ** 2
            viscosities.append(
                dict(
                    phase=phase,
                    temperature_c=temperature,
                    mu_pa_s=mu,
                    nu_lattice=nu,
                    wavelength_cells=cells,
                    shear_nu_lattice=float(measured),
                    ratio=float(measured / nu),
                )
            )
    result = dict(spectra=spectra, shear_viscosity=viscosities)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
