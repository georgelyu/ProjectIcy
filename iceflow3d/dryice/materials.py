"""SI carbon-dioxide properties, independent of the legacy air model.

Background viscosity: Fenghour, Wakeham & Vesovic (1998), equations 3--8,
tables 1/3 (critical enhancement is outside this atmospheric-gas model),
https://srd.nist.gov/jpcrdreprint/1.556013.pdf. No relaxation-time floor.
The fixed-density gas approximation evaluates these at the reservoir state.
"""

import math

MOLAR_MASS = 0.0440095
GAS_CONSTANT = 8.314462618
COLLISION = (0.235156, -0.491266, 0.05211155, 0.05347906, -0.01537102)


def co2_density(temperature_c, pressure_pa=101325.0):
    t = temperature_c + 273.15
    if (
        not math.isfinite(t)
        or t <= 0
        or not math.isfinite(pressure_pa)
        or pressure_pa <= 0
    ):
        raise ValueError("CO2 requires positive absolute temperature and pressure")
    return pressure_pa * MOLAR_MASS / (GAS_CONSTANT * t)


def co2_dynamic_viscosity_pa_s(temperature_c, density_kg_m3):
    t = temperature_c + 273.15
    if not math.isfinite(t) or not 200 <= t <= 1500:
        raise ValueError("Fenghour CO2 viscosity requires 200--1500 K")
    rho = density_kg_m3
    if not math.isfinite(rho) or not 0 <= rho <= 1400:
        raise ValueError("CO2 viscosity density is outside 0--1400 kg/m3")
    tr = t / 251.196
    logt = math.log(tr)
    dilute = (
        1.00697
        * math.sqrt(t)
        / math.exp(sum(a * logt**i for i, a in enumerate(COLLISION)))
    )
    excess = (
        0.004071119 * rho
        + 0.00007198037 * rho**2
        + 2.411697e-17 * rho**6 / tr**3
        + 2.971072e-23 * rho**8
        - 1.627888e-23 * rho**8 / tr
    )
    return (dilute + excess) * 1e-6
