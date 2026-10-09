"""Molecular dynamic viscosities in Pa s; no numerical viscosity is added.

Water: IAPWS SR6-08(2011), Eq. (7), liquid water at 0.1 MPa.
https://iapws.org/technical-guidance/release/LiquidWater.download
Air: NASA CAPE SutherlandMKS, evaluated at the prescribed air temperature.
https://nasa.github.io/cape-doc/2.3/api/cape/atm.html#cape.atm.SutherlandMKS
"""

import math


WATER_VISCOSITY_TERMS = (
    (280.68, -1.9),
    (511.45, -7.7),
    (61.131, -19.6),
    (0.45903, -40.0),
)


def water_dynamic_viscosity_pa_s(temperature_c: float) -> float:
    """IAPWS liquid-water correlation, valid from -20 to 110 degC."""

    if not math.isfinite(temperature_c) or not -20.0 <= temperature_c <= 110.0:
        raise ValueError(
            "IAPWS water viscosity requires temperatures in [-20, 110] degC"
        )
    reduced_temperature = (temperature_c + 273.15) / 300.0
    return 1.0e-6 * sum(
        coefficient * reduced_temperature**exponent
        for coefficient, exponent in WATER_VISCOSITY_TERMS
    )


def air_dynamic_viscosity_pa_s(temperature_c: float) -> float:
    """Sutherland's law for air, using NASA's SI reference constants."""

    if not math.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("air viscosity requires a temperature above absolute zero")
    temperature_k = temperature_c + 273.15
    reference_temperature_k = 273.15
    sutherland_temperature_k = 110.33333
    return (
        1.716e-5
        * (temperature_k / reference_temperature_k) ** 1.5
        * (reference_temperature_k + sutherland_temperature_k)
        / (temperature_k + sutherland_temperature_k)
    )
