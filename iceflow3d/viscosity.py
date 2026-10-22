"""Same molecular viscosity laws as iceflow2d_v3 (SI units)."""

import math

WATER_VISCOSITY_TERMS = (
    (280.68, -1.9),
    (511.45, -7.7),
    (61.131, -19.6),
    (0.45903, -40.0),
)


def water_dynamic_viscosity_pa_s(temperature_c):
    if not math.isfinite(temperature_c) or not -20 <= temperature_c <= 110:
        raise ValueError("IAPWS water viscosity requires [-20, 110] degC")
    t = (temperature_c + 273.15) / 300
    return 1e-6 * sum(c * t**p for c, p in WATER_VISCOSITY_TERMS)


def air_dynamic_viscosity_pa_s(temperature_c):
    if not math.isfinite(temperature_c) or temperature_c <= -273.15:
        raise ValueError("air temperature must exceed absolute zero")
    t = temperature_c + 273.15
    return 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.33333) / (t + 110.33333)
