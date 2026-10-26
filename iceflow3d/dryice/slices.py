"""Small, read-only GPU exports of the central physical z grid layer.

Arrays use (y, x[, component]) order; velocity components remain (x, y, z).
For an even grid the upper of the two central cell layers is selected, without
averaging across fluid/solid or water/gas interfaces. Its exact z is recorded.
"""

import numpy as np
import taichi as ti


@ti.kernel
def _extract(
    s: ti.template(),
    z: ti.i32,
    phase: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ids: ti.types.ndarray(dtype=ti.i32, ndim=2),
    velocity: ti.types.ndarray(dtype=ti.f32, ndim=3),
    temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    water_temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for y, x in ti.ndrange(s.ny - 2, s.nx - 2):
        cell = ti.Vector([x + 1, y + 1, z])
        body = s.solid_body_id[cell]
        phi = s.water_phase[cell]
        phase[y, x], ids[y, x] = phi, body
        for a in ti.static(range(3)):
            velocity[y, x, a] = s.velocity_lattice[cell][a] * s.scales.velocity_m_s
        tw = ti.cast(s.water_reference_temperature, ti.f32)
        if s.water_volume[cell] > 1e-30:
            tw = s._water_temperature(cell)
        water_temperature[y, x] = tw
        temp = ti.cast(s.config.thermal.initial_air_temperature_c, ti.f32)
        if body == -1 and phi >= 0.5:
            temp = tw
        elif body >= 0:
            material = ti.cast(
                ti.round(s._material_grid(ti.cast(cell, ti.f32) + 0.5, body)), ti.i32
            )
            material = ti.min(s.material_size[body] - 1, ti.max(0, material))
            temp = s._ice_temperature(body, material)
        temperature[y, x] = temp


def central_z_slice(sim):
    """Return only a 2D slice, without copying the full 3D fields to the host."""
    shape = sim.ny - 2, sim.nx - 2
    phase, temperature, water_temperature = (
        np.empty(shape, np.float32) for _ in range(3)
    )
    ids = np.empty(shape, np.int32)
    velocity = np.empty((*shape, 3), np.float32)
    z = sim.nz // 2
    _extract(sim, z, phase, ids, velocity, temperature, water_temperature)
    # An adiabatic wall has no thermal state in this model.
    temperature[ids == -2] = np.nan
    return {
        "water_phase": phase,
        "solid_body_id": ids,
        "velocity_m_s": velocity,
        "temperature_c": temperature,
        "water_temperature_c": water_temperature,
        "time_s": sim.physical_time_s,
        "step": sim.steps,
        "dx_m": sim.dx,
        "x_m": (np.arange(shape[1]) + 0.5) * sim.dx,
        "y_m": (np.arange(shape[0]) + 0.5) * sim.dx,
        "z_m": (z - 0.5) * sim.dx,
        "z_index_with_halo": z,
        "requested_z_m": sim.config.physical_size_m[2] / 2,
        "array_order": "y,x; velocity components x,y,z",
        "gas_temperature_model": "fixed isothermal reservoir",
        "gas_temperature_c": sim.config.thermal.initial_air_temperature_c,
    }
