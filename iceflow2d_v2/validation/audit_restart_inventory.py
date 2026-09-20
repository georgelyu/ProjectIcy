"""Independently integrate a complete checkpoint with NumPy, without Taichi."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_restart(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        cfg, initial = meta['config'], meta['initial']
        dx = cfg['width']/cfg['nx']
        dt = .1*dx/cfg['reference_velocity']
        mass_unit = cfg['rho_w']*dx**2
        energy_unit = mass_unit*cfg['cp_w']*cfg['temperature_scale']
        q = data['state.q'].astype(float)-data['state.comp'].astype(float)
        solid = data['geometry.solid'].astype(float)-data['geometry.solid_comp'].astype(float)
        ice_heat = data['geometry.ice_heat'].astype(float)-data['geometry.ice_heat_comp'].astype(float)
        ice = float(solid.sum()*cfg['rho_s']*dx**2)
        water = float(q[..., 0].sum()*mass_unit)
        air = float(q[..., 1].sum()*cfg['rho_a']*dx**2)
        water_heat = float(q[..., 2].sum()*energy_unit)
        solid_heat = float(ice_heat.sum()*energy_unit)
        air_heat = float(q[..., 3].sum()*energy_unit+air*cfg['cp_a']*cfg['air_temperature'])
        latent = water*cfg['latent']
        enthalpy = water_heat+solid_heat+latent
        air_source_heat = float((data['coupling.air_heat_total'].astype(float)
                               -data['coupling.air_heat_comp'].astype(float))*energy_unit)
        body_p = (data['rigid.momentum'].astype(float)-data['rigid.momentum_comp'].astype(float))
        body_p *= np.array([mass_unit*dx/dt, mass_unit*dx/dt, mass_unit*dx**2/dt])
        empty = bool(np.all(data['geometry.count'] == 0))
        geometry_mass = float(data['geometry.mass'])*mass_unit
        geometry_inertia = float(data['geometry.inertia'])*mass_unit*dx**2
        # Reintegrate the saved vertices, rather than trusting stored moments.
        poly, count = data['geometry.poly'].astype(float), data['geometry.count']
        vertex = np.arange(poly.shape[-2])
        following = (vertex+1) % np.maximum(count[..., None], 1)
        next_poly = np.take_along_axis(poly, following[..., None], axis=-2)
        cross = poly[..., 0]*next_poly[..., 1]-poly[..., 1]*next_poly[..., 0]
        polygon_area = float(.5*np.where(vertex < count[..., None], cross, 0).sum())
        polygon_mass = polygon_area*cfg['rho_s']*dx**2
        # mass_value multiplies f64 polygon moments by a default-f32 static
        # density ratio. Keep that known constant-rounding discrepancy visible.
        stored_density_ratio = float(np.float32(cfg['rho_s']/cfg['rho_w']))
        expected_stored_geometry_mass = polygon_area*stored_density_ratio*mass_unit
        moment = data['geometry.first_moment'].astype(float)
        origin = data['geometry.origin'].astype(float)-data['geometry.origin_comp'].astype(float)
        com = ((origin+moment/float(data['geometry.mass']))*dx).tolist() if geometry_mass > 0 else None
        state_finite = all(np.all(np.isfinite(value)) for value in (q, solid, ice_heat, body_p))
    initial_mass = initial['mass_ice_kg_m']+initial['mass_water_kg_m']
    values = {
        'mass_ice_kg_m': ice, 'mass_water_kg_m': water, 'mass_air_kg_m': air,
        'sensible_water_J_m': water_heat, 'sensible_ice_J_m': solid_heat,
        'sensible_air_J_m': air_heat, 'latent_water_J_m': latent,
        'condensed_enthalpy_J_m': enthalpy, 'all_enthalpy_J_m': enthalpy+air_heat,
        'numerical_air_heat_J_m': air_source_heat,
        'condensed_mass_relative_residual': (ice+water-initial_mass)/initial_mass,
        'condensed_energy_relative_residual': (enthalpy-initial['condensed_enthalpy_J_m'])/initial['condensed_enthalpy_J_m'],
        'all_energy_corrected_relative_residual': (
            enthalpy+air_heat-air_source_heat-initial['all_enthalpy_J_m'])/initial['all_enthalpy_J_m'],
        'air_unexpected_residual_kg_m': air-initial['mass_air_kg_m']-
            cfg['rho_a']*(initial['mass_ice_kg_m']-ice)*(1/cfg['rho_s']-1/cfg['rho_w']),
    }
    checks = {
        'finite_inventories': bool(state_finite and all(np.isfinite(v) for v in values.values())),
        'condensed_mass': abs(values['condensed_mass_relative_residual']) <= cfg['mass_tolerance'],
        'condensed_enthalpy': abs(values['condensed_energy_relative_residual']) <= cfg['energy_tolerance'],
        'air_corrected_enthalpy': abs(values['all_energy_corrected_relative_residual']) <= cfg['energy_tolerance'],
        'unexpected_air_mass': abs(values['air_unexpected_residual_kg_m']) <= 1e-6*initial['mass_air_kg_m'],
        'geometric_and_material_ice_agree': abs(polygon_mass-ice) <= 1e-12*max(initial['mass_ice_kg_m'], 1e-30),
        'stored_geometric_mass_matches_constant_precision': abs(geometry_mass-expected_stored_geometry_mass)
            <= 1e-12*max(initial['mass_ice_kg_m'], 1e-30),
        'empty_body_has_zero_mass_and_inertia': not empty or (geometry_mass == 0 and geometry_inertia == 0),
        'empty_body_has_zero_momentum': not empty or bool(np.all(body_p == 0)),
    }
    return {
        'path': str(path.resolve()), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'step': meta['steps'], 'time_s': meta['steps']*dt,
        'source_sha256': meta['source_sha256'], 'checks': checks, 'all_checks_passed': all(checks.values()),
        'inventory': values, 'no_surviving_ice_polygons': empty,
        'geometry_mass_kg_m': geometry_mass, 'geometry_inertia_kg_m': geometry_inertia,
        'independently_integrated_polygon_mass_kg_m': polygon_mass,
        'geometric_mass_density_roundoff_kg_m': geometry_mass-polygon_mass,
        'mass_matrix_density_ratio_f32': stored_density_ratio,
        'body_momentum_SI': body_p.tolist(), 'body_center_of_mass_m': com,
        'scope': 'Stored authoritative volumes and heat are integrated independently in NumPy. '
                 'This verifies an accepted checkpoint inventory, not every intervening state or physical convergence.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_restart(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    return 0 if result['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
