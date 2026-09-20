"""Spatial Stefan benchmark using the production operators and analytic PDE.

An initially isothermal melting square is submerged far from the free surface.
The central top face is compared before sidewall diffusion reaches its centre.
Conductivity is scaled to resolve a thermal layer in a short verification run;
all conversions, density ratio, latent heat and interface transfer are retained.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot


def similarity(cfg):
    stefan = cfg.cp_w*cfg.water_temperature/cfg.latent
    lo, hi = 0.0, 2.0
    for _ in range(80):
        z = (lo+hi)/2
        if z*math.exp(z*z)*(1+math.erf(z)) < stefan/math.sqrt(math.pi): lo = z
        else: hi = z
    return (lo+hi)/2/(cfg.rho_s/cfg.rho_w)


def top_recession(sim):
    geo, cfg = sim.geometry, sim.cfg
    polygon, counts = geo.poly.to_numpy(), geo.count.to_numpy()
    # Intersection of the material contour with x=0, independently of area.
    heights = []
    for a in (cfg.nb//2-1, cfg.nb//2):
        for b in range(cfg.nb//2, cfg.nb):
            for k in range(counts[a, b]):
                p = polygon[a, b, k]+[a+.5-cfg.nb/2, b+.5-cfg.nb/2]
                q = polygon[a, b, (k+1) % counts[a, b]]+[a+.5-cfg.nb/2, b+.5-cfg.nb/2]
                if p[0] == 0: heights.append(p[1])
                if p[0]*q[0] < 0: heights.append(p[1]-p[0]*(q[1]-p[1])/(q[0]-p[0]))
    return (cfg.nb/2-max(heights))*cfg.dx if heights else cfg.ice_size/2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--end-time', type=float, default=.1)
    parser.add_argument('--reference-velocity', type=float, default=4.0)
    parser.add_argument('--transport-iterations', type=int, default=120)
    parser.add_argument('--pressure-polish-iterations', type=int, default=600)
    parser.add_argument('--pressure-iterations', type=int, default=1200)
    parser.add_argument('--check-steps', type=int, default=160)
    parser.add_argument('--no-advection', action='store_true', help='Diagnostic isolation only; the physical Stefan benchmark retains flow advection.')
    parser.add_argument('--native-graph', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    cfg = Config(nx=args.nx, ny=2*args.nx, wall_cells=max(1, round(3*args.nx/100)),
                 fixed_body=True, angle_deg=0, water_level=.04, drop_height=-.02,
                 sigma=0, gravity=0, buoyancy=False, advection=not args.no_advection, k_w=60,
                 reference_velocity=args.reference_velocity, projection_solver='multigrid', pressure_iterations=args.pressure_iterations,
                 pressure_polish_iterations=args.pressure_polish_iterations, transport_iterations=args.transport_iterations)
    sim = Simulation(cfg); sim.enable_graph(recovery=True)
    if args.native_graph:
        from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
        sim.schedule = CudaKernelSchedule(sim)
    args.output.mkdir(parents=True, exist_ok=True)
    lam, alpha = similarity(cfg), cfg.k_w/(cfg.rho_w*cfg.cp_w)
    count = round(args.end_time/cfg.dt)
    rows = []
    retries = 0
    while sim.steps < count:
        try:
            d, retried = sim.advance_checked(min(args.check_steps, count-sim.steps))
            retries += int(retried)
        except Exception:
            sim.save(args.output/'last_accepted_state.npz')
            Snapshot(sim).save(args.output/'last_accepted_restart.npz')
            (sim.last_failure_snapshot or Snapshot(sim)).save(args.output/'failure_restart.npz')
            (args.output/'failure.json').write_text(json.dumps(sim.last_failure_diagnostics or sim.diagnostics(), indent=2))
            raise
        measured = top_recession(sim)
        expected = 2*lam*math.sqrt(alpha*d['time_s'])
        row = {'time_s': d['time_s'], 'front_recession_m': measured,
               'analytic_front_recession_m': expected, 'front_error_m': measured-expected,
               'mass_relative_residual': d['condensed_mass_relative_residual'],
               'enthalpy_relative_residual': d['condensed_energy_relative_residual']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (args.output/'front.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = {'config': cfg.metadata(), 'source_sha256': sim.source_sha256,
              'execution': sim.schedule.metadata if args.native_graph else 'taichi_graph',
              'similarity_lambda': lam, 'final': rows[-1],
              'capacity_solver_retries': retries,
              'side_diffusion_length_over_half_width': math.sqrt(alpha*args.end_time)/(cfg.ice_size/2),
              'diagnostics': sim.diagnostics()}
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    sim.save(args.output/'final_state.npz')
    print(json.dumps(result, indent=2))
    if args.native_graph: sim.schedule.close()


if __name__ == '__main__': main()
