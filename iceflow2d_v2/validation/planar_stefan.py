"""Production Stefan slab with an independent transported-volume measurement.

The square fills the interior width and touches the bottom insulating wall,
so only its top can melt. No special thermal or phase update is used. Compare
the actual upward volume flux through a fully liquid horizontal section with
the contraction implied by measured solid mass loss. This does not read the
prescribed interfacial liquid velocity from the coupling ledger.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import taichi as ti

from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.validation.stefan_front import similarity, top_recession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, choices=[1, 2], default=1)
    parser.add_argument('--end-time', type=float, default=.1)
    parser.add_argument('--thermal-block', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--native-graph', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    n = args.refinement
    cfg = Config(nx=40*n, ny=200*n, width=.01, height=.05, wall_cells=3*n,
                 ice_size=.0085, angle_deg=0, water_level=.03, drop_height=.00075-.03,
                 fixed_body=True, gravity=0, sigma=0, buoyancy=False, k_w=60,
                 pressure_iterations=1200, pressure_polish_iterations=24,
                 projection_solver='multigrid', transport_iterations=120,
                 coupling_steps=args.thermal_block)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    sim = Simulation(cfg)
    sim.enable_graph(recovery=True)
    if args.native_graph:
        from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
        sim.schedule = CudaKernelSchedule(sim)
    args.output.mkdir(parents=True, exist_ok=True)
    count, retries = round(args.end_time/cfg.dt), 0
    lam, alpha = similarity(cfg), cfg.k_w/(cfg.rho_w*cfg.cp_w)
    # The section lies above every solid polygon throughout retreat, below air.
    j = round(.011/cfg.dx)-1
    initial_area = cfg.ice_size**2
    rows = []
    with (args.output/'front.csv').open('w', newline='') as handle:
        writer = None
        while sim.steps < count:
            try:
                d, retried = sim.advance_checked(min(round(.0005/cfg.dt), count-sim.steps))
                retries += int(retried)
            except Exception:
                Snapshot(sim).save(args.output/'last_accepted_restart.npz')
                (sim.last_failure_snapshot or Snapshot(sim)).save(args.output/'failure_restart.npz')
                (args.output/'failure.json').write_text(json.dumps(sim.last_failure_diagnostics or sim.diagnostics(), indent=2))
                raise
            flux = sim.transport.flux.to_numpy().astype(float)
            # D2Q9 unique directions north, north-east and north-west each
            # cross this section once; both diagonal links must be included.
            actual_volume_rate = flux[:, j, 1:].sum()*cfg.dx**2/cfg.dt
            dm = d['phase_mass_step_kg_m']
            expected_volume_rate = -dm*(1/cfg.rho_s-1/cfg.rho_w)/cfg.dt
            t = d['time_s']
            front = top_recession(sim)
            expected_front = 2*lam*math.sqrt(alpha*t)
            area_front = (initial_area-d['mass_ice_kg_m']/cfg.rho_s)/cfg.ice_size
            row = {'time_s': t, 'front_recession_m': front,
                'analytic_front_recession_m': expected_front, 'front_error_m': front-expected_front,
                'width_mean_front_recession_m': area_front,
                'actual_section_velocity_m_s': actual_volume_rate/cfg.ice_size,
                'mass_jump_section_velocity_m_s': expected_volume_rate/cfg.ice_size,
                'analytic_liquid_velocity_m_s': -(1-cfg.rho_s/cfg.rho_w)*lam*math.sqrt(alpha/t),
                'section_volume_residual_m2_s': actual_volume_rate-expected_volume_rate,
                'section_relative_error': (actual_volume_rate-expected_volume_rate)/abs(expected_volume_rate),
                'mass_relative_residual': d['condensed_mass_relative_residual'],
                'enthalpy_relative_residual': d['condensed_energy_relative_residual']}
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=row); writer.writeheader()
            writer.writerow(row); handle.flush()
            print(json.dumps(row), flush=True)
    result = {'config': cfg.metadata(), 'source_sha256': sim.source_sha256,
              'benchmark_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'final': rows[-1], 'capacity_solver_retries': retries,
              'section_relative_error_peak': max(abs(r['section_relative_error']) for r in rows),
              'diagnostics': sim.diagnostics(),
              'interpretation': 'Area-averaged contraction velocity is measured from actual conservative graph flux. Wall shear and material PLIC corners can perturb the pointwise planar analytic front.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    sim.save(args.output/'final_state.npz')
    np.savez_compressed(args.output/'material_contour.npz', polygons=sim.geometry.poly.to_numpy(),
                        counts=sim.geometry.count.to_numpy(), normals=sim.geometry.normal.to_numpy())
    if args.native_graph: sim.schedule.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
