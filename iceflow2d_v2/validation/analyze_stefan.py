"""Independent error summary for completed production Stefan benchmarks."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    summaries = []
    for path in args.runs:
        result = json.loads((path/'result.json').read_text())
        with (path/'front.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        series = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        t = series['time_s']; error = series['front_error_m']
        cfg, d = result['config'], result['diagnostics']
        label = f"{cfg['nx']} x {cfg['ny']}"
        axes[0].plot(t, 1e3*series['front_recession_m'], label=label)
        axes[1].plot(t, 1e6*error, label=label)
        axes[2].semilogy(t, np.maximum(np.abs(series['mass_relative_residual']), 1e-16), label=label+' mass')
        axes[2].semilogy(t, np.maximum(np.abs(series['enthalpy_relative_residual']), 1e-16), '--', label=label+' enthalpy')
        final = result['final']
        summaries.append({'path': str(path.resolve()), 'nx': cfg['nx'], 'ny': cfg['ny'],
            'dx_m': cfg['dx_m'], 'dt_s': cfg['dt_s'], 'time_s': float(t[-1]),
            'front_error_m': final['front_error_m'],
            'front_relative_error': final['front_error_m']/final['analytic_front_recession_m'],
            'front_rms_error_last_half_m': float(np.sqrt(np.mean(error[t >= .5*t[-1]]**2))),
            'mass_residual_peak': float(np.abs(series['mass_relative_residual']).max()),
            'enthalpy_residual_peak': float(np.abs(series['enthalpy_relative_residual']).max()),
            'capacity_solver_retries': result['capacity_solver_retries'],
            'final_mach': d['mach_max'], 'source_sha256': result['source_sha256']})
    axes[0].plot(t, 1e3*series['analytic_front_recession_m'], 'k:', label='Similarity solution')
    axes[0].set(ylabel='Central face recession (mm)', xlabel='Time (s)')
    axes[1].axhline(0, color='k', lw=.7); axes[1].set(ylabel='Front error (micrometres)', xlabel='Time (s)')
    axes[2].axhline(1e-6, color='k', ls=':', label='Benchmark budget tolerance')
    axes[2].set(ylabel='Absolute relative residual', xlabel='Time (s)', ylim=(1e-13, 3e-6))
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure.suptitle('Accelerated Stefan benchmark: fixed ice, k_water = 60 W/(m K)')
    figure.savefig(args.output/'stefan_refinement.png', dpi=180)
    report = {'runs': summaries, 'interpretation':
        'Combined grid/time refinement (dt proportional to dx); two grids alone do not establish an asymptotic order or validate the full falling-ice scene.'}
    if len(summaries) == 2 and math.isclose(summaries[0]['time_s'], summaries[1]['time_s']):
        report['endpoint_error_reduction_factor'] = abs(summaries[0]['front_error_m']/summaries[1]['front_error_m'])
        report['last_half_rms_reduction_factor'] = summaries[0]['front_rms_error_last_half_m']/summaries[1]['front_rms_error_last_half_m']
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__': main()
