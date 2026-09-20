"""Compare independent geometry and transported liquid-volume measurements."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    reports = []
    for path in args.runs:
        result = json.loads((path/'result.json').read_text())
        rows = list(csv.DictReader((path/'front.csv').open()))
        series = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
        cfg = result['config']
        t = series['time_s']; late = t >= t[-1]/2
        expected = series['analytic_front_recession_m']
        error = series['width_mean_front_recession_m']-expected
        velocity, jump = series['actual_section_velocity_m_s'], series['mass_jump_section_velocity_m_s']
        difference = velocity-jump
        # Both integrals use identical sampled times; no unobserved t=0
        # singular velocity is silently filled in or extrapolated.
        integrate = lambda values: np.sum(.5*(values[1:]+values[:-1])*np.diff(t))
        integrated_relative = integrate(difference)/abs(integrate(jump))
        relative_l2 = np.sqrt(integrate(difference**2)/integrate(jump**2))
        label = f"dx={cfg['dx_m']*1e6:.0f} micrometres"
        axes[0, 0].plot(t, 1e3*series['width_mean_front_recession_m'], label=label)
        axes[0, 1].plot(t, 1e6*error, label=label)
        axes[1, 0].plot(t, 1e3*velocity, label=label+' measured')
        axes[1, 0].plot(t, 1e3*jump, '--', linewidth=.8, label=label+' mass jump')
        axes[1, 1].plot(t, 100*series['section_relative_error'], label=label)
        reports.append({'path': str(path.resolve()), 'dx_m': cfg['dx_m'], 'dt_s': cfg['dt_s'],
            'time_s': float(t[-1]), 'center_front_final_relative_error': float(series['front_error_m'][-1]/expected[-1]),
            'width_mean_front_final_relative_error': float(error[-1]/expected[-1]),
            'width_mean_front_last_half_rms_error_m': float(np.sqrt(np.mean(error[late]**2))),
            'section_velocity_jump_relative_l2': float(relative_l2),
            'section_velocity_jump_signed_integral_error': float(integrated_relative),
            'section_velocity_jump_relative_error_peak': float(np.max(np.abs(series['section_relative_error']))),
            'sampled_integral_start_time_s': float(t[0]),
            'source_sha256': result['source_sha256'],
            'interpretation': 'Width-mean front is derived from slab area; centre front independently intersects PLIC polygons. Actual section velocity comes from transport flux, not the prescribed interface velocity.'})
    axes[0, 0].plot(t, 1e3*expected, 'k:', label='Similarity front')
    axes[1, 0].plot(t, 1e3*series['analytic_liquid_velocity_m_s'], 'k:', label='Similarity liquid velocity')
    for ax, title in zip(axes.flat, ['Width-mean top recession (mm)', 'Width-mean front error (micrometres)',
                                   'Liquid section velocity (mm/s)', 'Actual flux vs mass jump: error (%)']):
        ax.set(xlabel='Time (s)', ylabel=title); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0, 1].axhline(0, color='k', linewidth=.6)
    axes[1, 1].axhline(0, color='k', linewidth=.6)
    fig.suptitle('Planar melting slab: geometry accuracy and independent contraction-flow check')
    fig.savefig(args.output/'planar_stefan.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    report = {'runs': reports, 'full_physics_validated': False}
    if len(reports) == 2:
        report['mean_front_last_half_rms_reduction_factor'] = (
            reports[0]['width_mean_front_last_half_rms_error_m']/reports[1]['width_mean_front_last_half_rms_error_m'])
        report['source_hashes_match'] = reports[0]['source_sha256'] == reports[1]['source_sha256']
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))


if __name__ == '__main__': main()
