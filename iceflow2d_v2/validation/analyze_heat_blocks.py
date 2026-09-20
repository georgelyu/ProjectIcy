"""Isolate heat-block sensitivity in completed fixed-body planar Stefan runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    result = json.loads((path/'result.json').read_text())
    with (path/'front.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    series = {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}
    if not all(np.isfinite(value).all() for value in series.values()):
        raise ValueError(f'Nonfinite planar data: {path}')
    if (path/'failure.json').exists():
        raise ValueError(f'Failed planar run: {path}')
    return result, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='Thermal block 1 first.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = [read(path) for path in args.runs]
    base, reference = data[0]
    assert base['config']['coupling_steps'] == 1
    assert base['config']['fixed_body']
    times = reference['time_s']
    records = []
    for path, (result, series) in zip(args.runs, data):
        assert result['source_sha256'] == base['source_sha256']
        changed = {key for key in base['config']
                   if base['config'][key] != result['config'].get(key)}
        assert changed <= {'coupling_steps'}, (path, changed)
        assert np.array_equal(series['time_s'], times), path
        assert result['diagnostics']['time_s'] == float(times[-1]), path
        cfg = result['config']
        guards = {
            'sampled_mass': bool(np.abs(series['mass_relative_residual']).max() <= cfg['mass_tolerance']),
            'sampled_enthalpy': bool(np.abs(series['enthalpy_relative_residual']).max() <= cfg['energy_tolerance']),
            'final_device_batch': result['diagnostics']['batch_failure_flags'] == 0,
        }
        differences = {}
        for key in ('front_recession_m', 'width_mean_front_recession_m'):
            delta = series[key]-reference[key]
            differences[key] = {'final_m': float(delta[-1]), 'peak_m': float(np.abs(delta).max()),
                                'rms_m': float(np.sqrt(np.mean(delta**2)))}
        area_delta = series['width_mean_front_recession_m']-reference['width_mean_front_recession_m']
        records.append({
            'path': str(path.resolve()), 'thermal_block': cfg['coupling_steps'],
            'time_s': float(times[-1]), 'guards': guards,
            'all_numerical_checks_passed': all(guards.values()),
            'front_differences': differences,
            'remaining_ice_difference_over_initial_mass': {
                'final': float(-area_delta[-1]/cfg['ice_size']),
                'rms': float(np.sqrt(np.mean(area_delta**2))/cfg['ice_size']),
                'peak': float(np.abs(area_delta).max()/cfg['ice_size']),
            },
            'final_center_front_relative_analytic_error': float(series['front_error_m'][-1]/series['analytic_front_recession_m'][-1]),
            'final_mean_front_relative_analytic_error': float(series['width_mean_front_recession_m'][-1]/series['analytic_front_recession_m'][-1]-1),
            'mass_residual_peak': float(np.abs(series['mass_relative_residual']).max()),
            'enthalpy_residual_peak': float(np.abs(series['enthalpy_relative_residual']).max()),
            'evidence_sha256': {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                               for name in ('result.json', 'front.csv')},
        })
    report = {
        'runs': records, 'source_sha256': base['source_sha256'],
        'all_numerical_checks_passed': all(item['all_numerical_checks_passed'] for item in records),
        'scope': 'Fixed planar ice, accelerated water conductivity 60 W/(m K), identical grid and fluid dt; '
                 'only bulk conduction block length changes. This isolates this benchmark and does not '
                 'establish thermal-block convergence for the freely moving production scene.',
        'postprocessor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 固定平面融化的热块对照', '',
             f'全部比较使用相同网格、流体时间步及数值源码，结束时间 {times[-1]:g} s；仅体内导热块长度不同。',
             '冰体固定，水导热率为 60 W/(m·K)。这项隔离实验不能代替原物性自由落冰的长时热块验收。', '',
             '| 热块 | 中心前沿末时差 / μm | 平均前沿末时差 / μm | 冰量差 RMS / 初始冰量 | 冰量末时差 / 初始冰量 |',
             '|---:|---:|---:|---:|---:|']
    for item in records:
        d, mass = item['front_differences'], item['remaining_ice_difference_over_initial_mass']
        lines.append(f"| {item['thermal_block']} | {1e6*d['front_recession_m']['final_m']:.6g} | "
                     f"{1e6*d['width_mean_front_recession_m']['final_m']:.6g} | "
                     f"{100*mass['rms']:.6g}% | {100*mass['final']:.6g}% |")
    lines += ['', '所有差值以热块 1 为参考。共同存在的空间/界面误差不因块长度对照接近而消失；解析前沿误差、来源哈希及质量/焓检查另见 report.json。', '']
    (args.output/'report_zh.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for result, series in data:
        label = f"block={result['config']['coupling_steps']}"
        axes[0].plot(times, series['width_mean_front_recession_m']*1e3, label=label)
        axes[1].plot(times, (series['width_mean_front_recession_m']-reference['width_mean_front_recession_m'])*1e6, label=label)
    axes[0].plot(times, reference['analytic_front_recession_m']*1e3, 'k--', label='Planar similarity')
    axes[0].set(ylabel='Width-mean recession (mm)')
    axes[1].set(ylabel='Width-mean difference from block 1 (micrometres)')
    for ax in axes:
        ax.set(xlabel='Physical time (s)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Fixed-body planar melting: bulk heat-block sensitivity')
    fig.savefig(args.output/'heat_blocks.png', dpi=180); plt.close(fig)
    print(json.dumps({'all_numerical_checks_passed': report['all_numerical_checks_passed'],
                      'mass_difference_rms_fraction': [r['remaining_ice_difference_over_initial_mass']['rms'] for r in records]}, indent=2))


if __name__ == '__main__':
    main()
