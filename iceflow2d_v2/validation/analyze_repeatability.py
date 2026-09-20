"""Compare a completed same-configuration repeat and heat blocks on one interval."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def read(path):
    payload = (path/'diagnostics.csv').read_bytes()
    # A live baseline can be appending its final line while this file is read.
    if not payload.endswith(b'\n'):
        payload = payload.rsplit(b'\n', 1)[0]+b'\n'
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    series = {key: np.asarray([float(row[key]) for row in rows]) for key in
              ('time_s', 'mass_ice_kg_m', 'body_origin_x_m', 'body_origin_y_m',
               'body_angle_rad', 'condensed_mass_relative_residual',
               'condensed_energy_relative_residual', 'batch_failure_flags')}
    config = json.loads((path/'config.json').read_text())
    return config, series, hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--variants', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repeat_result = json.loads((args.repeat/'result.json').read_text())
    if repeat_result['failure'] or not repeat_result['completed_requested_steps']:
        raise ValueError('The same-configuration repeat has not completed successfully.')
    paths = [args.reference, args.repeat, *args.variants]
    data = [read(path) for path in paths]
    reference_config, reference, _ = data[0]
    ignored = {'requested_steps', 'requested_time_s', 'command', 'source_sha256'}
    for index, (config, _, _) in enumerate(data[1:], 1):
        assert config['source_sha256'] == reference_config['source_sha256'], paths[index]
        changed = {key for key in reference_config if key not in ignored
                   and reference_config[key] != config.get(key)}
        assert changed <= (set() if index == 1 else {'coupling_steps'}), (paths[index], changed)
    end = min(series['time_s'][-1] for _, series, _ in data)
    times = reference['time_s'][reference['time_s'] <= end]
    mass0 = reference['mass_ice_kg_m'][0]
    records, plotted = [], []
    for path, (config, series, digest) in zip(paths, data):
        in_interval = series['time_s'] <= times[-1]
        values = {key: np.interp(times, series['time_s'], series[key]) for key in
                  ('mass_ice_kg_m', 'body_origin_x_m', 'body_origin_y_m', 'body_angle_rad')}
        differences = {key: values[key]-np.interp(times, reference['time_s'], reference[key])
                       for key in values}
        records.append({
            'path': str(path.resolve()), 'thermal_block': config['coupling_steps'],
            'recorded_csv_snapshot_sha256': digest, 'time_s': float(times[-1]),
            'differences': {key: {'final': float(delta[-1]), 'rms': float(np.sqrt(np.mean(delta**2))),
                                  'peak': float(np.max(np.abs(delta)))} for key, delta in differences.items()},
            'ice_mass_rms_difference_over_initial_mass': float(np.sqrt(np.mean(differences['mass_ice_kg_m']**2))/mass0),
            'all_compared_qoi_samples_exactly_equal': all(np.array_equal(values[key], reference[key][:len(times)])
                                                         for key in values),
            'sampled_mass_residual_peak': float(np.abs(series['condensed_mass_relative_residual'][in_interval]).max()),
            'sampled_enthalpy_residual_peak': float(np.abs(series['condensed_energy_relative_residual'][in_interval]).max()),
            'every_recorded_device_step_passed': bool(np.all(series['batch_failure_flags'][in_interval] == 0)),
        })
        plotted.append((path.name, values, differences))
    report = {
        'same_configuration_repeat': str(args.repeat.resolve()), 'common_end_time_s': float(times[-1]),
        'source_sha256': reference_config['source_sha256'], 'runs': records,
        'scope': 'One completed same-configuration repeat and thermal blocks, restricted to the same physical interval. '
                 'Equality of these macroscopic samples is not a claim about all state arrays or all future times. '
                 'One repeat does not establish an ensemble uncertainty or asymptotic convergence order.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# 同配置复跑' + ('与热块差异' if args.variants else ''), '', f'统一比较时间：0–{times[-1]:.2f} s。', '',
             '| 算例 | 热块 | 冰量差 RMS / 初始冰量 | 冰量末时差 / 初始冰量 | 转角差 RMS / rad |',
             '|---|---:|---:|---:|---:|']
    for record in records:
        lines.append(f"| {Path(record['path']).name} | {record['thermal_block']} | "
                     f"{100*record['ice_mass_rms_difference_over_initial_mass']:.6g}% | "
                     f"{100*record['differences']['mass_ice_kg_m']['final']/mass0:.6g}% | "
                     f"{record['differences']['body_angle_rad']['rms']:.6g} |")
    lines += ['', '同配置复跑只改变运行终止时间和输出位置。'
              + ('热块算例只额外改变体内导热块长度；界面热量仍按单个流体时间步计算。' if args.variants else ''),
              '一次复跑和宏观采样的一致性不等于所有状态数组逐位相同，也不构成统计误差估计或渐近收敛阶。', '']
    (args.output/'report_zh.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for index, (label, values, difference) in enumerate(plotted):
        style = '--' if index == 1 else '-'
        axes[0, 0].plot(times, values['mass_ice_kg_m']/mass0, style, label=label)
        axes[0, 1].plot(times, np.degrees(values['body_angle_rad']), style)
        axes[1, 0].plot(times, 100*difference['mass_ice_kg_m']/mass0, style)
        axes[1, 1].plot(times, np.degrees(difference['body_angle_rad']), style)
    for ax, label in zip(axes.flat, ['Ice mass / initial', 'Material angle (degrees)',
                                    'Ice mass difference (% initial)', 'Angle difference (degrees)']):
        ax.set(xlabel='Physical time (s)', ylabel=label); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Same-configuration repeat' + (' and heat blocks' if args.variants else '') + ': one common interval')
    fig.savefig(args.output/'repeatability.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    print(json.dumps({'common_end_time_s': report['common_end_time_s'],
                      'repeat_qoi_samples_equal': records[1]['all_compared_qoi_samples_exactly_equal'],
                      'mass_rms_fraction': [r['ice_mass_rms_difference_over_initial_mass'] for r in records]}, indent=2))


if __name__ == '__main__':
    main()
