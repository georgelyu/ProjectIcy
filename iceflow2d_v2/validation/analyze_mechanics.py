"""Summarize completed mechanics cases without loading the numerical solver."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(path):
    path = path.resolve()
    payload = (path / 'result.json').read_bytes()
    result = json.loads(payload)
    cfg, history = result['config'], result['history']
    values = {key: np.asarray([row[key] for row in history], dtype=float)
              for key in history[0] if not isinstance(history[0][key], bool)}
    peak = lambda key: float(np.abs(values[key]).max())
    checks = {
        'recorded_states_finite': all(row['finite'] for row in history),
        'every_device_step_passed': peak('batch_failure_flags') == 0,
        'receivers_available': peak('failure_code') == 0,
        'mass': peak('condensed_mass_relative_residual') <= cfg['mass_tolerance'],
        'enthalpy': peak('condensed_energy_relative_residual') <= cfg['energy_tolerance'],
        'capacity': max(peak('capacity_residual_max'), peak('batch_capacity_residual_peak'))
                    <= cfg['capacity_tolerance'],
        'mach': max(peak('mach_max'), peak('batch_mach_peak')) <= cfg['max_mach'],
        'ordered_samples': bool(np.all(np.diff(values['step']) > 0)),
        'time_matches_step': bool(np.allclose(values['time_s'], values['step']*cfg['dt_s'],
                                              rtol=0, atol=1e-12)),
        'no_failure_artifact': not (path / 'failure.json').exists(),
    }
    summary = {
        'case': result['case'], 'nx': cfg['nx'], 'ny': cfg['ny'],
        'time_s': result['final']['time_s'], 'path': str(path),
        'numerical_guards_passed': all(checks.values()), 'checks': checks,
        'mass_residual_peak': peak('condensed_mass_relative_residual'),
        'enthalpy_residual_peak': peak('condensed_energy_relative_residual'),
        'mach_peak': max(peak('mach_max'), peak('batch_mach_peak')),
        'source_sha256': result['source_sha256'],
        'result_sha256': hashlib.sha256(payload).hexdigest(),
        'capacity_solver_retries': result['capacity_solver_retries'],
    }
    for key in ('maximum_vertical_displacement_m', 'maximum_angle_change_rad',
                'initial_buoyancy_force_N_m', 'archimedes_force_N_m',
                'relative_buoyancy_error', 'fall_distance_m', 'zero_drag_fall_distance_m',
                'relative_freefall_error_including_gas_drag', 'phase_max_change', 'max_speed_m_s'):
        if key in result:
            summary[key] = result[key]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(path) for path in args.runs]
    report = {
        'runs': runs,
        'same_numerical_sources': all(run['source_sha256'] == runs[0]['source_sha256'] for run in runs),
        'all_numerical_guards_passed': all(run['numerical_guards_passed'] for run in runs),
        'full_mechanical_accuracy_validated': False,
        'limits': [
            'Freefall retains gas drag; the analytic comparison omits drag but includes gas buoyancy.',
            'Initial Archimedes force agreement is not a validation of added mass or moving-boundary torque.',
            'Report equilibrium drift and parasitic currents even when they increase on the finer grid.',
            'The prescribed shear relaxation floor changes effective physical viscosity with resolution.',
            'A completed short guarded run is not proof of long-time mechanical convergence.',
        ],
    }
    lines = ['# 当前版本短程力学基准', '',
             '下表保留实际误差；数值守卫通过不代表力学精度全部通过。', '',
             '| 算例 | 网格 | 时间 / s | 位移峰值 / μm | 转角峰值 / rad | 定量对照 |',
             '|---|---|---:|---:|---:|---|']
    for run in runs:
        comparison = ''
        if 'relative_freefall_error_including_gas_drag' in run:
            comparison = f"含气阻落距与无阻力解析差 {100*run['relative_freefall_error_including_gas_drag']:.4f}%"
        elif 'relative_buoyancy_error' in run:
            comparison = f"初始阿基米德浮力相对误差 {run['relative_buoyancy_error']:.5g}"
        elif 'max_speed_m_s' in run:
            comparison = f"末时寄生速度 {1e3*run['max_speed_m_s']:.4f} mm/s；相场最大变化 {run['phase_max_change']:.5g}"
        lines.append(f"| {run['case']} | {run['nx']}×{run['ny']} | {run['time_s']:.3f} | "
                     f"{1e6*run['maximum_vertical_displacement_m']:.4f} | "
                     f"{run['maximum_angle_change_rad']:.5g} | {comparison} |")
    lines += ['', '固定剪切松弛下限造成网格间有效黏度不同；两档对照不能直接解释为纯离散收敛率。',
              '原始结果路径、源码哈希、结果文件摘要与守卫逐项检查见 report.json。', '']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output / 'report_zh.md').write_text('\n'.join(lines))
    print(json.dumps({key: value for key, value in report.items() if key != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
