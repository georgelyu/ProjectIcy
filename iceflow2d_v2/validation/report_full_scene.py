"""Audit recorded full-scene runs without importing a numerical solver.

CSV samples include peaks/flags observed on every device step. A completed
trajectory and small conservation residuals are reported separately from
physical convergence; this report never infers A--H acceptance from a movie.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np


def read_run(path, required_time=10.0, expected_source=None, process_start=None):
    path = Path(path).resolve()
    config = json.loads((path / 'config.json').read_text())
    csv_payload = (path/'diagnostics.csv').read_bytes()
    # A live process can be in the middle of appending one diagnostic row.
    # Hash exactly the complete sample bytes that this report has inspected.
    if not csv_payload.endswith(b'\n'):
        csv_payload = csv_payload.rsplit(b'\n', 1)[0]+b'\n'
    rows = list(csv.DictReader(io.StringIO(csv_payload.decode())))
    numeric = {}
    # Ignore only an unfinished final write while monitoring a live process.
    rows = [row for row in rows if all(value is not None for value in row.values())]
    for key in rows[0]:
        if key not in ('finite', 'mechanical_energy_balance_validated'):
            numeric[key] = np.array([float(row[key]) for row in rows])
    result_path = path / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    interrupted_path = path / 'interrupted.json'
    interrupted = json.loads(interrupted_path.read_text()) if interrupted_path.exists() else None
    provenance_path = path / 'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    provenance_valid = not config.get('source_history')
    if provenance:
        provenance_valid = (provenance['verified'] and provenance['join_diagnostics_agree']
                            and provenance['source_history'] == config['source_history']
                            and all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
                                    for item in provenance['evidence']))
    first, last = rows[0], rows[-1]
    peak = lambda key: float(np.max(np.abs(numeric[key])))
    air0 = float(first['mass_air_kg_m'])
    ice0 = float(first['mass_ice_kg_m'])
    melt_ledger_error = (numeric['phase_mass_total_kg_m']-numeric['phase_mass_total_kg_m'][0]
                         -(ice0-numeric['mass_ice_kg_m']))
    melt_ledger_relative_peak = float(np.max(np.abs(melt_ledger_error)))/max(ice0, 1e-30)
    gates = {
        'completed_10s': bool(result and not result['failure'] and result['completed_requested_steps']
                              and math.isclose(float(last['time_s']), required_time,
                                               abs_tol=config['dt_s'] * .01)),
        'recorded_entire_interval': (float(first['step']) == 0 and float(first['time_s']) == 0
            and bool(np.all(np.diff(numeric['step']) > 0))
            and bool(np.allclose(numeric['time_s'], numeric['step']*config['dt_s'], rtol=0, atol=1e-12))),
        'all_sampled_states_finite': all(row['finite'] == 'True' for row in rows),
        'all_recorded_numeric_diagnostics_finite': all(np.isfinite(values).all() for values in numeric.values()),
        'every_accepted_device_step_passed': peak('batch_failure_flags') == 0,
        'interface_receivers_available': peak('failure_code') == 0,
        'condensed_mass': peak('condensed_mass_relative_residual') <= config['mass_tolerance'],
        'interface_mass_ledger': melt_ledger_relative_peak <= config['mass_tolerance'],
        'condensed_enthalpy': peak('condensed_energy_relative_residual') <= config['energy_tolerance'],
        'air_corrected_enthalpy': peak('all_energy_corrected_relative_residual') <= config['energy_tolerance'],
        'unexpected_air_mass': peak('air_unexpected_residual_kg_m') <= 1e-6 * air0,
        'capacity': max(peak('capacity_residual_max'), peak('batch_capacity_residual_peak')) <= config['capacity_tolerance'],
        'nonnegative_mass_to_roundoff': float(numeric['mass_min_lattice'].min()) >= -1e-7,
        'phase_bounds': float(numeric['phase_min'].min()) >= -1e-6 and float(numeric['phase_max'].max()) <= 1 + 1e-5,
        'low_mach': max(peak('mach_max'), peak('batch_mach_peak')) <= config['max_mach'],
        'geometry_displacement': peak('geometry_courant') <= .2,
        'water_temperature_bounds': float(numeric['temperature_water_min_C'].min()) >= -1e-5
            and float(numeric['temperature_water_max_C'].max()) <= config['water_temperature'] + 2e-4,
        'fixed_density_air': config['rho_a'] == 1.25 and config['rho_w'] == 1000 and config['rho_s'] == 917,
        'source_provenance_verified': bool(provenance_valid),
        'final_checkpoint_available': bool(result and result['completed_requested_steps']
                                           and (path/'restart.npz').exists()),
    }
    metrics = {key: peak(key) for key in (
        'condensed_mass_relative_residual', 'condensed_energy_relative_residual',
        'all_energy_corrected_relative_residual', 'air_unexpected_residual_kg_m',
        'capacity_residual_max', 'batch_capacity_residual_peak', 'gcl_residual_max',
        'geometry_area_residual_cells', 'mach_max', 'batch_mach_peak', 'geometry_courant',
        'momentum_balance_residual_x', 'momentum_balance_residual_y', 'angular_balance_residual',
        'phase_limiter_volume_cells', 'cut_cell_merge_water_volume_cells',
        'thermal_limiter_energy_J_m', 'zero_mass_momentum_l1_kg_s')}
    metrics['air_unexpected_relative_residual_peak'] = metrics['air_unexpected_residual_kg_m'] / air0
    metrics['interface_mass_ledger_relative_peak'] = melt_ledger_relative_peak
    for key in ('phase_momentum_roundoff_x', 'phase_momentum_roundoff_y', 'phase_angular_momentum_roundoff'):
        if key in numeric:
            metrics[key] = peak(key)
    source = {k: v for k, v in config['source_sha256'].items() if '/' not in k}
    if expected_source is not None:
        gates['matches_expected_numerical_source'] = source == expected_source
    summary = {
        'path': str(path), 'label': path.name, 'nx': config['nx'], 'ny': config['ny'],
        'dt_s': config['dt_s'], 'thermal_block': config['coupling_steps'],
        'time_s': float(last['time_s']), 'steps': int(float(last['step'])),
        'requested_time_s': config['requested_time_s'], 'gates': gates,
        'run_status': ('failed' if result and result['failure'] else
                       'completed' if result and result['completed_requested_steps'] else
                       'interrupted' if interrupted else 'running'),
        'numerical_10s_guards_passed': all(gates.values()),
        'full_A_H_physics_validated': False,
        'source_sha256': source, 'execution': config['execution'],
        'source_history': config.get('source_history', [{'source_sha256': source,
            'step_start': 0, 'step_end': int(float(last['step']))}]),
        'provenance': provenance,
        'capacity_solver_retries': result['capacity_solver_retries'] if result else None,
        'continuation_capacity_solver_retries': result.get('continuation_capacity_solver_retries') if result else None,
        'end_to_end_elapsed_seconds': result.get('end_to_end_elapsed_seconds') if result else None,
        'end_to_end_timing_scope': result.get('timing_scope') if result else None,
        'failure': result['failure'] if result else None,
        'rejected_diagnostics': result.get('rejected_diagnostics') if result else None,
        'peak_metrics_scope': 'Accepted CSV samples and their per-step batch peaks; rejected states, when present, are reported separately.',
        'interruption': interrupted,
        'peak_metrics': metrics,
        'initial_ice_mass_kg_m': float(first['mass_ice_kg_m']),
        'initial_water_mass_kg_m': float(first['mass_water_kg_m']),
        'initial_condensed_enthalpy_J_m': float(first['condensed_enthalpy_J_m']),
        'final': {key: float(last[key]) for key in (
            'mass_ice_kg_m', 'mass_water_kg_m', 'mass_air_kg_m',
            'air_total_relative_error', 'air_expected_increment_kg_m', 'numerical_air_heat_J_m',
            'body_origin_x_m', 'body_origin_y_m', 'body_angle_rad',
            'kinetic_energy_J_m', 'gravity_potential_J_m', 'wall_seconds', 'mlups')},
        'effective_shear_nu_range_m2_s': [float(numeric['effective_shear_nu_max_m2_s'].min()),
                                        float(numeric['effective_shear_nu_max_m2_s'].max())],
        'evidence_sha256': {name: hashlib.sha256(csv_payload if name == 'diagnostics.csv'
                                               else (path / name).read_bytes()).hexdigest()
                           for name in ('config.json', 'diagnostics.csv', 'result.json', 'provenance.json')
                           if (path / name).exists()},
    }
    summary['final']['mass_weighted_water_temperature_C'] = (
        float(last['sensible_water_J_m']) / (float(last['mass_water_kg_m']) * config['cp_w']))
    total_condensed_mass = float(first['mass_ice_kg_m']) + float(first['mass_water_kg_m'])
    summary['discrete_fully_melted_equilibrium_temperature_C'] = (
        float(first['condensed_enthalpy_J_m']) / total_condensed_mass - config['latent']) / config['cp_w']
    summary['temperature_scope'] = ('After complete melting the mass-weighted mean must match the finite-bath '
        'enthalpy prediction. This does not imply spatially uniform temperature or thermal equilibrium at 10 s.')
    if 'subgrid_viscous_active_steps' in numeric:
        active = np.flatnonzero(numeric['subgrid_viscous_active_steps'] > 0)
        summary['subgrid_viscous_coupling'] = {
            'first_active_sample_time_s': float(numeric['time_s'][active[0]]) if active.size else None,
            'active_steps': int(numeric['subgrid_viscous_active_steps'][-1]),
            'dissipated_kinetic_energy_J_m': float(numeric['subgrid_viscous_dissipated_energy_J_m'][-1]),
            'failure_peak': peak('subgrid_viscous_failure'),
            'scope': 'Conservative implicit coupling only when no SNIBB body links remain. '
                     'Recorded kinetic-energy loss is one component, not a complete mechanical-energy balance.',
        }
    if process_start is not None:
        end = result_path.stat().st_mtime if result else time.time()
        summary['end_to_end_elapsed_seconds'] = end-process_start['start_unix_s']
        summary['end_to_end_timing_scope'] = (
            'Linux process start to final result-file modification time; includes compilation, '
            'accepted work, retries and output. Live runs use report-generation time instead. '
            'Concurrent GPU processes share hardware; this is not standalone throughput.')
        summary['process_start_evidence'] = process_start
    fraction = numeric['mass_ice_kg_m']/ice0
    milestones = {}
    for remaining in (.5, .1, .01, .001):
        indices = np.flatnonzero(fraction <= remaining)
        value = None
        if indices.size:
            right = int(indices[0])
            if right == 0:
                value = float(numeric['time_s'][0])
            else:
                left = right-1
                weight = (fraction[left]-remaining)/(fraction[left]-fraction[right])
                value = float(numeric['time_s'][left]+weight*(numeric['time_s'][right]-numeric['time_s'][left]))
        milestones[str(remaining)] = value
    summary['remaining_ice_fraction_crossing_times_s'] = milestones
    summary['milestone_scope'] = 'Linear interpolation between accepted CSV samples; first crossing, not an analytic melting-time prediction.'
    summary['final_geometry'] = None
    restart = path/'restart.npz'
    if result and result['completed_requested_steps'] and restart.exists():
        with np.load(restart, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            assert meta['steps'] == int(float(last['step']))
            assert meta['source_sha256'] == source
            assert all(config[key] == value for key, value in meta['config'].items())
            assert math.isclose(meta['initial']['mass_ice_kg_m'], ice0, rel_tol=0, abs_tol=1e-15)
            count = archive['geometry.count']
            summary['final_geometry'] = {
                'material_cells_with_polygon': int(np.count_nonzero(count)),
                'no_surviving_ice_polygons': bool(np.all(count == 0)),
                'restart_sha256': hashlib.sha256(restart.read_bytes()).hexdigest(),
            }
        if __package__:
            from .audit_restart_inventory import audit_restart
        else:
            from audit_restart_inventory import audit_restart
        inventory_audit = audit_restart(restart)
        discrepancies = {key: inventory_audit['inventory'][key]-float(last[key])
                         for key in inventory_audit['inventory'] if key in last}
        # Allow only summation-order roundoff in this independent SI-unit sum.
        agrees = all(math.isclose(inventory_audit['inventory'][key], float(last[key]),
                                  rel_tol=2e-12, abs_tol=1e-12)
                     for key in discrepancies)
        summary['final_inventory_audit'] = inventory_audit
        summary['final_inventory_csv_discrepancies'] = discrepancies
        gates['final_checkpoint_inventory'] = inventory_audit['all_checks_passed']
        gates['final_checkpoint_matches_csv'] = agrees
        summary['numerical_10s_guards_passed'] = all(gates.values())
    return summary, numeric


def compare(reference, candidate):
    a, sa = reference
    b, sb = candidate
    times = sa['time_s'][sa['time_s'] <= min(a['time_s'], b['time_s'])]
    result = {'reference': a['label'], 'candidate': b['label'],
              'common_end_time_s': float(times[-1]),
              'same_final_numerical_sources': a['source_sha256'] == b['source_sha256'],
              'same_source_versions': [h['source_sha256'] for h in a['source_history']] ==
                                     [h['source_sha256'] for h in b['source_history']],
              'differences': {}}
    for key in ('mass_ice_kg_m', 'body_origin_x_m', 'body_origin_y_m', 'body_angle_rad'):
        values_a = np.interp(times, sa['time_s'], sa[key])
        values_b = np.interp(times, sb['time_s'], sb[key])
        difference = values_b - values_a
        # A square has pi/2 orientation symmetry, but retain the unwrapped
        # material angle because its angular history is a requested observable.
        result['differences'][key] = {'final': float(difference[-1]),
            'rms': float(np.sqrt(np.mean(difference ** 2))), 'peak': float(np.max(np.abs(difference)))}
    result['final_ice_mass_difference_over_initial_mass'] = (
        result['differences']['mass_ice_kg_m']['final'] / a['initial_ice_mass_kg_m'])
    result['rms_ice_mass_difference_over_initial_mass'] = (
        result['differences']['mass_ice_kg_m']['rms'] / a['initial_ice_mass_kg_m'])
    # The material origin remains stored after disappearance. Do not interpret
    # that coordinate as an observable body pose after the body is gone. Keep
    # the raw comparison above, and explicitly delimit this reporting subset.
    visible = ((np.interp(times, sa['time_s'], sa['mass_ice_kg_m']) >= .01*a['initial_ice_mass_kg_m']) &
               (np.interp(times, sb['time_s'], sb['mass_ice_kg_m']) >= .01*b['initial_ice_mass_kg_m']))
    pose = {'minimum_ice_fraction_each_run': .01, 'sample_count': int(visible.sum()),
            'last_compared_time_s': float(times[visible][-1]) if visible.any() else None,
            'scope': 'Reporting subset only: both bodies retain at least 1% of initial mass. '
                     'No simulation state or acceptance guard is changed.', 'differences': {}}
    if visible.any():
        for key in ('body_origin_x_m', 'body_origin_y_m', 'body_angle_rad'):
            delta = (np.interp(times, sb['time_s'], sb[key])-np.interp(times, sa['time_s'], sa[key]))[visible]
            pose['differences'][key] = {'final': float(delta[-1]),
                'rms': float(np.sqrt(np.mean(delta**2))), 'peak': float(np.abs(delta).max())}
    result['pose_while_both_retain_ice'] = pose
    result['milestone_time_differences_s'] = {
        key: (b['remaining_ice_fraction_crossing_times_s'][key]-value
              if value is not None and b['remaining_ice_fraction_crossing_times_s'][key] is not None else None)
        for key, value in a['remaining_ice_fraction_crossing_times_s'].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='First run is the reference.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--require-numerical-10s', action='store_true',
                        help='Exit nonzero unless every supplied run completes and passes the numerical 10 s checks.')
    parser.add_argument('--source-manifest', type=Path,
                        help='Require the recorded numerical source hashes to match this frozen manifest.')
    parser.add_argument('--process-start-times', type=Path,
                        help='Read-only Linux process start records, including compilation and retry time.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected_source = None
    if args.source_manifest:
        expected_source = {key: value for key, value in json.loads(args.source_manifest.read_text()).items()
                           if '/' not in key}
    starts = ({entry['run']: entry for entry in json.loads(args.process_start_times.read_text())}
              if args.process_start_times else {})
    runs = [read_run(path, expected_source=expected_source, process_start=starts.get(path.name))
            for path in args.runs]
    report = {'runs': [summary for summary, _ in runs],
              'comparisons': [compare(runs[0], item) for item in runs[1:]],
              'all_numerical_10s_guards_passed': all(item[0]['numerical_10s_guards_passed'] for item in runs),
              'full_A_H_physics_validated': False,
              'limits': [
                  'Numerical guards do not certify spatial or temporal convergence.',
                  'The mandated tau=0.8 stabilisation changes physical effective viscosity when dx or dt changes.',
                  'The mechanical energy budget is incomplete; conserved sensible plus latent heat does not include it.',
                  'Air total mass is allowed to change by the declared fixed-density volume source; only the unexpected residual is tested.',
                  'Liquid/ice mass and enthalpy are never normalised to pass these checks.',
                  'Recorded wall_seconds excludes compilation/output and is restored on rollback; concurrent-run MLUPS is not standalone throughput.',
                  'A CSV starting after a restart is only a partial trajectory; this report requires the complete interval from step zero.',
                  'Certified equivalent continuations retain both source versions and their transition steps; the full trajectory is not presented as a single-source run.',
              ]}
    report['audit_input_files_sha256'] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.source_manifest, args.process_start_times) if path is not None}
    report['postprocessor_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('report_full_scene.py', 'audit_restart_inventory.py')}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# 完整场景量化检查', '',
             '**数值长跑检查：' + ('全部完成并通过。**' if report['all_numerical_10s_guards_passed'] else '尚未全部完成或通过。**'),
             'A–H 全部物理验收仍须独立判断，不能由本报告自动认定。', '',
             '| 算例 | 时间 / s | 冰水质量残差峰值 | 焓残差峰值 | Mach 峰值 | 10 s 数值守卫 |',
             '|---|---:|---:|---:|---:|---|']
    for summary, _ in runs:
        m = summary['peak_metrics']
        lines.append(f"| {summary['label']} | {summary['time_s']:.3f} | {m['condensed_mass_relative_residual']:.3e} | "
                     f"{m['condensed_energy_relative_residual']:.3e} | {m['batch_mach_peak']:.4g} | "
                     f"{'通过' if summary['numerical_10s_guards_passed'] else '未完成/未通过'} |")
    lines += ['', '| 算例 | 当前剩余冰 / 初始 | 半融时间 / s | 融化 90% 时间 / s | 水质量加权温度 / °C | 空气质量增长 |',
              '|---|---:|---:|---:|---:|---:|']
    for summary, _ in runs:
        final, milestones = summary['final'], summary['remaining_ice_fraction_crossing_times_s']
        display = lambda value: f'{value:.4f}' if value is not None else '尚未达到'
        lines.append(f"| {summary['label']} | {final['mass_ice_kg_m']/summary['initial_ice_mass_kg_m']:.5g} | "
                     f"{display(milestones['0.5'])} | {display(milestones['0.1'])} | "
                     f"{final['mass_weighted_water_temperature_C']:.5f} | {100*final['air_total_relative_error']:.5f}% |")
    lines += ['', '| 与首个算例比较 | 共同时段 / s | 冰量差 RMS / 初始 | 冰量末时差 / 初始 | 位姿比较截至 / s | 材料参考点竖向差 RMS / mm | 转角差 RMS / ° |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for comparison in report['comparisons']:
        pose = comparison['pose_while_both_retain_ice']
        pd = pose['differences']
        pose_end = f"{pose['last_compared_time_s']:.3f}" if pd else '无共同样本'
        dy = f"{1e3*pd['body_origin_y_m']['rms']:.5g}" if pd else '—'
        angle = f"{180/np.pi*pd['body_angle_rad']['rms']:.5g}" if pd else '—'
        lines.append(f"| {comparison['candidate']} | {comparison['common_end_time_s']:.3f} | "
                     f"{100*comparison['rms_ice_mass_difference_over_initial_mass']:.5g}% | "
                     f"{100*comparison['final_ice_mass_difference_over_initial_mass']:.5g}% | {pose_end} | {dy} | {angle} |")
    lines += ['', '冰量比较覆盖各对照的全部共同时段。位姿表仅比较两组冰量均不低于各自初始值 1% 的采样点，避免把消失后仍存储的材料坐标当作物理刚体；这只是报告范围，不改变任何状态或验收阈值。位置为固定材料参考点，不是随非均匀融化而移动的质心。转角保留连续材料角，不按正方形对称性折叠。完整原始坐标差仍保留于 JSON。',
              '表中给出实际差异，未据此认定网格、时间步或热块精度通过。']
    lines += ['', '| 算例 | 线动量 x 余额残差峰值 / kg·s⁻¹ | 线动量 y 余额残差峰值 / kg·s⁻¹ | 角动量余额残差峰值 / kg·m·s⁻¹ |',
              '|---|---:|---:|---:|']
    for summary, _ in runs:
        metrics = summary['peak_metrics']
        lines.append(f"| {summary['label']} | {metrics['momentum_balance_residual_x']:.4e} | "
                     f"{metrics['momentum_balance_residual_y']:.4e} | {metrics['angular_balance_residual']:.4e} |")
    lines += ['', '上述为单位厚度下累计外力/壁面/接触及数值空气动量账本的余额残差，单独披露；质量和热焓守卫通过不代表这些动量误差已通过物理精度验收。']
    subgrid_runs = [(summary['label'], summary['subgrid_viscous_coupling']) for summary, _ in runs
                    if 'subgrid_viscous_coupling' in summary]
    if subgrid_runs:
        lines += ['', '| 算例 | 无链路黏性耦合首次活动采样 / s | 累计活动步数 | 累计动能耗散 / J·m⁻¹ |',
                  '|---|---:|---:|---:|']
        for label, coupling in subgrid_runs:
            first_active = coupling['first_active_sample_time_s']
            display = f'{first_active:.5f}' if first_active is not None else '未启用'
            lines.append(f"| {label} | {display} | {coupling['active_steps']} | "
                         f"{coupling['dissipated_kinetic_energy_J_m']:.6g} |")
        lines += ['', '上述局部隐式耦合仅在没有 SNIBB 固壁链路时启用；耗散单独记录，未加入热焓，不代表完整机械能余额。首次活动时间仅精确到诊断采样间隔。']
    lines += ['', '| 算例 | 运行状态 | 接受演化累计 / s | 进程实际经过 / s | 容量求解重试次数 |',
              '|---|---|---:|---:|---:|']
    for summary, _ in runs:
        elapsed = summary['end_to_end_elapsed_seconds']
        display_elapsed = f'{elapsed:.1f}' if elapsed is not None else '未记录'
        retries = summary['capacity_solver_retries']
        lines.append(f"| {summary['label']} | {summary['run_status']} | "
                     f"{summary['final']['wall_seconds']:.1f} | {display_elapsed} | "
                     f"{retries if retries is not None else '尚无完整统计'} |")
    lines += ['', '接受演化计时不含编译、输出及已回滚批次；进程实际经过时间包含这些开销。并发算例共享 GPU，不能视为独占性能。']
    lines += ['', '固定密度空气源单独核算；完整指标、源文件哈希和差异数据见 report.json。',
              '若使用等价接触搜索续算，报告保留两段源码、迁移凭证及状态数组不变的校验；不将分段运行称为单一源码版本。',
              '保留的 0.8 剪切松弛下限会改变加密/减半时间步时的物理有效黏度；不能把这种对照冒称纯粹的离散收敛率。',
              '机械能中的重力做功、黏性耗散、表面能、接触和数值体积源做功尚未形成完整余额。', '']
    (args.output / 'report_zh.md').write_text('\n'.join(lines))
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for summary, series in runs:
            label = f"{summary['nx']}x{summary['ny']}, dt={summary['dt_s']:.3g}, block={summary['thermal_block']}"
            t = series['time_s']
            axes[0, 0].plot(t, series['mass_ice_kg_m'] / summary['initial_ice_mass_kg_m'], label=label)
            retains_ice = series['mass_ice_kg_m'] >= .01*summary['initial_ice_mass_kg_m']
            axes[0, 1].plot(t, np.where(retains_ice, series['body_origin_y_m'] * 1e3, np.nan))
            axes[0, 2].plot(t, np.where(retains_ice, series['body_angle_rad'] * 180 / np.pi, np.nan))
            axes[1, 0].semilogy(t, np.maximum(np.abs(series['condensed_mass_relative_residual']), 1e-16))
            axes[1, 1].semilogy(t, np.maximum(np.abs(series['condensed_energy_relative_residual']), 1e-16))
            axes[1, 2].plot(t, series['air_total_relative_error'] * 100)
        labels = ['Ice mass / initial', 'Material origin y (mm), ice >= 1%', 'Material angle (degrees), ice >= 1%',
                  'Ice + water mass residual', 'Ice + water enthalpy residual', 'Numerical air mass growth (%)']
        for ax, label in zip(axes.flat, labels):
            ax.set(xlabel='Physical time (s)', ylabel=label)
            ax.grid(alpha=.2)
        axes[1, 0].axhline(1e-5, color='k', ls=':')
        axes[1, 1].axhline(1e-6, color='k', ls=':')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('Full falling/melting scene: recorded trajectories and conservative budgets')
        fig.savefig(args.output / 'trajectories.png', dpi=180)
        plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in ('runs', 'comparisons')}, indent=2))
    return 2 if args.require_numerical_10s and not report['all_numerical_10s_guards_passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
