"""Build auditable full trajectories from certified equivalent source segments."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as handle:
        return [r for r in csv.DictReader(handle) if all(v is not None for v in r.values())]


def collect(entry, root, certificate, certificate_path, starts):
    previous, current = Path(entry['previous_path']), Path(entry['path'])
    if not (current/'diagnostics.csv').exists():
        return {'name': entry['name'], 'status': 'initializing'}
    migration_path = current/'input_restart.migration.json'
    migration = json.loads(migration_path.read_text())
    assert migration['certificate_sha256'] == sha(certificate_path)
    assert migration['input_sha256'] == sha(previous/'restart.npz')
    assert migration['output_sha256'] == sha(current/'input_restart.npz')
    assert migration['old_source_sha256'] == certificate['reference_sources']
    assert migration['new_source_sha256'] == certificate['optimized_sources']
    old_config = json.loads((previous/'config.json').read_text())
    new_config = json.loads((current/'config.json').read_text())
    def core(config):
        return {k: v for k, v in config['source_sha256'].items() if '/' not in k}
    assert core(old_config) == certificate['reference_sources']
    assert core(new_config) == certificate['optimized_sources']
    for key in old_config:
        if key not in {'source_sha256', 'command', 'execution'}:
            assert old_config[key] == new_config[key], key
    old_rows, new_rows = read_csv(previous/'diagnostics.csv'), read_csv(current/'diagnostics.csv')
    if not new_rows:
        return {'name': entry['name'], 'status': 'initializing'}
    step = migration['steps']
    assert int(new_rows[0]['step']) == step
    matching = [row for row in old_rows if int(row['step']) == step]
    assert len(matching) == 1
    # Check shared diagnostics at the join independently of the array digest.
    for key in ('mass_ice_kg_m', 'mass_water_kg_m', 'mass_air_kg_m', 'condensed_enthalpy_J_m',
                'phase_mass_total_kg_m', 'body_origin_x_m', 'body_origin_y_m', 'body_angle_rad'):
        assert math.isclose(float(matching[0][key]), float(new_rows[0][key]), rel_tol=2e-12, abs_tol=1e-14), key
    selected = [row for row in old_rows if int(row['step']) < step]
    combined = selected+new_rows
    assert int(combined[0]['step']) == 0
    # A failed CLI repeats its final accepted row; retain it only once and keep
    # the rejected diagnostics in the original result, never in accepted peaks.
    deduplicated = []
    for row in combined:
        if deduplicated and row['step'] == deduplicated[-1]['step']:
            assert row == deduplicated[-1]
            continue
        deduplicated.append(row)
    combined = deduplicated
    assert all(int(b['step']) > int(a['step']) for a, b in zip(combined, combined[1:]))
    target = root/f"scene_10s_{entry['name']}"
    target.mkdir(exist_ok=True)
    columns = list(new_rows[0])
    for row in selected:
        row['geometry_courant_origin_bound'] = row['geometry_courant']
        assert set(row) == set(columns)
    with (target/'diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(combined)
    history = [
        {'path': str(previous), 'step_start': 0, 'step_end': step,
         'source_sha256': core(old_config), 'geometry_guard': 'initial-box conservative speed bound'},
        {'path': str(current), 'step_start': step, 'step_end': int(new_rows[-1]['step']),
         'source_sha256': core(new_config), 'geometry_guard': 'maximum surviving PLIC vertex speed'},
    ]
    metadata = {**new_config, 'source_history': history,
                'equivalent_transition_certificate': str(certificate_path),
                'equivalent_transition_certificate_sha256': sha(certificate_path),
                'data_kind': 'derived full trajectory; original accepted records retained in segment directories'}
    (target/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    evidence = [previous/'diagnostics.csv', previous/'config.json', current/'diagnostics.csv',
                current/'config.json', migration_path, certificate_path]
    provenance = {'verified': True, 'source_history': history, 'migration': migration,
                  'join_diagnostics_agree': True,
                  'discarded_original_tail_rows': sum(int(r['step']) > step for r in old_rows),
                  'joined_rows': len(combined),
                  'evidence': [{'path': str(p), 'sha256': sha(p)} for p in evidence],
                  'guard_note': 'Before the transition geometry_courant is the stricter initial-box bound; '
                                'afterwards it measures surviving vertices. Threshold remains 0.2 lattice units.',
                  'retry_note': 'Original interrupted process host retry count was not checkpointed. '
                                'Only the continuation count is available; do not interpret it as the full-run total.'}
    result = None
    if (current/'result.json').exists():
        result = json.loads((current/'result.json').read_text())
        result['continuation_capacity_solver_retries'] = result['capacity_solver_retries']
        result['capacity_solver_retries'] = None
        result['source_history'] = history
        result['provenance_verified'] = True
        result['end_to_end_elapsed_seconds'] = ((current/'result.json').stat().st_mtime-starts[previous.name])
        result['timing_scope'] = ('Original process start through continuation result write, including '
                                  'compilation, output, discarded tail, migration gap, and solver retries; '
                                  'runs shared a GPU concurrently.')
        (target/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    for directory, old in ((previous, True), (current, False)):
        for frame in directory.glob('frame_*.npz'):
            frame_step = int(frame.stem.split('_')[1])
            if (old and frame_step <= step) or (not old and frame_step > step):
                link = target/frame.name
                if link.is_symlink():
                    assert link.resolve() == frame.resolve()
                elif not link.exists():
                    link.symlink_to(frame.resolve())
    for name in ('restart.npz', 'final_state.npz', 'rejected_restart.npz'):
        frame = current/name
        if frame.exists() and not (target/name).exists():
            (target/name).symlink_to(frame.resolve())
    (target/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    if (current/'interrupted.json').exists():
        (target/'interrupted.json').write_text((current/'interrupted.json').read_text())
    return {'name': entry['name'], 'path': str(target), 'time_s': float(combined[-1]['time_s']),
            'completed': bool(result and result['completed_requested_steps']),
            'failure': result['failure'] if result else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    certificate_path = Path(manifest['certificate'])
    certificate = json.loads(certificate_path.read_text())
    assert certificate['validated']
    for item in certificate['evidence']:
        assert sha(Path(item['path'])) == item['sha256']
    starts = {item['run']: item['start_unix_s'] for item in json.loads(
        (root/'full_receiver_process_start_times.json').read_text())}
    summaries = [collect(entry, root, certificate, certificate_path, starts) for entry in manifest['runs']]
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
