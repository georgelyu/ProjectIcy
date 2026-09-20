"""Resume the six recorded scenes with certified, unchanged state arrays.

Each continuation has its own directory; original records remain untouched.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--certificate', type=Path, required=True)
    args = parser.parse_args()
    output, source = args.output_root.resolve(), args.source_dir.resolve()
    certificate = args.certificate.resolve()
    migrator = Path(__file__).with_name('migrate_equivalent_checkpoint.py')
    variants = [('100', []), ('200', []), ('dt_half_100', ['--reference-velocity', '8']),
                *[(f'block{k}_100', ['--thermal-block', str(k)]) for k in (2, 4, 8)]]
    manifest = output/'contact_parallel_continuations.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    report = {'status': 'preparing', 'certificate': str(certificate), 'runs': []}
    env = {**os.environ, 'PYTHONPATH': str(source)}
    def save():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(manifest)
    children, handles = [], []
    try:
        for name, extra in variants:
            previous = output/f'full_receiver_{name}'
            target = output/f'contact_parallel_{name}'
            target.mkdir(exist_ok=False)
            checkpoint = target/'input_restart.npz'
            with (target/'migration.log').open('w') as handle:
                subprocess.run([sys.executable, str(migrator), str(previous/'restart.npz'),
                                '--certificate', str(certificate), '--output', str(checkpoint)],
                               cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            migration = json.loads(checkpoint.with_suffix('.migration.json').read_text())
            with (previous/'diagnostics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            selected = [r for r in rows if int(r['step']) <= migration['steps']]
            assert selected and int(selected[0]['step']) == 0
            assert int(selected[-1]['step']) == migration['steps']
            interruption = {'reason': 'Certified equivalent contact search and geometry observer transition',
                            'continued_in': str(target), 'certificate': str(certificate),
                            'restart_step': migration['steps'], 'last_recorded_step': int(rows[-1]['step']),
                            'discarded_accepted_tail_rows': len(rows)-len(selected),
                            'note': 'Raw CSV and frames are retained. Combined trajectory must omit rows after restart_step.'}
            (previous/'interrupted.json').write_text(json.dumps(interruption, indent=2)+'\n')
            nx = '200' if name == '200' else '100'
            command = [sys.executable, '-u', '-m', 'iceflow2d_v2.examples.coupled_falling_ice_melting_2d',
                       '--nx', nx, '--end-time', '10', '--native-graph', '--projection-solver', 'multigrid',
                       '--pressure-iterations', '1200', '--output', str(target), '--resume', str(checkpoint), *extra]
            log = target.with_suffix('.log').open('w'); handles.append(log)
            started = datetime.now(timezone.utc).isoformat()
            child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            entry = {'name': name, 'path': str(target), 'previous_path': str(previous),
                     'resume_step': migration['steps'], 'command': command, 'pid': child.pid,
                     'started_at_utc': started, 'returncode': None}
            report['runs'].append(entry); save()
            print(json.dumps({'started': name, 'pid': child.pid, 'step': migration['steps']}), flush=True)
        report['status'] = 'running'; save()
        while any(child.poll() is None for child in children):
            for child, entry in zip(children, report['runs']):
                code = child.poll()
                if code is not None and entry['returncode'] is None:
                    entry['returncode'] = code
                    entry['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                    save()
                    print(json.dumps({'finished': entry['name'], 'returncode': code}), flush=True)
            time.sleep(5)
        for child, entry in zip(children, report['runs']):
            entry['returncode'] = child.returncode
            entry.setdefault('finished_at_utc', datetime.now(timezone.utc).isoformat())
        report['status'] = 'completed' if all(c.returncode == 0 for c in children) else 'failed'
        save()
    except BaseException:
        for child in children:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGINT)
        report['status'] = 'interrupted'; save()
        raise
    finally:
        for handle in handles:
            handle.close()
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
