"""Run full-duration dt/2 and heat-block comparisons from a frozen package."""
import argparse
import csv
import hashlib
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
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--start-after', type=Path, required=True)
    parser.add_argument('--start-time', type=float, default=2.0)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_root.resolve()
    package = source/'iceflow2d_v2'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted([*package.glob('*.py'), *package.glob('*.cpp')])}
    variants = [('dt_half_100', ['--reference-velocity', '8']),
                *[(f'block{k}_100', ['--thermal-block', str(k)]) for k in (2, 4, 8)]]
    for name, _ in variants:
        if (output/f'{args.prefix}_{name}').exists():
            raise FileExistsError(f'Refusing to overwrite {args.prefix}_{name}')
    manifest = output/f'{args.prefix}_parameter_matrix.json'
    report = {'source_sha256': hashes, 'dependency': str(args.start_after.resolve()),
              'dependency_time_s': args.start_time, 'status': 'waiting', 'runs': []}
    def save():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(manifest)
    save()
    while True:
        failure = args.start_after/'result.json'
        if failure.exists() and json.loads(failure.read_text())['failure']:
            report['status'] = 'dependency_failed'; save()
            return 2
        if (args.start_after/'interrupted.json').exists():
            report['status'] = 'dependency_interrupted'; save()
            return 2
        config = args.start_after/'config.json'
        samples = args.start_after/'diagnostics.csv'
        if config.exists() and samples.exists():
            metadata = json.loads(config.read_text())
            observed = {k: v for k, v in metadata['source_sha256'].items() if '/' not in k}
            if hashes != observed: raise ValueError('Dependency and comparison source hashes differ.')
            with samples.open() as handle:
                rows = [r for r in csv.DictReader(handle) if r.get('time_s') and r.get('mlups')]
            if rows and float(rows[-1]['time_s']) >= args.start_time: break
        time.sleep(5)
    children, handles = [], []
    try:
        for name, extra in variants:
            path = output/f'{args.prefix}_{name}'
            command = [sys.executable, '-u', '-m', 'iceflow2d_v2.examples.coupled_falling_ice_melting_2d',
                       '--nx', '100', '--end-time', '10', '--native-graph',
                       '--projection-solver', 'multigrid', '--pressure-iterations', '1200',
                       '--output', str(path), *extra]
            log = path.with_suffix('.log').open('w'); handles.append(log)
            child = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            entry = {'path': str(path), 'command': command, 'pid': child.pid, 'returncode': None}
            report['runs'].append(entry)
            print(json.dumps({'started': name, 'pid': child.pid}), flush=True)
        report['status'] = 'running'; save()
        while any(child.poll() is None for child in children):
            for child, entry in zip(children, report['runs']):
                code = child.poll()
                if code is not None and entry['returncode'] is None:
                    entry['returncode'] = code; save()
                    print(json.dumps({'finished': entry['path'], 'returncode': code}), flush=True)
            time.sleep(5)
        for child, entry in zip(children, report['runs']): entry['returncode'] = child.returncode
        report['status'] = 'completed' if all(c.returncode == 0 for c in children) else 'failed'
        save()
    except BaseException:
        for child in children:
            if child.poll() is None: os.kill(child.pid, signal.SIGINT)
        report['status'] = 'interrupted'; save()
        raise
    finally:
        for handle in handles: handle.close()
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
