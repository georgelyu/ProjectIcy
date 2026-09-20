"""Run the specified mechanics and Stefan cases against one frozen source."""
import argparse
from datetime import datetime, timezone
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
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_root.resolve()
    package = source/'iceflow2d_v2'
    manifest = output/f'{args.prefix}_benchmarks.json'
    assert args.workers > 0 and not manifest.exists()
    jobs = []
    for case, end in [('freefall', .01), ('buoyancy', .02), ('float-equilibrium', .1), ('static-interface', .05)]:
        for nx in (100, 200):
            jobs.append((f'{case}_{args.prefix}_{nx}', 'mechanics_benchmarks',
                         ['--case', case, '--nx', str(nx), '--end-time', str(end)]))
    for nx in (100, 200):
        jobs.append((f'planar_{args.prefix}_{nx}', 'planar_stefan', ['--refinement', str(nx//100), '--end-time', '.1']))
        jobs.append((f'stefan_{args.prefix}_{nx}', 'stefan_front',
                     ['--nx', str(nx), '--end-time', '.1', '--pressure-polish-iterations', '24']))
    assert all(not (output/name).exists() for name, _, _ in jobs)
    report = {'status': 'running', 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted([*package.glob('*.py'), *package.glob('*.cpp')])}, 'runs': []}
    active = []
    def save():
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(manifest)
    try:
        while jobs or active:
            while jobs and len(active) < args.workers:
                name, module, extra = jobs.pop(0)
                path = output/name
                command = [sys.executable, '-u', '-m', f'iceflow2d_v2.validation.{module}',
                           *extra, '--native-graph', '--output', str(path)]
                handle = path.with_suffix('.log').open('w')
                child = subprocess.Popen(command, cwd=source, stdout=handle, stderr=subprocess.STDOUT)
                entry = {'name': name, 'path': str(path), 'command': command, 'pid': child.pid,
                         'started_at_utc': datetime.now(timezone.utc).isoformat(), 'returncode': None}
                active.append((child, handle, entry)); report['runs'].append(entry); save()
                print(json.dumps({'started': name, 'pid': child.pid}), flush=True)
            for item in list(active):
                child, handle, entry = item
                if child.poll() is not None:
                    entry['returncode'] = child.returncode
                    entry['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                    handle.close(); active.remove(item); save()
                    print(json.dumps({'finished': entry['name'], 'returncode': child.returncode}), flush=True)
            time.sleep(5)
        report['status'] = 'completed' if all(e['returncode'] == 0 for e in report['runs']) else 'failed'
        save()
    except BaseException:
        for child, handle, _ in active:
            if child.poll() is None: os.kill(child.pid, signal.SIGINT)
            handle.close()
        report['status'] = 'interrupted'; save()
        raise
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
