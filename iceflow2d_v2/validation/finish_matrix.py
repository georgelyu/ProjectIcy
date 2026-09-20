"""Wait for an already running six-case matrix, then audit and render it.

This bounded batch helper does not launch or change numerical simulations.
It exits on a failed/interrupted run and never marks A--H physics as validated.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--process-start-times', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    comparison = root/f'{args.prefix}_comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    labels = ('100', '200', 'dt_half_100', 'block2_100', 'block4_100', 'block8_100')
    runs = [root/f'{args.prefix}_{label}' for label in labels]
    record = {'status': 'waiting_for_simulations', 'commands': [], 'runs': [str(p) for p in runs]}
    def save():
        temp = comparison/'finalization.tmp'
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(comparison/'finalization.json')
    save()
    while True:
        ready = []
        for run in runs:
            result = run/'result.json'
            interruption = run/'interrupted.json'
            if interruption.exists():
                record.update(status='simulation_interrupted', failed_run=str(run)); save(); return 2
            if not result.exists():
                ready.append(False); continue
            data = json.loads(result.read_text())
            if data['failure'] or not data['completed_requested_steps']:
                record.update(status='simulation_failed', failed_run=str(run), failure=data['failure']); save(); return 2
            ready.append(True)
        if all(ready):
            break
        time.sleep(5)
    commands = [[sys.executable, '-m', 'iceflow2d_v2.validation.report_full_scene',
                 *map(str, runs), '--output', str(comparison), '--require-numerical-10s',
                 '--source-manifest', str(args.source_manifest.resolve()),
                 '--process-start-times', str(args.process_start_times.resolve())]]
    for label in ('100', '200'):
        commands.append([sys.executable, '-m', 'iceflow2d_v2.examples.render_movie',
                         str(root/f'{args.prefix}_{label}'), '--output', str(comparison/f'full_10s_{label}.mp4')])
    record['status'] = 'postprocessing'; save()
    repo = Path(__file__).resolve().parents[2]
    for command in commands:
        entry = {'command': command, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
        record['commands'].append(entry); save()
        print(json.dumps({'started_postprocessor': command[2]}, ensure_ascii=False), flush=True)
        code = subprocess.run(command, cwd=repo).returncode
        entry.update(returncode=code, finished_at_utc=datetime.now(timezone.utc).isoformat()); save()
        if code:
            record['status'] = 'postprocessing_failed'; save(); return code
    record['status'] = 'numerical_checks_and_artifacts_completed_pending_review'
    record['full_A_H_physics_validated'] = False
    save()
    print(json.dumps({'status': record['status'], 'report': str(comparison/'report_zh.md')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
