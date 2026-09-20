"""Collect existing validation evidence without rerunning or accepting failed cases."""
import hashlib
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'baseline_100x200_0p02s': 'baseline_components',
    'refined_200x400_short': 'refined_current',
    'coarse_25x50_short': 'coarse_components',
}


def main():
    records = {}
    for label, directory in CASES.items():
        folder = ROOT/'outputs'/directory
        if not (folder/'result.json').exists():
            records[label] = {'status': 'not_run'}
            continue
        result = json.loads((folder/'result.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        records[label] = {'config': config, **result}
        # Preserve the whole diagnostic history, including the final rejected row.
        (ROOT/'validation'/f'{label}.csv').write_text((folder/'diagnostics.csv').read_text())
    xml = ROOT/'validation'/'pytest_cuda.xml'
    tests = {}
    if xml.exists():
        suites = ET.parse(xml).getroot().findall('testsuite')
        tests = {key: sum(int(s.get(key, 0)) for s in suites)
                 for key in ('tests', 'failures', 'errors', 'skipped')}
        tests['seconds'] = sum(float(s.get('time', 0)) for s in suites)
    sources = [*ROOT.glob('*.py'), *ROOT.joinpath('examples').glob('*.py')]
    payload = {
        'full_simulation_validated': False,
        'test_arch': 'cuda', 'gpu': 'NVIDIA GeForce RTX 5090',
        'python': platform.python_version(), 'taichi': '1.7.4',
        'gpu_state': 'f32', 'fast_math': False, 'host_diagnostics': 'float64',
        'units_per_unit_depth': {'mass': 'kg/m', 'energy': 'J/m', 'linear_momentum': 'kg/s',
                                 'angular_momentum': 'kg m/s', 'capacity_residual': 'dx^2'},
        'air_closure': 'fixed reference density; condensed mass strict, numerical air error separately reported',
        'tests': tests, 'cases': records,
        'historical_rejected_case': 'historical_coarse_failure.json',
        'delivered_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(sources)},
        'timing_note': 'Wall times are development measurements and include first-use compilation in step(). The final three cases ran sequentially. Not a throughput certification.',
        'reference_scope': 'research/reference_scope.md; no legacy simulator/thermal/rigid code imported or executed',
        'unvalidated': ['spatially coupled Stefan convergence', 'long-time fixed-air pressure response',
                        'complete linear/angular momentum balance', 'added-mass and buoyancy benchmarks',
                        'adaptive rollback/substepping', '10 second 100x200 and 200x400 convergence',
                        'thermal blocks 2/4/8', 'complete mechanical energy balance'],
    }
    (ROOT/'validation'/'status.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'tests': tests, 'cases': {k: {'steps': v.get('diagnostics', {}).get('step'),
          'failure': v.get('failure')} for k, v in records.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
