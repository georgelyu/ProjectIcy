"""Record the narrowly scoped v2 contact/observer provenance transition."""
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = Path(__file__).resolve().parents[1]
    output = package/'outputs'
    reference = output/'full_receiver_frozen_source'/'iceflow2d_v2'
    optimized = output/'contact_parallel_frozen_source'/'iceflow2d_v2'
    def sources(root):
        return {p.name: digest(p) for p in sorted([*root.glob('*.py'), *root.glob('*.cpp')])}
    old, new = sources(reference), sources(optimized)
    changed = [name for name in old if old[name] != new[name]]
    assert old.keys() == new.keys()
    assert set(changed) == {'rigid.py', 'health.py', 'simulation.py'}
    allowed = {'rigid.py': {'contact', 'contact_side'},
               'health.py': {'observe'}, 'simulation.py': {'diagnostics'}}
    checks = {}
    for name, methods in allowed.items():
        trees = [ast.parse((root/name).read_text()) for root in (reference, optimized)]
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    node.body = [item for item in node.body
                                 if not isinstance(item, ast.FunctionDef) or item.name not in methods]
        same = ast.dump(trees[0]) == ast.dump(trees[1])
        assert same, name
        checks[name] = {'changed_methods': sorted(methods), 'all_other_AST_identical': same}
    contact = json.loads((output/'contact_equivalence.json').read_text())
    assert contact['cases'] == 144 and contact['all_bitwise_equal']
    assert contact['reference_sha256'] == old['rigid.py']
    assert contact['candidate_sha256'] == new['rigid.py']
    probe = json.loads((output/'native_contact_probe/result.json').read_text())
    assert probe['recording_preserved_every_state_array']
    assert probe['input_sources'] == old and probe['output_sources'] == new
    assert probe['diagnostics']['finite'] and probe['diagnostics']['failure_code'] == 0
    assert probe['diagnostics']['batch_failure_flags'] == 0
    assert '61 passed' in (output/'contact_parallel_frozen_tests.log').read_text()
    evidence = [output/'contact_equivalence.json', output/'contact_parallel_frozen_tests.log',
                output/'native_contact_probe/result.json', output/'geometry_guard_observer_change.json',
                output/'profile_fine_slowdown/repeatability_comparison.json',
                package/'research/contact_extrema_zh.md', Path(__file__).resolve()]
    certificate = {'kind': 'equivalent_v2_contact_and_geometry_observer', 'validated': True,
                   'changed_core_files': changed, 'reference_sources': old, 'optimized_sources': new,
                   'AST_checks': checks,
                   'evidence': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in evidence],
                   'scope': 'Same contact law and f32 tie rule; parallel extremum search only. '
                            'Geometry CFL observes surviving vertices instead of the stricter initial-box bound. '
                            'No physical parameters, state-evolution methods outside contact, or field layouts change.',
                   'limitations': ['The complete coupled GPU trajectory is not claimed bitwise reproducible. '
                                   'Independent old-source replays also differ; the saved comparison records those differences.',
                                   'This certificate does not establish A-H physical convergence or terminal-melting stability.',
                                   'A migrated run has two source versions and must retain both segment histories.'],
                   'checkpoint_rule': 'Only source/provenance metadata may change; all arrays, config, step, '
                                      'and initial budgets must remain unchanged.'}
    target = output/'contact_optimization_certificate.json'
    target.write_text(json.dumps(certificate, indent=2)+'\n')
    print(json.dumps({'certificate': str(target), 'changed': changed, 'validated': True}))


if __name__ == '__main__':
    main()
