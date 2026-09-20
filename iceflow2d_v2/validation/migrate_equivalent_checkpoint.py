"""Explicit checkpoint provenance transition for a certified equivalent kernel.

This never relaxes Snapshot.restore or changes a state array. It requires the
exact input and imported target source hashes named by an equivalence
certificate and verifies every referenced evidence file. Ordinary numerical
changes must start a new run; this tool is only for the specified equivalent
contact optimization and read-only geometry observer update.
"""
import argparse
import hashlib
import json
from pathlib import Path

from iceflow2d_v2.checkpoint import Snapshot, source_hashes


def state_digest(arrays):
    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing migration checkpoint.')
    certificate = json.loads(args.certificate.read_text())
    if certificate['kind'] != 'equivalent_v2_contact_and_geometry_observer' or not certificate['validated']:
        raise ValueError('This is not a validated equivalence certificate for the supported change.')
    if set(certificate['changed_core_files']) != {'rigid.py', 'health.py', 'simulation.py'}:
        raise ValueError('The certificate includes unsupported changes.')
    for item in certificate['evidence']:
        evidence = Path(item['path'])
        if hashlib.sha256(evidence.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Equivalence evidence changed: {evidence}')
    saved = Snapshot.load(args.checkpoint)
    if saved.metadata['source_sha256'] != certificate['reference_sources']:
        raise ValueError('The input checkpoint was not produced by the certified reference version.')
    target = source_hashes()
    if target != certificate['optimized_sources']:
        raise ValueError('The imported target implementation is not the certified optimized version.')
    digest = state_digest(saved.arrays)
    record = {'kind': certificate['kind'], 'input_checkpoint': str(args.checkpoint.resolve()),
              'input_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'certificate': str(args.certificate.resolve()),
              'certificate_sha256': hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
              'old_source_sha256': saved.metadata['source_sha256'],
              'new_source_sha256': target, 'unchanged_state_arrays_sha256': digest,
              'steps': saved.metadata['steps'], 'config_unchanged': True,
              'initial_budgets_unchanged': True}
    saved.metadata['source_sha256'] = target
    saved.metadata['equivalent_source_transition'] = record
    saved.save(args.output)
    if state_digest(Snapshot.load(args.output).arrays) != digest:
        raise RuntimeError('State arrays changed while writing the provenance transition.')
    record['output_checkpoint'] = str(args.output.resolve())
    record['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.migration.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
