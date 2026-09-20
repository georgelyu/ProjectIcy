import numpy as np
import pytest
from iceflow2d_v2.checkpoint import Snapshot, fields
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation, SimulationError


def test_complete_restart_repeats_accepted_steps(tmp_path):
    sim = Simulation(Config(nx=25, ny=50, warmup_steps=0, projection_solver='multigrid'))
    sim.step(2)
    before = Snapshot(sim)
    path = tmp_path/'restart.npz'; before.save(path)
    sim.step(3)
    expected = Snapshot(sim)
    Snapshot.load(path).restore(sim)
    assert sim.steps == 2
    for name, field in fields(sim).items(): np.testing.assert_array_equal(field.to_numpy(), before.arrays[name], err_msg=name)
    sim.step(3)
    assert sim.steps == 5
    for name, field in fields(sim).items():
        # Union-find trees and unsorted member packing depend on scheduling;
        # canonical membership and every physical field must still agree.
        if name.endswith('.parent'): continue
        if name == 'merge.members':
            actual = field.to_numpy()
            counts = expected.arrays['merge.member_count']
            offsets = expected.arrays['merge.member_offset']
            groups = expected.arrays['merge.group_offset']
            for root in np.flatnonzero(counts):
                start = offsets[root]+groups[root//256]
                end = start+counts[root]
                np.testing.assert_array_equal(np.sort(actual[start:end]), np.sort(expected.arrays[name][start:end]))
            continue
        np.testing.assert_allclose(field.to_numpy(), expected.arrays[name], atol=2e-7, rtol=2e-5, err_msg=name)
    sim.validate()


def test_rejected_batch_retains_evidence_and_restores_every_field():
    sim = Simulation(Config(nx=25, ny=50, warmup_steps=0, max_mach=1e-9))
    before = Snapshot(sim)
    with pytest.raises(SimulationError): sim.advance_checked(2)
    assert sim.steps == 0
    assert sim.last_failure_snapshot.metadata['steps'] == 2
    assert sim.last_failure_diagnostics['batch_failure_flags'] != 0
    for name, field in fields(sim).items():
        np.testing.assert_array_equal(field.to_numpy(), before.arrays[name], err_msg=name)
