import csv
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from iceflow2d_v2.config import Config
from iceflow2d_v2.examples import coupled_falling_ice_melting_2d as example
from iceflow2d_v2.examples._fields import capture_fields, output_steps


class ArrayField:
    def __init__(self, values):
        self.values = np.asarray(values)

    def to_numpy(self):
        return self.values.copy()

    def __getitem__(self, key):
        return self.values


class RecordedSimulation:
    """Known fields with a controllable rejected batch, without advancing physics."""

    def __init__(self, config):
        self.cfg, self.steps = config, 0
        self.fail_at = None
        self.last_failure_diagnostics = self.last_failure_snapshot = None
        shape = (config.nx, config.ny)
        capacity = np.ones(shape)
        b = config.wall_cells
        capacity[:b] = capacity[-b:] = capacity[:, :b] = capacity[:, -b:] = 0
        q = np.zeros((*shape, 6))
        q[..., 0] = capacity
        q[:, config.ny//2:, 0] = 0
        q[..., 2] = q[..., 0]*.5
        compensation = np.zeros_like(q)
        compensation[..., 2] = q[..., 0]*.125
        self.state = SimpleNamespace(q=ArrayField(q), comp=ArrayField(compensation))
        velocity = np.zeros((*shape, 2)); velocity[..., 0] = .001
        self.flow = SimpleNamespace(velocity=ArrayField(velocity))
        solid = np.zeros((config.nb, config.nb)); solid[0, 0] = .5
        polygons = np.zeros((config.nb, config.nb, 12, 2))
        polygons[0, 0, :4] = [[-.5, -.5], [0, -.5], [0, .5], [-.5, .5]]
        counts = np.zeros_like(solid, dtype=int); counts[0, 0] = 4
        self.geometry = SimpleNamespace(
            capacity=ArrayField(capacity), solid=ArrayField(solid),
            solid_comp=ArrayField(np.zeros_like(solid)),
            ice_heat=ArrayField(solid*config.ice_capacity_lu*(-5)/config.temperature_scale),
            ice_heat_comp=ArrayField(np.zeros_like(solid)),
            poly=ArrayField(polygons), count=ArrayField(counts),
            rotation=ArrayField(np.eye(2)), origin=ArrayField(config.initial_origin))

    def diagnostics(self):
        return {'step': self.steps, 'time_s': self.steps*self.cfg.dt,
                'condensed_mass_relative_residual': 0., 'condensed_energy_relative_residual': 0.,
                'capacity_residual_max': 0., 'mach_max': 0.}

    def advance_checked(self, count):
        if self.fail_at is not None and self.steps >= self.fail_at:
            self.last_failure_diagnostics = {**self.diagnostics(), 'mach_max': float('nan')}
            self.last_failure_snapshot = SimpleNamespace(save=forbid_npz)
            raise example.SimulationError('injected rejected batch')
        self.steps += count
        return self.diagnostics(), False

    def save(self, path):
        forbid_npz(path)


def forbid_npz(*args, **kwargs):
    pytest.fail('The simulation entry point must never write NPZ files')


def test_physical_frame_targets_do_not_accumulate_rounding_drift():
    dt = Config().dt
    assert list(output_steps(0, 4000, dt, .01)) == [0, 1600, 3200, 4000]
    assert list(output_steps(800, 3200, dt, .01)) == [800, 1600, 3200]
    assert list(output_steps(0, 10, .003, .01)) == [0, 4, 7, 10]
    assert list(output_steps(0, 3, .03, .01)) == [0, 1, 2, 3]


def test_fields_use_compensated_heat_physical_velocity_and_material_ice():
    sim = RecordedSimulation(Config(nx=25, ny=50))
    temperature, speed, velocity, polygons, ice_temperature = capture_fields(sim)
    np.testing.assert_allclose(temperature.compressed(), 33.75)
    np.testing.assert_allclose(speed.compressed(), .04)
    np.testing.assert_allclose(velocity[..., 0], .04)
    assert temperature.mask[0, 0] and temperature.mask[12, 40]
    assert not temperature.mask[12, 12]
    np.testing.assert_allclose(ice_temperature, [-5])
    expected = (np.array([[-.5, -.5], [0, -.5], [0, .5], [-.5, .5]])
                +np.array([.5-sim.cfg.nb/2]*2)+sim.cfg.initial_origin)*sim.cfg.dx*1000
    np.testing.assert_allclose(polygons[0], expected)


@pytest.mark.parametrize('start, stop, fail_at, expected_steps', [
    (0, 840, None, [0, 400, 800, 840]),
    (100, 800, None, [100, 400, 800]),
    (0, 0, None, [0]),
    (0, 840, 600, [0, 400, 640]),
])
def test_entry_point_writes_only_images_and_text(monkeypatch, tmp_path, start, stop, fail_at, expected_steps):
    def simulation_factory(config):
        sim = RecordedSimulation(config)
        sim.fail_at = fail_at
        return sim

    monkeypatch.setattr(example, 'Simulation', simulation_factory)
    monkeypatch.setattr(example.ti, 'init', lambda **kwargs: None)
    monkeypatch.setattr(np, 'savez_compressed', forbid_npz)
    monkeypatch.setattr(example.Snapshot, 'save', forbid_npz)
    args = ['--arch', 'cpu', '--nx', '25', '--steps', str(stop), '--check-steps', '200',
            '--sample-interval', '.003', '--output', str(tmp_path), '--no-progress']
    if start:
        monkeypatch.setattr(example.Snapshot, 'load', lambda path: SimpleNamespace(
            restore=lambda sim: setattr(sim, 'steps', start)))
        args += ['--resume', 'existing_restart.npz']
    assert example.main(args) == (2 if fail_at is not None else 0)
    with (tmp_path/'frames.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    # Sampling/check chunks must never shift the independent 0.01 s image cadence.
    assert [int(row['step']) for row in rows] == expected_steps
    np.testing.assert_allclose([float(row['time_s']) for row in rows], np.array(expected_steps)*.000025)
    for kind in ('temperature', 'velocity'):
        assert len(list((tmp_path/f'{kind}_frames').glob('*.png'))) == len(rows)
        for row in rows:
            with Image.open(tmp_path/row[kind]) as image:
                assert image.format == 'PNG'
                image.verify()
    assert not list(tmp_path.rglob('*.npz'))
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['image_frames'] == len(rows)
    assert result['diagnostics']['step'] == expected_steps[-1]
    assert result['completed_requested_steps'] == (fail_at is None)
    if fail_at is not None:
        assert result['rejected_diagnostics']['mach_max'] is None
