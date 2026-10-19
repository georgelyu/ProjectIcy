"""Air/water visibility and display-only arrow caps, without loading CUDA."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from iceflow2d_v3.config import LatticeScales
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example
from iceflow2d_v3 import reporting


class AirVelocityVisualizationTests(unittest.TestCase):
    def test_air_is_visible_but_walls_and_ice_are_masked(self):
        config = example.create_config()
        scales = LatticeScales.from_iceflow_config(config)
        phase = np.zeros((config.ny, config.nx))
        phase[10:20, 10:20] = 1.0
        solid = np.zeros_like(phase)
        solid[30, 100] = 1
        velocity = np.zeros((*phase.shape, 2))
        velocity[30, 200] = (0.01, 0.02)
        snapshot = SimpleNamespace(
            physical_velocity_lattice=velocity, water_phase=phase, solid=solid
        )
        physical, visible = reporting._physical_flow_velocity(
            config, scales, snapshot, frame_index=0, include_air=True
        )
        _, water_only = reporting._physical_water_velocity(
            config, scales, snapshot, frame_index=0
        )
        self.assertTrue(visible[30, 200])
        self.assertFalse(water_only[30, 200])
        self.assertTrue(visible[15, 15])
        self.assertFalse(visible[30, 100])
        self.assertFalse(visible[0, 200])
        np.testing.assert_allclose(physical[30, 200], (0.4, 0.8))

    def test_arrow_cap_preserves_direction_and_original_velocity(self):
        velocity = np.array([[[4.0, 3.0], [0.04, 0.03]]])
        original = velocity.copy()
        x, y = reporting._scaled_quiver_components(
            velocity,
            np.array([0, 1]),
            np.array([0]),
            display_max_m_s=0.8,
            reference_arrow_length_m=0.001,
            clip_speed=True,
        )
        self.assertAlmostEqual(float(np.hypot(x[0, 0], y[0, 0])), 0.001)
        self.assertAlmostEqual(float(y[0, 0] / x[0, 0]), 0.75)
        self.assertAlmostEqual(float(np.hypot(x[0, 1], y[0, 1])), 0.001 * 0.05 / 0.8)
        np.testing.assert_array_equal(velocity, original)

    def test_stream_renders_moving_air_even_when_no_water_is_present(self):
        config = example.create_config()
        scales = LatticeScales.from_iceflow_config(config)
        phase = np.zeros((config.ny, config.nx))
        velocity = np.zeros((*phase.shape, 2))
        velocity[..., 0] = 0.03  # 1.2 m/s: above the requested color/arrow cap.
        snapshot = SimpleNamespace(
            physical_velocity_lattice=velocity,
            water_phase=phase,
            solid=np.zeros_like(phase),
            physical_time_s=0.01,
            melted_fraction=0.0,
        )
        with TemporaryDirectory() as directory:
            stream = reporting.VelocitySequenceStream(
                Path(directory), config, scales, display_max_m_s=0.8, include_air=True
            )
            frame = stream.append(snapshot)
            summary = stream.close()
            self.assertTrue(frame.is_file())
            self.assertTrue((Path(directory) / summary.animation).is_file())
            self.assertTrue(summary.include_air)
            self.assertEqual(summary.frame_count, 1)
            self.assertAlmostEqual(summary.observed_max_m_s, 1.2)
            self.assertEqual(summary.color_max_m_s, 0.8)
            self.assertEqual(summary.air_color_max_m_s, 4.0)
            self.assertEqual(
                (summary.water_colormap, summary.air_colormap), ("viridis", "Oranges")
            )

    def _rotating_phases(self):
        config = example.create_config()
        scales = LatticeScales.from_iceflow_config(config)
        x, y = np.meshgrid(
            (np.arange(config.nx) + 0.5) * config.dx,
            (np.arange(config.ny) + 0.5) * config.dx,
        )
        phase = (x < 0.075).astype(float)
        curl = np.where(phase >= 0.5, 100.0, 2000.0)
        velocity = (
            np.stack((-0.5 * curl * y, 0.5 * curl * x), axis=-1)
            / scales.velocity_scale_m_s
        )
        snapshot = SimpleNamespace(
            physical_velocity_lattice=velocity,
            water_phase=phase,
            solid=np.zeros_like(phase),
            physical_time_s=0.01,
            melted_fraction=0.0,
        )
        return config, scales, snapshot

    def test_phase_curls_match_rotation_without_cross_interface_spikes(self):
        config, scales, snapshot = self._rotating_phases()
        data = reporting._phase_flow_plot_data(config, scales, snapshot, frame_index=0)
        np.testing.assert_allclose(
            data["water_vorticity"].compressed(), 100.0, atol=1e-9
        )
        np.testing.assert_allclose(
            data["air_vorticity"].compressed(), 2000.0, atol=1e-9
        )
        self.assertTrue(np.ma.is_masked(data["water_vorticity"][30, 224]))
        self.assertTrue(np.ma.is_masked(data["air_vorticity"][30, 225]))

    def test_vorticity_stream_renders_both_phases_with_independent_palettes(self):
        config, scales, snapshot = self._rotating_phases()
        with TemporaryDirectory() as directory:
            stream = reporting.VorticitySequenceStream(
                Path(directory),
                config,
                scales,
                display_abs_max_s_1=250.0,
                velocity_display_max_m_s=0.8,
                include_air=True,
                air_display_abs_max_s_1=5000.0,
                air_velocity_display_max_m_s=4.0,
            )
            frame = stream.append(snapshot)
            summary = stream.close()
            self.assertTrue(frame.is_file())
            self.assertTrue((Path(directory) / summary.animation).is_file())
            self.assertTrue(summary.include_air)
            self.assertAlmostEqual(summary.observed_abs_max_s_1, 2000.0)
            self.assertEqual(
                (summary.water_colormap, summary.air_colormap), ("RdBu_r", "PuOr")
            )
            self.assertEqual(
                (summary.color_abs_max_s_1, summary.air_color_abs_max_s_1),
                (250.0, 5000.0),
            )


if __name__ == "__main__":
    unittest.main()
