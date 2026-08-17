from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "iceflow2d" / "examples" / "ice_fall_2d.py"
)
SPEC = importlib.util.spec_from_file_location(
    "iceflow2d_ice_fall_example_test", EXAMPLE_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot load {EXAMPLE_PATH}")
EXAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXAMPLE
SPEC.loader.exec_module(EXAMPLE)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "iceflow2d" / "config.py"
CONFIG_SPEC = importlib.util.spec_from_file_location(
    "iceflow2d_config_example_test", CONFIG_PATH
)
if CONFIG_SPEC is None or CONFIG_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {CONFIG_PATH}")
CONFIG = importlib.util.module_from_spec(CONFIG_SPEC)
sys.modules[CONFIG_SPEC.name] = CONFIG
CONFIG_SPEC.loader.exec_module(CONFIG)


class IceFallExampleGeometryTests(unittest.TestCase):
    def test_default_geometry_is_a_full_width_pool_with_ice_in_air(self):
        args = EXAMPLE.parse_args([])
        geometry = EXAMPLE.derive_ice_fall_geometry(args)

        self.assertEqual(
            (geometry["water_width"], geometry["water_height"]), (597, 120)
        )
        self.assertEqual((geometry["ice_width"], geometry["ice_height"]), (60, 60))
        self.assertAlmostEqual(geometry["drop_height"], 60.0)
        self.assertAlmostEqual(geometry["ice_lowest_y"], 180.0)
        self.assertGreater(geometry["ice_highest_y"], geometry["ice_lowest_y"])
        self.assertLessEqual(geometry["ice_highest_y"], 300 - args.boundary_cells)

        expected_extent = 30.0 * (
            abs(math.cos(math.radians(5.0))) + abs(math.sin(math.radians(5.0)))
        )
        self.assertAlmostEqual(
            geometry["ice_center_y"] - geometry["ice_lowest_y"], expected_extent
        )

    def test_default_fractional_gap_scales_to_a_small_grid(self):
        args = EXAMPLE.parse_args(["--resolution-x", "120", "--resolution-y", "60"])
        geometry = EXAMPLE.derive_ice_fall_geometry(args)

        self.assertEqual((geometry["water_width"], geometry["water_height"]), (117, 24))
        self.assertEqual((geometry["ice_width"], geometry["ice_height"]), (12, 12))
        self.assertAlmostEqual(geometry["drop_height"], 12.0)
        self.assertAlmostEqual(
            geometry["ice_lowest_y"] - geometry["water_height"], 12.0
        )
        self.assertLessEqual(geometry["ice_highest_y"], 57.0)

    def test_default_geometry_builds_the_expected_regular_config(self):
        args = EXAMPLE.parse_args([])
        fake_package = types.ModuleType("iceflow2d")
        fake_package.create_iceflow_config = CONFIG.create_iceflow_config
        with mock.patch.dict(sys.modules, {"iceflow2d": fake_package}):
            config = EXAMPLE.create_ice_fall_config(
                args, output_dir="outputs/iceflow2d/test_ice_fall"
            )

        self.assertEqual((config.water_width, config.water_height), (597, 120))
        self.assertEqual((config.ice_width, config.ice_height), (60, 60))
        self.assertEqual(config.output_dir, "outputs/iceflow2d/test_ice_fall")
        extent_y = (
            abs(math.sin(config.ice_initial_angle)) * 0.5 * config.ice_width
            + abs(math.cos(config.ice_initial_angle)) * 0.5 * config.ice_height
        )
        self.assertAlmostEqual(config.ice_initial_center[1] - extent_y, 180.0)

    def test_explicit_cell_gap_measures_the_lowest_rotated_corner(self):
        args = EXAMPLE.parse_args(
            [
                "--resolution-x",
                "120",
                "--resolution-y",
                "60",
                "--ice-angle-degrees",
                "20",
                "--drop-height-cells",
                "12.5",
            ]
        )
        geometry = EXAMPLE.derive_ice_fall_geometry(args)
        self.assertAlmostEqual(
            geometry["ice_lowest_y"] - geometry["water_height"], 12.5
        )

    def test_rejects_a_gap_inside_the_diffuse_interface(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                EXAMPLE.parse_args(["--drop-height-cells", "9"])

    def test_rejects_a_drop_that_crosses_the_top_wall(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                EXAMPLE.parse_args(["--drop-height-fraction", "0.80"])

    def test_rejects_pool_or_ice_that_discretizes_too_small(self):
        cases = (
            ("--water-level-fraction", "0.001"),
            ("--ice-width-fraction", "0.001"),
            ("--ice-height-fraction", "0.001"),
        )
        for option, value in cases:
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        EXAMPLE.parse_args([option, value])


if __name__ == "__main__":
    unittest.main()
