"""Configuration must fail before reaching geometry or GPU compilation."""

import unittest

from iceflow2d.config import IceFlowConfig, PhaseChangeProperties


class NumericConfigurationTests(unittest.TestCase):
    def test_float_like_inputs_are_stored_as_numbers(self):
        config = IceFlowConfig(dx="0.000125", rho_water="1000", gravity=("0", "-9.8"))
        self.assertIsInstance(config.dx, float)
        self.assertIsInstance(config.rho_water, float)
        self.assertEqual(config.gravity, (0.0, -9.8))
        self.assertGreater(config.ice_mass_lattice, 0.0)

    def test_bad_physical_values_raise_named_errors(self):
        for value in (True, None, "invalid", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "latent_heat_j_kg"
            ):
                PhaseChangeProperties(latent_heat_j_kg=value)


if __name__ == "__main__":
    unittest.main()
