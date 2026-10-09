"""CPU contracts for molecular viscosities and explicit physical overrides."""

from dataclasses import replace
import unittest

from iceflow2d_v3.config import IceFlowConfig, ThermalBoundary, ThermalBoundarySet
from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import (
    create_config,
    parse_args,
)
from iceflow2d_v3.viscosity import (
    air_dynamic_viscosity_pa_s,
    water_dynamic_viscosity_pa_s,
)


class MolecularViscosityTests(unittest.TestCase):
    def test_water_matches_reference_properties_over_the_melting_range(self):
        # Atmospheric-pressure liquid-water reference values in Pa s.
        for temperature, reference in (
            (0.0, 1.792e-3),
            (20.0, 1.002e-3),
            (90.0, 3.15e-4),
        ):
            with self.subTest(temperature=temperature):
                self.assertAlmostEqual(
                    water_dynamic_viscosity_pa_s(temperature) / reference,
                    1.0,
                    delta=0.005,
                )

    def test_air_matches_room_temperature_reference(self):
        self.assertAlmostEqual(
            air_dynamic_viscosity_pa_s(20.0) / 1.813e-5, 1.0, delta=0.001
        )

    def test_default_models_use_configured_density_and_local_temperature(self):
        config = create_config()
        self.assertAlmostEqual(config.water_viscosity_at(90.0), 3.14173e-7, delta=1e-12)
        self.assertAlmostEqual(config.air_viscosity_at(20.0), 1.45065e-5, delta=1e-10)
        self.assertGreater(
            config.water_viscosity_at(0.0), config.water_viscosity_at(90.0)
        )
        denser = replace(
            config, rho_water=2 * config.rho_water, rho_air=2 * config.rho_air
        )
        self.assertEqual(
            denser.water_viscosity_at(90.0), 0.5 * config.water_viscosity_at(90.0)
        )
        self.assertEqual(
            denser.air_viscosity_at(20.0), 0.5 * config.air_viscosity_at(20.0)
        )

    def test_explicit_cli_values_remain_constant_physical_viscosities(self):
        config = create_config(
            parse_args(
                [
                    "--kinematic-viscosity-water",
                    "1e-6",
                    "--kinematic-viscosity-air",
                    "1.5e-5",
                ]
            )
        )
        for temperature in (0.0, 20.0, 90.0):
            self.assertEqual(config.water_viscosity_at(temperature), 1e-6)
            self.assertEqual(config.air_viscosity_at(temperature), 1.5e-5)
        numeric = IceFlowConfig(viscosity_water="1e-6", viscosity_air="1.5e-5")
        self.assertEqual(numeric.viscosity_water, 1e-6)
        self.assertEqual(numeric.viscosity_air, 1.5e-5)

    def test_invalid_overrides_and_out_of_range_water_models_fail_early(self):
        for name in ("viscosity_water", "viscosity_air"):
            for value in (True, 0, -1, "invalid", float("nan"), float("inf")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    IceFlowConfig(**{name: value})
        config = create_config()
        for thermal in (
            replace(config.thermal, initial_water_temperature_c=111.0),
            replace(
                config.thermal,
                boundaries=ThermalBoundarySet(left=ThermalBoundary.dirichlet(111.0)),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "IAPWS"):
                replace(config, thermal=thermal)
        for temperature in (-273.15, float("nan")):
            with self.assertRaisesRegex(ValueError, "air viscosity"):
                air_dynamic_viscosity_pa_s(temperature)


if __name__ == "__main__":
    unittest.main()
