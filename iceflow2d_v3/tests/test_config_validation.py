"""Configuration must fail before reaching geometry or GPU compilation."""

import unittest

from iceflow2d_v3.config import IceFlowConfig, PhaseChangeProperties


class NumericConfigurationTests(unittest.TestCase):
    def test_stability_controls_are_independent_and_serialized(self):
        defaults = IceFlowConfig()
        self.assertEqual(defaults.phase_normal_epsilon, 1e-3)
        self.assertEqual(defaults.cm_acoustic_relaxation_rate, 0.25)
        for epsilon, rate in ((0.0, 0.25), (1e-3, 1.0), (1e-3, 0.25)):
            config = IceFlowConfig(
                momentum_collision_model="cm_mrt",
                phase_normal_epsilon=str(epsilon),
                cm_acoustic_relaxation_rate=str(rate),
            )
            self.assertEqual(config.to_dict()["phase_normal_epsilon"], epsilon)
            self.assertEqual(config.to_dict()["cm_acoustic_relaxation_rate"], rate)
        # Phase compression is independent of the momentum collision frame.
        self.assertEqual(
            IceFlowConfig(phase_normal_epsilon=1e-3).phase_normal_epsilon, 1e-3
        )

    def test_invalid_stability_controls_fail_before_gpu_setup(self):
        for name, invalid in (
            ("phase_normal_epsilon", (-1e-3, True, None, float("nan"), float("inf"))),
            (
                "cm_acoustic_relaxation_rate",
                (0.0, -0.25, 2.0, True, "invalid", float("nan"), float("inf")),
            ),
        ):
            for value in invalid:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    IceFlowConfig(momentum_collision_model="cm_mrt", **{name: value})
        with self.assertRaisesRegex(
            ValueError, "requires momentum_collision_model='cm_mrt'"
        ):
            IceFlowConfig(
                momentum_collision_model="rm_mrt", cm_acoustic_relaxation_rate=0.25
            )

    def test_momentum_collision_model_is_validated_and_serialized(self):
        self.assertEqual(IceFlowConfig().momentum_collision_model, "cm_mrt")
        for model in ("rm_mrt", "cm_mrt"):
            with self.subTest(model=model):
                config = IceFlowConfig(momentum_collision_model=model)
                self.assertEqual(config.to_dict()["momentum_collision_model"], model)
                self.assertEqual(
                    config.to_dict()["cm_acoustic_relaxation_rate"],
                    0.25 if model == "cm_mrt" else 1.0,
                )
        for value in (None, "bgk", "CM-MRT", 1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "momentum_collision_model"),
            ):
                IceFlowConfig(momentum_collision_model=value)

    def test_float_like_inputs_are_stored_as_numbers(self):
        config = IceFlowConfig(dx="0.000125", rho_water="1000", gravity=("0", "-9.8"))
        self.assertIsInstance(config.dx, float)
        self.assertIsInstance(config.rho_water, float)
        self.assertEqual(config.gravity, (0.0, -9.8))
        self.assertGreater(config.ice_mass_lattice, 0.0)

    def test_bad_physical_values_raise_named_errors(self):
        for value in (True, None, "invalid", float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "latent_heat_j_kg"),
            ):
                PhaseChangeProperties(latent_heat_j_kg=value)


if __name__ == "__main__":
    unittest.main()
