import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from iceflow2d_v2.config import Config


def test_units_and_selected_air_closure():
    c = Config()
    assert c.dx == .00025
    assert c.dt == 6.25e-6
    assert round(10/c.dt) == 1600000
    np.testing.assert_allclose(np.array(c.initial_origin)*c.dx, [.0125, .035833401763357614])
    np.testing.assert_allclose(c.thermal_weight('water'), 5.7333970377448644e-5)
    np.testing.assert_allclose(c.thermal_weight('ice'), .00045697668380329236)
    np.testing.assert_allclose(c.thermal_weight('air'), .008278606965174129)
    assert c.coupling_steps == 1 and not c.ice_air_heat_exchange


@pytest.mark.parametrize('change', [{'coupling_steps': 3}, {'ice_air_heat_exchange': True}, {'thermal_tau': .8},
                                    {'ice_temperature': 1}, {'rho_a': 0}, {'pressure_iterations': 0}])
def test_unsupported_or_invalid_physics_fails_explicitly(change):
    with pytest.raises(ValueError): replace(Config(), **change)


def test_production_never_imports_legacy_or_derivation_solver():
    root = Path(__file__).resolve().parents[1]
    for path in [*root.glob('*.py'), *root.joinpath('examples').glob('*.py')]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [item.name for item in node.names]
            if isinstance(node, ast.ImportFrom): names = [node.module or '']
            assert not any(n == 'iceflow2d' or n.startswith('iceflow2d.') or 'derivation' in n for n in names), path
