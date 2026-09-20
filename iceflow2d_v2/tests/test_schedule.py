import numpy as np
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation


def test_graph_capture_does_not_advance_and_matches_python():
    config = Config(nx=25, ny=50, warmup_steps=8, projection_solver='multigrid')
    python, graph = Simulation(config), Simulation(config)
    initial = graph.state.q.to_numpy().copy()
    origin = np.array(graph.geometry.origin[None])
    graph.enable_graph()
    assert graph.schedule.dispatches > 100
    np.testing.assert_array_equal(graph.state.q.to_numpy(), initial)
    np.testing.assert_array_equal(graph.geometry.origin[None], origin)
    python.step(8); graph.step(8)
    for a, b in ((python.state.q, graph.state.q), (python.flow.f, graph.flow.f),
                 (python.geometry.solid, graph.geometry.solid), (python.rigid.momentum, graph.rigid.momentum)):
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=3e-8, rtol=2e-5)
    np.testing.assert_allclose(python.geometry.origin[None], graph.geometry.origin[None], atol=2e-9)
    python.validate(); graph.validate()
