from collections import deque
import numpy as np
import taichi as ti
from iceflow2d_v2.connectivity import Components


def test_gpu_components_match_cpu_bfs_with_islands_and_narrow_channels():
    nx, ny = 31, 47
    rng = np.random.default_rng(313)
    active = (rng.random((nx, ny)) > .29).astype(np.int32)
    links = rng.random((nx, ny, 4)).astype(np.float32)
    links[links < .22] = 0
    a = ti.field(ti.i32, (nx, ny)); a.from_numpy(active)
    k = ti.Vector.field(4, ti.f32, (nx, ny)); k.from_numpy(links)
    labels = ti.field(ti.i32, (nx, ny))
    components = Components(nx, ny)
    components.seed(a, 0); components.connect(k, True); components.labels(labels)
    actual = labels.to_numpy()
    expected = np.full((nx, ny), -1)
    edges = ((1, 0), (0, 1), (1, 1), (-1, 1))
    for i, j in np.ndindex(nx, ny):
        if not active[i, j] or expected[i, j] >= 0: continue
        label = i*ny+j
        queue = deque([(i, j)]); expected[i, j] = label
        while queue:
            x, y = queue.popleft()
            for edge, (dx, dy) in enumerate(edges):
                for sign in (-1, 1):
                    xx, yy = x+sign*dx, y+sign*dy
                    if not (0 <= xx < nx and 0 <= yy < ny): continue
                    weight = links[x, y, edge] if sign == 1 else links[xx, yy, edge]
                    if weight > 0 and active[xx, yy] and expected[xx, yy] < 0:
                        expected[xx, yy] = label; queue.append((xx, yy))
    smallest = np.flatnonzero(active)[0]
    expected[expected == smallest] = -2
    expected = np.where(expected == -2, 0, np.where(expected >= 0, expected+1, -1))
    np.testing.assert_array_equal(actual, expected)
    top_labels = set(actual[:, -1][active[:, -1] != 0])
    components.mark_top(ny-1); components.connected_to_top(labels)
    np.testing.assert_array_equal(labels.to_numpy(), np.isin(actual, list(top_labels)))
