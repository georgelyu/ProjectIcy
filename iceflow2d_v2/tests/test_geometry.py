import math
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.polygon import square, segment_interval


@ti.kernel
def thin_face_intervals(out: ti.types.ndarray(dtype=ti.f64, ndim=2)):
    for k in range(3):
        gap = ti.cast(1e-10, ti.f64)
        if k == 1: gap = -gap
        start, end = ti.Vector([.5+gap, -.5], dt=ti.f64), ti.Vector([.5+gap, .5], dt=ti.f64)
        if k == 2:
            start.x, end.x = .5+ti.cast(1e-14, ti.f64), .5-ti.cast(1e-14, ti.f64)
        interval = segment_interval(square(), 4, start, end)
        out[k, 0], out[k, 1] = interval.x, interval.y


def test_thin_open_faces_are_not_closed_by_a_distance_tolerance():
    out = np.zeros((3, 2)); thin_face_intervals(out)
    assert out[0, 1] < out[0, 0]
    np.testing.assert_allclose(out[1], [0, 1], atol=1e-14)
    np.testing.assert_allclose(out[2], [.5, 1], atol=1e-14)


def test_exact_rotated_area_centroid_and_inertia():
    config = Config(nx=25, ny=50)
    geo = Geometry(config)
    for angle in (0, .13, .72, 1.3, 2.1):
        geo.angle[None] = angle
        geo.rasterize()
        area = geo.occupancy.to_numpy().astype(float).sum()
        np.testing.assert_allclose(area, config.nb**2, atol=2e-5)
        np.testing.assert_allclose(geo.first_moment[None], 0, atol=2e-5)
        np.testing.assert_allclose(geo.inertia[None], config.rho_s_lu*config.nb**4/6, rtol=2e-6)
        assert geo.capacity.to_numpy().min() >= 0
        np.testing.assert_allclose(geo.surface_length.to_numpy().astype(float).sum(), 4*config.nb, atol=1e-5)


def test_partial_cell_area_is_a_polygon_not_uniform_porous_square():
    geo = Geometry(Config(nx=25, ny=50))
    fraction = np.ones((8, 8), np.float32)
    fraction[0, :] = .25
    geo.solid.from_numpy(fraction)
    geo.reconstruct(); geo.rasterize()
    area = geo.integrals.to_numpy()[..., 0]
    np.testing.assert_allclose(area, fraction, atol=2e-7)
    np.testing.assert_allclose(geo.occupancy.to_numpy().astype(float).sum(), fraction.sum(dtype=float), atol=2e-5)
    # The quarter cell centroid shifts inward; scaling an entire unit square
    # would leave this first moment at zero and violate the geometry model.
    assert np.all(geo.integrals.to_numpy()[0, 1:-1, 1] > 0)


def test_melting_polygons_are_nested_when_neighbour_gradients_change():
    geo = Geometry(Config(nx=25, ny=50))
    fraction = np.ones((8, 8), np.float32)
    fraction[0, :] = .6
    geo.solid.from_numpy(fraction); geo.reconstruct()
    normals, offsets = geo.normal.to_numpy(), geo.offset.to_numpy()
    fraction[0, :] = .4
    fraction[1, :4] = .2
    geo.solid.from_numpy(fraction); geo.reconstruct()
    polys, counts = geo.poly.to_numpy(), geo.count.to_numpy()
    for b in range(8):
        assert np.max(polys[0, b, :counts[0, b]]@normals[0, b]-offsets[0, b]) <= 2e-14
    np.testing.assert_allclose(geo.integrals.to_numpy()[..., 0], fraction, atol=2e-14)


def test_face_driven_retreat_preserves_area_and_fluid_connectivity():
    from collections import deque
    cfg = Config(nx=25, ny=50)
    geo = Geometry(cfg)
    dm = ti.field(ti.f32, (cfg.nb, cfg.nb, 12))
    rng = np.random.default_rng(51)
    for _ in range(12):
        old_capacity = geo.capacity.to_numpy()
        amounts = (geo.surface_length.to_numpy()*rng.uniform(.001, .01, dm.shape)).astype(np.float32)
        fraction = geo.solid.to_numpy().astype(float)-geo.solid_comp.to_numpy()
        target = fraction-amounts.sum(axis=-1, dtype=float)/cfg.rho_s_lu
        assert target.min() > 0
        high = target.astype(np.float32)
        geo.solid.from_numpy(high); geo.solid_comp.from_numpy((high.astype(float)-target).astype(np.float32))
        dm.from_numpy(amounts); geo.erode(dm); geo.rasterize()
        np.testing.assert_allclose(geo.integrals.to_numpy()[..., 0], target, atol=2e-14, rtol=0)
        capacity, faces = geo.capacity.to_numpy(), geo.face_open.to_numpy()
        reached = old_capacity > 0
        queue = deque(map(tuple, np.argwhere(reached)))
        while queue:
            i, j = queue.popleft()
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i+di, j+dj
                if 0 <= ni < cfg.nx and 0 <= nj < cfg.ny and not reached[ni, nj] and capacity[ni, nj] > 0:
                    opening = faces[min(i, ni), min(j, nj), 0 if di else 1]
                    if opening > 0:
                        reached[ni, nj] = True; queue.append((ni, nj))
        assert np.all(reached[capacity > 1e-10]), 'retreat created a sealed empty fluid pocket'


def test_partly_exposed_face_melts_its_corner_without_opening_a_full_height_channel():
    cfg = Config(nx=25, ny=50, angle_deg=0)
    geo = Geometry(cfg)
    fraction = geo.solid.to_numpy(); fraction[4, 6] = .98
    normals = geo.normal.to_numpy(); normals[4, 6] = [0, 1]
    geo.normal.from_numpy(normals); geo.solid.from_numpy(fraction)
    geo.reconstruct(); geo.rasterize()
    lengths = geo.surface_length.to_numpy()
    np.testing.assert_allclose(lengths[3, 6].sum(), .02, atol=3e-8)
    accepted = np.zeros(lengths.shape, np.float32)
    accepted[3, 6] = lengths[3, 6]/lengths[3, 6].sum()*.0001*cfg.rho_s_lu
    fraction[3, 6] = 1-accepted[3, 6].sum(dtype=float)/cfg.rho_s_lu
    geo.solid.from_numpy(fraction)
    dm = ti.field(ti.f32, lengths.shape); dm.from_numpy(accepted)
    geo.erode(dm)
    p = geo.poly.to_numpy()[3, 6, :geo.count[3, 6]]
    assert np.any(np.all(p == [.5, -.5], axis=1)), 'buried lower-right corner was removed'
    assert not np.any(np.all(p == [.5, .5], axis=1)), 'heated upper-right corner was retained'


def test_no_mass_loss_after_many_rigid_rasterizations():
    geo = Geometry(Config(nx=25, ny=50))
    initial = geo.totals()
    origin = np.array(geo.origin[None], float)
    maximum = 0.0
    for k in range(31):
        geo.angle[None] = k*math.pi/62
        geo.origin[None] = origin+[.4*math.sin(k), .3*math.cos(k)]
        geo.rasterize()
        maximum = max(maximum, abs(geo.occupancy.to_numpy().astype(float).sum()-64))
    assert geo.totals() == initial
    assert maximum < 2e-5


def test_exhausted_ice_leaves_no_frozen_original_rectangle():
    config = Config(nx=25, ny=50)
    geo = Geometry(config)
    geo.solid.fill(0); geo.reconstruct(); geo.rasterize()
    assert geo.occupancy.to_numpy().sum() == 0
    assert geo.mask.to_numpy().sum() == 0
    interior = (slice(config.wall_cells, -config.wall_cells),)*2
    np.testing.assert_array_equal(geo.capacity.to_numpy()[interior], 1)
    assert geo.surface_length.to_numpy().sum() == 0
    assert geo.mass[None] == 0


def test_fully_covered_world_cell_is_not_a_zero_volume_fluid_node():
    # Four material triangles, each of area 1/4, form one complete world cell
    # after a 45-degree rotation. The approved smoothed SDF is positive at
    # their common vertex (1/2 - 1/4); it does not measure exact fluid volume.
    # Treating that fully covered node as air creates an artificial density
    # jump and capillary acceleration when a moving remnant covers it.
    cfg = Config(nx=25, ny=50, angle_deg=45)
    geo = Geometry(cfg)
    fractions = np.zeros((cfg.nb, cfg.nb), np.float32)
    a = cfg.nb//2
    fractions[a-1:a+1, a-1:a+1] = .25
    geo.solid.from_numpy(fractions)
    geo.solid_comp.fill(0)
    geo.reconstruct()
    geo.origin[None] = [12.5, 25.5]
    geo.rasterize()
    np.testing.assert_allclose(geo.occupancy.to_numpy().sum(), 1, atol=2e-14)
    assert geo.sdf[12, 25] > 0  # Preserve the whitelisted SDF, not its sign as volume.
    assert geo.capacity[12, 25] == 0
    assert geo.mask[12, 25] == 1

    # No positive-volume cutoff: even a newly opened thin sliver is retained.
    geo.origin[None] = [12.5+1e-8, 25.5]
    geo.rasterize()
    assert 0 < geo.capacity[12, 25] < 2e-8
    assert geo.sdf[12, 25] > 0 and geo.mask[12, 25] == 0
    np.testing.assert_allclose(geo.occupancy.to_numpy().sum(), 1, atol=2e-14)


def test_clipping_through_vertices_has_no_duplicate_corners():
    import taichi as ti
    from iceflow2d_v2.polygon import square, clip, moments

    @ti.kernel
    def check(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        # Match production's per-cell invocation: the polygon's sequential
        # vertex scan must be nested inside the kernel's parallel work loop.
        for _ in range(1):
            triangle, count = clip(square(), 4, ti.Vector([1.0, 1.0]), 0.0)
            out[0], out[1] = count, moments(triangle, count)[0]
            line, n = clip(triangle, count, ti.Vector([-1.0, -1.0]), 0.0)
            out[2], out[3] = n, moments(line, n)[0]

    result = np.zeros(4, np.float32); check(result)
    np.testing.assert_array_equal(result, [3, .5, 2, 0])
