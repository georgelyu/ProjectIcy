"""Derive full-grid surfaces and scientific attributes from dry-ice HDF5.

This is display-only: no solver fields are altered. The saved material grid,
not a single Eulerian section, determines the dry-ice geometry.
"""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from .dryice_export import sha256, write_json


AXES = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)


def surface(values, dx, sigma=0.35):
    a = np.pad(np.asarray(values, np.float32), 2)
    a = ndimage.gaussian_filter(a, sigma=sigma, mode="constant")
    if a.max() <= 0.5:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    vertices, faces, _, _ = marching_cubes(a, 0.5, spacing=(dx,) * 3, allow_degenerate=False)
    vertices += (0.5 - 2) * dx
    volume6 = np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum()
    if volume6 < 0:
        faces = faces[:, [0, 2, 1]]
    return vertices.astype(np.float32), faces.astype(np.int32)


def read_xyz(f, name):
    a = f[f"fields/{name}"][:]
    return a.transpose((2, 1, 0, 3) if a.ndim == 4 else (2, 1, 0))


def extract(field_path, cache_path):
    start = time.perf_counter()
    cache = {}
    with h5py.File(field_path, "r") as f:
        dx = float(f.attrs["dx_m"])
        lengths = f.attrs["physical_size_m"]
        shift = np.array([-lengths[0] / 2, lengths[2] / 2, 0])
        phi, ids = read_xyz(f, "water_phase"), read_xyz(f, "solid_body_id")
        fluid = ids == -1
        optical = phi.copy()
        optical[ids == -2] = 0
        # Fill only the solid interior from the surrounding resolved fluid.
        # A local EDT avoids a full-domain 3-vector nearest-neighbour allocation.
        where = np.where(ids >= 0)
        if where[0].size:
            bounds = tuple(slice(max(0, int(a.min()) - 8), min(phi.shape[i], int(a.max()) + 9)) for i, a in enumerate(where))
            local_ids = ids[bounds]
            nearest = ndimage.distance_transform_edt(local_ids != -1, return_distances=False, return_indices=True)
            local = optical[bounds]
            solid = local_ids >= 0
            local[solid] = local[tuple(a[solid] for a in nearest)]
        vertices, faces = surface(optical, dx)
        coordinates = (vertices / dx - 0.5).T
        weight = ndimage.map_coordinates(fluid.astype(np.float32), coordinates, order=1, mode="nearest")

        def sample(a):
            return (ndimage.map_coordinates(np.where(fluid, a, 0), coordinates, order=1, mode="nearest") / np.maximum(weight, 1e-8)).astype(np.float32)

        velocity = read_xyz(f, "velocity_m_s")
        water_velocity = np.stack([sample(velocity[..., i]) for i in range(3)], axis=-1)
        cache.update(water_vertices=(vertices @ AXES.T + shift).astype(np.float32), water_faces=faces,
                     water_velocity_m_s=water_velocity @ AXES.T,
                     water_speed_m_s=np.linalg.norm(water_velocity, axis=-1).astype(np.float32))
        for key, field in (("water_temperature_c", "water_temperature_c"), ("water_pressure_pa", "pressure_pa")):
            a = read_xyz(f, field)
            cache[key] = sample(a)
        for b, size in enumerate(f["body/material_size"][:]):
            fraction = f["body/material_fraction"][b, :size[0], :size[1], :size[2]]
            v, triangles = surface(fraction, dx, sigma=0.35)
            local = v - size * dx / 2
            world = local @ f["body/rotation"][b].T + f["body/origin_m"][b]
            cache[f"ice_{b}_vertices"] = (world @ AXES.T + shift).astype(np.float32)
            cache[f"ice_{b}_faces"] = triangles
            cache[f"ice_{b}_mass_kg"] = f["body/mass_kg"][b]
        # A flat scientific cut through the centre, including CO2. Display only
        # the tank neighbourhood; full-domain fields remain in the HDF5 archive.
        n = phi.shape[0]
        ix = np.arange(max(0, n // 4 - n // 18), min(n, 3 * n // 4 + n // 18))
        iy = np.arange(0, min(n, n // 2))
        xx, yy = np.meshgrid(ix, iy, indexing="ij")
        z = n // 2
        slice_v = np.stack(((xx + .5) * dx, (yy + .5) * dx, np.full(xx.shape, (z + .5) * dx)), axis=-1)
        cache["slice_vertices"] = (slice_v.reshape(-1, 3) @ AXES.T + shift).astype(np.float32)
        face_list = []
        indices = np.arange(xx.size).reshape(xx.shape)
        a, b, c, d = indices[:-1, :-1].ravel(), indices[1:, :-1].ravel(), indices[1:, 1:].ravel(), indices[:-1, 1:].ravel()
        face_list = np.concatenate((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)))
        cache["slice_faces"] = face_list.astype(np.int32)
        cache["slice_shape"] = np.array(xx.shape)
        for name in ("water_phase", "solid_body_id", "temperature_c", "pressure_pa"):
            a = read_xyz(f, name) if name not in ("water_phase", "solid_body_id") else (phi if name == "water_phase" else ids)
            cache["slice_" + name] = np.asarray(a[ix[:, None], iy[None, :], z]).ravel()
        cache["slice_speed_m_s"] = np.linalg.norm(velocity[ix[:, None], iy[None, :], z], axis=-1).ravel().astype(np.float32)
        cache.update(time_s=float(f.attrs["time_s"]), step=int(f.attrs["step"]), dx_m=dx,
                     physical_size_m=lengths, body_count=len(f["body/mass_kg"]))
    if not all(np.isfinite(a).all() for a in cache.values()):
        raise FloatingPointError(f"nonfinite mesh data: {field_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **cache)
    tmp.replace(cache_path)
    info = dict(source_sha256=sha256(field_path), mesh_sha256=sha256(cache_path),
                time_s=float(cache["time_s"]), step=int(cache["step"]),
                water_vertices=len(vertices), water_triangles=len(faces),
                ice_triangles=sum(len(v) for k, v in cache.items() if k.startswith("ice_") and k.endswith("_faces")),
                wall_time_s=time.perf_counter() - start,
                geometry="full-resolution phi=0.5 and material-fraction=0.5; Gaussian sigma=0.35 cells for display only",
                coordinates="Blender=(solver_x-Lx/2,Lz/2-solver_z,solver_y); metres")
    write_json(cache_path.with_suffix(".json"), info)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        manifest = args.simulation / "simulation.json"
        if not manifest.exists():
            if not args.watch:
                raise FileNotFoundError(manifest)
            time.sleep(5)
            continue
        sim = json.loads(manifest.read_text())
        for f in sim["frames"]:
            path = args.output / f"frame_{f['index']:06d}.npz"
            info = path.with_suffix(".json")
            if info.exists() and path.exists() and json.loads(info.read_text())["source_sha256"] == f["sha256"]:
                continue
            result = extract(args.simulation / f["path"], path)
            if result["source_sha256"] != f["sha256"]:
                raise RuntimeError("source changed during meshing")
            print(json.dumps(dict(index=f["index"], **result)), flush=True)
        frames = sorted(args.output.glob("frame_*.json"))
        write_json(args.output / "meshes.json", dict(status=sim["status"], frame_count=len(frames),
            source_status=sim["status"], last_frame=frames[-1].name if frames else None))
        if sim["status"] in ("completed", "failed") or not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
