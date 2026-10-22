"""Extract full-resolution, closed water/ice surfaces for offline Blender."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from simulate import save_npz, write_json


# Right-handed mapping: solver (x, up-y, depth-z) -> Blender (x, -z, up-y).
AXES = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
CACHE_VERSION = 3


def closed_surface(values, dx, sigma=0.45):
    """Subcell filtering only; no spatial downsampling or invented splashes."""
    field = np.pad(np.asarray(values, np.float32), 2)
    if sigma:
        field = ndimage.gaussian_filter(field, sigma=sigma, mode="constant")
    if float(field.max()) <= 0.5:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
    vertices, faces, _, _ = marching_cubes(
        field, level=0.5, spacing=(dx, dx, dx), allow_degenerate=False
    )
    # Values are at cell centres; account for padding and the half-cell origin.
    vertices += (0.5 - 2) * dx
    faces = np.asarray(faces, np.int32)
    volume = (
        np.einsum(
            "ij,ij->i",
            vertices[faces[:, 0]],
            np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
        ).sum()
        / 6
    )
    if volume < 0:
        faces = faces[:, [0, 2, 1]].copy()
    return np.asarray(vertices, np.float32), faces


def extract_frame(field_path, output_path):
    with np.load(field_path, allow_pickle=False) as fields:
        dx = float(fields["dx_m"])
        shape = fields["water_phase"].shape
        lengths = np.array(fields["domain_size_m"])
        shift = np.array([-lengths[0] / 2, lengths[2] / 2, 0])
        phi = fields["water_phase"].copy()
        ids = fields["solid_body_id"]
        if np.any(ids >= 0):
            # The free surface must continue through submerged ice rather than
            # introduce a fictitious air shell around every ice boundary.
            nearest = ndimage.distance_transform_edt(
                ids != -1, return_distances=False, return_indices=True
            )
            solid = ids >= 0
            phi[solid] = phi[tuple(axis[solid] for axis in nearest)]
            del nearest
        vertices, faces = closed_surface(phi, dx)
        optical_phase = ndimage.gaussian_filter(phi, sigma=0.45, mode="constant")
        sample_coords = (vertices / dx - 0.5).T
        cache = {
            "water_vertices": (vertices @ AXES.T + shift).astype(np.float32),
            "water_faces": faces,
            "time_s": fields["time_s"],
            "step": fields["step"],
            "dx_m": dx,
            "domain_size_m": lengths,
            "boundary_cells": fields["boundary_cells"],
        }
        # The current fixed-shutter renderer only consumes geometry. Preserve
        # real velocities when supplied, without inventing zero velocities.
        if "velocity_m_s" in fields:
            velocity_field = fields["velocity_m_s"]
            velocity = np.stack([
                ndimage.map_coordinates(velocity_field[..., a], sample_coords,
                    order=1, mode="nearest") for a in range(3)
            ], axis=-1)
            cache["water_velocity_m_s"] = (velocity @ AXES.T).astype(np.float32)
        counts = {"water_vertices": len(vertices), "water_triangles": len(faces)}
        for body, size in enumerate(fields["material_size"]):
            fraction = fields["ice_material_fraction"][
                body, : size[0], : size[1], : size[2]
            ]
            points, triangles = closed_surface(fraction, dx, sigma=0.4)
            local = points - 0.5 * size * dx
            world = (
                local @ fields["body_rotation"][body].T + fields["body_origin_m"][body]
            )
            wet_phase = ndimage.map_coordinates(
                optical_phase, (world / dx - 0.5).T, order=1, mode="nearest"
            )
            # Cycles Glass IOR is a relative interface ratio. A submerged
            # ice face meets water, while an exposed face meets air.
            medium_ior = 1.0 + 0.333 * np.clip((wet_phase - 0.45) / 0.1, 0, 1)
            transform = np.eye(4)
            transform[:3, :3] = AXES @ fields["body_rotation"][body] @ AXES.T
            transform[:3, 3] = fields["body_origin_m"][body] @ AXES.T + shift
            cache[f"ice_{body}_vertices"] = (local @ AXES.T).astype(np.float32)
            cache[f"ice_{body}_faces"] = triangles
            cache[f"ice_{body}_transform"] = transform
            cache[f"ice_{body}_mass_kg"] = fields["body_mass_kg"][body]
            cache[f"ice_{body}_medium_ior"] = medium_ior.astype(np.float32)
            counts[f"ice_{body}_triangles"] = len(triangles)
        assert all(np.isfinite(value).all() for value in cache.values())
        save_npz(output_path, cache)
        return {
            "cache_version": CACHE_VERSION,
            "time_s": float(fields["time_s"]),
            "step": int(fields["step"]),
            "grid": list(shape),
            **counts,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = {
        "cache_version": CACHE_VERSION,
        "status": "waiting",
        "frames": [],
        "coordinate_mapping": "Blender=(solver_x-Lx/2,Lz/2-solver_z,solver_y), metres",
        "water_surface": "phi=0.5; nearest-fluid extension inside ice; Gaussian sigma=0.45 cells",
        "ice_surface": "material fraction=0.5; Gaussian sigma=0.4 cells; rigid body transform",
        "ice_optics": "relative Glass IOR=1.31/medium_ior; surrounding medium sampled at ice surface",
        "resolution": "full simulation grid; no downsampling",
    }
    while True:
        metadata_path = args.simulation / "simulation.json"
        if not metadata_path.exists():
            if not args.watch:
                raise FileNotFoundError(metadata_path)
            time.sleep(5)
            continue
        simulation = json.loads(metadata_path.read_text())
        for frame in simulation["frames"]:
            filename = f"frame_{frame['index']:06d}.npz"
            info_path = args.output / filename.replace(".npz", ".json")
            if info_path.exists():
                info = json.loads(info_path.read_text())
                if (
                    info["field_sha256"] == frame["sha256"]
                    and info.get("cache_version") == CACHE_VERSION
                ):
                    continue
            print(f"Extracting {filename} at t={frame['time_s']:.2f}s", flush=True)
            field_path = args.simulation / frame["path"]
            if hashlib.sha256(field_path.read_bytes()).hexdigest() != frame["sha256"]:
                raise RuntimeError(f"input field checksum changed: {field_path}")
            info = extract_frame(field_path, args.output / filename)
            info.update(
                index=frame["index"],
                path=filename,
                field_sha256=frame["sha256"],
            )
            write_json(info_path, info)
        manifest["frames"] = [
            json.loads((args.output / f"frame_{f['index']:06d}.json").read_text())
            for f in simulation["frames"]
            if (args.output / f"frame_{f['index']:06d}.json").exists()
        ]
        manifest["status"] = (
            "completed"
            if simulation["status"] == "completed"
            else (
                "failed"
                if simulation["status"] == "failed"
                else "waiting_for_simulation"
            )
        )
        manifest["wall_time_s"] = time.perf_counter() - start
        write_json(args.output / "meshes.json", manifest)
        if simulation["status"] == "failed":
            raise RuntimeError(f"simulation failed: {simulation.get('error')}")
        if simulation["status"] == "completed" or not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
