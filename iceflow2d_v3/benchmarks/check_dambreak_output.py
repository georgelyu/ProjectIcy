"""Audit dam-break visualizations and inventories, including a user-stopped run.

Run from the repository root:
python -m iceflow2d_v3.benchmarks.check_dambreak_output [output_directory]
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit(directory: Path) -> dict:
    directory = directory.resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    config = metadata["config"]
    assert not (directory / "failure.json").exists(), "failed run is not a completed result"
    count = int(metadata["results"]["snapshots"])
    assert count >= 2
    with (directory / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    body_count = len(config["ice_bodies"])
    assert len(rows) == count * body_count
    series = np.array([[float(row[key]) for key in (
        "physical_time_s", "lbm_steps", "water_volume_current_cells",
        "water_mean_temperature_c", "body_mass_kg_m", "body_center_x_m",
        "body_center_y_m", "body_angle_rad",
    )] for row in rows]).reshape(count, body_count, 8)
    assert np.isfinite(series).all()
    targets = np.asarray(metadata["lattice_scaling"]["scheduled_snapshot_steps"])
    assert targets.shape == (count,) and np.all(np.diff(targets) > 0)
    # The common runner rounds requested times up to a thermal-update boundary.
    # Compare with the recorded schedule, including its floating-point ceil,
    # rather than assuming all intermediate targets are exactly 1200 apart.
    dt = metadata["lattice_scaling"]["dt_s"]
    interval = config["thermal"]["update_interval_lbm_steps"]
    assert np.max(np.abs(targets * dt - np.arange(count) * .01)) <= interval * dt + 1e-12
    np.testing.assert_allclose(series[:, :, 0], targets[:, None] * dt + np.zeros((1, body_count)), atol=1e-12)
    np.testing.assert_array_equal(series[:, :, 1], targets[:, None] + np.zeros((1, body_count)))
    assert metadata["results"]["physical_time_s"] == series[-1, 0, 0]
    assert metadata["results"]["lbm_steps"] == targets[-1]
    water_mass = series[:, 0, 2] * config["rho_water"] * config["dx"]**2
    ice_mass = series[:, :, 4].sum(axis=1)
    total_mass = water_mass + ice_mass
    mass_error = float(np.max(np.abs(total_mass / total_mass[0] - 1)))
    thermal = config["thermal"]
    properties = thermal["properties"]
    melting_temperature = properties["melting_temperature_c"]
    assert thermal["initial_ice_temperature_c"] == melting_temperature
    energy = water_mass * properties["specific_heat_water_j_kg_k"] * (series[:, 0, 3] - melting_temperature)
    energy += (ice_mass[0] - ice_mass) * properties["latent_heat_j_kg"]
    energy_error = float(np.max(np.abs(energy / energy[0] - 1)))
    assert mass_error < 1e-8, mass_error
    assert energy_error < 1e-8, energy_error
    result = {
        "output_directory": str(directory), "physical_time_s": float(series[-1, 0, 0]),
        "lbm_steps": int(targets[-1]), "snapshots": count,
        "status": metadata.get("status", "completed"),
        "max_relative_total_mass_error": mass_error,
        "max_relative_total_energy_error": energy_error,
        "initial_ice_mass_kg_m": float(ice_mass[0]),
        "final_ice_mass_kg_m": float(ice_mass[-1]),
        "final_melted_fraction": float(1 - ice_mass[-1] / ice_mass[0]),
        "final_water_mean_temperature_c": float(series[-1, 0, 3]),
        "max_body_displacement_m": np.linalg.norm(series[:, :, 5:7] - series[0, :, 5:7], axis=2).max(axis=0).tolist(),
        "max_body_rotation_rad": np.abs(series[:, :, 7] - series[0, :, 7]).max(axis=0).tolist(),
        "visualizations": {},
    }
    for name in ("velocity", "vorticity", "temperature", "phase"):
        frames = sorted((directory / f"{name}_frames").glob("frame_*.png"))
        assert len(frames) == count, (name, len(frames))
        for index in (0, count//2, count-1):
            with Image.open(frames[index]) as frame:
                frame.verify()
        with Image.open(directory / f"{name}_field.gif") as animation:
            assert animation.n_frames == count, (name, animation.n_frames)
            animation.seek(count-1)
            animation.load()
        result["visualizations"][name] = {"png_frames": count, "gif_frames": count}
    result["raw_fields_audited"] = False
    if (directory / "fields.npz").exists():
        with np.load(directory / "fields.npz") as data:
            np.testing.assert_allclose(data["physical_time_s"], series[:, 0, 0], atol=1e-12)
            np.testing.assert_array_equal(data["lbm_steps"], series[:, 0, 1])
            for key in ("water_phase", "water_volume_cells", "fluid_capacity", "temperature_c", "physical_velocity_lattice", "water_sensible_energy_j_m"):
                field = data[key]
                assert field.shape[:3] == (count, *config["resolution"]), (key, field.shape)
                assert np.isfinite(field).all(), key
                if key in ("water_phase", "water_volume_cells", "fluid_capacity"):
                    assert field.min() >= -1e-10 and field.max() <= 1 + 1e-10, key
                if key == "water_volume_cells":
                    np.testing.assert_allclose(field.sum(axis=(1, 2)), series[:, 0, 2], atol=1e-8, rtol=1e-12)
                if key == "temperature_c":
                    result["temperature_range_c"] = [float(field.min()), float(field.max())]
                if key == "physical_velocity_lattice":
                    speed = np.linalg.norm(field, axis=-1)
                    result["max_full_domain_speed_m_s"] = float(speed.max() * metadata["lattice_scaling"]["velocity_scale_m_s"])
                del field
            np.testing.assert_allclose(data["body_mass_kg_m"], series[:, :, 4], atol=1e-12)
            result["raw_fields_audited"] = True
    manifest = json.loads((directory / "run_manifest.json").read_text())
    root = Path(__file__).resolve().parents[2]
    source_root = directory / manifest["source_snapshot_directory"] if "source_snapshot_directory" in manifest else root
    current_differences = []
    for relative_path, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((source_root / relative_path).read_bytes()).hexdigest() == digest, relative_path
        if hashlib.sha256((root / relative_path).read_bytes()).hexdigest() != digest:
            current_differences.append(relative_path)
    result["run_sources_match_manifest"] = True
    result["current_worktree_sources_match_run"] = not current_differences
    result["current_worktree_sources_different_from_run"] = current_differences
    result["passed"] = True
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path, nargs="?", default=Path("iceflow2d_v3/outputs/dambreak_four_ice_melting_2d"))
    print(json.dumps(audit(parser.parse_args().output_directory), indent=2))


if __name__ == "__main__":
    main()
