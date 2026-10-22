"""Complete, atomic restart files; distinct from visualization snapshots."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .config import IceBodyConfig, IceFlowConfig, PhaseChangeProperties, ThermalConfig


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }


def _fields(simulation):
    return {
        name: value
        for name, value in vars(simulation).items()
        if name not in getattr(simulation, "_transient_field_names", ())
        and callable(getattr(value, "to_numpy", None))
        and callable(getattr(value, "from_numpy", None))
    }


def save_checkpoint(simulation, path):
    """Stream all device fields and host rigid state into an atomic ZIP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = _fields(simulation)
    rigid = {
        name: value
        for name, value in vars(simulation.rigid).items()
        if isinstance(value, np.ndarray)
    }
    metadata = {
        "format_version": 2,
        "configuration": simulation.config.to_dict(),
        "source_sha256": source_hashes(),
        "steps": simulation.steps,
        "thermal_pending": simulation._thermal_pending,
        "pending_cells": simulation.pending_cells,
        "initial_water_volume_m3": simulation.initial_water_volume_m3,
        "initial_phase_volume_m3": simulation.initial_phase_volume_m3,
        "fields": list(fields),
        "rigid": list(rigid),
    }
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:

        def write(name, array):
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)

        write("metadata", np.array(json.dumps(metadata)))
        write("initial_ice_mass_kg", simulation.initial_ice_mass_kg)
        for name, value in fields.items():
            write("field/" + name, value.to_numpy())
        for name, value in rigid.items():
            write("rigid/" + name, value)
    temporary.replace(path)


def load_checkpoint(path, *, backend=None, allow_source_change=False):
    from .simulator import IceFlow3D

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["format_version"] != 2:
            raise ValueError("unsupported checkpoint format: HOME requires a version-2 moment/AA checkpoint")
        if metadata["source_sha256"] != source_hashes() and not allow_source_change:
            raise ValueError(
                "checkpoint source differs; explicit allow_source_change required"
            )
        values = dict(metadata["configuration"])
        thermal = dict(values["thermal"])
        thermal["properties"] = PhaseChangeProperties(**thermal["properties"])
        values["thermal"] = ThermalConfig(**thermal)
        values["ice_bodies"] = tuple(IceBodyConfig(**b) for b in values["ice_bodies"])
        if backend is not None:
            values["backend"] = backend
        config = IceFlowConfig(**values)
        sim = IceFlow3D(replace(config, phase_warmup_steps=0))
        sim.config = config
        if set(metadata["fields"]) != set(_fields(sim)):
            raise ValueError("checkpoint field schema differs from the solver")
        for name in metadata["fields"]:
            getattr(sim, name).from_numpy(archive["field/" + name])
        for name in metadata["rigid"]:
            setattr(sim.rigid, name, archive["rigid/" + name].copy())
        sim.steps = metadata["steps"]
        sim._thermal_pending = metadata["thermal_pending"]
        sim.pending_cells = metadata["pending_cells"]
        sim.initial_water_volume_m3 = metadata["initial_water_volume_m3"]
        sim.initial_phase_volume_m3 = metadata["initial_phase_volume_m3"]
        sim.initial_ice_mass_kg = archive["initial_ice_mass_kg"].copy()
    # Older checkpoints may contain a gradient preceding the final geometry
    # update. Rebuild once, including when source changes were allowed.
    sim.invalidate_gradient_cache()
    return sim
