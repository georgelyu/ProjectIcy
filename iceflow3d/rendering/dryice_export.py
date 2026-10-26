"""Lossless 3-D scientific output for the unmodified dry-ice example.

Run with ``python -m iceflow3d.rendering.dryice_export --output-dir ...``.
HDF5 arrays use z,y,x storage, vector components remain solver x,y,z (y up).
The XDMF time series opens directly in ParaView. These are visualization
snapshots, not complete solver restart files.
"""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from xml.etree import ElementTree as ET

import h5py
import numpy as np

from iceflow3d.examples.dry_ice_in_hot_water_3d import build_parser, create_config


UNITS = {
    "water_phase": "1", "solid_body_id": "1", "ice_fraction": "1",
    "velocity_m_s": "m/s", "pressure_pa": "Pa (gauge, ambient at top opening)",
    "temperature_c": "degC", "water_temperature_c": "degC",
    "gas_mass_source_kg_m3_s": "kg/(m^3 s)",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_xdmf(root, record):
    n = record["physical_grid"][0]
    dx = record["dx_m"]
    doc = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(doc, "Domain")
    collection = ET.SubElement(domain, "Grid", Name="Dry ice in hot water",
                               GridType="Collection", CollectionType="Temporal")
    for f in record["frames"]:
        grid = ET.SubElement(collection, "Grid", Name=f"frame_{f['index']:06d}", GridType="Uniform")
        ET.SubElement(grid, "Time", Value=str(f["time_s"]))
        # Point locations are the physical cell centres, not the domain corners.
        ET.SubElement(grid, "Topology", TopologyType="3DCoRectMesh", Dimensions=f"{n} {n} {n}")
        geometry = ET.SubElement(grid, "Geometry", GeometryType="ORIGIN_DXDYDZ")
        ET.SubElement(geometry, "DataItem", Dimensions="3", Format="XML").text = " ".join([str(dx / 2)] * 3)
        ET.SubElement(geometry, "DataItem", Dimensions="3", Format="XML").text = " ".join([str(dx)] * 3)
        for name in UNITS:
            vector = name == "velocity_m_s"
            integer = name == "solid_body_id"
            a = ET.SubElement(grid, "Attribute", Name=name, Center="Node",
                              AttributeType="Vector" if vector else "Scalar")
            ET.SubElement(a, "DataItem", Dimensions=f"{n} {n} {n}" + (" 3" if vector else ""),
                          NumberType="Int" if integer else "Float", Precision="1" if integer else "4",
                          Format="HDF").text = f"{f['path']}:/fields/{name}"
    tmp = root / "fields.xdmf.tmp"
    ET.indent(doc)
    ET.ElementTree(doc).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(root / "fields.xdmf")


def save_frame(sim, path):
    from iceflow3d.rigid import rotation_matrix
    fields = sim.snapshot()
    n = sim.nx - 2
    slices = (slice(1, -1),) * 3
    fluid = fields["solid_body_id"][slices] == -1
    statistics = {}
    tmp = path.with_suffix(".h5.tmp")
    dx = sim.scales.dx_m
    with h5py.File(tmp, "w") as f:
        f.attrs.update(format_version=1, time_s=sim.physical_time_s, step=sim.steps,
                       dx_m=dx, dt_s=sim.scales.dt_s, axis_order="zyx",
                       vector_component_order="xyz; y is vertical",
                       sample_origin_m=np.full(3, dx / 2),
                       physical_size_m=sim.config.physical_size_m,
                       gas_temperature_c=sim.config.thermal.initial_air_temperature_c,
                       gas_thermal_model="prescribed isothermal CO2 reservoir",
                       pressure_reference="gauge zero at ambient top opening; includes hydrostatic head",
                       valid_fluid="solid_body_id == -1; -2=wall, >=0=dry ice")
        group = f.create_group("fields")
        for name in UNITS:
            a = fields[name][slices]
            if not np.isfinite(a[fluid]).all():
                raise FloatingPointError(f"nonfinite fluid {name} at step {sim.steps}")
            axes = (2, 1, 0, 3) if a.ndim == 4 else (2, 1, 0)
            a = np.ascontiguousarray(a.transpose(axes), dtype=np.int8 if name == "solid_body_id" else np.float32)
            chunks = (min(16, n), min(54, n), min(54, n)) + ((3,) if a.ndim == 4 else ())
            d = group.create_dataset(name, data=a, chunks=chunks, compression="lzf",
                                     shuffle=True, fletcher32=True)
            d.attrs["units"] = UNITS[name]
            values = fields[name][slices][fluid]
            statistics[name] = {"min": float(values.min()), "max": float(values.max())}
        body = f.create_group("body")
        body.create_dataset("material_fraction", data=sim.material_mass.to_numpy() / sim.initial_cell_mass,
                            compression="lzf", shuffle=True, fletcher32=True)
        body.create_dataset("material_mass_kg", data=sim.material_mass.to_numpy())
        body.create_dataset("material_energy_j", data=sim.material_energy.to_numpy())
        body.create_dataset("material_size", data=sim.material_size.to_numpy())
        # Body origins in the same physical coordinate system as the trimmed grid.
        body.create_dataset("origin_m", data=(sim.rigid.origin - 1) * dx)
        body.create_dataset("rotation", data=np.array([rotation_matrix(q) for q in sim.rigid.q]))
        body.create_dataset("quaternion_wxyz", data=sim.rigid.q)
        body.create_dataset("mass_kg", data=sim.rigid.mass * sim.scales.mass_kg)
        body.create_dataset("velocity_m_s", data=sim.rigid.velocity * sim.scales.velocity_m_s)
        body.create_dataset("angular_velocity_s_1", data=sim.rigid.omega / sim.scales.dt_s)
    tmp.replace(path)
    # Keep an inexpensive central section as well as the complete 3D fields.
    z = sim.nz // 2
    central = {k: v[1:-1, 1:-1, z].swapaxes(0, 1) for k, v in fields.items()}
    central.update(time_s=sim.physical_time_s, step=sim.steps, dx_m=dx,
                   z_m=(z - 0.5) * dx, axis_order="yx", vector_component_order="xyz")
    slice_path = path.parent.parent / "slices" / (path.stem + ".npz")
    with slice_path.with_suffix(".npz.tmp").open("wb") as f:
        np.savez_compressed(f, **central)
    slice_path.with_suffix(".npz.tmp").replace(slice_path)
    return statistics


def main():
    args = build_parser().parse_args()
    config = create_config(args)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "simulation.json"
    if manifest.exists() or any((root / "fields").glob("*.h5")):
        raise FileExistsError(f"refusing to overwrite {root}")
    for folder in ("fields", "slices"):
        (root / folder).mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    paths = list(source.glob("*.py")) + list((source / "dryice").glob("*.py")) + [
        source / "examples/dry_ice_in_hot_water_3d.py", Path(__file__).resolve()]
    record = dict(status="initializing", started_utc=now(), pid=os.getpid(),
                  configuration=asdict(config), frames=[], physical_grid=[args.resolution] * 3,
                  requested_end_time_s=args.end_time_s, output_interval_s=args.output_interval_s,
                  source_root=str(source.parent),
                  source_sha256={str(p.relative_to(source)): sha256(p) for p in paths},
                  data_format="HDF5 LZF+shuffle lossless float32; z,y,x; SI units",
                  gas_thermal_model="CO2 prescribed isothermal 20 degC; no gas temperature equation",
                  pressure_definition="gauge Pa relative to ambient at top opening, includes hydrostatic head",
                  timing_s={k: 0.0 for k in ("initialization", "stepping", "diagnostics", "output")})
    write_json(manifest, record)
    start = time.perf_counter()
    sim = None
    try:
        from iceflow3d.dryice.simulator import DryIceFlow3D
        sim = DryIceFlow3D(config)
        record["timing_s"]["initialization"] = time.perf_counter() - start
        dt = sim.scales.dt_s
        end = args.steps if args.steps is not None else round(args.end_time_s / dt)
        stride = round(args.output_interval_s / dt)
        if stride <= 0 or end % stride or not np.isclose(stride * dt, args.output_interval_s, rtol=0, atol=1e-12):
            raise ValueError("requested sample times must align exactly with dt")
        record.update(status="running", target_steps=end, output_stride_steps=stride,
                      expected_frames=end // stride + 1, dt_s=dt, dx_m=sim.scales.dx_m)
        with (root / "history.csv").open("w", newline="") as stream:
            writer = None
            for index, target in enumerate(range(0, end + 1, stride)):
                t = time.perf_counter()
                sim.step(target - sim.steps)
                if target == end:
                    sim.flush_thermal()
                record["timing_s"]["stepping"] += time.perf_counter() - t
                t = time.perf_counter()
                diag = sim.diagnostics()
                if not all(np.isfinite(v).all() for v in diag.values() if v is not None):
                    raise FloatingPointError("nonfinite diagnostic")
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(diag))
                    writer.writeheader()
                writer.writerow(diag)
                stream.flush()
                record["timing_s"]["diagnostics"] += time.perf_counter() - t
                t = time.perf_counter()
                relative = f"fields/frame_{index:06d}.h5"
                statistics = save_frame(sim, root / relative)
                record["frames"].append(dict(index=index, step=sim.steps, time_s=sim.physical_time_s,
                    path=relative, bytes=(root / relative).stat().st_size,
                    sha256=sha256(root / relative), statistics=statistics))
                write_xdmf(root, record)
                record["timing_s"]["output"] += time.perf_counter() - t
                record.update(current_step=sim.steps, current_time_s=sim.physical_time_s,
                              last_diagnostics=diag, wall_time_s=time.perf_counter() - start,
                              updated_utc=now())
                write_json(manifest, record)
                print(json.dumps(dict(frame=index, step=sim.steps, time_s=sim.physical_time_s,
                    elapsed_s=record["wall_time_s"], file_mb=record["frames"][-1]["bytes"] / 1e6)), flush=True)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      failure_step=sim.steps if sim else 0,
                      failure_time_s=sim.physical_time_s if sim else 0)
        raise
    finally:
        record.update(ended_utc=now(), wall_time_s=time.perf_counter() - start)
        write_json(manifest, record)


if __name__ == "__main__":
    main()
