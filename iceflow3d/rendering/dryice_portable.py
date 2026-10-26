"""Losslessly repack the fast h5py archive into portable GZIP HDF5/XDMF.

LZF is bundled with h5py but is not a mandatory filter in other HDF5 readers.
GZIP is portable (https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline).
The simulation's immutable original frames and checksums are retained.
"""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from .dryice_export import sha256, write_json, write_xdmf


def convert(source, target):
    tmp = target.with_suffix(".h5.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as src, h5py.File(tmp, "w") as dst:
        dst.attrs.update(src.attrs)
        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                dst.require_group(name).attrs.update(obj.attrs)
            else:
                a = obj[()]
                kwargs = dict(compression="gzip", compression_opts=1, shuffle=True, fletcher32=True,
                              chunks=obj.chunks or True) if a.size > 1024 else {}
                d = dst.create_dataset(name, data=a, **kwargs)
                d.attrs.update(obj.attrs)
        src.visititems(copy)
    # Compare every numeric dataset after reopening, before publishing it.
    with h5py.File(source, "r") as src, h5py.File(tmp, "r") as dst:
        def compare(name, obj):
            if isinstance(obj, h5py.Dataset):
                assert np.array_equal(obj[()], dst[name][()], equal_nan=True), name
        src.visititems(compare)
    tmp.replace(target)
    return dict(source_sha256=sha256(source), sha256=sha256(target),
                bytes=target.stat().st_size, array_values_identical=True)


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
        record = {k: sim[k] for k in ("physical_grid", "dx_m") if k in sim}
        record.update(status=sim["status"], frames=[], compression="portable GZIP level 1 + shuffle, lossless",
                      originals=str(args.simulation.resolve()), values_identical_to_original=True)
        for frame in sim["frames"]:
            output = args.output / frame["path"]
            info_path = output.with_suffix(".json")
            info = json.loads(info_path.read_text()) if info_path.exists() and output.exists() else None
            if info is None or info["source_sha256"] != frame["sha256"]:
                t = time.perf_counter()
                info = convert(args.simulation / frame["path"], output)
                assert info["source_sha256"] == frame["sha256"]
                info.update(index=frame["index"], step=frame["step"], time_s=frame["time_s"],
                            path=frame["path"], wall_time_s=time.perf_counter() - t)
                write_json(info_path, info)
                print(json.dumps(info), flush=True)
            record["frames"].append(info)
        if "dx_m" in record:
            write_xdmf(args.output, record)
        write_json(args.output / "portable.json", record)
        if sim["status"] in ("completed", "failed") or not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
