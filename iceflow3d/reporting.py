"""Streaming 3D/slice PNG+GIF, optional VTK volumes and per-sample NPZ."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import numpy as np


class StreamingGif:
    """Keep at most one decoded frame in memory, as in the v3 reporter."""

    def __init__(self, path, duration_ms=120):
        self.path = Path(path)
        self.stream = self.path.open("wb")
        self.count = 0
        self.duration_ms = duration_ms

    def append(self, png):
        from PIL import GifImagePlugin, Image

        with Image.open(png) as source:
            frame = source.convert("RGB").convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=256
            )
        try:
            info = {"duration": self.duration_ms, "disposal": 2, "optimize": False}
            if self.count == 0:
                info["loop"] = 0
                for block in GifImagePlugin._get_global_header(frame, info):
                    self.stream.write(block)
            else:
                info["include_color_table"] = True
            GifImagePlugin._write_frame_data(self.stream, frame, (0, 0), info)
            self.stream.flush()
            self.count += 1
        finally:
            frame.close()

    def close(self):
        if not self.stream.closed:
            if self.count:
                self.stream.write(b";")
            self.stream.close()
            if not self.count:
                self.path.unlink(missing_ok=True)


def write_vti(path, fields, dx):
    """VTK ImageData with SI cell coordinates and raw little-endian arrays."""
    shape = fields["water_phase"].shape
    extent = f"0 {shape[0]} 0 {shape[1]} 0 {shape[2]}"
    root = ET.Element(
        "VTKFile",
        type="ImageData",
        version="1.0",
        byte_order="LittleEndian",
        header_type="UInt64",
    )
    image = ET.SubElement(
        root, "ImageData", WholeExtent=extent, Origin="0 0 0", Spacing=f"{dx} {dx} {dx}"
    )
    piece = ET.SubElement(image, "Piece", Extent=extent)
    data = ET.SubElement(
        piece, "CellData", Scalars="water_phase", Vectors="velocity_m_s"
    )
    arrays, offset = [], 0
    for name, values in fields.items():
        dtype = "<i4" if np.issubdtype(values.dtype, np.integer) else "<f4"
        values = np.asarray(values, dtype=dtype)
        # VTK x is fastest, with vector components contiguous at each cell.
        order = (2, 1, 0, 3) if values.ndim == 4 else (2, 1, 0)
        payload = np.ascontiguousarray(values.transpose(order))
        components = values.shape[3] if values.ndim == 4 else 1
        ET.SubElement(
            data,
            "DataArray",
            type="Int32" if dtype == "<i4" else "Float32",
            Name=name,
            NumberOfComponents=str(components),
            format="appended",
            offset=str(offset),
        )
        arrays.append(payload)
        offset += 8 + payload.nbytes
    prefix = ET.tostring(root, encoding="utf-8")
    with Path(path).open("wb") as stream:
        stream.write(b'<?xml version="1.0"?>\n')
        stream.write(prefix.removesuffix(b"</VTKFile>"))
        stream.write(b'<AppendedData encoding="raw">_')
        for array in arrays:
            stream.write(struct.pack("<Q", array.nbytes))
            stream.write(memoryview(array).cast("B"))
        stream.write(b"</AppendedData></VTKFile>\n")


def write_pvd(path, samples):
    root = ET.Element(
        "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
    )
    collection = ET.SubElement(root, "Collection")
    for time, filename in samples:
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time:.16g}",
            group="",
            part="0",
            file=filename,
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def boundary_mesh(mask, coordinates):
    """Only exposed voxel faces; rendering decimation never affects the solver."""
    faces = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            neighbor = np.roll(mask, -sign, axis=axis)
            end = [slice(None)] * 3
            end[axis] = -1 if sign == 1 else 0
            neighbor[tuple(end)] = False
            cells = np.array(np.nonzero(mask & ~neighbor)).T
            if not len(cells):
                continue
            verts = np.empty((len(cells), 4, 3))
            for corner, offsets in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                for a in range(3):
                    shift = (
                        (1 if sign == 1 else 0)
                        if a == axis
                        else offsets[others.index(a)]
                    )
                    verts[:, corner, a] = coordinates[a][cells[:, a] + shift]
            # Display horizontal depth as z and height as the plotting z axis.
            faces.append(verts[:, :, [0, 2, 1]])
    return np.concatenate(faces) if faces else np.empty((0, 4, 3))


def render_frame(path, config, scales, fields, time_s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    phi, ids = fields["water_phase"], fields["solid_body_id"]
    shape = phi.shape
    stride = max(1, math_ceil(shape[0] / 120))
    indices = [np.arange(0, n, stride) for n in shape]
    view = np.ix_(*indices)
    coordinates = [np.r_[idx, n] * scales.dx_m * 1000 for idx, n in zip(indices, shape)]
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.1, 1))
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    wet = (phi[view] >= 0.5) & (ids[view] == -1)
    for mask, color, alpha in (
        (wet, "#2e9cca", 0.22),
        (ids[view] >= 0, "#edae49", 1.0),
    ):
        mesh = boundary_mesh(mask, coordinates)
        if len(mesh):
            ax.add_collection3d(
                Poly3DCollection(
                    mesh,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=(0.15, 0.35, 0.4, 0.12),
                    linewidth=0.12,
                )
            )
    lengths = np.array(config.domain_size_m) * 1000
    ax.set(
        xlim=(0, lengths[0]),
        ylim=(0, lengths[2]),
        zlim=(0, lengths[1]),
        xlabel="x [mm]",
        ylabel="z [mm]",
        zlabel="y [mm]",
        title=f"3D water and melting ice | t = {time_s:.6f} s",
    )
    ax.set_box_aspect(lengths[[0, 2, 1]], zoom=0.82)
    ax.view_init(elev=25, azim=-62)
    ax.tick_params(labelsize=8, pad=1)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#2e9cca", label="Water"),
            Patch(facecolor="#edae49", label="Ice"),
        ],
        loc="upper left",
    )
    mid = shape[2] // 2
    solid = ids[:, :, mid] != -1
    extent = (0, lengths[0], 0, lengths[1])
    temperature = np.where(
        (phi[:, :, mid] < 0.5) & ~solid,
        config.thermal.initial_air_temperature_c,
        fields["temperature_c"][:, :, mid],
    )
    speed = np.linalg.norm(fields["velocity_m_s"][:, :, mid], axis=-1)
    for slot, values, cmap, vmax, label in (
        (
            gs[0, 1],
            temperature,
            "inferno",
            config.thermal.initial_water_temperature_c,
            "Temperature [degC]",
        ),
        (gs[1, 1], speed, "viridis", 0.8, "Speed [m/s]"),
    ):
        axis = fig.add_subplot(slot)
        mask = (ids[:, :, mid] < -1) if label.startswith("Temperature") else solid
        image = axis.imshow(
            np.ma.masked_where(mask, values).T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            aspect="equal",
        )
        axis.imshow(
            np.ma.masked_where(~mask, np.ones(solid.shape)).T,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=2,
            interpolation="nearest",
        )
        phase = phi[:, :, mid].T
        if np.min(phase) < 0.5 < np.max(phase):
            xs = (np.arange(shape[0]) + 0.5) * scales.dx_m * 1000
            ys = (np.arange(shape[1]) + 0.5) * scales.dx_m * 1000
            axis.contour(
                xs,
                ys,
                np.ma.masked_where(solid.T, phase),
                levels=[0.5],
                colors="white",
                linewidths=0.65,
            )
        axis.set(
            xlabel="x [mm]",
            ylabel="y [mm]",
            title=f"{label}, z = {(mid + 0.5) * scales.dx_m * 1000:.2f} mm",
        )
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def math_ceil(x):
    return int(np.ceil(x))
