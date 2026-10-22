"""Blender 5 / Cycles studio film of the actual cached water and melting ice.

Usage: blender -b --python blender_scene.py -- --cache ... --output ... --all
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix, Vector
import numpy as np


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def material(name, color, roughness=0.3, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat, bsdf


def dielectric(name, ior, roughness, absorption=None, density=0):
    mat, old = material(name, (1, 1, 1), roughness)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(old)
    shader = nodes.new("ShaderNodeBsdfGlass")
    shader.name = "Dielectric"
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["IOR"].default_value = ior
    links.new(shader.outputs["BSDF"], nodes.get("Material Output").inputs["Surface"])
    if absorption is not None:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        volume = nodes.new("ShaderNodeVolumeAbsorption")
        volume.inputs["Color"].default_value = (*absorption, 1)
        volume.inputs["Density"].default_value = density
        links.new(
            volume.outputs["Volume"], nodes.get("Material Output").inputs["Volume"]
        )
    return mat


def ice_material():
    mat = dielectric("ICE · frost, microstructure and transmission", 1.31, 0.115)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    medium = nodes.new("ShaderNodeAttribute")
    medium.attribute_name = "medium_ior"
    relative_ior = nodes.new("ShaderNodeMath")
    relative_ior.operation = "DIVIDE"
    relative_ior.inputs[0].default_value = 1.31
    links.new(medium.outputs["Fac"], relative_ior.inputs[1])
    links.new(relative_ior.outputs[0], nodes.get("Dielectric").inputs["IOR"])
    texture = nodes.new("ShaderNodeTexCoord")
    grain = nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 4200
    grain.inputs["Detail"].default_value = 2
    links.new(texture.outputs["Object"], grain.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.000012
    links.new(grain.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes.get("Dielectric").inputs["Normal"])
    cloud = nodes.new("ShaderNodeTexNoise")
    cloud.inputs["Scale"].default_value = 750
    cloud.inputs["Detail"].default_value = 3
    links.new(texture.outputs["Object"], cloud.inputs["Vector"])
    density = nodes.new("ShaderNodeMapRange")
    density.inputs["From Min"].default_value = 0.2
    density.inputs["From Max"].default_value = 0.8
    density.inputs["To Min"].default_value = 30
    density.inputs["To Max"].default_value = 230
    links.new(cloud.outputs["Fac"], density.inputs["Value"])
    # One volume closure avoids the 64-closure budget consumed by adding
    # separate scatter and absorption nodes (each reserves a volume stack).
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Color"].default_value = (0.86, 0.94, 0.99, 1)
    volume.inputs["Absorption Color"].default_value = (0.98, 0.995, 1.0, 1)
    volume.inputs["Anisotropy"].default_value = 0.15
    links.new(density.outputs["Result"], volume.inputs["Density"])
    links.new(volume.outputs["Volume"], nodes.get("Material Output").inputs["Volume"])
    return mat


def box(name, location, size, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("Machined edge", "BEVEL")
        modifier.width, modifier.segments = bevel, 5
        modifier = obj.modifiers.new("Corner normals", "WEIGHTED_NORMAL")
        modifier.keep_sharp = True
    return obj


def aim(obj, target):
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def light(name, location, target, energy, size, color=(1, 1, 1), size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.color = color
    data.shape = "RECTANGLE" if size_y else "DISK"
    data.size = size
    if size_y:
        data.size_y = size_y
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    if hasattr(data.cycles, "is_caustics_light"):
        data.cycles.is_caustics_light = True
    return obj


def mesh_object(name, vertices, faces, mat, transform=None, medium_ior=None):
    obj = bpy.data.objects.get(name)
    previous = obj.data if obj else None
    mesh = bpy.data.meshes.new(name + " surface")
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", np.asarray(vertices, np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", np.asarray(faces, np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(len(faces), dtype=np.int32) * 3)
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    mesh.materials.append(mat)
    if medium_ior is not None:
        attribute = mesh.attributes.new("medium_ior", "FLOAT", "POINT")
        attribute.data.foreach_set("value", np.asarray(medium_ior, np.float32))
    mesh.update()
    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
    else:
        obj.data = mesh
    obj.hide_render = not len(faces)
    if transform is not None:
        obj.matrix_world = Matrix(transform.tolist())
    if previous is not None and previous.users == 0:
        bpy.data.meshes.remove(previous)
    if hasattr(obj.cycles, "is_caustics_caster"):
        obj.cycles.is_caustics_caster = True
    return obj


def load_frame(cache, index, materials):
    with np.load(cache / f"frame_{index:06d}.npz", allow_pickle=False) as fields:
        mesh_object(
            "WATER", fields["water_vertices"], fields["water_faces"], materials[0]
        )
        for body in range(4):
            mesh_object(
                f"ICE {body + 1}",
                fields[f"ice_{body}_vertices"],
                fields[f"ice_{body}_faces"],
                materials[1],
                fields[f"ice_{body}_transform"],
                fields[f"ice_{body}_medium_ior"],
            )
        bpy.context.scene["physical_time_s"] = float(fields["time_s"])
        bpy.context.scene["simulation_step"] = int(fields["step"])


def build_scene(args):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = args.noise_threshold
    scene.cycles.adaptive_min_samples = args.min_samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = args.denoiser
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = args.device != "CPU" and args.denoise_device == "GPU"
    scene.cycles.max_bounces = args.max_bounces
    scene.cycles.diffuse_bounces = args.diffuse_bounces
    scene.cycles.glossy_bounces = args.glossy_bounces
    scene.cycles.transmission_bounces = args.transmission_bounces
    scene.cycles.transparent_max_bounces = args.max_bounces
    scene.cycles.volume_bounces = args.volume_bounces
    # Blender 5.0.1's null-scattering octree feeds tiny mesh dimensions to
    # OpenVDB, whose determinant tolerance aborts on millimetre-scale ice.
    # Ray marching preserves the heterogeneous volume without that SDF path.
    scene.cycles.volume_biased = args.volume_integrator == "ray_marching"
    scene.cycles.volume_step_rate = args.volume_step_rate
    scene.cycles.volume_max_steps = 4096
    scene.cycles.sample_clamp_indirect = 8.0
    scene.cycles.seed = 37
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    scene.render.fps = 50
    scene.frame_start, scene.frame_end = 1, args.frame_count
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 25
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.3
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.threads_mode = "FIXED"
    scene.render.threads = args.threads
    if args.device == "CPU":
        scene.cycles.device = "CPU"
    else:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = args.device
        preferences.get_devices()
        enabled = []
        for device in preferences.devices:
            device.use = device.type == args.device
            if device.use:
                enabled.append(device.name)
        if not enabled:
            raise RuntimeError(f"No {args.device} renderer available")
        scene.cycles.device = "GPU"
        print("GPU_RENDER_DEVICES", enabled, flush=True)

    world = bpy.data.worlds.new("Studio ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.34,
        0.40,
        0.50,
        1,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    scene.world = world

    water = dielectric(
        "WATER · clear aqua, IOR 1.333",
        1.333,
        0.018,
        absorption=(0.72, 0.93, 0.98),
        density=8,
    )
    ice = ice_material()
    glass = dielectric(
        "Low-iron optical glass",
        1.46,
        0.035,
        absorption=(0.88, 0.98, 0.97),
        density=0.8,
    )
    porcelain, _ = material("Warm ceramic floor", (0.32, 0.33, 0.315), 0.44)
    base, _ = material("Satin graphite plinth", (0.032, 0.043, 0.05), 0.26, 0.45)
    ground, _ = material("Charcoal studio", (0.038, 0.047, 0.058), 0.34)
    trim, _ = material("Brushed silver", (0.32, 0.36, 0.39), 0.23, 0.85)
    box("Studio surface", (0, 0, -0.0095), (1.5, 1.5, 0.005), ground)
    box("Floating plinth", (0, 0, -0.0035), (0.162, 0.042, 0.007), base, 0.0018)
    floor = box("Tank floor", (0, 0, 0.00030), (0.15, 0.03, 0.0013), porcelain, 0.0003)
    if hasattr(floor.cycles, "is_caustics_receiver"):
        floor.cycles.is_caustics_receiver = True
    for side in (-1, 1):
        wall = box(
            f"Glass {'front' if side < 0 else 'back'}",
            (0, side * 0.01490, 0.026),
            (0.1513, 0.0015, 0.05),
            glass,
            0.00022,
        )
        wall.visible_shadow = False
        wall = box(
            f"Glass end {side}",
            (side * 0.07490, 0, 0.026),
            (0.0015, 0.0283, 0.05),
            glass,
            0.00022,
        )
        wall.visible_shadow = False
        box(
            f"Lower edge {side}",
            (0, side * 0.016, -0.00025),
            (0.154, 0.00065, 0.00065),
            trim,
            0.0002,
        )

    light(
        "Key softbox",
        (-0.045, -0.04, 0.18),
        (0, 0, 0.015),
        1.0,
        0.14,
        (1.0, 0.94, 0.86),
        0.08,
    )
    light(
        "Long rim strip",
        (0.015, 0.085, 0.105),
        (0, 0, 0.02),
        2.0,
        0.17,
        (0.77, 0.89, 1.0),
        0.016,
    )
    light(
        "Front fill",
        (0.07, -0.14, 0.07),
        (0.025, 0, 0.012),
        0.25,
        0.10,
        (0.92, 0.96, 1.0),
        0.07,
    )
    camera = bpy.data.cameras.new("Macro camera")
    obj = bpy.data.objects.new("Macro camera", camera)
    scene.collection.objects.link(obj)
    obj.location = (0.145, -0.215, 0.145)
    target = (0.002, 0, 0.020)
    aim(obj, target)
    camera.lens = 56
    camera.clip_start, camera.clip_end = 0.001, 10
    camera.dof.use_dof = True
    camera.dof.focus_distance = (obj.location - Vector(target)).length
    camera.dof.aperture_fstop = 16
    camera.dof.aperture_blades = 8
    scene.camera = obj
    scene["cache_directory"] = str(args.cache.resolve())
    scene["simulation_grid"] = "450 x 150 x 90"
    scene["description"] = (
        "Dambreak with four freely moving, melting ice bodies; no field color maps"
    )
    return water, ice


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("CPU", "OPTIX", "CUDA"), default="OPTIX")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--noise-threshold", type=float, default=0.015)
    p.add_argument("--min-samples", type=int, default=16)
    p.add_argument("--max-bounces", type=int, default=16)
    p.add_argument("--diffuse-bounces", type=int, default=3)
    p.add_argument("--glossy-bounces", type=int, default=8)
    p.add_argument("--transmission-bounces", type=int, default=12)
    p.add_argument("--volume-bounces", type=int, default=2)
    p.add_argument("--denoiser", choices=("OPENIMAGEDENOISE", "OPTIX"), default="OPENIMAGEDENOISE")
    p.add_argument("--denoise-device", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument(
        "--volume-integrator",
        choices=("ray_marching", "null_scattering"),
        default="ray_marching",
    )
    p.add_argument("--volume-step-rate", type=float, default=1.0)
    p.add_argument("--frames", type=int, nargs="+", default=[0])
    p.add_argument("--all", action="store_true")
    args = p.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    available = sorted(int(p.stem.removeprefix("frame_")) for p in args.cache.glob("frame_*.npz"))
    if not available or available != list(range(len(available))):
        raise ValueError("mesh cache must contain a contiguous sequence starting at zero")
    args.frame_count = len(available)
    frames = available if args.all else args.frames
    if any(index not in available for index in frames):
        raise ValueError("requested frame is absent from the mesh cache")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["scene_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    record_path = args.output / "render.json"
    record = (
        json.loads(record_path.read_text())
        if record_path.exists()
        else {
            "status": "initializing",
            "settings": settings,
            "frames": [],
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    if record["settings"] != settings:
        raise ValueError("render settings changed: use a new output directory")
    materials = build_scene(args)
    load_frame(args.cache, 0, materials)
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(
        filepath=str((args.output / "dambreak_four_ice.blend").resolve())
    )
    start = time.perf_counter()
    try:
        record["status"] = "rendering"
        for index in frames:
            path = args.output / "frames" / f"frame_{index:06d}.png"
            if any(f["index"] == index for f in record["frames"]) and path.exists():
                continue
            t0 = time.perf_counter()
            load_frame(args.cache, index, materials)
            bpy.context.scene.frame_set(index + 1)
            temporary = path.with_name(path.stem + ".tmp.png")
            bpy.context.scene.render.filepath = str(temporary.resolve())
            bpy.ops.render.render(write_still=True)
            temporary.replace(path)
            item = {
                "index": index,
                "time_s": bpy.context.scene["physical_time_s"],
                "path": str(path.resolve()),
                "wall_time_s": time.perf_counter() - t0,
            }
            record["frames"].append(item)
            record["total_frame_render_time_s"] = sum(
                f["wall_time_s"] for f in record["frames"]
            )
            write_json(record_path, record)
            print("FRAME_COMPLETED", json.dumps(item), flush=True)
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["last_session_wall_time_s"] = time.perf_counter() - start
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record_path, record)


if __name__ == "__main__":
    main()
