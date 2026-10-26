"""Cycles film and three-field cutaway of the actual dry-ice mesh cache.

blender -b --python dryice_blender.py -- --cache ... --output ... --all
No synthetic bubbles, smoke, foam, or rigid-body animation are added.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_scene import aim, box, dielectric, light, material, mesh_object, write_json


def configure(scene, args, science=False):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if args.device != "CPU" else "CPU"
    scene.cycles.samples = args.science_samples if science else args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_min_samples = 16
    scene.cycles.adaptive_threshold = .02 if science else .012
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = args.device != "CPU"
    scene.cycles.max_bounces = 16
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.volume_bounces = 0
    scene.cycles.volume_biased = True
    scene.cycles.sample_clamp_indirect = 8
    scene.cycles.seed = 29
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 20
    scene.render.fps = 25
    scene.render.threads_mode, scene.render.threads = "FIXED", 8
    scene.view_settings.view_transform = "Standard" if science else "AgX"
    if not science:
        scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0 if science else -.35
    scene.unit_settings.system = "METRIC"
    world = bpy.data.worlds.new(scene.name + " ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.22, .29, .42, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .25 if science else .35
    scene.world = world


def camera(scene, location, target, scale):
    data = bpy.data.cameras.new(scene.name + " camera")
    obj = bpy.data.objects.new(scene.name + " camera", data)
    scene.collection.objects.link(obj)
    obj.location = location
    aim(obj, target)
    data.type, data.ortho_scale = "ORTHO", scale
    data.clip_start, data.clip_end = .0001, 5
    scene.camera = obj


def dry_ice_material():
    mat, bsdf = material("Solid CO2 · fine white crystalline surface", (.80, .89, .94), .29)
    bsdf.inputs["Subsurface Weight"].default_value = .07
    bsdf.inputs["Subsurface Radius"].default_value = (.00025, .0002, .00015)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6500
    noise.inputs["Detail"].default_value = 2
    tex = nodes.new("ShaderNodeTexCoord")
    links.new(tex.outputs["Object"], noise.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .13
    bump.inputs["Distance"].default_value = .000008
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def tank(prefix, glass, base, offset=0, science=False):
    # Display shell follows the 36 x 36 x 24 mm tank. A 0.08 mm outward
    # display offset avoids coplanar refraction; no physical geometry changes.
    depth, centre_y = (.018, .009) if science else (.036, 0)
    floor = box(prefix + " floor", (offset, centre_y, .00045), (.036, depth, .0009), base, .00018)
    for side in (-1, 1):
        obj = box(prefix + f" end {side}", (offset + side * .01758, centre_y, .012),
                  (.001, depth, .024), glass, .0001)
        obj.visible_shadow = False
        if science and side == -1:
            obj.hide_render = True
        obj = box(prefix + f" {'front' if side < 0 else 'back'}", (offset, side * .01758, .012),
                  (.034, .001, .024), glass, .0001)
        obj.visible_shadow = False
        if science and side < 0:
            obj.hide_render = True
    return floor


def attribute_material(name, attribute, emission=False, alpha=.85):
    mat, bsdf = material(name, (1, 1, 1), .36)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = attribute
    if emission:
        emit = nodes.new("ShaderNodeEmission")
        links.new(attr.outputs["Color"], emit.inputs["Color"])
        emit.inputs["Strength"].default_value = 1
        links.new(emit.outputs[0], nodes.get("Material Output").inputs["Surface"])
    else:
        links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = .32
        bsdf.inputs["Alpha"].default_value = alpha
    return mat


def build(args):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if args.device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = args.device
        prefs.get_devices()
        devices = []
        for device in prefs.devices:
            device.use = device.type == args.device
            if device.use:
                devices.append(device.name)
        if not devices:
            raise RuntimeError(f"No {args.device} device")
        print("RENDER_DEVICES", devices, flush=True)
    hero = bpy.context.scene
    hero.name = "Beauty"
    configure(hero, args)
    water = dielectric("Water · resolved liquid-gas interface", 1.333, .025,
                       absorption=(.63, .89, .96), density=9)
    glass = dielectric("Tank · optical display shell", 1.46, .025)
    ice = dry_ice_material()
    floor_mat, _ = material("Tank base · satin porcelain", (.32, .40, .43), .32)
    plinth, _ = material("Pedestal · graphite", (.025, .043, .057), .28, .45)
    ground, _ = material("Studio · midnight blue", (.018, .027, .043), .38)
    trim, _ = material("Metal trim", (.28, .45, .51), .23, .75)
    box("Studio", (0, 0, -.0058), (.6, .6, .004), ground)
    box("Pedestal", (0, 0, -.0018), (.0415, .0415, .0036), plinth, .0008)
    box("Pedestal accent", (0, 0, -.0004), (.040, .040, .00025), trim, .0002)
    tank("Beauty tank", glass, floor_mat)
    light("Beauty key", (-.045, -.045, .10), (0, 0, .012), .4, .07, (1, .92, .80), .05)
    light("Beauty rim", (.015, .052, .065), (0, 0, .014), .65, .055, (.70, .85, 1), .012)
    light("Beauty strip", (.055, -.025, .035), (0, 0, .012), .12, .04, (.78, .91, 1), .009)
    light("Beauty fill", (-.01, -.07, .02), (0, 0, .012), .045, .04, (1, 1, 1), .03)
    camera(hero, (.070, -.105, .070), (0, 0, .010), .120)
    science = bpy.data.scenes.new("Scientific cutaways")
    bpy.context.window.scene = science
    configure(science, args, science=True)
    science.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .25
    dark, _ = material("Scientific background", (.018, .027, .043), .7)
    box("Scientific stage", (0, 0, -.005), (.5, .2, .006), dark)
    surface_mat = attribute_material("Scientific interface", "scientific_color", alpha=.35)
    cut_mat = attribute_material("Scientific centre section", "scientific_color", emission=True)
    science_floor, _ = material("Cutaway base", (.045, .065, .085), .6)
    for offset, name in zip((-.052, 0, .052), ("Speed", "Temperature", "Pressure")):
        tank(name + " tank", glass, science_floor, offset=offset, science=True)
    light("Scientific key", (0, -.06, .16), (0, 0, .015), .35, .19, (1, 1, 1), .09)
    camera(science, (0, -.20, .095), (0, 0, .015), .174)
    bpy.context.window.scene = hero
    return hero, science, water, ice, surface_mat, cut_mat


def linear_rgb(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def colors(values, lo, hi, lut):
    position = np.clip((values - lo) / (hi - lo), 0, 1) * (len(lut) - 1)
    left = np.floor(position).astype(int)
    right = np.minimum(left + 1, len(lut) - 1)
    t = (position - left)[:, None]
    rgb = lut[left, :3] * (1 - t) + lut[right, :3] * t
    return np.column_stack((linear_rgb(rgb), np.ones(len(rgb)))).astype(np.float32)


def add_attributes(obj, data):
    for name, a in data.items():
        vector = a.ndim == 2
        attr = obj.data.attributes.new(name, "FLOAT_VECTOR" if vector else "FLOAT", "POINT")
        attr.data.foreach_set("vector" if vector else "value", np.asarray(a, np.float32).ravel())


def set_colors(obj, rgba):
    a = obj.data.color_attributes.new(name="scientific_color", type="FLOAT_COLOR", domain="POINT")
    a.data.foreach_set("color", np.asarray(rgba, np.float32).ravel())


def load(cache, index, scenes, luts):
    hero, science, water, ice, surface_mat, cut_mat = scenes
    with np.load(cache / f"frame_{index:06d}.npz", allow_pickle=False) as f:
        bpy.context.window.scene = hero
        obj = mesh_object("Resolved water", f["water_vertices"], f["water_faces"], water)
        add_attributes(obj, {"velocity_m_s": f["water_velocity_m_s"], "speed_m_s": f["water_speed_m_s"],
                             "temperature_c": f["water_temperature_c"], "pressure_pa": f["water_pressure_pa"]})
        for b in range(int(f["body_count"])):
            mesh_object(f"Resolved dry ice {b}", f[f"ice_{b}_vertices"], f[f"ice_{b}_faces"], ice)
        bpy.context.window.scene = science
        modes = [
            ("Speed", -.052, "speed_m_s", (0, .5), (0, 1.5), "viridis", "magma"),
            ("Temperature", 0, "temperature_c", (0, 60), (0, 60), "inferno", "Blues"),
            ("Pressure", .052, "pressure_pa", (-250, 250), (-250, 250), "coolwarm", "PuOr"),
        ]
        v, faces = f["water_vertices"], f["water_faces"]
        far_half = v[faces].mean(axis=1)[:, 1] > 0.0002
        for label, offset, key, bounds, gas_bounds, cmap, gas_cmap in modes:
            shift = np.array([offset, 0, 0])
            obj = mesh_object(label + " interface", v + shift, faces[far_half], surface_mat)
            set_colors(obj, colors(f["water_" + key], *bounds, luts[cmap]))
            obj = mesh_object(label + " z centre section", f["slice_vertices"] + shift, f["slice_faces"], cut_mat)
            values = f["slice_" + key]
            rgba = colors(values, *bounds, luts[cmap])
            gas = f["slice_water_phase"] < .5
            rgba[gas] = colors(values[gas], *gas_bounds, luts[gas_cmap])
            ids = f["slice_solid_body_id"]
            rgba[ids == -2, :3] = linear_rgb(np.array([.24, .31, .37]))
            rgba[ids >= 0, :3] = linear_rgb(np.array([.90, .96, .98]))
            set_colors(obj, rgba)
            for b in range(int(f["body_count"])):
                # Keep the real 3D body visible where it lies in front of the cut.
                mesh_object(label + f" dry ice {b}", f[f"ice_{b}_vertices"] + shift, f[f"ice_{b}_faces"], ice)
        for scene in (hero, science):
            scene.frame_set(index + 1)
            scene["physical_time_s"] = float(f["time_s"])
            scene["simulation_step"] = int(f["step"])
            scene["gas_thermal_model"] = "CO2 prescribed isothermal 20 C"
        bpy.context.window.scene = hero


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--luts", type=Path, required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--science-samples", type=int, default=48)
    p.add_argument("--device", choices=("OPTIX", "CUDA", "CPU"), default="OPTIX")
    p.add_argument("--frames", type=int, nargs="+", default=[0])
    p.add_argument("--all", action="store_true")
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    args.output.mkdir(parents=True, exist_ok=True)
    frames = sorted(int(p.stem.removeprefix("frame_")) for p in args.cache.glob("frame_*.npz")) if args.all else args.frames
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record_path = args.output / "render.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(status="initializing", settings=settings, frames=[], started_utc=datetime.now(timezone.utc).isoformat())
    if record["settings"] != settings:
        raise ValueError("render settings changed; choose a new output directory")
    luts = {k: np.array(v) for k, v in json.loads(args.luts.read_text()).items()}
    scenes = build(args)
    load(args.cache, frames[0], scenes, luts)
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / "dry_ice_scene.blend").resolve()))
    try:
        record["status"] = "rendering"
        for index in frames:
            load(args.cache, index, scenes, luts)
            for kind, scene in zip(("beauty", "science"), scenes[:2]):
                path = args.output / (kind + "_raw") / "frames" / f"frame_{index:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and any(f["index"] == index and f["kind"] == kind for f in record["frames"]):
                    continue
                t = time.perf_counter()
                scene.render.filepath = str(path.with_name(path.stem + ".tmp.png").resolve())
                bpy.context.window.scene = scene
                bpy.ops.render.render(write_still=True, scene=scene.name)
                Path(scene.render.filepath).replace(path)
                item = dict(index=index, kind=kind, time_s=scene["physical_time_s"],
                            path=str(path.resolve()), wall_time_s=time.perf_counter() - t)
                record["frames"].append(item)
                write_json(record_path, record)
                print("FRAME_COMPLETED", json.dumps(item), flush=True)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record_path, record)


if __name__ == "__main__":
    main()
