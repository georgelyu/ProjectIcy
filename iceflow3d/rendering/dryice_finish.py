"""Add scientific legends, encode 25 fps, and verify the delivered film."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dryice_export import write_json


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def decorate(raw, destination, kind, time_s, luts, total_s=6.0):
    with Image.open(raw) as image:
        im = image.convert("RGB")
    width, height = im.size
    scale = width / 1920
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)

    def text(x, y, message, size=24, color="#d2e2ed", bold=False):
        draw.text((round(x * scale), round(y * scale)), message,
                  font=ImageFont.truetype(BOLD if bold else FONT, round(size * scale)), fill=color)

    def rect(box, color):
        draw.rectangle(tuple(round(x * scale) for x in box), fill=color)

    rect((0, 0, 1920, 160), (11, 22, 34, 205))
    rect((58, 47, 64, 116), "#7fe2d1")
    text(88, 38, "DRY ICE / HOT WATER", 44, "#f4f8fb", True)
    text(90, 100, "216³  ·  CO₂ sublimation  ·  adiabatic open tank", 22)
    text(1560, 45, f"{time_s:05.2f} s", 48, "#f4f8fb", True)
    text(1566, 106, "physical time", 19)
    if kind == "beauty":
        rect((0, 970, 1920, 1080), (11, 22, 34, 218))
        text(64, 988, "Computed liquid–gas interface and solid CO₂ geometry", 23, "#f4f8fb")
        text(64, 1026, "0.01 s / sample   ·   25 fps   ·   4× slow motion", 21)
        text(1260, 989, "Gas: prescribed isothermal 20 °C", 23)
        text(1260, 1026, "Research model · display shell is transparent", 18, "#9dafbd")
    else:
        modes = [
            ("SPEED", "m/s", "viridis", "magma", "0", "≥ 0.5", "0", "≥ 1.5"),
            ("TEMPERATURE", "°C", "inferno", "Blues", "0", "60", "0", "60"),
            ("GAUGE PRESSURE", "Pa", "coolwarm", "PuOr", "≤ −250", "≥ 250", "≤ −250", "≥ 250"),
        ]
        rect((0, 810, 1920, 1080), (11, 22, 34, 236))
        for i, (name, unit, liquid, gas, low, high, glow, ghigh) in enumerate(modes):
            x = 116 + 574 * i
            text(x, 239, name, 29, "#f4f8fb", True)
            text(x, 283, unit + "  /  centre plane + 3D cutaway", 18)
            for y, cmap, phase, a, b in ((846, liquid, "WATER", low, high), (918, gas, "CO₂ GAS", glow, ghigh)):
                text(x, y - 22, phase, 17)
                bar = (np.asarray(luts[cmap])[:, :3] * 255 + .5).astype(np.uint8)[None]
                bar = Image.fromarray(bar).resize((round(430 * scale), round(13 * scale)))
                overlay.paste(bar, (round(x * scale), round(y * scale)))
                text(x, y + 18, a, 17)
                text(x + 347, y + 18, b, 17)
        text(116, 997, "Upper central z plane: 36.167 mm  ·  pressure relative to ambient at top opening", 20)
        text(116, 1030, "White: solid CO₂ · grey: wall · gas prescribed at 20 °C · fixed colour limits; raw data are not clipped.", 18, "#9dafbd")
    rect((0, 1075, 1920 * min(time_s / max(total_s, 1e-12), 1), 1080), "#7fe2d1")
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp.png")
    im.save(tmp, compress_level=3)
    tmp.replace(destination)


def video(frames, output, count, width, height):
    if not count:
        raise ValueError("cannot encode empty sequence")
    temporary = output.with_suffix(".tmp.mp4")
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "warning", "-framerate", "25", "-start_number", "0",
        "-i", str(frames / "frame_%06d.png"), "-frames:v", str(count),
        "-vf", "zscale=matrixin=gbr:transferin=iec61966-2-1:primariesin=709:rangein=full:matrix=709:transfer=709:primaries=709:range=limited,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-threads", "8",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-movflags", "+faststart", str(temporary),
    ], check=True)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration,codec_name",
        "-of", "json", str(temporary),
    ]))
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"]) == (width, height)
    assert stream["r_frame_rate"] == "25/1" and stream["avg_frame_rate"] == "25/1"
    assert int(stream["nb_frames"]) == count and int(stream["nb_read_frames"]) == count
    assert abs(float(stream["duration"]) - count / 25) < 1e-3
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(temporary), "-f", "null", "-"], check=True)
    temporary.replace(output)
    return probe


def finish(root, render, *, encode=True):
    render_record = json.loads((render / "render.json").read_text())
    luts = json.loads((root / "colormaps.json").read_text())
    result = dict(status="processing", frame_count={}, movies={},
                  scalar_limits=dict(liquid_speed_m_s=[0, .5], gas_speed_m_s=[0, 1.5],
                                     temperature_c=[0, 60], gauge_pressure_pa=[-250, 250]))
    for kind in ("beauty", "science"):
        items = sorted((f for f in render_record["frames"] if f["kind"] == kind), key=lambda f: f["index"])
        output = render / kind
        for f in items:
            target = output / "frames" / f"frame_{f['index']:06d}.png"
            decorate(Path(f["path"]), target, kind, f["time_s"], luts)
            with Image.open(target) as im:
                width, height = im.size
                im.verify()
        result["frame_count"][kind] = len(items)
        if not items:
            continue
        # Timed contact sheet, including early impact as well as the full run.
        selected = sorted(set([items[0]["index"], items[-1]["index"]] + [items[i]["index"] for i in np.linspace(0, len(items) - 1, 9, dtype=int)]))[:9]
        sheet = Image.new("RGB", (1920, 3 * 388), "#101c28")
        draw = ImageDraw.Draw(sheet)
        lookup = {f["index"]: f for f in items}
        for cell, index in enumerate(selected):
            with Image.open(output / "frames" / f"frame_{index:06d}.png") as im:
                preview = im.resize((640, 360), Image.Resampling.LANCZOS)
            x, y = (cell % 3) * 640, (cell // 3) * 388
            sheet.paste(preview, (x, y))
            draw.text((x + 14, y + 362), f"t = {lookup[index]['time_s']:.2f} s", fill="#edf5fa", font=ImageFont.truetype(FONT, 19))
        sheet.save(output / "contact_sheet.jpg", quality=95)
        if encode:
            if [f["index"] for f in items] != list(range(len(items))):
                raise ValueError("video requires contiguous frames starting from zero")
            output_video = output / f"dry_ice_{kind}_25fps.mp4"
            result["movies"][kind] = dict(path=str(output_video.resolve()),
                verification=video(output / "frames", output_video, len(items), width, height))
        result.setdefault("last_time_s", {})[kind] = items[-1]["time_s"]
    result["status"] = "completed" if encode else "previews_decorated"
    write_json(render / "delivery.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--render", type=Path, required=True)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    print(json.dumps(finish(args.root, args.render, encode=not args.no_video), indent=2))


if __name__ == "__main__":
    main()
