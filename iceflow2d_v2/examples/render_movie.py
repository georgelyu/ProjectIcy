"""Render recorded physical-time frames; never synthesize missing evolution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def read_frame(path):
    with np.load(path, allow_pickle=False) as data:
        cfg = json.loads(str(data['config']))
        q = data['state'].astype(float)-data['compensation'].astype(float)
        capacity = data['capacity'].astype(float)
        dt = .1*(cfg['width']/cfg['nx'])/cfg['reference_velocity']
        phase = np.divide(q[..., 0], capacity, out=np.zeros_like(capacity), where=capacity > 1e-8)
        temperature = np.divide(q[..., 2]*cfg['temperature_scale'], q[..., 0],
                                out=np.full_like(capacity, np.nan), where=q[..., 0] > 1e-12)
        return cfg, int(data['step'])*dt, np.ma.array(phase, mask=capacity < 1e-8), np.ma.array(
            temperature, mask=(capacity < 1e-8) | (phase < .01) | ~np.isfinite(temperature)), capacity


def ice_overlay(capacity):
    # Visualization of measured cell-average occupied volume, including walls.
    # Do not reconstruct a new contour from a thresholded solid mask.
    rgba = np.empty((*capacity.T.shape, 4))
    rgba[..., :3] = np.array([184, 199, 204])/255
    rgba[..., 3] = np.clip(1-capacity.T, 0, 1)
    return rgba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()
    paths = sorted(args.run.glob('frame_*.npz'))
    if not paths: raise ValueError('No accepted recorded frames found.')
    cfg, t, phase, temperature, capacity = read_frame(paths[0])
    time_digits = 2
    if len(paths) > 1:
        first_interval = read_frame(paths[1])[1]-t
        if first_interval <= 0:
            raise ValueError('Recorded frame times must increase.')
        time_digits = max(2, min(8, int(np.ceil(-np.log10(first_interval)))))
    extent = (0, 1e3*cfg['width'], 0, 1e3*cfg['height'])
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), constrained_layout=True)
    artists, overlays = [], []
    for ax, field, cmap, maximum, label in zip(axes, [phase, temperature], ['Blues', 'inferno'],
            [1, cfg['water_temperature']], ['Liquid fraction in fluid volume', 'Water temperature (C), fraction > 1%']):
        artist = ax.imshow(field.T, origin='lower', extent=extent, cmap=cmap,
                           vmin=0, vmax=maximum, interpolation='nearest')
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=label)
        ax.set_facecolor('#f4f6f7')
        overlays.append(ax.imshow(ice_overlay(capacity), origin='lower', extent=extent,
                                   interpolation='nearest'))
        fig.colorbar(artist, ax=ax, shrink=.8)
        artists.append(artist)
    title = fig.suptitle('')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=2200,
                         extra_args=['-pix_fmt', 'yuv420p'])
    frames = []
    with writer.saving(fig, str(args.output), dpi=140):
        for path in paths:
            _, t, phase, temperature, capacity = read_frame(path)
            artists[0].set_data(phase.T); artists[1].set_data(temperature.T)
            for overlay in overlays:
                overlay.set_data(ice_overlay(capacity))
            title.set_text(f'Falling and melting ice | physical time {t:.{time_digits}f} s\n'
                           f'{cfg["nx"]} x {cfg["ny"]} | fixed-density air | ice/wall volume in gray')
            writer.grab_frame()
            frames.append({'file': path.name, 'time_s': t,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.savefig(args.output.with_suffix('.png'), dpi=140)
    plt.close(fig)
    times = np.asarray([frame['time_s'] for frame in frames])
    intervals = np.diff(times)
    sample_interval = (float(intervals[0]) if len(intervals)
                       and np.allclose(intervals, intervals[0], rtol=0, atol=1e-12) else None)
    args.output.with_suffix('.json').write_text(json.dumps({
        'run': str(args.run.resolve()), 'fps': args.fps, 'frames': frames,
        'recorded_start_time_s': float(times[0]), 'recorded_end_time_s': float(times[-1]),
        'uniform_sample_interval_s': sample_interval,
        'physical_seconds_per_video_second': sample_interval*args.fps if sample_interval is not None else None,
        'note': 'Visualization of accepted recorded samples; does not certify physical convergence.'}, indent=2))
    print(args.output.resolve())


if __name__ == '__main__': main()
