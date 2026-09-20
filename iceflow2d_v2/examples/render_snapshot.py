"""Render an experimental NPZ snapshot; no legacy visualization is imported."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .render_movie import ice_overlay
else:
    from render_movie import ice_overlay


def render(path, output):
    with np.load(path, allow_pickle=False) as snapshot:
        cfg = json.loads(str(snapshot['config']))
        q = snapshot['state'].astype(float)-snapshot['compensation']
        capacity = snapshot['capacity'].astype(float)
        velocity = snapshot['velocity'].astype(float)
        step = int(snapshot['step'])
    dx = cfg['width']/cfg['nx']
    dt = .1*dx/cfg['reference_velocity']
    phase = np.divide(q[..., 0], capacity, out=np.zeros_like(capacity), where=capacity > 1e-8)
    temperature = np.divide(q[..., 2]*cfg['temperature_scale'], q[..., 0],
                            out=np.full_like(capacity, np.nan), where=q[..., 0] > 1e-8)
    speed = np.linalg.norm(velocity, axis=-1)*dx/dt
    zero_speed = bool(np.all(speed == 0))
    speed_upper = float(speed.max()) if not zero_speed else 1.0
    extent = (0, cfg['width']*1000, 0, cfg['height']*1000)
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), constrained_layout=True)
    for ax, data, title, cmap, limits in zip(axes, (phase, temperature, speed),
             ('Liquid fraction', 'Water temperature (C), liquid > 1%', 'Speed (m/s)'),
             ('Blues', 'inferno', 'viridis'), ((0, 1), (0, cfg['water_temperature']), (0, speed_upper))):
        shown = np.ma.array(data, mask=(capacity < 1e-8) | ~np.isfinite(data))
        if data is temperature:
            shown = np.ma.masked_where(phase < .01, shown)
        artist = ax.imshow(shown.T, origin='lower', extent=extent, cmap=cmap,
                            vmin=limits[0], vmax=limits[1], interpolation='nearest')
        ax.set_facecolor('#f4f6f7')
        ax.imshow(ice_overlay(capacity), origin='lower', extent=extent, interpolation='nearest')
        ax.set_title(title); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        if data is speed and zero_speed:
            ax.set_title('Speed: 0 m/s')
        else:
            fig.colorbar(artist, ax=ax, shrink=.7)
    fig.suptitle(f'Recorded snapshot | step {step} | t={step*dt:.6f} s\n'
                 'Fixed-density air | ice/wall volume in gray', fontsize=12)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(render(args.snapshot, args.output or args.snapshot.with_suffix('.png')))


if __name__ == '__main__':
    main()
