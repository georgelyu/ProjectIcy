"""Stream temperature and water-velocity images directly from accepted states."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


def output_steps(start, stop, dt, interval):
    """Include both endpoints and round each physical-time target up separately."""
    yield start
    index = int(np.floor(start*dt/interval))+1
    current = start
    while current < stop:
        current = min(stop, max(current+1, int(np.ceil(index*interval/dt-1e-9))))
        yield current
        index = max(index+1, int(np.floor(current*dt/interval))+1)


def capture_fields(simulation):
    """Convert compensated cell inventories and lattice velocities to SI units."""
    cfg, geometry = simulation.cfg, simulation.geometry
    q = simulation.state.q.to_numpy().astype(float)-simulation.state.comp.to_numpy().astype(float)
    capacity = geometry.capacity.to_numpy()
    phase = np.divide(q[..., 0], capacity, out=np.zeros_like(capacity), where=capacity > 1e-8)
    visible = (capacity > 1e-8) & (phase >= .5)
    temperature = np.divide(q[..., 2]*cfg.temperature_scale, q[..., 0],
                            out=np.zeros_like(capacity), where=q[..., 0] > 1e-12)
    velocity = simulation.flow.velocity.to_numpy().astype(float)*cfg.dx/cfg.dt
    speed = np.linalg.norm(velocity, axis=-1)

    # Use the actual material polygons and pose, including partially melted cells.
    solid = geometry.solid.to_numpy().astype(float)-geometry.solid_comp.to_numpy().astype(float)
    heat = geometry.ice_heat.to_numpy().astype(float)-geometry.ice_heat_comp.to_numpy().astype(float)
    polygons, counts = geometry.poly.to_numpy(), geometry.count.to_numpy()
    rotation, origin = np.asarray(geometry.rotation[None]), np.asarray(geometry.origin[None])
    ice_polygons, ice_temperatures = [], []
    for a, b in np.argwhere((counts >= 3) & (solid > 1e-12)):
        local = polygons[a, b, :counts[a, b]]+np.array([a+.5-cfg.nb/2, b+.5-cfg.nb/2])
        ice_polygons.append((local@rotation.T+origin)*cfg.dx*1000)
        ice_temperatures.append(heat[a, b]*cfg.temperature_scale/(solid[a, b]*cfg.ice_capacity_lu))
    return (np.ma.array(temperature, mask=~visible), np.ma.array(speed, mask=~visible),
            velocity, ice_polygons, np.asarray(ice_temperatures))


class FieldImageWriter:
    """Reuse two figures and keep only the current frame in host memory."""

    def __init__(self, output, config, velocity_max=.1):
        self.output, self.cfg = Path(output), config
        self.frame_count, self.last_step = 0, None
        self.figures, self.axes, self.images, self.ice = [], [], [], []
        extent = (0, config.width*1000, 0, config.height*1000)
        blank = np.ma.masked_all((config.ny, config.nx))
        wall = np.ones((config.nx, config.ny), dtype=bool)
        b = config.wall_cells
        wall[b:-b, b:-b] = False
        walls = np.zeros((config.ny, config.nx, 4))
        walls[..., :3] = .65
        walls[..., 3] = wall.T
        temperature_min = min(0, config.ice_temperature)
        temperature_max = max(temperature_min+1, config.water_temperature)
        for name, label, cmap, limits in (
                ('temperature', 'Temperature (°C)', 'inferno', (temperature_min, temperature_max)),
                ('velocity', 'Water speed |u| (m/s)', 'viridis', (0, velocity_max))):
            directory = self.output/f'{name}_frames'
            directory.mkdir(parents=True, exist_ok=True)
            for stale in directory.glob('frame_*.png'):
                stale.unlink()
            figure, axis = plt.subplots(figsize=(6.4, 8), constrained_layout=True)
            artist = axis.imshow(blank, origin='lower', extent=extent, interpolation='nearest',
                                 cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set(xlabel='x (mm)', ylabel='y (mm)', xlim=extent[:2], ylim=extent[2:])
            axis.set_facecolor('#eef3f6')
            axis.imshow(walls, origin='lower', extent=extent, interpolation='nearest')
            if name == 'temperature':
                ice = PolyCollection([], cmap=artist.cmap, norm=artist.norm,
                                     edgecolors='none', antialiaseds=False)
            else:
                ice = PolyCollection([], facecolors='#b8c7cc', edgecolors='none', antialiaseds=False)
            axis.add_collection(ice)
            figure.colorbar(artist, ax=axis, label=label, extend='max' if name == 'velocity' else 'neither')
            self.figures.append(figure); self.axes.append(axis)
            self.images.append(artist); self.ice.append(ice)

        stride = max(1, int(np.ceil(config.nx/16)))
        self.sample = (slice(stride//2, None, stride), slice(stride//2, None, stride))
        x, y = np.meshgrid((np.arange(config.nx)[self.sample[0]]+.5)*config.dx*1000,
                           (np.arange(config.ny)[self.sample[1]]+.5)*config.dx*1000)
        self.quiver = self.axes[1].quiver(
            x, y, np.zeros_like(x), np.zeros_like(y), angles='xy', scale_units='xy',
            scale=velocity_max/(.75*stride*config.dx*1000), pivot='mid', color='white', width=.004)
        self.velocity_max = velocity_max
        self.axes[1].quiverkey(self.quiver, X=.90, Y=.035, U=velocity_max,
                               label=f'{velocity_max:g} m/s', labelpos='W', labelcolor='white')
        self.manifest = (self.output/'frames.csv').open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.manifest)
        self.writer.writerow(('frame', 'step', 'time_s', 'temperature', 'velocity'))
        self.manifest.flush()

    def append(self, simulation):
        if simulation.steps == self.last_step:
            return
        temperature, speed, velocity, polygons, ice_temperatures = capture_fields(simulation)
        self.images[0].set_data(temperature.T)
        self.images[1].set_data(speed.T)
        for ice in self.ice:
            ice.set_verts(polygons)
        self.ice[0].set_array(ice_temperatures)
        sampled_speed = speed[self.sample]
        hidden = np.ma.getmaskarray(sampled_speed) | (sampled_speed.data <= 0)
        # Cap arrow length at the displayed reference; colors retain the speed scale.
        factor = np.minimum(1, self.velocity_max/np.maximum(sampled_speed.data, 1e-30))
        arrows = velocity[self.sample]*factor[..., None]
        self.quiver.set_UVC(np.ma.array(arrows[..., 0], mask=hidden).T,
                            np.ma.array(arrows[..., 1], mask=hidden).T)
        time_s = simulation.steps*self.cfg.dt
        paths = []
        for name, title, figure, axis in zip(('temperature', 'velocity'),
                ('Water and ice temperature', 'Water velocity'), self.figures, self.axes):
            axis.set_title(f'{title} | t={time_s:.6f} s\nStep {simulation.steps}')
            path = Path(f'{name}_frames')/f'frame_{self.frame_count:05d}.png'
            figure.savefig(self.output/path, dpi=160)
            paths.append(str(path))
        self.writer.writerow((self.frame_count, simulation.steps, time_s, *paths))
        self.manifest.flush()
        self.frame_count += 1
        self.last_step = simulation.steps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.manifest.close()
        for figure in self.figures:
            plt.close(figure)
