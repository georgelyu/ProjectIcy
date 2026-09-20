"""Independent flat/curved Couette benchmarks for the retained local SNIBB.

This isolates the imported v2 D2Q9 pressure/MRT/SNIBB formulas at uniform
viscosity. It is not a validation of the full cut-volume coupling, material
SDF proxy, melting, or the production spatially varying viscosity floor.
No legacy package is imported or executed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import taichi as ti

from iceflow2d_v2 import lattice
from iceflow2d_v2.lattice import (
    basis_moments, direction, momentum_collision, opposite,
    pressure_equilibrium, reconstruct, snibb,
)


@ti.data_oriented
class Couette:
    def __init__(self, n, curved, speed=.002):
        self.n, self.curved, self.speed = n, curved, speed
        self.nu, self.tau = .1, .8
        self.low, self.high = .173*n, .821*n
        self.ri, self.ro = .20*n, .42*n
        self.cx, self.cy = .501*n, .497*n
        self.a = -speed*self.ri/(self.ro**2-self.ri**2)
        self.b = speed*self.ri*self.ro**2/(self.ro**2-self.ri**2)
        self.f = ti.field(ti.f32, (9, n, n))
        self.post = ti.field(ti.f32, (9, n, n))
        self.u = ti.Vector.field(2, ti.f32, (n, n))
        self.p = ti.field(ti.f32, (n, n))
        self.initialize()
        if ti.__version__ != (1, 7, 4):
            raise RuntimeError('Class-kernel graph adapter requires Taichi 1.7.4.')
        builder = ti.graph.GraphBuilder()
        sequence = builder.create_sequential()
        for method in (self.collide, self.stream, self.recover):
            primal, owner = method._primal, method._kernel_owner
            specialization = primal.ensure_compiled(owner)
            sequence.seq_.dispatch(primal.compiled_kernels[specialization], [])
        builder.append(sequence)
        self.graph = builder.compile()

    @ti.func
    def distance(self, x, y):
        d = ti.min(y-self.low, self.high-y)
        if ti.static(self.curved):
            radius = ti.sqrt((x-self.cx)**2+(y-self.cy)**2)
            d = ti.min(radius-self.ri, self.ro-radius)
        return d

    @ti.func
    def exact(self, x, y):
        u = ti.Vector([self.speed*(y-self.low)/(self.high-self.low), 0.])
        p = 0.
        if ti.static(self.curved):
            relative = ti.Vector([x-self.cx, y-self.cy])
            radius = relative.norm()
            theta_speed = self.a*radius+self.b/radius
            u = theta_speed/radius*ti.Vector([-relative.y, relative.x])
            # Integrate dp/dr=u_theta^2/r, choosing p(ro)=0.
            p = (.5*self.a**2*(radius**2-self.ro**2)
                 +2*self.a*self.b*ti.log(radius/self.ro)
                 -.5*self.b**2*(1/radius**2-1/self.ro**2))
        return u, p

    @ti.func
    def wall_velocity(self, point):
        v = ti.Vector([0., 0.])
        if ti.static(self.curved):
            r = point-ti.Vector([self.cx, self.cy])
            if r.norm() < .5*(self.ri+self.ro):
                v = self.speed/self.ri*ti.Vector([-r.y, r.x])
        elif point.y > .5*(self.low+self.high):
            v.x = self.speed
        return v

    @ti.kernel
    def initialize(self):
        for i, j in self.u:
            if self.distance(i+.5, j+.5) > 0:
                u, p = self.exact(i+.5, j+.5)
                self.u[i, j], self.p[i, j] = u, p
                for q in range(9):
                    self.f[q, i, j] = pressure_equilibrium(q, p, 1., u)

    @ti.kernel
    def collide(self):
        for i, j in self.u:
            if self.distance(i+.5, j+.5) > 0:
                raw, eq = ti.Vector.zero(ti.f32, 9), ti.Vector.zero(ti.f32, 9)
                for q in range(9):
                    e = direction(q)
                    basis = basis_moments(e.x, e.y)
                    raw += basis*self.f[q, i, j]
                    eq += basis*pressure_equilibrium(q, self.p[i, j], 1., self.u[i, j])
                post = momentum_collision(raw, eq, ti.Vector.zero(ti.f32, 9), self.tau, self.tau)
                populations = ti.Vector.zero(ti.f32, 9)
                for q in range(9):
                    populations[q] = reconstruct(q, ti.Vector([0., 0.]), post)
                here = ti.Vector([i+.5, j+.5])
                d = self.distance(here.x, here.y)
                for q in range(9):
                    e = direction(q)
                    neighbor_d = self.distance(here.x+e.x, here.y+e.y)
                    value = populations[q]
                    if neighbor_d <= 0:
                        # Prepare before streaming, so f[q,i,j] is the actual
                        # pre-collision population, never a racing push write.
                        fraction = ti.min(.95, ti.max(.05, d/(d-neighbor_d)))
                        point = here+fraction*ti.cast(e, ti.f32)
                        value = snibb(q, fraction, populations, self.f[q, i, j],
                                      self.p[i, j], 1., self.u[i, j],
                                      self.wall_velocity(point))
                    self.post[q, i, j] = value

    @ti.kernel
    def stream(self):
        for i, j in self.u:
            here = ti.Vector([i+.5, j+.5])
            d = self.distance(here.x, here.y)
            if d > 0:
                for q in range(9):
                    e = direction(q)
                    ni, nj = (i+e.x+self.n) % self.n, j+e.y
                    neighbor_d = self.distance(here.x+e.x, here.y+e.y)
                    if neighbor_d > 0:
                        self.f[q, ni, nj] = self.post[q, i, j]
                    else:
                        # The production cut-link rule is distance ratio with
                        # the mandated [0.05, 0.95] clamp. Analytic signed
                        # distances here isolate that rule from rasterization.
                        self.f[opposite(q), i, j] = self.post[q, i, j]

    @ti.kernel
    def recover(self):
        for i, j in self.u:
            if self.distance(i+.5, j+.5) > 0:
                u, moving = ti.Vector([0., 0.]), 0.
                for q in range(9):
                    u += ti.cast(direction(q), ti.f32)*self.f[q, i, j]
                    if q > 0:
                        moving += self.f[q, i, j]
                self.u[i, j] = u
                self.p[i, j] = .6*(moving-(2/3)*u.norm_sqr())

    def oracle(self):
        x, y = np.indices((self.n, self.n), dtype=float)+.5
        exact = np.zeros((self.n, self.n, 2))
        if self.curved:
            x, y = x-self.cx, y-self.cy
            r = np.hypot(x, y)
            mask = (r > self.ri) & (r < self.ro)
            omega = self.a+self.b/r**2
            exact[..., 0], exact[..., 1] = -omega*y, omega*x
            gap = self.ro-self.ri
        else:
            mask = (y > self.low) & (y < self.high)
            exact[..., 0] = self.speed*(y-self.low)/(self.high-self.low)
            gap = self.high-self.low
        return mask, exact, gap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['flat', 'curved'], required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    sim = Couette(args.n, args.case == 'curved')
    mask, exact, gap = sim.oracle()
    steps = math.ceil(2*gap*gap/sim.nu)
    batch, history = 500, []
    previous = sim.u.to_numpy().astype(float)
    done = 0
    while done < steps:
        count = min(batch, steps-done)
        for _ in range(count):
            sim.graph.run({})
        done += count
        u = sim.u.to_numpy().astype(float)
        difference = u[mask]-exact[mask]
        relative_l2 = float(np.linalg.norm(difference)/np.linalg.norm(exact[mask]))
        increment = float(np.max(np.linalg.norm(u[mask]-previous[mask], axis=-1))/sim.speed)
        history.append({'step': done, 'relative_l2': relative_l2, 'max_increment_over_wall_speed': increment})
        print(json.dumps(history[-1]), flush=True)
        previous = u
        if not np.isfinite(u).all() or np.max(np.linalg.norm(u[mask], axis=-1)) > .1:
            raise RuntimeError('Couette benchmark diverged')
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'case': args.case, 'n': args.n, 'steps': done, 'tau': sim.tau,
        'wall_speed_lattice': sim.speed, 'viscous_times': done*sim.nu/gap**2,
        'relative_velocity_l2': relative_l2,
        'max_velocity_error_over_wall_speed': float(np.max(np.linalg.norm(difference, axis=-1))/sim.speed),
        'final_batch_increment_over_wall_speed': increment,
        'lattice_sha256': hashlib.sha256(Path(lattice.__file__).read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'history': history,
        'scope': 'Uniform-viscosity local MRT/SNIBB benchmark; analytic signed distances. Does not validate full cut-volume/phase coupling or the material SDF proxy.',
    }
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output/'velocity.npz', velocity=u, exact=exact, mask=mask)
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
