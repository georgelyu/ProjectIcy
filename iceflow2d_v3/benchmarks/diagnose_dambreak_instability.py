"""Scalar-only controls and substep probes for the W=5 dam-break instability.

This file defines diagnostic subclasses; production defaults/kernels are not
changed. No grid arrays or restart files are written.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example
from iceflow2d_v3.lattice import (
    cross_product_2d,
    inside_grid,
    lattice_direction,
    lattice_weight,
    mixture_density,
    momentum_equilibrium,
    opposite_direction_index,
)
from iceflow2d_v3.body_state import BodyField
from iceflow2d_v3.simulator import ensure_taichi_cuda


def original_config():
    """Keep the historical controls reproducible after the example default changes."""
    return replace(example.create_config(), phase_gradient_scheme="legacy")


@ti.data_oriented
class FrozenPose(IceFlow2D):
    def _advance_body(self):
        self.hydrodynamic_impulse.fill(0.0)
        self.hydrodynamic_torque.fill(0.0)


@ti.data_oriented
class TranslationOnly(IceFlow2D):
    @ti.kernel
    def _cancel_rotation(self):
        for b in range(self.body_count):
            center = self.body_center[b]
            self.body_angle[b] = 0.0
            self.body_angular_velocity[b] = 0.0
            self.body_reference_origin[b] = center - ti.cast(
                self.body_local_center_of_mass[b], ti.f32
            )

    def _advance_body(self):
        self._integrate_rigid_ice()
        self._cancel_rotation()
        self._refresh_fluid_domain(resolve_contact=True)


@ti.data_oriented
class HalfwayLinks(IceFlow2D):
    @ti.func
    def _cut_link_fraction(self, i, j, neighbor_i, neighbor_j):
        return 0.5


PROBE_NAMES = (
    "phi",
    "phi_population_sum",
    "rho",
    "pressure_dynamic",
    "ux",
    "uy",
    "ax",
    "ay",
    "momentum_x",
    "momentum_y",
    "g_abs_sum",
    "h_abs_sum",
    "h_min",
    "base_ax",
    "base_ay",
    "flux_force_x",
    "flux_force_y",
    "grad_phi_x",
    "grad_phi_y",
    "fluid_neighbors",
    "cardinal_neighbors",
    "eta_min",
    "eta_max",
    "solid",
    "owner",
    "previous_owner",
    "coverage",
    "neighbor_phi_min",
    "neighbor_phi_max",
    "body_vx",
    "body_vy",
    "body_omega",
)
PROBE_SIZE = len(PROBE_NAMES)


@ti.data_oriented
class PointProbe(IceFlow2D):
    def __init__(self, config):
        super().__init__(config)
        self.probe_points_host = np.array(
            [
                (180, 3),
                (179, 3),
                (181, 3),
                (180, 4),
                (174, 3),
                (189, 3),
                (180, 2),
                (179, 2),
                (181, 2),
                (179, 4),
                (181, 4),
            ],
            dtype=np.int32,
        )
        self.probe_points = ti.Vector.field(
            2, ti.i32, shape=len(self.probe_points_host)
        )
        self.probe_values = ti.Vector.field(
            len(PROBE_NAMES), ti.f64, shape=len(self.probe_points_host)
        )
        self.probe_points.from_numpy(self.probe_points_host)

    @ti.kernel
    def sample_points(self):
        for k in self.probe_points:
            i, j = self.probe_points[k].x, self.probe_points[k].y
            v = ti.Vector.zero(ti.f64, PROBE_SIZE)
            v[0] = self.water_phase[i, j]
            v[2] = mixture_density(
                self.water_phase[i, j],
                self._water_density_lattice,
                self._air_density_lattice,
            )
            v[3] = (
                self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
            )
            physical = self._physical_velocity_at(i, j)
            v[4], v[5] = physical.x, physical.y
            v[6], v[7] = (
                self.fluid_acceleration_lattice[i, j].x,
                self.fluid_acceleration_lattice[i, j].y,
            )
            v[12] = 1e30
            for q in range(9):
                c = lattice_direction(q)
                g, h = (
                    self.momentum_populations[i, j, q],
                    self.phase_populations[i, j, q],
                )
                v[1] += h
                v[8] += c.x * g
                v[9] += c.y * g
                v[10] += ti.abs(g)
                v[11] += ti.abs(h)
                v[12] = ti.min(v[12], h)
            if self._is_fluid_cell(i, j):
                base = self._fluid_force_and_gradient(i, j)
                flux = self._phase_flux_force_density(i, j)
                v[13], v[14] = base[0], base[1]
                v[15], v[16] = flux.x, flux.y
                v[17], v[18] = base[2], base[3]
            v[21], v[22] = 1.0, 0.0
            v[27], v[28] = 1.0, 0.0
            body = ti.max(self.solid_body_id[i, j], 0)
            for q in range(1, 9):
                c = lattice_direction(q)
                ni, nj = i + c.x, j + c.y
                if inside_grid(ni, nj, self.nx, self.ny):
                    if self._is_fluid_cell(ni, nj):
                        v[19] += 1.0
                        if q <= 4:
                            v[20] += 1.0
                        v[27] = ti.min(v[27], self.water_phase[ni, nj])
                        v[28] = ti.max(v[28], self.water_phase[ni, nj])
                    elif self.solid_mask[ni, nj] != 0:
                        body = self.solid_body_id[ni, nj]
                        eta = self._cut_link_fraction(i, j, ni, nj)
                        v[21] = ti.min(v[21], eta)
                        v[22] = ti.max(v[22], eta)
            v[23] = self.solid_mask[i, j]
            v[24] = self.solid_body_id[i, j]
            v[25] = self.previous_body_id[i, j]
            v[26] = self._moving_body_fraction[i, j]
            v[29], v[30] = self.body_velocity[body].x, self.body_velocity[body].y
            v[31] = self.body_angular_velocity[body]
            self.probe_values[k] = v

    def point_rows(self):
        self.sample_points()
        return [
            dict(
                cell=[int(x) for x in cell],
                **dict(zip(PROBE_NAMES, row.tolist(), strict=True)),
            )
            for cell, row in zip(
                self.probe_points_host, self.probe_values.to_numpy(), strict=True
            )
        ]


@ti.data_oriented
class BoundaryExperiment(PointProbe):
    """Runtime switch for matched-checkpoint boundary tests (sigma=0 only)."""

    def __init__(self, config):
        assert config.sigma == 0 and len(config.bodies) > 1
        super().__init__(config)
        self.boundary_experiment_mode = ti.field(ti.i32, shape=())
        self.boundary_experiment_mode[None] = 0

    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        prepared = raw_post
        density = mixture_density(
            self.water_phase[i, j],
            self._water_density_lattice,
            self._air_density_lattice,
        )
        pressure = (
            self.pressure_lattice[i, j] - self.hydrostatic_reference_pressure[i, j]
        )
        velocity = (
            self.momentum_velocity_lattice[i, j]
            + 0.5 * self.fluid_acceleration_lattice[i, j]
        )
        for q in range(1, 9):
            direction = lattice_direction(q)
            ni, nj = i + direction.x, j + direction.y
            if inside_grid(ni, nj, self.nx, self.ny) and self.solid_mask[ni, nj] == 1:
                b = self.solid_body_id[ni, nj]
                eta = self._cut_link_fraction(i, j, ni, nj)
                c = ti.cast(direction, ti.f32)
                point = self._cell_point(i, j) + eta * c
                ub = self._body_velocity_at(point, b)
                opposite = opposite_direction_index(q)
                reflected = raw_post[q] - 6 * lattice_weight(q) * density * c.dot(ub)
                if self.boundary_experiment_mode[None] == 0:
                    neq = self.momentum_populations[i, j, q] - momentum_equilibrium(
                        q, pressure, density, velocity
                    )
                    reflected = (
                        momentum_equilibrium(opposite, pressure, density, ub)
                        + neq
                        + eta * raw_post[opposite]
                    ) / (1 + eta)
                prepared[q] = reflected
                impulse = (c - ub) * raw_post[q] - (-c - ub) * reflected
                pref = (1 - eta) * self.hydrostatic_reference_pressure[
                    i, j
                ] + eta * self.hydrostatic_reference_pressure[ni, nj]
                impulse += 6 * lattice_weight(q) * pref * c
                impulse64 = ti.cast(impulse, ti.f64)
                arm = ti.cast(point - self.body_center[b], ti.f64)
                ti.atomic_add(self.hydrodynamic_impulse[b], impulse64)
                ti.atomic_add(
                    self.hydrodynamic_torque[b], cross_product_2d(arm, impulse64)
                )
        return prepared


@ti.data_oriented
class GhostExperiment(PointProbe):
    def __init__(self, config):
        ensure_taichi_cuda()
        self.ghost_experiment_mode = ti.field(ti.i32, shape=())
        self.ghost_experiment_mode[None] = 0
        super().__init__(config)
        self.stencil_probe = ti.Vector.field(9, ti.f64, shape=8)

    @ti.func
    def _locally_connected(self, fi, fj, si, sj):
        dx, dy = si - fi, sj - fj
        ax, ay = ti.abs(dx), ti.abs(dy)
        connected = False
        if ax <= 2 and ay <= 2:
            for pattern in range(16):
                x, y = fi, fj
                cx, cy = 0, 0
                valid = True
                for step in range(4):
                    if step < ax + ay:
                        if (pattern >> step) & 1:
                            x += 1 if dx > 0 else -1
                            cx += 1
                        else:
                            y += 1 if dy > 0 else -1
                            cy += 1
                        if (
                            cx > ax
                            or cy > ay
                            or not inside_grid(x, y, self.nx, self.ny)
                        ):
                            valid = False
                        elif not self._is_fluid_cell(x, y):
                            valid = False
                if valid and x == si and y == sj:
                    connected = True
        return connected

    @ti.func
    def _ice_phase_ghost(self, fi, fj, si, sj, sample_i, sample_j):
        value = self.water_phase[fi, fj]
        if self.ghost_experiment_mode[None] != 1:
            mirror = self._ice_mirror_sample(fi, fj, si, sj, sample_i, sample_j)
            base = ti.cast(ti.floor(ti.Vector([mirror.x, mirror.y])), ti.i32)
            center = self.water_phase[fi, fj]
            difference, weight = 0.0, 0.0
            for di, dj in ti.static(ti.ndrange(2, 2)):
                i, j = base.x + di, base.y + dj
                w = ti.max(0.0, 1.0 - ti.abs(mirror.x - i)) * ti.max(
                    0.0, 1.0 - ti.abs(mirror.y - j)
                )
                index = self._phase_sample_inside_walls(i, j)
                if inside_grid(
                    index.x, index.y, self.nx, self.ny
                ) and self._is_fluid_cell(index.x, index.y):
                    accept = True
                    if self.ghost_experiment_mode[None] == 2:
                        accept = self._locally_connected(fi, fj, index.x, index.y)
                    if accept:
                        difference += w * (self.water_phase[index.x, index.y] - center)
                        weight += w
            if weight > 1e-10:
                value += difference / weight
        return value

    @ti.kernel
    def sample_stencil(self):
        for k in range(8):
            q = k + 1
            c = lattice_direction(q)
            first = self._phase_sample_inside_walls(180 + c.x, 3 + c.y)
            v = ti.Vector.zero(ti.f64, 9)
            v[0], v[1] = c.x, c.y
            v[2] = self._phase_neighbor(180, 3, c.x, c.y, 1)
            v[3], v[4] = first.x, first.y
            v[5] = self.solid_body_id[first.x, first.y]
            if self.solid_mask[first.x, first.y] != 0:
                mirror = self._ice_mirror_sample(
                    180, 3, first.x, first.y, first.x, first.y
                )
                v[6], v[7], v[8] = mirror.x, mirror.y, 1.0
            self.stencil_probe[k] = v


@ti.data_oriented
class OpenFaceGradient(PointProbe):
    @ti.func
    def _phase_central_gradient(self, i, j):
        gradient = ti.Vector.zero(ti.f32, 2)
        phi = self.water_phase[i, j]
        for q in range(1, 9):
            c = lattice_direction(q)
            neighbor = self._phase_sample_inside_walls(i + c.x, j + c.y)
            if inside_grid(
                neighbor.x, neighbor.y, self.nx, self.ny
            ) and self._is_fluid_cell(neighbor.x, neighbor.y):
                gradient += (
                    3
                    * lattice_weight(q)
                    * ti.cast(c, ti.f32)
                    * (self.water_phase[neighbor.x, neighbor.y] - phi)
                )
        return gradient


@ti.data_oriented
class LimitedGradient(PointProbe):
    @ti.func
    def _phase_central_gradient(self, i, j):
        gradient = ti.Vector.zero(ti.f32, 2)
        phi = self.water_phase[i, j]
        for axis in ti.static(range(2)):
            before, after = phi, phi
            for sign in ti.static(range(2)):
                ni, nj = i, j
                if ti.static(axis == 0):
                    ni += 2 * sign - 1
                else:
                    nj += 2 * sign - 1
                if inside_grid(ni, nj, self.nx, self.ny) and self._is_fluid_cell(
                    ni, nj
                ):
                    if ti.static(sign == 0):
                        before = self.water_phase[ni, nj]
                    else:
                        after = self.water_phase[ni, nj]
            dl, dr = phi - before, after - phi
            slope = 0.0
            if dl * dr > 0:
                slope = (1.0 if dl > 0 else -1.0) * ti.min(
                    ti.abs(0.5 * (dl + dr)), ti.min(ti.abs(2 * dl), ti.abs(2 * dr))
                )
            gradient[axis] = slope
        return gradient


def device_snapshot(s):
    seen, result = set(), []
    for prefix, owner in (("flow", s), ("thermal", s.thermal)):
        for name, value in vars(owner).items():
            field = value.field if isinstance(value, BodyField) else value
            if isinstance(field, (ti.Field, ti.MatrixField)) and id(field) not in seen:
                seen.add(id(field))
                result.append((prefix, name, field.to_numpy().copy()))
    return result


def opening_operator_control(out):
    source = PointProbe(original_config())
    source._thermal_stability_check_enabled = True
    source.step(8544)
    snapshot = device_snapshot(source)  # RAM only; never serialize grids.
    s = BoundaryExperiment(source.config)
    results = []
    for mode in (0, 1):
        for prefix, name, values in snapshot:
            owner = s if prefix == "flow" else s.thermal
            field = getattr(owner, name)
            if isinstance(field, BodyField):
                field = field.field
            field.from_numpy(values)
        s.steps = 8544
        s.boundary_experiment_mode[None] = mode
        s.body_velocity.fill(0.0)
        s.body_angular_velocity.fill(0.0)
        rows = []
        for k in range(240):
            s._advance_fluid()
            s.steps += 1
            state = state_summary(s)
            point = s.point_rows()[0]
            rows.append(
                dict(
                    step=s.steps,
                    max_speed_l1=state["speed_l1_max"],
                    point_speed_l1=abs(point["ux"]) + abs(point["uy"]),
                    point_pressure=point["pressure_dynamic"],
                    point_phi=point["phi"],
                )
            )
            if state["speed_l1_max"] > 16 or not np.isfinite(state["speed_l1_max"]):
                break
        result = dict(
            boundary="current" if mode == 0 else "postcollision-bounceback",
            frozen_geometry=True,
            stationary_walls=True,
            start_step=8544,
            steps_run=len(rows),
            last=rows[-1],
            history=rows,
        )
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True
        )
    (out / "matched_opening_operator.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def state_summary(s):
    u = s.physical_velocity_lattice.to_numpy().astype(float)
    phi = s.water_phase.to_numpy().astype(float)
    active = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
    speed = np.abs(u).sum(axis=-1)
    ranked = np.where(active, speed, -np.inf)
    cell = np.unravel_index(int(np.argmax(ranked)), ranked.shape)
    air, water = active & (phi < 0.5), active & (phi >= 0.5)
    return dict(
        step=s.steps,
        time_s=s.physical_time_s,
        speed_l1_max=float(ranked[cell]),
        peak_cell=[int(x) for x in cell],
        peak_phi=float(phi[cell]),
        air_max=float(speed[air].max()) if air.any() else 0.0,
        water_max=float(speed[water].max()) if water.any() else 0.0,
        centers=s.body_center.to_numpy().tolist(),
        angles=s.body_angle.to_numpy().tolist(),
        ice_mass_sum=float(s.body_mass_lattice.array().sum()),
    )


def run_control(mode, end_s, out):
    c = original_config()
    cls = IceFlow2D
    if mode == "frozen-pose":
        cls = FrozenPose
    elif mode == "no-rotation":
        cls = TranslationOnly
    elif mode == "halfway-links":
        cls = HalfwayLinks
    elif mode == "no-erosion":
        c = replace(
            c,
            thermal=replace(
                c.thermal,
                properties=replace(c.thermal.properties, latent_heat_j_kg=1e30),
            ),
        )
    elif mode == "legacy":
        c = replace(c, phase_momentum_coupling="legacy")
    elif mode == "non-equilibrium":
        c = replace(c, exposed_node_refill="non_equilibrium")
    elif mode == "postcollision-bounceback":
        cls = BoundaryExperiment
    elif mode == "acoustic-1":
        c = replace(c, cm_acoustic_relaxation_rate=1.0)
    elif mode == "rm-mrt":
        c = replace(
            c, momentum_collision_model="rm_mrt", cm_acoustic_relaxation_rate=1.0
        )
    elif mode in ("local-ghost", "connected-ghost"):
        cls = GhostExperiment
    elif mode == "open-face-gradient":
        cls = OpenFaceGradient
    elif mode == "limited-gradient":
        cls = LimitedGradient
    elif mode in ("hrr", "limited-hrr", "connected-hrr"):
        c = replace(c, momentum_collision_model="cm_hrr")
        if mode == "limited-hrr":
            cls = LimitedGradient
        elif mode == "connected-hrr":
            cls = GhostExperiment
    started = time.monotonic()
    s = cls(c)
    if mode == "postcollision-bounceback":
        s.boundary_experiment_mode[None] = 1
    if mode in ("local-ghost", "connected-ghost", "connected-hrr"):
        s.ghost_experiment_mode[None] = 1 if mode == "local-ghost" else 2
    s._thermal_stability_check_enabled = True
    target = round(end_s / s.scales.dt_s)
    rows = [state_summary(s)]
    error = None
    try:
        while s.steps < target:
            s.step(min(240, target - s.steps))
            rows.append(state_summary(s))
    except RuntimeError as exc:
        error = str(exc)
        rows.append(state_summary(s))
    result = dict(
        mode=mode,
        completed=s.steps == target,
        requested_s=end_s,
        elapsed_s=time.monotonic() - started,
        error=error,
        last=rows[-1],
        peak_speed_l1=max(r["speed_l1_max"] for r in rows),
        configuration=c.to_dict(),
    )
    (out / f"{mode}_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / f"{mode}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("mode", "completed", "elapsed_s", "last", "error")}
        ),
        flush=True,
    )


def run_trace(start, stop, out, mode="trace", probe_cell=(180, 3)):
    cls = GhostExperiment if mode in ("trace-local", "trace-connected") else PointProbe
    if mode == "trace-limited":
        cls = LimitedGradient
    s = cls(original_config())
    if mode in ("trace-local", "trace-connected"):
        s.ghost_experiment_mode[None] = 1 if mode == "trace-local" else 2
    x, y = probe_cell
    s.probe_points_host = np.array(
        [
            (x, y),
            (x - 1, y),
            (x + 1, y),
            (x, y + 1),
            (x - 6, y),
            (x + 9, y),
            (x, y - 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ],
        dtype=np.int32,
    )
    s.probe_points.from_numpy(s.probe_points_host)
    s._thermal_stability_check_enabled = True
    s.step(start)
    records = []
    error = None
    try:
        while s.steps < stop:
            step = s.steps + 1
            for stage, operation in [
                ("before_collision", lambda: None),
                ("after_collision", s._collide_momentum),
                ("after_stream", s._stream),
                ("after_macros", s._update_streamed_macroscopic_fields),
                ("after_pressure", s._update_pressure),
                ("after_phase_bound", s._bound_water_phase),
                ("after_body", s._advance_body),
                ("after_thermal", s._advance_thermal),
            ]:
                operation()
                records.append(dict(step=step, stage=stage, points=s.point_rows()))
            s.steps += 1
    except RuntimeError as exc:
        error = str(exc)
        records.append(dict(step=s.steps + 1, stage="failure", points=s.point_rows()))
    (out / "substep_point_metrics.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    summary = dict(
        start_step=start, last=state_summary(s), error=error, records=len(records)
    )
    (out / "substep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def linear_pressure_control(out):
    source = PointProbe(original_config())
    source._thermal_stability_check_enabled = True
    source.step(8544)
    snapshot = device_snapshot(source)
    results = []
    for model, acoustic in (("cm_mrt", 0.25), ("cm_mrt", 1.0), ("rm_mrt", 1.0)):
        s = BoundaryExperiment(
            replace(
                source.config,
                momentum_collision_model=model,
                cm_acoustic_relaxation_rate=acoustic,
            )
        )
        for boundary in (0, 1):
            for prefix, name, values in snapshot:
                owner = s if prefix == "flow" else s.thermal
                field = getattr(owner, name)
                if isinstance(field, BodyField):
                    field = field.field
                field.from_numpy(values)
            s.steps = 0
            s.boundary_experiment_mode[None] = boundary
            s.body_velocity.fill(0)
            s.body_angular_velocity.fill(0)
            s.water_phase.fill(0)
            s.phase_populations.fill(0)
            s.momentum_populations.fill(0)
            s.momentum_velocity_lattice.fill(0)
            s.fluid_acceleration_lattice.fill(0)
            reference = s.hydrostatic_reference_pressure.to_numpy()
            s.pressure_lattice.from_numpy(reference)
            seed = 1e-9
            weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
            populations = np.zeros((s.nx, s.ny, 9), np.float32)
            populations[180, 3] = 3 * weights * seed
            populations[180, 3, 0] -= 3 * seed
            s.momentum_populations.from_numpy(populations)
            s.pressure_lattice[180, 3] = float(reference[180, 3]) + seed
            rows = []
            for k in range(120):
                s._advance_fluid()
                s.steps += 1
                u = s.physical_velocity_lattice.to_numpy()
                pressure = s.pressure_lattice.to_numpy() - reference
                active = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
                peak = float(np.abs(u[active]).sum(-1).max())
                ppeak = float(np.abs(pressure[active]).max())
                rows.append(
                    dict(
                        step=k + 1,
                        pressure_growth=ppeak / seed,
                        speed_l1=peak,
                        point_pressure=float(pressure[180, 3]),
                    )
                )
                if peak > 0.01 or not np.isfinite(peak):
                    break
            result = dict(
                model=model,
                acoustic_rate=acoustic,
                boundary="current" if boundary == 0 else "postcollision-bounceback",
                seed_pressure=seed,
                steps_run=len(rows),
                last=rows[-1],
                history=rows,
            )
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
    (out / "linear_pressure_controls.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def ghost_operator_control(out):
    source = PointProbe(original_config())
    source._thermal_stability_check_enabled = True
    source.step(8544)
    snapshot = device_snapshot(source)
    s = GhostExperiment(source.config)
    results = []
    for mode in (0, 1, 2):
        for prefix, name, values in snapshot:
            owner = s if prefix == "flow" else s.thermal
            field = getattr(owner, name)
            if isinstance(field, BodyField):
                field = field.field
            field.from_numpy(values)
        s.steps = 8544
        s.ghost_experiment_mode[None] = mode
        s.body_velocity.fill(0)
        s.body_angular_velocity.fill(0)
        initial_point = s.point_rows()[0]
        if mode == 0:
            s.sample_stencil()
            stencils = s.stencil_probe.to_numpy()
            phi = s.water_phase.to_numpy()
            solid = s.solid_mask.to_numpy()
            detail = []
            for stencil in stencils:
                item = dict(
                    direction=stencil[:2].tolist(),
                    ghost_phi=float(stencil[2]),
                    first_cell=stencil[3:5].astype(int).tolist(),
                    owner=int(stencil[5]),
                )
                if stencil[8]:
                    mx, my = stencil[6:8]
                    base = np.floor([mx, my]).astype(int)
                    samples = []
                    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
                        i, j = base + [di, dj]
                        w = max(0, 1 - abs(mx - i)) * max(0, 1 - abs(my - j))
                        if j < 3:
                            j = 5 - j
                        if i < 3:
                            i = 5 - i
                        samples.append(
                            dict(
                                cell=[int(i), int(j)],
                                weight=w,
                                phi=float(phi[i, j]),
                                solid=int(solid[i, j]),
                            )
                        )
                    item.update(mirror=[float(mx), float(my)], samples=samples)
                detail.append(item)
            (out / "ghost_stencil_detail.json").write_text(
                json.dumps(detail, indent=2) + "\n"
            )
        rows = []
        for k in range(240):
            s._advance_fluid()
            s.steps += 1
            state = state_summary(s)
            point = s.point_rows()[0]
            rows.append(
                dict(
                    step=s.steps,
                    max_speed_l1=state["speed_l1_max"],
                    point_speed_l1=abs(point["ux"]) + abs(point["uy"]),
                    point_pressure=point["pressure_dynamic"],
                    grad_phi_x=point["grad_phi_x"],
                    grad_phi_y=point["grad_phi_y"],
                )
            )
            if state["speed_l1_max"] > 16 or not np.isfinite(state["speed_l1_max"]):
                break
        result = dict(
            ghost_mode=mode,
            initial_point=initial_point,
            steps_run=len(rows),
            last=rows[-1],
            history=rows,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("history", "initial_point")
                }
            ),
            flush=True,
        )
    (out / "matched_ghost_operator.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def translation_control(out):
    """Uniform periodic translation, without solids or field serialization."""
    from iceflow2d_v3.benchmarks.check_interface_translation import TranslationProbe
    from iceflow2d_v3.test_iceflow import _falling_config

    results = []
    for width, ratio, coupling in (
        (5.0, 800.0, "consistent_flux"),
        (12.0, 800.0, "consistent_flux"),
        (5.0, 1.0, "consistent_flux"),
    ):
        config = _falling_config(
            resolution=(32, 128),
            dx=1 / 3000,
            reference_velocity=4.0,
            interface_width=width,
            phase_warmup_steps=0,
            rho_air=1000.0 / ratio,
            sigma=0.0,
            gravity=(0.0, 0.0),
            viscosity_water=3.141732579601075e-7,
            viscosity_air=1.4506451972115707e-5,
            phase_momentum_coupling=coupling,
        )
        s = TranslationProbe(config)
        y = np.arange(s.ny) + 0.5
        phi = (
            np.broadcast_to(
                0.5 * (np.tanh(2 * (y - 32) / width) - np.tanh(2 * (y - 96) / width)),
                (s.nx, s.ny),
            )
            .astype(np.float32)
            .copy()
        )
        c = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )
        w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
        u0 = np.array([0.0, 0.01], dtype=np.float32)
        cu = c @ u0
        poly = 3 * cu + 4.5 * cu**2 - 1.5 * (u0 @ u0)
        s.water_phase.from_numpy(phi)
        s.momentum_velocity_lattice.from_numpy(
            np.broadcast_to(u0, (s.nx, s.ny, 2)).copy()
        )
        s.fluid_acceleration_lattice.fill(0)
        s.pressure_lattice.fill(0)
        s.reference_pressure_by_row.fill(0)
        s.phase_populations.from_numpy(
            (phi[..., None] * w * (1 + poly)).astype(np.float32)
        )
        for _ in range(2000):
            s._collide_phase()
            s.stream_phase()
            s._update_streamed_phase()
            s._bound_water_phase()
        phi = s.water_phase.to_numpy()
        density = 1 / ratio + (1 - 1 / ratio) * phi
        s.momentum_populations.from_numpy(
            (density[..., None] * w * poly).astype(np.float32)
        )
        s._update_pressure()
        initial_amount = float(phi.astype(float).sum())
        rows = []
        for step in range(12001):
            if step % 240 == 0:
                # The generic host view masks container walls even though this
                # periodic fixture overrides every device cell as fluid.
                u = (
                    s.momentum_velocity_lattice.to_numpy()
                    + 0.5 * s.fluid_acceleration_lattice.to_numpy()
                )
                ph = s.water_phase.to_numpy()
                rows.append(
                    dict(
                        step=step,
                        max_speed=float(np.linalg.norm(u, axis=-1).max()),
                        max_velocity_error=float(np.linalg.norm(u - u0, axis=-1).max()),
                        phase_amount=float(ph.astype(float).sum()),
                        phi_min=float(ph.min()),
                        phi_max=float(ph.max()),
                    )
                )
                if not np.isfinite(u).all() or rows[-1]["max_speed"] > 1:
                    break
            if step < 12000:
                s._advance_fluid()
        result = dict(
            interface_width=width,
            density_ratio=ratio,
            coupling=coupling,
            completed=rows[-1]["step"] == 12000,
            initial_velocity_lattice=u0.tolist(),
            initial_phase_amount=initial_amount,
            last=rows[-1],
            history=rows,
        )
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True
        )
    (out / "uniform_translation_controls.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "frozen-pose",
            "no-erosion",
            "no-rotation",
            "legacy",
            "halfway-links",
            "non-equilibrium",
        ],
    )
    p.add_argument("--end-time-s", type=float, default=0.2)
    p.add_argument("--trace-start", type=int, default=8400)
    p.add_argument("--trace-stop", type=int, default=8560)
    p.add_argument("--probe-cell", nargs=2, type=int, default=[180, 3])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        if mode in ("trace", "trace-local", "trace-connected", "trace-limited"):
            run_trace(
                args.trace_start,
                args.trace_stop,
                args.output_dir,
                mode,
                args.probe_cell,
            )
        elif mode == "opening-operator":
            opening_operator_control(args.output_dir)
        elif mode == "linear-pressure":
            linear_pressure_control(args.output_dir)
        elif mode == "ghost-operator":
            ghost_operator_control(args.output_dir)
        elif mode == "translation":
            translation_control(args.output_dir)
        else:
            run_control(mode, args.end_time_s, args.output_dir)


if __name__ == "__main__":
    main()
