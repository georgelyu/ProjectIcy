"""Four-body W=5 instability reproduction and controlled checkpoint restarts.

Diagnostic ablations deliberately freeze parts of the model; they are not fixes.
Production sources are hashed and left unchanged. Arrays use (x, y[, q]).
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.body_state import BodyField
from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config, parse_args
from iceflow2d_v3.lattice import lattice_direction, mixture_density
from iceflow2d_v3.simulator import IceFlow2D, _MAX_D2Q9_LATTICE_VELOCITY_L1


def storage(s):
    seen = set()
    owners = [("flow", s), ("thermal", s.thermal)]
    for name, value in vars(s).items():
        if value.__class__.__name__ == "BodyContacts":
            owners.append(("contacts", value))
    for prefix, owner in owners:
        for name, value in vars(owner).items():
            field = value.field if isinstance(value, BodyField) else value
            if isinstance(field, (ti.Field, ti.MatrixField)) and id(field) not in seen:
                seen.add(id(field))
                yield prefix + "__" + name, field


def checkpoint(s, folder):
    folder.mkdir()
    np.savez_compressed(
        folder / "state.npz", **{name: field.to_numpy() for name, field in storage(s)}
    )
    host = dict(
        flow={k: getattr(s, k) for k in ("steps", "_thermal_lbm_steps_pending")},
        thermal={k: getattr(s.thermal, k) for k in ("steps", "time_s")},
        config=s.config.to_dict(),
        format_version=2,
    )
    (folder / "host.json").write_text(json.dumps(host, indent=2) + "\n")


def restore(s, folder):
    host = json.loads((folder / "host.json").read_text())
    assert host["format_version"] == 2
    current = json.loads(json.dumps(s.config.to_dict()))
    for key in ("resolution", "dx", "reference_velocity"):
        assert host["config"][key] == current[key]
    with np.load(folder / "state.npz") as state:
        for name, field in storage(s):
            field.from_numpy(state[name])
        for name, field in storage(s):
            assert np.array_equal(field.to_numpy(), state[name], equal_nan=True), name
    for prefix, owner in (("flow", s), ("thermal", s.thermal)):
        for name, value in host[prefix].items():
            setattr(owner, name, value)


@ti.kernel
def recover_fixed_phase(s: ti.template()):
    for i, j in s.water_phase:
        if s._is_fluid_cell(i, j):
            momentum = ti.Vector.zero(ti.f32, 2)
            for q in range(9):
                momentum += (
                    ti.cast(lattice_direction(q), ti.f32)
                    * s.momentum_populations[i, j, q]
                )
            rho = mixture_density(
                s.water_phase[i, j], s._water_density_lattice, s._air_density_lattice
            )
            s.momentum_velocity_lattice[i, j] = momentum / rho
        else:
            velocity = ti.Vector.zero(ti.f32, 2)
            if s.solid_mask[i, j] != 0:
                velocity = s._body_velocity_at(
                    s._cell_point(i, j), ti.max(s.solid_body_id[i, j], 0)
                )
            s.momentum_velocity_lattice[i, j] = velocity
            s.fluid_acceleration_lattice[i, j] = ti.Vector.zero(ti.f32, 2)


@ti.kernel
def sample_terms(s: ti.template(), out: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i, j in s.water_phase:
        if s._is_fluid_cell(i, j):
            data = s._fluid_force_and_gradient(i, j)
            out[i, j, 0] = data[0]
            out[i, j, 1] = data[1]
            out[i, j, 2] = data[2]
            out[i, j, 3] = data[3]
            out[i, j, 4] = 0.5 + 3.0 * s._fluid_kinematic_viscosity_lattice(i, j)


def fields(s):
    terms = np.zeros((s.nx, s.ny, 5), dtype=np.float32)
    sample_terms(s, terms)
    return dict(
        phi=s.water_phase.to_numpy(),
        u=s.physical_velocity_lattice.to_numpy(),
        momentum_u=s.momentum_velocity_lattice.to_numpy(),
        f=s.momentum_populations.to_numpy(),
        h=s.phase_populations.to_numpy(),
        solid=s.solid_mask.to_numpy(),
        wall=s.wall_mask.to_numpy(),
        body_id=s.solid_body_id.to_numpy(),
        pressure=s.pressure_lattice.to_numpy(),
        reference_pressure=s.hydrostatic_reference_pressure.to_numpy(),
        reference_density=s.hydrostatic_reference_density.to_numpy(),
        accel=s.fluid_acceleration_lattice.to_numpy(),
        terms=terms,
        temperature=s.thermal.water_temperature.to_numpy(),
        center=s.body_center.to_numpy(),
        body_velocity=s.body_velocity.array(),
        body_mass=s.body_mass_lattice.array(),
    )


def metrics(s, d, step):
    active = (d["solid"] == 0) & (d["wall"] == 0)
    speed = np.abs(d["u"]).sum(axis=-1)
    pos = np.unravel_index(np.argmax(np.where(active, speed, -np.inf)), speed.shape)
    rho = s._air_density_lattice + (
        s._water_density_lattice - s._air_density_lattice
    ) * np.clip(d["phi"], 0, 1)
    water = active & (d["phi"] > 0.5)
    gas = active & (d["phi"] < 0.01)
    return dict(
        step=step,
        time_s=step * s.scales.dt_s,
        max_l1=float(speed[pos]),
        max_l2=float(np.linalg.norm(d["u"][active], axis=-1).max()),
        cell=[int(x) for x in pos],
        phi_at_max=float(d["phi"][pos]),
        u_at_max=d["u"][pos].tolist(),
        accel_at_max=d["accel"][pos].tolist(),
        density_at_max_kg_m3=float(rho[pos] * s.config.rho_water),
        tau_at_max=float(d["terms"][pos][4]),
        water_max_l1=float(speed[water].max()) if water.any() else 0.0,
        gas_max_l1=float(speed[gas].max()) if gas.any() else 0.0,
        phase_amount=float(d["phi"][active].sum(dtype=np.float64)),
        phase_min=float(d["phi"][active].min()),
        phase_max=float(d["phi"][active].max()),
        centers=d["center"].tolist(),
        body_mass=d["body_mass"].tolist(),
        finite=all(
            np.isfinite(d[k]).all()
            for k in ("u", "phi", "f", "h", "pressure", "temperature")
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart", type=Path)
    p.add_argument("--end", type=float, default=0.15)
    p.add_argument("--sample-steps", type=int, default=120)
    p.add_argument("--reference", type=float, default=4.0)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument(
        "--case",
        choices=(
            "baseline",
            "no_thermal",
            "fixed_bodies",
            "fixed_bodies_phase",
            "fixed_phase",
            "fixed_from_start",
        ),
        default="baseline",
    )
    p.add_argument("--checkpoints", type=float, nargs="*", default=[])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = create_config(
        parse_args(
            [
                "--reference-velocity-m-s",
                str(args.reference),
                "--interface-width",
                str(args.width),
            ]
        )
    )
    source_paths = [
        Path("iceflow2d_v3") / x
        for x in ("simulator.py", "config.py", "lattice.py", "thermal_d2q5.py")
    ]
    source_hashes = {
        str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths
    }
    manifest = dict(
        case=args.case,
        solver_class=IceFlow2D.__name__,
        config=config.to_dict(),
        speed_guard_l1=_MAX_D2Q9_LATTICE_VELOCITY_L1,
        source_sha256=source_hashes,
        restart=str(args.restart) if args.restart else None,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    s = IceFlow2D(config)
    if args.restart:
        restore(s, args.restart)
    s._thermal_stability_check_enabled = True
    frozen_body = args.case in (
        "fixed_bodies",
        "fixed_bodies_phase",
        "fixed_from_start",
    )
    frozen_phase = args.case in ("fixed_bodies_phase", "fixed_phase")
    no_thermal = args.case in (
        "no_thermal",
        "fixed_bodies",
        "fixed_bodies_phase",
        "fixed_phase",
        "fixed_from_start",
    )
    if frozen_body:
        s.body_velocity.fill(0.0)
        s.body_angular_velocity.fill(0.0)
    if frozen_phase:
        s._update_streamed_macroscopic_fields = lambda: recover_fixed_phase(s)
    if frozen_body:
        s._advance_body = lambda: None
    if no_thermal:

        def only_check():
            s._measure_water_outflow_rate()
            s._check_thermal_advection_stability(target_step=s.steps + 1)

        s._advance_thermal = only_check
    targets = {round(t / s.scales.dt_s) for t in args.checkpoints}
    end = round(args.end / s.scales.dt_s)
    started = time.monotonic()
    rows = []
    last_step = s.steps
    error = None
    with (args.output / "metrics.jsonl").open("w") as out:
        while True:
            d = fields(s)
            row = metrics(s, d, last_step)
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if s.steps in targets and error is None:
                checkpoint(s, args.output / f"checkpoint_{s.steps:07d}")
            if s.steps in targets or error or s.steps >= end:
                np.savez_compressed(args.output / f"fields_{last_step:07d}.npz", **d)
                print(json.dumps(row), flush=True)
            if error or s.steps >= end:
                break
            stride = min(
                args.sample_steps,
                8
                if s.steps >= round(0.1 / s.scales.dt_s) and not args.restart
                else args.sample_steps,
            )
            target = min(
                end,
                s.steps + stride,
                min((t for t in targets if t > s.steps), default=end),
            )
            try:
                s.step(target - s.steps)
                last_step = s.steps
            except RuntimeError as exc:
                error = str(exc)
                last_step = s.steps + 1
    summary = dict(
        case=args.case,
        completed=error is None,
        error=error,
        last=rows[-1],
        peak_sampled_l1=max(x["max_l1"] for x in rows),
        elapsed_s=time.monotonic() - started,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert source_hashes == {
        str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in source_paths
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
