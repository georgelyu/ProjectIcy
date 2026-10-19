"""Substep evidence and matched controls for the production four-ice run.

Only sparse diagnostics and rendered images are saved; checkpoint grids stay
in RAM. Runtime controls belong to a diagnostic subclass, not production.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iceflow2d_v3.benchmarks.diagnose_single_dambreak import SingleProbe
from iceflow2d_v3.benchmarks.diagnose_connected_instability import (
    host_snapshot,
    restore,
)
from iceflow2d_v3.benchmarks.diagnose_dambreak_instability import (
    device_snapshot,
    state_summary,
)
from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def record(s, step, stage):
    summary = state_summary(s)
    s.probe_points_host[-1] = summary["peak_cell"]
    s.probe_points.from_numpy(s.probe_points_host)
    points = s.point_rows()
    names = (
        "phase_gradient_lattice",
        "source_gradient_lattice",
        "phase_gradient_status",
        "phase_gradient_data_rank",
        "phase_gradient_spatial_rank",
        "phase_gradient_boundary_residual",
        "phase_reconstruction_coefficients",
        "phase_donor_mask",
        "phase_wall_geometry",
        "phase_mass_flux",
        "previous_solid_mask",
        "phase_populations",
        "momentum_populations",
        "phase_stream_buffer",
        "momentum_stream_buffer",
        "refill_state",
    )
    fields = {name: getattr(s, name).to_numpy() for name in names}
    for point in points:
        index = tuple(point["cell"])
        for name, array in fields.items():
            point[name] = array[index].tolist()
    phi = s.water_phase.to_numpy()
    solid = s.solid_mask.to_numpy()
    previous = s.previous_solid_mask.to_numpy()
    wall = s.wall_mask.to_numpy()
    active = (solid == 0) & (wall == 0)
    summary.update(
        raw_phi_min=float(phi[active].min()), raw_phi_max=float(phi[active].max())
    )
    out = dict(attempted_step=step, stage=stage, state=summary, points=points)
    if stage in ("body", "thermal", "before"):
        exposed = np.argwhere(active & (previous != 0))
        out["newly_exposed"] = [
            dict(cell=cell.tolist(), phi=float(phi[tuple(cell)])) for cell in exposed
        ]
    if stage in ("body", "thermal") or step >= 8525:
        out["floor_line"] = {
            "i_start": 140,
            "j": 3,
            "phi": phi[140:181, 3].tolist(),
            "solid": solid[140:181, 3].tolist(),
            "source_gradient": fields["source_gradient_lattice"][140:181, 3].tolist(),
            "velocity": s.physical_velocity_lattice.to_numpy()[140:181, 3].tolist(),
            "pressure": s.pressure_lattice.to_numpy()[140:181, 3].tolist(),
        }
    return out


def visual_state(s, step, stage, out):
    from iceflow2d_v3.benchmarks.render_single_dambreak_step import (
        setup_fonts,
        classify,
        COLORS,
        LABELS,
    )
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, SymLogNorm
    from matplotlib.patches import Patch

    setup_fonts()
    phi = s.water_phase.to_numpy()
    solid = s.solid_mask.to_numpy()
    wall = s.wall_mask.to_numpy()
    vel = s.physical_velocity_lattice.to_numpy() * 40
    gradient = s.source_gradient_lattice.to_numpy()
    blocked = (solid != 0) | (wall != 0)
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), layout="constrained")
    nx, ny = phi.shape
    dx = 1 / 3
    for ax, array, title, cmap, norm in (
        (axes[0, 0], phi, "全域相场 φ", "Blues", None),
        (
            axes[0, 1],
            np.linalg.norm(vel, axis=-1),
            "全域速度模 (m/s)",
            "magma",
            SymLogNorm(
                0.1, vmin=0, vmax=max(4, np.linalg.norm(vel, axis=-1)[~blocked].max())
            ),
        ),
    ):
        im = ax.imshow(
            np.ma.masked_where(blocked, array).T,
            origin="lower",
            extent=(0, nx * dx, 0, ny * dx),
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, shrink=0.65)
        ax.plot((152 + 0.5) * dx, (3 + 0.5) * dx, "*", color="crimson", ms=12)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    kinds = classify(dict(phi=phi, solid=solid, wall=wall))
    for ax in axes[1]:
        ax.imshow(
            kinds.T,
            origin="lower",
            extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
            cmap=ListedColormap(COLORS),
            vmin=-0.5,
            vmax=4.5,
            interpolation="nearest",
        )
        ax.set(xlim=(145.5, 157.5), ylim=(1.5, 6.5), xlabel="i", ylabel="j")
        ax.set_xticks(range(146, 158))
        ax.set_yticks(range(2, 7))
        ax.set_xticks(np.arange(145.5, 158, 0.999999), minor=True)
        ax.set_yticks(np.arange(1.5, 7, 1), minor=True)
        ax.grid(which="minor", color="#aaa", lw=0.4)
        ax.plot(152, 3, "s", mfc="none", mec="crimson", ms=36, mew=2)
    for i in range(146, 158):
        for j in range(2, 7):
            text = (
                f"{phi[i, j]:.3g}"
                if not blocked[i, j]
                else ("冰" if solid[i, j] else "墙")
            )
            axes[1, 0].text(i, j, text, ha="center", va="center", fontsize=8)
            if not blocked[i, j]:
                gx, gy = gradient[i, j]
                axes[1, 1].text(
                    i, j, f"{gx:.3g}\n{gy:.3g}", ha="center", va="center", fontsize=7
                )
    axes[1, 0].set_title("局部点类型与 φ（红框为最终超限点）")
    axes[1, 1].set_title("实际缓存 ∇φ：上行为 x，下行为 y，格子单位")
    fig.legend(
        handles=[
            Patch(color=color, label=label) for color, label in zip(COLORS, LABELS)
        ],
        loc="outside lower center",
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(
        f"四冰块数据优先重构：第 {step:,} 步，{stage}，t={step / 120000:.9f} s\n局部颜色表示点类型；梯度取当前实际缓存",
        fontsize=14,
    )
    path = out / f"state_{step:06d}_{stage}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=8000)
    p.add_argument("--end", type=int, default=8540)
    p.add_argument("--detail-start", type=int, default=8480)
    p.add_argument(
        "--checkpoints", type=int, nargs="+", default=[8200, 8400, 8480, 8510]
    )
    p.add_argument("--control-steps", type=int, default=600)
    p.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "fixed_pose",
            "stationary_pose",
            "no_thermal",
            "zero_boundary_source",
            "zero_all_source",
            "no_flux_force",
            "positive_phase_populations",
            "postcollision_bounceback",
            "halfway_links",
        ],
    )
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s = SingleProbe(create_config())
    s._thermal_stability_check_enabled = True
    s.probe_points_host[:] = np.array(
        [
            (152, 3),
            (151, 3),
            (153, 3),
            (150, 3),
            (154, 3),
            (152, 4),
            (151, 4),
            (153, 4),
            (149, 3),
            (155, 3),
            (152, 3),
        ],
        np.int32,
    )
    s.probe_points.from_numpy(s.probe_points_host)
    started = time.monotonic()
    s.step(args.start)
    print(
        json.dumps(
            dict(
                event="ready",
                elapsed_s=time.monotonic() - started,
                state=state_summary(s),
            )
        ),
        flush=True,
    )
    snapshots = {}
    trace = []
    failure = None
    images = []
    for step in range(args.start + 1, args.end + 1):
        if s.steps in args.checkpoints:
            snapshots[s.steps] = (device_snapshot(s), host_snapshot(s))
        detailed = step >= args.detail_start

        def capture(stage):
            if detailed or stage == "thermal" and step % 10 == 0:
                trace.append(record(s, step, stage))

        try:
            capture("before")
            s._collide_momentum()
            capture("collision")
            s._stream()
            capture("stream")
            s._update_streamed_macroscopic_fields()
            capture("macros")
            s._bound_water_phase()
            capture("bounded_phase")
            s._prepare_phase_gradients()
            s._update_pressure()
            capture("pressure")
            s._advance_body()
            capture("body")
            s._advance_thermal()
            s.steps += 1
            capture("thermal")
            if step in (8400, 8480, 8510, 8520, 8530):
                images.append(visual_state(s, step, "end", out))
        except RuntimeError as exc:
            failure = dict(attempted_step=step, error=str(exc), state=state_summary(s))
            images.append(visual_state(s, step, "rejected_before_thermal", out))
            break
        if step % 100 == 0:
            print(json.dumps(dict(event="trace", state=state_summary(s))), flush=True)
    save(out / "trace.json", trace)
    save(out / "failure.json", failure)
    print(json.dumps(dict(event="failure", failure=failure)), flush=True)
    results = []
    for checkpoint, snapshot in snapshots.items():
        for mode in args.modes:
            restore(s, snapshot)
            s.control_mode[None] = {
                "zero_boundary_source": 1,
                "no_flux_force": 2,
                "zero_all_source": 3,
                "positive_phase_populations": 4,
                "postcollision_bounceback": 5,
                "halfway_links": 6,
            }.get(mode, 0)
            if mode == "stationary_pose":
                s.body_velocity.fill(0)
                s.body_angular_velocity.fill(0)
            history = []
            error = None
            for k in range(args.control_steps):
                try:
                    s._advance_fluid()
                    if mode not in ("fixed_pose", "stationary_pose"):
                        s._advance_body()
                    if mode not in ("fixed_pose", "stationary_pose", "no_thermal"):
                        s._advance_thermal()
                    s.steps += 1
                    if k % 10 == 0 or k == args.control_steps - 1:
                        state = state_summary(s)
                        history.append(state)
                        if state["speed_l1_max"] > 1 or not np.isfinite(
                            state["speed_l1_max"]
                        ):
                            break
                except RuntimeError as exc:
                    error = str(exc)
                    break
            result = dict(
                checkpoint=checkpoint,
                mode=mode,
                completed_steps=s.steps - checkpoint,
                error=error,
                last=state_summary(s),
                history=history,
            )
            results.append(result)
            save(out / "matched_controls.json", results)
            print(
                json.dumps({k: v for k, v in result.items() if k != "history"}),
                flush=True,
            )
    save(
        out / "manifest.json",
        dict(
            source_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in map(
                    Path,
                    (
                        "iceflow2d_v3/simulator.py",
                        "iceflow2d_v3/phase_reconstruction.py",
                    ),
                )
            },
            images=images,
            raw_fields_saved=False,
        ),
    )


if __name__ == "__main__":
    main()
