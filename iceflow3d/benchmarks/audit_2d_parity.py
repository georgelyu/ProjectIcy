"""Local operator comparisons against iceflow2d_v3, without editing either solver.

Run from the repository root with the existing Taichi-enabled Python:
    python iceflow3d/benchmarks/audit_2d_parity.py --output /tmp/parity.json

These are controlled counterexamples to algorithmic parity, not a long-time
stability or accuracy benchmark. Third-axis-independent inputs are used where
the comparison admits a planar reduction. Units are stated in result keys.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from iceflow2d_v3.config import IceFlowConfig as Config2D, ThermalConfig as Thermal2D
from iceflow2d_v3.lattice import (
    momentum_equilibrium as equilibrium2,
    phase_equilibrium as phase2,
)
from iceflow2d_v3.simulator import IceFlow2D
from iceflow3d import IceBodyConfig, IceFlow3D, IceFlowConfig
from iceflow3d.lattice import (
    DIRECTIONS,
    mixture_kinematic_viscosity,
    momentum_equilibrium as equilibrium3,
    phase_equilibrium as phase3,
)

C2 = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
               (1, 1), (-1, 1), (-1, -1), (1, -1)])
C3 = np.array(DIRECTIONS)
I2, I3 = (8, 8), (8, 8, 6)


@ti.kernel
def equilibrium_probe(two: ti.types.ndarray(), three: ti.types.ndarray()):
    for q in range(9):
        two[q] = equilibrium2(q, 0.012, 0.7, ti.Vector([0.08, 0.06]))
    for q in range(27):
        three[q] = equilibrium3(q, 0.012, 0.7, ti.Vector([0.08, 0.06, 0.0]))


@ti.kernel
def viscosity_probe(a: ti.template(), b: ti.template(), out: ti.types.ndarray()):
    out[0] = a._fluid_kinematic_viscosity_lattice(8, 8) * a.config.dx**2 / a.scales.dt_s
    out[1] = mixture_kinematic_viscosity(
        0.5, 90.0, b.config.rho_water, b.config.rho_air, b.mu_air
    )


@ti.kernel
def gradient_probe(a: ti.template(), b: ti.template(), out: ti.types.ndarray()):
    # First interior row; a linear field normal to the stationary bottom wall.
    out[0] = a._phase_central_gradient(8, 2).y
    out[1] = b.phase_gradient_lattice[8, 2, 6].y
    out[2] = b.phase_gradient_lattice[8, 2, 6].y


@ti.kernel
def flux_probe(a: ti.template(), b: ti.template(), out: ti.types.ndarray()):
    out[0] = a._phase_flux_force_density(8, 8).x
    out[1] = b._force_at(ti.Vector([8, 8, 6]), True).x


@ti.kernel
def ice_ghost_probe(a: ti.template(), b: ti.template(), out: ti.types.ndarray()):
    out[0] = a._ice_capillary_stress(11, 32, 12, 32)[1]
    out[1] = b._capillary_neighbor(ti.Vector([11, 32, 6]), ti.Vector([1, 0, 0]))[0, 1]
    out[2] = a._regularization_velocity_neighbor(11, 32, 1, 0).x
    out[3] = b._stress_neighbor_velocity(ti.Vector([11, 32, 6]), ti.Vector([1, 0, 0])).x


@ti.kernel
def seed_overshoot(a: ti.template(), b: ti.template()):
    a.water_phase[8, 8] = 1.1
    a.momentum_velocity_lattice[8, 8] = ti.Vector([0.07, -0.03])
    for q in range(9):
        a.phase_populations[8, 8, q] = phase2(q, 1.1, ti.Vector([0.07, -0.03]))
    for q in range(27):
        b.h[8, 8, 6, q] = phase3(q, 1.1, ti.Vector([0.07, -0.03, 0.0]))


@ti.kernel
def seed_matched_phase(a: ti.template(), b: ti.template()):
    for q in range(9):
        a.phase_populations[8, 8, q] = phase2(q, 0.6, ti.Vector([0.03, -0.02]))
    for q in range(27):
        b.h[8, 8, 6, q] = phase3(q, 0.6, ti.Vector([0.03, -0.02, 0.0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}

    def record(name, value):
        results[name] = value
        print(json.dumps({name: value}, ensure_ascii=False), flush=True)

    a = IceFlow2D(Config2D(
        resolution=(32, 48), dx=0.0005, boundary_cells=2,
        water_width_fraction=30 / 32, water_height_fraction=24 / 48,
        ice_width_fraction=8 / 32, ice_height_fraction=8 / 48, ice_base_y_cells=30,
        phase_warmup_steps=0, ice_initial_angle=0, gravity=(0, 0), sigma=0,
        viscosity_water=None, viscosity_air=None,
        phase_gradient_scheme="connected_ls", phase_momentum_coupling="consistent_flux",
        phase_normal_epsilon=1e-8,
        thermal=Thermal2D(initial_water_temperature_c=90),
    ))
    b = IceFlow3D(IceFlowConfig(
        nx=32, ny=48, nz=12, domain_size_m=(0.016, 0.024, 0.006),
        boundary_cells=2, water_width=30, water_height=24, floor_water_depth_cells=0,
        ice_bodies=(IceBodyConfig((16, 34, 6), (8, 8, 4)),),
        phase_warmup_steps=0, gravity=(0, 0, 0), surface_tension_n_m=0,
        backend="cuda",
    ))
    record("fixture", {"backend": "cuda", "taichi": list(ti.__version__),
                       "dx_m": a.config.dx, "dt_s": a.scales.dt_s,
                       "shape_2d": [a.nx, a.ny], "shape_3d": list(b.shape)})
    record("initial_interface_without_warmup", {
        "2d_phi_below_surface": float(a.water_phase[8, 23]),
        "3d_phi_below_surface": float(b.water_phase[8, 23, 6]),
        "2d_phi_above_surface": float(a.water_phase[8, 24]),
        "3d_phi_above_surface": float(b.water_phase[8, 24, 6]),
    })

    two, three = np.zeros(9, np.float32), np.zeros(27, np.float32)
    equilibrium_probe(two, three)
    projected = np.array([three[(C3[:, :2] == c).all(axis=1)].sum(dtype=float) for c in C2])
    record("momentum_equilibrium_planar_reduction", {
        "u": [0.08, 0.06, 0], "rho": 0.7, "p": 0.012,
        "max_population_difference": float(abs(projected - two).max()),
        "raw_21_2d": float((two * C2[:, 0]**2 * C2[:, 1]).sum()),
        "raw_21_3d": float((three * C3[:, 0]**2 * C3[:, 1]).sum()),
    })

    a.water_phase.fill(0.6)
    b.water_phase.fill(0.6)
    a.momentum_velocity_lattice.fill([0.03, -0.02])
    a.fluid_acceleration_lattice.fill(0)
    b.velocity_lattice.fill([0.03, -0.02, 0])
    a.phase_gradient_lattice.fill([0.02, -0.01])
    b.phase_gradient_lattice.fill([0.02, -0.01, 0])
    seed_matched_phase(a, b)
    a._collide_phase(gradients_prepared=True)
    b._collide_phase()
    post2, post3 = a.phase_stream_buffer.to_numpy()[I2], b.h_post.to_numpy()[I3]
    marginal = np.array([post3[(C3[:, :2] == c).all(axis=1)].sum(dtype=float) for c in C2])
    phase_error = float(abs(marginal - post2).max())
    assert phase_error < 2e-7, phase_error
    record("matched_phase_collision_control", {
        "epsilon_both": 1e-8, "max_population_difference": phase_error,
        "agreement_tolerance": 2e-7,
    })

    a.water_phase.fill(0.5)
    a.thermal.water_volume_m2.fill(0.5 * a.config.dx**2)
    a.thermal.water_sensible_energy.fill(0.5 * a.config.dx**2 * 1000 * 4186 * 90)
    out = np.zeros(2, np.float64)
    viscosity_probe(a, b, out)
    record("mixture_viscosity_phi_0_5_T90", {
        "nu_2d_m2_s": float(out[0]), "nu_3d_m2_s": float(out[1]),
        "ratio_2d_over_3d": float(out[0] / out[1]),
    })

    _, y2 = np.indices((a.nx, a.ny))
    _, y3, _ = np.indices(b.shape)
    a.water_phase.from_numpy((0.2 + 0.01 * y2).astype(np.float32))
    b.water_phase.from_numpy((0.2 + 0.01 * y3).astype(np.float32))
    a._prepare_phase_gradients()
    b._gradients()
    out3 = np.zeros(3, np.float64)
    gradient_probe(a, b, out3)
    record("wall_normal_density_source_gradient", {
        "2d_shared_gradient": float(out3[0]), "3d_phase_gradient": float(out3[1]),
        "3d_density_source_gradient": float(out3[2]),
    })

    a.capillary_stress_lattice.fill([2, 1, 3])
    b.capillary_stress.fill([[2, 1, 0], [1, 3, 0], [0, 0, 5]])
    ghost_u2 = np.zeros((a.nx, a.ny, 2), np.float32)
    ghost_u3 = np.zeros((*b.shape, 3), np.float32)
    ghost_u2[11, :, 0], ghost_u2[10, :, 0] = 0.04, 0.1
    ghost_u3[11, :, :, 0], ghost_u3[10, :, :, 0] = 0.04, 0.1
    a.stability_velocity.from_numpy(ghost_u2)
    b.velocity_lattice.from_numpy(ghost_u3)
    ghost_out = np.zeros(4, np.float64)
    ice_ghost_probe(a, b, ghost_out)
    record("ice_boundary_ghosts", {
        "2d_Kxy_ghost": float(ghost_out[0]), "3d_Kxy_ghost": float(ghost_out[1]),
        "2d_hrr_velocity_ghost": float(ghost_out[2]), "3d_hrr_velocity_ghost": float(ghost_out[3]),
    })

    u2, u3 = np.zeros((a.nx, a.ny, 2), np.float32), np.zeros((*b.shape, 3), np.float32)
    u2[8, :, 0], u2[9, :, 0] = 0.04, 0.1
    u3[8, :, :, 0], u3[9, :, :, 0] = 0.04, 0.1
    a.stability_velocity.from_numpy(u2)
    b.velocity_lattice.from_numpy(u3)
    j2, j3 = np.zeros_like(u2), np.zeros_like(u3)
    j2[..., 0], j3[..., 0] = 0.002, 0.002
    a.phase_mass_flux.from_numpy(j2)
    b.phase_mass_flux.from_numpy(j3)
    flux_probe(a, b, out)
    record("phase_flux_momentum_force_lattice", {
        "2d_centered": float(out[0]), "3d_upwind": float(out[1]),
    })

    seed_overshoot(a, b)
    a._bound_water_phase()
    b._recover_phase()
    h2, h3 = a.phase_populations.to_numpy()[I2], b.h.to_numpy()[I3]
    record("phase_overshoot_repair", {
        "initial_phi": 1.1, "initial_velocity": [0.07, -0.03],
        "bounded_phi_2d": float(a.water_phase[I2]), "bounded_phi_3d": float(b.water_phase[I3]),
        "first_phase_moment_2d": (C2.T @ h2).tolist(),
        "first_phase_moment_3d": (C3.T @ h3).tolist(),
    })

    valid2 = np.ones((a.nx, a.ny), np.int32)
    valid3 = np.ones(b.shape, np.int8)
    valid2[I2], valid3[8, 8, :] = 0, 0
    state2 = np.zeros((a.nx, a.ny, 4), np.float32)
    state3 = np.zeros((*b.shape, 5), np.float32)
    state2[9, 8, 0], state3[9, 8, :, 0] = 1, 1
    a.refill_valid.from_numpy(valid2)
    a.refill_state.from_numpy(state2)
    b.refill_valid.from_numpy(valid3)
    b.refill_state.from_numpy(state3)
    a._propagate_connected_refill()
    b._propagate_refill()
    record("connected_refill_weights", {
        "2d_new_phi": float(a.refill_next_state[I2][0]),
        "3d_new_phi": float(b.refill_next[I3][0]),
    })

    a.water_phase.fill(1)
    b.water_phase.fill(1)
    a.thermal.water_volume_m2.fill(0)
    a.thermal.water_sensible_energy.fill(0)
    b.water_volume.fill(0)
    b.water_energy.fill(0)
    a.thermal.water_volume_m2[10, 8] = a.config.dx**2
    a.thermal.water_sensible_energy[10, 8] = a.config.dx**2 * 1000 * 4186 * 80
    v3 = np.zeros(b.shape, np.float64)
    v3[10, 8, :] = b.cell_volume
    b.water_volume.from_numpy(v3)
    b.water_energy.from_numpy(v3 * 1000 * 4186 * 80)
    a.thermal._build_phase_aperture_targets(a.water_phase, a.wall_mask, a.solid_mask)
    b._build_water_targets()
    record("newly_wet_temperature_donor_two_cells_away", {
        "2d_temperature_c": float(a.thermal._phase_target_energy[I2] / (a.config.dx**2 * 1000 * 4186)),
        "3d_temperature_c": float(b.target_energy[I3] / (b.cell_volume * b.rho_cp)),
    })

    origin2, origin3 = a.body_reference_origin.to_numpy().copy(), b.rigid.origin.copy()
    a.body_velocity.fill(0)
    a.body_angular_velocity.fill(0)
    a.hydrodynamic_impulse[None] = [float(a.body_mass_lattice[None]) * 0.02, 0]
    a.hydrodynamic_torque.fill(0)
    a._prepare_body_motion()
    a.body_motion.commit()
    b.rigid.velocity[:] = 0
    b.rigid.omega[:] = 0
    b.hydrodynamic_force.from_numpy(np.array([[b.rigid.mass[0] * 0.02, 0, 0]]))
    b.hydrodynamic_torque.fill(0)
    b.rigid.advance()
    record("rigid_order_from_rest_same_impulse", {
        "2d_displacement_cells": (a.body_reference_origin.to_numpy() - origin2).tolist(),
        "3d_displacement_cells": (b.rigid.origin[0] - origin3[0]).tolist(),
        "2d_velocity_lattice": a.body_velocity.to_numpy().tolist(),
        "3d_velocity_lattice": b.rigid.velocity[0].tolist(),
    })

    masses2 = a.thermal.body_solid_mass.to_numpy()
    masses3 = b.material_mass.to_numpy()
    for fraction in (0.75, 0.25):
        a.thermal.body_solid_mass.from_numpy(masses2 * fraction)
        b.material_mass.from_numpy(masses3 * fraction)
        a._update_moving_body_mass_properties()
        b.rigid.update_mass()
        record(f"eroded_body_fraction_{fraction}", {
            "2d_Izz_per_mass_cells2": float(a.body_inertia_lattice[None] / a.body_mass_lattice[None]),
            "3d_Izz_per_mass_cells2": float(b.rigid.inertia[0, 2, 2] / b.rigid.mass[0]),
            "2d_contact_lower": a.body_motion.material_lower.to_numpy()[0].tolist(),
            "2d_contact_upper": a.body_motion.material_upper.to_numpy()[0].tolist(),
            "3d_contact_lower": b.rigid.lower[0].tolist(),
            "3d_contact_upper": b.rigid.upper[0].tolist(),
        })

    from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example2
    from iceflow3d.examples import dambreak_four_ice_melting_3d as example3
    case2 = example2.create_config(example2.build_parser().parse_args([]))
    case3 = example3.create_config(example3.build_parser().parse_args([]))
    names = ("momentum_collision_model", "phase_normal_epsilon", "phase_warmup_steps",
             "cm_acoustic_relaxation_rate", "density_gradient_source_limit",
             "phase_flux_acceleration_limit", "gas_state_repair", "body_contact_iterations")
    defaults = {name: {"2d": getattr(case2, name), "3d": getattr(case3, name)} for name in names}
    defaults["surface_tension_n_m"] = {"2d": case2.sigma, "3d": case3.surface_tension_n_m}
    record("four_ice_example_defaults", defaults)

    hashes = {}
    for folder in ("iceflow2d_v3", "iceflow3d"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if not path.name.startswith("test_"):
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    results["source_sha256"] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
