"""First-order experimental orchestration with explicit conservation diagnostics."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import taichi as ti
from .config import Config
from .geometry import Geometry
from .flow import Flow
from .state import State
from .thermal import Thermal
from .coupling import Coupling
from .transport import Transport
from .rigid import Rigid
from .diagnostics import Budgets
from .momentum import Momentum
from .merge import CutCellMerge
from .health import Health
from .phase_momentum import PhaseMomentum
from .subgrid_viscous import SubgridViscous


class SimulationError(RuntimeError):
    pass


class Simulation:
    """Taichi must be initialized by the caller with fast_math=False.

    A completed run is not an A-H validation certificate. See validation/status.json.
    Diagnostics never repair fields or renormalize mass/energy.
    """
    def __init__(self, config=None):
        from .checkpoint import source_hashes
        self.source_sha256 = source_hashes()
        self.cfg = config or Config()
        self.geometry = Geometry(self.cfg)
        self.flow = Flow(self.cfg, self.geometry)
        self.state = State(self.cfg, self.geometry, self.flow)
        self.thermal = Thermal(self.cfg, self.geometry, self.state)
        self.coupling = Coupling(self.cfg, self.geometry, self.state, self.flow)
        self.transport = Transport(self.cfg, self.geometry, self.state, self.flow)
        self.momentum = Momentum(self.cfg, self.geometry, self.state, self.flow, self.transport)
        self.merge = CutCellMerge(self.cfg, self.geometry, self.state, self.flow)
        self.rigid = Rigid(self.cfg, self.geometry)
        self.rigid.phase = PhaseMomentum(self.cfg, self.geometry, self.rigid, self.coupling)
        self.rigid.subgrid = SubgridViscous(self.cfg, self.geometry, self.state, self.flow, self.rigid, self.merge)
        self.budgets = Budgets(self.cfg, self.geometry, self.state, self.flow, self.rigid, self.coupling)
        self.health = Health(self.cfg, self.geometry, self.state, self.flow)
        for _ in range(self.cfg.warmup_steps):
            self.flow.warmup_collision()
            self.flow.warmup_stream()
            self.transport.phase_step()
            self.state.temperatures()
            self.state.restore_populations()
        # Measure baselines only after the final discrete initialization.
        self.merge.step()
        self.state.temperatures()
        self.thermal.reconstruct()
        self.state.restore_populations()
        self.flow.freeze_reference_density()
        self.steps = 0
        self.wall_seconds = 0.0
        self.schedule = None
        self.recovery_schedule = None
        self.last_failure_snapshot = None
        self.last_failure_diagnostics = None
        self.initial = None
        self.initial = self.diagnostics()

    def _advance(self):
        self.geometry.remember_geometry()
        self.state.temperatures()
        self.flow.collide()
        self.rigid.begin()
        for _ in range(self.cfg.rigid_iterations):
            self.flow.prepare_boundary()
            self.flow.reduce_boundary()
            self.rigid.iterate(self.flow)
        self.flow.prepare_boundary()
        self.flow.reduce_boundary()
        self.flow.stream()
        self.flow.recover_moments()
        self.flow.recover_pressure()
        self.rigid.commit_force(self.flow)
        self.rigid.recover_velocity()
        self.budgets.gather_forces()
        self.budgets.accumulate()
        self.transport.build_predictor()
        self.momentum.step()
        self.rigid.subgrid.step()
        self.transport.phase_step()
        self.state.temperatures()
        self.thermal.step()
        self.rigid.phase.capture()
        self.coupling.prepare()
        self.geometry.erode(self.coupling.mass)
        self.geometry.update_mass()
        self.rigid.phase.finish()
        self.coupling.finish()
        self.budgets.accumulate_numerical_air()
        self.rigid.recover_velocity()
        self.rigid.advance_pose()
        self.geometry.update_rotation()
        # First moments are expressed in the world frame. Refresh them before
        # contact and velocity recovery at the new orientation.
        self.geometry.update_mass()
        self.rigid.recover_velocity()
        self.rigid.contact()
        self.rigid.recover_velocity()
        self.geometry.rasterize()
        self.rigid.recover_velocity()
        self.transport.remap()
        self.merge.step()
        self.state.temperatures()
        self.thermal.reconstruct()
        self.state.prepare_new_nodes()
        self.state.restore_populations()

        self.health.observe()
        self.rigid.subgrid.observe(self.health)

    def enable_graph(self, recovery=False):
        from .schedule import KernelSchedule
        from .checkpoint import source_hashes
        if self.source_sha256 != source_hashes():
            raise RuntimeError('Production sources changed while initializing this run.')
        self.schedule = KernelSchedule(self)
        if recovery:
            self.transport.polish_override = max(240, self.cfg.pressure_iterations)
            try: self.recovery_schedule = KernelSchedule(self)
            finally: self.transport.polish_override = None
        if self.source_sha256 != source_hashes():
            raise RuntimeError('Production sources changed while compiling this run.')

    def step(self, count=1):
        start = time.perf_counter()
        self.health.reset()
        if self.schedule is not None:
            self.schedule.run(count)
        else:
            for _ in range(count): self._advance()
        self.steps += count
        ti.sync()
        self.wall_seconds += time.perf_counter()-start

    def advance_checked(self, count):
        """Commit a batch only after validation; retry capacity failures strictly.

        The retry changes the linear-solver work, never the accepted physical
        time interval or the interface mass/heat ledger. Other failures retain
        the last accepted state and raise for diagnosis.
        """
        from .checkpoint import Snapshot
        self.last_failure_snapshot = None
        self.last_failure_diagnostics = None
        before = Snapshot(self)
        try:
            self.step(count)
            return self.validate(), False
        except SimulationError:
            failed = self.diagnostics()
            self.last_failure_diagnostics = failed
            self.last_failure_snapshot = Snapshot(self)
            before.restore(self)
            # A capacity failure early in a device batch can cause later
            # velocity/NaN flags. Retrying the original healthy state with a
            # stricter solve is still valid in that case.
            capacity_failure = (bool(failed['batch_failure_flags'] & 9)
                                or failed['capacity_residual_max'] > self.cfg.capacity_tolerance
                                or failed['phase_min'] < -1e-6 or failed['phase_max'] > 1+1e-5)
            if self.cfg.projection_solver != 'multigrid' or not capacity_failure:
                raise
            original = self.schedule
            self.transport.polish_override = max(240, self.cfg.pressure_iterations)
            self.schedule = self.recovery_schedule
            try:
                self.step(count)
                result = self.validate()
                self.last_failure_snapshot = None
                self.last_failure_diagnostics = None
                return result, True
            except SimulationError:
                self.last_failure_diagnostics = self.diagnostics()
                self.last_failure_snapshot = Snapshot(self)
                before.restore(self)
                raise
            finally:
                self.schedule = original
                self.transport.polish_override = None

    def diagnostics(self):
        c, g = self.cfg, self.geometry
        q = self.state.q.to_numpy().astype(np.float64)-self.state.comp.to_numpy().astype(np.float64)
        solid = g.solid.to_numpy().astype(np.float64)-g.solid_comp.to_numpy().astype(np.float64)
        solid_heat = g.ice_heat.to_numpy().astype(np.float64)-g.ice_heat_comp.to_numpy().astype(np.float64)
        capacity = g.capacity.to_numpy().astype(np.float64)
        occupied = g.occupancy.to_numpy().astype(np.float64)
        velocity = self.flow.velocity.to_numpy().astype(np.float64)
        wet, gas = q[..., 0] > 1e-10, q[..., 1] > 1e-10
        active = capacity > 1e-8
        tw, ta, ts = self.state.temperature_w.to_numpy(), self.state.temperature_a.to_numpy(), g.ice_temperature.to_numpy()
        mass_s = float(solid.sum()*c.rho_s_lu*c.mass_unit)
        mass_w = float(q[..., 0].sum()*c.mass_unit)
        mass_a = float(q[..., 1].sum()*c.rho_a_lu*c.mass_unit)
        sensible_w, sensible_s = float(q[..., 2].sum()*c.energy_unit), float(solid_heat.sum()*c.energy_unit)
        sensible_a = float((q[..., 3]+q[..., 1]*c.air_capacity_lu*c.air_temperature/c.temperature_scale).sum()*c.energy_unit)
        latent = mass_w*c.latent
        air_heat = (float(self.coupling.air_heat_total[None])-float(self.coupling.air_heat_comp[None]))*c.energy_unit
        initial = self.initial
        condensed0 = mass_s+mass_w if initial is None else initial['mass_ice_kg_m']+initial['mass_water_kg_m']
        air0 = mass_a if initial is None else initial['mass_air_kg_m']
        ice0 = mass_s if initial is None else initial['mass_ice_kg_m']
        enthalpy = sensible_s+sensible_w+latent
        h0 = enthalpy if initial is None else initial['condensed_enthalpy_J_m']
        full_h0 = enthalpy+sensible_a if initial is None else initial['all_enthalpy_J_m']
        phi = q[..., 0][active]/capacity[active]
        expected_air = c.rho_a*(ice0-mass_s)*(1/c.rho_s-1/c.rho_w)
        body_p = np.array(self.rigid.momentum[None], dtype=float)-np.array(self.rigid.momentum_comp[None], dtype=float)
        # Linear momentum unit is mass_unit*dx/dt; angular adds another dx.
        p_unit, l_unit = c.mass_unit*c.dx/c.dt, c.mass_unit*c.dx**2/c.dt
        ii, jj = np.indices(capacity.shape)
        angular = float(((ii+0.5)*q[..., 5]-(jj+0.5)*q[..., 4]).sum()+body_p[2])
        expected = (np.array(self.budgets.external[None], dtype=float)-np.array(self.budgets.external_comp[None], dtype=float)
                    +np.array(self.rigid.contact_impulse[None], dtype=float)-np.array(self.rigid.contact_impulse_comp[None], dtype=float)
                    +np.array(self.rigid.support_impulse[None], dtype=float)-np.array(self.rigid.support_impulse_comp[None], dtype=float))
        measured = np.array([q[..., 4].sum()+body_p[0], q[..., 5].sum()+body_p[1], angular])
        momentum_residual = measured-expected
        phase_roundoff = np.asarray(self.rigid.phase.roundoff_total[None], dtype=float)*np.array([p_unit, p_unit, l_unit])
        fluid_mass = q[..., 0]+c.rho_a_lu*q[..., 1]
        has_mass = fluid_mass > 0.0
        # P^2/M has no meaning in an empty cell. A denominator floor would
        # turn its compensated roundoff momentum into a fictitious huge energy.
        # Keep that momentum in the global balance and disclose it separately.
        kinetic = .5*np.sum((q[..., 4][has_mass]**2+q[..., 5][has_mass]**2)/fluid_mass[has_mass])
        empty_momentum = np.abs(q[..., 4:6][~has_mass]).sum()*p_unit
        local_body_p = body_p.copy()
        origin = np.array(g.origin[None], float)
        local_body_p[2] -= origin[0]*body_p[1]-origin[1]*body_p[0]
        kinetic += .5*float(np.dot(local_body_p, np.array(g.velocity[None], float)))
        kinetic *= c.mass_unit*(c.dx/c.dt)**2
        potential = (np.sum((q[..., 0]+c.rho_a_lu*q[..., 1])*(jj+.5))
                     +g.mass[None]*origin[1]+g.first_moment[None][1])*c.mass_unit*c.gravity*c.dx
        physical_nu = (self.flow.tau_shear.to_numpy()-0.5)*c.dx**2/(3*c.dt)
        polygons = g.poly.to_numpy().astype(float)
        counts = g.count.to_numpy()
        material_i, material_j = np.indices(counts.shape)
        centres = np.stack([material_i+.5-.5*c.nb, material_j+.5-.5*c.nb], axis=-1)
        relative_vertices = (polygons+centres[..., None, :])@np.asarray(g.rotation[None], float).T
        body_velocity = np.asarray(g.velocity[None], float)
        vertex_velocity = body_velocity[:2]+body_velocity[2]*np.stack(
            [-relative_vertices[..., 1], relative_vertices[..., 0]], axis=-1)
        valid_vertices = np.arange(polygons.shape[-2]) < counts[..., None]
        geometry_courant = float(np.linalg.norm(vertex_velocity[valid_vertices], axis=-1).max(initial=0))
        return {
            'step': self.steps, 'time_s': self.steps*c.dt,
            'mass_ice_kg_m': mass_s, 'mass_water_kg_m': mass_w, 'mass_air_kg_m': mass_a,
            'condensed_mass_relative_residual': (mass_s+mass_w-condensed0)/max(condensed0, 1e-30),
            'air_total_relative_error': (mass_a-air0)/max(air0, 1e-30),
            'air_expected_increment_kg_m': expected_air,
            'air_unexpected_residual_kg_m': mass_a-air0-expected_air,
            'sensible_water_J_m': sensible_w, 'sensible_ice_J_m': sensible_s,
            'sensible_air_J_m': sensible_a, 'latent_water_J_m': latent,
            'condensed_enthalpy_J_m': enthalpy, 'all_enthalpy_J_m': enthalpy+sensible_a,
            'condensed_energy_relative_residual': (enthalpy-h0)/max(abs(h0), 1e-30),
            'numerical_air_heat_J_m': air_heat,
            'all_energy_corrected_relative_residual': (enthalpy+sensible_a-full_h0-air_heat)/max(abs(full_h0), 1e-30),
            'phase_mass_step_kg_m': float(self.coupling.melt_step[None])*c.mass_unit,
            'phase_mass_total_kg_m': (float(self.coupling.melt_total[None])-float(self.coupling.melt_comp[None]))*c.mass_unit,
            'geometry_area_residual_cells': float(occupied.sum()-solid.sum()),
            'capacity_residual_max': float(np.max(np.abs(q[..., 0]+q[..., 1]-capacity))),
            'gcl_residual_max': float(np.max(np.abs(self.transport.gcl.to_numpy()))),
            'pressure_iterations_used': int(self.transport.iterations_used[None]),
            'capacity_min': float(capacity.min()), 'phase_min': float(phi.min(initial=0)), 'phase_max': float(phi.max(initial=0)),
            'mass_min_lattice': float(min(q[..., 0].min(), q[..., 1].min(), solid.min())),
            'temperature_water_min_C': float(tw[wet].min()*c.temperature_scale) if wet.any() else 0.0,
            'temperature_water_max_C': float(tw[wet].max(initial=0)*c.temperature_scale),
            'temperature_air_min_C': float(ta[gas].min()*c.temperature_scale) if gas.any() else 0.0,
            'temperature_air_max_C': float(ta[gas].max(initial=0)*c.temperature_scale),
            'temperature_ice_min_C': float(ts[solid > 1e-10].min()*c.temperature_scale) if (solid > 1e-10).any() else 0.0,
            'mach_max': float(np.linalg.norm(velocity[active], axis=-1).max(initial=0)*math.sqrt(3)),
            'thermal_fourier_water': c.thermal_weight('water')/4,
            'thermal_fourier_ice': c.thermal_weight('ice')/4,
            'thermal_fourier_air': c.thermal_weight('air')/4,
            'geometry_courant': geometry_courant,
            'geometry_courant_origin_bound': float(np.linalg.norm(body_velocity[:2])+abs(body_velocity[2])*c.nb),
            'phase_limiter_volume_cells': float(self.transport.limited.to_numpy().astype(float).sum()),
            'cut_cell_merge_water_volume_cells': float(self.merge.activity.to_numpy().astype(float).sum()),
            'thermal_limiter_energy_J_m': float(self.thermal.limited.to_numpy().astype(float).sum()*c.energy_unit),
            'snibb_phase_boundary_volume_cells': float(self.flow.boundary_phase.to_numpy().astype(float).sum()),
            'body_origin_x_m': float(g.origin[None][0]*c.dx), 'body_origin_y_m': float(g.origin[None][1]*c.dx),
            'body_angle_rad': float(g.angle[None]),
            'total_momentum_x': float((q[..., 4].sum()+body_p[0])*p_unit),
            'total_momentum_y': float((q[..., 5].sum()+body_p[1])*p_unit),
            'total_angular_momentum': angular*l_unit,
            'momentum_balance_residual_x': float(momentum_residual[0]*p_unit),
            'momentum_balance_residual_y': float(momentum_residual[1]*p_unit),
            'angular_balance_residual': float(momentum_residual[2]*l_unit),
            'gravity_impulse_y': float((self.budgets.gravity[None][1]-self.budgets.gravity_comp[None][1])*p_unit),
            'wall_impulse_y': float((self.budgets.wall[None][1]-self.budgets.wall_comp[None][1])*p_unit),
            'contact_impulse_y': float((self.rigid.contact_impulse[None][1]-self.rigid.contact_impulse_comp[None][1])*p_unit),
            'interface_transfer_momentum_y': float((self.rigid.phase_impulse[None][1]-self.rigid.phase_impulse_comp[None][1])*p_unit),
            'kinetic_energy_J_m': float(kinetic), 'gravity_potential_J_m': float(potential),
            'zero_mass_momentum_l1_kg_s': float(empty_momentum),
            'mechanical_energy_balance_validated': False,
            'phase_momentum_roundoff_x': float(phase_roundoff[0]),
            'phase_momentum_roundoff_y': float(phase_roundoff[1]),
            'phase_angular_momentum_roundoff': float(phase_roundoff[2]),
            'effective_shear_nu_max_m2_s': float(physical_nu.max(initial=0)),
            'rigid_iteration_residual': float(self.rigid.iteration_residual[None]),
            'subgrid_viscous_active_steps': int(self.rigid.subgrid.active_steps[None]),
            'subgrid_viscous_failure': int(self.rigid.subgrid.failure[None]),
            'subgrid_viscous_dissipated_energy_J_m': float(self.rigid.subgrid.dissipated_total[None]*c.mass_unit*(c.dx/c.dt)**2),
            'failure_code': int(self.coupling.failure[None]),
            'batch_failure_flags': int(self.health.flags[None]),
            'batch_capacity_residual_peak': float(self.health.capacity_peak[None]),
            'batch_mach_peak': float(self.health.mach_peak[None]),
            'finite': bool(np.isfinite(q).all() and np.isfinite(velocity).all() and np.isfinite(solid_heat).all()),
            'wall_seconds': self.wall_seconds,
            'mlups': self.steps*c.nx*c.ny/max(self.wall_seconds, 1e-30)/1e6,
        }

    def validate(self, diagnostics=None):
        """Current numerical/enthalpy guards, not an A-H physics certificate."""
        d = diagnostics or self.diagnostics()
        failures = []
        if not d['finite']: failures.append('NaN/Inf')
        if d['batch_failure_flags']: failures.append(f"per-step batch guard ({d['batch_failure_flags']})")
        if d['failure_code'] == 1: failures.append('no connected gas receiver for contraction')
        if d['failure_code'] == 2: failures.append('no connected receiver for interface torque')
        if d['mass_min_lattice'] < -1e-7: failures.append('negative phase mass')
        if abs(d['condensed_mass_relative_residual']) > self.cfg.mass_tolerance: failures.append('condensed mass budget')
        if abs(d['condensed_energy_relative_residual']) > self.cfg.energy_tolerance: failures.append('condensed energy budget')
        if d['capacity_residual_max'] > self.cfg.capacity_tolerance: failures.append('unclosed fluid capacity')
        if d['phase_min'] < -1e-6 or d['phase_max'] > 1+1e-5: failures.append('phase outside bounds')
        if d['mach_max'] > self.cfg.max_mach: failures.append('low Mach limit')
        if d['geometry_courant'] > 0.2: failures.append('geometry displacement limit')
        if failures: raise SimulationError(', '.join(failures))
        return d

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, config=json.dumps(asdict(self.cfg)), step=self.steps,
                            state_format=2,
                            state=self.state.q.to_numpy(), compensation=self.state.comp.to_numpy(),
                            solid=self.geometry.solid.to_numpy(), solid_comp=self.geometry.solid_comp.to_numpy(),
                            ice_heat=self.geometry.ice_heat.to_numpy(), ice_heat_comp=self.geometry.ice_heat_comp.to_numpy(),
                            capacity=self.geometry.capacity.to_numpy(), velocity=self.flow.velocity.to_numpy(),
                            pressure=self.flow.pressure.to_numpy(), origin=np.array(self.geometry.origin[None]),
                            angle=self.geometry.angle[None])
