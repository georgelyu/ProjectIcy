"""Analytic and conservation verification. Run with conda run -n icy python.

Only this new directory and the user-supplied PDF are used as task inputs.
All reference solutions below are derived directly from the stated PDEs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from thermal import ADIABATIC, DIRICHLET, PERIODIC, D2Q5, ThermalParameters


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def save_json(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + "\n")


def emit(name, value):
    print(json.dumps({"test": name, **value}, ensure_ascii=False, allow_nan=False), flush=True)


def check_finite(solver):
    data = solver.arrays()
    if not all(np.isfinite(value).all() for value in data.values()):
        raise AssertionError("Nonfinite thermal state")
    if data["liquid_fraction"].min() < -1e-6 or data["liquid_fraction"].max() > 1.000001:
        raise AssertionError("Invalid liquid fraction")
    return data


def moment_test():
    c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], float)
    w = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    M = np.array([[1, 1, 1, 1, 1], [0, 1, 0, -1, 0],
                  [0, 0, 1, 0, -1], [0, 1, 1, 1, 1],
                  [0, 1, -1, 1, -1]], float)
    worst = 0.0
    for H, T in ((0.8, 0.0), (-0.3, -0.6), (6.4, 1.4)):
        geq = w * T
        geq[0] += H - T
        expected = np.array([H, 0, 0, 0.4 * T, 0])
        worst = max(worst, float(np.max(np.abs(M @ geq - expected))))
    errors = [abs(w.sum() - 1), np.abs(w @ c).max(),
              np.abs(np.einsum("i,ia,ib->ab", w, c, c) - 0.2 * np.eye(2)).max(), worst]
    result = {"maximum_moment_residual": float(max(errors)),
              "matrix_rank": int(np.linalg.matrix_rank(M)),
              "passes": bool(max(errors) < 1e-14)}
    assert result["passes"]
    return result


def uniform_heat_test():
    p = ThermalParameters(c_s=0.5, c_l=1.0, latent=2.0, k_s=0.1, k_l=0.1)
    solver = D2Q5(8, 8, p)
    h0, heat_per_step, steps = -0.5, 0.002, 1800
    solver.initialize(h0)
    history = []
    max_t, max_f, max_h = 0.0, 0.0, 0.0
    for step in range(1, steps + 1):
        solver.add_uniform_source(heat_per_step)
        solver.diffuse()
        if step % 25 == 0:
            exact_h = h0 + step * float(np.float32(heat_per_step))
            exact_t = exact_h / p.c_s if exact_h < 0 else max(0.0, (exact_h - p.latent) / p.c_l)
            exact_f = min(1.0, max(0.0, exact_h / p.latent))
            data = check_finite(solver)
            t = float(data["temperature"].mean(dtype=np.float64))
            f = float(data["liquid_fraction"].mean(dtype=np.float64))
            h = float(data["enthalpy"].mean(dtype=np.float64))
            max_t, max_f, max_h = max(max_t, abs(t-exact_t)), max(max_f, abs(f-exact_f)), max(max_h, abs(h-exact_h))
            history.append([step, t, f, h, exact_t, exact_f])
    # Reverse the heat flux: a separate freezing branch checks the latent sign.
    for step in range(steps):
        solver.add_uniform_source(-heat_per_step)
        solver.diffuse()
    reverse_error = abs(solver.energy() / 64 - h0)
    result = {"max_temperature_error": max_t, "max_fraction_error": max_f,
              "max_enthalpy_error": max_h, "heat_then_cool_error": reverse_error,
              "passes": bool(max_t < 2e-5 and max_f < 2e-5 and reverse_error < 2e-5)}
    np.savez_compressed(RESULTS / "uniform_heating.npz", history=np.array(history))
    assert result["passes"], result
    return result


def diffusion_test(grids):
    rows, profiles = [], {}
    capacity, conductivity, final_fourier = 0.6, 0.1, 0.03
    for n in grids:
        p = ThermalParameters(capacity, capacity, 0.0, conductivity, conductivity, 0.4)
        solver = D2Q5(n, n, p)
        x = (np.arange(n) + 0.5) / n
        wave = np.sin(2 * np.pi * x[:, None]) * np.cos(2 * np.pi * x[None, :])
        initial = 0.3 * wave
        solver.initialize(capacity * initial)
        e0 = solver.energy()
        steps = round(final_fourier * n*n / (conductivity / capacity))
        for step in range(steps):
            solver.diffuse()
        actual_fo = steps * conductivity / capacity / (n*n)
        exact = initial * np.exp(-8 * np.pi**2 * actual_fo)
        data = check_finite(solver)
        error = float(np.sqrt(np.mean((data["temperature"] - exact)**2)))
        amplitude = float(np.sum(data["temperature"] * wave) / np.sum(wave**2))
        measured_alpha = -math.log(amplitude / 0.3) * n*n / (8*np.pi**2*steps)
        row = {"n": n, "steps": steps, "l2_error": error,
               "relative_diffusivity_error": abs(measured_alpha / (conductivity/capacity) - 1),
               "energy_drift_per_cell": abs(solver.energy() - e0)/(n*n)}
        if rows:
            row["order"] = math.log(rows[-1]["l2_error"]/error) / math.log(n/rows[-1]["n"])
        rows.append(row)
        emit("diffusion_grid", row)
        profiles[f"T_{n}"] = data["temperature"]
    result = {"rows": rows, "passes": bool(rows[-1]["relative_diffusivity_error"] < 0.003
                                             and rows[-1]["l2_error"] < rows[0]["l2_error"] / 3)}
    np.savez_compressed(RESULTS / "diffusion.npz", **profiles)
    assert result["passes"], result
    return result


def manufactured_test(grids):
    """Transient C(x,y)T_t=div(k(x,y)grad(T))+Q, all terms analytic."""
    rows=[]
    for n in grids:
        x=(np.arange(n)+0.5)/n
        xx,yy=np.meshgrid(x,x,indexing="ij")
        sx,cx,sy,cy=np.sin(2*np.pi*xx),np.cos(2*np.pi*xx),np.sin(2*np.pi*yy),np.cos(2*np.pi*yy)
        capacity=0.8+0.2*sx*cy
        k=1+0.3*cx
        dt=0.08/n**2
        solver=D2Q5(n,n,ThermalParameters(0.8,0.8,0,0.08,0.08,0.4))
        for field,value in ((solver.c_s,capacity),(solver.c_l,capacity),
                            (solver.k_s,k*0.08),(solver.k_l,k*0.08)):
            field.from_numpy(value.astype(np.float32))
        base=0.2*sx*sy
        # k_x*T_x = -0.2*1.2*pi^2 sin(2pi x)cos(2pi x)sin(2pi y).
        qcoef=(-capacity+8*np.pi**2*k)*base+0.2*1.2*np.pi**2*sx*cx*sy
        source=ti.field(ti.f32,(n,n))
        source.from_numpy(qcoef.astype(np.float32))
        solver.initialize(capacity*base)
        steps=round(0.03/dt)
        for step in range(steps):
            a,b=step*dt,(step+1)*dt
            mid=(a+b)/2
            solver.add_source_field(source,math.exp(-a)-math.exp(-mid))
            solver.diffuse()
            solver.add_source_field(source,math.exp(-mid)-math.exp(-b))
        exact=base*math.exp(-steps*dt)
        temp=check_finite(solver)["temperature"]
        error=float(np.sqrt(np.mean((temp-exact)**2)))
        row={"n":n,"steps":steps,"temperature_l2_error":error}
        if rows:row["order"]=math.log(rows[-1]["temperature_l2_error"]/error)/math.log(n/rows[-1]["n"])
        rows.append(row)
        emit("manufactured_grid",row)
    result={"rows":rows,"passes":bool(rows[-1]["temperature_l2_error"]<5e-4
                                         and rows[-1]["temperature_l2_error"]<rows[0]["temperature_l2_error"]/3)}
    assert result["passes"],result
    return result


def advection_diffusion_test(grids):
    rows=[]
    capacity,conductivity=0.6,0.1
    alpha=conductivity/capacity
    for n in grids:
        solver=D2Q5(n,n,ThermalParameters(capacity,capacity,0,conductivity,conductivity,0.4))
        x=(np.arange(n)+0.5)/n
        xx,yy=np.meshgrid(x,x,indexing="ij")
        initial=0.3*np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)
        vx,vy=8*alpha/n,-4*alpha/n
        solver.set_rigid_velocity(vx,vy,0.0,0.0,0.0)
        solver.initialize(capacity*initial)
        steps=round(0.03*n*n/alpha)
        for step in range(steps):solver.step(advect=True)
        exact=0.3*np.sin(2*np.pi*(xx-vx*steps/n))*np.cos(2*np.pi*(yy-vy*steps/n))*np.exp(-8*np.pi**2*steps*alpha/n**2)
        data=check_finite(solver)
        error=float(np.sqrt(np.mean((data["temperature"]-exact)**2)))
        row={"n":n,"steps":steps,"temperature_l2_error":error,"cell_peclet":abs(vx)/alpha}
        if rows:row["order"]=math.log(rows[-1]["temperature_l2_error"]/error)/math.log(n/rows[-1]["n"])
        rows.append(row)
        emit("advection_diffusion_grid",row)
    result={"rows":rows,"passes":bool(rows[-1]["temperature_l2_error"]<2e-4
                                         and rows[-1]["temperature_l2_error"]<rows[0]["temperature_l2_error"]/3)}
    assert result["passes"],result
    return result


def layer_test(grids):
    rows, profiles = [], {}
    for n in grids:
        p = ThermalParameters(0.6, 0.6, 0.0, 0.1, 0.1, 0.4)
        solver = D2Q5(n, 4, p, (DIRICHLET, DIRICHLET, PERIODIC, PERIODIC))
        x = (np.arange(n) + 0.5) / n
        capacity = np.broadcast_to(np.where(x[:, None] < 0.5, 0.5, 1.0), (n,4)).copy()
        k = np.broadcast_to(np.where(x[:, None] < 0.5, 0.2, 0.05), (n,4)).copy()
        solver.c_s.from_numpy(capacity.astype(np.float32))
        solver.c_l.from_numpy(capacity.astype(np.float32))
        solver.k_s.from_numpy(k.astype(np.float32))
        solver.k_l.from_numpy(k.astype(np.float32))
        heat_flux = (1.0 - 0.0)/(0.5/0.2 + 0.5/0.05)
        exact = np.where(x <= 0.5, 1.0 - heat_flux*x/0.2,
                         heat_flux*(1.0-x)/0.05)
        # Start from the independently derived exact profile and verify stationarity.
        solver.initialize(capacity * exact[:, None])
        e0 = solver.energy()
        steps = 6*n*n
        for step in range(steps):
            solver.diffuse((1.0, 0.0, 0.0, 0.0))
        data = check_finite(solver)
        temp = data["temperature"].mean(axis=1, dtype=np.float64)
        slope_left = np.polyfit(x[2:n//2-2], temp[2:n//2-2], 1)[0]
        slope_right = np.polyfit(x[n//2+2:-2], temp[n//2+2:-2], 1)[0]
        qleft, qright = -0.2*slope_left, -0.05*slope_right
        row = {"n": n, "steps": steps, "max_temperature_error": float(np.max(abs(temp-exact))),
               "left_flux_relative_error": abs(qleft/heat_flux-1),
               "right_flux_relative_error": abs(qright/heat_flux-1),
               "energy_balance_per_cell": abs(solver.energy()-e0-solver.boundary_heat())/(n*4)}
        rows.append(row)
        profiles[f"T_{n}"] = temp
        emit("layer_grid", row)
    result = {"rows": rows, "passes": bool(rows[-1]["max_temperature_error"] < 0.004
                                             and rows[-1]["left_flux_relative_error"] < 0.02
                                             and rows[-1]["right_flux_relative_error"] < 0.02)}
    np.savez_compressed(RESULTS / "layer.npz", **profiles)
    assert result["passes"], result
    return result


def bisect(function, lo, hi, iterations=90):
    flo = function(lo)
    if flo * function(hi) >= 0:
        raise ValueError("Root not bracketed")
    for _ in range(iterations):
        mid = (lo + hi)/2
        fm = function(mid)
        if flo*fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo+hi)/2


def similarity_root(cs, cl, latent, ks, kl, hot, cold):
    al, ass = kl/cl, ks/cs
    def equation(lam):
        eta = lam*math.sqrt(al/ass)
        left = kl*hot*math.exp(-lam*lam)/(math.sqrt(math.pi*al)*math.erf(lam))
        right = ks*(-cold)*math.exp(-eta*eta)/(math.sqrt(math.pi*ass)*math.erfc(eta))
        return left-right-latent*lam*math.sqrt(al)
    return bisect(equation, 1e-7, 5.0)


def similarity_temperature(x, t, lam, cs, cl, ks, kl, hot, cold):
    al, ass = kl/cl, ks/cs
    front = 2*lam*math.sqrt(al*t)
    x = np.asarray(x)
    erf_l = np.array([math.erf(float(xx)/(2*math.sqrt(al*t))) for xx in x.flat]).reshape(x.shape)
    erfc_s = np.array([math.erfc(float(xx)/(2*math.sqrt(ass*t))) for xx in x.flat]).reshape(x.shape)
    tl = hot*(1-erf_l/math.erf(lam))
    ts = cold + (-cold)*erfc_s/math.erfc(lam*math.sqrt(al/ass))
    return np.where(x < front, tl, ts), front


def cell_average_similarity(n, length, t, lam, cs, cl, latent, ks, kl, hot, cold):
    # Resolve the latent discontinuity by integrating each phase separately.
    nodes, weights = np.polynomial.legendre.leggauss(12)
    edges = np.arange(n+1)*length/n
    front = 2*lam*math.sqrt((kl/cl)*t)
    h = np.zeros(n)
    for i, (a,b) in enumerate(zip(edges[:-1], edges[1:])):
        for lo, hi, liquid in ((a,min(b,front),True), (max(a,front),b,False)):
            if hi > lo:
                x = (lo+hi)/2 + (hi-lo)/2 * nodes
                temp, _ = similarity_temperature(x,t,lam,cs,cl,ks,kl,hot,cold)
                energy = cl*temp+latent if liquid else cs*temp
                h[i] += (hi-lo)/2*np.dot(weights,energy)/(b-a)
    return h


def stefan_test(grids, two_phase=False):
    cs, cl, latent, ks, kl = 0.5, 1.0, 2.0, 2.0, 1.0
    hot, cold, length, t0, tend = 1.0, (-0.25 if two_phase else 0.0), 1.0, 0.002, 0.02
    lam = similarity_root(cs,cl,latent,ks,kl,hot,cold)
    alpha_lattice = 0.08
    rows, profiles = [], {}
    for n in grids:
        dx, dt = length/n, alpha_lattice*(length/n)**2
        p = ThermalParameters(cs, cl, latent, ks*dt/dx**2, kl*dt/dx**2, 0.4)
        solver = D2Q5(n,4,p,(DIRICHLET,DIRICHLET,PERIODIC,PERIODIC))
        initial = cell_average_similarity(n,length,t0,lam,cs,cl,latent,ks,kl,hot,cold)
        solver.initialize(initial[:,None])
        e0 = solver.energy()
        steps = round((tend-t0)/dt)
        times, numerical_front, exact_front = [], [], []
        for step in range(steps):
            t_mid = t0 + (step+0.5)*dt
            boundary_right = cold + (-cold)*math.erfc(length/(2*math.sqrt((ks/cs)*t_mid)))/math.erfc(lam*math.sqrt((kl/cl)/(ks/cs)))
            solver.diffuse((hot,boundary_right,0.0,0.0))
            if (step+1) % max(1,steps//40) == 0 or step+1 == steps:
                time_now = t0+(step+1)*dt
                fraction = solver.liquid.to_numpy().mean(axis=1,dtype=np.float64)
                times.append(time_now)
                numerical_front.append(float(fraction.sum()*dx))
                exact_front.append(2*lam*math.sqrt((kl/cl)*time_now))
        data = check_finite(solver)
        x = (np.arange(n)+0.5)*dx
        final_time = t0+steps*dt
        exact_t, exact_s = similarity_temperature(x,final_time,lam,cs,cl,ks,kl,hot,cold)
        temp = data["temperature"].mean(axis=1,dtype=np.float64)
        front = data["liquid_fraction"].mean(axis=1,dtype=np.float64).sum()*dx
        # Front by liquid volume is independent of choosing an f_l=0.5 contour.
        row = {"n":n,"steps":steps,"lambda":lam,"time":final_time,
               "front":float(front),"exact_front":exact_s,
               "relative_front_error":abs(float(front)/exact_s-1),
               "temperature_l2_error":float(np.sqrt(np.mean((temp-exact_t)**2))),
               "min_temperature":float(temp.min()),"max_temperature":float(temp.max()),
               "energy_balance_relative":abs(solver.energy()-e0-solver.boundary_heat())/max(abs(solver.boundary_heat()),1e-20)}
        exact_h_final=cell_average_similarity(n,length,final_time,lam,cs,cl,latent,ks,kl,hot,cold)
        exact_heat=(exact_h_final.sum()-initial.sum())*4
        row["boundary_heat_relative_error_vs_analytic"]=abs(solver.boundary_heat()/exact_heat-1)
        if rows:
            row["front_order"] = math.log(rows[-1]["relative_front_error"]/row["relative_front_error"])/math.log(n/rows[-1]["n"])
        rows.append(row)
        profiles.update({f"x_{n}":x,f"T_{n}":temp,f"T_exact_{n}":exact_t,
                         f"times_{n}":np.array(times),f"front_{n}":np.array(numerical_front),
                         f"front_exact_{n}":np.array(exact_front)})
        emit("two_phase_stefan_grid" if two_phase else "one_phase_stefan_grid",row)
    result = {"parameters":{"c_s":cs,"c_l":cl,"latent":latent,"k_s":ks,"k_l":kl,
                            "hot":hot,"cold":cold,"initial_time":t0},
              "rows":rows,"passes":bool(rows[-1]["relative_front_error"] < 0.025
                                           and rows[-1]["temperature_l2_error"] < 0.02
                                           and rows[-1]["relative_front_error"] < rows[0]["relative_front_error"]
                                           and rows[-1]["energy_balance_relative"] < 5e-5)}
    name = "stefan_two_phase" if two_phase else "stefan_one_phase"
    np.savez_compressed(RESULTS/f"{name}.npz",**profiles)
    assert result["passes"],result
    return result


def transport_test(grids, rotation=False):
    rows, snapshots = [], {}
    for n in grids:
        p = ThermalParameters(0.5,1.0,5.0,0.15,0.05,0.4)
        solver = D2Q5(n,n,p)
        x = np.arange(n)+0.5-n/2
        xx,yy = np.meshgrid(x,x,indexing="ij")
        initial_solid = (((xx+0.10*n)/(0.18*n))**2 + ((yy-0.05*n)/(0.11*n))**2 < 1).astype(float)
        h0 = p.latent*(1-initial_solid)
        solver.initialize(h0)
        e0, solid0 = solver.energy(), initial_solid.sum()
        if rotation:
            steps = 4*n
            omega = (np.pi/2)/steps
            solver.set_rigid_velocity(0.0,0.0,float(omega),n/2,n/2)
            exact = np.rot90(initial_solid)
        else:
            steps = 5*n
            solver.set_rigid_velocity(0.2,0.0,0.0,n/2,n/2)
            exact = initial_solid
        for step in range(steps):
            solver.step(advect=True)
        data = check_finite(solver)
        solid = 1-data["liquid_fraction"].astype(float)
        cx = float(np.sum(xx*solid)/solid.sum()/n)
        cy = float(np.sum(yy*solid)/solid.sum()/n)
        cx_ref = float(np.sum(xx*exact)/exact.sum()/n)
        cy_ref = float(np.sum(yy*exact)/exact.sum()/n)
        inertia=float(np.sum(((xx-cx*n)**2+(yy-cy*n)**2)*solid))
        inertia_ref=float(np.sum(((xx-cx_ref*n)**2+(yy-cy_ref*n)**2)*exact))
        row = {"n":n,"steps":steps,"relative_solid_volume_error":abs(solid.sum()/solid0-1),
               "relative_energy_error":abs(solver.energy()-e0)/abs(e0),
               "shape_l1_per_solid_volume":float(np.sum(abs(solid-exact))/solid0),
               "centroid_error_over_domain":math.hypot(cx-cx_ref,cy-cy_ref),
               "relative_geometric_inertia_error":abs(inertia/inertia_ref-1),
               "maximum_absolute_temperature":float(np.max(abs(data["temperature"]))) }
        if rows:
            row["shape_order"] = math.log(rows[-1]["shape_l1_per_solid_volume"]/row["shape_l1_per_solid_volume"])/math.log(n/rows[-1]["n"])
        rows.append(row)
        snapshots.update({f"solid_{n}":solid,f"exact_{n}":exact})
        emit("rotation_grid" if rotation else "translation_grid",row)
    result = {"rows":rows,"passes":bool(rows[-1]["relative_solid_volume_error"] < 2e-5
                                          and rows[-1]["maximum_absolute_temperature"] < 1e-4
                                          and rows[-1]["centroid_error_over_domain"] < 0.004
                                          and rows[-1]["shape_l1_per_solid_volume"] < rows[0]["shape_l1_per_solid_volume"])}
    np.savez_compressed(RESULTS/("rotation.npz" if rotation else "translation.npz"),**snapshots)
    assert result["passes"],result
    return result


def closed_melting_test(n=80):
    p = ThermalParameters(0.5,1.0,5.0,0.2,0.1,0.4)
    solver = D2Q5(n,n,p,(ADIABATIC,)*4)
    x = (np.arange(n)+0.5)/n-0.5
    solid = (abs(x[:,None])<0.25)&(abs(x[None,:])<0.25)
    initial = np.where(solid,-0.25,6.0)
    solver.initialize(initial)
    e0 = solver.energy()
    expected_solid = n*n-e0/p.latent
    snapshots = {"initial_temperature":solver.temperature.to_numpy(),"initial_solid":solid}
    history = []
    max_steps = 180000
    for step in range(max_steps):
        solver.diffuse()
        if (step+1)%1000==0:
            data = check_finite(solver)
            actual_solid = float(np.sum(1-data["liquid_fraction"],dtype=np.float64))
            max_t = float(np.max(abs(data["temperature"])))
            history.append([step+1,actual_solid,max_t,solver.energy()])
            if step+1 in (1000,5000,15000,40000):
                snapshots[f"T_{step+1}"]=data["temperature"]
                snapshots[f"solid_{step+1}"]=1-data["liquid_fraction"]
            if max_t < 2e-4:
                break
    snapshots.update({"final_temperature":data["temperature"],"final_solid":1-data["liquid_fraction"],
                      "history":np.array(history)})
    result = {"n":n,"steps":step+1,"initial_solid_volume":float(solid.sum()),
              "predicted_final_solid_volume":expected_solid,"final_solid_volume":actual_solid,
              "relative_final_volume_error":abs(actual_solid/expected_solid-1),
              "relative_energy_error":abs(solver.energy()-e0)/abs(e0),"max_final_temperature":max_t,
              "passes":bool(max_t<3e-4 and abs(actual_solid/expected_solid-1)<0.002
                             and abs(solver.energy()-e0)/abs(e0)<2e-5)}
    np.savez_compressed(RESULTS/"closed_melting.npz",**snapshots)
    assert result["passes"],result
    return result


def translating_melt_test(grids):
    """Galilean check: a melting ice block and its surrounding water translate.

    This checks the thermal/melting operator with given motion, not the force
    balance of a freely rising rigid body. The PDE predicts an exact translation
    of the stationary reference solution in this periodic domain.
    """
    rows=[]
    snapshots={}
    for n in grids:
        p=ThermalParameters(0.5,1.0,5.0,0.2,0.1,0.4)
        reference=D2Q5(n,n,p)
        moving=D2Q5(n,n,p)
        x=(np.arange(n)+0.5)/n-0.5
        xx,yy=np.meshgrid(x,x,indexing="ij")
        shape=(abs(xx)<0.20)&(abs(yy)<0.25)
        initial=np.where(shape,-0.25,6.0)
        reference.initialize(initial)
        moving.initialize(initial)
        e0=moving.energy()
        steps=round(0.04*n*n/p.k_l)
        shift=n//4
        moving.set_rigid_velocity(shift/steps,0.0,0.0,0.0,0.0)
        for step in range(steps):
            reference.diffuse()
            moving.step(advect=True)
        ref=check_finite(reference)
        data=check_finite(moving)
        target_t=np.roll(ref["temperature"],shift,axis=0)
        target_f=np.roll(ref["liquid_fraction"],shift,axis=0)
        solid0=float(shape.sum())
        solid_ref=float(np.sum(1-ref["liquid_fraction"],dtype=np.float64))
        solid_moving=float(np.sum(1-data["liquid_fraction"],dtype=np.float64))
        row={"n":n,"steps":steps,"initial_solid_volume":solid0,
             "reference_solid_volume":solid_ref,"moving_solid_volume":solid_moving,
             "relative_solid_volume_difference":abs(solid_moving/solid_ref-1),
             "temperature_l2_error":float(np.sqrt(np.mean((data["temperature"]-target_t)**2))),
             "fraction_l1_error":float(np.mean(abs(data["liquid_fraction"]-target_f))),
             "relative_energy_error":abs(moving.energy()-e0)/abs(e0)}
        rows.append(row)
        snapshots.update({f"temperature_{n}":data["temperature"],f"solid_{n}":1-data["liquid_fraction"],
                          f"target_temperature_{n}":target_t,f"target_solid_{n}":1-target_f})
        emit("translating_melt_grid",row)
    result={"interpretation":"Prescribed comoving water/ice velocity; no hydrodynamic force solver.",
            "rows":rows,"passes":bool(rows[-1]["relative_solid_volume_difference"]<0.04
                                         and rows[-1]["fraction_l1_error"]<rows[0]["fraction_l1_error"]
                                         and rows[-1]["relative_energy_error"]<2e-5)}
    np.savez_compressed(RESULTS/"translating_melt.npz",**snapshots)
    assert result["passes"],result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",default="all",choices=["all","core","motion","closed","moments","uniform","diffusion","layers","stefan","manufactured","advection_diffusion","translating_melt"])
    parser.add_argument("--grids",nargs="+",type=int,default=[64,128,256])
    parser.add_argument("--arch",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ti.init(arch=ti.cuda if args.arch=="cuda" else ti.cpu,default_fp=ti.f32,
            default_ip=ti.i32,offline_cache=False,random_seed=0,fast_math=False)
    start=time.perf_counter()
    suite={"metadata":{"taichi_version":".".join(map(str,ti.__version__)),"arch":args.arch,
                       "gpu_precision":"f32","host_diagnostics_precision":"float64","grids":args.grids}}
    tests={"moments":moment_test,"uniform":uniform_heat_test,
           "diffusion":lambda:diffusion_test(args.grids),"layers":lambda:layer_test(args.grids),
           "stefan_one_phase":lambda:stefan_test(args.grids),
           "stefan_two_phase":lambda:stefan_test(args.grids,True),
           "manufactured":lambda:manufactured_test(args.grids),
           "advection_diffusion":lambda:advection_diffusion_test(args.grids),
           "translation":lambda:transport_test(args.grids),"rotation":lambda:transport_test(args.grids,True),
           "closed":closed_melting_test,"translating_melt":lambda:translating_melt_test(args.grids)}
    names=list(tests)
    if args.suite=="core":names=names[:6]
    elif args.suite=="motion":names=["translation","rotation"]
    elif args.suite=="stefan":names=["stefan_one_phase","stefan_two_phase"]
    elif args.suite!="all":names=[args.suite]
    for name in names:
        t0=time.perf_counter()
        result=tests[name]()
        result["runtime_seconds"]=time.perf_counter()-t0
        suite[name]=result
        save_json(name,result)
        emit(name,result)
    suite["runtime_seconds"]=time.perf_counter()-start
    save_json(f"suite_{args.suite}",suite)
    print("All selected verification checks passed.",flush=True)


if __name__=="__main__":
    main()
