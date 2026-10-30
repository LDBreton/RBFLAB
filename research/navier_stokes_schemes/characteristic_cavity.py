"""Bounded, restartable first-order characteristic mixed-LHI cavity study.

Only interior arrival sites are traced. Feet use the nodal advector U, and
the old mixed field is evaluated with the arrival owner's frozen stencil.
Wall data enter the new algebraic constraint directly. No clipping or
exterior continuation is permitted. All global algebra is Float64.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import splu


def input_hashes(source, baseline, parameters, initial):
    """Hash only the owning numerical inputs, never dependencies or caches."""
    names = ('A', 'B', 'SL', 'PY', 'PB', 'PL', 'DxU', 'DxB', 'DxL',
             'DyU', 'DyB', 'DyL')
    paths = [(f'corrected/{name}', Path(source)/'maps'/(name+'.npz')) for name in names]
    paths += [(f'raw/{name}', Path(baseline)/(name+'.npz')) for name in names]
    paths += [('parameters', Path(parameters)), ('initial', Path(initial))]
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths}


class DepartureOutside(ValueError):
    """A rejected step, retaining the actual offending feet for the report."""
    def __init__(self, step, time, feet, outside):
        self.record = dict(step=step, time=time, status="departure_outside",
                           exterior_count=int(outside.sum()),
                           owners=np.flatnonzero(outside).tolist(),
                           feet=feet[outside].tolist())
        super().__init__(f"Characteristic foot left physical domain at step {step}")


def evolve(problem, evaluator, dt, steps, *, scheme="characteristic", start_step=1,
           on_step=None, speed_bound=10.):
    """Advance a bounded chunk; reuse one LU for all characteristic steps.

    ``evaluator.evaluate(U,L,g,feet)`` returns owner-ordered Nx2 velocity.
    The optional ``be_oseen`` comparison uses first-order history/advection
    at every step. The saved historical BDF2 engine is a separate comparison.
    ``on_step(record,U,L)`` is called after each finite accepted state.
    """
    if not np.isfinite(dt) or dt <= 0 or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("dt must be positive and steps a nonnegative integer")
    if not isinstance(start_step, (int, np.integer)) or start_step < 1:
        raise ValueError("start_step must be a positive integer")
    if scheme not in ("characteristic", "be_oseen"):
        raise ValueError("Unsupported first-order scheme")
    p = problem
    U, L = (np.asarray(z, dtype=float).ravel().copy() for z in p.initial())
    n = len(U)
    I = eye(n, format="csc")
    matrix = bmat([[p.A, -p.M], [I/dt, I]], format="csc")
    solver = splu(matrix) if scheme == "characteristic" else None
    records = []
    reference = np.asarray(getattr(p, "_stability_reference", U)).ravel()
    bounds = np.asarray(p.bounds)
    if bounds.shape != (2, 2) or len(U) != 2*len(p.points):
        raise ValueError("Expected two-dimensional interior node-major velocity")
    factorizations = int(solver is not None)
    for step in range(start_step, start_step + steps):
        t, tn = (step-1)*dt, step*dt
        old_g, g = p.boundary(t), p.boundary(tn)
        rhs = np.r_[-p.B@g, p.force(tn)+U/dt]
        foot_min = foot_max = None
        if scheme == "characteristic":
            feet = np.asarray(p.points)-dt*U.reshape(-1, 2)
            outside = np.any((feet < bounds[:, 0]) | (feet > bounds[:, 1]), axis=1)
            if outside.any():
                raise DepartureOutside(step, tn, feet, outside)
            if not np.isfinite(feet).all():
                raise FloatingPointError(f"Nonfinite departure at step {step}")
            departure = np.asarray(evaluator.evaluate(U, L, old_g, feet))
            if departure.shape != (len(p.points), 2) or not np.isfinite(departure).all():
                raise FloatingPointError(f"Invalid mixed departure velocity at step {step}")
            rhs[n:] = p.force(tn)+departure.ravel()/dt
            foot_min, foot_max = feet.min(axis=0).tolist(), feet.max(axis=0).tolist()
        else:
            from .coupled import _transport
            Vx, Vy = _transport(U)
            OU, OL = Vx@p.DxU+Vy@p.DyU, Vx@p.DxL+Vy@p.DyL
            matrix = bmat([[p.A, -p.M], [I/dt+OU, I+OL]], format="csc")
            rhs[n:] -= Vx@(p.DxB@g)+Vy@(p.DyB@g)
            solver = splu(matrix)
            factorizations += 1
        state = solver.solve(rhs)
        new, new_l = np.split(state, 2)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite first-order state at step {step}")
        dx = p.DxU@new+p.DxB@g+p.DxL@new_l
        dy = p.DyU@new+p.DyB@g+p.DyL@new_l
        speeds = np.linalg.norm(new.reshape(-1, 2), axis=1)
        row = dict(step=int(step), time=tn, scheme=scheme,
                   solved_residual_inf=float(np.max(abs(matrix@state-rhs))),
                   constraint_residual_inf=float(np.max(abs(p.A@new-p.M@new_l+p.B@g))),
                   nodal_mapped_divergence_inf=float(np.max(abs(dx[::2]+dy[1::2]))),
                   velocity_inf=float(np.max(abs(new))), max_nodal_speed=float(speeds.max()),
                   kinetic_mean=float(.5*np.mean(speeds**2)),
                   perturbation_velocity_inf=float(np.max(abs(new-reference))),
                   exterior_count=0, foot_min=foot_min, foot_max=foot_max)
        U, L = new, new_l
        records.append(row)
        if on_step is not None:
            on_step(row, U, L)
        if speeds.max() > speed_bound:
            raise RuntimeError(f"Numerical speed bound {speed_bound} exceeded at step {step}")
    final_time = (start_step+steps-1)*dt
    gradient = p.PY@U+p.PB@p.boundary(final_time)+p.PL@L
    return dict(U=U, L=L, gradient=np.asarray(gradient).ravel(), diagnostics=records,
                next_step=start_step+steps, factorizations=factorizations)


def advance(source, cells, output, initial, baseline, parameters, cache,
            scheme, dt, control, steps, end, amplitude=1e-3, ranks=8):
    """Run at most ``steps`` foreground steps, preserving last accepted fields."""
    from .cavity_accuracy_verify import problem
    from .cavity_candidates import reference_at_nodal_lines
    p = problem(Path(source), cells)
    if control == "perturb" and (not np.isfinite(amplitude) or amplitude <= 0):
        raise ValueError("Perturbation amplitude must be finite and positive")
    initial = Path(initial)
    saved = np.load(initial)
    Uref, Lref = saved["U"].copy(), saved["L"].copy()
    if control == "pressure":
        Uref[:] = 0
        Lref = np.column_stack((p.points[:, 1], p.points[:, 0])).ravel()
        p.g0 = np.zeros_like(p.g0)
        p.force = lambda t: Lref.copy()
    case = Path(output)/f"{control}_{scheme}_{dt}"
    case.mkdir(parents=True, exist_ok=True)
    configuration = dict(source=str(Path(source).resolve()), cells=cells,
        initial=str(initial.resolve()), baseline=str(baseline), parameters=str(parameters),
        scheme=scheme, dt=dt, control=control, end=end, amplitude=amplitude,
        bounds=p.bounds.tolist(), nu=float(p.nu), reynolds=1/float(p.nu),
        input_sha256=input_hashes(source, baseline, parameters, initial),
        departure="x-dt*U; same arrival-owner 37-site mixed field",
        exterior_policy="reject; no clipping or extension", scope="nodal diagnostics only")
    config_path = case/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Checkpoint configuration mismatch")
    config_path.write_text(json.dumps(configuration, indent=2))
    if (case/"checkpoint.npz").exists():
        with np.load(case/"checkpoint.npz") as cp:
            U, L, start = cp["U"].copy(), cp["L"].copy(), int(cp["next_step"])
    elif control == "startup":
        U, L, start = np.zeros_like(Uref), np.zeros_like(Lref), 1
    else:
        mode = saved["mode"].real if control == "perturb" else np.zeros(2*len(Uref))
        a = amplitude if control == "perturb" else 0.
        U, L, start = Uref+a*mode[:len(Uref)], Lref+a*mode[len(Uref):], 1
    p.initial = lambda: (U.copy(), L.copy())
    p._stability_reference = Uref
    total = round(end/dt)
    if abs(total*dt-end) > 1e-12 or steps < 1:
        raise ValueError("end must be an integer multiple of dt and chunk steps positive")
    count = min(steps, total-start+1)
    if count <= 0:
        return
    history_path = case/"diagnostics.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    trace_path = case/"trajectory.npz"
    if trace_path.exists():
        with np.load(trace_path) as trace:
            trace_steps, trace_U, trace_L = list(trace["steps"]), list(trace["U"]), list(trace["L"])
        if int(trace_steps[-1]) != start-1:
            raise ValueError("Trajectory/checkpoint step mismatch")
    elif start > 1:
        raise ValueError("Checkpoint lacks per-step trajectory; preserve as pilot and use a fresh case")
    else:
        trace_steps, trace_U, trace_L = [0], [U.copy()], [L.copy()]
    latest = dict(U=U, L=L, next_step=start)
    def checkpoint():
        checkpoint_temp = case/"checkpoint_pending.npz"
        trajectory_temp = case/"trajectory_pending.npz"
        np.savez(checkpoint_temp, points=p.points, **latest)
        np.savez(trajectory_temp, steps=np.asarray(trace_steps), times=np.asarray(trace_steps)*dt,
                 U=np.asarray(trace_U), L=np.asarray(trace_L))
        trajectory_temp.replace(trace_path)
        checkpoint_temp.replace(case/"checkpoint.npz")
        history_path.write_text(json.dumps(history, indent=2))
    def accept(row, u, ell):
        history.append(row)
        trace_steps.append(row["step"])
        trace_U.append(u.copy())
        trace_L.append(ell.copy())
        latest.update(U=u.copy(), L=ell.copy(), next_step=row["step"]+1)
        if row["step"] % 100 == 0:
            checkpoint()
            print(json.dumps(dict(status="progress", step=row["step"], time=row["time"],
                                  control=control, dt=dt, speed=row["max_nodal_speed"])), flush=True)
    evaluator = None
    if scheme == "characteristic":
        from .characteristic_reconstruction import CorrectedCharacteristicReconstruction
        from .nullspace_lifting import NullLifting
        from .pressure_lifting import load_problem
        model = NullLifting(load_problem(cells, Path(baseline)))
        coefficients = np.load(parameters)
        reconstructed = model.matrices(coefficients, full=True)
        # The evaluator and time step must describe exactly the same field.
        for name in ("A", "B", "M", "PY", "PB", "PL", "DxU", "DxB", "DxL", "DyU", "DyB", "DyL"):
            saved_map, rebuilt_map = getattr(p, name), getattr(reconstructed, name)
            if saved_map.shape != rebuilt_map.shape:
                raise ValueError(f"Saved/evaluator map shape mismatch for {name}")
            difference = saved_map-rebuilt_map
            error = float(np.max(abs(difference.data))) if difference.nnz else 0.
            scale = max(1., float(np.max(abs(saved_map.data))) if saved_map.nnz else 0.)
            if error > 1e-11*scale:
                raise ValueError(f"Saved/evaluator field mismatch for {name}: {error:g}")
        evaluator = CorrectedCharacteristicReconstruction(model, coefficients, Path(cache), ranks=ranks)
    started = time.perf_counter()
    failure = None
    try:
        evolve(p, evaluator, dt, count, scheme=scheme, start_step=start, on_step=accept)
    except Exception as exc:
        failure = getattr(exc, "record", dict(status="failed", error=str(exc)))
        failure["error"] = str(exc)
    finally:
        if evaluator is not None and hasattr(evaluator, "close"):
            evaluator.close()
    checkpoint()
    U, L = latest["U"], latest["L"]
    t = (latest["next_step"]-1)*dt
    gradient = p.PY@U+p.PB@p.boundary(t)+p.PL@L
    np.savez(case/"fields.npz", points=p.points, U=U, L=L, gradient=gradient)
    row = dict(status="completed" if latest["next_step"]>total else "checkpoint",
        time=t, next_step=latest["next_step"], scheme=scheme, control=control, dt=dt,
        velocity_change_inf=float(np.max(abs(U-Uref))), seconds=time.perf_counter()-started,
        linear_factorizations_this_chunk=1 if scheme == "characteristic" else latest["next_step"]-start,
        factorization_reuse_scope="within this foreground chunk",
        diagnostics_steps=len(history), failure=failure)
    if history:
        row.update(peak_nodal_speed=max(d["max_nodal_speed"] for d in history),
                   max_constraint_residual=max(d["constraint_residual_inf"] for d in history),
                   max_solved_residual=max(d["solved_residual_inf"] for d in history),
                   max_nodal_mapped_divergence=max(d["nodal_mapped_divergence_inf"] for d in history))
    if failure:
        row["status"] = failure["status"]
    if control == "pressure":
        row["pressure_gradient_error_inf"] = float(np.max(abs(gradient-Lref)))
    else:
        row["ghia"] = reference_at_nodal_lines(p.points, U)
    if control == "perturb":
        row["raw_root_distance_over_amplitude"] = row["velocity_change_inf"]/amplitude
    (case/"result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps({k:v for k,v in row.items() if k != "ghia"}), flush=True)
    return row


def paired_difference(hold, perturb, output):
    """Measure perturbation against its matching unperturbed discrete trajectory."""
    hold, perturb, output = Path(hold), Path(perturb), Path(output)
    hc = json.loads((hold/"configuration.json").read_text())
    pc = json.loads((perturb/"configuration.json").read_text())
    if hc.get("control") != "hold" or pc.get("control") != "perturb":
        raise ValueError("Expected an unperturbed hold and its perturbed counterpart")
    for key in ("source", "cells", "nu", "reynolds", "bounds", "scheme", "dt", "end", "input_sha256"):
        if hc[key] != pc[key]:
            raise ValueError(f"Paired trajectory mismatch for {key}")
    h, p = np.load(hold/"trajectory.npz"), np.load(perturb/"trajectory.npz")
    steps, hi, pi = np.intersect1d(h["steps"], p["steps"], return_indices=True)
    delta = p["U"][pi]-h["U"][hi]
    norms = np.max(abs(delta), axis=1)
    rms = np.sqrt(np.mean(delta**2, axis=1))
    if not len(steps) or steps[0] != 0 or norms[0] == 0:
        raise ValueError("Pair lacks a nonzero initial perturbation")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/"paired_trajectory.npz", steps=steps, times=steps*pc["dt"],
             delta_U=delta, delta_L=p["L"][pi]-h["L"][hi], velocity_inf=norms,
             component_rms=rms, gain_inf=norms/norms[0])
    report = dict(scope="perturbed minus matched unperturbed discrete trajectory",
        initial_velocity_inf=float(norms[0]), time=float(steps[-1]*pc["dt"]),
        final_velocity_inf=float(norms[-1]), final_gain_inf=float(norms[-1]/norms[0]),
        peak_gain_inf=float(np.max(norms)/norms[0]),
        hold_root_distance_inf=float(np.max(abs(h["U"][hi[-1]]-h["U"][0]))),
        matched_steps=len(steps), hold=str(hold), perturb=str(perturb))
    (output/"paired_difference.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "initial", "baseline", "parameters", "cache"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--scheme", choices=("characteristic", "be_oseen"), default="characteristic")
    parser.add_argument("--control", choices=("startup", "hold", "perturb", "pressure"), required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--end", type=float, default=.1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--amplitude", type=float, default=1e-3)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        advance(**vars(parser.parse_args()))
