"""Cavity evolution by nodal least squares constrained by linear Stokes.

Each time step is a control window. Its reverse-time discrete adjoint is
the transpose of the actual Stokes equations, and the sparse factor is
reused throughout the run. No global pressure unknown is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .control_maps import load_maps
from .stokes_control import StokesControl
from .cavity_candidates import reference_at_nodal_lines


def write_json(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def ghia_at_nodes(points, velocity):
    if not all(np.any(np.isclose(points[:, axis], .5, rtol=0, atol=1e-14))
               for axis in (0, 1)):
        return dict(status='unavailable', reason='This grid has no nodes on the Ghia centerlines; no solution interpolation is used.')
    return dict(status='available', **reference_at_nodal_lines(points, velocity))


def run(maps, output, dt, end=.1, max_steps=None, method='gauss_newton'):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(end) or end <= 0:
        raise ValueError('dt and end must be finite and positive')
    if max_steps is not None and (not isinstance(max_steps, (int, np.integer)) or max_steps < 1):
        raise ValueError('max_steps must be a positive integer')
    if method not in ('gauss_newton', 'lbfgs', 'newton_krylov'):
        raise ValueError('Unsupported control optimizer')
    total = round(end/dt)
    if not np.isclose(total*dt, end, rtol=0, atol=1e-13):
        raise ValueError('End time must be an integer multiple of dt')
    previous_report = output/'result.json'
    if previous_report.exists() and json.loads(previous_report.read_text())['status'] == 'failed':
        raise ValueError('Prior run failed; preserve its evidence and use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    problem = load_maps(maps)
    augmentation = getattr(problem, 'map_configuration', {}).get('augmentation', [6, 1])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(maps.glob('*.npz'))}
    configuration = dict(maps=str(maps.resolve()), input_hashes=hashes,
        cells=problem.cells, total_nodes=(problem.cells+1)**2,
        interior_nodes=len(problem.points), dt=dt, target_time=end, viscosity=problem.nu,
        method=method, scheme='Backward Euler; one-step control windows',
        objective='0.5*dt*h^2*sum_interior_components((f-convection(U,L,g))^2)',
        adjoint='Exact transpose discrete Stokes sweep in reversed time',
        local_pressure_elimination=True, stencil=37,
        augmentation='/'.join(map(str, augmentation)),
        pressure_scaling='nu^2*q^4, q=cells/44, cbar_p=1', corner_subtraction=False,
        initial='rest; impulsive unit lid, stationary corner nodes',
        residual_tolerance=1e-8, speed_bound=10.,
        scope='All reported solution diagnostics are nodal; transient Ghia differences are not steady accuracy.')
    config_path = output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Existing run has different numerical inputs')
    write_json(config_path, configuration)
    history_path, checkpoint_path = output/'history.json', output/'checkpoint.npz'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    U, L = problem.initial()
    controls = np.zeros((1, len(U)))
    completed = 0
    if checkpoint_path.exists():
        with np.load(checkpoint_path) as saved:
            U, L, controls = saved['U'], saved['L'], saved['controls']
            completed = int(saved['completed'])
        # Each record belongs to an accepted step; refuse mixed checkpoint files.
        if len(history) != completed:
            raise ValueError('Checkpoint and history do not describe the same step')
    elif history:
        raise ValueError('History exists without its checkpoint')
    if completed >= total:
        print('Already completed', output, flush=True)
        return
    start = time.perf_counter()
    engine = StokesControl(problem, dt, 1, initial=(U,L), start_time=completed*dt,
        node_weights=np.full(len(problem.points), 1/problem.cells**2))
    factor_seconds = time.perf_counter()-start
    print(json.dumps(dict(event='factor_ready', cells=problem.cells, dt=dt,
          seconds=factor_seconds, nnz=int(engine.factorization.L.nnz+engine.factorization.U.nnz))), flush=True)
    limit = total if max_steps is None else min(total, completed+max_steps)
    failure = None

    def checkpoint():
        pending = output/'checkpoint.pending.npz'
        np.savez(pending, U=U, L=L, controls=controls, completed=completed)
        write_json(history_path, history)
        pending.replace(checkpoint_path)

    for step in range(completed+1, limit+1):
        before = time.perf_counter()
        oldU = U.copy()
        result = None
        try:
            engine.reset_window((U,L), (step-1)*dt)
            result = engine.solve(controls, method=method, maxiter=30, gtol=1e-14,
                                  residual_tolerance=1e-8, linear_maxiter=200)
            candidateU, candidateL = np.split(result['states'][-1], 2)
            speed = np.linalg.norm(candidateU.reshape(-1,2), axis=1)
            g = problem.boundary(step*dt)
            convection = engine.convection(result['states'])[-1]
            constraint = problem.A@candidateU + problem.B@g - problem.M@candidateL
            momentum = (candidateU-oldU)/dt + candidateL + convection - problem.force(step*dt)
            dx, dy = engine.derivatives(result['states'])
            if not all(np.isfinite(v).all() for v in
                       (candidateU, candidateL, speed, convection, constraint, momentum,
                        dx, dy, result['cost'], result['residual_inf'])):
                raise FloatingPointError('Nonfinite candidate or diagnostics')
        except Exception as exc:
            failure = dict(reason='solver_exception', step=step, time=step*dt,
                           error=f'{type(exc).__name__}: {exc}')
            if result is not None:
                np.savez(output/'failed_candidate.npz', states=result['states'],
                         controls=result['controls'], residual=result['residual'])
            break
        row = dict(step=step, time=step*dt, seconds=time.perf_counter()-before,
            cost=result['cost'], control_residual_inf=result['residual_inf'],
            constraint_residual_inf=float(np.max(abs(constraint))),
            momentum_residual_inf=float(np.max(abs(momentum))),
            mapped_divergence_inf=float(np.max(abs(dx[0,::2]+dy[0,1::2]))),
            max_nodal_speed=float(np.max(speed)), kinetic_nodal_mean=float(.5*np.mean(speed**2)),
            optimizer_success=result['optimizer_success'], optimizer_message=result['optimizer_message'],
            optimizer_function_evaluations=result.get('optimizer_function_evaluations'),
            optimizer_iterations=result.get('optimizer_iterations'),
            krylov_iterations=sum(r.get('linear_iterations',0) for r in result['history']),
            initial_cost=result['history'][0]['cost'],
            forward_sweeps=engine.forward_sweeps, adjoint_sweeps=engine.adjoint_sweeps)
        if not result['converged'] or not np.isfinite(speed).all():
            failure = dict(reason='unresolved_control_residual', **row)
            np.savez(output/'failed_candidate.npz', U=candidateU, L=candidateL,
                     controls=result['controls'], residual=result['residual'])
            break
        U, L, controls = candidateU, candidateL, result['controls']
        completed = step
        history.append(row)
        if step % 20 == 0 or step == limit:
            checkpoint()
            print(json.dumps(dict(event='progress', cells=problem.cells, dt=dt, **row)), flush=True)
        if row['max_nodal_speed'] > configuration['speed_bound']:
            failure = dict(reason='speed_bound_exceeded', **row)
            break
    checkpoint()
    gradient = problem.PY@U+problem.PB@problem.boundary(completed*dt)+problem.PL@L
    np.savez(output/'fields.npz', points=problem.points, boundary_points=problem.boundary_points,
             U=U, L=L, gradient=gradient, controls=controls, time=completed*dt)
    status = 'failed' if failure else ('completed' if completed == total else 'checkpoint')
    report = dict(status=status, completed_steps=completed, time=completed*dt,
        target_time=end, cells=problem.cells, total_nodes=(problem.cells+1)**2, dt=dt,
        failure=failure, factor_seconds=factor_seconds,
        seconds_this_process=time.perf_counter()-start,
        global_factorizations_this_process=1,
        forward_sweeps_this_process=engine.forward_sweeps,
        adjoint_sweeps_this_process=engine.adjoint_sweeps,
        max_control_residual_inf=max((r['control_residual_inf'] for r in history),default=0),
        max_constraint_residual_inf=max((r['constraint_residual_inf'] for r in history),default=0),
        max_momentum_residual_inf=max((r['momentum_residual_inf'] for r in history),default=0),
        max_mapped_divergence_inf=max((r['mapped_divergence_inf'] for r in history),default=0),
        peak_nodal_speed=max((r['max_nodal_speed'] for r in history),default=0),
        ghia=ghia_at_nodes(problem.points,U))
    write_json(output/'result.json',report)
    print(json.dumps(dict(event='result', **{k:v for k,v in report.items() if k!='ghia'})),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--end',type=float,default=.1)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--method',choices=['gauss_newton','lbfgs','newton_krylov'],default='gauss_newton')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.maps,args.output,args.dt,args.end,args.max_steps,args.method)
