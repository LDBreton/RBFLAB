"""Evolve the exact pressure balance u=0, p=xy with the control solver."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .control_maps import load_maps
from .stokes_control import StokesControl


def run(maps, output, dt, end=.02):
    output.mkdir(parents=True, exist_ok=False)
    problem = load_maps(maps)
    x, y = problem.points.T
    exact_gradient = np.column_stack((y,x)).ravel()
    problem.g0 = np.zeros_like(problem.g0)
    problem.force = lambda t: exact_gradient.copy()
    U, L = np.zeros_like(exact_gradient), exact_gradient.copy()
    count = round(end/dt)
    if not np.isclose(count*dt,end):
        raise ValueError('End time must be an integer multiple of dt')
    configuration = dict(cells=problem.cells, total_nodes=(problem.cells+1)**2,
        dt=dt, end=end, exact_velocity='zero', exact_pressure='xy',
        force='grad(xy)=(y,x)', boundary='homogeneous velocity',
        maps=str(maps.resolve()), augmentation=problem.map_configuration['augmentation'],
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(maps.glob('*.npz'))})
    (output/'configuration.json').write_text(json.dumps(configuration,indent=2))
    started=time.perf_counter()
    engine=StokesControl(problem,dt,1,initial=(U,L),
        node_weights=np.full(len(problem.points),1/problem.cells**2))
    controls=np.zeros((1,len(U)))
    history=[]
    failure=None
    for k in range(count):
        engine.reset_window((U,L),k*dt)
        try:
            result=engine.solve(controls,method='newton_krylov',maxiter=30,
                gtol=1e-14,residual_tolerance=1e-8,linear_maxiter=200)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failure=dict(reason='solver_exception',step=k+1,error=str(exc))
            break
        if not result['converged']:
            failure=dict(reason='unresolved_control',step=k+1,
                         residual_inf=result['residual_inf'])
            np.savez(output/'failed_candidate.npz',states=result['states'],
                     controls=result['controls'],residual=result['residual'])
            break
        U,L=np.split(result['states'][-1],2)
        controls=result['controls']
        gradient=problem.PY@U+problem.PL@L
        history.append(dict(step=k+1,time=(k+1)*dt,
            velocity_component_max=float(np.max(abs(U))),
            velocity_vector_rms=float(np.sqrt(np.mean(np.sum(U.reshape(-1,2)**2,axis=1)))),
            gradient_error_component_max=float(np.max(abs(gradient-exact_gradient))),
            control_residual_inf=result['residual_inf'],cost=result['cost']))
        if history[-1]['velocity_component_max']>1:
            failure=dict(reason='spurious_velocity_bound_exceeded',**history[-1])
            break
    gradient=problem.PY@U+problem.PL@L
    np.savez(output/'fields.npz',points=problem.points,U=U,L=L,gradient=gradient,
             exact_gradient=exact_gradient,time=len(history)*dt)
    report=dict(status='failed' if failure else 'completed',failure=failure,
        cells=problem.cells,total_nodes=(problem.cells+1)**2,dt=dt,time=len(history)*dt,
        peak_velocity_component_max=max((r['velocity_component_max'] for r in history),default=0),
        peak_gradient_error_component_max=max((r['gradient_error_component_max'] for r in history),default=0),
        seconds=time.perf_counter()-started,scope='Absolute errors only at interior mesh nodes')
    (output/'history.json').write_text(json.dumps(history,indent=2))
    (output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dt',type=float,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.maps,args.output,args.dt)
