"""Compare schemes and save every success or failure before continuing.

python -m research.navier_stokes_schemes.run --output outputs/ns_schemes_run
Scheme 7 must be requested separately, after the other tests finish.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits

from .problem import Problem


def run(args):
    fixed_schemes = {'oseen', 'implicit', 'oseen_newton', 'rotational', 'imex', 'imex_ars222'}
    if getattr(args, 'reuse_stencils', False) and not set(args.schemes) <= fixed_schemes:
        raise ValueError('Exact template reuse currently supports these fixed-map research schemes only')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = vars(args).copy()
    metadata.update(stencil=37, augmentation='6/1', scaling='q=cells/44',
                    pressure='pure Gaussian cbar=1', local_digits=80,
                    global_dtype='float64', domain='unit square', offset=0,
                    error='absolute interior-nodal vector RMS and max',
                    boundary='homogeneous', initial='exact manufactured U and Stokes L')
    (output/'configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    results = []
    for cells in args.cells:
        start = time.perf_counter()
        if getattr(args, 'maps_from', None):
            from .cached import load_problem
            problem = load_problem(args.maps_from, cells)
            if problem.nu != args.nu or not set(args.schemes) <= fixed_schemes:
                raise ValueError('Cached maps require matching viscosity and fixed-map schemes')
            problem.linear_solver = args.linear_solver
            problem.global_ordering = getattr(args, 'global_ordering', 'COLAMD')
        else:
            problem = Problem(cells=cells, nu=args.nu, ranks=args.ranks,
                          native_timeout=getattr(args, 'native_timeout', 3600),
                          linear_solver=getattr(args, 'linear_solver', 'direct'),
                          global_ordering=getattr(args, 'global_ordering', 'COLAMD'),
                          reuse_stencils=getattr(args, 'reuse_stencils', False),
                          weight_folder=output/f'weights_cells{cells}')
        # Preserve expensive assembled maps for inspection and later reuse.
        map_directory = output/f'maps_cells{cells}'
        map_directory.mkdir()
        for name in ('M', 'A', 'B', 'SL', 'PY', 'PB', 'PL',
                     'DxU', 'DxB', 'DxL', 'DyU', 'DyB', 'DyL'):
            save_npz(map_directory/f'{name}.npz', getattr(problem, name))
        np.savez(map_directory/'nodes.npz', points=problem.points,
                 boundary_points=problem.boundary_points)
        fixed_schemes = {'oseen', 'implicit', 'oseen_newton', 'rotational',
                         'imex', 'imex_ars222'}
        if set(args.schemes) <= fixed_schemes:
            # The fixed schemes use the detached sparse blocks, not local fits.
            problem.maps = problem.local = None
            gc.collect()
        action = 'Loaded cached maps for' if getattr(args, 'maps_from', None) else 'Built'
        print(f'{action} {cells}: {(cells+1)**2} nodes in {time.perf_counter()-start:.1f}s', flush=True)
        for scheme in args.schemes:
            if scheme == 'adaptive_hermite':
                from .adaptive_hermite import evolve
            elif scheme in ('imex', 'imex_ars222', 'characteristic', 'characteristic_bdf2'):
                from .transport import evolve
            else:
                from .coupled import evolve
            for dt in args.dt:
                steps = round(args.end_time/dt)
                if steps < 1 or not np.isclose(steps*dt, args.end_time, rtol=0, atol=1e-14):
                    raise ValueError('end-time must be a positive multiple of each dt')
                row = dict(cells=cells, nodes=(cells+1)**2, scheme=scheme, dt=dt,
                           end_time=args.end_time, steps=steps)
                start = time.perf_counter()
                print(f'Start {scheme}, cells={cells}, dt={dt}, steps={steps}', flush=True)
                try:
                    with threadpool_limits(limits=getattr(args, 'global_threads', 1)):
                        state = evolve(problem, scheme, dt, steps)
                    exact_u, exact_g = problem.exact(args.end_time)
                    for name, computed, exact in [('velocity', state['U'], exact_u),
                                                   ('gradient', state['gradient'], exact_g)]:
                        lengths = np.linalg.norm((computed-exact).reshape(-1, 2), axis=1)
                        if not np.isfinite(lengths).all():
                            raise FloatingPointError('Nonfinite error')
                        row[name+'_rms'] = float(np.sqrt(np.mean(lengths**2)))
                        row[name+'_max'] = float(lengths.max())
                    row['status'] = 'completed'
                    row['nodal_mean_kinetic_energy'] = float(.5*np.mean(np.sum(state['U'].reshape(-1, 2)**2, axis=1)))
                    row['diagnostics'] = state['diagnostics']
                    if 'linear_solver' in state:
                        row['linear_solver'] = state['linear_solver']
                    np.savez(output/f'{scheme}_cells{cells}_dt{dt}.npz',
                             U=state['U'], L=state['L'], gradient=state['gradient'],
                             points=problem.points)
                except Exception as error:
                    row.update(status='failed', error=str(error), traceback=traceback.format_exc())
                row['seconds'] = time.perf_counter()-start
                results.append(row)
                (output/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                summary = {k: v for k, v in row.items() if k not in ('diagnostics', 'traceback', 'linear_solver')}
                if 'linear_solver' in row:
                    summary['linear_solver'] = {k: v for k, v in row['linear_solver'].items() if k != 'solves'}
                print(summary, flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, nargs='+', default=[6, 8])
    parser.add_argument('--nu', type=float, default=1.)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--native-timeout', type=float, default=3600,
                        help='Maximum seconds allowed for native local weight execution')
    parser.add_argument('--linear-solver', choices=['direct', 'gmres', 'gmres_scaled'], default='direct')
    parser.add_argument('--reuse-stencils', action='store_true',
                        help='Reuse exactly translated and permuted constant-operator stencils')
    parser.add_argument('--maps-from', help='Load this study\'s saved fixed maps; do not recompute weights')
    parser.add_argument('--global-threads', type=int, default=1,
                        help='BLAS threads during global time integration; MPI weights use one per rank')
    parser.add_argument('--global-ordering', choices=['COLAMD', 'MMD_AT_PLUS_A'], default='COLAMD',
                        help='Sparse ordering for iterative-solver LU preconditioners')
    parser.add_argument('--dt', type=float, nargs='+', default=[.00025, .000125])
    parser.add_argument('--end-time', type=float, default=.001)
    parser.add_argument('--schemes', nargs='+', default=[
        'oseen', 'implicit', 'oseen_newton', 'imex', 'imex_ars222', 'rotational',
        'characteristic', 'characteristic_bdf2'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args)
