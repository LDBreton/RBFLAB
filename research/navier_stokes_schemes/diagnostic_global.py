"""Bounded global-solver comparisons on cached maps and identical equations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import LinearOperator, gmres, splu
from threadpoolctl import threadpool_limits

from .probe_global import load_problem


def predictor(problem, U, Uold, step, dt):
    n = len(U)
    alpha = (1 if step == 1 else 1.5)/dt
    history = U/dt if step == 1 else (2*U-.5*Uold)/dt
    advector = U if step == 1 else 2*U-Uold
    vx = diags(np.repeat(advector[::2], 2))
    vy = diags(np.repeat(advector[1::2], 2))
    ou = vx @ problem.DxU + vy @ problem.DyU
    ol = vx @ problem.DxL + vy @ problem.DyL
    g, f = problem.boundary(step*dt), problem.force(step*dt)
    cb = vx @ (problem.DxB @ g)+vy @ (problem.DyB @ g)
    ident = eye(n, format='csr')
    matrix = bmat([[problem.A, -problem.M], [alpha*ident+ou, ident+ol]], format='csc')
    rhs = np.concatenate((-problem.B @ g, f+history-cb))
    return matrix, rhs


def factor(matrix, ordering='COLAMD'):
    before = time.perf_counter()
    lu = splu(matrix, permc_spec=ordering)
    report = dict(seconds=time.perf_counter()-before, ordering=ordering,
                  matrix_nnz=int(matrix.nnz), L_nnz=int(lu.L.nnz), U_nnz=int(lu.U.nnz))
    report['fill_ratio'] = (report['L_nnz']+report['U_nnz'])/matrix.nnz
    return lu, report


def direct(lu, rhs):
    before = time.perf_counter()
    answer = lu.solve(rhs)
    return answer, time.perf_counter()-before


def residuals(matrix, rhs, answer):
    residual = matrix @ answer-rhs
    norm_b = float(np.linalg.norm(rhs))
    denominator = abs(matrix) @ abs(answer)+abs(rhs)
    return dict(residual_l2=float(np.linalg.norm(residual)),
                residual_inf=float(np.linalg.norm(residual, np.inf)),
                rhs_l2=norm_b, gmres_threshold=max(1e-11, 1e-14*norm_b),
                max_componentwise_backward_error=float(np.max(
                    abs(residual)/np.maximum(denominator, np.finfo(float).tiny))),
                finite=bool(np.isfinite(answer).all()))


def compare(problem, answer, reference):
    du, dl = np.split(answer-reference, 2)
    dg = problem.PY @ du+problem.PL @ dl
    return dict(velocity_max=float(np.max(abs(du))), observation_max=float(np.max(abs(dl))),
                gradient_max=float(np.max(abs(dg))),
                physical_parity=bool(np.max(abs(du)) <= 1e-10 and np.max(abs(dg)) <= 1e-7))


def iterative(matrix, rhs, lu):
    history = []
    before = time.perf_counter()
    answer, info = gmres(matrix, rhs,
                         M=LinearOperator(matrix.shape, matvec=lu.solve, dtype=float),
                         rtol=1e-14, atol=1e-11, restart=40, maxiter=5,
                         callback=lambda value: history.append(float(value)), callback_type='pr_norm')
    return answer, dict(seconds=time.perf_counter()-before, info=int(info),
                        inner_iterations=len(history), preconditioned_residual_history=history,
                        **residuals(matrix, rhs, answer))


def scaling(matrix):
    row_max = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    rows = 1/np.where(row_max > 0, row_max, 1.)
    row_scaled = diags(rows) @ matrix
    col_max = np.asarray(abs(row_scaled).max(axis=0).toarray()).ravel()
    columns = 1/np.where(col_max > 0, col_max, 1.)
    return rows, columns, (row_scaled @ diags(columns)).tocsc()


def run(maps_from, output, cells=126, steps=2, dt=.00025):
    if steps not in (2, 3):
        raise ValueError('Bounded diagnostic supports only two or three steps')
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status='running', cells=cells, steps=steps, dt=dt, threads=4,
                  maps_from=str(maps_from), equations='unchanged coupled Oseen; common direct-reference trajectory',
                  stages={}, source_sha256={})
    for name in ('diagnostic_global.py', 'coupled.py', 'linear.py', 'probe_global.py'):
        report['source_sha256'][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()

    def save(label):
        report['seconds'] = time.perf_counter()-started
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{label}: {report["seconds"]:.1f}s', flush=True)

    save('START')
    try:
        problem = load_problem(maps_from, cells)
        U0, _ = problem.initial()
        a1, b1 = predictor(problem, U0, None, 1, dt)
        lu1, info = factor(a1)
        x1, solve_time = direct(lu1, b1)
        report['stages']['startup'] = dict(factor=info, solve_seconds=solve_time, **residuals(a1, b1, x1))
        save('STARTUP_LU')
        U1, _ = np.split(x1, 2)
        a2, b2 = predictor(problem, U1, U0, 2, dt)
        plain_stale, stale_info = iterative(a2, b2, lu1)
        report['stages']['step2_BE_preconditioner'] = stale_info
        save('STEP2_BE_GMRES')
        del lu1, a1
        gc.collect()
        lu2, fresh_factor = factor(a2)
        x2, solve_time = direct(lu2, b2)
        report['stages']['step2_direct'] = dict(factor=fresh_factor, solve_seconds=solve_time,
                                               **residuals(a2, b2, x2))
        stale_info.update(compare(problem, plain_stale, x2))
        save('STEP2_DIRECT_LU')
        plain_fresh, fresh_info = iterative(a2, b2, lu2)
        fresh_info.update(compare(problem, plain_fresh, x2))
        report['stages']['step2_BDF_preconditioner'] = fresh_info
        save('STEP2_FRESH_GMRES')
        row_scale, col_scale, scaled2 = scaling(a2)
        scaled_rhs = row_scale*b2
        scaled_lu, scaled_factor = factor(scaled2)
        ys, scaled_direct_time = direct(scaled_lu, scaled_rhs)
        report['stages']['step2_scaled_direct'] = dict(
            factor=scaled_factor, solve_seconds=scaled_direct_time,
            row_scale_min=float(row_scale.min()), row_scale_max=float(row_scale.max()),
            column_scale_min=float(col_scale.min()), column_scale_max=float(col_scale.max()),
            scaled_residual=residuals(scaled2, scaled_rhs, ys),
            original_residual=residuals(a2, b2, col_scale*ys),
            **compare(problem, col_scale*ys, x2))
        save('STEP2_SCALED_LU')
        ys, scaled_info = iterative(scaled2, scaled_rhs, scaled_lu)
        scaled_info['original_residual'] = residuals(a2, b2, col_scale*ys)
        scaled_info.update(compare(problem, col_scale*ys, x2))
        report['stages']['step2_scaled_BDF_preconditioner'] = scaled_info
        save('STEP2_SCALED_GMRES')
        if steps == 3:
            U2, _ = np.split(x2, 2)
            a3, b3 = predictor(problem, U2, U1, 3, dt)
            plain3, plain_info = iterative(a3, b3, lu2)
            report['stages']['step3_BDF_preconditioner'] = plain_info
            save('STEP3_PLAIN_GMRES')
            scaled3 = (diags(row_scale) @ a3 @ diags(col_scale)).tocsc()
            y3, scaled_info = iterative(scaled3, row_scale*b3, scaled_lu)
            scaled_info['original_residual'] = residuals(a3, b3, col_scale*y3)
            report['stages']['step3_scaled_BDF_preconditioner'] = scaled_info
            save('STEP3_SCALED_GMRES')
            del lu2, scaled_lu, scaled2, scaled3
            gc.collect()
            lu3, info = factor(a3)
            x3, solve_time = direct(lu3, b3)
            report['stages']['step3_direct'] = dict(factor=info, solve_seconds=solve_time,
                                                   **residuals(a3, b3, x3))
            plain_info.update(compare(problem, plain3, x3))
            scaled_info.update(compare(problem, col_scale*y3, x3))
            save('STEP3_DIRECT_CHECK')
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(report['status'].upper())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-from', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cells', type=int, default=126)
    parser.add_argument('--steps', type=int, choices=(2, 3), default=2)
    parser.add_argument('--dt', type=float, default=.00025)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.maps_from, args.output, args.cells, args.steps, args.dt)
