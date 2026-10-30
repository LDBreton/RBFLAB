"""Measure one alternative sparse ordering on the cached startup equations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .cached import load_problem
from .diagnostic_global import predictor, factor, direct, residuals


def resources():
    result = dict(logical_cpus=os.cpu_count(), process_cpu_seconds=sum(os.times()[:2]))
    try:
        import psutil
        process = psutil.Process()
        result.update(memory=process.memory_info()._asdict(),
                      memory_available=psutil.virtual_memory().available,
                      memory_total=psutil.virtual_memory().total,
                      cpu_affinity=process.cpu_affinity(),
                      system_cpu_times=psutil.cpu_times()._asdict(),
                      process_threads=process.num_threads())
    except (ImportError, AttributeError) as exc:
        result['resource_note'] = str(exc)
    return result


def run(maps_from, output, cells=126, dt=.00025):
    path = Path(output)
    if path.exists() or path.with_suffix('.npz').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(status='running', cells=cells, dt=dt, ordering='MMD_AT_PLUS_A',
                  threads=4, maps_from=str(maps_from), resources_start=resources(),
                  concurrent_work='Other requested Oseen trajectories may be active; timing is not isolated.',
                  scope='One alternative LU; no parameter sweep, no new COLAMD reference, no field-parity claim',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save(label):
        report['total_seconds'] = time.perf_counter()-started
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{label}: {report["total_seconds"]:.2f}s', flush=True)

    save('START')
    try:
        problem = load_problem(maps_from, cells)
        U0, _ = problem.initial()
        matrix, rhs = predictor(problem, U0, None, 1, dt)
        report['resources_before_factor'] = resources()
        report['threadpools'] = threadpool_info()
        before_cpu = sum(os.times()[:2])
        lu, factor_report = factor(matrix, ordering='MMD_AT_PLUS_A')
        factor_report['process_cpu_seconds'] = sum(os.times()[:2])-before_cpu
        report['factor'] = factor_report
        report['resources_after_factor'] = resources()
        save('FACTORIZED')
        answer, solve_seconds = direct(lu, rhs)
        U, L = np.split(answer, 2)
        gradient = problem.PY @ U+problem.PB @ problem.boundary(dt)+problem.PL @ L
        report.update(solve_seconds=solve_seconds, residual=residuals(matrix, rhs, answer),
                      velocity_max=float(np.max(abs(U))), gradient_max=float(np.max(abs(gradient))),
                      finite_fields=bool(np.isfinite(U).all() and np.isfinite(gradient).all()),
                      physical_parity='Not assessed: no second factorization was performed.')
        np.savez_compressed(path.with_suffix('.npz'), U=U, L=L, gradient=gradient,
                            points=problem.points, dt=dt)
        report.update(status='completed', resources_end=resources())
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
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.maps_from, args.output, args.cells)
