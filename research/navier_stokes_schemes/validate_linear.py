"""Compare direct and preconditioned Oseen solves with identical MPI maps."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled import evolve
from .problem import Problem


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(f'Choose a new validation output: {path}')
    problem = Problem(cells=6, ranks=2)
    rows = []
    for dt in (.00025, .000125):
        problem.linear_solver = 'direct'
        direct = evolve(problem, 'oseen', dt, round(.005/dt))
        problem.linear_solver = 'gmres'
        iterative = evolve(problem, 'oseen', dt, round(.005/dt))
        row = {'dt': dt, 'linear_solver': iterative['linear_solver']}
        for name in ('U', 'gradient'):
            difference = float(np.max(np.abs(direct[name]-iterative[name])))
            row[name+'_difference_max'] = difference
            np.testing.assert_allclose(iterative[name], direct[name], rtol=0,
                                       atol=1e-10 if name == 'U' else 1e-7)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSON output file')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
