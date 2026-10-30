"""Inspect a few global steps using already assembled, unchanged sparse maps."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from .coupled import evolve
from .cached import load_problem




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-from', required=True)
    parser.add_argument('--cells', type=int, default=126)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    problem = load_problem(args.maps_from, args.cells)
    before = time.perf_counter()
    result = dict(cells=args.cells, steps=args.steps, dt=.00025,
                  cached_maps=args.maps_from, purpose='linear solver diagnostic; no weight rebuild')
    try:
        with threadpool_limits(limits=4):
            state = evolve(problem, 'oseen', .00025, args.steps)
        result.update(status='completed', linear_solver=state['linear_solver'],
                      diagnostics=state['diagnostics'])
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
    result['seconds'] = time.perf_counter()-before
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
