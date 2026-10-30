"""Checkpointed fresh public MPI weights, compared with a saved cavity baseline.

One invocation computes a small consecutive group of owners. This avoids long
shell sessions while leaving each actual local solve on eight MPI ranks.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits
import rbflab as r
from .pressure_lifting import load_problem, NAMES


def run(cells, source, output, start, count):
    p = load_problem(cells, source)
    owners = np.arange(start, min(start + count, len(p.points)))
    if not len(owners):
        raise ValueError("No owners selected")
    folder = output / f"owners_{start:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    momentum = -p.nu * r.laplacian(p.u) + r.gradient(p.p)
    samples = {
        "velocity": r.Samples(p.points, operator=p.u, within=p.stencil),
        "boundary": r.Samples(p.boundary_points, operator=p.u, within=p.stencil),
        "momentum": r.Samples(p.points, operator=momentum,
                              within=p.stencil, target="exclude"),
    }
    operators = dict(L=momentum, P=r.gradient(p.p), Dx=r.dx(p.u), Dy=r.dy(p.u))
    before = time.perf_counter()
    local = r.LocalApproximation(
        source=samples, trial=p.space.representers(samples),
        backend=r.CppBackend(threads=1, timeout=900),
        execution=r.LocalExecution(scheduler="mpi", ranks=8),
        precision=r.Precision(local_digits=80))
    maps = local.operators(targets=p.points[owners], operators=operators)
    rows = (2 * owners[:, None] + np.arange(2)).ravel()
    report = dict(cells=cells, source=str(source), owners=owners.tolist(),
                  execution=maps.diagnostics["execution"], blocks={})
    for target, names in zip(operators, NAMES):
        for group, name in zip(("velocity", "boundary", "momentum"), names):
            fresh = maps[target][group].to_scipy()
            saved = getattr(p, name)[rows]
            error = float(np.max(abs((fresh - saved).data), initial=0.))
            report["blocks"][name] = error
            np.testing.assert_allclose(fresh.toarray(), saved.toarray(),
                                       rtol=1e-12, atol=1e-9, err_msg=name)
            save_npz(folder / (name + ".npz"), fresh)
    report.update(status="passed", seconds=time.perf_counter() - before)
    (folder / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(start=start, count=len(owners),
                         maximum_error=max(report["blocks"].values()),
                         seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(**vars(args))
