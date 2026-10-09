"""Bounded planar rejection sampling shared by geometric construction APIs."""
import numpy as np
from scipy.stats import qmc
from shapely import points, distance


def planar_interior(count, bounds, contains, polygon, method, seed, clearance, budget):
    bounds = np.asarray(bounds, dtype=float)
    engine = np.random.default_rng(seed) if method == "random" else (
        qmc.Halton(2, scramble=True, seed=seed) if method == "halton"
        else qmc.Sobol(2, scramble=True, seed=seed))
    accepted, total, attempted = [], 0, 0
    while total < count and attempted < budget:
        n = min(1024, budget-attempted)
        unit = engine.random((n, 2)) if method == "random" else engine.random(n)
        p = bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])
        keep = contains(p)
        if clearance:
            keep &= distance(points(p), polygon.boundary) >= clearance
        p = p[keep][:count-total]
        accepted.append(p)
        total += len(p)
        attempted += n
    if total != count:
        raise ValueError("Interior sampling exhausted max_candidates; check domain and clearance")
    return np.concatenate(accepted) if accepted else np.empty((0, 2))
