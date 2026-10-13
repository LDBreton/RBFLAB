"""Bounded planar rejection sampling shared by geometric construction APIs."""
import numpy as np
from scipy.stats import qmc
from shapely import points, distance


def planar_interior(count, bounds, contains, polygon, method, seed, clearance, budget,
                    padua_degree=None):
    bounds = np.asarray(bounds, dtype=float)
    if method == "padua":
        from .padua import padua_points
        if (padua_degree+1)*(padua_degree+2)//2 > budget:
            raise ValueError("Padua grid exceeds max_candidates")
        reference = padua_points(padua_degree)
        p = bounds[:, 0] + (reference+1)/2*(bounds[:, 1]-bounds[:, 0])
        # Exclude reference edges and physical boundaries: boundary samples
        # belong to the existing curve sampler, including at corners.
        keep = np.all(np.abs(reference) < 1, axis=1) & contains(p)
        separation = distance(points(p), polygon.boundary)
        tolerance = 64*np.finfo(float).eps*np.max(bounds[:, 1]-bounds[:, 0])
        keep &= separation > tolerance
        if clearance:
            keep &= separation >= clearance
        return p[keep]
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
