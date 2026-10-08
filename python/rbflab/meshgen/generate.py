"""Reproducible node generation returning the shared PointCloud contract."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from ..geometry import PointCloud, ParametricDomain, ImplicitRegion
from .volumes import RBFMesh3D


def generate(domain, *, interior=400, boundary=100, method="halton", seed=0,
             boundary_distance=0., max_candidates=1000000) -> PointCloud:
    """Generate a 2D parametric or 3D implicit cloud, without connectivity.

    Args:
        domain (ParametricDomain | ImplicitRegion): ParametricDomain (including primitives) or ImplicitRegion.
        interior (int): Exact nonnegative interior node count.
        boundary (int | dict): Total boundary count; 2D also accepts counts by arc label.
        method (str): 'random', 'halton', or 'sobol'. No minimum separation guarantee.
        seed (int | None): Nonnegative integer or None. Does not change global random state.
        boundary_distance (float): Physical clearance from the boundary. 2D generic
            domains use the polygon approximation; 3D needs region clearance.
        max_candidates (int): Rejection budget; failure raises instead of returning
            fewer interior points.

    Returns:
        PointCloud: Float64 coordinates, boundary labels, unit normals and
            region membership. Rejection/truncation weakens Sobol balance.
    """
    if type(interior) is not int or interior < 0:
        raise ValueError("interior must be a nonnegative integer")
    if method not in ("random", "halton", "sobol"):
        raise ValueError("method must be random, halton or sobol")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    if not np.isfinite(boundary_distance) or boundary_distance < 0:
        raise ValueError("boundary_distance must be finite and nonnegative")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if isinstance(domain, ImplicitRegion):
        if type(boundary) is not int or boundary < 1:
            raise ValueError("3D boundary must be a positive total count")
        # The legacy API intentionally uses random's global state for seed=None;
        # isolate the unified API while retaining that legacy behavior separately.
        sample_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**32))
        mesh = RBFMesh3D(domain)
        mesh.generate_points(interior, boundary_distance, method=method, seed=sample_seed,
                             append=False, max_candidates=max_candidates)
        mesh.generate_boundary_points(boundary, method=method, seed=sample_seed,
                                      append=False, max_candidates=max_candidates)
        p = np.array([(v.x,v.y,v.z) for v in mesh.Points+mesh.Boundary_Points])
        groups, normals = {}, {}
        for i, v in enumerate(mesh.Boundary_Points, interior):
            groups.setdefault(v.boundary_label, []).append(i)
            normals.setdefault(v.boundary_label, []).append(v.normal)
        return PointCloud(p, groups, normals, regions={domain.label or "domain":np.arange(interior)})
    if not isinstance(domain, ParametricDomain):
        raise TypeError("domain must be ParametricDomain or ImplicitRegion")
    groups = domain.sample_boundary(boundary)
    bounds = np.asarray(domain.bounds)
    engine = np.random.default_rng(seed) if method == "random" else (
        qmc.Halton(2, scramble=True, seed=seed) if method == "halton" else qmc.Sobol(2, scramble=True, seed=seed))
    accepted, count, attempted = [], 0, 0
    # Fixed power-of-two batches avoid Sobol warnings on ordinary calls.
    while count < interior and attempted < max_candidates:
        n = min(1024, max_candidates-attempted)
        unit = engine.random((n, 2)) if method == "random" else engine.random(n)
        p = bounds[:,0]+unit*(bounds[:,1]-bounds[:,0])
        keep = domain.contains_points(p)
        if boundary_distance:
            from shapely import points, distance
            keep &= distance(points(p), domain.polygon.boundary) >= boundary_distance
        p = p[keep][:interior-count]
        accepted.append(p);count += len(p);attempted += n
    if count != interior:
        raise ValueError("Interior sampling exhausted max_candidates; check domain and clearance")
    points = [np.concatenate(accepted) if accepted else np.empty((0,2))]
    ids, normals, offset = {}, {}, interior
    for label, (p, normal) in groups.items():
        ids[label]=np.arange(offset,offset+len(p));normals[label]=normal
        points.append(p);offset += len(p)
    return PointCloud(np.concatenate(points), ids, normals, regions={"domain":np.arange(interior)})


def quality(cloud, *, stencil_size=25, polynomial_degree=2):
    """Report nearest-node spacing and normalized local polynomial geometry.

    Returns minimum/median spacing, min/max spacing ratio, worst stencil
    separation ratio and worst polynomial singular-value ratio. These are
    geometric diagnostics, not bounds on PDE error or stability.
    """
    from ..stencils import geometry_quality
    p = cloud.points
    if type(stencil_size) is not int or not 2 <= stencil_size <= len(p):
        raise ValueError("stencil_size must be between 2 and the node count")
    if type(polynomial_degree) is not int or polynomial_degree < 0:
        raise ValueError("polynomial_degree must be a nonnegative integer")
    tree = cKDTree(p)
    nearest = tree.query(p, k=2)[0][:,1]
    indices = tree.query(p, k=stencil_size)[1]
    diagnostics = [geometry_quality(p[ids], target, polynomial_degree) for target,ids in zip(p,indices)]
    return dict(nodes=len(p), minimum_spacing=float(nearest.min()), median_spacing=float(np.median(nearest)),
                spacing_ratio=float(nearest.min()/nearest.max()),
                worst_separation_ratio=min(v['separation_ratio'] for v in diagnostics),
                worst_polynomial_ratio=min(v['polynomial_ratio'] for v in diagnostics))
