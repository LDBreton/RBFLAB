"""Reproducible node generation returning the shared PointCloud contract."""
import numpy as np
from scipy.spatial import cKDTree
from ..geometry import PointCloud, ParametricDomain, ImplicitRegion, Border
from .volumes import RBFMesh3D
from .polygons import RBFMesh
from .surfaces import ParametricSurface3D
from ._sampling import planar_interior


def generate(domain, *, interior=400, boundary=None, method="halton", seed=0,
             boundary_distance=0., max_candidates=1000000, normals=None) -> PointCloud:
    """Generate labeled 2D/3D nodes through one geometry-independent entry point.

    Args:
        domain (ParametricDomain | ImplicitRegion | ParametricSurface3D | Border | Sequence | RBFMesh):
            Primitive/parametric 2D domain, implicit 3D region, sampled Border,
            list/tuple of sampled borders, an RBFMesh construction, or a
            ParametricSurface3D (use interior=0 for a surface).
        interior (int): Nonnegative number of points sampled in the material.
            Retained internal-interface samples add further interior unknowns.
        boundary (int | dict | None): Boundary count for domains (default 100).
            2D parametric domains also accept a count per arc label. For borders,
            omit this argument: counts are already fixed by border(n).
        method (str): random, halton, or sobol. No minimum separation guarantee.
        seed (int | None): Nonnegative seed or None; global random state is unchanged.
        boundary_distance (float): Physical boundary clearance. Generic 2D domains
            use a polygon approximation; custom 3D regions need clearance support.
        max_candidates (int): Positive rejection budget; failure raises rather
            than silently returning fewer interior points.
        normals (dict | None): Optional label -> normal arrays or callbacks of
            the label's point coordinates. Overrides generated normals; finite
            nonzero vectors are normalized and their supplied direction is retained.

    Returns:
        PointCloud: Float64 coordinates, exterior boundary labels, outward unit
            normals, and region/interface metadata. No connectivity is invented.

    Border geometry is rebuilt from its signed counts without mutating its
    objects or an input RBFMesh's samples. Internal interfaces remain interior;
    no interface normals or transmission equations are inferred. A label cannot
    simultaneously describe an exterior boundary and an internal interface.
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
    if normals is not None and not isinstance(normals, dict):
        raise TypeError("normals must be a mapping from boundary labels to vectors or callbacks")
    overrides = normals or {}
    if isinstance(domain, (Border, RBFMesh, list, tuple)):
        if boundary is not None:
            raise ValueError("omit boundary for Border geometry; choose resolution with border(n)")
        from ._border_cloud import generate_borders
        cloud = generate_borders(domain, interior=interior, method=method, seed=seed,
            boundary_distance=boundary_distance, max_candidates=max_candidates,
            normal_overrides=overrides)
        return _with_normals(cloud, overrides)
    if boundary is None:
        boundary = 100
    if isinstance(domain, ParametricSurface3D):
        if interior != 0 or boundary_distance != 0:
            raise ValueError("a parametric surface has no volume interior; use interior=0 and boundary_distance=0")
        if type(boundary) is not int or boundary < 1:
            raise ValueError("surface boundary must be a positive total count")
        from copy import copy
        surface = copy(domain)
        surface.Boundary_Points = []
        sample_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**32))
        samples = surface.generate_points(boundary, method=method, seed=sample_seed,
            append=False, max_candidates=max_candidates)
        p = np.array([(v.x, v.y, v.z) for v in samples])
        vectors = np.array([v.normal for v in samples])
        cloud = PointCloud(p, {domain.label: np.arange(len(p))}, {domain.label: vectors})
        return _with_normals(cloud, overrides)
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
        return _with_normals(PointCloud(p, groups, normals, regions={domain.label or "domain":np.arange(interior)}), overrides)
    if not isinstance(domain, ParametricDomain):
        raise TypeError("domain must be a 2D domain, 3D region/surface, Border collection, or RBFMesh")
    groups = domain.sample_boundary(boundary)
    interior_points = planar_interior(interior, domain.bounds, domain.contains_points,
        domain.polygon, method, seed, boundary_distance, max_candidates)
    points = [interior_points]
    ids, normals, offset = {}, {}, interior
    for label, (p, normal) in groups.items():
        ids[label]=np.arange(offset,offset+len(p));normals[label]=normal
        points.append(p);offset += len(p)
    return _with_normals(PointCloud(np.concatenate(points), ids, normals, regions={"domain":np.arange(interior)}), overrides)


def _with_normals(cloud, overrides):
    if set(overrides)-set(cloud.boundary):
        raise ValueError("normal overrides must refer to exterior boundary labels")
    for label, spec in overrides.items():
        locations = cloud.points[cloud.boundary[label]]
        raw = np.asarray(spec(locations) if callable(spec) else spec)
        if np.iscomplexobj(raw):
            raise ValueError("normal override vectors must be real-valued")
        vectors = np.asarray(raw, dtype=float)
        if vectors.shape != locations.shape or not np.isfinite(vectors).all():
            raise ValueError("normal override must match the boundary's coordinate shape")
        length = np.linalg.norm(vectors, axis=1)
        if np.any(length == 0):
            raise ValueError("normal override vectors must be nonzero")
        cloud.normals[label] = vectors/length[:, None]
    return cloud


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
