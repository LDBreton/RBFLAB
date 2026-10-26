"""Deterministic constructors for compact ball patch covers."""

from numbers import Integral

import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _options(points, n_patches, points_per_patch, overlap, min_points, seed):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or not len(points):
        raise ValueError('points must have shape (N, 2) or (N, 3), N > 0')
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('points must be finite and distinct')
    target = _positive_int(points_per_patch, 'points_per_patch')
    minimum = points.shape[1] + 1 if min_points is None else _positive_int(min_points, 'min_points')
    if minimum > len(points):
        raise ValueError('min_points exceeds the number of points')
    if isinstance(overlap, (bool, np.bool_)) or not np.isscalar(overlap):
        raise ValueError('overlap must be finite and in [0, 1)')
    try:
        overlap = float(overlap)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('overlap must be finite and in [0, 1)') from exc
    if not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must be finite and in [0, 1)')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    count = max(1, int(np.ceil(len(points) / target))) if n_patches is None else _positive_int(n_patches, 'n_patches')
    if count > len(points):
        raise ValueError('n_patches exceeds the number of points')
    return points, count, overlap, minimum, int(seed)


def _allocation(sizes, count):
    if count < len(sizes):
        raise ValueError('n_patches must cover every connected mesh component')
    if count > sum(sizes):
        raise ValueError('n_patches exceeds the number of points')
    result = np.ones(len(sizes), dtype=int)
    for _ in range(count - len(sizes)):
        candidates = [i for i, size in enumerate(sizes) if result[i] < size]
        i = max(candidates, key=lambda j: (sizes[j] / result[j], -j))
        result[i] += 1
    return result


def _cluster(points, count, rng):
    # Farthest-first starts Lloyd at distinct observed locations. Tie-breaking
    # follows input order, and empty clusters keep their existing center.
    indices = [int(rng.integers(len(points)))]
    distances = np.linalg.norm(points - points[indices[0]], axis=1)
    for _ in range(1, count):
        i = int(np.argmax(distances))
        indices.append(i)
        distances = np.minimum(distances, np.linalg.norm(points - points[i], axis=1))
    centers = points[indices].copy()
    for _ in range(30):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        if not np.isfinite(distances).all():
            raise ValueError('point coordinate range is too large for distances')
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for j in range(count):
            selected = points[labels == j]
            if len(selected):
                updated[j] = selected.mean(axis=0)
        if np.array_equal(updated, centers):
            break
        centers = updated
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    return centers, distances, labels


def _strict_radius(distance, overlap):
    if not np.isfinite(distance):
        raise ValueError('point coordinate range is too large for distances')
    # The runtime tests squared distances, including in higher precision. A
    # small relative margin absorbs rounding in the norm and squared sum.
    margin = 1 + 64 * np.finfo(float).eps
    floor = 16 * np.sqrt(np.finfo(float).tiny)
    value = np.nextafter(max(distance * (1 + overlap) * margin, floor), np.inf)
    if value <= distance:
        value = np.nextafter(distance, np.inf)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('could not construct a finite positive support radius')
    return value


def _build(cls, points, count, overlap, minimum, seed, components, triangles=None):
    components = np.asarray(components, dtype=int)
    ids = np.unique(components)
    groups = [np.flatnonzero(components == component) for component in ids]
    if any(len(group) < minimum for group in groups):
        raise ValueError('a connected mesh component has fewer than min_points vertices')
    allocated = _allocation([len(group) for group in groups], count)
    rng = np.random.default_rng(seed)
    centers_all, radii_all = [], []
    for component, group, n_local in zip(ids, groups, allocated):
        local = points[group]
        centers, distances, labels = _cluster(local, int(n_local), rng)
        triangle_rows = None if triangles is None else triangles[components[triangles[:, 0]] == component]
        triangle_assignment = None
        if triangle_rows is not None:
            centroids = points[triangle_rows].mean(axis=1)
            triangle_assignment = np.argmin(
                np.linalg.norm(centroids[:, None, :] - centers[None, :, :], axis=2), axis=1)
        for j, center in enumerate(centers):
            assigned = distances[labels == j, j]
            occupancy = np.partition(distances[:, j], minimum - 1)[minimum - 1]
            limit = max(float(assigned.max()) if len(assigned) else 0.0, float(occupancy))
            if triangle_rows is not None:
                selected = triangle_rows[triangle_assignment == j]
                if len(selected):
                    vertex_distances = np.linalg.norm(points[selected] - center, axis=2)
                    limit = max(limit, float(vertex_distances.max()))
            if limit == 0:
                other = np.linalg.norm(local - center, axis=1)
                positive = other[other > 0]
                limit = float(positive.min() / 2) if len(positive) else 1.0
            centers_all.append(center)
            radii_all.append(_strict_radius(limit, overlap))
    cover = cls(np.asarray(centers_all), np.asarray(radii_all))
    membership = np.zeros((len(points), len(cover.centers)), dtype=bool)
    for j, rows in enumerate(cover.memberships(points)):
        membership[rows, j] = True
    if not membership.any(axis=1).all() or np.any(membership.sum(axis=0) < minimum):
        raise ValueError('could not cover all samples with the requested minimum occupancy')
    if triangles is not None:
        if not np.all(np.any(np.all(membership[triangles], axis=1), axis=1)):
            raise ValueError('could not cover every mesh triangle')
    return cover


def cover_from_points(cls, points, *, n_patches, points_per_patch, overlap, min_points, seed):
    points, count, overlap, minimum, seed = _options(
        points, n_patches, points_per_patch, overlap, min_points, seed)
    return _build(cls, points, count, overlap, minimum, seed,
                  np.zeros(len(points), dtype=int))


def cover_from_mesh(cls, mesh, *, n_patches, points_per_patch, overlap, min_points, seed):
    from .geometry import TriangleMesh2D
    if not isinstance(mesh, TriangleMesh2D):
        raise TypeError('mesh must be a TriangleMesh2D')
    points, count, overlap, minimum, seed = _options(
        mesh.vertices, n_patches, points_per_patch, overlap, min_points, seed)
    triangles = mesh.triangles
    parent = np.arange(len(points))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in triangles:
        ra = root(int(a))
        parent[root(int(b))] = ra
        parent[root(int(c))] = ra
    components = np.array([root(i) for i in range(len(points))])
    count = max(count, len(np.unique(components))) if n_patches is None else count
    return _build(cls, points, count, overlap, minimum, seed, components, triangles)
