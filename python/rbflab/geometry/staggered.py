"""Explicit 2D triangle topology and vertex/edge-midpoint point clouds.

No triangulation is inferred from point coordinates. See docs/staggered.md for
the boundary, normal, indexing, and portable archive contracts.
"""

from dataclasses import dataclass, field
import json
from numbers import Integral

import numpy as np
from shapely.geometry import Polygon, LineString, MultiPoint, Point
from shapely.strtree import STRtree


def _indices(values, width, size, name):
    array = np.asarray(values)
    if array.size == 0:
        if array.shape not in ((0,), (0, width)):
            raise ValueError(f'{name} must have shape (N, {width})')
        return np.empty((0, width), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != width or array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be an integer array of shape (N, {width})')
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f'{name} indices are out of bounds')
    return array.astype(np.int64, copy=True)


def _labels(mapping, size, name):
    result = {}
    for label, pairs in (mapping or {}).items():
        if isinstance(label, bool) or not isinstance(label, (str, Integral)):
            raise ValueError('Labels must be strings or integers (not booleans)')
        label = int(label) if isinstance(label, Integral) else label
        pairs = np.sort(_indices(pairs, 2, size, name), axis=1)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError(f'{name} contains a self edge')
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError(f'{name} contains duplicate edges')
        result[label] = pairs
    return result


class TriangleMesh2D:
    """Validated conforming planar mesh, copied from zero-based arrays.

    ``boundary_edges`` and ``interface_edges`` map str/int labels to (K, 2)
    vertex-index arrays. If boundary_edges is omitted, all exterior edges use
    label 'boundary'. Explicit boundary labels must cover every exterior edge.
    Triangle rows and vertex rows retain their original ordering/orientation.
    """

    def __init__(self, vertices, triangles, boundary_edges=None, *, interface_edges=None):
        self.vertices = np.array(vertices, dtype=np.float64, copy=True)
        v = self.vertices
        if v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all():
            raise ValueError('vertices must be finite coordinates with shape (N, 2)')
        if len(np.unique(v, axis=0)) != len(v):
            raise ValueError('vertices must be distinct')
        self.triangles = _indices(triangles, 3, len(v), 'triangles')
        t = self.triangles
        if not len(t):
            raise ValueError('At least one triangle is required')
        if len(np.unique(t)) != len(v):
            raise ValueError('Every vertex must belong to a triangle')
        if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
            raise ValueError('Duplicate triangles are not allowed')
        a, b = v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]
        scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
        if not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError('Degenerate triangle or coordinate range too large')
        a, b = a / scale[:, None], b / scale[:, None]
        area = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if np.any(np.abs(area) <= 64 * np.finfo(float).eps):
            raise ValueError('Degenerate or numerically singular triangle')
        local = np.stack((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]), axis=1)
        self.edges, inverse = np.unique(np.sort(local.reshape(-1, 2), axis=1),
                                        axis=0, return_inverse=True)
        self.triangle_edges = inverse.reshape(-1, 3)
        counts = np.bincount(inverse, minlength=len(self.edges))
        if np.any(counts > 2):
            raise ValueError('Nonmanifold edge: more than two incident triangles')
        self.edge_triangles = np.full((len(self.edges), 2), -1, dtype=np.int64)
        for triangle_id, edge_ids in enumerate(self.triangle_edges):
            for edge_id in edge_ids:
                slot = 0 if self.edge_triangles[edge_id, 0] == -1 else 1
                self.edge_triangles[edge_id, slot] = triangle_id
        self.exterior_edges = np.flatnonzero(counts == 1)

        # A conforming embedding may intersect only along shared indexed entities.
        polygons = [Polygon(v[row]) for row in t]
        tree = STRtree(polygons)
        for i, polygon in enumerate(polygons):
            for j in tree.query(polygon, predicate='intersects'):
                if j <= i:
                    continue
                shared = np.intersect1d(t[i], t[j])
                intersection = polygon.intersection(polygons[j])
                expected = (LineString(v[shared]) if len(shared) == 2 else
                            Point(v[shared[0]]) if len(shared) == 1 else MultiPoint([]))
                if not intersection.equals(expected):
                    raise ValueError('Overlapping or nonconforming triangles')

        midpoints = v[self.edges[:, 0]] / 2 + v[self.edges[:, 1]] / 2
        if len(np.unique(midpoints, axis=0)) != len(midpoints):
            raise ValueError('Distinct edges have coincident midpoint nodes')
        self.boundary_edges = _labels(
            {'boundary': self.edges[self.exterior_edges]} if boundary_edges is None
            else boundary_edges, len(v), 'boundary_edges')
        self.interface_edges = _labels(interface_edges, len(v), 'interface_edges')
        lookup = {tuple(edge): i for i, edge in enumerate(self.edges)}
        self.boundary_edge_indices = self._map_edges(self.boundary_edges, lookup, counts, 1)
        self.interface_edge_indices = self._map_edges(self.interface_edges, lookup, counts, 2)
        covered = {int(i) for ids in self.boundary_edge_indices.values() for i in ids}
        if covered != set(self.exterior_edges.tolist()):
            raise ValueError('boundary_edges must label every exterior edge')

    @staticmethod
    def _map_edges(mapping, lookup, counts, expected_count):
        result = {}
        for label, pairs in mapping.items():
            try:
                ids = np.array([lookup[tuple(pair)] for pair in pairs], dtype=np.int64)
            except KeyError as exc:
                raise ValueError('Labeled edge is not in the triangle mesh') from exc
            if np.any(counts[ids] != expected_count):
                raise ValueError('Boundary edges must be exterior; interface edges must be interior')
            result[label] = np.sort(ids)
        return result


# One validated cloud contract for geometry and numerical methods.
from .cloud import PointCloud


@dataclass
class StaggeredLayout:
    """Vertex and edge-midpoint clouds with explicit 2D triangle topology."""
    vertices: PointCloud
    edge_midpoints: PointCloud
    edges: np.ndarray
    triangle_edges: np.ndarray
    edge_triangles: np.ndarray
    normal_policy: dict


def _normal_override(values, count, label):
    values = np.array(values, dtype=np.float64, copy=True)
    if values.shape != (count, 2) or not np.isfinite(values).all():
        raise ValueError(f'Normal override for {label!r} must have shape ({count}, 2)')
    if not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=1e-7, atol=1e-12):
        raise ValueError(f'Normal override for {label!r} must contain unit vectors')
    return values


def staggered_clouds(mesh, triangles=None, boundary_edges=None, *, interface_edges=None,
                     vertex_normals=None, edge_midpoint_normals=None):
    """Construct vertex and unique-edge-midpoint clouds.

    Accept TriangleMesh2D or (vertices, triangles, boundary_edges) arrays.
    Optional normal maps override computed normals per label, aligned with
    sorted vertex indices or sorted edge indices respectively.
    Supplied normals are trusted to follow the caller's CAD orientation.
    """
    if isinstance(mesh, TriangleMesh2D):
        if triangles is not None or boundary_edges is not None or interface_edges is not None:
            raise ValueError('Pass either a mesh or arrays, not both')
        # Revalidate public mutable input arrays and keep outputs independent.
        mesh = TriangleMesh2D(mesh.vertices, mesh.triangles, mesh.boundary_edges,
                              interface_edges=mesh.interface_edges)
    else:
        mesh = TriangleMesh2D(mesh, triangles, boundary_edges, interface_edges=interface_edges)
    vertex_overrides, edge_overrides = vertex_normals or {}, edge_midpoint_normals or {}
    for overrides in (vertex_overrides, edge_overrides):
        if set(overrides) - set(mesh.boundary_edges):
            raise ValueError('Normal override has an unknown boundary label')
    v, edges = mesh.vertices, mesh.edges
    midpoints = v[edges[:, 0]] / 2 + v[edges[:, 1]] / 2
    vertex_boundary, edge_boundary, vertex_normals_map, edge_normals_map, policies = {}, {}, {}, {}, {}
    for label, ids in mesh.boundary_edge_indices.items():
        pairs = edges[ids]
        nodes = np.unique(pairs)
        delta = v[pairs[:, 1]] - v[pairs[:, 0]]
        lengths = np.hypot(delta[:, 0], delta[:, 1])
        if not np.isfinite(lengths).all() or np.any(lengths == 0):
            raise ValueError('Boundary edge length is outside the supported numeric range')
        normals = np.column_stack((delta[:, 1], -delta[:, 0])) / lengths[:, None]
        for k, edge_id in enumerate(ids):
            triangle = mesh.triangles[mesh.edge_triangles[edge_id, 0]]
            opposite = triangle[~np.isin(triangle, pairs[k])][0]
            if np.dot(normals[k], v[opposite] - midpoints[edge_id]) > 0:
                normals[k] *= -1
        vertex_boundary[label], edge_boundary[label] = nodes, ids.copy()
        edge_normals_map[label] = (_normal_override(edge_overrides[label], len(ids), label)
                     if label in edge_overrides else normals)
        if label in vertex_overrides:
            vertex_normals_map[label] = _normal_override(vertex_overrides[label], len(nodes), label)
        else:
            averages = np.zeros((len(nodes), 2), dtype=np.float64)
            # Scale all weights in this label to avoid overflow in accumulation.
            weights = lengths / lengths.max() if len(lengths) else lengths
            for side in (0, 1):
                np.add.at(averages, np.searchsorted(nodes, pairs[:, side]),
                          normals * weights[:, None])
            magnitudes = np.linalg.norm(averages, axis=1)
            if np.any(magnitudes <= 64 * np.finfo(float).eps):
                raise ValueError(f'Ambiguous vertex normal for {label!r}; split side labels or supply normals')
            vertex_normals_map[label] = averages / magnitudes[:, None]
        policies[label] = {
            'vertices': 'supplied' if label in vertex_overrides else 'length-weighted edge average',
            'edge_midpoints': 'supplied' if label in edge_overrides else 'outward straight-edge normal',
        }
    vertex_interfaces = {label: np.unique(edges[ids]) for label, ids in mesh.interface_edge_indices.items()}
    edge_interfaces = {label: ids.copy() for label, ids in mesh.interface_edge_indices.items()}
    return StaggeredLayout(
        PointCloud(v.copy(), vertex_boundary, vertex_normals_map, mesh.triangles.copy(), vertex_interfaces),
        PointCloud(midpoints, edge_boundary, edge_normals_map, interfaces=edge_interfaces),
        edges.copy(), mesh.triangle_edges.copy(), mesh.edge_triangles.copy(), policies)


def save_staggered_layout(path, layout):
    """Save one compressed NPZ with numeric arrays and JSON metadata; no pickle."""
    labels = list(layout.edge_midpoints.boundary)
    interfaces = list(layout.edge_midpoints.interfaces)
    arrays = {'vertices': layout.vertices.points, 'triangles': layout.vertices.triangles}
    for i, label in enumerate(labels):
        arrays[f'b{i}'] = layout.edges[layout.edge_midpoints.boundary[label]]
        arrays[f'vertex_normals_{i}'] = layout.vertices.normals[label]
        arrays[f'edge_midpoint_normals_{i}'] = layout.edge_midpoints.normals[label]
    for i, label in enumerate(interfaces):
        arrays[f'i{i}'] = layout.edges[layout.edge_midpoints.interfaces[label]]
    arrays['metadata'] = np.array(json.dumps({
        'version': 2, 'labels': labels, 'interfaces': interfaces,
        'normal_policy': [layout.normal_policy[label] for label in labels]}))
    np.savez_compressed(path, **arrays)


def load_staggered_layout(path):
    """Read an archive without pickle and rebuild/validate deterministic topology."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['version'] != 2:
            raise ValueError('Unsupported staggered archive version')
        labels = metadata['labels']
        result = staggered_clouds(
            archive['vertices'], archive['triangles'],
            {label: archive[f'b{i}'] for i, label in enumerate(labels)},
            interface_edges={label: archive[f'i{i}'] for i, label in enumerate(metadata['interfaces'])},
            vertex_normals={label: archive[f'vertex_normals_{i}'] for i, label in enumerate(labels)},
            edge_midpoint_normals={label: archive[f'edge_midpoint_normals_{i}'] for i, label in enumerate(labels)})
        result.normal_policy = dict(zip(labels, metadata['normal_policy']))
        return result
