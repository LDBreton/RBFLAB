"""Build a shared PointCloud from oriented borders, without mutating inputs."""
import numpy as np
from shapely import points, covers, distance, intersects_xy
from shapely.ops import unary_union
from ..geometry import PointCloud, Border
from .polygons import RBFMesh
from ._sampling import planar_interior


def _outward(material, points_on_curve, next_points, vectors, tolerance):
    """Choose material-facing sign at the associated polygon edge, not its curve sagitta."""
    midpoint = (points_on_curve+next_points)/2
    use_midpoint = distance(points(midpoint), material.boundary) <= tolerance
    origin = np.where(use_midpoint[:, None], midpoint, points_on_curve)
    bounds = material.bounds
    scale = max(bounds[2]-bounds[0], bounds[3]-bounds[1])
    edge_length = np.linalg.norm(next_points-points_on_curve, axis=1)
    step = np.minimum(scale*1e-6, edge_length*.01)
    if np.any(step <= 0):
        raise ValueError("degenerate sampled border segment; split or reparameterize the curve")
    plus = covers(material, points(origin+step[:, None]*vectors))
    minus = covers(material, points(origin-step[:, None]*vectors))
    ambiguous = plus == minus
    if np.any(ambiguous):
        # Clipping/intersection may remove the midpoint of a sampled segment.
        plus[ambiguous] = covers(material, points(points_on_curve[ambiguous]+step[ambiguous, None]*vectors[ambiguous]))
        minus[ambiguous] = covers(material, points(points_on_curve[ambiguous]-step[ambiguous, None]*vectors[ambiguous]))
    if np.any(plus == minus):
        raise ValueError("cannot determine outward normal at an intersecting/singular border; split the arc or supply normal explicitly")
    return vectors*np.where(minus, 1., -1.)[:, None]


def generate_borders(source, *, interior, method, seed, boundary_distance, max_candidates,
                     normal_overrides):
    if isinstance(source, RBFMesh):
        mesh = RBFMesh(*source.borders, abs_tol=source.abs_tol)
    else:
        borders = [source] if isinstance(source, Border) else list(source)
        if not borders or any(not isinstance(b, Border) for b in borders):
            raise TypeError("a border collection must contain sampled Border objects")
        mesh = RBFMesh(*borders)
    material = unary_union(mesh.region_polygons)
    if material.is_empty or material.area <= 0:
        raise ValueError("borders must enclose positive-area material; check contour orientation")
    x0, y0, x1, y1 = material.bounds
    inside = planar_interior(interior, ((x0, x1), (y0, y1)),
        lambda p: intersects_xy(material, p[:, 0], p[:, 1]), material,
        method, seed, boundary_distance, max_candidates)
    coordinates = [tuple(p) for p in inside]
    lookup = {p: i for i, p in enumerate(coordinates)}
    groups, interfaces, normals = {}, {}, {}
    region_lines = unary_union([region.boundary for region in mesh.region_polygons])
    for border in mesh.borders:
        if not border.is_border:
            continue
        q = np.linspace(border.t_start, border.t_end, abs(border.n_segments)+1)
        if border.reverse:
            q = q[::-1]
        all_points = np.array([border.parametric_function(t) for t in q])
        p = all_points[:-1]
        exterior = distance(points(p), material.boundary) < mesh.abs_tol
        retained = distance(points(p), region_lines) < mesh.abs_tol
        normal_values = None
        if np.any(exterior) and border.label not in normal_overrides:
            normal_values = border._geometry.normals(q[:-1][exterior])
            if border.normal is None:
                normal_values = _outward(material, p[exterior], all_points[1:][exterior],
                                          normal_values, mesh.abs_tol)
        normal_index = 0
        for point, keep, on_boundary in zip(p, retained, exterior):
            if not keep:
                continue
            key = tuple(point)
            if key not in lookup:
                lookup[key] = len(coordinates)
                coordinates.append(key)
            index = lookup[key]
            dest = groups if on_boundary else interfaces
            other = interfaces if on_boundary else groups
            if border.label in other:
                raise ValueError("use separate labels for exterior boundaries and internal interfaces")
            ids = dest.setdefault(border.label, [])
            if index not in ids:
                ids.append(index)
                if on_boundary and normal_values is not None:
                    normals.setdefault(border.label, []).append(normal_values[normal_index])
            elif on_boundary and normal_values is not None:
                previous = normals[border.label][ids.index(index)]
                if not np.allclose(previous, normal_values[normal_index]):
                    raise ValueError("conflicting normals for one point and boundary label; split labels at this junction")
            if on_boundary:
                normal_index += 1
    p = np.asarray(coordinates)
    regions = {i: np.flatnonzero(intersects_xy(region, p[:, 0], p[:, 1]))
               for i, region in enumerate(mesh.region_polygons)}
    return PointCloud(p, groups, normals, interfaces=interfaces, regions=regions)
