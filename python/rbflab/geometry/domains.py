"""Smooth planar domains with explicit boundary labels and outward normals.

Parametric domains use a dense polygon for membership tests. Primitive disk,
ellipse, annulus and flower domains use analytic membership instead.
"""
from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon as _Polygon
from shapely import intersects_xy


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _points(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("points must be finite (N, 2) coordinates")
    return p


def _center(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("center must contain two finite coordinates")
    return result


@dataclass(frozen=True)
class ParametricBoundary:
    """A regular oriented curve, evaluated with scalar parameter values.

    Args:
        curve: Callable t -> (x, y).
        tangent: Analytic derivative t -> (dx/dt, dy/dt); never guessed.
        label (object): Boundary-condition group name.
        interval: Increasing finite parameter endpoints.

    Curves in each domain contour must connect in the supplied order.
    Sampling is approximately uniform in arc length, excluding the endpoint.
    """
    curve: object
    tangent: object
    label: str = "boundary"
    interval: tuple = (0., 2*np.pi)

    def __post_init__(self):
        a = np.asarray(self.interval, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all() or a[0] >= a[1]:
            raise ValueError("interval must have two increasing finite endpoints")
        if not callable(self.curve) or not callable(self.tangent):
            raise TypeError("curve and tangent must be callable")

    def _table(self):
        t = np.linspace(*self.interval, 2049)
        points = np.asarray([self.curve(v) for v in t], dtype=float)
        if points.shape != (len(t), 2) or not np.isfinite(points).all():
            raise ValueError("curve must return finite 2D coordinates")
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if np.any(np.diff(length) <= 0):
            raise ValueError("curve must be regular and have positive arc length")
        return t, points, length

    def sample(self, count):
        """Return points and unit right-hand normals before contour orientation."""
        if type(count) is not int or count < 1:
            raise ValueError("boundary counts must be positive integers")
        t, _, length = self._table()
        q = np.interp(np.linspace(0, length[-1], count, endpoint=False), length, t)
        points = np.asarray([self.curve(v) for v in q], dtype=float)
        tangents = np.asarray([self.tangent(v) for v in q], dtype=float)
        if tangents.shape != points.shape or not np.isfinite(tangents).all():
            raise ValueError("tangent must return finite 2D vectors")
        norms = np.linalg.norm(tangents, axis=1)
        if np.any(norms == 0):
            raise ValueError("boundary tangent must be nonzero")
        return points, np.column_stack((tangents[:, 1], -tangents[:, 0]))/norms[:, None]


class ParametricDomain:
    """One closed outer contour with optional closed holes.

    Args:
        outer (ParametricBoundary | Sequence): ParametricBoundary or sequence of connected boundary arcs.
        holes (Sequence): Sequence of boundaries or arc sequences, one per hole.

    Orientation may be clockwise or counterclockwise: outward normals are
    corrected automatically, including on holes. Labels must be unique across
    arcs. Membership uses 2048 straight segments per arc; refine the geometric
    model separately when studying very small spatial errors.
    """
    dimension = 2

    def __init__(self, outer, holes=()):
        def contour(value):
            result = [value] if isinstance(value, ParametricBoundary) else list(value)
            if not result or any(not isinstance(b, ParametricBoundary) for b in result):
                raise TypeError("contours require ParametricBoundary objects")
            return result
        self.contours = [contour(outer), *(contour(h) for h in holes)]
        rings, self.boundaries, self._signs, self._lengths = [], {}, {}, {}
        for k, arcs in enumerate(self.contours):
            tables = [b._table() for b in arcs]
            for j in range(len(arcs)):
                if not np.allclose(tables[j][1][-1], tables[(j+1) % len(arcs)][1][0], atol=1e-10, rtol=0):
                    raise ValueError("boundary arcs must form a closed contour")
            ring = np.concatenate([v[1][:-1] for v in tables])
            polygon = _Polygon(ring)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError("contours must be simple and enclose positive area")
            sign = (1 if polygon.exterior.is_ccw else -1) * (1 if k == 0 else -1)
            for b, table in zip(arcs, tables):
                if b.label in self.boundaries:
                    raise ValueError("use a unique label for each boundary arc")
                self.boundaries[b.label] = b
                self._signs[b.label] = sign
                self._lengths[b.label] = table[2][-1]
            rings.append(ring)
        self.polygon = _Polygon(rings[0], rings[1:])
        if not self.polygon.is_valid or self.polygon.area <= 0:
            raise ValueError("holes must lie inside the outer contour without intersecting")
        x0, y0, x1, y1 = self.polygon.bounds
        self.bounds = ((x0, x1), (y0, y1))

    def contains_points(self, points):
        """Return membership in the closed computational domain."""
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("points must be finite (N, 2) coordinates")
        return intersects_xy(self.polygon, p[:, 0], p[:, 1])

    def sample_boundary(self, counts):
        """Return label -> (coordinates, outward unit normals)."""
        if type(counts) is int:
            labels = list(self.boundaries)
            if counts < len(labels)*3:
                raise ValueError("boundary total must allow at least three nodes per arc")
            lengths = np.array([self._lengths[k] for k in labels])
            allocation = (counts - 3*len(labels))*lengths/lengths.sum()
            n = np.floor(allocation).astype(int)+3
            for i in np.argsort(-(allocation % 1), kind="stable")[:counts-n.sum()]:
                n[i] += 1
            counts = dict(zip(labels, map(int, n)))
        if not isinstance(counts, dict):
            raise ValueError("boundary must be an integer or a count mapping")
        if set(counts) != set(self.boundaries):
            raise ValueError("boundary counts must match all domain labels")
        result = {}
        for label, b in self.boundaries.items():
            p, normal = b.sample(counts[label])
            result[label] = p, self._signs[label]*normal
        return result


class Ellipse(ParametricDomain):
    """Ellipse with semi-axes a, b and a counterclockwise angle in radians.

    Pass labels=(...) to split its parameter interval into equally sized arcs,
    useful for mixed boundary conditions. Each arc starts at its first node.
    """
    def __init__(self, a=1.3, b=0.8, *, center=(0., 0.), angle=0., labels=("boundary",)):
        self.a, self.b = _positive(a, "a"), _positive(b, "b")
        self.center = _center(center)
        if not np.isfinite(angle) or not labels:
            raise ValueError("finite angle and nonempty labels required")
        c, s = np.cos(angle), np.sin(angle)
        self.rotation = np.array([[c, -s], [s, c]])
        def curve(t): return self.center + self.rotation @ [a*np.cos(t), b*np.sin(t)]
        def tangent(t): return self.rotation @ [-a*np.sin(t), b*np.cos(t)]
        super().__init__([ParametricBoundary(curve, tangent, label, (2*np.pi*i/len(labels), 2*np.pi*(i+1)/len(labels))) for i, label in enumerate(labels)])
        extents = np.sqrt((self.rotation**2) @ np.array([a*a, b*b]))
        self.bounds = tuple(zip(self.center-extents, self.center+extents))

    def contains_points(self, points):
        p = (_points(points)-self.center) @ self.rotation
        return np.sum((p / [self.a, self.b])**2, axis=1) <= 1+1e-14


class Disk(Ellipse):
    """Circular 2D domain with analytic membership and outward normals."""
    def __init__(self, radius=1., *, center=(0., 0.), label="boundary"):
        self.radius = _positive(radius, "radius")
        super().__init__(radius, radius, center=center, labels=(label,))


class Annulus(ParametricDomain):
    """Concentric circular boundaries labeled inner and outer by default."""
    def __init__(self, inner_radius=0.4, outer_radius=1., *, center=(0., 0.)):
        self.inner_radius = _positive(inner_radius, "inner_radius")
        self.outer_radius = _positive(outer_radius, "outer_radius")
        if inner_radius >= outer_radius:
            raise ValueError("inner_radius must be smaller than outer_radius")
        self.center = _center(center)
        def circle(radius, label):
            return ParametricBoundary(lambda t: self.center+radius*np.array([np.cos(t), np.sin(t)]),
                                      lambda t: radius*np.array([-np.sin(t), np.cos(t)]), label)
        super().__init__(circle(outer_radius, "outer"), [circle(inner_radius, "inner")])

    def contains_points(self, points):
        radius = np.linalg.norm(_points(points)-self.center, axis=1)
        return (radius >= self.inner_radius-1e-14) & (radius <= self.outer_radius+1e-14)


class Flower(ParametricDomain):
    """Smooth star-shaped boundary r(theta)=radius*(1+amplitude*cos(petals*theta))."""
    def __init__(self, radius=1., amplitude=.18, petals=5, *, center=(0., 0.), label="boundary"):
        self.radius = _positive(radius, "radius")
        if not np.isfinite(amplitude) or not 0 <= amplitude < 1:
            raise ValueError("amplitude must lie in [0, 1)")
        if type(petals) is not int or petals < 1:
            raise ValueError("petals must be a positive integer")
        self.amplitude, self.petals, self.center = amplitude, petals, _center(center)
        def curve(t): return self.center + self._radius(t)*np.array([np.cos(t), np.sin(t)])
        def tangent(t):
            dr = -radius*amplitude*petals*np.sin(petals*t)
            return dr*np.array([np.cos(t), np.sin(t)])+self._radius(t)*np.array([-np.sin(t), np.cos(t)])
        super().__init__(ParametricBoundary(curve, tangent, label))
        extent=radius*(1+amplitude)
        self.bounds=tuple(zip(self.center-extent,self.center+extent))

    def _radius(self, theta): return self.radius*(1+self.amplitude*np.cos(self.petals*theta))

    def contains_points(self, points):
        p = _points(points)-self.center
        return np.linalg.norm(p, axis=1) <= self._radius(np.arctan2(p[:, 1], p[:, 0]))+1e-14


class Polygon(ParametricDomain):
    """Simple planar polygon with one named straight boundary per edge.

    Args:
        vertices (array-like): At least three distinct finite (x, y) vertices in contour
            order. A repeated closing vertex is accepted and removed.
        labels (Sequence[str] | None): One distinct label per edge; defaults to edge_0, edge_1, ... .

    Both orientations and concave polygons are supported. Self-intersections,
    repeated vertices and zero-area contours are rejected. Edge j joins vertex
    j to j+1 (cyclically); a corner belongs to the edge starting there. Its
    normal is that edge's outward normal, not an averaged corner normal.
    """
    def __init__(self, vertices, *, labels=None):
        vertices = _points(vertices).copy()
        if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        if len(vertices) < 3 or len(np.unique(vertices, axis=0)) != len(vertices):
            raise ValueError("polygon needs at least three distinct vertices")
        shape = _Polygon(vertices)
        if not shape.is_valid or shape.area <= 0:
            raise ValueError("polygon must be simple and enclose positive area")
        labels = tuple(f"edge_{i}" for i in range(len(vertices))) if labels is None else tuple(labels)
        if len(labels) != len(vertices) or len(set(labels)) != len(labels):
            raise ValueError("provide one unique label per polygon edge")
        arcs = []
        for i, label in enumerate(labels):
            start = vertices[i].copy()
            direction = vertices[(i+1) % len(vertices)]-start
            arcs.append(ParametricBoundary(
                lambda t, a=start, v=direction: a+t*v,
                lambda t, v=direction: v.copy(), label, (0., 1.)))
        super().__init__(arcs)
        # Exact straight edges need no dense polygon for membership.
        self.polygon = shape
        self.vertices = vertices


class Rectangle(Polygon):
    """Rectangle with optional rotation and individually labeled sides.

    Args:
        width (float): Positive length along the local x axis.
        height (float): Positive length along the local y axis.
        center (tuple): Two finite center coordinates.
        angle (float): Counterclockwise rotation in radians.
        labels (Sequence[str]): Labels for local bottom, right, top, left sides, in that order.

    Labels rotate with the rectangle. Junction ownership follows Polygon.
    """
    def __init__(self, width=2., height=1., *, center=(0., 0.), angle=0.,
                 labels=("bottom", "right", "top", "left")):
        width, height = _positive(width, "width"), _positive(height, "height")
        center = _center(center)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        vertices = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[width/2,height/2]
        super().__init__(vertices @ rotation.T + center, labels=labels)


def with_holes(outer, holes) -> ParametricDomain:
    """Cut named, disjoint holes from a planar domain without changing inputs.

    Args:
        outer (ParametricDomain): Outer domain, optionally already perforated.
        holes (dict): Nonempty mapping from a nonempty string name to a
            simply connected ParametricDomain such as Disk, Ellipse or Polygon.

    Returns:
        ParametricDomain: A new domain with preserved exterior labels. A
            single-arc hole gets its mapping key as label; multi-arc holes
            get name/edge_label. Hole normals point out of the material.

    Holes must lie strictly inside the existing material, without touching or
    overlapping each other. Existing holes are preserved. Topology and generic
    membership use the polygon representation of curved contours (2048
    segments per arc), while sampling retains the analytic curves/tangents.
    This is a hole-composition helper, not a general Boolean CAD engine.
    """
    from dataclasses import replace
    if not isinstance(outer, ParametricDomain):
        raise TypeError("outer must be a planar ParametricDomain")
    if not isinstance(holes, dict) or not holes:
        raise ValueError("holes must be a nonempty mapping of names to domains")
    contours, shapes = [], []
    for name, hole in holes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("hole names must be nonempty strings")
        if not isinstance(hole, ParametricDomain) or len(hole.contours) != 1:
            raise ValueError("each hole must be a simply connected planar domain")
        if (not outer.polygon.contains(hole.polygon)
                or not outer.polygon.boundary.disjoint(hole.polygon)):
            raise ValueError("holes must lie strictly inside the existing material")
        if any(not hole.polygon.disjoint(shape) for shape in shapes):
            raise ValueError("holes must not touch or overlap")
        arcs = hole.contours[0]
        contours.append([replace(b, label=name if len(arcs)==1 else f"{name}/{b.label}") for b in arcs])
        shapes.append(hole.polygon)
    return ParametricDomain(outer.contours[0], [*outer.contours[1:], *contours])
