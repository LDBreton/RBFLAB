"""Arc-length/radial Padua maps for domains with separated star-shaped holes."""
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import contains_xy, distance, points as shapely_points

from ..geometry import Border, ParametricBoundary, ParametricDomain, PointCloud, find_polygons
from .polygons import RBFMesh
from .padua import padua_points


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _as_domain(source):
    """Rebuild directed border contours with dense curve geometry, without mutation."""
    if isinstance(source, ParametricDomain):
        return source
    borders = source.borders if isinstance(source, RBFMesh) else (
        [source] if isinstance(source, Border) else source)
    if not isinstance(borders, (list, tuple)) or not borders or any(
            not isinstance(b, Border) or b.n_segments is None for b in borders):
        raise TypeError("mapped_padua requires a ParametricDomain or sampled Border collection")
    if any(not b.is_border for b in borders):
        raise ValueError("mapped_padua requires all contour borders to be retained")
    contours = find_polygons(borders)
    if sum(map(len, contours)) != len(borders):
        raise ValueError("mapped_padua does not support shared/internal border contours")
    outer, holes = [], []
    for k, contour in enumerate(contours):
        arcs = []
        for j, border in enumerate(contour):
            a, b = border.t_start, border.t_end
            if border.reverse:
                a, b = b, a
            curve = border._geometry
            arcs.append(ParametricBoundary(
                lambda t, c=curve, a=a, b=b: c.point(a+(b-a)*t),
                tangent=lambda t, c=curve, a=a, b=b: (b-a)*c.tangent(a+(b-a)*t),
                interval=(0., 1.), label=(k, j)))
        ring = Polygon(np.concatenate([arc._table()[1][:-1] for arc in arcs]))
        (outer if ring.exterior.is_ccw else holes).append(arcs)
    if len(outer) != 1:
        raise ValueError("mapped_padua requires exactly one outer contour and disjoint holes")
    return ParametricDomain(outer[0], holes=holes)


class _Contour:
    def __init__(self, arcs):
        self.arcs = arcs
        self.tables = [arc._table() for arc in arcs]
        self.lengths = np.array([table[2][-1] for table in self.tables])
        self.ends = np.r_[0., np.cumsum(self.lengths)]
        self.ring = Polygon(np.concatenate([table[1][:-1] for table in self.tables]))

    def evaluate(self, fraction):
        length = np.asarray(fraction)*self.ends[-1]
        indices = np.minimum(np.searchsorted(self.ends, length, side="right")-1,
                             len(self.arcs)-1)
        result = np.empty((len(length), 2))
        for j, (arc, (t, _, s)) in enumerate(zip(self.arcs, self.tables)):
            take = indices == j
            parameter = np.interp(length[take]-self.ends[j], s, t)
            result[take] = [arc.evaluate(v) for v in parameter] if take.any() else np.empty((0, 2))
        return result


def _clip(vertices, normal, offset, origin):
    """Clip a convex enclosing polygon by a center-relative half plane."""
    result = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        da, db = (a-origin)@normal-offset, (b-origin)@normal-offset
        if da <= 0:
            result.append(a)
        if (da <= 0) != (db <= 0):
            result.append(a+(b-a)*da/(da-db))
    return np.asarray(result).reshape(-1, 2)


class _StarRing:
    def __init__(self, polygon, center, scale, name):
        if polygon.geom_type != "Polygon" or polygon.interiors or not polygon.contains(Point(center)):
            raise ValueError(f"{name} must be a single star-shaped region containing its center")
        self.center = center
        v = np.asarray(orient(polygon, sign=1).exterior.coords)[:-1]-center
        # Intersections through an existing vertex may emit a second copy a
        # few ulps away. Remove only consecutive geometric duplicates.
        v = v[np.linalg.norm(v-np.roll(v, 1, axis=0), axis=1) > scale*1e-13]
        edge = np.roll(v, -1, axis=0)-v
        # Positive signed distance to every oriented edge places the center
        # strictly in the polygon kernel; this prevents radial folds.
        signed_distance = _cross(v, edge)/np.linalg.norm(edge, axis=1)
        if np.min(signed_distance) <= scale*1e-10:
            raise ValueError(f"{name} is not strictly star-shaped about its center; choose centers or different geometry")
        angles = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2*np.pi)
        start = np.argmin(angles)
        self.v = np.roll(v, -start, axis=0)
        self.angles = np.roll(angles, -start)
        if np.any(np.diff(self.angles) <= 0):
            raise ValueError(f"{name} has a folded radial parametrization")

    def radius(self, directions):
        angle = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2*np.pi)
        j = (np.searchsorted(self.angles, angle, side="right")-1) % len(self.v)
        a, b = self.v[j], self.v[(j+1) % len(self.v)]
        return _cross(a, b)/_cross(directions, b-a)


def _dividers(outer, centers, scale):
    """Return each interior Voronoi edge once, clipped to the outer contour."""
    result = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            ci, cj = centers[i], centers[j]
            delta = cj-ci
            direction = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
            midpoint = (ci+cj)/2
            lo, hi = -4*scale, 4*scale
            for k, ck in enumerate(centers):
                if k in (i, j):
                    continue
                normal = ck-ci
                rhs = normal@normal/2-(midpoint-ci)@normal
                slope = direction@normal
                if abs(slope) < scale*1e-14:
                    if rhs < 0:
                        hi = lo-1
                        break
                elif slope > 0:
                    hi = min(hi, rhs/slope)
                else:
                    lo = max(lo, rhs/slope)
            if hi-lo <= scale*1e-12:
                continue
            cut = outer.intersection(LineString([midpoint+lo*direction, midpoint+hi*direction]))
            parts = [cut] if cut.geom_type == "LineString" else getattr(cut, "geoms", [])
            result.extend(part for part in parts if part.geom_type == "LineString" and part.length > scale*1e-12)
    return result


def mapped_padua(domain, *, degree, boundary=None, clustering=2., centers=None,
                 min_spacing=0., max_candidates=1000000) -> PointCloud:
    """Map Padua nodes around holes using arc length and graded radial patches.

    Args:
        domain (ParametricDomain | Border | list | RBFMesh): Input geometry. One
            exterior and at least one hole; internal material interfaces unsupported.
        degree (int): Positive Padua degree per hole patch, not a total node count.
        boundary (int | dict | None): Existing boundary counts for ParametricDomain. Omit for borders.
        clustering (float): In [0, 8]; zero gives linear radial interpolation.
            Increasing this value compresses nodes toward holes via expm1(beta*r).
        centers (array-like | None): Optional (number_of_holes, 2) centers in contour order; default
            hole polygon centroids. Each center must lie in the strict kernel
            of its hole and surrounding nearest-center cell.
        min_spacing (float): Optional physical minimum allowed separation. Raises
            if violated; never silently thins the mapped cloud.
        max_candidates (int): Upper bound on reference, divider and boundary nodes.

    The Padua cosine index is mapped to uniform periodic arc length around
    each hole, removing artificial crowding at the angular seam. Rays
    from its center reach the outer boundary or an internal Voronoi divider.
    The second coordinate spans this interval with exponential grading. The
    cells partition the whole exterior; each must contain its complete hole.
    Unsupported geometries raise rather than falling back to clipping.

    Physical boundaries keep their existing sampler, labels and normals. The
    periodic seam is half-open; radial endpoint nodes are replaced by physical
    boundary samples and shared divider samples. ``regions['patch_i']`` records
    nearest-center membership. Dividers are not physical boundaries or PDE
    interfaces. No triangle connectivity is returned.

    Mapping uses dense polygonal curve tables (2048 segments per arc), including
    for borders. Checks are geometric checks at this resolution, not a proof
    for unresolved curve features. Clustering can worsen RBF conditioning; no
    polynomial interpolation or PDE accuracy guarantee follows from this map.
    """
    from .generate import generate
    if type(degree) is not int or degree < 1:
        raise ValueError("degree must be a positive integer")
    if not np.isscalar(clustering) or not np.isfinite(clustering) or not 0 <= clustering <= 8:
        raise ValueError("clustering must be finite and in [0, 8]")
    if not np.isscalar(min_spacing) or not np.isfinite(min_spacing) or min_spacing < 0:
        raise ValueError("min_spacing must be finite and nonnegative")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    geometry = _as_domain(domain)
    holes = [_Contour(arcs) for arcs in geometry.contours[1:]]
    if not holes:
        raise ValueError("mapped_padua requires at least one hole")
    if len(holes)*(degree+1)*(degree+2)//2 > max_candidates:
        raise ValueError("mapped Padua reference grids exceed max_candidates")
    outer = Polygon(geometry.polygon.exterior)
    x0, y0, x1, y1 = outer.bounds
    scale = max(x1-x0, y1-y0)
    tolerance = scale*1e-10
    if centers is None:
        centers = np.array([h.ring.centroid.coords[0] for h in holes])
    else:
        centers = np.asarray(centers, dtype=float)
    if centers.shape != (len(holes), 2) or not np.isfinite(centers).all():
        raise ValueError("centers must have finite shape (number_of_holes, 2)")
    if len(np.unique(centers, axis=0)) != len(centers):
        raise ValueError("hole centers must be distinct")
    cells, stars = [], []
    for i, (hole, center) in enumerate(zip(holes, centers)):
        _StarRing(hole.ring, center, scale, f"hole {i}")
        vertices = np.array([[x0-scale, y0-scale], [x1+scale, y0-scale],
                             [x1+scale, y1+scale], [x0-scale, y1+scale]])
        for j, other in enumerate(centers):
            if i != j:
                normal = other-center
                vertices = _clip(vertices, normal, normal@normal/2, center)
        cell = outer.intersection(Polygon(vertices))
        star = _StarRing(cell, center, scale, f"patch {i}")
        if not cell.covers(hole.ring) or cell.boundary.distance(hole.ring) <= tolerance:
            raise ValueError(f"hole {i} crosses its patch boundary; choose centers or different geometry")
        cells.append(cell)
        stars.append(star)
    if unary_union(cells).symmetric_difference(outer).area > scale**2*1e-9:
        raise ValueError("mapped patches do not cover the outer domain")
    if sum(c.area for c in cells)-outer.area > scale**2*1e-9:
        raise ValueError("mapped patches overlap")

    # Preserve the established physical boundary contract, including overrides
    # supplied on the curves themselves. No interior random sampling occurs.
    base = generate(domain, interior=0, boundary=boundary)
    if base.interfaces:
        raise ValueError("mapped_padua does not support internal material interfaces")
    reference = padua_points(degree)
    s = np.arccos(reference[:, 0])/np.pi
    r = (1-reference[:, 1])/2
    take = (s < 1) & (r > 0) & (r < 1)
    s, r = s[take], r[take]
    weight = np.expm1(clustering*r)/np.expm1(clustering) if clustering else r
    mapped = []
    for i, (hole, center, star) in enumerate(zip(holes, centers, stars)):
        wall = hole.evaluate(s)
        vectors = wall-center
        inner_radius = np.linalg.norm(vectors, axis=1)
        direction = vectors/inner_radius[:, None]
        gap = star.radius(direction)-inner_radius
        if np.any(gap <= tolerance):
            raise ValueError(f"patch {i} has a folded or vanishing radial span")
        mapped.append(wall+(weight*gap)[:, None]*direction)

    patch_node_count = sum(map(len, mapped))
    for edge in _dividers(outer, centers, scale):
        count = max(2, int(np.ceil(edge.length/scale*degree*2))+1)
        if sum(map(len, mapped))+len(base.points)+count > max_candidates:
            raise ValueError("mapped Padua cloud exceeds max_candidates")
        samples = np.array([edge.interpolate(t, normalized=True).coords[0]
                            for t in np.linspace(0, 1, count)])
        # Endpoints on physical curves are owned by the boundary sampler.
        keep = distance(shapely_points(samples), geometry.polygon.boundary) > tolerance
        mapped.append(samples[keep])
    interior = np.concatenate(mapped)
    if len(interior)+len(base.points) > max_candidates:
        raise ValueError("mapped Padua cloud exceeds max_candidates")
    if not np.all(contains_xy(geometry.polygon, interior[:, 0], interior[:, 1])):
        raise ValueError("mapped points leave the material at the curve-table resolution; reduce degree/clustering or refine geometry")
    # Merge coincident divider junctions. Physical boundary coordinates take
    # precedence, while genuine near-collisions are checked below.
    all_points = np.concatenate([base.points, interior])
    tree = cKDTree(all_points)
    keep = np.ones(len(all_points), dtype=bool)
    for a, b in sorted(tree.query_pairs(tolerance)):
        if b < len(base.points)+patch_node_count:
            raise ValueError("mapped or physical boundary samples are too close; reduce degree/clustering or boundary resolution")
        if keep[a]:
            keep[b] = False
    all_points = all_points[keep]
    minimum = cKDTree(all_points).query(all_points, k=2)[0][:, 1].min()
    if minimum < max(min_spacing, 10*tolerance):
        raise ValueError(f"mapped node separation {minimum:.6g} is too small; reduce degree/clustering or min_spacing")
    distances = np.linalg.norm(all_points[:, None, :]-centers[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    regions = {f"patch_{i}": np.flatnonzero(distances[:, i] <= nearest+tolerance)
               for i in range(len(holes))}
    return PointCloud(all_points, base.boundary, base.normals, regions=regions)
