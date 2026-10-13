"""Optional constrained triangulation with sizing inherited from boundary spacing."""
from numbers import Integral
import numpy as np
from shapely.geometry import Polygon
from ..geometry import Border, ParametricDomain, TriangleMesh2D, find_polygons
from .polygons import RBFMesh


def _boundary_pslg(source, boundary):
    """Sample oriented loops into a labeled planar straight-line graph."""
    loops = []
    if isinstance(source, ParametricDomain):
        groups = source.sample_boundary(100 if boundary is None else boundary)
        loops = [[(arc.label, groups[arc.label][0]) for arc in contour]
                 for contour in source.contours]
    else:
        if boundary is not None:
            raise ValueError("omit boundary for Border geometry; choose resolution with border(n)")
        borders = source.borders if isinstance(source, RBFMesh) else (
            [source] if isinstance(source, Border) else source)
        if not isinstance(borders, (list, tuple)) or not borders or any(
                not isinstance(b, Border) or b.n_segments is None for b in borders):
            raise TypeError("triangulate requires a ParametricDomain or sampled Border collection")
        if any(not b.is_border for b in borders):
            raise ValueError("triangulate requires all contour borders to be retained")
        contours = find_polygons(borders)
        if sum(map(len, contours)) != len(borders):
            raise ValueError("triangulate does not support shared/internal border contours")
        outer, holes = [], []
        for contour in contours:
            parts = [(b.label, np.array([(p.x, p.y) for p in b.generate_points()]))
                     for b in contour]
            ring = Polygon(np.concatenate([p for _, p in parts]))
            (outer if ring.exterior.is_ccw else holes).append(parts)
        if len(outer) != 1:
            raise ValueError("triangulate requires exactly one outer contour and disjoint holes")
        loops = outer+holes
    vertices, facets, markers, labels, rings = [], [], [], [], []
    for loop in loops:
        start = len(vertices)
        for label, coordinates in loop:
            if isinstance(label, bool) or not isinstance(label, (str, Integral)):
                raise ValueError("triangulation labels must be strings or integers")
            if label not in labels:
                labels.append(label)
            marker = labels.index(label)+1
            for p in coordinates:
                index = len(vertices)
                vertices.append(p)
                facets.append([index, index+1])
                markers.append(marker)
        facets[-1][1] = start
        ring = np.array(vertices[start:])
        if len(ring) < 3:
            raise ValueError("each contour needs at least three boundary samples")
        rings.append(ring)
    polygon = Polygon(rings[0], rings[1:])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("sampled contours must define valid positive-area material; refine the borders")
    exterior = Polygon(rings[0])
    holes = [Polygon(ring) for ring in rings[1:]]
    if any(not exterior.contains(hole) or exterior.boundary.intersects(hole) for hole in holes):
        raise ValueError("holes must lie strictly inside the sampled outer contour")
    vertices, facets = np.asarray(vertices), np.asarray(facets)
    if len(np.unique(vertices, axis=0)) != len(vertices):
        raise ValueError("sampled contours contain duplicate boundary nodes")
    return vertices, facets, np.asarray(markers), labels, polygon, holes


def triangulate(domain, *, boundary=None, growth=.25, max_spacing=None,
                min_angle=25., fixed_boundary=True) -> TriangleMesh2D:
    """Generate a constrained triangle mesh with boundary-driven point density.

    Args:
        domain (ParametricDomain | Border | list | RBFMesh): Input geometry. One
            exterior plus disjoint holes, including non-star-shaped geometry.
        boundary (int | dict | None): Total/per-label arc-length counts for ParametricDomain. Omit
            for Border inputs, which use their existing signed counts.
        growth (float): Positive rate at which target edge spacing grows away
            from boundary segments. Smaller values extend fine resolution farther.
        max_spacing (float | None): Optional positive cap on target interior
            edge spacing; the default lets spacing grow with boundary distance.
        min_angle (float): Triangle's quality target in degrees, in [0, 28].
            Geometric corners and a fixed boundary can prevent meeting it.
        fixed_boundary (bool): Keep exactly the supplied boundary nodes by
            default. False permits additional nodes on the sampled segments.

    The sizing field is h(x) = min_e(length(e) + growth*distance(x,e)), capped
    by max_spacing when provided. Triangle refinement requests area <=
    sqrt(3)/4*h(centroid)^2. Increasing samples on an arc therefore refines the
    adjacent interior. This is a local size target, not an exact edge-length bound.

    Uses Triangle through the optional MeshPy dependency; install
    ``rbflab[triangulation]``. This offers a FreeFEM-like boundary-driven workflow,
    not the FreeFEM/BAMG algorithm. No dependency is imported until called.

    Curves are approximated by the sampled straight boundary segments. Labels
    attach to edges, so corner vertices can belong to multiple boundary groups.
    ``geometry.staggered_clouds(mesh).vertices`` produces a PDE PointCloud with
    the mesh's piecewise-linear normals and connectivity. Explicit analytic
    normal overrides can be supplied to staggered_clouds. Internal material
    interfaces and overlapping positive contours are currently unsupported.
    """
    if not np.isscalar(growth) or not np.isfinite(growth) or growth <= 0:
        raise ValueError("growth must be finite and positive")
    if max_spacing is not None and (not np.isscalar(max_spacing) or
                                   not np.isfinite(max_spacing) or max_spacing <= 0):
        raise ValueError("max_spacing must be finite and positive")
    if not np.isscalar(min_angle) or not np.isfinite(min_angle) or not 0 <= min_angle <= 28:
        raise ValueError("min_angle must be finite and in [0, 28]")
    if type(fixed_boundary) is not bool:
        raise ValueError("fixed_boundary must be boolean")
    vertices, facets, markers, labels, polygon, holes = _boundary_pslg(domain, boundary)
    a, b = vertices[facets[:, 0]], vertices[facets[:, 1]]
    edges = b-a
    lengths = np.linalg.norm(edges, axis=1)
    scale = max(polygon.bounds[2]-polygon.bounds[0], polygon.bounds[3]-polygon.bounds[1])
    if np.min(lengths) <= 1e-12*scale:
        raise ValueError("boundary segments are too short relative to domain size")
    try:
        from meshpy.triangle import MeshInfo, build
    except ImportError as exc:
        raise ImportError("triangulate requires MeshPy; install rbflab[triangulation]") from exc
    info = MeshInfo()
    info.set_points(vertices.tolist())
    info.set_facets(facets.tolist(), facet_markers=markers.tolist())
    info.set_holes([list(h.representative_point().coords[0]) for h in holes])
    squared = lengths**2

    def needs_refinement(triangle, area):
        centroid = np.asarray(triangle).mean(axis=0)
        fraction = np.clip(np.sum((centroid-a)*edges, axis=1)/squared, 0, 1)
        distances = np.linalg.norm(centroid-a-fraction[:, None]*edges, axis=1)
        spacing = np.min(lengths+growth*distances)
        if max_spacing is not None:
            spacing = min(spacing, max_spacing)
        return bool(area > np.sqrt(3)/4*spacing**2)

    output = build(info, refinement_func=needs_refinement, min_angle=float(min_angle),
                   allow_boundary_steiner=not fixed_boundary)
    p, t = np.array(output.points), np.array(output.elements, dtype=int)
    boundary_edges = np.array(output.facets, dtype=int)
    edge_markers = np.array(output.facet_markers)
    groups = {label: boundary_edges[edge_markers == i+1] for i, label in enumerate(labels)}
    result = TriangleMesh2D(p, t, groups)
    u, v = p[t[:, 1]]-p[t[:, 0]], p[t[:, 2]]-p[t[:, 0]]
    areas = np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0])/2
    if not np.isclose(areas.sum(), polygon.area, rtol=1e-9, atol=scale**2*1e-12):
        raise ValueError("triangulation does not cover the sampled material")
    return result
