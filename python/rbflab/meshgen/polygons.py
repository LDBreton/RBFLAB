from ..geometry._borders import MeshPoint, find_polygons, Border
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, polygonize
from shapely.validation import explain_validity
from shapely import prepare, contains_xy
import random
import math
from numbers import Integral


class RBFMesh:
    """
    RBFMesh class for generating random points within polygons.

    Args:
        *borders: Variable length argument list of Border objects representing the borders of the polygons.
        abs_tol (float, optional): Absolute tolerance for geometric calculations. Defaults to 1e-04.

    Attributes:
        borders (list): List of Border objects representing the borders of the polygons.
        Points (list): List of generated MeshPoint objects.
        Boundary_Points (list): Sampled border points on external or internal region boundaries.
        outer_polygons (list): List of outer Polygon objects.
        holes_polygons (list): List of hole Polygon objects.
        abs_tol (float): Absolute tolerance for geometric calculations.
    Methods:
        generate_points(num_points): Generates random points within the polygons.

        process_polygons(): Validates contours and partitions the domain.
    """

    def __init__(self, *borders: Border, abs_tol=1e-04):
        self.borders = list(borders)
        self.Points = []
        self.Boundary_Points = []
        self.outer_polygons = []
        self.holes_polygons = []
        self.region_polygons = []
        self.abs_tol = abs_tol
        self.process_polygons()  # Process polygons during initialization

    def process_polygons(self):
        """
        Processes polygons to prepare them for point generation by classifying
        them into outer polygons and holes, and then refining them to ensure
        that they are properly nested and non-overlapping. This method sets up
        the initial geometric configuration by finding polygons based on the
        provided borders, classifying them based on orientation, and generating
        points along these borders.

        The method performs several key operations:
        1. Identifies and orients polygons based on the input borders, distinguishing
           between counter-clockwise (outer polygons) and clockwise (holes).
        2. Generates points along each border and collects these points, noting
           which are on boundary borders.
        3. Resolve overlaps among multiple polygons by calculating unique and intersecting areas.
        4. Subtracts hole polygons from outer polygons to finalize distinct regions.
        5. Combines region boundaries, preserving internal interfaces, to filter
           border samples that remain after hole subtraction.

        Modifies:
            self.outer_polygons: List of shapely.geometry.Polygon objects representing the outer boundaries.
            self.holes_polygons: List of shapely.geometry.Polygon objects representing the holes.
            self.region_polygons: List of shapely.geometry.Polygon objects representing the final regions
                                  after subtraction of holes from the outer polygons.
            self.Boundary_Points: Samples on any surviving region boundary,
                                  including internal interfaces, within the absolute tolerance.

        This setup is crucial for ensuring that the subsequent point generation by `generate_points`
        occurs within properly defined and non-overlapping geometric regions.
        """
        polygons = find_polygons(self.borders, self.abs_tol)

        # Sample each input border once, even if multiple contours share it.
        sampled = {id(border): border.generate_points() for border in self.borders}
        polygons_with_points = []
        tentative_boundary_points = [point for border in self.borders if border.is_border
                                     for point in sampled[id(border)]]

        for polygon in polygons:
            polygon_points = []
            for border in polygon:
                border_point = sampled[id(border)]
                polygon_points.extend([(p.x, p.y) for p in border_point])  # Add to polygon definition
            polygons_with_points.append(polygon_points)

        polygons = []
        for index, coordinates in enumerate(polygons_with_points):
            if len(set(coordinates)) < 3:
                raise ValueError(f'Contour {index} needs at least three distinct sampled points; '
                                 'increase its border segment counts')
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                raise ValueError(f'Invalid contour {index}: {explain_validity(polygon)}')
            if not math.isfinite(polygon.area) or polygon.area <= 0:
                raise ValueError(f'Contour {index} must enclose a finite positive area')
            polygons.append(polygon)

        # Determine orientation and classify as outer or holes
        self.outer_polygons = [poly for poly in polygons if poly.exterior.is_ccw]
        self.holes_polygons = [poly for poly in polygons if not poly.exterior.is_ccw]

        # Partition all positive contours in one pass, including nested ones.
        self.outer_polygons = resolve_multiple_overlaps(self.outer_polygons)

        # Step 2: generate_regions
        self.region_polygons = generate_regions(self.outer_polygons, self.holes_polygons)

        # Union the boundary lines, not the region areas: merging areas would
        # erase internal interfaces and discard their explicitly requested samples.
        boundary_line = unary_union([region.boundary for region in self.region_polygons])
        self.Boundary_Points = [p for p in tentative_boundary_points if
                                boundary_line.distance(Point(p.x, p.y)) < self.abs_tol]

    def generate_points(self, num_points, boundary_distance=1.0e-5, *, append=True,
                        method='random', seed=None) -> list:
        """
        Generates random points within the polygons defined by the borders.

        Args:
            boundary_distance (float):  distance from generated point to the boundary
            num_points (int): Number of points to generate.
            append (bool): Keep existing interior points (default True).
                Set False to replace them after successful generation.
            method (str): 'random', 'halton', or 'sobol'.
            seed (int or None): Non-negative seed for a fresh sampler per call.
                Reusing a seed reproduces points; use append=False to replace them.

        Returns:
            list: List of generated MeshPoint objects.
        """
        # Step 1: Calculate points allocation
        points_allocation = calculate_point_allocation(self.region_polygons, num_points)

        # Step 2: Generate points
        points = generate_points_within_polygons(
            self.region_polygons, points_allocation, boundary_distance, method=method, seed=seed)
        if append:
            self.Points.extend(points)
        else:
            self.Points[:] = points

        return self.Points


def _polygon_parts(geometry):
    """Return valid positive-area Polygon components; ignore line/point contacts."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        if not geometry.is_valid:
            raise ValueError(f'Invalid polygon: {explain_validity(geometry)}')
        if not math.isfinite(geometry.area):
            raise ValueError('Polygon area must be finite')
        return [geometry] if geometry.area > 0 else []
    if hasattr(geometry, 'geoms'):
        return [part for child in geometry.geoms for part in _polygon_parts(child)]
    return []


def resolve_multiple_overlaps(polygons):
    """
    Partition nested and overlapping polygons into disjoint positive-area faces.
    Shared edges and point contacts are not regions. Existing holes are preserved.

    Args:
        polygons (list of shapely.geometry.Polygon): List of Polygon objects that might overlap.

    Returns:
        list of shapely.geometry.Polygon: List of disjoint Polygon objects including unique areas and individual intersection areas without duplicates.
    """
    polygons = [part for polygon in polygons for part in _polygon_parts(polygon)]
    if not polygons:
        return []
    # Noding boundaries before polygonizing creates disjoint planar faces.
    linework = unary_union([polygon.boundary for polygon in polygons])
    domain = unary_union(polygons)
    return [face for face in polygonize(linework)
            if face.area > 0 and domain.covers(face.representative_point())]


def exclude_nested_polygons(outer_polygons) -> list:
    """
    Subtract contained polygons using the original containment relationships.
    This handles nesting and duplicates; use resolve_multiple_overlaps for
    partially overlapping inputs.

    Args:
        outer_polygons (list): List of outer Polygon objects.

    Returns:
        list: Polygon components with contained areas removed from their parents.
    """
    polygons = [part for polygon in outer_polygons for part in _polygon_parts(polygon)]
    unique = []
    for polygon in sorted(polygons, key=lambda p: p.area, reverse=True):
        if not any(polygon.equals(other) for other in unique):
            unique.append(polygon)
    result = []
    # Compare against the original polygons, not already-subtracted shells.
    for i, polygon in enumerate(unique):
        children = [other for other in unique[i + 1:] if polygon.covers(other)]
        result.extend(_polygon_parts(polygon.difference(unary_union(children))))
    return result


def calculate_point_allocation(region_polygons, num_points) -> list:
    """
    Calculates the point allocation for each region_polygons based on their area.

    Args:
        region_polygons (list): List of outer Polygon objects.
        num_points (int): Number of points to allocate.

    Returns:
        list: List of integers representing the point allocation for each outer polygon.
    """
    if isinstance(num_points, bool) or not isinstance(num_points, Integral) or num_points < 0:
        raise ValueError('num_points must be a non-negative integer')
    areas = [poly.area for poly in region_polygons]
    if any(not math.isfinite(area) or area < 0 for area in areas):
        raise ValueError('Region areas must be finite and non-negative')
    if num_points == 0:
        return [0] * len(areas)
    total_area = sum(areas)
    if not math.isfinite(total_area) or total_area <= 0:
        raise ValueError('Cannot generate points without a positive-area region')
    quotas = [area / total_area * num_points for area in areas]
    allocation = [math.floor(quota) for quota in quotas]
    # Largest remainders preserve the requested total, with stable tie-breaking.
    remaining = int(num_points) - sum(allocation)
    order = sorted(range(len(areas)), key=lambda i: quotas[i] - allocation[i], reverse=True)
    for i in order[:remaining]:
        allocation[i] += 1
    return allocation


def generate_regions(outer_polygons, hole_polygons) -> list:
    """
    Subtract all holes without mutating the input list. Fully removed regions
    disappear, and disconnected results become separate Polygon components.

    Args:
        outer_polygons (list): List of outer Polygon objects.
        hole_polygons (list): List of hole Polygon objects.

    Returns:
        list: List of modified outer Polygon objects.
    """
    holes = unary_union([part for hole in hole_polygons for part in _polygon_parts(hole)])
    return [part for polygon in outer_polygons
            for outer in _polygon_parts(polygon)
            for part in _polygon_parts(outer.difference(holes))]


def generate_points_within_polygons(region_polygons, points_allocation, boundary_distance=1.0e-5,
                                    *, method='random', seed=None) -> list:
    """
    Generates random points within the outer polygons.

    Args:
        region_polygons (list): List of outer Polygon objects.
        points_allocation (list): List of integers representing the point allocation for each outer polygon.
        boundary_distance (float, optional): Distance to buffer the polygons. Defaults to 1.0e-5.
        method (str): 'random' (default), scrambled 'halton', or scrambled 'sobol'.
        seed (int or None): Non-negative seed, reset on each call. With random
            sampling and no seed, the global random state is used as before.

    QMC candidates are mapped from the unit square to each region's bounding
    box and rejected outside the buffered region. This preserves exact counts,
    but clipping/truncation does not preserve Sobol's power-of-two balance.

    Returns:
        list: List of generated MeshPoint objects.
    """
    if method not in ('random', 'halton', 'sobol'):
        raise ValueError("method must be 'random', 'halton', or 'sobol'")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('seed must be a non-negative integer or None')
        seed = int(seed)
    if not math.isfinite(boundary_distance) or boundary_distance < 0:
        raise ValueError('boundary_distance must be finite and non-negative')
    if len(region_polygons) != len(points_allocation):
        raise ValueError('Each region must have a point allocation')
    prepared_regions = []
    for poly, num_pts in zip(region_polygons, points_allocation):
        if isinstance(num_pts, bool) or not isinstance(num_pts, Integral) or num_pts < 0:
            raise ValueError('Point allocations must be non-negative integers')
        shrunk = poly.buffer(-boundary_distance) if num_pts else poly
        if num_pts and (shrunk.is_empty or not shrunk.is_valid or
                        not math.isfinite(shrunk.area) or shrunk.area <= 0):
            raise ValueError('boundary_distance leaves no valid sampling area in a requested region')
        prepared_regions.append(shrunk)

    if not any(points_allocation):
        return []

    sampler = None
    rng = random if seed is None else random.Random(seed)
    if method != 'random':
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError('Halton and Sobol require SciPy. Reinstall with: pip install rbflab') from exc
        engine = qmc.Halton if method == 'halton' else qmc.Sobol
        # SciPy 1.15 renamed seed to rng; support older Python-compatible releases.
        import inspect
        options = {'rng': seed} if 'rng' in inspect.signature(engine).parameters else {'seed': seed}
        sampler = engine(d=2, scramble=True, **options)

    points = []
    total_points_generated = 0

    for i, (poly, num_pts) in enumerate(zip(prepared_regions, points_allocation)):
        if num_pts == 0:
            continue
        prepare(poly)  # Optional: prepare the polygon for faster operations if supported
        target_points_count = total_points_generated + num_pts
        min_x, min_y, max_x, max_y = poly.bounds

        while len(points) < target_points_count:
            if sampler is None:
                x = rng.uniform(min_x, max_x)
                y = rng.uniform(min_y, max_y)
                if poly.contains_properly(Point(x, y)):
                    points.append(MeshPoint(x, y, f'region {i + 1}', False))
            else:
                # Fixed power-of-two blocks keep memory bounded and avoid the
                # first-draw Sobol warning for arbitrary requested point counts.
                candidates = sampler.random(1024)
                x = min_x + candidates[:, 0] * (max_x - min_x)
                y = min_y + candidates[:, 1] * (max_y - min_y)
                inside = contains_xy(poly, x, y)
                remaining = target_points_count - len(points)
                points.extend(MeshPoint(px, py, f'region {i + 1}', False)
                              for px, py in zip(x[inside][:remaining], y[inside][:remaining]))

        total_points_generated += num_pts

    return points
