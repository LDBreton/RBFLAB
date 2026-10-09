import numpy as np
from collections import deque
from numbers import Integral


def _coordinates(value, label):
    """Validate a parametric function result before geometric processing."""
    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Border {label!r} must return two finite coordinates') from exc
    if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
        raise ValueError(f'Border {label!r} must return two finite coordinates')
    return tuple(coordinates)


class MeshPoint:
    """Legacy labeled planar sample; use PointCloud for array-based PDE work."""
    def __init__(self, x: float, y: float, label=0, is_border=True):
        """
        Represents a point in the mesh.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            label (int or string, optional): The label of the point. Defaults to 0.
            is_border (bool, optional): Indicates if the point is on the border. Defaults to True.
        """
        self.x = x
        self.y = y
        self.label = label
        self.is_border = is_border


class Border:
    """Labeled parametric curve with FreeFEM-inspired segment-count syntax.

    Call border(n) before passing it to meshgen.RBFMesh. The sign of n selects
    orientation: a negative count reverses the parameter traversal. Sampling
    is uniform in the parameter, not in arc length. At a junction the curve
    starting there owns the point; the terminal endpoint is omitted. Calling border(n)
    updates this object in place, so use distinct Border objects for distinct
    arcs. See the parametric-borders tutorial for conversion to a PDE PointCloud.
    """
    def __init__(self, parametric_function, label, t_start, t_end, is_border=True):
        """
        Represents a border in the mesh.

        Args:
            parametric_function (function): The parametric function that defines the border.
            label (str or int): The label of the border.
            t_start (float): The start parameter value of the border.
            t_end (float): The end parameter value of the border.
            is_border (bool, optional): Indicates if the border is a boundary. Defaults to True.
        """
        self.parametric_function = parametric_function
        self.label = label
        self.t_start = t_start
        self.t_end = t_end
        # Calculate start and end points using the parametric function
        self.start_point = _coordinates(parametric_function(t_start), label)
        self.end_point = _coordinates(parametric_function(t_end), label)
        self.is_border = is_border
        self.n_segments = None
        self.reverse = False  # Attribute to control direction

    def __call__(self, n) -> "Border":
        """
        Sets the number of segments for the border.

        Args:
            n (int): The number of segments.

        Returns:
            Border: The updated Border object.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n == 0:
            raise ValueError('The number of border segments must be a non-zero integer')
        self.n_segments = int(n)
        self.reverse = n < 0  # Set reverse flag based on the sign of n
        # Reverse start and end if n is negative
        if self.reverse:
            self.start_point = _coordinates(self.parametric_function(self.t_end), self.label)
            self.end_point = _coordinates(self.parametric_function(self.t_start), self.label)
        else:
            self.start_point = _coordinates(self.parametric_function(self.t_start), self.label)
            self.end_point = _coordinates(self.parametric_function(self.t_end), self.label)
        return self

    def get_midpoint(self) -> tuple:
        """
        Calculates the midpoint of the border.

        Returns:
            tuple: The coordinates of the midpoint.
        """
        mid_t = (self.t_start + self.t_end) / 2
        return self.parametric_function(mid_t)

    def generate_points(self) -> list:
        """
        Generates mesh points along the border.

        Returns:
            list: A list of MeshPoint objects representing the generated points.
        """
        if self.n_segments is None:
            raise ValueError(f'Border {self.label!r} needs a segment count: call border(n) first')
        t_values = np.linspace(self.t_start, self.t_end, abs(self.n_segments) + 1, endpoint=True)
        points = [
            MeshPoint(x, y, self.label, self.is_border)
            for x, y in [_coordinates(self.parametric_function(t), self.label) for t in t_values]
        ]
        return points[:-1] if not self.reverse else points[::-1][:-1]


def find_next_border(current_end, remaining_borders, abs_tol=1e-6) -> tuple:
    """
    Finds the next border connected to the current end point.

    Args:
        current_end (tuple): The coordinates of the current end point.
        remaining_borders (list): A list of Border objects representing the remaining borders.
        abs_tol (float, optional): The absolute tolerance for distance comparison. Defaults to 1e-6.

    Returns:
        tuple: A tuple containing a boolean indicating if multiple candidates were found and the first candidate border.
    """
    first_candidate = None
    found_multiple = False

    for border in remaining_borders:
        if is_close(border.start_point, current_end, abs_tol):
            if first_candidate is None:
                first_candidate = border
            else:
                # As soon as a second candidate is found, stop checking further
                found_multiple = True
                break

    if first_candidate:
        return found_multiple, first_candidate
    else:
        return found_multiple, None


def is_close(point1, point2, tolerance=1e-6) -> bool:
    """
    Checks if two points are close to each other within a given tolerance.

    Args:
        point1 (tuple): The coordinates of the first point.
        point2 (tuple): The coordinates of the second point.
        tolerance (float, optional): The tolerance for distance comparison. Defaults to 1e-6.

    Returns:
        bool: True if the points are close, False otherwise.
    """
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance < tolerance


def find_polygons(borders: list[Border], tolerance=1e-6) -> list:
    """
    Find directed closed contours without discarding open borders.

    For shared junctions, each border uses a shortest directed closing path
    (in number of borders). Duplicate cycles are removed. Input order breaks
    ties; border direction is never silently reversed.

    Args:
        borders (list[Border]): A list of Border objects representing the borders.
        tolerance (float, optional): The tolerance for distance comparison. Defaults to 1e-6.

    Returns:
        list: A list of lists, where each inner list represents a group of connected borders forming a polygon.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if any(not isinstance(border, Border) for border in borders):
        raise TypeError('All contours must be Border objects')
    if len({id(border) for border in borders}) != len(borders):
        raise ValueError('The same Border object was supplied more than once')
    successors = [
        [j for j, other in enumerate(borders)
         if is_close(border.end_point, other.start_point, tolerance)]
        for border in borders
    ]
    contours = []
    seen = set()
    for start, border in enumerate(borders):
        queue = deque([start])
        parents = {start: None}
        end = None
        while queue:
            current = queue.popleft()
            if is_close(borders[current].end_point, border.start_point, tolerance):
                end = current
                break
            for candidate in successors[current]:
                if candidate not in parents:
                    parents[candidate] = current
                    queue.append(candidate)
        if end is None:
            raise ValueError(f'Border {border.label!r} (index {start}) is not part of a closed directed contour; '
                             'check endpoints, orientation, and tolerance')
        cycle = []
        while end is not None:
            cycle.append(end)
            end = parents[end]
        cycle.reverse()
        # Canonical rotation retains direction and shared-border cycles.
        pivot = cycle.index(min(cycle))
        key = tuple(cycle[pivot:] + cycle[:pivot])
        if key not in seen:
            seen.add(key)
            contours.append([borders[i] for i in key])
    return contours
