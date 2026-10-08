"""Geometry definitions for three-dimensional RBF point clouds."""

import math

import numpy as np


def _vector3(value, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain three finite numbers') from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must contain three finite numbers')
    return vector


class MeshPoint3D:
    """A labeled point in three-dimensional space."""

    def __init__(self, x, y, z, label=0, is_border=True, *, normal=None,
                 boundary_label=None):
        coordinates = _vector3((x, y, z), 'point coordinates')
        self.x, self.y, self.z = coordinates.tolist()
        self.label = label
        self.is_border = is_border
        if normal is None:
            self.normal = None
        else:
            normal = _vector3(normal, 'normal')
            magnitude = np.linalg.norm(normal)
            if magnitude == 0:
                raise ValueError('normal must be non-zero')
            self.normal = tuple((normal / magnitude).tolist())
        self.boundary_label = (boundary_label if boundary_label is not None
                               else label if is_border else None)


class ImplicitRegion:
    """A bounded volume described by ``field(x, y, z) <= 0``.

    Args:
        field (Callable): Scalar or NumPy-vectorized implicit function. Negative values
            are inside the region and zero is its boundary.
        bounds (Sequence): Three ``(minimum, maximum)`` pairs for x, y, and z.
        label (object): Optional point label. An automatic region label is used when
            omitted.
        volume (float | None): Optional known volume. It avoids numerical volume estimation
            when points are allocated across multiple regions.
        clearance (Callable | None): Optional function returning a positive physical distance
            to the boundary for interior points.
        max_clearance (float | None): Optional greatest possible interior clearance.
    """

    def __init__(self, field, bounds, *, label=None, boundary_label=None,
                 volume=None, surface_area=None, clearance=None,
                 gradient=None, max_clearance=None):
        if not callable(field):
            raise TypeError('field must be callable')
        if clearance is not None and not callable(clearance):
            raise TypeError('clearance must be callable or None')
        if gradient is not None and not callable(gradient):
            raise TypeError('gradient must be callable or None')
        try:
            limits = np.asarray(bounds, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('bounds must contain three finite (minimum, maximum) pairs') from exc
        if (limits.shape != (3, 2) or not np.isfinite(limits).all() or
                np.any(limits[:, 0] >= limits[:, 1])):
            raise ValueError('bounds must contain three finite increasing (minimum, maximum) pairs')
        if volume is not None and (isinstance(volume, bool) or
                                   not isinstance(volume, (int, float)) or
                                   not math.isfinite(volume) or volume <= 0):
            raise ValueError('volume must be finite and positive or None')
        if surface_area is not None and (isinstance(surface_area, bool) or
                                         not isinstance(surface_area, (int, float)) or
                                         not math.isfinite(surface_area) or surface_area <= 0):
            raise ValueError('surface_area must be finite and positive or None')
        if max_clearance is not None and (not math.isfinite(max_clearance) or
                                          max_clearance <= 0):
            raise ValueError('max_clearance must be finite and positive or None')

        self.field = field
        self.bounds = tuple((float(lower), float(upper)) for lower, upper in limits)
        self.label = label
        self.boundary_label = (boundary_label if boundary_label is not None
                               else label if label is not None else 'boundary')
        self.volume = float(volume) if volume is not None else None
        self.surface_area = float(surface_area) if surface_area is not None else None
        self.clearance = clearance
        self.gradient = gradient
        self.max_clearance = max_clearance

    @staticmethod
    def _evaluate(function, points, name):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points must have shape (n, 3)')
        try:
            values = np.asarray(function(points[:, 0], points[:, 1], points[:, 2]),
                                dtype=float)
            if values.ndim == 0:
                values = np.full(len(points), float(values))
            elif values.shape != (len(points),):
                raise ValueError
        except (TypeError, ValueError, IndexError):
            try:
                values = np.asarray([function(x, y, z) for x, y, z in points], dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{name} must return finite scalar values') from exc
        if values.shape != (len(points),) or not np.isfinite(values).all():
            raise ValueError(f'{name} must return finite scalar values')
        return values

    def contains_points(self, points, *, strict=False, tolerance=1e-12):
        """Return a Boolean mask indicating which points are in the region."""
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('tolerance must be finite and non-negative')
        values = self._evaluate(self.field, points, 'field')
        return values < 0 if strict else values <= tolerance

    def contains(self, x, y, z, *, strict=False, tolerance=1e-12):
        """Return whether one point is in the region."""
        return bool(self.contains_points(
            [[x, y, z]], strict=strict, tolerance=tolerance)[0])

    def clearance_points(self, points):
        """Return physical interior boundary clearance for the supplied points."""
        if self.clearance is None:
            raise ValueError('boundary_distance requires a region with a clearance function')
        return self._evaluate(self.clearance, points, 'clearance')

    def gradient_points(self, points):
        """Evaluate an outward-pointing field gradient at multiple points."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points must have shape (n, 3)')
        if self.gradient is not None:
            try:
                values = np.asarray(
                    self.gradient(points[:, 0], points[:, 1], points[:, 2]), dtype=float)
                if values.shape == (3, len(points)):
                    values = values.T
                if values.shape != points.shape:
                    raise ValueError
            except (TypeError, ValueError, IndexError):
                try:
                    values = np.asarray(
                        [self.gradient(x, y, z) for x, y, z in points], dtype=float)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError('gradient must return three finite values') from exc
            if values.shape != points.shape or not np.isfinite(values).all():
                raise ValueError('gradient must return three finite values')
            return values

        scales = np.asarray([upper - lower for lower, upper in self.bounds])
        steps = np.maximum(scales * 1e-6, 1e-8)
        gradients = np.empty_like(points)
        for axis in range(3):
            forward = points.copy()
            backward = points.copy()
            forward[:, axis] += steps[axis]
            backward[:, axis] -= steps[axis]
            gradients[:, axis] = (
                self._evaluate(self.field, forward, 'field') -
                self._evaluate(self.field, backward, 'field')) / (2 * steps[axis])
        return gradients

    def boundary_candidates(self, unit_points, *, tolerance=1e-8, max_iterations=30):
        """Project bounding-box candidates onto the implicit zero level set."""
        unit_points = np.asarray(unit_points, dtype=float)
        bounds = np.asarray(self.bounds)
        points = bounds[:, 0] + unit_points * (bounds[:, 1] - bounds[:, 0])
        active = np.ones(len(points), dtype=bool)
        for _ in range(max_iterations):
            values = self._evaluate(self.field, points, 'field')
            converged = np.abs(values) <= tolerance
            active &= ~converged
            if not np.any(active):
                break
            gradients = self.gradient_points(points[active])
            norms_squared = np.sum(gradients * gradients, axis=1)
            usable = norms_squared > np.finfo(float).eps
            active_indices = np.flatnonzero(active)
            active[active_indices[~usable]] = False
            update_indices = active_indices[usable]
            points[update_indices] -= (
                values[update_indices, None] * gradients[usable] /
                norms_squared[usable, None])

        values = np.abs(self._evaluate(self.field, points, 'field'))
        inside_bounds = np.all((points >= bounds[:, 0] - tolerance) &
                               (points <= bounds[:, 1] + tolerance), axis=1)
        gradients = self.gradient_points(points)
        magnitudes = np.linalg.norm(gradients, axis=1)
        valid = (values <= tolerance) & inside_bounds & (magnitudes > np.finfo(float).eps)
        normals = gradients[valid] / magnitudes[valid, None]
        labels = np.full(np.count_nonzero(valid), self.boundary_label, dtype=object)
        return points[valid], normals, labels


class Sphere(ImplicitRegion):
    """A solid sphere implicit region."""

    def __init__(self, center=(0, 0, 0), radius=1, *, label=None,
                 boundary_label=None):
        center = _vector3(center, 'center')
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or \
                not math.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be finite and positive')
        radius = float(radius)

        def field(x, y, z):
            return ((x - center[0]) ** 2 + (y - center[1]) ** 2 +
                    (z - center[2]) ** 2 - radius ** 2)

        def clearance(x, y, z):
            return radius - np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 +
                                    (z - center[2]) ** 2)

        def gradient(x, y, z):
            return np.asarray((2 * (x - center[0]), 2 * (y - center[1]),
                               2 * (z - center[2])))

        bounds = [(coordinate - radius, coordinate + radius) for coordinate in center]
        super().__init__(field, bounds, label=label, boundary_label=boundary_label,
                         volume=4 * math.pi * radius ** 3 / 3,
                         surface_area=4 * math.pi * radius ** 2,
                         clearance=clearance, gradient=gradient,
                         max_clearance=radius)
        self.center = tuple(center.tolist())
        self.radius = radius

    def boundary_candidates(self, unit_points, **kwargs):
        unit_points = np.asarray(unit_points, dtype=float)
        azimuth = 2 * math.pi * unit_points[:, 0]
        cosine = 1 - 2 * unit_points[:, 1]
        sine = np.sqrt(np.maximum(0, 1 - cosine * cosine))
        normals = np.column_stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine))
        points = np.asarray(self.center) + self.radius * normals
        labels = np.full(len(points), self.boundary_label, dtype=object)
        return points, normals, labels


class Box(ImplicitRegion):
    """An axis-aligned solid box implicit region."""

    def __init__(self, minimum=(-1, -1, -1), maximum=(1, 1, 1), *, label=None,
                 boundary_labels=None):
        minimum = _vector3(minimum, 'minimum')
        maximum = _vector3(maximum, 'maximum')
        if np.any(minimum >= maximum):
            raise ValueError('minimum coordinates must be less than maximum coordinates')

        def field(x, y, z):
            return np.maximum.reduce((minimum[0] - x, x - maximum[0],
                                      minimum[1] - y, y - maximum[1],
                                      minimum[2] - z, z - maximum[2]))

        def clearance(x, y, z):
            return np.minimum.reduce((x - minimum[0], maximum[0] - x,
                                      y - minimum[1], maximum[1] - y,
                                      z - minimum[2], maximum[2] - z))

        lengths = maximum - minimum
        default_labels = {
            'xmin': 'xmin', 'xmax': 'xmax', 'ymin': 'ymin',
            'ymax': 'ymax', 'zmin': 'zmin', 'zmax': 'zmax',
        }
        if boundary_labels is not None:
            unknown = set(boundary_labels) - set(default_labels)
            if unknown:
                raise ValueError(f'Unknown box boundary labels: {sorted(unknown)}')
            default_labels.update(boundary_labels)
        self.boundary_labels = default_labels
        surface_area = 2 * (lengths[0] * lengths[1] + lengths[0] * lengths[2] +
                            lengths[1] * lengths[2])
        super().__init__(field, np.column_stack((minimum, maximum)), label=label,
                         volume=float(np.prod(lengths)), surface_area=float(surface_area),
                         clearance=clearance,
                         max_clearance=float(np.min(lengths) / 2))
        self.minimum = tuple(minimum.tolist())
        self.maximum = tuple(maximum.tolist())

    def boundary_candidates(self, unit_points, **kwargs):
        unit_points = np.asarray(unit_points, dtype=float)
        minimum, maximum = np.asarray(self.minimum), np.asarray(self.maximum)
        lengths = maximum - minimum
        face_areas = np.asarray((lengths[1] * lengths[2], lengths[1] * lengths[2],
                                 lengths[0] * lengths[2], lengths[0] * lengths[2],
                                 lengths[0] * lengths[1], lengths[0] * lengths[1]))
        cumulative = np.cumsum(face_areas) / np.sum(face_areas)
        faces = np.searchsorted(cumulative, unit_points[:, 0], side='right')
        points = minimum + unit_points[:, 1:3].repeat(2, axis=1)[:, :3] * lengths
        normals = np.zeros_like(points)
        names = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        labels = np.empty(len(points), dtype=object)
        for face, name in enumerate(names):
            selected = faces == face
            axis, side = divmod(face, 2)
            other_axes = [index for index in range(3) if index != axis]
            points[selected, axis] = minimum[axis] if side == 0 else maximum[axis]
            points[selected, other_axes[0]] = (minimum[other_axes[0]] +
                                               unit_points[selected, 1] * lengths[other_axes[0]])
            points[selected, other_axes[1]] = (minimum[other_axes[1]] +
                                               unit_points[selected, 2] * lengths[other_axes[1]])
            normals[selected, axis] = -1 if side == 0 else 1
            labels[selected] = self.boundary_labels[name]
        return points, normals, labels


class Cylinder(ImplicitRegion):
    """A finite axis-aligned solid cylinder implicit region."""

    def __init__(self, center=(0, 0, 0), radius=1, height=2, *, axis='z', label=None,
                 boundary_labels=None):
        center = _vector3(center, 'center')
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or \
                not math.isfinite(radius) or radius <= 0:
            raise ValueError('radius must be finite and positive')
        if isinstance(height, bool) or not isinstance(height, (int, float)) or \
                not math.isfinite(height) or height <= 0:
            raise ValueError('height must be finite and positive')
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        radius, height = float(radius), float(height)
        axial = {'x': 0, 'y': 1, 'z': 2}[axis]
        radial = [index for index in range(3) if index != axial]

        def field(x, y, z):
            coordinates = (x, y, z)
            radial_squared = sum((coordinates[index] - center[index]) ** 2
                                 for index in radial)
            axial_offset = np.abs(coordinates[axial] - center[axial])
            return np.maximum(radial_squared - radius ** 2, axial_offset - height / 2)

        def clearance(x, y, z):
            coordinates = (x, y, z)
            radial_distance = np.sqrt(sum((coordinates[index] - center[index]) ** 2
                                          for index in radial))
            axial_offset = np.abs(coordinates[axial] - center[axial])
            return np.minimum(radius - radial_distance, height / 2 - axial_offset)

        half_sizes = np.full(3, radius)
        half_sizes[axial] = height / 2
        bounds = np.column_stack((center - half_sizes, center + half_sizes))
        default_labels = {'side': 'side', 'bottom': 'bottom', 'top': 'top'}
        if boundary_labels is not None:
            unknown = set(boundary_labels) - set(default_labels)
            if unknown:
                raise ValueError(f'Unknown cylinder boundary labels: {sorted(unknown)}')
            default_labels.update(boundary_labels)
        self.boundary_labels = default_labels
        surface_area = 2 * math.pi * radius * height + 2 * math.pi * radius ** 2
        super().__init__(field, bounds, label=label,
                         volume=math.pi * radius ** 2 * height,
                         surface_area=surface_area, clearance=clearance,
                         max_clearance=min(radius, height / 2))
        self.center = tuple(center.tolist())
        self.radius = radius
        self.height = height
        self.axis = axis

    def boundary_candidates(self, unit_points, **kwargs):
        unit_points = np.asarray(unit_points, dtype=float)
        center = np.asarray(self.center)
        axial = {'x': 0, 'y': 1, 'z': 2}[self.axis]
        radial = [index for index in range(3) if index != axial]
        side_area = 2 * math.pi * self.radius * self.height
        cap_area = math.pi * self.radius ** 2
        side_fraction = side_area / (side_area + 2 * cap_area)
        bottom_fraction = cap_area / (side_area + 2 * cap_area)
        side = unit_points[:, 0] < side_fraction
        bottom = ((unit_points[:, 0] >= side_fraction) &
                  (unit_points[:, 0] < side_fraction + bottom_fraction))
        top = ~(side | bottom)
        points = np.tile(center, (len(unit_points), 1))
        normals = np.zeros_like(points)
        labels = np.empty(len(points), dtype=object)

        angle = 2 * math.pi * unit_points[:, 1]
        points[side, radial[0]] += self.radius * np.cos(angle[side])
        points[side, radial[1]] += self.radius * np.sin(angle[side])
        points[side, axial] += (unit_points[side, 2] - 0.5) * self.height
        normals[side, radial[0]] = np.cos(angle[side])
        normals[side, radial[1]] = np.sin(angle[side])
        labels[side] = self.boundary_labels['side']

        cap_points = bottom | top
        radial_distance = self.radius * np.sqrt(unit_points[cap_points, 1])
        cap_angle = 2 * math.pi * unit_points[cap_points, 2]
        points[cap_points, radial[0]] += radial_distance * np.cos(cap_angle)
        points[cap_points, radial[1]] += radial_distance * np.sin(cap_angle)
        points[bottom, axial] -= self.height / 2
        points[top, axial] += self.height / 2
        normals[bottom, axial] = -1
        normals[top, axial] = 1
        labels[bottom] = self.boundary_labels['bottom']
        labels[top] = self.boundary_labels['top']
        return points, normals, labels
