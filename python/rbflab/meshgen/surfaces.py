"""Random point sampling on continuous parametric surfaces in 3D."""

import math
import random
from numbers import Integral

import numpy as np

from ..geometry._regions3d import MeshPoint3D
from .volumes import _qmc_engine, _validate_method_seed


class ParametricSurface3D:
    """A surface ``F(u, v) = (x, y, z)`` sampled without triangulation.

    Points are sampled in parameter space and accepted in proportion to the
    local surface Jacobian ``|dF/du x dF/dv|``. Consequently, accepted points
    are approximately uniform in physical surface area rather than merely
    uniform in the parameter rectangle.

    Args:
        parametric_function (Callable): Callable returning three coordinates.
        u_range (tuple): Finite increasing ``(minimum, maximum)`` pair.
        v_range (tuple): Finite increasing ``(minimum, maximum)`` pair.
        label (object): Boundary label assigned to generated points.
        derivatives (tuple | None): Optional ``(dF_du, dF_dv)`` callables. Finite differences
            are used when omitted.
        periodic_u (bool): Wrap finite-difference evaluations across the u seam.
        periodic_v (bool): Wrap finite-difference evaluations across the v seam.
        orientation (int): ``1`` for ``dF/du x dF/dv`` or ``-1`` to reverse normals.
    """

    def __init__(self, parametric_function, u_range, v_range, *, label='boundary',
                 derivatives=None, periodic_u=False, periodic_v=False,
                 orientation=1):
        if not callable(parametric_function):
            raise TypeError('parametric_function must be callable')
        self.u_range = self._validate_range(u_range, 'u_range')
        self.v_range = self._validate_range(v_range, 'v_range')
        if derivatives is not None:
            if (not isinstance(derivatives, (tuple, list)) or len(derivatives) != 2 or
                    not all(callable(derivative) for derivative in derivatives)):
                raise TypeError('derivatives must contain callable dF_du and dF_dv functions')
            derivatives = tuple(derivatives)
        if not isinstance(periodic_u, bool) or not isinstance(periodic_v, bool):
            raise TypeError('periodic_u and periodic_v must be Boolean')
        if isinstance(orientation, bool) or orientation not in (-1, 1):
            raise ValueError('orientation must be 1 or -1')

        self.parametric_function = parametric_function
        self.derivatives = derivatives
        self.periodic_u = periodic_u
        self.periodic_v = periodic_v
        self.orientation = orientation
        self.label = label
        self.Boundary_Points = []

        midpoint_u = sum(self.u_range) / 2
        midpoint_v = sum(self.v_range) / 2
        self._evaluate(self.parametric_function,
                       np.asarray([midpoint_u]), np.asarray([midpoint_v]),
                       'parametric_function')

    @staticmethod
    def _validate_range(value, name):
        try:
            limits = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite increasing pair') from exc
        if (limits.shape != (2,) or not np.isfinite(limits).all() or
                limits[0] >= limits[1]):
            raise ValueError(f'{name} must be a finite increasing pair')
        return float(limits[0]), float(limits[1])

    @staticmethod
    def _evaluate(function, u, v, name):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        shape = u.shape
        try:
            raw = function(u, v)
            if isinstance(raw, (tuple, list)) and len(raw) == 3:
                coordinates = np.stack(
                    [np.broadcast_to(np.asarray(value, dtype=float), shape) for value in raw],
                    axis=-1)
            else:
                coordinates = np.asarray(raw, dtype=float)
                if coordinates.shape == (3,) + shape:
                    coordinates = np.moveaxis(coordinates, 0, -1)
                if coordinates.shape != shape + (3,):
                    raise ValueError
        except (TypeError, ValueError, IndexError):
            try:
                flat = [function(float(a), float(b))
                        for a, b in zip(u.ravel(), v.ravel())]
                coordinates = np.asarray(flat, dtype=float).reshape(shape + (3,))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{name} must return three finite coordinates') from exc
        if coordinates.shape != shape + (3,) or not np.isfinite(coordinates).all():
            raise ValueError(f'{name} must return three finite coordinates')
        return coordinates

    @staticmethod
    def _offset(values, step, limits, periodic):
        lower, upper = limits
        shifted = values + step
        if periodic:
            return lower + np.mod(shifted - lower, upper - lower)
        return np.clip(shifted, lower, upper)

    def _derivative_vectors(self, u, v):
        if self.derivatives is not None:
            derivative_u = self._evaluate(self.derivatives[0], u, v, 'dF_du')
            derivative_v = self._evaluate(self.derivatives[1], u, v, 'dF_dv')
            return derivative_u, derivative_v

        step_u = (self.u_range[1] - self.u_range[0]) * 1e-6
        step_v = (self.v_range[1] - self.v_range[0]) * 1e-6
        forward_u = self._offset(u, step_u, self.u_range, self.periodic_u)
        backward_u = self._offset(u, -step_u, self.u_range, self.periodic_u)
        forward_v = self._offset(v, step_v, self.v_range, self.periodic_v)
        backward_v = self._offset(v, -step_v, self.v_range, self.periodic_v)
        delta_u = np.where(self.periodic_u, 2 * step_u, forward_u - backward_u)
        delta_v = np.where(self.periodic_v, 2 * step_v, forward_v - backward_v)
        derivative_u = (
            self._evaluate(self.parametric_function, forward_u, v, 'parametric_function') -
            self._evaluate(self.parametric_function, backward_u, v, 'parametric_function')
        ) / delta_u[..., None]
        derivative_v = (
            self._evaluate(self.parametric_function, u, forward_v, 'parametric_function') -
            self._evaluate(self.parametric_function, u, backward_v, 'parametric_function')
        ) / delta_v[..., None]
        return derivative_u, derivative_v

    def _surface_data(self, u, v):
        coordinates = self._evaluate(
            self.parametric_function, u, v, 'parametric_function')
        derivative_u, derivative_v = self._derivative_vectors(u, v)
        cross_product = np.cross(derivative_u, derivative_v)
        jacobian = np.linalg.norm(cross_product, axis=-1)
        normals = np.zeros_like(cross_product)
        valid = jacobian > np.finfo(float).eps
        normals[valid] = (self.orientation * cross_product[valid] /
                          jacobian[valid, None])
        return coordinates, normals, jacobian, valid

    def _estimate_max_jacobian(self, resolution):
        u = np.linspace(*self.u_range, resolution, endpoint=not self.periodic_u)
        v = np.linspace(*self.v_range, resolution, endpoint=not self.periodic_v)
        grid_u, grid_v = np.meshgrid(u, v, indexing='ij')
        _, _, jacobian, valid = self._surface_data(grid_u, grid_v)
        if not np.any(valid):
            raise ValueError('parametric_function defines a zero-area surface')
        return float(np.max(jacobian[valid]) * 1.05)

    @staticmethod
    def _unit_candidates(count, method, rng, sampler):
        if method == 'random':
            return np.asarray([[rng.random(), rng.random(), rng.random()]
                               for _ in range(count)])
        return sampler.random(count)

    def generate_points(self, num_points, *, method='random', seed=None, append=True,
                        jacobian_resolution=48, max_candidates=None):
        """Generate labeled random points and normals on the surface."""
        if isinstance(num_points, bool) or not isinstance(num_points, Integral) or num_points < 0:
            raise ValueError('num_points must be a non-negative integer')
        seed = _validate_method_seed(method, seed)
        if (isinstance(jacobian_resolution, bool) or
                not isinstance(jacobian_resolution, Integral) or
                jacobian_resolution < 2):
            raise ValueError('jacobian_resolution must be an integer greater than one')
        if max_candidates is not None and (isinstance(max_candidates, bool) or
                                           not isinstance(max_candidates, Integral) or
                                           max_candidates <= 0):
            raise ValueError('max_candidates must be a positive integer or None')
        if num_points == 0:
            if not append:
                self.Boundary_Points.clear()
            return self.Boundary_Points

        jacobian_bound = self._estimate_max_jacobian(int(jacobian_resolution))
        candidate_limit = (int(max_candidates) if max_candidates is not None
                           else max(100000, int(num_points) * 1000))

        # Restart from the same deterministic sequence if the preliminary grid
        # underestimated the maximum surface Jacobian.
        while True:
            rng = random if seed is None else random.Random(seed)
            sampler = None if method == 'random' else _qmc_engine(method, seed)
            generated = []
            tested = 0
            underestimated = False
            while len(generated) < num_points and tested < candidate_limit:
                batch_size = min(1024, candidate_limit - tested)
                unit = self._unit_candidates(batch_size, method, rng, sampler)
                u = self.u_range[0] + unit[:, 0] * (self.u_range[1] - self.u_range[0])
                v = self.v_range[0] + unit[:, 1] * (self.v_range[1] - self.v_range[0])
                coordinates, normals, jacobian, valid = self._surface_data(u, v)
                actual_maximum = float(np.max(jacobian))
                if actual_maximum > jacobian_bound:
                    jacobian_bound = actual_maximum * 1.05
                    underestimated = True
                    break
                accepted = valid & (unit[:, 2] < jacobian / jacobian_bound)
                remaining = num_points - len(generated)
                for coordinate, normal in zip(
                        coordinates[accepted][:remaining], normals[accepted][:remaining]):
                    generated.append(MeshPoint3D(
                        *coordinate, self.label, True, normal=normal,
                        boundary_label=self.label))
                tested += batch_size
            if underestimated:
                continue
            if len(generated) < num_points:
                raise ValueError('Could not sample enough surface points; check the '
                                 'parametrization or increase max_candidates')
            break

        if append:
            self.Boundary_Points.extend(generated)
        else:
            self.Boundary_Points[:] = generated
        return self.Boundary_Points
