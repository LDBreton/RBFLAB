"""Random point-cloud generation inside bounded three-dimensional regions."""

import inspect
import math
import random
from numbers import Integral

import numpy as np

from ..geometry._regions3d import ImplicitRegion, MeshPoint3D


_METHODS = ('random', 'halton', 'sobol')


def _validate_method_seed(method, seed):
    if method not in _METHODS:
        raise ValueError("method must be 'random', 'halton', or 'sobol'")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError('seed must be a non-negative integer or None')
        seed = int(seed)
    return seed


def _qmc_engine(method, seed, dimension=3):
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError('Halton and Sobol require SciPy. Reinstall with: pip install rbflab') from exc
    engine = qmc.Halton if method == 'halton' else qmc.Sobol
    options = {'rng': seed} if 'rng' in inspect.signature(engine).parameters else {'seed': seed}
    return engine(d=dimension, scramble=True, **options)


def _unit_candidates(count, method, seed=None, *, rng=None, sampler=None, dimension=3):
    if method == 'random':
        rng = random if rng is None else rng
        return np.asarray([[rng.random() for _ in range(dimension)] for _ in range(count)])
    sampler = _qmc_engine(method, seed, dimension) if sampler is None else sampler
    return sampler.random(count)


def _map_to_bounds(unit_points, bounds):
    bounds = np.asarray(bounds, dtype=float)
    return bounds[:, 0] + unit_points * (bounds[:, 1] - bounds[:, 0])


def estimate_region_volume(region, samples=32768, *, method='halton', seed=0):
    """Estimate an implicit region's volume inside its declared bounds."""
    if not isinstance(region, ImplicitRegion):
        raise TypeError('region must be an ImplicitRegion')
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples <= 0:
        raise ValueError('samples must be a positive integer')
    seed = _validate_method_seed(method, seed)
    unit_points = _unit_candidates(int(samples), method, seed,
                                  rng=random if seed is None else random.Random(seed))
    candidates = _map_to_bounds(unit_points, region.bounds)
    fraction = np.count_nonzero(region.contains_points(candidates, strict=True)) / samples
    bounds = np.asarray(region.bounds)
    return float(fraction * np.prod(bounds[:, 1] - bounds[:, 0]))


def calculate_volume_allocation(regions, num_points, *, estimate_samples=32768):
    """Allocate an exact point count among regions in proportion to volume."""
    if isinstance(num_points, bool) or not isinstance(num_points, Integral) or num_points < 0:
        raise ValueError('num_points must be a non-negative integer')
    if any(not isinstance(region, ImplicitRegion) for region in regions):
        raise TypeError('All regions must be ImplicitRegion objects')
    if num_points == 0:
        return [0] * len(regions)
    if not regions:
        raise ValueError('Cannot generate points without a positive-volume region')

    volumes = [region.volume if region.volume is not None
               else estimate_region_volume(region, estimate_samples) for region in regions]
    if any(not math.isfinite(volume) or volume <= 0 for volume in volumes):
        raise ValueError('Cannot generate points in an empty region; check its field and bounds')
    total_volume = sum(volumes)
    quotas = [volume / total_volume * num_points for volume in volumes]
    allocation = [math.floor(quota) for quota in quotas]
    remaining = int(num_points) - sum(allocation)
    order = sorted(range(len(regions)), key=lambda index: quotas[index] - allocation[index],
                   reverse=True)
    for index in order[:remaining]:
        allocation[index] += 1
    return allocation


def calculate_boundary_allocation(regions, num_points):
    """Allocate boundary points in proportion to known surface areas."""
    if isinstance(num_points, bool) or not isinstance(num_points, Integral) or num_points < 0:
        raise ValueError('num_points must be a non-negative integer')
    if any(not isinstance(region, ImplicitRegion) for region in regions):
        raise TypeError('All regions must be ImplicitRegion objects')
    if num_points == 0:
        return [0] * len(regions)
    if not regions:
        raise ValueError('Cannot generate boundary points without a region')
    if len(regions) == 1:
        return [int(num_points)]
    if any(region.surface_area is None for region in regions):
        raise ValueError('Multiple custom regions require explicit boundary point allocations')

    total_area = sum(region.surface_area for region in regions)
    quotas = [region.surface_area / total_area * num_points for region in regions]
    allocation = [math.floor(quota) for quota in quotas]
    remaining = int(num_points) - sum(allocation)
    order = sorted(range(len(regions)), key=lambda index: quotas[index] - allocation[index],
                   reverse=True)
    for index in order[:remaining]:
        allocation[index] += 1
    return allocation


def generate_points_within_regions(regions, points_allocation, boundary_distance=0,
                                   *, method='random', seed=None, max_candidates=None):
    """Generate unconnected 3D points inside bounded implicit regions."""
    seed = _validate_method_seed(method, seed)
    if not math.isfinite(boundary_distance) or boundary_distance < 0:
        raise ValueError('boundary_distance must be finite and non-negative')
    if len(regions) != len(points_allocation):
        raise ValueError('Each region must have a point allocation')

    for region, count in zip(regions, points_allocation):
        if not isinstance(region, ImplicitRegion):
            raise TypeError('All regions must be ImplicitRegion objects')
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('Point allocations must be non-negative integers')
        if count and boundary_distance:
            if region.clearance is None:
                raise ValueError('boundary_distance requires every sampled region to provide clearance')
            if (region.max_clearance is not None and
                    boundary_distance >= region.max_clearance):
                raise ValueError('boundary_distance leaves no positive-volume sampling region')
    if max_candidates is not None and (isinstance(max_candidates, bool) or
                                       not isinstance(max_candidates, Integral) or
                                       max_candidates <= 0):
        raise ValueError('max_candidates must be a positive integer or None')
    if not any(points_allocation):
        return []

    rng = random if seed is None else random.Random(seed)
    sampler = None if method == 'random' else _qmc_engine(method, seed)
    points = []

    for index, (region, count) in enumerate(zip(regions, points_allocation)):
        if count == 0:
            continue
        accepted = 0
        tested = 0
        candidate_limit = (max_candidates if max_candidates is not None
                           else max(100000, int(count) * 10000))
        label = region.label if region.label is not None else f'region {index + 1}'

        while accepted < count and tested < candidate_limit:
            batch_size = min(1024, candidate_limit - tested)
            unit_points = _unit_candidates(batch_size, method, seed, rng=rng, sampler=sampler)
            candidates = _map_to_bounds(unit_points, region.bounds)
            keep = region.contains_points(candidates, strict=True)
            if boundary_distance:
                keep &= region.clearance_points(candidates) >= boundary_distance
            remaining = count - accepted
            for x, y, z in candidates[keep][:remaining]:
                points.append(MeshPoint3D(x, y, z, label, False))
                accepted += 1
            tested += batch_size

        if accepted < count:
            raise ValueError('Could not sample enough points; check the region bounds, field, '
                             'boundary distance, or increase max_candidates')

    return points


def generate_boundary_points(regions, points_allocation, *, method='random', seed=None,
                             tolerance=1e-8, max_iterations=30, max_candidates=None):
    """Generate labeled boundary points with outward unit normals."""
    seed = _validate_method_seed(method, seed)
    if len(regions) != len(points_allocation):
        raise ValueError('Each region must have a boundary point allocation')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or \
            max_iterations <= 0:
        raise ValueError('max_iterations must be a positive integer')
    if max_candidates is not None and (isinstance(max_candidates, bool) or
                                       not isinstance(max_candidates, Integral) or
                                       max_candidates <= 0):
        raise ValueError('max_candidates must be a positive integer or None')
    for region, count in zip(regions, points_allocation):
        if not isinstance(region, ImplicitRegion):
            raise TypeError('All regions must be ImplicitRegion objects')
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('Boundary point allocations must be non-negative integers')
    if not any(points_allocation):
        return []

    rng = random if seed is None else random.Random(seed)
    sampler = None if method == 'random' else _qmc_engine(method, seed)
    points = []
    for region, count in zip(regions, points_allocation):
        accepted = 0
        tested = 0
        candidate_limit = (max_candidates if max_candidates is not None
                           else max(100000, int(count) * 100))
        while accepted < count and tested < candidate_limit:
            # Power-of-two blocks retain Sobol balance and avoid SciPy's
            # arbitrary-size draw warning; surplus accepted points are discarded.
            batch_size = min(1024, candidate_limit - tested)
            unit_points = _unit_candidates(
                batch_size, method, seed, rng=rng, sampler=sampler)
            locations, normals, labels = region.boundary_candidates(
                unit_points, tolerance=tolerance, max_iterations=max_iterations)
            remaining = count - accepted
            for location, normal, label in zip(
                    locations[:remaining], normals[:remaining], labels[:remaining]):
                points.append(MeshPoint3D(*location, label, True, normal=normal,
                                          boundary_label=label))
                accepted += 1
            tested += batch_size
        if accepted < count:
            raise ValueError('Could not project enough boundary points; check the implicit field, '
                             'bounds, gradient, or increase max_candidates')
    return points


class RBFMesh3D:
    """Generate random, Halton, or Sobol point clouds in 3D implicit regions."""

    def __init__(self, *regions):
        if any(not isinstance(region, ImplicitRegion) for region in regions):
            raise TypeError('All regions must be ImplicitRegion objects')
        self.regions = list(regions)
        self.Points = []
        self.Boundary_Points = []

    def generate_points(self, num_points, boundary_distance=0, *, append=True,
                        method='random', seed=None, estimate_samples=32768,
                        max_candidates=None):
        """Generate exactly ``num_points`` interior points across the regions."""
        allocation = calculate_volume_allocation(
            self.regions, num_points, estimate_samples=estimate_samples)
        points = generate_points_within_regions(
            self.regions, allocation, boundary_distance, method=method, seed=seed,
            max_candidates=max_candidates)
        if append:
            self.Points.extend(points)
        else:
            self.Points[:] = points
        return self.Points

    def generate_boundary_points(self, num_points, *, append=True, method='random', seed=None,
                                 tolerance=1e-8, max_iterations=30,
                                 max_candidates=None):
        """Generate explicit boundary nodes with labels and outward normals."""
        allocation = calculate_boundary_allocation(self.regions, num_points)
        points = generate_boundary_points(
            self.regions, allocation, method=method, seed=seed, tolerance=tolerance,
            max_iterations=max_iterations, max_candidates=max_candidates)
        if append:
            self.Boundary_Points.extend(points)
        else:
            self.Boundary_Points[:] = points
        return self.Boundary_Points
