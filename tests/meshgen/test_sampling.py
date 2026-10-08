import math
import random
import unittest
from collections import Counter
from unittest.mock import patch

from shapely.geometry import Point, box

from rbflab.meshgen import Border, RBFMesh, generate_points_within_polygons


def coordinates(points):
    return [(p.x, p.y) for p in points]


class SamplingTests(unittest.TestCase):
    def test_each_method_respects_holes_counts_and_labels(self):
        regions = [box(0, 0, 3, 3).difference(box(1, 1, 2, 2)), box(5, 0, 6, 1)]
        for method in ('random', 'halton', 'sobol'):
            with self.subTest(method=method):
                points = generate_points_within_polygons(regions, [1137, 37], 0.02,
                                                        method=method, seed=42)
                self.assertEqual(Counter(p.label for p in points), {'region 1': 1137, 'region 2': 37})
                self.assertEqual(len(set(coordinates(points))), 1174)
                for point in points:
                    region = regions[int(point.label.split()[-1]) - 1]
                    location = Point(point.x, point.y)
                    self.assertTrue(region.contains(location))
                    self.assertGreaterEqual(region.boundary.distance(location), 0.019)
                    self.assertFalse(point.is_border)

    def test_seed_reproducibility_and_global_state_isolation(self):
        for method in ('random', 'halton', 'sobol'):
            with self.subTest(method=method):
                state = random.getstate()
                first = generate_points_within_polygons([box(0, 0, 1, 1)], [50], method=method, seed=7)
                again = generate_points_within_polygons([box(0, 0, 1, 1)], [50], method=method, seed=7)
                other = generate_points_within_polygons([box(0, 0, 1, 1)], [50], method=method, seed=8)
                self.assertEqual(coordinates(first), coordinates(again))
                self.assertNotEqual(coordinates(first), coordinates(other))
                self.assertEqual(random.getstate(), state)

    def test_random_default_keeps_legacy_sequence(self):
        state = random.getstate()
        try:
            random.seed(12)
            expected = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(10)]
            random.seed(12)
            points = generate_points_within_polygons([box(0, 0, 1, 1)], [10], 0)
            self.assertEqual(coordinates(points), expected)
        finally:
            random.setstate(state)

    def test_mesh_replacement_and_borders_are_preserved(self):
        circle = Border(lambda t: (math.cos(t), math.sin(t)), 'border', 0, 2 * math.pi)(50)
        mesh = RBFMesh(circle)
        borders = list(mesh.Boundary_Points)
        for method in ('random', 'halton', 'sobol'):
            self.assertEqual(len(mesh.generate_points(101, method=method, seed=42, append=False)), 101)
            self.assertEqual(mesh.Boundary_Points, borders)
            self.assertEqual(len(mesh.generate_points(7, method=method, seed=43)), 108)
        previous = list(mesh.Points)
        with self.assertRaises(ValueError):
            mesh.generate_points(10, method='unknown', append=False)
        self.assertEqual(mesh.Points, previous)

    def test_invalid_options_and_zero_requests(self):
        for seed in (-1, 0.5, True, '42'):
            with self.assertRaisesRegex(ValueError, 'seed'):
                generate_points_within_polygons([], [], seed=seed)
        with self.assertRaisesRegex(ValueError, 'method'):
            generate_points_within_polygons([], [], method='invalid')
        for method in ('random', 'halton', 'sobol'):
            self.assertEqual(generate_points_within_polygons([], [], method=method), [])

    def test_missing_scipy_has_installation_hint(self):
        with patch.dict('sys.modules', {'scipy.stats': None}):
            with self.assertRaisesRegex(ImportError, 'pip install rbflab'):
                generate_points_within_polygons([box(0, 0, 1, 1)], [10], method='halton')
            self.assertEqual(len(generate_points_within_polygons([box(0, 0, 1, 1)], [10])), 10)


if __name__ == '__main__':
    unittest.main()
