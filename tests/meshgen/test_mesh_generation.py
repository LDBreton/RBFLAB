import math
import random
import unittest

from shapely.geometry import Point, box

from rbflab.meshgen import Border, RBFMesh, calculate_point_allocation, generate_points_within_polygons


class MeshGenerationTests(unittest.TestCase):
    def test_exact_area_allocation(self):
        regions = [box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)]
        self.assertEqual(calculate_point_allocation(regions, 10), [4, 3, 3])
        self.assertEqual(calculate_point_allocation(regions, 2), [1, 1, 0])
        self.assertEqual(calculate_point_allocation([box(0, 0, 1, 1), box(2, 0, 5, 1)], 10), [3, 7])

    def test_invalid_counts_and_empty_regions(self):
        for count in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                calculate_point_allocation([box(0, 0, 1, 1)], count)
        self.assertEqual(calculate_point_allocation([], 0), [])
        with self.assertRaises(ValueError):
            calculate_point_allocation([], 1)

    def test_invalid_sampling_requests(self):
        for distance in [-1, math.nan, math.inf, 1]:
            with self.assertRaises(ValueError):
                generate_points_within_polygons([box(0, 0, 1, 1)], [1], distance)
        with self.assertRaises(ValueError):
            generate_points_within_polygons([box(0, 0, 1, 1)], [])
        self.assertEqual(generate_points_within_polygons([box(0, 0, 1, 1)], [0], 1), [])

    def test_points_stay_in_domain_with_hole(self):
        region = box(0, 0, 3, 3).difference(box(1, 1, 2, 2))
        random.seed(42)
        points = generate_points_within_polygons([region], [100], 0.05)
        self.assertEqual(len(points), 100)
        for point in points:
            position = Point(point.x, point.y)
            self.assertTrue(region.contains(position))
            self.assertGreaterEqual(region.boundary.distance(position), 0.049)
            self.assertFalse(point.is_border)

    def test_append_replace_and_failure_preserve_state(self):
        border = Border(lambda t: (math.cos(t), math.sin(t)), 1, 0, 2 * math.pi)
        mesh = RBFMesh(border(40))
        self.assertEqual(len(mesh.generate_points(10)), 10)
        self.assertEqual(len(mesh.generate_points(5)), 15)
        self.assertEqual(len(mesh.generate_points(7, append=False)), 7)
        previous = list(mesh.Points)
        with self.assertRaises(ValueError):
            mesh.generate_points(1, boundary_distance=2, append=False)
        self.assertEqual(mesh.Points, previous)
        self.assertEqual(mesh.generate_points(0, append=False), [])


if __name__ == '__main__':
    unittest.main()
