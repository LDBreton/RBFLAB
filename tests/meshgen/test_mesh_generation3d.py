import math
import random
import unittest
from collections import Counter

from rbflab.meshgen import (
    Box, Cylinder, ImplicitRegion, MeshPoint3D, RBFMesh3D, Sphere,
    calculate_boundary_allocation, calculate_volume_allocation,
    estimate_region_volume, generate_boundary_points,
    generate_points_within_regions,
)


def coordinates(points):
    return [(point.x, point.y, point.z) for point in points]


class MeshGeneration3DTests(unittest.TestCase):
    def test_3d_point_validates_coordinates(self):
        point = MeshPoint3D(1, 2, 3, 'sample', False)
        self.assertEqual((point.x, point.y, point.z), (1, 2, 3))
        self.assertEqual(point.label, 'sample')
        self.assertFalse(point.is_border)
        with self.assertRaisesRegex(ValueError, 'finite'):
            MeshPoint3D(0, math.nan, 0)
        boundary = MeshPoint3D(1, 0, 0, 'wall', True, normal=(2, 0, 0))
        self.assertEqual(boundary.normal, (1, 0, 0))
        self.assertEqual(boundary.boundary_label, 'wall')

    def test_each_sampler_generates_reproducible_sphere_points(self):
        sphere = Sphere((1, -2, 0.5), 2, label='sphere')
        for method in ('random', 'halton', 'sobol'):
            with self.subTest(method=method):
                first = generate_points_within_regions(
                    [sphere], [257], 0.1, method=method, seed=42)
                again = generate_points_within_regions(
                    [sphere], [257], 0.1, method=method, seed=42)
                self.assertEqual(coordinates(first), coordinates(again))
                self.assertEqual(len(set(coordinates(first))), 257)
                self.assertTrue(all(point.label == 'sphere' for point in first))
                self.assertTrue(all(not point.is_border for point in first))
                for point in first:
                    distance = math.dist((point.x, point.y, point.z), sphere.center)
                    self.assertLess(distance, sphere.radius)
                    self.assertGreaterEqual(sphere.radius - distance, 0.1 - 1e-12)

    def test_seed_does_not_change_global_random_state(self):
        state = random.getstate()
        generate_points_within_regions([Box()], [20], method='random', seed=7)
        self.assertEqual(random.getstate(), state)

    def test_implicit_region_and_estimated_volume(self):
        ball = ImplicitRegion(
            lambda x, y, z: x * x + y * y + z * z - 1,
            [(-1, 1), (-1, 1), (-1, 1)], label='implicit ball')
        self.assertTrue(ball.contains(0, 0, 0))
        self.assertFalse(ball.contains(1, 1, 1))
        volume = estimate_region_volume(ball, 16384, method='halton', seed=5)
        self.assertAlmostEqual(volume, 4 * math.pi / 3, delta=0.08)
        points = RBFMesh3D(ball).generate_points(100, method='sobol', seed=3)
        self.assertEqual(len(points), 100)
        self.assertTrue(all(ball.contains(p.x, p.y, p.z, strict=True) for p in points))

    def test_primitives_and_exact_volume_allocation(self):
        regions = [Box((0, 0, 0), (1, 1, 1), label='box'),
                   Cylinder((3, 0, 0), 1, 1, axis='x', label='cylinder')]
        expected_box = round(1000 / (1 + math.pi))
        allocation = calculate_volume_allocation(regions, 1000)
        self.assertEqual(allocation[0], expected_box)
        self.assertEqual(sum(allocation), 1000)
        points = RBFMesh3D(*regions).generate_points(1000, method='halton', seed=9)
        self.assertEqual(Counter(point.label for point in points),
                         {'box': allocation[0], 'cylinder': allocation[1]})
        for point in points:
            region = regions[0] if point.label == 'box' else regions[1]
            self.assertTrue(region.contains(point.x, point.y, point.z, strict=True))

    def test_append_replace_and_failed_generation_preserve_points(self):
        mesh = RBFMesh3D(Sphere())
        self.assertEqual(len(mesh.generate_points(10, seed=1)), 10)
        self.assertEqual(len(mesh.generate_points(5, seed=2)), 15)
        self.assertEqual(len(mesh.generate_points(7, seed=3, append=False)), 7)
        previous = list(mesh.Points)
        with self.assertRaises(ValueError):
            mesh.generate_points(1, boundary_distance=1, append=False)
        self.assertEqual(mesh.Points, previous)

    def test_sphere_boundary_points_have_outward_normals(self):
        sphere = Sphere((1, -2, 0.5), 2, label='fluid', boundary_label='wall')
        for method in ('random', 'halton', 'sobol'):
            with self.subTest(method=method):
                points = generate_boundary_points(
                    [sphere], [300], method=method, seed=12)
                again = generate_boundary_points(
                    [sphere], [300], method=method, seed=12)
                self.assertEqual(coordinates(points), coordinates(again))
                self.assertEqual(len(points), 300)
                for point in points:
                    radial = tuple((coordinate - origin) / sphere.radius
                                   for coordinate, origin in
                                   zip((point.x, point.y, point.z), sphere.center))
                    self.assertAlmostEqual(math.dist(
                        (point.x, point.y, point.z), sphere.center), sphere.radius)
                    self.assertEqual(point.label, 'wall')
                    self.assertEqual(point.boundary_label, 'wall')
                    self.assertTrue(point.is_border)
                    for actual, expected in zip(point.normal, radial):
                        self.assertAlmostEqual(actual, expected)

    def test_box_and_cylinder_boundary_labels(self):
        box = Box((0, 0, 0), (1, 2, 3), boundary_labels={'zmax': 'outlet'})
        box_points = generate_boundary_points([box], [2000], method='halton', seed=2)
        labels = {point.boundary_label for point in box_points}
        self.assertEqual(labels, {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'outlet'})
        for point in box_points:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in point.normal)), 1)
            self.assertTrue(box.contains(point.x, point.y, point.z))

        cylinder = Cylinder(radius=2, height=3,
                            boundary_labels={'side': 'wall', 'top': 'outlet',
                                             'bottom': 'inlet'})
        cylinder_points = generate_boundary_points(
            [cylinder], [2000], method='sobol', seed=4)
        self.assertEqual({point.boundary_label for point in cylinder_points},
                         {'wall', 'outlet', 'inlet'})
        self.assertTrue(all(cylinder.contains(point.x, point.y, point.z)
                            for point in cylinder_points))

    def test_custom_implicit_boundary_projection_and_finite_difference_normals(self):
        region = ImplicitRegion(
            lambda x, y, z: x * x + y * y + z * z - 1,
            [(-1.1, 1.1)] * 3, boundary_label='implicit wall')
        points = RBFMesh3D(region).generate_boundary_points(
            250, method='halton', seed=8, tolerance=1e-9)
        self.assertEqual(len(points), 250)
        for point in points:
            radius = math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            self.assertAlmostEqual(radius, 1, places=7)
            dot = point.x * point.normal[0] + point.y * point.normal[1] + \
                point.z * point.normal[2]
            self.assertGreater(dot, 0.999999)

    def test_boundary_allocation_and_separate_storage(self):
        sphere = Sphere(radius=1)
        box = Box((-1, -1, -1), (1, 1, 1))
        allocation = calculate_boundary_allocation([sphere, box], 101)
        self.assertEqual(sum(allocation), 101)
        expected_sphere = round(101 * sphere.surface_area /
                                (sphere.surface_area + box.surface_area))
        self.assertEqual(allocation[0], expected_sphere)

        mesh = RBFMesh3D(sphere)
        mesh.generate_points(20, boundary_distance=0.1, seed=1)
        mesh.generate_boundary_points(30, seed=2)
        self.assertEqual(len(mesh.Points), 20)
        self.assertEqual(len(mesh.Boundary_Points), 30)
        self.assertTrue(all(not point.is_border for point in mesh.Points))
        self.assertTrue(all(point.is_border and point.normal is not None
                            for point in mesh.Boundary_Points))
        mesh.generate_boundary_points(7, append=False, seed=3)
        self.assertEqual(len(mesh.Boundary_Points), 7)

    def test_validation_and_empty_requests(self):
        with self.assertRaisesRegex(ValueError, 'bounds'):
            ImplicitRegion(lambda x, y, z: -1, [(0, 0), (0, 1), (0, 1)])
        with self.assertRaisesRegex(ValueError, 'radius'):
            Sphere(radius=0)
        with self.assertRaisesRegex(ValueError, 'axis'):
            Cylinder(axis='q')
        with self.assertRaisesRegex(ValueError, 'boundary_distance'):
            generate_points_within_regions([Sphere()], [1], -1)
        with self.assertRaisesRegex(ValueError, 'clearance'):
            generate_points_within_regions(
                [ImplicitRegion(lambda x, y, z: x * x + y * y + z * z - 1,
                                [(-1, 1)] * 3)], [1], 0.1)
        with self.assertRaisesRegex(ValueError, 'explicit boundary'):
            calculate_boundary_allocation([
                ImplicitRegion(lambda x, y, z: x, [(-1, 1)] * 3),
                Sphere()], 10)
        self.assertEqual(RBFMesh3D().generate_points(0), [])
        self.assertEqual(RBFMesh3D().generate_boundary_points(0), [])
        with self.assertRaisesRegex(ValueError, 'positive-volume'):
            RBFMesh3D().generate_points(1)


if __name__ == '__main__':
    unittest.main()
