import math
import random
import unittest

import numpy as np

from rbflab.meshgen import ParametricSurface3D


def coordinates(points):
    return [(point.x, point.y, point.z) for point in points]


class ParametricSurface3DTests(unittest.TestCase):
    @staticmethod
    def sphere():
        return ParametricSurface3D(
            lambda u, v: (np.sin(v) * np.cos(u),
                          np.sin(v) * np.sin(u),
                          np.cos(v)),
            (0, 2 * math.pi), (0, math.pi), label='sphere',
            periodic_u=True, orientation=-1)

    def test_sphere_points_are_uniformly_sampled_with_outward_normals(self):
        points = self.sphere().generate_points(4000, method='halton', seed=42)
        self.assertEqual(len(points), 4000)
        self.assertTrue(all(point.is_border for point in points))
        self.assertTrue(all(point.boundary_label == 'sphere' for point in points))
        for point in points:
            radius = math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            self.assertAlmostEqual(radius, 1, places=10)
            alignment = (point.x * point.normal[0] + point.y * point.normal[1] +
                         point.z * point.normal[2])
            self.assertGreater(alignment, 0.999999)
        # Uniform sphere-area sampling makes z uniform on [-1, 1].
        northern_fraction = sum(point.z > 0 for point in points) / len(points)
        self.assertAlmostEqual(northern_fraction, 0.5, delta=0.025)

    def test_analytic_derivatives_define_constant_plane_normal(self):
        surface = ParametricSurface3D(
            lambda u, v: (u, v, 2 * u + 3 * v), (0, 1), (0, 1),
            derivatives=(lambda u, v: (1, 0, 2),
                         lambda u, v: (0, 1, 3)), label='plane')
        points = surface.generate_points(100, seed=5)
        expected = np.asarray((-2, -3, 1)) / math.sqrt(14)
        for point in points:
            self.assertAlmostEqual(point.z, 2 * point.x + 3 * point.y)
            np.testing.assert_allclose(point.normal, expected, atol=1e-12)

    def test_methods_are_reproducible_and_seeded_random_is_isolated(self):
        for method in ('random', 'halton', 'sobol'):
            with self.subTest(method=method):
                state = random.getstate()
                first = self.sphere().generate_points(137, method=method, seed=7)
                again = self.sphere().generate_points(137, method=method, seed=7)
                other = self.sphere().generate_points(137, method=method, seed=8)
                self.assertEqual(coordinates(first), coordinates(again))
                self.assertNotEqual(coordinates(first), coordinates(other))
                self.assertEqual(random.getstate(), state)

    def test_append_replace_and_zero_requests(self):
        surface = self.sphere()
        self.assertEqual(len(surface.generate_points(10, seed=1)), 10)
        self.assertEqual(len(surface.generate_points(5, seed=2)), 15)
        self.assertEqual(len(surface.generate_points(7, seed=3, append=False)), 7)
        self.assertEqual(len(surface.generate_points(0)), 7)
        self.assertEqual(surface.generate_points(0, append=False), [])

    def test_invalid_and_degenerate_surfaces_raise(self):
        with self.assertRaisesRegex(ValueError, 'u_range'):
            ParametricSurface3D(lambda u, v: (u, v, 0), (1, 1), (0, 1))
        with self.assertRaisesRegex(ValueError, 'orientation'):
            ParametricSurface3D(lambda u, v: (u, v, 0), (0, 1), (0, 1),
                                orientation=0)
        with self.assertRaisesRegex(ValueError, 'three finite'):
            ParametricSurface3D(lambda u, v: (u, math.nan), (0, 1), (0, 1))
        degenerate = ParametricSurface3D(
            lambda u, v: (u, 0 * v, 0 * v), (0, 1), (0, 1))
        with self.assertRaisesRegex(ValueError, 'zero-area'):
            degenerate.generate_points(1)


if __name__ == '__main__':
    unittest.main()
