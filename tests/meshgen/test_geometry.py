import itertools
import math
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch



from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from rbflab.meshgen import Border, RBFMesh, exclude_nested_polygons, find_polygons, generate_regions
from rbflab.meshgen.polygons import resolve_multiple_overlaps


def edges(coordinates):
    return [Border(lambda t, a=a, b=b: (a[0] + t * (b[0] - a[0]),
                                       a[1] + t * (b[1] - a[1])),
                   i, 0, 1)(1)
            for i, (a, b) in enumerate(zip(coordinates, coordinates[1:] + coordinates[:1]))]


def circle(radius, segments=60):
    return Border(lambda t: (radius * math.cos(t), radius * math.sin(t)),
                  radius, 0, 2 * math.pi)(segments)


class GeometryTests(unittest.TestCase):
    def assert_partition(self, regions, expected):
        for region in regions:
            self.assertEqual(region.geom_type, 'Polygon')
            self.assertTrue(region.is_valid)
            self.assertGreater(region.area, 0)
        for a, b in itertools.combinations(regions, 2):
            self.assertAlmostEqual(a.intersection(b).area, 0, places=10)
        self.assertAlmostEqual(unary_union(regions).symmetric_difference(expected).area, 0, places=10)

    def test_nested_contours_form_three_regions(self):
        mesh = RBFMesh(circle(1), circle(0.8), circle(0.3))
        self.assertEqual(len(mesh.region_polygons), 3)
        expected = Polygon([(p.x, p.y) for p in mesh.borders[0].generate_points()])
        self.assert_partition(mesh.region_polygons, expected)
        self.assertEqual(len(mesh.Boundary_Points), 180)

    def test_dense_internal_circle_preserves_requested_border_points(self):
        internal = circle(0.8, 10000)
        internal.label = 'interface'
        mesh = RBFMesh(circle(1, 100), internal, circle(0.3, -100))
        points = [p for p in mesh.Boundary_Points if p.label == 'interface']
        self.assertEqual(len(points), 10000)
        self.assertEqual(len(mesh.Boundary_Points), 10200)
        self.assertTrue(all(p.is_border for p in points))
        for point in points:
            self.assertAlmostEqual(math.hypot(point.x, point.y), 0.8)

    def test_internal_border_flag_and_hole_clipping_are_respected(self):
        internal = circle(0.8, 100)
        internal.label = 'interface'
        internal.is_border = False
        mesh = RBFMesh(circle(1), internal)
        self.assertFalse(any(p.label == 'interface' for p in mesh.Boundary_Points))
        internal.is_border = True
        mesh = RBFMesh(circle(1), internal, circle(0.9, -100))
        self.assertFalse(any(p.label == 'interface' for p in mesh.Boundary_Points))

    def test_nested_helper_preserves_union_without_overlap(self):
        polygons = [box(-3, -3, 3, 3), box(-2, -2, 2, 2), box(-1, -1, 1, 1)]
        for order in itertools.permutations(polygons):
            regions = exclude_nested_polygons(order)
            self.assertEqual(len(regions), 3)
            self.assert_partition(regions, polygons[0])

    def test_overlaps_touching_duplicates_and_existing_holes(self):
        cases = [
            [box(0, 0, 2, 2), box(1, 0, 3, 2), box(0.5, 1, 2.5, 3)],
            [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            [box(0, 0, 1, 1), box(1, 1, 2, 2)],
            [box(0, 0, 1, 1), box(0, 0, 1, 1)],
            [box(0, 0, 3, 3).difference(box(1, 1, 2, 2))],
        ]
        for polygons in cases:
            with self.subTest(polygons=polygons):
                regions = resolve_multiple_overlaps(polygons)
                self.assert_partition(regions, unary_union(polygons))
                self.assertEqual(len(regions), len(resolve_multiple_overlaps(list(reversed(polygons)))))

    def test_holes_remove_whole_region_and_split_components(self):
        outer = [box(0, 0, 3, 3)]
        self.assertEqual(generate_regions(outer, [box(-1, -1, 4, 4)]), [])
        self.assertEqual(generate_regions(outer, outer), [])
        hole = box(1, -1, 2, 4)
        regions = generate_regions(outer, [hole])
        self.assertEqual(len(regions), 2)
        self.assert_partition(regions, outer[0].difference(hole))
        self.assertEqual(outer[0].area, 9)  # The caller's input is untouched.

    def test_empty_domain_is_valid_but_cannot_sample_positive_count(self):
        mesh = RBFMesh(circle(1), circle(2, -60))
        self.assertEqual(mesh.region_polygons, [])
        self.assertEqual(mesh.Boundary_Points, [])
        self.assertEqual(mesh.generate_points(0), [])
        with self.assertRaisesRegex(ValueError, 'positive-area'):
            mesh.generate_points(1)

    def test_unordered_edges_close_once(self):
        borders = edges([(0, 0), (1, 0), (1, 1), (0, 1)])
        for order in itertools.permutations(borders):
            contours = find_polygons(order)
            self.assertEqual(len(contours), 1)
            self.assertEqual(len(contours[0]), 4)
            self.assert_partition(RBFMesh(*order).region_polygons, box(0, 0, 1, 1))

    def test_open_and_reversed_edges_raise(self):
        borders = edges([(0, 0), (1, 0), (1, 1), (0, 1)])
        with self.assertRaisesRegex(ValueError, 'closed directed contour'):
            RBFMesh(*borders[:-1])
        borders[0](-1)
        with self.assertRaisesRegex(ValueError, 'closed directed contour'):
            RBFMesh(*borders)

    def test_closure_uses_requested_tolerance(self):
        borders = edges([(0, 0), (1, 0), (1, 1), (0, 1)])
        borders[-1] = Border(lambda t: (0, 1 - t * (1 - 1e-5)), 'almost closed', 0, 1)(1)
        self.assertEqual(len(RBFMesh(*borders, abs_tol=1e-4).region_polygons), 1)
        with self.assertRaisesRegex(ValueError, 'closed directed contour'):
            RBFMesh(*borders, abs_tol=1e-6)

    def test_dangling_branch_is_not_silently_ignored(self):
        borders = edges([(0, 0), (1, 0), (1, 1), (0, 1)])
        dangling = Border(lambda t: (1 + t, 1), 'dangling', 0, 1)(1)
        with self.assertRaisesRegex(ValueError, 'dangling'):
            RBFMesh(*borders, dangling)

    def test_self_intersection_and_degenerate_contours_raise(self):
        with self.assertRaisesRegex(ValueError, 'Invalid contour'):
            RBFMesh(*edges([(0, 0), (1, 1), (0, 1), (1, 0)]))
        with self.assertRaisesRegex(ValueError, 'Invalid contour'):
            RBFMesh(*edges([(0, 0), (1, 0), (2, 0)]))
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            RBFMesh(circle(1, 2))

    def test_segment_counts_coordinates_and_tolerances(self):
        border = circle(1)
        for count in [0, True, 2.5]:
            with self.assertRaisesRegex(ValueError, 'non-zero integer'):
                border(count)
        with self.assertRaisesRegex(ValueError, 'segment count'):
            RBFMesh(Border(lambda t: (math.cos(t), math.sin(t)), 0, 0, 2 * math.pi))
        with self.assertRaisesRegex(ValueError, 'finite coordinates'):
            Border(lambda t: (math.nan, t), 0, 0, 1)
        with self.assertRaisesRegex(ValueError, 'finite coordinates'):
            Border(lambda t: (t, math.inf if t == 0.5 else 0), 0, 0, 1)(2).generate_points()
        for tolerance in [0, -1, math.nan, math.inf]:
            with self.assertRaisesRegex(ValueError, 'tolerance'):
                RBFMesh(circle(1), abs_tol=tolerance)

    def test_all_examples_generate_exact_counts_and_disjoint_regions(self):
        root = Path(__file__).resolve().parent
        for index in range(1, 6):
            with self.subTest(example=index):
                mesh = runpy.run_path(str(root / 'examples' / f'example_{index}.py'))['random_mesh']
                self.assertEqual(len(mesh.Points), 10000)
                self.assert_partition(mesh.region_polygons,
                    unary_union(mesh.outer_polygons).difference(unary_union(mesh.holes_polygons)))
                if index == 2:
                    self.assertEqual(len(mesh.region_polygons), 2)
                    self.assertAlmostEqual(unary_union(mesh.region_polygons).area, 1.5)



if __name__ == '__main__':
    unittest.main()
