import tempfile
from pathlib import Path
import unittest

import numpy as np

from rbflab.geometry import (TriangleMesh2D, staggered_clouds,
                        save_staggered_layout, load_staggered_layout)


VERTICES = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])
WALLS = {'bottom': [[0, 1]], 'right': [[1, 2]],
         'top': [[2, 3]], 'left': [[3, 0]]}


class StaggeredTests(unittest.TestCase):
    def test_square_topology_and_boundaries(self):
        result = staggered_clouds(VERTICES, TRIANGLES, WALLS)
        np.testing.assert_array_equal(result.edges, [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])
        np.testing.assert_array_equal(result.triangle_edges, [[0, 3, 1], [1, 4, 2]])
        np.testing.assert_array_equal(result.edge_triangles, [[0, -1], [0, 1], [1, -1], [0, -1], [1, -1]])
        np.testing.assert_array_equal(result.vertices.points, VERTICES)
        np.testing.assert_array_equal(result.vertices.triangles, TRIANGLES)
        self.assertEqual(result.edge_midpoints.points.shape, (5, 2))
        self.assertEqual(result.edge_midpoints.triangles.shape, (0, 3))
        np.testing.assert_array_equal(result.edge_midpoints.interior_indices, [1])
        self.assertEqual(len(result.vertices.interior_indices), 0)
        self.assertIn(0, result.vertices.boundary['bottom'])
        self.assertIn(0, result.vertices.boundary['left'])
        for label, normal in [('bottom', [0, -1]), ('right', [1, 0]),
                              ('top', [0, 1]), ('left', [-1, 0])]:
            np.testing.assert_allclose(result.edge_midpoints.normals[label], [normal])
            np.testing.assert_allclose(result.vertices.normals[label], [normal, normal])

    def test_orientation_and_determinism(self):
        reference = staggered_clouds(VERTICES, TRIANGLES, WALLS)
        for triangles in [TRIANGLES.copy(), TRIANGLES[:, ::-1], np.array([[2, 1, 0], [0, 2, 3]])]:
            result = staggered_clouds(VERTICES, triangles, WALLS)
            np.testing.assert_array_equal(result.edges, reference.edges)
            np.testing.assert_array_equal(result.edge_triangles, reference.edge_triangles)
            for cloud in ['vertices', 'edge_midpoints']:
                actual, expected = getattr(result, cloud), getattr(reference, cloud)
                np.testing.assert_array_equal(actual.points, expected.points)
                for label in WALLS:
                    np.testing.assert_allclose(actual.normals[label], expected.normals[label])
            # Local edge columns follow the supplied triangle vertex order.
            for row, ids in zip(triangles, result.triangle_edges):
                np.testing.assert_array_equal(result.edges[ids],
                    np.sort([row[[0, 1]], row[[1, 2]], row[[2, 0]]], axis=1))

    def test_mesh_and_array_inputs_are_independent(self):
        vertices, triangles = VERTICES.copy(), TRIANGLES.copy()
        mesh = TriangleMesh2D(vertices, triangles, WALLS)
        result = staggered_clouds(mesh)
        result.vertices.points[0] = 20
        result.edges[0] = 0
        np.testing.assert_array_equal(vertices, VERTICES)
        np.testing.assert_array_equal(triangles, TRIANGLES)
        np.testing.assert_array_equal(mesh.vertices, VERTICES)
        np.testing.assert_array_equal(mesh.edges[0], [0, 1])
        with self.assertRaises(ValueError):
            staggered_clouds(mesh, triangles)

    def test_hole_normals(self):
        outer = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
        inner = outer / 2
        triangles = []
        for i in range(4):
            j = (i + 1) % 4
            triangles.extend([[i, j, j + 4], [i, j + 4, i + 4]])
        walls = {'outer': [[i, (i + 1) % 4] for i in range(4)],
                 'hole': [[i + 4, (i + 1) % 4 + 4] for i in range(4)]}
        result = staggered_clouds(np.vstack([outer, inner]), triangles, walls)
        for cloud in [result.vertices, result.edge_midpoints]:
            for label, sign in [('outer', 1), ('hole', -1)]:
                indices = cloud.boundary[label]
                normals = cloud.normals[label]
                self.assertTrue(np.all(sign * np.sum(cloud.points[indices] * normals, axis=1) > 0))
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)

    def test_interface_is_not_exterior(self):
        result = staggered_clouds(VERTICES, TRIANGLES, WALLS,
                                  interface_edges={'diagonal': [[0, 2]]})
        np.testing.assert_array_equal(result.edge_midpoints.interfaces['diagonal'], [1])
        np.testing.assert_array_equal(result.vertices.interfaces['diagonal'], [0, 2])
        self.assertNotIn('diagonal', result.edge_midpoints.boundary)
        np.testing.assert_array_equal(result.edge_midpoints.interior_indices, [1])

    def test_invalid_inputs(self):
        cases = [
            (VERTICES, [[0, 1, 4]], None),
            (VERTICES, [[-1, 1, 2]], None),
            (VERTICES, TRIANGLES.astype(float), None),
            (VERTICES, [[0, 0, 1], [0, 2, 3]], None),
            (VERTICES, [[0, 1, 2], [2, 1, 0], [0, 2, 3]], None),
            (np.array([[0, 0], [1, 0], [2, 0]]), [[0, 1, 2]], None),
            (np.array([[0, 0], [1, 0], [0, 0]]), [[0, 1, 2]], None),
            (np.array([[0, 0], [np.nan, 0], [0, 1]]), [[0, 1, 2]], None),
            (VERTICES, TRIANGLES, {}),
            (VERTICES, TRIANGLES, {'wrong': [[0, 2]]}),
            (VERTICES, TRIANGLES, {'wrong': [[1, 3]]}),
            (VERTICES, TRIANGLES, {'wrong': [[0, 0]]}),
            (VERTICES, TRIANGLES, {'wrong': [[0, 1], [1, 0]]}),
            (VERTICES, [], None),
        ]
        for vertices, triangles, walls in cases:
            with self.subTest(triangles=triangles, walls=walls), self.assertRaises(ValueError):
                staggered_clouds(vertices, triangles, walls)
        with self.assertRaisesRegex(ValueError, 'Nonmanifold'):
            staggered_clouds([[0, 0], [1, 0], [0, 1], [0, -1], [1, 1]],
                              [[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        with self.assertRaises(ValueError):
            staggered_clouds(VERTICES, TRIANGLES, WALLS, interface_edges={'bad': [[0, 1]]})

    def test_overlapping_and_nonconforming_meshes(self):
        for vertices, triangles in [
            ([[0, 0], [2, 0], [0, 2], [.2, .2], [.3, .2], [.2, .3]], [[0, 1, 2], [3, 4, 5]]),
            ([[0, 0], [2, 0], [0, 2], [1, 0], [1, -1]], [[0, 1, 2], [0, 3, 4]]),
            ([[0, 0], [2, 0], [0, 2], [1, 1]], [[0, 1, 2], [0, 1, 3]]),
        ]:
            with self.subTest(vertices=vertices), self.assertRaisesRegex(ValueError, 'nonconforming'):
                staggered_clouds(vertices, triangles)

    def test_normal_overrides_and_ambiguity(self):
        result = staggered_clouds(VERTICES, TRIANGLES, WALLS,
                                  vertex_normals={'bottom': [[0, -1], [0, -1]]},
                                  edge_midpoint_normals={'bottom': [[0, -1]]})
        self.assertEqual(result.normal_policy['bottom']['vertices'], 'supplied')
        for normals in [[[0, 0]], [[np.nan, 1]], [[1, 1]], [[0, -1], [0, -1]]]:
            with self.assertRaises(ValueError):
                staggered_clouds(VERTICES, TRIANGLES, WALLS, edge_midpoint_normals={'bottom': normals})
        with self.assertRaises(ValueError):
            staggered_clouds(VERTICES, TRIANGLES, vertex_normals={'unknown': []})
        # Two components touch at a vertex: one label has cancelling normals.
        vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            staggered_clouds(vertices, [[0, 1, 2], [0, 3, 4]])

    def test_portable_round_trip(self):
        walls = dict(WALLS)
        walls[7] = walls.pop('bottom')
        original = staggered_clouds(VERTICES, TRIANGLES, walls,
                                    interface_edges={'diagonal': [[0, 2]]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layout.npz'
            save_staggered_layout(path, original)
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, 'O')
            restored = load_staggered_layout(path)
        self.assertEqual(restored.normal_policy, original.normal_policy)
        for name in ['edges', 'triangle_edges', 'edge_triangles']:
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        for name in ['vertices', 'edge_midpoints']:
            actual, expected = getattr(restored, name), getattr(original, name)
            np.testing.assert_array_equal(actual.points, expected.points)
            np.testing.assert_array_equal(actual.triangles, expected.triangles)
            for mapping in ['boundary', 'normals', 'interfaces']:
                self.assertEqual(set(getattr(actual, mapping)), set(getattr(expected, mapping)))
                for label in getattr(expected, mapping):
                    np.testing.assert_array_equal(getattr(actual, mapping)[label], getattr(expected, mapping)[label])


if __name__ == '__main__':
    unittest.main()
