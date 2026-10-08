"""Optional real Gmsh integration; install gmsh to enable these tests."""
import importlib.util
from pathlib import Path
import runpy
import unittest

import numpy as np

from rbflab.geometry import staggered_clouds


@unittest.skipUnless(importlib.util.find_spec('gmsh'), 'Optional gmsh package is not installed')
class GmshIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        example = runpy.run_path(str(Path(__file__).parents[2] / 'examples' / 'perforated_plate.py'))
        cls.mesh, cls.tags = example['build_perforated_plate']()
        cls.holes = example['HOLES']
        cls.layout = staggered_clouds(cls.mesh)

    def test_topology_and_tag_mapping(self):
        mesh, layout = self.mesh, self.layout
        self.assertGreater(len(mesh.triangles), 100)
        self.assertTrue(np.all(np.diff(self.tags) > 0))
        self.assertFalse(np.array_equal(self.tags, np.arange(len(self.tags))))
        self.assertEqual(len(self.tags), len(mesh.vertices))
        # Euler characteristic for one connected planar region with three holes.
        self.assertEqual(len(mesh.vertices) - len(mesh.edges) + len(mesh.triangles), -2)
        self.assertEqual(3 * len(mesh.triangles), 2 * len(mesh.edges) - len(mesh.exterior_edges))
        np.testing.assert_allclose(layout.edge_midpoints.points,
                                   mesh.vertices[mesh.edges].mean(axis=1))
        boundary = np.unique(np.concatenate(list(layout.edge_midpoints.boundary.values())))
        np.testing.assert_array_equal(boundary, mesh.exterior_edges)

    def test_physical_labels_and_hole_normals(self):
        for cloud in [self.layout.vertices, self.layout.edge_midpoints]:
            self.assertEqual(set(cloud.boundary), {'bottom', 'right', 'top', 'left',
                                                  'hole_1', 'hole_2', 'hole_3'})
            for i, (x, y, radius) in enumerate(self.holes):
                label = f'hole_{i + 1}'
                points = cloud.points[cloud.boundary[label]]
                normals = cloud.normals[label]
                toward_center = np.array([x, y]) - points
                cosine = np.sum(toward_center * normals, axis=1) / np.linalg.norm(toward_center, axis=1)
                self.assertTrue(np.all(cosine > .99))
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
                if cloud is self.layout.vertices:
                    np.testing.assert_allclose(np.linalg.norm(toward_center, axis=1), radius)
            for label, expected in [('bottom', [0, -1]), ('right', [1, 0]),
                                    ('top', [0, 1]), ('left', [-1, 0])]:
                np.testing.assert_allclose(cloud.normals[label],
                    np.tile(expected, (len(cloud.boundary[label]), 1)), atol=1e-12)
        # Every rectangular corner keeps both adjacent physical groups.
        for first, second in [('bottom', 'left'), ('bottom', 'right'),
                              ('top', 'left'), ('top', 'right')]:
            self.assertEqual(len(np.intersect1d(self.layout.vertices.boundary[first],
                                               self.layout.vertices.boundary[second])), 1)


if __name__ == '__main__':
    unittest.main()
