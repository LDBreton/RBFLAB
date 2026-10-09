"""Integration check for the documented Border -> PointCloud -> symbolic PDE route."""
import numpy as np
from examples.parametric_borders import run


def test_border_recipe_labels_normals_and_mixed_pde():
    mesh, cloud, _, metrics = run()
    assert len(cloud.points) == 384
    assert {name: len(ids) for name, ids in cloud.boundary.items()} == {"outer": 96, "hole": 48}
    assert cloud.triangles.shape == (0, 3)
    assert not mesh.holes_polygons[0].exterior.is_ccw
    for label, sign in (("outer", 1), ("hole", -1)):
        p = cloud.points[cloud.boundary[label]]
        normals = cloud.normals[label]
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        assert np.all(sign*np.sum(normals*p, axis=1) > 0)
    assert metrics["nodal_max"] < .0015
