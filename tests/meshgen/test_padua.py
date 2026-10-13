import numpy as np
import pytest
from rbflab import geometry, meshgen


@pytest.mark.parametrize("degree", [1, 2, 3, 8, 15])
def test_reference_count_and_polynomial_unisolvence(degree):
    p = meshgen.padua_points(degree)
    assert len(p) == (degree+1)*(degree+2)//2
    assert len(np.unique(p, axis=0)) == len(p)
    assert np.all(np.abs(p) <= 1)
    if degree <= 8:
        # Independent total-degree polynomial interpolation criterion.
        v = np.column_stack([p[:, 0]**i*p[:, 1]**j
                             for i in range(degree+1)
                             for j in range(degree+1-i)])
        assert np.linalg.matrix_rank(v) == len(p)


def test_arc_length_holes_labels_and_normals():
    domain = geometry.Annulus(inner_radius=.35, outer_radius=1)
    cloud = meshgen.generate(domain, method="padua", degree=20,
                             boundary={"inner": 30, "outer": 60})
    baseline = meshgen.generate(domain, interior=0,
                                boundary={"inner": 30, "outer": 60})
    assert len(cloud.interior) > 0
    radii = np.linalg.norm(cloud.interior, axis=1)
    assert np.all((radii > .35) & (radii < 1))
    for label in baseline.boundary:
        np.testing.assert_array_equal(cloud.points[cloud.boundary[label]],
                                      baseline.points[baseline.boundary[label]])
        np.testing.assert_array_equal(cloud.normals[label], baseline.normals[label])
    assert len(cloud.triangles) == 0


def test_border_generation_preserves_samples_and_input():
    border = geometry.Border(lambda t: (np.cos(t), np.sin(t)), "wall", 0, 2*np.pi)(60)
    cloud = meshgen.generate(border, method="padua", degree=18)
    baseline = meshgen.generate(border, interior=0)
    assert len(cloud.interior) > 0
    np.testing.assert_array_equal(cloud.points[cloud.boundary["wall"]],
                                  baseline.points[baseline.boundary["wall"]])
    np.testing.assert_allclose(cloud.normals["wall"], baseline.normals["wall"])
    assert border.n_segments == 60


def test_clearance_and_reproducibility():
    domain = geometry.Disk()
    a = meshgen.generate(domain, method="padua", degree=18,
                         boundary_distance=.1, seed=1)
    b = meshgen.generate(domain, method="padua", degree=18,
                         boundary_distance=.1, seed=99)
    np.testing.assert_array_equal(a.points, b.points)
    assert np.all(np.linalg.norm(a.interior, axis=1) <= .9+1e-12)


@pytest.mark.parametrize("kwargs", [dict(method="padua"),
    dict(method="padua", degree=True), dict(method="padua", degree=0),
    dict(method="padua", degree=3, interior=10), dict(degree=3),
    dict(method="padua", degree=20, max_candidates=10)])
def test_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        meshgen.generate(geometry.Disk(), **kwargs)


def test_reject_3d_and_allow_empty_interior():
    with pytest.raises(ValueError, match="planar"):
        meshgen.generate(geometry.Sphere(), method="padua", degree=4)
    cloud = meshgen.generate(geometry.Disk(), method="padua", degree=1)
    assert len(cloud.interior) == 0
