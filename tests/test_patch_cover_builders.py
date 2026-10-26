import numpy as np
import pytest
import mpmath

import rbflab as r
from rbflab.geometry import TriangleMesh2D


@pytest.mark.parametrize('dimension', [2, 3])
def test_points_are_strictly_covered_and_reproducible(dimension):
    rng = np.random.default_rng(43)
    points = rng.random((71, dimension))
    first = r.PatchCover.from_points(points, points_per_patch=11,
                                     min_points=dimension + 1, seed=7)
    second = r.PatchCover.from_points(points, points_per_patch=11,
                                      min_points=dimension + 1, seed=7)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.radii, second.radii)
    distances = np.linalg.norm(points[:, None, :] - first.centers[None, :, :], axis=2)
    members = distances < first.radii
    assert len(first.centers) == 7
    assert members.any(axis=1).all()
    assert (members.sum(axis=0) >= dimension + 1).all()


@pytest.mark.parametrize('points', [
    np.empty((0, 2)), np.zeros((3, 1)), [[0, 0], [0, 0], [1, 1]],
    [[0, 0], [np.inf, 1]], [[0, 0], [np.nan, 1]],
])
def test_bad_point_cloud_rejected(points):
    with pytest.raises(ValueError):
        r.PatchCover.from_points(points)


@pytest.mark.parametrize('kwargs', [
    {'n_patches': 0}, {'n_patches': True}, {'points_per_patch': 0},
    {'points_per_patch': 2.5}, {'min_points': 0}, {'min_points': True},
    {'overlap': -0.1}, {'overlap': 1}, {'overlap': np.nan},
    {'seed': -1}, {'seed': True},
])
def test_bad_options_rejected(kwargs):
    with pytest.raises(ValueError):
        r.PatchCover.from_points([[0, 0], [1, 0], [0, 1]], **kwargs)


def test_singleton_clusters_receive_positive_radii():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cover = r.PatchCover.from_points(points, n_patches=3, min_points=1, overlap=0)
    assert (cover.radii > 0).all()
    assert all(len(row) >= 1 for row in cover.memberships(points))


def test_one_point_cover_supports_interpolation_and_query():
    points = np.array([[2., -3.]])
    cover = r.PatchCover.from_points(points, min_points=1, overlap=0)
    assert cover.memberships(points)[0].tolist() == [0]
    fit = r.interpolate(r.Gaussian(1), points, np.array([7.]), patches=cover)
    np.testing.assert_allclose(fit.evaluate(points), [7.], atol=1e-12)


def test_zero_overlap_actual_membership_at_high_precision():
    rng = np.random.default_rng(711)
    points = rng.normal(size=(43, 3))
    cover = r.PatchCover.from_points(points, n_patches=8,
                                     min_points=4, overlap=0, seed=17)
    ctx = mpmath.mp.clone()
    ctx.dps = 60
    for rows in (cover.memberships(points), cover.memberships(points, ctx)):
        mask = np.zeros((len(points), len(cover.centers)), dtype=bool)
        for j, members in enumerate(rows):
            mask[members, j] = True
        assert mask.any(axis=1).all()
        assert (mask.sum(axis=0) >= 4).all()


def test_separated_point_groups_remain_separate_with_sufficient_patches():
    group = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    points = np.vstack((group, group + [100., 0.]))
    cover = r.PatchCover.from_points(points, n_patches=2, min_points=3)
    assert cover.centers[0, 0] < 10 or cover.centers[1, 0] < 10
    assert cover.centers[0, 0] > 90 or cover.centers[1, 0] > 90
    assert np.max(cover.radii) < 10


def test_mesh_every_triangle_is_strictly_inside_one_ball():
    vertices = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = TriangleMesh2D(vertices, [[0, 1, 2], [0, 2, 3]])
    cover = r.PatchCover.from_mesh(mesh, n_patches=2)
    ctx = mpmath.mp.clone()
    ctx.dps = 60
    for rows in (cover.memberships(vertices), cover.memberships(vertices, ctx)):
        mask = np.zeros((len(vertices), len(cover.centers)), dtype=bool)
        for j, members in enumerate(rows):
            mask[members, j] = True
        assert np.all(np.any(np.all(mask[mesh.triangles], axis=1), axis=1))
    for triangle in mesh.triangles:
        distance = np.linalg.norm(vertices[triangle][:, None, :] - cover.centers[None, :, :], axis=2)
        assert np.any(np.all(distance < cover.radii, axis=0))
        interior = vertices[triangle].mean(axis=0)
        assert np.any(np.linalg.norm(cover.centers - interior, axis=1) < cover.radii)


def test_disconnected_mesh_allocates_a_patch_per_component():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    vertices = np.vstack((square, square + [100., 0.]))
    mesh = TriangleMesh2D(vertices, [[0, 1, 2], [0, 2, 3],
                                     [4, 5, 6], [4, 6, 7]])
    cover = r.PatchCover.from_mesh(mesh, points_per_patch=100)
    assert len(cover.centers) == 2
    assert np.max(cover.radii) < 10
    with pytest.raises(ValueError, match='n_patches'):
        r.PatchCover.from_mesh(mesh, n_patches=1)


def test_mesh_type_required():
    with pytest.raises(TypeError):
        r.PatchCover.from_mesh([[0, 0], [1, 0], [0, 1]])
