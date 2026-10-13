"""Physical-map checks independent of the point generator's implementation."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from shapely import contains_xy
from rbflab import geometry as g, meshgen as m


def annulus():
    return g.Annulus(inner_radius=.35, outer_radius=1)


@pytest.mark.parametrize("degree", [1, 2, 7, 16])
@pytest.mark.parametrize("beta", [0., 2.])
def test_annulus_matches_analytic_arc_length_radial_map(degree, beta):
    cloud = m.mapped_padua(annulus(), degree=degree, boundary=60, clustering=beta)
    ref = m.padua_points(degree)
    s, r = np.arccos(ref[:, 0])/np.pi, (1-ref[:, 1])/2
    take = (s < 1) & (r > 0) & (r < 1)
    s, r = s[take], r[take]
    radial = np.expm1(beta*r)/np.expm1(beta) if beta else r
    expected_radius = .35+.65*radial
    # Annulus inner coordinates run counterclockwise; normals point inward.
    expected = expected_radius[:, None]*np.column_stack((np.cos(2*np.pi*s), np.sin(2*np.pi*s)))
    np.testing.assert_allclose(cloud.interior, expected, atol=1.3e-6, rtol=0)
    assert len(cloud.triangles) == 0


def test_grading_increases_density_without_changing_count():
    linear = m.mapped_padua(annulus(), degree=20, boundary=80, clustering=0)
    graded = m.mapped_padua(annulus(), degree=20, boundary=80, clustering=3)
    assert linear.points.shape == graded.points.shape
    r0 = np.linalg.norm(linear.interior, axis=1)
    r1 = np.linalg.norm(graded.interior, axis=1)
    assert np.all(r1 < r0)
    assert np.count_nonzero(r1 < .45) > np.count_nonzero(r0 < .45)
    base = m.generate(annulus(), interior=0, boundary=80)
    for label, ids in base.boundary.items():
        np.testing.assert_array_equal(graded.points[graded.boundary[label]], base.points[ids])
        np.testing.assert_array_equal(graded.normals[label], base.normals[label])


def test_arc_length_mapping_is_invariant_to_curve_speed():
    def domain(warp):
        def angle(t):
            return t+warp*np.sin(t)
        hole = g.ParametricBoundary(lambda t: (.35*np.cos(angle(t)), .35*np.sin(angle(t))), label="hole")
        outer = g.ParametricBoundary(lambda t: (np.cos(t), np.sin(t)), label="outer")
        return g.ParametricDomain(outer, holes=[hole])
    a = m.mapped_padua(domain(0), degree=18, boundary=80)
    b = m.mapped_padua(domain(.4), degree=18, boundary=80)
    np.testing.assert_allclose(a.interior, b.interior, atol=2e-6, rtol=0)


def two_holes():
    return g.with_holes(g.Rectangle(width=4, height=3), {
        "hole_left": g.Ellipse(a=.45, b=.65, center=(-.9, 0)),
        "hole_right": g.Disk(radius=.4, center=(.9, 0)),
    })


def test_multi_hole_coverage_divider_and_normals():
    domain = two_holes()
    c = m.mapped_padua(domain, degree=18, boundary=160)
    assert contains_xy(domain.polygon, c.interior[:, 0], c.interior[:, 1]).all()
    # The x=0 shared divider belongs to both regions, and is not a BC group.
    shared = np.intersect1d(c.regions["patch_0"], c.regions["patch_1"])
    assert len(shared) > 5
    np.testing.assert_allclose(c.points[shared, 0], 0, atol=1e-12)
    # A divider may meet an existing exterior sample at its endpoint.
    endpoints = np.intersect1d(shared, c.boundary_indices)
    np.testing.assert_allclose(np.abs(c.points[endpoints, 1]), 1.5)
    assert not c.interfaces
    assert cKDTree(c.points).query(c.points, k=2)[0][:, 1].min() > 1e-5
    # Both outer and hole neighborhoods are populated in each half.
    for sign in [-1, 1]:
        p = c.interior[c.interior[:, 0]*sign > 0]
        assert p[:, 1].max() > 1.3 and p[:, 1].min() < -1.3
    for label in ("hole_left", "hole_right"):
        center = np.array([-.9 if label == "hole_left" else .9, 0.])
        p = c.points[c.boundary[label]]
        assert np.all(np.sum((p-center)*c.normals[label], axis=1) < 0)


def test_border_and_rbfmesh_keep_counts_normals_and_input():
    outer = g.Border(lambda t: (np.cos(t), np.sin(t)), "outer", 0, 2*np.pi)(70)
    hole = g.Border(lambda t: (.35*np.cos(t), .35*np.sin(t)), "hole", 0, 2*np.pi)(-42)
    mesh = m.RBFMesh(outer, hole)
    mesh.generate_points(5, seed=4)
    old = list(mesh.Points)
    cloud = m.mapped_padua(mesh, degree=16)
    base = m.generate(mesh, interior=0)
    for label, ids in base.boundary.items():
        np.testing.assert_array_equal(cloud.points[cloud.boundary[label]], base.points[ids])
        np.testing.assert_array_equal(cloud.normals[label], base.normals[label])
    assert mesh.Points == old and hole.n_segments == -42
    other = m.mapped_padua([outer, hole], degree=16)
    np.testing.assert_array_equal(cloud.points, other.points)


def test_repeated_border_labels_remain_supported():
    borders = [g.Border(lambda t: (t, -1), "wall", -1, 1)(20),
               g.Border(lambda t: (1, t), "wall", -1, 1)(20),
               g.Border(lambda t: (-t, 1), "wall", -1, 1)(20),
               g.Border(lambda t: (-1, -t), "wall", -1, 1)(20),
               g.Border(lambda t: (.25*np.cos(t), .25*np.sin(t)), "hole", 0, 2*np.pi)(-32)]
    c = m.mapped_padua(borders, degree=12)
    assert len(c.boundary["wall"]) == 80 and len(c.boundary["hole"]) == 32


@pytest.mark.parametrize("kwargs", [dict(degree=0), dict(degree=True),
    dict(degree=10, clustering=-1), dict(degree=10, clustering=np.inf),
    dict(degree=10, clustering=9), dict(degree=10, min_spacing=-1),
    dict(degree=10, max_candidates=10), dict(degree=10, max_candidates=0),
    dict(degree=10, centers=[[0, 0, 0]]), dict(degree=10, centers=[[np.nan, 0]])])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        m.mapped_padua(annulus(), **kwargs)


def test_spacing_is_checked_without_thinning():
    with pytest.raises(ValueError, match="separation"):
        m.mapped_padua(annulus(), degree=12, min_spacing=.1)


def test_unsupported_topologies_and_patch_crossings_fail():
    with pytest.raises(ValueError, match="at least one hole"):
        m.mapped_padua(g.Disk(), degree=12)
    with pytest.raises(ValueError, match="containing its center"):
        m.mapped_padua(annulus(), degree=12, centers=[[.8, 0]])
    crossing = g.with_holes(g.Disk(radius=3), {
        "a": g.Ellipse(a=.6, b=.2, center=(-.4, 0)),
        "b": g.Disk(radius=.1, center=(.6, 0))})
    with pytest.raises(ValueError, match="crosses its patch"):
        m.mapped_padua(crossing, degree=12)
    concave = g.with_holes(g.Polygon([(-2,-2), (2,-2), (2,2), (.3,2),
                                      (.3,0), (-.3,0), (-.3,2), (-2,2)]),
                          {"hole": g.Disk(radius=.15, center=(-1, 1))})
    with pytest.raises(ValueError, match="star-shaped"):
        m.mapped_padua(concave, degree=12)


def test_shared_junctions_do_not_duplicate_nodes():
    domain = g.with_holes(g.Disk(radius=2), {
        "a": g.Disk(radius=.2, center=(-.7, -.4)),
        "b": g.Disk(radius=.2, center=(.7, -.4)),
        "c": g.Disk(radius=.2, center=(0, .8))})
    cloud = m.mapped_padua(domain, degree=15, boundary=150)
    assert len(np.unique(cloud.points, axis=0)) == len(cloud.points)
    shared = set(cloud.regions["patch_0"]) & set(cloud.regions["patch_1"]) & set(cloud.regions["patch_2"])
    assert len(shared) == 1
