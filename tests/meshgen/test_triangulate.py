"""Optional boundary-driven constrained triangulation checks."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely import polygons, covers
from rbflab import geometry as g, meshgen as m

pytest.importorskip("meshpy.triangle")


def test_annulus_constraints_coverage_and_normals():
    domain = g.Annulus(inner_radius=.35, outer_radius=1.)
    counts = {"inner": 48, "outer": 64}
    mesh = m.triangulate(domain, boundary=counts)
    cloud = g.staggered_clouds(mesh).vertices
    assert len(cloud.boundary_indices) == sum(counts.values())
    sampled = domain.sample_boundary(counts)
    for label, (p, _) in sampled.items():
        assert len(mesh.boundary_edges[label]) == counts[label]
        np.testing.assert_allclose(cKDTree(mesh.vertices).query(p)[0], 0, atol=1e-14)
    polygon = Polygon(sampled["outer"][0], [sampled["inner"][0]])
    cells = polygons(mesh.vertices[mesh.triangles])
    assert covers(polygon.buffer(1e-12), cells).all()
    assert sum(cell.area for cell in cells) == pytest.approx(polygon.area, rel=1e-12)
    for label, sign in [("inner", -1), ("outer", 1)]:
        p = cloud.points[cloud.boundary[label]]
        assert np.all(sign*np.sum(p*cloud.normals[label], axis=1) > 0)


def test_denser_hole_boundary_refines_adjacent_interior():
    domain = g.Annulus(inner_radius=.35, outer_radius=1.)
    coarse = g.staggered_clouds(m.triangulate(domain, boundary={"inner": 24, "outer": 48})).vertices
    fine = g.staggered_clouds(m.triangulate(domain, boundary={"inner": 72, "outer": 48})).vertices
    def near_statistics(cloud):
        inside_band = np.linalg.norm(cloud.interior, axis=1) < .55
        spacing = cKDTree(cloud.points).query(cloud.interior, k=2)[0][:, 1]
        return inside_band.sum(), np.median(spacing[inside_band])
    n0, h0 = near_statistics(coarse)
    n1, h1 = near_statistics(fine)
    assert n1 > 2*n0
    assert h1 < .65*h0


def test_border_api_preserves_coordinates_counts_and_inputs():
    outer = g.Border(lambda t: (np.cos(t), np.sin(t)), "outer", 0, 2*np.pi)(48)
    hole = g.Border(lambda t: (.3*np.cos(t), .3*np.sin(t)), "hole", 0, 2*np.pi)(-40)
    source = m.RBFMesh(outer, hole)
    source.generate_points(3, seed=2)
    old = list(source.Points)
    mesh = m.triangulate(source)
    assert len(mesh.boundary_edges["outer"]) == 48
    assert len(mesh.boundary_edges["hole"]) == 40
    for border in (outer, hole):
        p = np.array([(v.x, v.y) for v in border.generate_points()])
        np.testing.assert_allclose(cKDTree(mesh.vertices).query(p)[0], 0, atol=1e-14)
    assert source.Points == old and hole.n_segments == -40
    with pytest.raises(ValueError, match="omit boundary"):
        m.triangulate([outer, hole], boundary=80)


def test_non_star_shaped_domain_with_hole():
    domain = g.with_holes(g.Polygon([(-2,-2), (2,-2), (2,2), (.3,2),
                                      (.3,0), (-.3,0), (-.3,2), (-2,2)]),
                          {"hole": g.Disk(radius=.2, center=(-1, 1))})
    mesh = m.triangulate(domain, boundary=160)
    cells = polygons(mesh.vertices[mesh.triangles])
    # Compare with the sampled geometry, since circular holes are chords.
    assert all(cell.centroid.x < -.3 or cell.centroid.x > .3 or cell.centroid.y < 0
               for cell in cells)
    assert len(mesh.boundary_edges["hole"]) > 0


def test_max_spacing_and_free_boundary():
    domain = g.Rectangle(width=2, height=2)
    coarse = m.triangulate(domain, boundary=16)
    fine = m.triangulate(domain, boundary=16, max_spacing=.15, fixed_boundary=False)
    assert len(fine.vertices) > 2*len(coarse.vertices)
    assert len(fine.exterior_edges) > 16
    assert set(fine.boundary_edges) == {"bottom", "right", "top", "left"}


@pytest.mark.parametrize("kwargs", [dict(growth=0), dict(growth=np.nan),
    dict(max_spacing=0), dict(min_angle=35), dict(min_angle=-1), dict(fixed_boundary=1)])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        m.triangulate(g.Disk(), **kwargs)


def test_repeated_border_labels_and_corner_membership():
    borders = [g.Border(lambda t: (t, 0), "wall", 0, 1)(12),
               g.Border(lambda t: (1, t), "right", 0, 1)(12),
               g.Border(lambda t: (1-t, 1), "wall", 0, 1)(12),
               g.Border(lambda t: (0, 1-t), "left", 0, 1)(12)]
    mesh = m.triangulate(borders)
    assert len(mesh.boundary_edges["wall"]) == 24
    cloud = g.staggered_clouds(mesh).vertices
    assert len(np.intersect1d(cloud.boundary["wall"], cloud.boundary["right"])) == 2
