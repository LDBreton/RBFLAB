"""Topology, normals and labels for reusable planar composition."""
import numpy as np
import pytest
from rbflab import geometry as g, meshgen


def test_polygon_orientation_concavity_and_closure():
    vertices=np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]])
    for points in (vertices,vertices[::-1],np.vstack((vertices,vertices[0]))):
        domain=g.Polygon(points)
        cloud=meshgen.generate(domain,interior=80,boundary=48,seed=7)
        assert len(cloud.points)==128 and len(cloud.interior_indices)==80
        assert domain.contains_points(cloud.points).all()
        assert not domain.contains_points([[1.5,1.5]])[0]
        for label,ids in cloud.boundary.items():
            p=cloud.points[ids][1:]; n=cloud.normals[label][1:]
            assert domain.contains_points(p-1e-6*n).all()
            assert not domain.contains_points(p+1e-6*n).any()


def test_rotated_rectangle_labels_normals_and_exact_counts():
    domain=g.Rectangle(3,1,center=(2,-1),angle=.4,labels=('a','b','c','d'))
    counts={'a':7,'b':5,'c':9,'d':4}
    cloud=meshgen.generate(domain,interior=50,boundary=counts)
    assert {k:len(v) for k,v in cloud.boundary.items()}==counts
    for label,ids in cloud.boundary.items():
        n=cloud.normals[label]
        np.testing.assert_allclose(np.linalg.norm(n,axis=1),1)
        assert np.all(np.sum((cloud.points[ids]-[2,-1])*n,axis=1)>0)


def test_named_holes_and_original_domains_unchanged():
    outer=g.Rectangle(4,2)
    hole=g.Disk(.3,center=(-.7,0))
    original=list(hole.boundaries)
    domain=g.with_holes(outer,{'obstacle':hole,'slot':g.Ellipse(.35,.2,center=(.7,0))})
    cloud=meshgen.generate(domain,interior=100,boundary=90,seed=3)
    assert len(cloud.points)==190 and len(cloud.interior_indices)==100
    assert set(domain.boundaries)=={'bottom','right','top','left','obstacle','slot'}
    assert not hole.contains_points(cloud.interior).any()
    ids=cloud.boundary['obstacle']
    assert np.all(np.sum((cloud.points[ids]-[-.7,0])*cloud.normals['obstacle'],axis=1)<0)
    assert list(hole.boundaries)==original and len(outer.contours)==1
    assert not domain.contains_points([[-.7,0],[.7,0]]).any()
    more=g.with_holes(domain,{'third':g.Disk(.1,center=(0,.6))})
    assert len(more.contours)==4 and len(domain.contours)==3


def test_polygon_hole_labels_and_validation():
    outer=g.Rectangle(4,3)
    domain=g.with_holes(outer,{'cut':g.Rectangle(.4,.4)})
    assert 'cut/bottom' in domain.boundaries
    with pytest.raises(ValueError,match='strictly inside'):
        g.with_holes(outer,{'outside':g.Disk(.5,center=(2,0))})
    with pytest.raises(ValueError,match='strictly inside'):
        g.with_holes(outer,{'touch':g.Rectangle(4,1)})
    with pytest.raises(ValueError,match='touch or overlap'):
        g.with_holes(outer,{'a':g.Disk(.4),'b':g.Disk(.4,center=(.5,0))})
    with pytest.raises(ValueError,match='unique label'):
        g.with_holes(outer,{'bottom':g.Disk(.4)})
    with pytest.raises(ValueError,match='simply connected'):
        g.with_holes(outer,{'ring':g.Annulus()})
    with pytest.raises(ValueError):g.Polygon([[0,0],[1,1],[0,1],[1,0]])
    with pytest.raises(ValueError):g.Polygon([[0,0],[1,0],[2,0]])
    with pytest.raises(ValueError):g.Polygon([[0,0],[1,0],[1,1]],labels=['x','x','z'])
    with pytest.raises(ValueError):g.Rectangle(width=-1)


def test_perforated_poisson_independent_clouds():
    from examples.poisson_perforated import run
    for seed in (42, 17):
        _, _, _, metrics = run(seed=seed)
        assert metrics["nodal_max"] < 2e-4
        assert metrics["off_node_max"] < 2e-4
