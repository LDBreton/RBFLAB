"""Geometry-to-PDE invariants for the integrated array API."""
import subprocess
import sys
import numpy as np
import pytest
import rbflab as rbf
from rbflab import geometry as g, meshgen as m


@pytest.mark.parametrize("domain",[g.Disk(),g.Annulus(),g.Ellipse(),g.Flower(),g.Sphere()])
@pytest.mark.parametrize("method",["random","halton","sobol"])
def test_counts_labels_normals_reproducibility(domain,method):
    cloud=m.generate(domain,interior=70,boundary=40,method=method,seed=23)
    repeated=m.generate(domain,interior=70,boundary=40,method=method,seed=23)
    assert isinstance(cloud,rbf.PointCloud)
    assert len(cloud.points)==110 and len(cloud.interior_indices)==70
    assert domain.contains_points(cloud.points).all()
    np.testing.assert_array_equal(cloud.points,repeated.points)
    assert cloud.triangles.shape==(0,3)
    for label,ids in cloud.boundary.items():
        n=cloud.normals[label];p=cloud.points[ids]
        np.testing.assert_allclose(np.linalg.norm(n,axis=1),1)
        assert domain.contains_points(p-1e-4*n).all()
        assert not domain.contains_points(p+1e-4*n).any()


def test_parametric_hole_and_mixed_arcs():
    def circle(radius,label,sign):
        return g.ParametricBoundary(lambda t: radius*np.array([np.cos(t),sign*np.sin(t)]),
                 lambda t:radius*np.array([-np.sin(t),sign*np.cos(t)]),label)
    domain=g.ParametricDomain(circle(1,"wall",-1),[circle(.4,"hole",1)])
    cloud=m.generate(domain,interior=100,boundary={"wall":50,"hole":30})
    assert np.all(np.linalg.norm(cloud.interior,axis=1)>.4)
    ids=cloud.boundary["hole"]
    assert np.all(np.sum(cloud.normals["hole"]*cloud.points[ids],axis=1)<0)
    assert set(m.generate(g.Ellipse(labels=("a","b","c")),boundary=99).boundary)=={"a","b","c"}


def test_shared_staggered_contract():
    mesh=g.TriangleMesh2D([[0,0],[1,0],[1,1],[0,1]],[[0,1,2],[0,2,3]],
        interface_edges={"diagonal":[[0,2]]})
    layout=g.staggered_clouds(mesh)
    assert isinstance(layout.vertices,rbf.PointCloud)
    assert isinstance(layout.edge_midpoints,rbf.PointCloud)
    assert len(layout.edge_midpoints.interfaces["diagonal"])==1
    assert len(layout.edge_midpoints.interior_indices)==1


def test_quality_and_invalid_geometry():
    cloud=m.generate(g.Disk(),interior=80,boundary=32)
    q=m.quality(cloud)
    assert 0<q["spacing_ratio"]<=1 and q["worst_polynomial_ratio"]>0
    with pytest.raises(ValueError):g.Annulus(1,.4)
    with pytest.raises(ValueError):g.Flower(amplitude=1)
    with pytest.raises(ValueError):m.generate(g.Disk(),seed=-1)
    with pytest.raises(ValueError):m.generate(g.Disk(),boundary={"wrong":30})
    with pytest.raises(ValueError,match="exhausted"):
        m.generate(g.Disk(),interior=10,boundary_distance=3,max_candidates=100)
    with pytest.raises(ValueError):rbf.PointCloud([[0,0],[1,1]],{}, {},interfaces={"bad":[.5]})


def test_import_without_optional_dependencies():
    script="""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('matplotlib','gmsh','torch'):
            raise ImportError('optional dependency deliberately blocked: '+fullname)
sys.meta_path.insert(0,Block())
import rbflab
from rbflab import meshgen, geometry, viz
assert len(meshgen.generate(geometry.Disk(),interior=10,boundary=12).points)==22
assert len(meshgen.generate(geometry.Sphere(),interior=10,boundary=12).points)==22
"""
    subprocess.run([sys.executable,"-c",script],check=True)


def test_backend_source_discovery():
    from rbflab.native_paths import source_root
    assert (source_root()/"cpp"/"real.hpp").is_file()


def test_sampling_does_not_touch_global_random_state():
    import random
    for domain in (g.Disk(),g.Sphere()):
        before=random.getstate()
        m.generate(domain,interior=10,boundary=12,method="random",seed=None)
        assert random.getstate()==before


def test_pointcloud_rejects_fractional_connectivity():
    with pytest.raises(ValueError,match="boundary"):
        rbf.PointCloud([[0,0],[1,1]],{"wall":[.5]}, {})
    with pytest.raises(ValueError,match="integers"):
        rbf.PointCloud([[0,0],[1,0],[0,1]],{}, {},triangles=[[0,1,1.5]])


def test_legacy_adapter_retains_interfaces_and_regions():
    from types import SimpleNamespace
    from rbflab.geometry import MeshPoint
    mesh=SimpleNamespace(Points=[MeshPoint(.5,.5,"fluid",False)],
        Boundary_Points=[MeshPoint(0,0,"wall"),MeshPoint(1,0,"wall"),MeshPoint(.5,.2,"interface")])
    cloud=rbf.from_rbfmeshgen(mesh,boundary_labels=["wall"],interface_labels=["interface"])
    assert list(cloud.regions["fluid"])==[0]
    assert list(cloud.interfaces["interface"])==[3]
    assert 3 in cloud.interior_indices
