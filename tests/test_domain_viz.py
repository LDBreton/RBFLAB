"""Domain masks prevent evaluation or drawing in holes; plotting stays optional."""
import numpy as np
import pytest
from types import SimpleNamespace
from rbflab import geometry,meshgen,viz


def test_mask_before_evaluation():
    domain=geometry.Annulus()
    p=np.array([[0,0],[.6,0],[2,0]])
    def evaluate(points):
        assert domain.contains_points(points).all()
        return points[:,0]
    result=viz._sample_domain(evaluate,p,domain)
    np.testing.assert_array_equal(result.mask,[True,False,True])
    assert result[1]==.6


def test_plots_and_animation(tmp_path):
    mpl=pytest.importorskip("matplotlib");mpl.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    domain=geometry.Annulus()
    class Field:
        def __init__(self,time):self.time=time
        def evaluate(self,p):return (p[:,0]+2)*(1-self.time)
        def velocity(self,p):return np.column_stack((-p[:,1],p[:,0]))
    initial,final=Field(0),Field(.5)
    fig,ax=viz.plot_scalar(initial,domain=domain,resolution=30)
    assert np.ma.getmaskarray(ax.images[0].get_array()).any();plt.close(fig)
    fig,_=viz.plot_velocity(initial,domain=domain,resolution=30);plt.close(fig)
    cloud=meshgen.generate(domain,interior=30,boundary=30)
    fig,_=viz.plot_cloud(cloud,normals=True,stencil=[0,1]);plt.close(fig)
    ball=geometry.Sphere();fig,_=viz.plot_slice(initial,ball,resolution=30);plt.close(fig)
    fig,_=viz.plot_cloud(meshgen.generate(ball,interior=10,boundary=10));plt.close(fig)
    path=viz.animate_scalar(SimpleNamespace(initial=initial,states=[final],final=final),tmp_path/'masked.gif',domain=domain,resolution=30)
    assert Image.open(path).n_frames==2
