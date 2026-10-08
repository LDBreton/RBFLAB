"""Regenerate curved-domain documentation assets from computed solutions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from rbflab import viz
from examples import annulus,ellipse_boundary,heat_flower,ball_poisson,stokes_annulus


def main(backend="python",output=Path("docs/assets")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    a,ac,asol,am=annulus.run(backend=backend)
    e,ec,esol,em=ellipse_boundary.run(backend=backend)
    h,hc,trajectory,hm=heat_flower.run(backend=backend)
    b,bc,bsol,bm=ball_poisson.run(backend=backend)
    s,sc,ssol,sm=stokes_annulus.run()
    # A consistent gallery: generated points, scalar solution, flow, 3D slice.
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    fig.patch.set_facecolor("#0b1220")
    fig.suptitle("RBFLAB  /  from geometry to equations",color="#e8f1ff",fontsize=23,weight="bold")
    stencil=cKDTree(hc.points).query([.75,.05],k=35)[1]
    viz.plot_cloud(hc,ax=axes[0,0],title="01 / A cloud shaped for your problem",stencil=stencil)
    viz.plot_scalar(asol,domain=a,resolution=160,ax=axes[0,1],title="02 / Harmonic field around a hole")
    viz.plot_velocity(ssol,domain=s,resolution=150,ax=axes[1,0],title="03 / Divergence-free Stokes flow")
    viz.plot_slice(bsol,b,resolution=150,ax=axes[1,1],title="04 / A section through a 3D solution")
    fig.savefig(output/'curved_domains.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    viz.plot_cloud(ac,ax=axes[0],normals=True,title="Boundary normals / including the hole")
    truth=lambda p:np.log(np.linalg.norm(p,axis=1)/.4)/np.log(2.5)
    error=SimpleNamespace(evaluate=lambda p:np.abs(asol.evaluate(p)-truth(p)))
    viz.plot_scalar(error,domain=a,resolution=160,ax=axes[1],title="Absolute error / independent display points",cmap="inferno")
    fig.savefig(output/'annulus_error.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    viz.plot_cloud(ec,ax=axes[0],normals=True,title="Three boundary conditions / one ellipse")
    viz.plot_scalar(esol,domain=e,resolution=160,ax=axes[1],title="Computed reaction-diffusion solution")
    fig.savefig(output/'ellipse_boundary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    viz.plot_cloud(sc,ax=axes[0],title="Rotating inner wall / fixed outer wall")
    viz.plot_velocity(ssol,domain=s,resolution=150,ax=axes[1],title="Stokes Couette flow / computed velocity")
    fig.savefig(output/'annular_stokes.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(11,4.7),layout="constrained")
    viz.plot_cloud(bc,ax=fig.add_subplot(121,projection="3d"),title="A 3D cloud / ball")
    viz.plot_slice(bsol,b,ax=fig.add_subplot(122),resolution=150,title="Numerical solution / z = 0")
    fig.savefig(output/'ball_poisson.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained")
    states=[trajectory.initial,trajectory.states[9],trajectory.final]
    for ax,state in zip(axes,states):
        viz.plot_scalar(state,domain=h,resolution=150,ax=ax,vmin=0,vmax=1.25,
                        title=f"Temperature / t = {float(getattr(state,'time',0)):.2f}")
    fig.savefig(output/'flower_heat.png',dpi=160);plt.close(fig)
    viz.animate_scalar(trajectory,output/'flower_heat.gif',domain=h,resolution=100,title="Manufactured heat diffusion / flower")
    (output/'curved_gallery.json').write_text(json.dumps([am,em,hm,bm,sm],indent=2)+"\n",encoding="utf-8")
    print("Gallery saved to",output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=("python","cpp","torch"),default="python")
    p.add_argument("--output",type=Path,default=Path("docs/assets"))
    main(**vars(p.parse_args()))
