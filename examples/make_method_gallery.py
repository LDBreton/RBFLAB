"""Render compact website cards from numerical solutions, not stock imagery.

Run: python -m examples.make_method_gallery [--backend python|cpp]
The JSON beside the cards records the recipe and measured diagnostics.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
import rbflab as rbf
from examples.tutorials.heat_matrices import rbf_fd_laplacian, march, exact
from examples.stokes_annulus import run as stokes_run
from examples.navier_stokes_cavity import solve as cavity_solve

BG, INK, MUTED = "#0c1828", "#edf5ff", "#a9bfd2"
OUT = Path(__file__).resolve().parents[1]/"docs/assets"


def canvas():
    fig = plt.figure(figsize=(6.6, 5.4), facecolor=BG)
    return fig


def save(fig, name):
    fig.savefig(OUT/name, dpi=190, facecolor=BG)
    plt.close(fig)


def bar(fig, image, label, ticks):
    ax = fig.add_axes([.22, .09, .56, .026])
    colorbar = fig.colorbar(image, cax=ax, orientation="horizontal", ticks=ticks)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=5)
    colorbar.set_label(label, color=INK, fontsize=10, labelpad=4)


def heat():
    cloud = rbf.unit_box_grid(18)
    matrix, _ = rbf_fd_laplacian(cloud, 25)
    states, _ = march(matrix, cloud, dt=.001, steps=20, scheme="bdf2")
    error = float(np.max(abs(states[-1]-exact(cloud.points,.02))))
    assert error < .015
    fig = canvas()
    ax = fig.add_axes([-.02,.19,1.04,.85], projection="3d", facecolor=BG)
    x,y = cloud.points.T
    temp = states[-1].reshape(19,19)
    sample=np.linspace(0,1,85)
    temp=RectBivariateSpline(np.linspace(0,1,19),np.linspace(0,1,19),temp)(sample,sample)
    X,Y=np.meshgrid(sample,sample,indexing="ij")
    norm=Normalize(0,.7)
    surface=ax.plot_surface(X,Y,temp,cmap="plasma",norm=norm,
        linewidth=0,antialiased=True,rcount=85,ccount=85)
    ax.contour(X,Y,temp,zdir="z",offset=-.16,levels=10,cmap="plasma",norm=norm,linewidths=.9)
    ax.plot([0,1,1,0,0],[0,0,1,1,0],[-.16]*5,color=MUTED,lw=.8,alpha=.5)
    ax.set(zlim=(-.16,.75),xlim=(0,1),ylim=(0,1))
    ax.view_init(elev=31,azim=-56);ax.set_box_aspect((1,1,.63),zoom=1.12);ax.set_axis_off()
    bar(fig,surface,"temperature  /  initial peak",[0,.35,.7])
    save(fig,"method_heat.png")
    return dict(method="RBF-FD",nodes=len(cloud.points),kernel="PHS5",degree=2,
                stencil=25,time=.02,dt=.001,scheme="BE-started BDF2",nodal_max_error=error,
                rendering="cubic display interpolation of computed nodal temperatures; no analytic replacement")


def stokes():
    domain,cloud,solution,metrics=stokes_run()
    xy=np.linspace(-1,1,180);X,Y=np.meshgrid(xy,xy);points=np.c_[X.ravel(),Y.ravel()]
    inside=np.asarray(domain.contains_points(points),dtype=bool)
    velocity=np.full((len(points),2),np.nan)
    velocity[inside]=solution.velocity(points[inside])
    U=np.ma.masked_invalid(velocity[:,0].reshape(X.shape));V=np.ma.masked_invalid(velocity[:,1].reshape(X.shape))
    fig=canvas();ax=fig.add_axes([.14,.24,.72,.72],facecolor=BG)
    im=ax.pcolormesh(X,Y,np.hypot(U,V),cmap="viridis",vmin=0,vmax=.5,shading="gouraud",rasterized=True)
    ax.streamplot(xy,xy,U,V,color="#f0fff9",density=1.4,linewidth=.75,arrowsize=.8,minlength=.12)
    for radius in (.5,1):ax.add_patch(plt.Circle((0,0),radius,fill=False,color="#c7e6ec",lw=1.3))
    ax.text(0,0,"ROTATING\nINNER WALL",ha="center",va="center",color=MUTED,fontsize=10,linespacing=1.6)
    ax.set(xlim=(-1.04,1.04),ylim=(-1.04,1.04),aspect="equal");ax.axis("off")
    bar(fig,im,"speed",[0,.25,.5]);save(fig,"method_stokes.png")
    return dict(metrics,rendering="computed velocity sampled inside annulus; masked hole")


def cavity(backend):
    points,times,velocities,metrics=cavity_solve(cells=16,end=20,backend=backend,frames=2)
    xy=np.linspace(0,1,190);X,Y=np.meshgrid(xy,xy)
    field=LinearNDInterpolator(points,velocities[-1])(X,Y)
    U,V=np.ma.masked_invalid(field[:,:,0]),np.ma.masked_invalid(field[:,:,1])
    speed=np.hypot(U,V)
    fig=canvas();ax=fig.add_axes([.16,.24,.68,.72],facecolor=BG)
    im=ax.pcolormesh(X,Y,speed,cmap="magma",vmin=0,vmax=1,shading="gouraud",rasterized=True)
    ax.streamplot(xy,xy,U,V,color="#f1f2ff",density=1.55,linewidth=.75,arrowsize=.8,minlength=.12)
    ax.plot([0,0,1,1],[1,0,0,1],color=MUTED,lw=1.2)
    ax.annotate("",xy=(.91,1.03),xytext=(.09,1.03),arrowprops=dict(arrowstyle="->",lw=2,color="#ffd18a"))
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.07),aspect="equal");ax.axis("off")
    bar(fig,im,"speed  /  lid speed",[0,.5,1]);save(fig,"method_cavity.png")
    return dict(metrics,Re=100,cells=16,time=float(times[-1]),dt=.00125,backend=backend,
                rendering="linear interpolation of computed nodal velocities; no extrapolation",
                validation="illustrative coarse run; not a grid-converged benchmark")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("python","cpp"),default="python")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})
    report={"heat":heat(),"stokes":stokes(),"cavity":cavity(args.backend)}
    (OUT/"method_gallery.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
