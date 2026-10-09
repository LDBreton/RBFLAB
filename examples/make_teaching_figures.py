"""Regenerate the tutorial illustrations from the documented numerical recipes."""
from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rbflab as rbf
from rbflab import geometry, meshgen

OUT = Path(__file__).resolve().parents[1] / "docs/assets"
BLUE, TEAL, ORANGE = "#254e78", "#00897b", "#d87930"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": BLUE, "text.color": BLUE,
                         "axes.edgecolor": "#c5cdd6", "figure.facecolor": "white"})
    rng = np.random.default_rng(7)
    X = rng.uniform(-1., 1., (64, 2))
    X_card = X.copy()
    d = np.sin(X[:, 0])+np.cos(X[:, 1])
    fit = rbf.interpolate(rbf.PHS(5), X, d, polynomial_degree=2)
    gx, gy = np.meshgrid(np.linspace(-.7, .7, 85), np.linspace(-.7, .7, 85))
    Q = np.column_stack((gx.ravel(), gy.ravel()))
    values = fit.evaluate(Q).reshape(gx.shape)
    dx = fit.evaluate(Q, rbf.Derivative(0)).reshape(gx.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    im = axes[0].scatter(*X.T, c=d, cmap="viridis", s=33, edgecolor="white", linewidth=.5)
    fig.colorbar(im, ax=axes[0], shrink=.75, label="sample value")
    for ax, field, label in zip(axes[1:], (values, dx), ("s(x, y)", "partial s / partial x")):
        im = ax.contourf(gx, gy, field, levels=22, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=.75, label=label)
    for ax, title in zip(axes, ("01  Supplied samples", "02  PHS reconstruction", "03  Differentiate the expansion")):
        style(ax, title)
    fig.savefig(OUT/"teaching_interpolation.png", dpi=170); plt.close(fig)

    s=sp.Symbol("s",nonnegative=True);c=sp.Symbol("c",positive=True);beta=sp.Symbol("beta",nonnegative=True)
    family=rbf.Kernel((1+c*s)**sp.Rational(-1,2)+beta*sp.exp(-c*s),s,(c,beta))
    k=family(c="2",beta="0.1");r=np.linspace(0,2.5,240);z=np.column_stack((r,np.zeros_like(r)))
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),constrained_layout=True)
    for ax,alpha,title,color in zip(axes,((0,0),(1,0)),("01  IMQ + Gaussian", "02  Cartesian derivative at (r, 0)"),(BLUE,TEAL)):
        val=k.derivative(z,alpha);ax.plot(r,val,color=color,lw=2.5)
        ax.scatter([0],[val[0]],color=ORANGE,zorder=3,s=35)
        ax.axhline(0,color="#c5cdd6",lw=.8)
        ax.set(title=title,xlabel="r",ylabel="kernel value" if alpha==(0,0) else "partial / partial z1")
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"teaching_kernel.png",dpi=170);plt.close(fig)

    cloud=rbf.geometry.unit_box_grid(5);model=rbf.SymbolicScalar(2);u=model.field;x,y=model.coordinates
    exact=sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem=model.stationary(sp.Eq(-model.laplacian(u),2*sp.pi**2*exact),boundary=[model.bc("boundary",sp.Eq(u,0))])
    system=rbf.LHI(rbf.PHS(5),20,polynomial_degree=2).assemble(problem,cloud)
    patch=system.stencils[0];X=cloud.points
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    axes[0].scatter(*cloud.interior.T,c=BLUE,s=45,label="16 PDE rows")
    axes[0].scatter(*cloud.points[cloud.boundary_indices].T,c=ORANGE,marker="s",s=40,label="20 Dirichlet rows")
    axes[1].scatter(*X.T,c="#dce1e7",s=25)
    axes[1].scatter(*X[patch.solution_indices].T,c=BLUE,s=45,label="12 solution values")
    axes[1].scatter(*X[patch.boundary_indices].T,c=ORANGE,marker="s",s=45,label="8 boundary data")
    axes[1].scatter(*X[patch.pde_indices].T,facecolors="none",edgecolors=TEAL,marker="^",s=140,lw=1.4,label="11 PDE data")
    axes[1].scatter(*X[patch.center],c=ORANGE,marker="*",s=160,edgecolors="white",zorder=4,label="target")
    for ax,title in zip(axes,("01  Global collocation centers", "02  First LHI neighborhood")):
        style(ax,title);ax.set(xlim=(-.08,1.08),ylim=(-.08,1.08));ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1,1),frameon=False)
    fig.savefig(OUT/"teaching_centers.png",dpi=170);plt.close(fig)

    domain=geometry.Ellipse(a=1.2,b=.8,labels=("wall",))
    cloud=meshgen.generate(domain,interior=140,boundary=60,seed=42)
    ops=rbf.RBFFD(rbf.PHS(5),35,polynomial_degree=3,stencil_policy=rbf.StencilPolicy(scaling="local"),local_backend=rbf.PythonBackend(compute_condition=False)).operators(source=cloud.points,operators={"lap":rbf.Laplacian()})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    axes[0].scatter(*cloud.interior.T,c=BLUE,s=15,label="140 interior values")
    axes[0].scatter(*cloud.points[cloud.boundary_indices].T,c=ORANGE,s=20,label="60 prescribed wall values")
    style(axes[0],"01  Locations and equation roles");axes[0].legend(fontsize=8,loc="lower center",frameon=False)
    axes[1].spy(ops.lap.matrix,markersize=1,color=TEAL)
    axes[1].set(title="02  Sparse Laplacian: 35 entries per row",xlabel="source value index",ylabel="target index")
    axes[1].xaxis.tick_bottom()
    fig.savefig(OUT/"teaching_operators.png",dpi=170);plt.close(fig)
    from examples.tutorials.heat_matrices import rbf_fd_laplacian, march
    cloud=rbf.geometry.unit_box_grid(6);matrix,_=rbf_fd_laplacian(cloud)
    states,_=march(matrix,cloud,dt=.01,steps=5,scheme="bdf2")
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    axes[0].scatter(*cloud.interior.T,c=BLUE,s=28,label="interior")
    axes[0].scatter(*cloud.points[cloud.boundary_indices].T,c=ORANGE,s=28,label="Dirichlet")
    axes[0].legend(frameon=False,fontsize=8)
    for ax,field in zip(axes[1:],(states[0],states[-1])):
        im=ax.tricontourf(*cloud.points.T,field,levels=np.linspace(0,1,21),cmap="inferno")
        fig.colorbar(im,ax=ax,shrink=.75,label="temperature")
    for ax,title in zip(axes,("01  Same 49 nodes in both routes", "02  Initial values", "03  BDF2 at t = 0.05")):
        style(ax,title)
    fig.savefig(OUT/"teaching_heat.png",dpi=170);plt.close(fig)
    # Single-panel previews remain legible on narrow learning-path cards.
    fig,ax=plt.subplots(figsize=(4.8,3.6),constrained_layout=True)
    ax.contourf(gx,gy,values,levels=20,cmap="viridis")
    ax.scatter(*X_card.T,s=22,c="white",edgecolors=BLUE,linewidths=.7)
    ax.set(xlim=(-.75,.75),ylim=(-.75,.75),aspect="equal",xlabel="x",ylabel="y")
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"teaching_data_card.png",dpi=160);plt.close(fig)

    domain=geometry.Ellipse(a=1.3,b=.8,labels=("wall",))
    cloud=meshgen.generate(domain,interior=200,boundary=80,seed=42)
    m=rbf.SymbolicScalar(2);u=m.field;x,y=m.coordinates
    problem=m.stationary(sp.Eq(-m.laplacian(u),2*sp.sin(x)*sp.cos(y)),boundary=[m.bc("wall",sp.Eq(u,sp.sin(x)*sp.cos(y)))])
    sol=problem.solve(cloud,rbf.RBFFD(rbf.PHS(5),35,polynomial_degree=3,stencil_policy=rbf.StencilPolicy(scaling="local")))
    fig,ax=plt.subplots(figsize=(4.8,3.6),constrained_layout=True)
    ax.tricontourf(*cloud.points.T,sol.evaluate(cloud.points),levels=22,cmap="inferno")
    ax.scatter(*cloud.points[cloud.boundary_indices].T,c=ORANGE,s=8)
    ax.set(aspect="equal",xlabel="x",ylabel="y")
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"teaching_pde_card.png",dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(4.8,3.6),constrained_layout=True)
    ax.spy(ops.lap.matrix,markersize=1.5,color=TEAL)
    ax.set(xlabel="source values",ylabel="target equations");ax.xaxis.tick_bottom()
    fig.savefig(OUT/"teaching_algorithm_card.png",dpi=160);plt.close(fig)
    print("Wrote five teaching figures and three card previews to",OUT)


if __name__=="__main__":
    main()
