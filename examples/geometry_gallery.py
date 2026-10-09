"""Rebuild the geometry gallery from actual generated, labeled point clouds.

Run: python -m examples.geometry_gallery
Requires rbflab[examples]. Images are illustrative clouds, not mesh-quality
or PDE-accuracy claims. No triangle connectivity is inferred.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rbflab import geometry, meshgen
from examples.poisson_perforated import run as poisson

BG, FG, MUTED = "#0c1828", "#eef6ff", "#a9bfd2"
COLORS = ["#67e8d5", "#ffb96d", "#c5adff", "#63bdff", "#f58ba8", "#e4ed91"]
OUT = Path(__file__).resolve().parents[1]/"docs/assets"


def domains():
    """Small reproducible recipes reused in the geometry documentation."""
    channel = geometry.with_holes(
        geometry.Rectangle(4., 1.8, labels=("bottom", "outlet", "top", "inlet")),
        {"cylinder": geometry.Disk(.32, center=(-.8, 0)),
         "ellipse": geometry.Ellipse(.4, .2, center=(.7, 0), angle=.45)})
    flower = geometry.with_holes(geometry.Flower(petals=6, amplitude=.16),
                                {"hole": geometry.Disk(.38)})
    elbow = geometry.Polygon([(0,0),(2,0),(2,.7),(.8,.7),(.8,2),(0,2)],
        labels=("base", "exit", "inner_floor", "inner_wall", "entry", "outer_wall"))
    return channel, flower, elbow


def planar(ax, domain, cloud):
    ax.set_facecolor(BG)
    ax.scatter(*cloud.interior.T, s=5, color="#7298bc", alpha=.8, linewidths=0)
    for i, (label, indices) in enumerate(cloud.boundary.items()):
        color = COLORS[i % len(COLORS)]
        boundary = domain.boundaries[label]
        ts = np.linspace(*boundary.interval, 300)
        coords = np.array([boundary.curve(t) for t in ts])
        ax.plot(*coords.T, color=color, lw=1, alpha=.6)
        ax.scatter(*cloud.points[indices].T, s=10, color=color, linewidths=0)
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(.07)


def caption(fig, x, y, title, subtitle):
    fig.text(x,y,title,color=FG,fontsize=14,fontweight="bold")
    fig.text(x,y-.032,subtitle,color=MUTED,fontsize=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    fig=plt.figure(figsize=(13,8.5),facecolor=BG)
    fig.text(.045,.95,"GEOMETRY → NODES → EQUATIONS",color="#67e8d5",fontsize=12,fontweight="bold")
    fig.text(.045,.897,"Start with the shape of your problem.",color=FG,fontsize=27,fontweight="bold")
    report={}
    boxes=[(.035,.485,.45,.31),(.53,.47,.43,.34),(.045,.065,.43,.29)]
    names=[("Obstacles & named walls","One domain · six boundary groups"),
           ("Curved contours & holes","Analytic curves · outward normals"),
           ("Concave polygons","Labeled edges · deterministic corners")]
    for i,(domain,box,(title,sub)) in enumerate(zip(domains(),boxes,names)):
        cloud=meshgen.generate(domain,interior=380,boundary=180,seed=42)
        ax=fig.add_axes(box);planar(ax,domain,cloud)
        if i==1:
            for label,ind in cloud.boundary.items():
                p=cloud.points[ind][::8];n=cloud.normals[label][::8]
                ax.quiver(*p.T,*n.T,color=FG,scale=18,width=.004,alpha=.8)
        caption(fig,box[0]+.01,box[1]-.006,title,sub)
        report[title]={"nodes":len(cloud.points),"labels":list(cloud.boundary),"seed":42}
    sphere=geometry.Sphere(radius=1.,boundary_label="surface")
    cloud=meshgen.generate(sphere,interior=550,boundary=500,seed=42)
    ax=fig.add_axes([.55,.052,.4,.34],projection="3d",facecolor=BG)
    p=cloud.interior; p=p[p[:,1]>-.1]
    ax.scatter(*p.T,s=8,c=p[:,2],cmap="cool",alpha=.85,depthshade=False,edgecolors="none")
    p=cloud.points[cloud.boundary["surface"]];p=p[p[:,1]>.05]
    ax.scatter(*p.T,s=12,color=COLORS[0],alpha=.95,depthshade=False,edgecolors="none")
    ax.set_box_aspect((1,1,1),zoom=1.3);ax.view_init(elev=23,azim=-65);ax.axis("off")
    caption(fig,.55,.059,"Volumes in three dimensions","Sphere cutaway · surface and interior nodes")
    report["sphere"]={"nodes":len(cloud.points),"seed":42,"display":"front removed to reveal interior"}
    fig.savefig(OUT/"geometry_gallery.png",dpi=165,facecolor=BG);plt.close(fig)
    domain,cloud,solution,metrics=poisson()
    fig=plt.figure(figsize=(12,4.8),facecolor=BG)
    ax=fig.add_axes([.035,.15,.44,.72]);planar(ax,domain,cloud)
    caption(fig,.05,.9,"01 / Generate the cloud",f"{len(cloud.points)} nodes · three named boundaries")
    ax=fig.add_axes([.53,.15,.44,.72],facecolor=BG)
    grid=np.linspace(-1.7,1.7,210);ys=np.linspace(-1.1,1.1,140)
    X,Y=np.meshgrid(grid,ys);q=np.c_[X.ravel(),Y.ravel()]
    inside=domain.contains_points(q);v=np.full(len(q),np.nan)
    v[inside]=solution.evaluate(q[inside])
    im=ax.contourf(X,Y,v.reshape(X.shape),levels=np.linspace(-1,1,35),cmap="coolwarm")
    ax.contour(X,Y,v.reshape(X.shape),levels=12,colors="#172e49",linewidths=.35,alpha=.55)
    for boundary in domain.boundaries.values():
        coords=np.array([boundary.curve(t) for t in np.linspace(*boundary.interval,300)])
        ax.plot(*coords.T,color=FG,lw=1)
    ax.set_aspect("equal");ax.axis("off")
    caption(fig,.55,.9,"02 / Solve the equation","Symbolic Poisson · local RBF-FD weights")
    cax=fig.add_axes([.65,.13,.24,.023]);bar=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[-1,0,1])
    bar.ax.tick_params(colors=MUTED,length=0,labelsize=9);bar.outline.set_visible(False)
    fig.text(.05,.07,"Computed solution, masked at every hole. Colors show u; lines are solution contours.",color=MUTED,fontsize=10)
    fig.savefig(OUT/"perforated_poisson.png",dpi=170,facecolor=BG);plt.close(fig)
    report["poisson"]=metrics
    (OUT/"geometry_gallery.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
