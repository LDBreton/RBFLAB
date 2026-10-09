"""Reproduce the mesh-generation manual's geometry-only illustrations.

python -m examples.mesh_construction [--gmsh]
Every image is generated from RBFLAB coordinates/labels. No PDE solve is needed.
"""
import argparse
from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rbflab import geometry as g, meshgen as m

OUT=Path(__file__).resolve().parents[1]/"docs/assets/meshes"
BG, INK, MUTED="#0c1828", "#edf5ff", "#a9bfd2"
COLORS=["#67e8d5", "#ffb96d", "#c5adff", "#63bdff", "#f58ba8", "#e4ed91", "#ddd"]


def channel_borders():
    """A bulged channel with a round obstacle; all labels are geometric groups."""
    t=sp.symbols("t", real=True)
    return [g.Border((3*t,0),"bottom",0,1)(60),
            g.Border((3,t),"outlet",0,1)(24),
            g.Border((3*(1-t),1+sp.sin(sp.pi*t)/4),"top",0,1)(60),
            g.Border((0,1-t),"inlet",0,1)(24),
            g.Border((1+sp.cos(t)/4, sp.Rational(1,2)+sp.sin(t)/4),
                     "obstacle",0,2*sp.pi)(-40)]


def channel_contours():
    """The same physical channel using explicit outer/hole contours."""
    t=sp.symbols("t", real=True)
    outer=[g.ParametricBoundary((3*t,0),label="bottom",interval=(0,1)),
           g.ParametricBoundary((3,t),label="outlet",interval=(0,1)),
           g.ParametricBoundary((3*(1-t),1+sp.sin(sp.pi*t)/4),label="top",interval=(0,1)),
           g.ParametricBoundary((0,1-t),label="inlet",interval=(0,1))]
    hole=g.ParametricBoundary((1+sp.cos(t)/4,sp.Rational(1,2)+sp.sin(t)/4),label="obstacle")
    return g.ParametricDomain(outer, holes=[hole])


def torus_surface():
    """A 3D surface; it does not enclose sampled volume points."""
    R, r=1., .35
    return m.ParametricSurface3D(
        lambda u,v: ((R+r*np.cos(v))*np.cos(u),(R+r*np.cos(v))*np.sin(u),r*np.sin(v)),
        (0,2*np.pi),(0,2*np.pi),label="torus",periodic_u=True,periodic_v=True,
        derivatives=(lambda u,v: (-(R+r*np.cos(v))*np.sin(u),(R+r*np.cos(v))*np.cos(u),0*u),
                     lambda u,v: (-r*np.sin(v)*np.cos(u),-r*np.sin(v)*np.sin(u),r*np.cos(v))))


def staggered_example():
    """Two triangle vertex/edge point sets with explicit shared connectivity."""
    mesh=g.TriangleMesh2D([[0,0],[2,0],[1.6,1.2],[-.2,1]],[[0,1,2],[0,2,3]],
        boundary_edges={"bottom":[[0,1]],"right":[[1,2]],"top":[[2,3]],"left":[[3,0]]})
    return mesh,g.staggered_clouds(mesh)


def panel(ax,cloud,title,normals=False):
    ax.set_facecolor(BG)
    ax.scatter(*cloud.interior.T,s=9,c="#7199bd",alpha=.8,lw=0,label="interior")
    for i,(label,indices) in enumerate(cloud.boundary.items()):
        color=COLORS[i%len(COLORS)];p=cloud.points[indices]
        ax.scatter(*p.T,s=16,c=color,lw=0,label=label)
        if normals:
            stride=max(1,len(p)//10);n=cloud.normals[label][::stride]
            ax.quiver(*p[::stride].T,*n.T,color=color,scale=18,width=.0035)
    for label,indices in cloud.interfaces.items():
        ax.scatter(*cloud.points[indices].T,s=14,c="#ef94e1",lw=0,label=label)
    ax.set_title(title,loc="left",color=INK,fontsize=13,pad=16,fontweight="bold")
    ax.set_aspect("equal");ax.margins(.12);ax.tick_params(colors=MUTED,labelsize=9)
    for spine in ax.spines.values():spine.set_color("#334c64")
    ax.set_xlabel("x",color=MUTED);ax.set_ylabel("y",color=MUTED)
    legend=ax.legend(loc="upper center",bbox_to_anchor=(.5,-.24),ncol=4,
                     facecolor=BG,edgecolor="none",fontsize=9,labelcolor=INK)


def save(fig,name):
    fig.savefig(OUT/name,dpi=155,facecolor=BG,bbox_inches="tight")
    plt.close(fig)


def plane_panels(name,items,cols=2,normals=False):
    rows=(len(items)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(6.2*cols,5.1*rows),squeeze=False,facecolor=BG)
    for ax,(title,cloud) in zip(axes.flat,items):panel(ax,cloud,title,normals)
    for ax in list(axes.flat)[len(items):]:ax.axis("off")
    fig.subplots_adjust(wspace=.24,hspace=.65,bottom=.23)
    save(fig,name)


def main(include_gmsh=False):
    OUT.mkdir(parents=True,exist_ok=True)
    primitives=[("Disk(radius=.8)",g.Disk(.8,label="wall")),("Ellipse(a=1.5, b=.7, angle=.35)",g.Ellipse(1.5,.7,angle=.35)),
                ("Annulus(inner=.35, outer=1)",g.Annulus(.35,1)),("Flower(amplitude=.2, petals=6)",g.Flower(amplitude=.2,petals=6))]
    plane_panels("primitives.png",[(name,m.generate(d,interior=200,boundary=96,seed=42)) for name,d in primitives])
    polygon=g.Polygon([(0,0),(2,0),(2,.7),(.8,.7),(.8,2),(0,2)],
                      labels=("base","exit","inner_floor","inner_wall","entry","outer_wall"))
    plane_panels("polygons.png",[("Rectangle(width=2, height=1, angle=.4)",m.generate(g.Rectangle(2,1,angle=.4),interior=200,boundary=96)),
                                 ("Concave polygon / one label per edge",m.generate(polygon,interior=200,boundary=120))])
    plate=g.with_holes(g.Rectangle(4,2),{"circle":g.Disk(.3,center=(-.8,0)),
                                        "slot":g.Ellipse(.45,.2,center=(.7,0),angle=.4)})
    plane_panels("holes.png",[("One material domain / two named holes",m.generate(plate,interior=400,boundary=200))],cols=1)
    plane_panels("channels.png",[("Signed borders / parameter spacing",m.generate(channel_borders(),interior=350,seed=42)),
        ("Explicit contours / arc-length spacing",m.generate(channel_contours(),interior=350,
         boundary={"bottom":60,"outlet":24,"top":60,"inlet":24,"obstacle":40},seed=42))])
    t=sp.symbols("t",real=True)
    half=[g.Border((sp.cos(t),sp.sin(t)),"wall",0,sp.pi)(80),g.Border((-1+2*t,0),"base",0,1)(40)]
    plane_panels("half-disk.png",[("Two arcs / one closed contour",m.generate(half,interior=200))],cols=1)
    labels=g.with_holes(g.Ellipse(1.5,.9,labels=("arc_0","arc_1","arc_2","arc_3")),{"hole":g.Disk(.3)})
    plane_panels("normals.png",[("Arc labels and outward normals",m.generate(labels,interior=220,
                 boundary={"arc_0":30,"arc_1":30,"arc_2":30,"arc_3":30,"hole":36}))],cols=1,normals=True)
    plane_panels("sampling.png",[(method,m.generate(g.Disk(),interior=180,boundary=60,method=method,seed=42))
                                  for method in ("random","halton","sobol")],cols=3)
    plane_panels("annulus.png",[("Annulus / outward from the material",m.generate(g.Annulus(.35,1),
        interior=200,boundary={"inner":40,"outer":80},seed=42))],cols=1,normals=True)
    ring=[g.Border((sp.cos(t),sp.sin(t)),"outer",0,2*sp.pi)(96),
          g.Border((sp.Rational(2,5)*sp.cos(t),sp.Rational(2,5)*sp.sin(t)),"hole",0,2*sp.pi)(-48)]
    plane_panels("border-ring.png",[("Oriented circular contours",m.generate(ring,interior=240,seed=42))],cols=1,normals=True)
    ellipse_arcs=g.Ellipse(1.5,.9,labels=("arc_0","arc_1","arc_2","arc_3"))
    plane_panels("arc-labels.png",[("Split an ellipse into four named arcs",m.generate(ellipse_arcs,interior=220,
        boundary={"arc_0":30,"arc_1":30,"arc_2":30,"arc_3":30},seed=42))],cols=1)
    curve_sources=[g.ParametricBoundary((2*sp.cos(t),sp.sin(t)),label="wall"),
        g.ParametricBoundary(lambda t:(2*np.cos(t),np.sin(t)),label="wall",tangent=lambda t:(-2*np.sin(t),np.cos(t))),
        g.ParametricBoundary(lambda t:(2*np.cos(t),np.sin(t)),label="wall")]
    plane_panels("curve-sources.png",[(c.derivative_source.replace("_"," "),m.generate(g.ParametricDomain(c),
        interior=200,boundary=96,seed=42)) for c in curve_sources],cols=3,normals=True)
    regions=[("Sphere / radius 1",g.Sphere(boundary_label="wall")),("Box / labeled faces",g.Box()),
             ("Cylinder / side and two caps",g.Cylinder(radius=.7,height=2))]
    fig=plt.figure(figsize=(15,5.5),facecolor=BG)
    for i,(name,region) in enumerate(regions):
        c=m.generate(region,interior=200,boundary=300,seed=42)
        ax=fig.add_subplot(1,3,i+1,projection="3d",facecolor=BG)
        p=c.interior;ax.scatter(*p.T,s=4,c="#789cbf",alpha=.4,depthshade=False)
        for j,(label,ids) in enumerate(c.boundary.items()):
            ax.scatter(*c.points[ids].T,s=9,c=COLORS[j%len(COLORS)],alpha=.65,depthshade=False,label=label)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.axis("off")
        ax.set_title(name,color=INK,fontsize=13)
        ax.legend(loc="lower center",fontsize=8,labelcolor=INK,facecolor=BG,edgecolor="none",ncol=3)
    save(fig,"volumes.png")
    # General implicit volume and parametric surface are separate constructions.
    ellipsoid=g.ImplicitRegion(lambda x,y,z:(x/1.5)**2+y*y+(z/.65)**2-1,
       [(-1.5,1.5),(-1,1),(-.65,.65)],gradient=lambda x,y,z:(2*x/2.25,2*y,2*z/.65**2),boundary_label="wall")
    c=m.generate(ellipsoid,interior=240,boundary=400,seed=42)
    fig=plt.figure(figsize=(7,5.5),facecolor=BG);ax=fig.add_subplot(projection="3d",facecolor=BG)
    p=c.points;ax.scatter(*p.T,c=p[:,2],cmap="viridis",s=9,alpha=.8)
    ax.set_box_aspect((1.5,1,.65));ax.axis("off");ax.set_title("Implicit ellipsoid / volume and surface nodes",color=INK)
    save(fig,"implicit.png")
    c=m.generate(torus_surface(),interior=0,boundary=900,seed=42)
    fig=plt.figure(figsize=(8,5.5),facecolor=BG);ax=fig.add_subplot(projection="3d",facecolor=BG)
    p=c.points;ax.scatter(*p.T,c=p[:,2],cmap="cool",s=8,alpha=.9,depthshade=False)
    ids=np.arange(0,len(p),55);n=c.normals["torus"][ids]
    ax.quiver(*p[ids].T,*n.T,length=.16,color="#fff",linewidth=.6)
    ax.set_box_aspect((1,1,.38));ax.view_init(32,-60);ax.axis("off")
    ax.set_title("Parametric torus / surface samples only",color=INK,fontsize=14)
    save(fig,"surface.png")
    mesh,layout=staggered_example()
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),facecolor=BG)
    for i,ax in enumerate(axes):
        ax.set_facecolor(BG);ax.triplot(*mesh.vertices.T,mesh.triangles,color="#536c81",lw=1)
        if i in (0,2):ax.scatter(*layout.vertices.points.T,c=COLORS[0],s=75,zorder=3,label="vertices")
        if i in (1,2):ax.scatter(*layout.edge_midpoints.points.T,c=COLORS[1],s=65,marker="s",zorder=3,label="edge midpoints")
        if i<2:
            p=layout.vertices.points if i==0 else layout.edge_midpoints.points
            for k,(x,y) in enumerate(p):ax.annotate(str(k),(x,y),xytext=(5,7),textcoords="offset points",color=INK)
        ax.set_aspect("equal");ax.margins(.14);ax.axis("off")
        ax.set_title(["4 vertex locations","5 unique edge midpoints","Two interlaced clouds"][i],color=INK,fontsize=13)
        ax.legend(loc="lower center",bbox_to_anchor=(.5,-.13),ncol=2,facecolor=BG,edgecolor="none",labelcolor=INK)
    save(fig,"staggered.png")
    if include_gmsh:
        from examples.perforated_plate import build_perforated_plate, plot_layout
        mesh,_=build_perforated_plate()
        plot_layout(mesh,g.staggered_clouds(mesh),str(OUT/"gmsh.png"))
    print("Geometry images written to",OUT)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--gmsh",action="store_true")
    main(parser.parse_args().gmsh)
