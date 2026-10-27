"""Build the six geometry figures used by the FreeFEM-style mesh manual.

Run from the repository root with ``python -m examples.make_mesh_manual_gallery``.
The planar triangulation figures require the optional ``rbflab[triangulation]``
extra. All samples are deterministic and generated from public RBFLAB geometry APIs.
"""
from pathlib import Path
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rbflab import geometry as g, meshgen as m

OUT = Path(__file__).resolve().parents[1] / "docs/assets/meshes"
BG, PANEL, INK, MUTED = "#f5f7fa", "#ffffff", "#172b3a", "#52697a"
COLORS = ["#087f8c", "#d97706", "#7856a5", "#2878b5", "#c44569", "#668c28"]


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", color=INK, fontsize=12, weight="bold", pad=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#d6dee5")
    ax.set_xlabel("x", color=MUTED, fontsize=9)
    ax.set_ylabel("y", color=MUTED, fontsize=9)


def draw_cloud(ax, cloud, domain=None, size=8):
    if domain is not None:
        for boundary in domain.boundaries.values():
            ts = np.linspace(*boundary.interval, 500)
            p = np.array([boundary.curve(t) for t in ts])
            ax.plot(*p.T, color="#263c4c", lw=1.2, zorder=2)
    ax.scatter(*cloud.interior.T, s=size, c="#567a96", alpha=.72,
               linewidths=0, zorder=3, label="interior nodes")
    for i, (label, ids) in enumerate(cloud.boundary.items()):
        ax.scatter(*cloud.points[ids].T, s=size*1.5, color=COLORS[i % len(COLORS)],
                   linewidths=0, zorder=4, label=str(label))


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def planar_atlas():
    shapes = [
        ("Disk", g.Disk(.82, center=(-.15, 0))),
        ("Rotated ellipse", g.Ellipse(1.08, .58, center=(0, 0), angle=.42)),
        ("Annulus", g.Annulus(.38, .95)),
        ("Six-petal flower", g.Flower(radius=.9, amplitude=.2, petals=6)),
        ("Two curved holes", g.with_holes(g.Ellipse(1.55, .95, angle=.12),
             {"round": g.Disk(.24, center=(-.62, 0)),
              "slot": g.Ellipse(.36, .17, center=(.58, .08), angle=-.35)})),
        ("Concave polygon", g.Polygon([(0,0),(1.8,0),(1.8,.65),(.75,.65),(.75,1.6),(0,1.6)])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1), facecolor=BG)
    for ax, (name, domain) in zip(axes.flat, shapes):
        cloud = m.generate(domain, interior=145, boundary=84, method="halton", seed=42)
        draw_cloud(ax, cloud, domain)
        style(ax, name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("One point-cloud workflow · six planar geometries", color=INK,
                 fontsize=18, weight="bold", y=.99)
    fig.text(.5, .01, "Deterministic Halton interiors · 145 interior + 84 boundary nodes per panel",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    save(fig, "manual-planar-atlas.png")


def border_workflow():
    t = sp.symbols("t", real=True)
    borders = [
        g.Border((2.8*t, 0), "base", 0, 1)(48),
        g.Border((2.8, t), "outlet", 0, 1)(20),
        g.Border((2.8*(1-t), 1 + .22*sp.sin(sp.pi*t)), "curved_wall", 0, 1)(48),
        g.Border((0, 1-t), "inlet", 0, 1)(20),
        g.Border((1.05 + .24*sp.cos(t), .5 + .24*sp.sin(t)), "obstacle", 0, 2*sp.pi)(-32),
    ]
    cloud = m.generate(borders, interior=235, method="halton", seed=11)
    fig = plt.figure(figsize=(12.2, 5.5), facecolor=BG)
    ax = fig.add_axes([.055, .16, .47, .72])
    draw_cloud(ax, cloud)
    style(ax, "Signed Border segments · labeled cloud")
    ax.set_xlim(-.18, 3.0); ax.set_ylim(-.15, 1.5)
    tx = fig.add_axes([.57, .15, .39, .72]); tx.axis("off")
    tx.text(0, .98, "A FreeFEM-style boundary recipe", transform=tx.transAxes,
            fontsize=14, weight="bold", color=INK, va="top")
    tx.text(0, .83, "Positive n  →  preserve parameter traversal\nNegative n  →  reverse traversal and loop orientation",
            transform=tx.transAxes, fontsize=10.5, color=MUTED, va="top", linespacing=1.6)
    tx.text(0, .62, "One label per physical boundary group", transform=tx.transAxes,
            fontsize=11, weight="bold", color=INK)
    for j, (name, color) in enumerate(zip(("base", "outlet", "curved_wall", "inlet", "obstacle"), COLORS)):
        y = .52 - j*.075
        tx.scatter([.02], [y], s=75, color=color, transform=tx.transAxes, clip_on=False)
        tx.text(.07, y, name, transform=tx.transAxes, color=INK, va="center", fontsize=10)
    tx.text(0, .04, "Normals are computed from oriented curves.\nThe obstacle normal points into the removed region.",
            transform=tx.transAxes, color=MUTED, fontsize=10, va="bottom", linespacing=1.55)
    fig.suptitle("Border(n) → geometry → labeled boundary nodes", color=INK,
                 fontsize=18, weight="bold", y=.97)
    save(fig, "manual-border-workflow.png")


def refinement():
    domain = g.with_holes(g.Ellipse(2.35, 1.35, angle=.08,
                                   labels=("arc_0", "arc_1", "arc_2", "arc_3")), {
        "upper_hole": g.Ellipse(.36, .2, center=(-.65, .36), angle=.3),
        "lower_hole": g.Disk(.25, center=(.55, -.35)),
    })
    base = {"arc_0": 25, "arc_1": 25, "arc_2": 25, "arc_3": 25,
            "upper_hole": 24, "lower_hole": 24}
    refined = {**base, "upper_hole": 96}
    meshes = [m.triangulate(domain, boundary=counts, growth=.26) for counts in (base, refined)]
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.4), facecolor=BG)
    for ax, mesh, title, count in zip(axes, meshes, ("Uniform boundary sampling", "Dense sampling at one hole"), (24, 96)):
        ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
                   color="#8fa2b1", lw=.42, zorder=1)
        ax.scatter(*mesh.vertices.T, s=4, color="#28465b", linewidths=0, zorder=2)
        # Highlight the refined boundary group, preserving actual label indices.
        for label, edges in mesh.boundary_edges.items():
            ids = np.unique(edges)
            ax.scatter(*mesh.vertices[ids].T, s=8,
                       color=COLORS[1] if label == "upper_hole" else COLORS[0],
                       linewidths=0, zorder=3)
        style(ax, f"{title}\n{len(mesh.vertices):,} vertices · {len(mesh.triangles):,} triangles · {count} upper-hole samples")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Boundary spacing controls local triangle size", color=INK,
                 fontsize=18, weight="bold", y=.99)
    fig.text(.5, .015, "Same analytic curved domain and growth setting; only upper_hole boundary count changes.",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .92))
    save(fig, "manual-refinement.png")


def distributions():
    domain = g.Annulus(.34, 1.)
    boundary = {"inner": 42, "outer": 84}
    items = [(method, m.generate(domain, interior=200, boundary=boundary,
                                  method=method, seed=42))
             for method in ("random", "halton", "sobol")]
    items.append(("Padua · clipped", m.generate(domain, method="padua", degree=22,
                                                  boundary=boundary)))
    items.append(("Mapped Padua", m.mapped_padua(domain, degree=22, clustering=2.0,
                                                  boundary=boundary)))
    mesh = m.triangulate(domain, boundary=boundary, growth=.28)
    items.append(("Triangle mesh vertices", g.staggered_clouds(mesh).vertices))
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.5), facecolor=BG)
    for ax, (name, cloud) in zip(axes.flat, items):
        if name == "Triangle mesh vertices":
            ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
                       color="#a7b4bd", lw=.42, zorder=1)
        draw_cloud(ax, cloud, domain=None, size=7)
        # Analytic circles keep the annular material boundary apparent.
        for radius in (.34, 1.):
            theta = np.linspace(0, 2*np.pi, 400)
            ax.plot(radius*np.cos(theta), radius*np.sin(theta), color="#263c4c", lw=1)
        style(ax, f"{name}\n{len(cloud.interior)} interior · {len(cloud.boundary_indices)} boundary")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Node placement methods on the same annulus", color=INK,
                 fontsize=18, weight="bold", y=.99)
    fig.text(.5, .012, "Random / Halton / Sobol: exactly 200 interior nodes. Padua and triangulation counts are method-defined; no equal-count claim.",
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, .05, 1, .93))
    save(fig, "manual-node-distributions.png")


def three_dimensional():
    sphere = g.Sphere(radius=1., boundary_label="surface")
    volume = m.generate(sphere, interior=800, boundary=800, method="halton", seed=42)
    torus = m.ParametricSurface3D(
        lambda u, v: ((1 + .35*np.cos(v))*np.cos(u),
                      (1 + .35*np.cos(v))*np.sin(u), .35*np.sin(v)),
        (0, 2*np.pi), (0, 2*np.pi), label="torus",
        periodic_u=True, periodic_v=True,
    )
    surface = m.generate(torus, interior=0, boundary=700, method="sobol", seed=7)
    fig = plt.figure(figsize=(11.8, 5.4), facecolor=BG)
    for i, (cloud, title) in enumerate(((volume, "Sphere volume · interior + surface"),
                                         (surface, "Parametric torus · surface nodes only"))):
        ax = fig.add_subplot(1, 2, i+1, projection="3d", facecolor=BG)
        if i == 0:
            p = cloud.interior
            # A front half-space cutaway reveals volume nodes without implying a boundary mesh.
            p = p[p[:, 1] > -.12]
            ax.scatter(*p.T, s=4, c=p[:, 2], cmap="viridis", alpha=.68,
                       depthshade=False, edgecolors="none")
            bp = cloud.points[cloud.boundary["surface"]]
            bp = bp[bp[:, 1] > -.12]
            ax.scatter(*bp.T, s=8, color=COLORS[0], alpha=.9, depthshade=False, edgecolors="none")
        else:
            p = cloud.points[cloud.boundary["torus"]]
            ax.scatter(*p.T, s=8, c=p[:, 2], cmap="plasma", alpha=.9,
                       depthshade=False, edgecolors="none")
        ax.set_box_aspect((1, 1, 1 if i == 0 else .4)); ax.view_init(elev=20, azim=-56)
        ax.set_axis_off(); ax.set_title(title, loc="left", color=INK, fontsize=12, weight="bold", pad=8)
    fig.suptitle("Three dimensions: volume clouds and surface clouds", color=INK,
                 fontsize=18, weight="bold", y=.98)
    fig.text(.5, .04, "Left: cutaway view of a sampled volume and its surface. Right: sampled surface only; neither view invents mesh connectivity.",
             ha="center", color=MUTED, fontsize=9.5)
    save(fig, "manual-three-dimensional.png")


def staggered():
    vertices = [[0, 0], [2, 0], [1.6, 1.2], [-.2, 1]]
    triangles = [[0, 1, 2], [0, 2, 3]]
    mesh = g.TriangleMesh2D(vertices, triangles, boundary_edges={
        "bottom": [[0, 1]], "right": [[1, 2]], "top": [[2, 3]], "left": [[3, 0]],
    })
    layout = g.staggered_clouds(mesh)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.9), facecolor=BG)
    ax = axes[0]
    ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
               color="#8194a2", lw=1.1)
    for n, tri in enumerate(mesh.triangles):
        c = mesh.vertices[tri].mean(axis=0)
        ax.text(*c, f"T{n+1}", color=MUTED, ha="center", va="center", fontsize=10)
    style(ax, "Explicit two-triangle mesh")
    ax = axes[1]
    ax.scatter(*layout.vertices.points.T, s=75, color=COLORS[0], edgecolor="white", zorder=3)
    for i, p in enumerate(layout.vertices.points):
        ax.annotate(f"v{i}", p, xytext=(5, 6), textcoords="offset points", color=INK, fontsize=9)
    style(ax, f"Vertex locations · {len(layout.vertices.points)}")
    ax = axes[2]
    mid = layout.edge_midpoints.points
    ax.scatter(*mid.T, s=75, color=COLORS[2], edgecolor="white", zorder=3)
    for (i, j), p in zip(layout.edges, mid):
        ax.annotate(f"m{i}{j}", p, xytext=(5, 6), textcoords="offset points", color=INK, fontsize=9)
    style(ax, f"Unique edge midpoints · {len(mid)}")
    for ax in axes:
        ax.set_xlim(-.4, 2.2); ax.set_ylim(-.25, 1.45)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Staggered layout: vertices and one midpoint per unique edge", color=INK,
                 fontsize=17, weight="bold", y=.99)
    fig.text(.5, .025, "The shared diagonal contributes one midpoint. Labels and connectivity come from the explicit TriangleMesh2D.",
             ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, .07, 1, .91))
    save(fig, "manual-staggered.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    planar_atlas()
    border_workflow()
    refinement()
    distributions()
    three_dimensional()
    staggered()
    print(f"Manual mesh figures saved in {OUT}")


if __name__ == "__main__":
    main()
