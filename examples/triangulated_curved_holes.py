"""Border density drives interior refinement, using optional rbflab[triangulation].

Run: python examples/triangulated_curved_holes.py
Writes outputs/triangulated_curved_holes.png and a density comparison.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from shapely import points, distance
from shapely.geometry import LineString
from rbflab import geometry, meshgen
from padua_curved_holes import build_geometry


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    borders, domain, _, _ = build_geometry()
    # Keep the outer contour and other holes identical; refine only hole_1.
    coarse_counts = {f"outer_{i}": 24 for i in range(1, 5)}
    coarse_counts.update({f"hole_{i}": 24 for i in range(1, 4)})
    fine_counts = {**coarse_counts, "hole_1": 96}
    coarse = meshgen.triangulate(domain, boundary=coarse_counts, growth=.25)
    fine = meshgen.triangulate(domain, boundary=fine_counts, growth=.25)
    # The same count-driven workflow through the existing Border(n) API.
    for b in borders:
        b(fine_counts[b.label] * (-1 if b.label.startswith("hole") else 1))
    border_mesh = meshgen.triangulate(borders, growth=.25)
    print(f"Border version: {len(border_mesh.vertices)} vertices, {len(border_mesh.triangles)} triangles")
    hole = LineString(domain.polygon.interiors[0].coords)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="white")
    for mesh, ax, title, show_edges in zip(
            (coarse, fine, fine), axes,
            ("24 samples on each hole", "96 samples on the left hole", "Refined mesh vertices for RBF methods"),
            (True, True, False)):
        p, t = mesh.vertices, mesh.triangles
        cloud = geometry.staggered_clouds(mesh).vertices
        if show_edges:
            ax.triplot(p[:, 0], p[:, 1], t, color="#677e91", lw=.45)
        ax.scatter(*p.T, s=5, color="#273f52", linewidths=0, zorder=3)
        for label, edges in mesh.boundary_edges.items():
            color = "#c4491c" if label == "hole_1" else "#006d77"
            ids = np.unique(edges)
            ax.scatter(*p[ids].T, color=color, s=10, zorder=4)
        near = distance(points(cloud.interior), hole) < .2
        spacing = cKDTree(p).query(cloud.interior, k=2)[0][:, 1]
        print(f"{title}: {len(p)} vertices, {len(t)} triangles, "
              f"{near.sum()} interior nodes within .2 of left hole, "
              f"median local spacing {np.median(spacing[near]):.5f}")
        ax.set_title(f"{title}\n{len(p)} vertices · {len(t)} triangles", fontsize=11)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f7fa")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Boundary-driven triangulation on the same curved domain", fontsize=16)
    fig.tight_layout()
    path = Path(__file__).resolve().parents[1]/"outputs"/"triangulated_curved_holes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
