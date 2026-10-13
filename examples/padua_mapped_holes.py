"""Whole-domain mapped Padua clouds, graded toward three curved holes.

Run: python examples/padua_mapped_holes.py
Output: outputs/padua_mapped_holes.png
"""
from pathlib import Path
import numpy as np
from shapely import points, distance
from shapely.geometry import LineString
from shapely.ops import unary_union
from rbflab import meshgen
from padua_curved_holes import build_geometry


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    borders, domain, counts, _ = build_geometry()
    clipped = meshgen.generate(domain, method="padua", degree=54,
                               boundary=counts, boundary_distance=.025)
    linear = meshgen.mapped_padua(domain, degree=18, boundary=counts, clustering=0)
    graded = meshgen.mapped_padua(domain, degree=18, boundary=counts, clustering=2)
    border_cloud = meshgen.mapped_padua(borders, degree=18, clustering=2)
    # Both APIs retain precisely the requested physical boundary counts.
    assert len(border_cloud.boundary_indices) == len(graded.boundary_indices) == sum(counts.values())
    hole_lines = unary_union([LineString(ring.coords) for ring in domain.polygon.interiors])
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="white")
    colors = ["#005f73", "#bb3e03", "#6a4c93"]
    titles = ["Clipped square grid", "Mapped · linear radial spacing", "Mapped · concentrated near holes"]
    for cloud, ax, title in zip((clipped, linear, graded), axes, titles):
        ax.scatter(*cloud.interior.T, s=8, color="#405970", linewidths=0)
        exterior = np.asarray(domain.polygon.exterior.coords)
        ax.plot(*exterior.T, color="#263849", lw=1.3)
        for ring, color in zip(domain.polygon.interiors, colors):
            p = np.asarray(ring.coords)
            ax.fill(*p.T, color="white", zorder=3)
            ax.plot(*p.T, color=color, lw=1.5, zorder=4)
        for label, ids in cloud.boundary.items():
            color = colors[int(label[-1])-1] if label.startswith("hole") else "#263849"
            ax.scatter(*cloud.points[ids].T, color=color, s=7, zorder=5)
        near = np.count_nonzero(distance(points(cloud.interior), hole_lines) < .12)
        fraction = near/len(cloud.interior)
        ax.set_title(f"{title}\n{len(cloud.interior)} interior · {fraction:.0%} within 0.12 of holes",
                     fontsize=11, color="#263849")
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f4f7")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        print(f"{title}: {len(cloud.interior)} interior, {near} near holes ({fraction:.1%})")
    fig.suptitle("Arc-length Padua maps on a curved domain with three holes", fontsize=16)
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1]/"outputs"/"padua_mapped_holes.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
