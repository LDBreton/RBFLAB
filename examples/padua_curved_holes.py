"""Compare Border and arc-length boundaries on a perforated curved domain.

Run from the repository: python examples/padua_curved_holes.py
Writes outputs/padua_curved_holes.png and prints node counts.
"""
from pathlib import Path
import numpy as np
import sympy as sp
from rbflab import geometry as g, meshgen, viz


def build_geometry():
    t = sp.symbols("t", real=True)
    radius = 1 + sp.Rational(3, 25)*sp.cos(5*t)
    exterior = (sp.Rational(11, 5)*radius*sp.cos(t),
                sp.Rational(8, 5)*radius*sp.sin(t))
    curves = [(f"outer_{i+1}", exterior, (i*np.pi/2, (i+1)*np.pi/2), 70)
              for i in range(4)]
    # Three separated ellipses; the third is rotated by 35 degrees.
    holes = [(-.95, .15, .38, .55, 0),
             (.05, -.55, .42, .23, 0),
             (.85, .35, .46, .28, 35*np.pi/180)]
    for i, (cx, cy, a, b, angle) in enumerate(holes):
        c, s = np.cos(angle), np.sin(angle)
        curve = (cx+a*c*sp.cos(t)-b*s*sp.sin(t),
                 cy+a*s*sp.cos(t)+b*c*sp.sin(t))
        curves.append((f"hole_{i+1}", curve, (0, 2*np.pi), 56))

    borders = [g.Border(curve, label, *interval, parameter=t)(
        count if label.startswith("outer") else -count)
        for label, curve, interval, count in curves]
    arcs = [g.ParametricBoundary(curve, label=label, interval=interval, parameter=t)
            for label, curve, interval, _ in curves]
    domain = g.ParametricDomain(arcs[:4], holes=arcs[4:])
    counts = {label: count for label, _, _, count in curves}
    return borders, domain, counts, holes


def build_clouds(degree=54):
    borders, domain, counts, holes = build_geometry()
    border_cloud = meshgen.generate(borders, method="padua", degree=degree,
                                   boundary_distance=.025)
    arc_cloud = meshgen.generate(domain, method="padua", degree=degree,
                                boundary=counts, boundary_distance=.025)
    # Check hole exclusion independently from the polygon representation.
    for cloud in (border_cloud, arc_cloud):
        for cx, cy, a, b, angle in holes:
            delta = cloud.interior-np.array([cx, cy])
            c, s = np.cos(angle), np.sin(angle)
            local = delta @ np.array([[c, -s], [s, c]])
            assert np.all((local[:, 0]/a)**2+(local[:, 1]/b)**2 > 1)
        assert set(cloud.boundary) == set(counts)
        assert len(cloud.boundary_indices) == sum(counts.values())
    return border_cloud, arc_cloud


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clouds = build_clouds()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    titles = ["Border(n): uniform curve parameter",
              "ParametricBoundary: uniform arc length"]
    for cloud, ax, title in zip(clouds, axes, titles):
        viz.plot_cloud(cloud, ax=ax, title=title)
        ax.collections[0].set_color("#526a86")
        ax.set_facecolor("#f5f7fa")
        for label, ids in cloud.boundary.items():
            chosen = np.arange(0, len(ids), 12)
            p = cloud.points[ids[chosen]]
            normal = cloud.normals[label][chosen]
            ax.quiver(p[:, 0], p[:, 1], normal[:, 0], normal[:, 1],
                      angles="xy", scale_units="xy", scale=9, color="#202b38",
                      width=.003)
        ax.legend(loc="upper right", fontsize=8, facecolor="white", labelcolor="#202b38")
        ax.set_xlabel("x", color="#202b38")
        ax.set_ylabel("y", color="#202b38")
        ax.tick_params(colors="#202b38")
        ax.set_title(title, loc="left", color="#202b38")
        print(f"{title}: {len(cloud.interior)} interior, "
              f"{len(cloud.boundary_indices)} boundary nodes")
    fig.set_facecolor("white")
    fig.suptitle("Padua degree 54 · curved exterior · three elliptical holes",
                 color="#202b38", fontsize=15)
    fig.tight_layout()
    path = Path(__file__).resolve().parents[1]/"outputs"/"padua_curved_holes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
