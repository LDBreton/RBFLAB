"""Generate the manual's original stencil figures with RBFLAB and Matplotlib.

Run from a source checkout: python -m examples.make_stencil_figures
The irregular stencil is computed; the LHI diagram is a labeled schematic.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import rbflab as rbf

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
BLUE, ORANGE, TEAL, GRAY = "#245b91", "#bc5b26", "#087f8c", "#c5ced9"

def save(fig, name):
    """Save vector and raster copies of an original documentation figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, bbox_inches="tight",
                    facecolor="white", metadata={"Creator": "RBFLAB"})
    plt.close(fig)

def rbf_fd():
    """Compute a real 20-node Laplacian stencil on a seeded irregular cloud."""
    rng = np.random.default_rng(17)
    cloud = rbf.unit_box_grid(10)
    points = cloud.points.copy()
    points[cloud.interior_indices] += rng.uniform(-0.025, 0.025,
                                                  (len(cloud.interior_indices), 2))
    i = np.argmin(np.linalg.norm(points - [0.5, 0.5], axis=1))
    target = points[i]
    op = rbf.RBFFD(
        spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)}, stencil_size=20,
        local_backend=rbf.PythonBackend(compute_condition=False),
    ).operators(source=points, targets=target[None], space="u",
                operators={"lap": rbf.Laplacian(2)}).lap
    local = op.local(0)
    selected = points[local.indices]
    weights = local.weights[:, 0]
    radius = np.linalg.norm(selected - target, axis=1).max()
    assert abs(weights.sum()) < 1e-9
    assert abs(weights @ np.sum(selected**2, axis=1) - 4) < 1e-9

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained")
    ax.scatter(*points.T, s=15, c=GRAY, label="Other cloud nodes")
    for p in selected:
        ax.plot([target[0], p[0]], [target[1], p[1]], c=BLUE, alpha=.16, lw=.8)
    ax.add_patch(Circle(target, radius, fill=False, ec=BLUE, ls="--", lw=1.2))
    ax.scatter(*selected.T, s=35, c=BLUE, label="20 stencil nodes", zorder=3)
    ax.scatter(*target, s=135, marker="*", c=ORANGE, edgecolor="white",
               linewidth=.6, label="Target (also a value node)", zorder=4)
    ax.set(xlim=(-.04, 1.04), ylim=(-.04, 1.04), xlabel="$x$", ylabel="$y$",
           title="A. Select a local neighborhood")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=.96)
    # Show the actual sparse row in global cloud ordering.
    values = np.zeros(len(points))
    values[local.indices] = radius**2 * weights
    bx.axhline(0, c=GRAY, lw=1)
    bx.vlines(local.indices, 0, values[local.indices],
              colors=[ORANGE if w < 0 else BLUE for w in weights], lw=1.5)
    bx.scatter(local.indices, values[local.indices],
               c=[ORANGE if w < 0 else BLUE for w in weights], s=22, zorder=3)
    bx.set(xlim=(-2,len(points)+1), xlabel="Global node index $j$",
           ylabel=r"Dimensionless weight $R^2 w_{ij}$",
           title="B. Insert weights into one sparse row")
    for a in (ax,bx):
        a.spines[["top","right"]].set_visible(False)
    save(fig,"rbf_fd_stencil")
    print(f"Stencil checks: sum(w)={weights.sum():.2e}; Lap(r^2)={weights @ np.sum(selected**2,axis=1):.12g}")

def lhi():
    """Draw data-functional roles; this is not a measured solver stencil."""
    solution = np.array([[.22,.36],[.35,.18],[.40,.42],[.56,.28],[.61,.54],[.28,.62]])
    target = solution[0]
    pde = solution[1:]
    boundary = np.array([[0,.12],[0,.36],[0,.61]])
    fig, ax = plt.subplots(figsize=(8.4,4.8), layout="constrained")
    ax.axvspan(-.08,0,color="#eef1f6")
    ax.axvline(0,c="#526176",lw=1.7)
    ax.scatter(*solution.T,s=120,facecolors="white",edgecolors=BLUE,lw=2,
               label=r"Solution centers: $u(x_j)$ unknown",zorder=3)
    ax.scatter(*pde.T,s=32,marker="^",c=TEAL,
               label=r"PDE centers: $\mathcal{L}u=f$ known",zorder=4)
    ax.scatter(*boundary.T,s=85,marker="s",c=ORANGE,
               label=r"Boundary centers: $\mathcal{B}u=g$ known",zorder=4)
    ax.scatter(*target,s=155,marker="*",c=ORANGE,zorder=5)
    ax.annotate("Target: solution value only\nPDE is enforced by the assembled row",
                xy=target,xytext=(.33,.79),fontsize=10,
                arrowprops=dict(arrowstyle="->",color="#526176"),color="#142d52")
    ax.text(.43,.09,"Circle + triangle:\nsame location, two functionals",
            fontsize=10,color="#526176")
    ax.text(-.04,.85,r"$\partial\Omega$",fontsize=16,color="#526176")
    ax.text(.73,.60,r"$\Omega$",fontsize=20,color="#526176")
    ax.set(xlim=(-.09,.88),ylim=(-.03,.94),
           title="LHI: geometry and data play different roles")
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.13),fontsize=10,frameon=False)
    save(fig,"lhi_centers")

def main():
    """Rebuild both figures without downloading or copying paper illustrations."""
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.titlesize":12,"axes.titleweight":"bold",
                         "axes.labelcolor":"#142d52","text.color":"#142d52",
                         "svg.fonttype":"none"})
    rbf_fd(); lhi()

if __name__ == "__main__":
    main()
