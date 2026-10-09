"""Regenerate the functional-center diagram without solving a PDE."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rbflab as rbf


def main():
    cloud = rbf.geometry.unit_box_grid(3)
    pde = np.array([[.2, .3], [.7, .3], [.3, .7], [.7, .7], [.5, .5]])
    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="#94a3b8", lw=1.5)
    boundary, solution = cloud.points[cloud.boundary_indices], cloud.interior
    for points, color, marker, label in (
        (boundary, "#0f766e", "s", "Boundary: prescribed values"),
        (solution, "#c2410c", "o", "Solution: unknown values"),
        (pde, "#4338ca", "D", "PDE: prescribed forcing"),
    ):
        ax.scatter(*points.T, s=100, color=color, marker=marker, label=label, zorder=3)
    ax.scatter(*solution[0], s=260, facecolors="none", edgecolors="#172554", lw=1.5, zorder=5)
    ax.annotate("Equation target", solution[0], xytext=(.02, .5),
                arrowprops={"arrowstyle": "->", "color": "#172554"}, color="#172554")
    ax.set(xlim=(-.09, 1.09), ylim=(-.08, 1.08), xlabel="$x$", ylabel="$y$", aspect="equal")
    ax.set_title("One domain, independent functional centers", loc="left", fontsize=15, color="#172554", pad=17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
    output = Path(__file__).resolve().parents[1] / "docs/assets/figures/lhi-centers.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
