"""Generate teaching figures by executing the same snippet sections as the manual.

Run: python -m examples.make_teaching_figures
Only trusted, repository-owned tutorial source is executed. Numerical recipes
remain in the example files; plotting does not maintain a second implementation.
"""
from pathlib import Path
from textwrap import dedent
import hashlib
import json
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import rbflab as rbf
from rbflab import geometry, meshgen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
INK, BLUE, TEAL, ORANGE = "#16324f", "#2765b0", "#008c86", "#e27b3d"
CMAP = LinearSegmentedColormap.from_list("rbflab_field", ["#e6f5ed", "#31b8a6", "#247eb1", "#192e66"])


def lesson(name, sections, **values):
    """Run the exact named snippets used by MkDocs, in the requested order."""
    path = ROOT / "examples/tutorials" / (name + ".py")
    text = path.read_text(encoding="utf-8")
    state = dict(np=np, sp=sp, sparse=sparse, spsolve=spsolve, splu=splu,
                 rbf=rbf, geometry=geometry, meshgen=meshgen)
    state.update(values)
    for section in sections:
        start = "# --8<-- [start:" + section + "]"
        end = "# --8<-- [end:" + section + "]"
        code = text.split(start, 1)[1].split(end, 1)[0]
        exec(compile(dedent(code), str(path) + ":" + section, "exec"), state)
    return state


def style(ax, title, square=True):
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=14, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#bacbda")
    ax.tick_params(colors="#577086", labelsize=9)
    if square:
        ax.set(xlabel="x", ylabel="y", aspect="equal",
               xlim=(-.04, 1.04), ylim=(-.04, 1.04))
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def nodes(ax, cloud, title):
    ax.scatter(*cloud.interior.T, color=BLUE, s=26, zorder=3, label="interior values")
    ax.scatter(*cloud.points[cloud.boundary_indices].T, color=ORANGE, s=28,
               marker="s", zorder=3, label="boundary values")
    style(ax, title)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2,
              fontsize=9, frameon=False)


def field(ax, X, U, title, levels=None, cmap=CMAP):
    im = ax.tricontourf(*X.T, U, levels=levels if levels is not None else 24, cmap=cmap)
    ax.tricontour(*X.T, U, levels=7, colors="white", linewidths=.45, alpha=.5)
    style(ax, title)
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "axes.facecolor": "#f7fafc"})
    # Data and reconstruction are exactly those of the interpolation tutorial.
    data = lesson("interpolation", ["data", "interpolants", "evaluate"])
    X, fit, U = data["centers"], data["phs"], data["values"]
    g = np.linspace(0, 1, 70); gx, gy = np.meshgrid(g, g)
    Q = np.column_stack((gx.ravel(), gy.ravel()))
    predicted = fit.evaluate(Q).reshape(gx.shape)
    derivative = fit.evaluate(Q, rbf.Derivative(0)).reshape(gx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    for ax, Z, title in zip(axes, (predicted, derivative), ("Interpolated field", "Derivative in x")):
        im = ax.contourf(gx, gy, Z, levels=24, cmap=CMAP)
        ax.scatter(*X.T, s=14, c="white", edgecolors=INK, linewidths=.35)
        style(ax, title)
        fig.colorbar(im, ax=ax, shrink=.74, pad=.025)
    save(fig, "teaching_interpolation.png")
    fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
    ax.contourf(gx, gy, predicted, levels=24, cmap=CMAP)
    ax.scatter(*X.T, s=24, c="white", edgecolors=INK, linewidths=.5)
    style(ax, "64 scattered samples")
    save(fig, "teaching_data_card.png")

    custom = lesson("custom_kernel", ["family", "mixture"])
    kernel = custom["kernel"]
    r = np.linspace(0, 2.5, 250); offsets = np.column_stack((r, np.zeros_like(r)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, alpha, title, color in zip(axes, ((0, 0), (1, 0)),
                                      ("IMQ + Gaussian", "Derivative along (r, 0)"), (BLUE, TEAL)):
        val = kernel.derivative(offsets, alpha)
        ax.plot(r, val, lw=2.8, color=color)
        ax.fill_between(r, val, 0, color=color, alpha=.08)
        ax.scatter([0], [val[0]], color=ORANGE, s=40, zorder=4)
        ax.axhline(0, color="#bacbda", lw=.8)
        style(ax, title, square=False); ax.set(xlabel="r", ylabel="value")
    save(fig, "teaching_kernel.png")

    # Actual group membership, including coincident PDE/solution locations.
    lhi = lesson("lhi_construction", ["problem", "solve", "patch"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7), layout="constrained")
    nodes(axes[0], lhi["cloud"], "36 nodes on the square")
    ax = axes[1]; cloud = lhi["cloud"]
    ax.scatter(*cloud.points.T, c="#d7e1e8", s=25)
    for points, indices, color, marker, size, label in (
        (lhi["Xu"], lhi["S"], BLUE, "o", 40, "12 solution values"),
        (lhi["Xb"], lhi["B"], ORANGE, "s", 40, "8 boundary data"),
    ):
        ax.scatter(*points[indices].T, c=color, marker=marker, s=size, label=label)
    ax.scatter(*lhi["Xf"][lhi["F"]].T, facecolors="none", edgecolors=TEAL,
               marker="^", s=130, lw=1.5, label="11 PDE data")
    ax.scatter(*lhi["Xu"][0], c="#a33150", marker="*", s=180, edgecolors="white",
               zorder=5, label="target")
    style(ax, "One Hermite neighborhood")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2,
              fontsize=8.5, frameon=False)
    save(fig, "teaching_centers.png")

    stencil = lesson("one_stencil", ["geometry", "operator", "inspect", "weights"],
                     cells=5, stencil_size=20)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    ax = axes[0]
    ax.scatter(*stencil["cloud"].points.T, c="#d7e1e8", s=32)
    ax.scatter(*stencil["points"].T, c=BLUE, s=45)
    ax.scatter(*stencil["target"][0], c=ORANGE, marker="*", s=180,
               edgecolors="white", zorder=5)
    radius = np.max(np.linalg.norm(stencil["points"] - stencil["target"][0], axis=1))
    ax.add_patch(plt.Circle(stencil["target"][0], radius, fill=False,
                           color=TEAL, ls="--", lw=1.3))
    style(ax, "20 selected / 36 available")
    weights = stencil["op"].matrix.toarray()[0] * radius**2
    axes[1].bar(np.arange(len(weights)), weights,
                color=np.where(weights < 0, ORANGE, BLUE), width=.85)
    axes[1].axhline(0, color=INK, lw=.7)
    style(axes[1], "One signed Laplacian row", square=False)
    axes[1].set(xlabel="source node index", ylabel="weight × radius²")
    save(fig, "tutorial_stencil.png")

    assembly = lesson("custom_assembly", ["cloud", "operators", "combine", "data", "solve"],
                      local=rbf.PythonBackend(compute_condition=False))
    cloud, X, values = assembly["cloud"], assembly["X"], assembly["U"]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.4), layout="constrained")
    nodes(axes[0], cloud, "01  Sample the square")
    axes[1].spy(assembly["A"], color=TEAL, markersize=1.3)
    axes[1].xaxis.tick_bottom()
    style(axes[1], "02  Build the operator", square=False)
    axes[1].set(xlabel="source column", ylabel="equation row")
    im = field(axes[2], X, values, "03  Solve for the field")
    fig.colorbar(im, ax=axes[2], shrink=.7, pad=.025, label="u")
    save(fig, "tutorial_overview.png")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout="constrained")
    nodes(axes[0], cloud, "81 unknown / 40 boundary")
    axes[1].spy(assembly["Lap"], color=TEAL, markersize=1.8)
    axes[1].xaxis.tick_bottom()
    style(axes[1], "Laplacian / 35-node stencils", square=False)
    axes[1].set(xlabel="source column", ylabel="target row")
    save(fig, "teaching_operators.png")
    fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
    ax.spy(assembly["A"], color=TEAL, markersize=1.8); ax.xaxis.tick_bottom()
    style(ax, "Sparse equations", square=False)
    ax.set(xlabel="source column", ylabel="equation row")
    save(fig, "teaching_algorithm_card.png")

    first = lesson("first_problem", ["geometry", "equation", "solve"])
    values = first["solution"].evaluate(first["cloud"].points)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7), layout="constrained")
    nodes(axes[0], first["cloud"], "121 labeled nodes")
    im = field(axes[1], first["cloud"].points, values, "Computed Poisson field")
    fig.colorbar(im, ax=axes[1], shrink=.72, label="u")
    save(fig, "first_problem.png")
    fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
    field(ax, first["cloud"].points, values, "Poisson on the square")
    save(fig, "teaching_pde_card.png")

    heat = lesson("heat_equation", ["settings", "operators", "factors", "loop"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    for ax, state, title in zip(axes, (heat["states"][0], heat["states"][-1]),
                                ("Initial field / t = 0", "BDF2 field / t = 0.05")):
        im = field(ax, heat["X"], state, title, levels=np.linspace(0, 1, 26), cmap="magma")
        ax.scatter(*heat["X"].T, s=8, c="white", alpha=.55)
    fig.colorbar(im, ax=axes, shrink=.75, label="temperature (shared scale)")
    save(fig, "teaching_heat.png")

    names = ("interpolation", "custom_kernel", "one_stencil", "lhi_construction",
             "custom_assembly", "first_problem", "heat_equation")
    manifest = {
        "source_sha256": {name: hashlib.sha256((ROOT / "examples/tutorials" / (name+".py")).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                          for name in names},
        "square_nodes": len(cloud.points), "stencil_nodes": len(stencil["points"]),
        "lhi_groups": {name: len(ids) for name, ids in lhi["patch"].groups.items()},
        "nodal_pde_error": float(np.max(np.abs(assembly["U"]-assembly["boundary_data"]))),
        "heat_peak": [float(np.max(heat["states"][0])), float(np.max(heat["states"][-1]))],
    }
    (OUT / "tutorial_figures.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
