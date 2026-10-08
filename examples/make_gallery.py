"""Rebuild the README image and GIF from numerical RBFLAB solutions.

From the repository root: python -m examples.make_gallery
Install the optional plotting extra first: pip install -e ".[examples]"
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import rbflab as r
from rbflab import viz
from examples.stokes_spaces import make_solution


def heat_trajectory(shape=10):
    model = r.SymbolicScalar(2, transient=True)
    u, t = model.field, model.time
    x, y = model.coordinates
    pulse = 40*x*(1-x)*y*(1-y)*sp.exp(-55*((x-sp.Rational(7, 20))**2
                                          + (y-sp.Rational(17, 25))**2))
    problem = model.evolution(
        sp.Eq(sp.diff(u, t)-sp.Rational(1, 25)*model.laplacian(u), 0),
        initial=pulse,
        boundary=[model.bc("boundary", sp.Eq(u, 0))],
    )
    cloud = r.unit_box_grid(12)
    trajectory = problem.solve(cloud, r.GlobalCollocation(r.IMQ(shape),
                               scheme="asymmetric"), dt="0.0075", steps=40)
    return cloud, trajectory


def main(output=Path("docs/assets")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cloud, trajectory = heat_trajectory()
    nodal = [np.asarray(state.evaluate(cloud.points))
             for state in (trajectory.initial, *trajectory.states)]
    peaks = np.array([values.max() for values in nodal])
    if np.any(np.diff(peaks) > 1e-6) or min(values.min() for values in nodal) < -1e-3:
        raise RuntimeError("Heat gallery failed its decay/positivity sanity check")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), layout="constrained")
    vmax = float(peaks[0])
    snapshots = (trajectory.initial, trajectory.states[19], trajectory.final)
    for state, ax, caption in zip(snapshots, axes,
                                  ("START  ·  t = 0", "DIFFUSION  ·  t = 0.15",
                                   "LATER  ·  t = 0.30")):
        viz.plot_scalar(state, ax=ax, title=caption, resolution=85,
                        vmin=0, vmax=vmax, colorbar=False,
                        cloud=cloud if state is trajectory.initial else None)
    fig.suptitle("A heat pulse, evolved by global RBF collocation",
                 color="#e8f1ff", fontsize=17, weight="bold")
    fig.savefig(output/"heat_diffusion.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    viz.animate_scalar(trajectory, output/"heat_diffusion.gif",
                       resolution=80, fps=9, every=2,
                       title="Heat pulse · numerical solution")

    stokes = make_solution()
    fig, _ = viz.plot_velocity(stokes, resolution=80,
                               title="Divergence-free steady Stokes")
    fig.savefig(output/"stokes_velocity.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote heat PNG, heat GIF, and Stokes PNG to {output}")


if __name__ == "__main__":
    main()
