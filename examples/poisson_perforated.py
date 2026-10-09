"""Symbolic Poisson on an ellipse with two named holes.

Run: python -m examples.poisson_perforated --plot solution.png
Optional: --backend cpp or --backend torch (see installation guide).
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import local_method, solve, errors, report


def make_domain():
    """Preserve the exterior label and give each hole its own boundary group."""
    return geometry.with_holes(
        geometry.Ellipse(1.6, 1., labels=("wall",)),
        {"round_hole": geometry.Disk(.28, center=(-.65, 0)),
         "slot": geometry.Ellipse(.24, .43, center=(.6, 0), angle=-.3)},
    )


def run(interior=500, seed=42, backend="python", plot=None):
    """Solve a nonpolynomial manufactured problem and report independent errors."""
    domain = make_domain()
    cloud = meshgen.generate(domain, interior=interior,
        boundary={"wall": 120, "round_hole": 32, "slot": 40}, seed=seed)
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(x)*sp.cos(y)
    problem = model.stationary(
        sp.Eq(-model.laplacian(u), 2*exact),
        boundary=[model.bc(label, sp.Eq(u, exact)) for label in domain.boundaries],
    )
    solution, timing = solve(problem, cloud, local_method(backend))
    exact_values = lambda points: np.sin(points[:, 0])*np.cos(points[:, 1])
    metrics = report("perforated/"+backend, cloud, timing,
                     errors(solution, cloud, domain, exact_values))
    if plot:
        from rbflab import viz
        fig, _ = viz.plot_scalar(solution, domain=domain, title="Poisson / two holes")
        fig.savefig(plot, dpi=180)
    return domain, cloud, solution, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interior", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    parser.add_argument("--plot")
    run(**vars(parser.parse_args()))
