"""FreeFEM-inspired Border syntax, connected to a symbolic Poisson problem.

Run: python -m examples.parametric_borders --plot borders.png
Unified border generation and automatic normals require RBFLAB 0.4 or later.
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab.geometry import Border
from rbflab.meshgen import RBFMesh
from examples._curved import local_method, solve, report


def make_cloud(interior=240, seed=42):
    """Generate an oriented annulus with automatic curve normals."""
    outer = Border(lambda t: (np.cos(t), np.sin(t)),
                   label="outer", t_start=0, t_end=2*np.pi)
    hole = Border(lambda t: (.4*np.cos(t), .4*np.sin(t)),
                  label="hole", t_start=0, t_end=2*np.pi)
    mesh = RBFMesh(outer(96), hole(-48))
    cloud = rbf.meshgen.generate(mesh, interior=interior, method="halton", seed=seed)
    return mesh, cloud


def run(interior=240, seed=42, backend="python", plot=None):
    """Check border labels and hole normals through Dirichlet/Neumann assembly."""
    mesh, cloud = make_cloud(interior, seed)
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.exp(x/2)*sp.cos(y)
    lhs = -model.laplacian(u)+u
    dn = model.normal_derivative(u)
    problem = model.stationary(sp.Eq(lhs, lhs.subs(u, exact).doit()), boundary=[
        model.bc("outer", sp.Eq(u, exact)),
        model.bc("hole", sp.Eq(dn, dn.subs(u, exact).doit())),
    ])
    solution, timings = solve(problem, cloud, local_method(backend))
    reference = np.exp(cloud.points[:, 0]/2)*np.cos(cloud.points[:, 1])
    metrics = report("parametric borders/"+backend, cloud, timings,
        {"nodal_max": float(np.max(abs(solution.evaluate(cloud.points)-reference)))})
    if plot:
        from rbflab import viz
        fig, _ = viz.plot_cloud(cloud, normals=True, title="Parametric borders / labeled nodes")
        fig.savefig(plot, dpi=180)
    return mesh, cloud, solution, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interior", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    parser.add_argument("--plot")
    run(**vars(parser.parse_args()))
