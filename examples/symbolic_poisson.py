"""Solve one symbolic Poisson problem with global, LHI, or RBF-FD.

Run: python examples/symbolic_poisson.py --method global
     python examples/symbolic_poisson.py --method lhi
     python examples/symbolic_poisson.py --method rbffd
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as r


def run(method_name="global", cells=5):
    model = r.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    lhs = -model.laplacian(u)
    problem = model.stationary(
        sp.Eq(lhs, lhs.subs(u, exact).doit()),
        boundary=[model.bc("boundary", sp.Eq(u, 0))],
    )
    cloud = r.unit_box_grid(cells)
    methods = {
        "global": lambda: r.GlobalCollocation(r.IMQ(2), scheme="asymmetric"),
        "lhi": lambda: r.LHI(r.PHS(5), min(20, len(cloud.points)), polynomial_degree=2),
        "rbffd": lambda: r.RBFFD(r.PHS(5), min(20, len(cloud.points)), polynomial_degree=2),
    }
    solution = problem.solve(cloud, methods[method_name]())
    query = np.random.default_rng(17).uniform(0.1, 0.9, (40, 2))
    expected = np.sin(np.pi * query[:, 0]) * np.sin(np.pi * query[:, 1])
    error = float(np.max(np.abs(solution.evaluate(query) - expected)))
    nodal = cloud.points
    nodal_exact = np.sin(np.pi*nodal[:, 0])*np.sin(np.pi*nodal[:, 1])
    nodal_error = float(np.max(np.abs(solution.evaluate(nodal)-nodal_exact)))
    print(f"{method_name}: {len(cloud.points)} nodes; nodal max error {nodal_error:.3e}; "
          f"off-node max error {error:.3e}")
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("global", "lhi", "rbffd"), default="global")
    parser.add_argument("--cells", type=int, default=5)
    args = parser.parse_args()
    run(args.method, args.cells)
