"""Compare global symmetric/asymmetric, LHI, and RBF-FD on one Poisson PDE.

Run: python -m examples.tutorials.compare_methods
"""
import numpy as np
import sympy as sp
import rbflab as rbf


def run(cells=5):
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    lhs = -model.laplacian(u)
    problem = model.stationary(
        sp.Eq(lhs, lhs.subs(u, exact).doit()),
        boundary=[model.bc("boundary", sp.Eq(u, 0))],
    )
    cloud = rbf.unit_box_grid(cells)
    query = np.random.default_rng(19).uniform(.1, .9, (40, 2))
    target = np.sin(np.pi*query[:, 0])*np.sin(np.pi*query[:, 1])
    methods = {
        "global_symmetric": rbf.GlobalCollocation(rbf.IMQ(2), scheme="symmetric"),
        "global_asymmetric": rbf.GlobalCollocation(rbf.IMQ(2), scheme="asymmetric"),
        "lhi": rbf.LHI(rbf.PHS(5), min(20, len(cloud.points)), polynomial_degree=2),
        "rbffd": rbf.RBFFD(rbf.PHS(5), min(20, len(cloud.points)), polynomial_degree=2),
    }
    results = {}
    for name, method in methods.items():
        system = method.assemble(problem, cloud)
        solution = system.solve()
        error = float(np.max(np.abs(solution.evaluate(query)-target)))
        results[name] = error
        print(f"{name}: matrix={system.matrix.shape}, "
              f"off_node_max_error={error:.3e}")
    return results


if __name__ == "__main__":
    run()
