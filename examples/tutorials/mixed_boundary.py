"""Poisson problem with a Robin top and Dirichlet values elsewhere.

Run: python -m examples.tutorials.mixed_boundary
"""
import numpy as np
import sympy as sp
import rbflab as rbf


def run(cells=5):
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = x*x + y*y
    operator = -model.laplacian(u)
    robin = 2*u + model.normal_derivative(u)
    problem = model.stationary(
        sp.Eq(operator, operator.subs(u, exact).doit()),
        boundary=[
            model.bc("top", sp.Eq(robin, robin.subs(u, exact).doit())),
            model.bc("boundary", sp.Eq(u, exact)),
        ],
    )
    cloud = rbf.unit_box_grid(cells)
    method = rbf.RBFFD(rbf.PHS(5), min(20, len(cloud.points)), polynomial_degree=2)
    system = method.assemble(problem, cloud)
    solution = system.solve()
    nodal_error = float(np.max(np.abs(solution.evaluate(cloud.points) -
                                      (cloud.points[:, 0]**2 + cloud.points[:, 1]**2))))
    print(f"nodes={len(cloud.points)}, matrix={system.matrix.shape}, "
          f"nnz={system.matrix.nnz}, nodal_max_error={nodal_error:.3e}")
    return nodal_error


if __name__ == "__main__":
    run()
