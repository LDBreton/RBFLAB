"""Three-dimensional scalar Poisson solve and reusable operators.

Run: python -m examples.tutorials.scalar_3d
"""
import numpy as np
import sympy as sp
import rbflab as rbf


def run(cells=3):
    model = rbf.SymbolicScalar(3)
    u = model.field
    x, y, z = model.coordinates
    exact = x*x + y*y + z*z
    lhs = -model.laplacian(u)
    problem = model.stationary(
        sp.Eq(lhs, lhs.subs(u, exact).doit()),
        boundary=[model.bc("boundary", sp.Eq(u, exact))],
    )
    cloud = rbf.unit_box_grid(cells, dimension=3)
    method = rbf.RBFFD(rbf.PHS(5), min(30, len(cloud.points)), polynomial_degree=2)
    solution = problem.solve(cloud, method)
    nodal = solution.evaluate(cloud.points)
    expected = np.sum(cloud.points**2, axis=1)
    error = float(np.max(np.abs(nodal-expected)))
    operators = rbf.RBFFD(
        spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)},
        stencil_size=min(30, len(cloud.points)),
        local_backend=rbf.PythonBackend(compute_condition=False),
    ).operators(source=cloud.points, targets=cloud.interior, space="u",
                operators={"lap": rbf.Laplacian(3)})
    reproduction = float(np.max(np.abs(operators.lap @ expected - 6)))
    print(f"dimension={cloud.dimension}, nodes={len(cloud.points)}, "
          f"nodal_max_error={error:.3e}, lap_reproduction_error={reproduction:.3e}")
    return error, reproduction


if __name__ == "__main__":
    run()
