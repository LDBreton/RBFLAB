"""Global asymmetric and symmetric collocation for a small Poisson equation."""
import numpy as np
import sympy as sp
import rbflab as rbf


def run():
    # --8<-- [start:problem]
    cloud = rbf.geometry.unit_box_grid(5)
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem = model.stationary(sp.Eq(-model.laplacian(u), 2*sp.pi**2*exact),
                               boundary=[model.bc("boundary", sp.Eq(u, 0))])
    # --8<-- [end:problem]
    # --8<-- [start:asymmetric]
    kernel = rbf.IMQ(2)
    method = rbf.GlobalCollocation(kernel, scheme="asymmetric")
    system = method.assemble(problem, cloud)
    solution = system.solve()
    coefficients = solution.coefficients
    query = np.array([[.25, .3], [.6, .7]])
    predicted = solution.evaluate(query)
    # --8<-- [end:asymmetric]
    # --8<-- [start:rows]
    X = cloud.points
    interior, boundary = cloud.interior_indices, cloud.boundary_indices
    A = np.empty((len(X), len(X)))
    A[interior] = kernel.matrix(X[interior], X, left=-rbf.Laplacian())
    A[boundary] = kernel.matrix(X[boundary], X)
    direct_values = kernel.matrix(query, X) @ coefficients
    # --8<-- [end:rows]
    # --8<-- [start:symmetric]
    hermite = rbf.GlobalCollocation(kernel, scheme="symmetric")
    hermite_system = hermite.assemble(problem, cloud)
    hermite_solution = hermite_system.solve()
    hermite_values = hermite_solution.evaluate(query)
    # --8<-- [end:symmetric]
    expected = np.sin(np.pi*query[:, 0])*np.sin(np.pi*query[:, 1])
    result = {"assembly_difference": float(np.max(np.abs(A-system.matrix))),
              "evaluation_difference": float(np.max(np.abs(direct_values-predicted))),
              "asymmetric_error": float(np.max(np.abs(predicted-expected))),
              "symmetric_error": float(np.max(np.abs(hermite_values-expected)))}
    print(f"dense system={A.shape}, coefficient vector={coefficients.shape}")
    print(result)
    return result


if __name__ == "__main__":
    run()
