"""Inspect a scalar LHI patch and connect it to the interior sparse equation."""
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
    # --8<-- [start:solve]
    method = rbf.LHI(rbf.PHS(5), stencil_size=20, polynomial_degree=2)
    system = method.assemble(problem, cloud)
    solution = system.solve()
    interior_values = solution.interior_values
    # --8<-- [end:solve]
    # --8<-- [start:patch]
    row = 0  # first interior equation, not necessarily cloud node zero
    patch = system.stencils[row]
    S = patch.solution_indices
    B = patch.boundary_indices
    F = patch.pde_indices
    nS, nB = len(S), len(B)
    wS = patch.weights[:nS]
    wB = patch.weights[nS:nS+nB]
    wF = patch.weights[nS+nB:]
    # --8<-- [end:patch]
    # --8<-- [start:row]
    X = cloud.points
    forcing = 2*np.pi**2*np.sin(np.pi*X[:, 0])*np.sin(np.pi*X[:, 1])
    boundary_values = np.zeros(nB)  # Dirichlet values for this problem
    reduced_rhs = forcing[patch.center] - wB @ boundary_values - wF @ forcing[F]
    interior_column = {int(node): col for col, node in enumerate(cloud.interior_indices)}
    sparse_row = np.zeros(len(cloud.interior_indices))
    sparse_row[[interior_column[int(node)] for node in S]] = wS
    # --8<-- [end:row]
    # --8<-- [start:evaluate]
    query = np.array([[.25, .3], [.6, .7]])
    predicted = solution.evaluate(query)
    gradient_x = solution.evaluate(query, rbf.Derivative(0))
    # --8<-- [end:evaluate]
    expected = np.sin(np.pi*query[:, 0])*np.sin(np.pi*query[:, 1])
    result = {"row_difference": float(np.max(np.abs(sparse_row-system.matrix.getrow(row).toarray()[0]))),
              "rhs_difference": float(abs(reduced_rhs-system.rhs[row])),
              "off_node_error": float(np.max(np.abs(predicted-expected))),
              "center_excluded": bool(patch.center not in F)}
    print(f"centers: solution={nS}, boundary={nB}, PDE={len(F)}; global={system.matrix.shape}")
    print(f"local functional count={len(patch.points)}, polynomial terms=6; derivative shape={gradient_x.shape}")
    print(result)
    return result


if __name__ == "__main__":
    run()
