"""Build scalar LHI from named functional maps and an explicit sparse equation."""
import numpy as np
from scipy.sparse.linalg import spsolve
import rbflab as rbf


def run():
    # --8<-- [start:problem]
    cloud = rbf.geometry.unit_box_grid(5)
    Xu = cloud.interior
    Xb = cloud.points[cloud.boundary_indices]
    Xf = Xu.copy()
    L = -rbf.Laplacian(2)

    def exact(X):
        return np.sin(np.pi*X[:, 0])*np.sin(np.pi*X[:, 1])

    fu, ff, gb = 2*np.pi**2*exact(Xu), 2*np.pi**2*exact(Xf), exact(Xb)
    # --8<-- [end:problem]
    # --8<-- [start:solve]
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
    source = {
        "u": rbf.Samples(Xu, size=12, target="require"),
        "boundary": rbf.Samples(Xb, size=8),
        "pde": rbf.Samples(Xf, operator=L, size=11, target="exclude"),
    }
    approximation = rbf.LocalApproximation(source=source, trial=space.representers(source))
    op = approximation.operators(targets=Xu, operators={"L": L}).L
    Su, Sb, Sf = (op[name].matrix for name in ("u", "boundary", "pde"))
    rhs = fu - Sb @ gb - Sf @ ff
    interior_values = spsolve(Su, rhs)
    # --8<-- [end:solve]
    # --8<-- [start:patch]
    row = 0
    patch = op.local(row)
    algebra = op.reconstruct_local(row)
    S, B, F = (patch.groups[name] for name in ("u", "boundary", "pde"))
    nS, nB = len(S), len(B)
    w = patch.weights[:, 0]  # source insertion order: u, boundary, pde
    wS, wB, wF = w[:nS], w[nS:nS+nB], w[nS+nB:]
    # --8<-- [end:patch]
    # --8<-- [start:row]
    reduced_rhs = fu[row] - wB @ gb[B] - wF @ ff[F]
    sparse_row = np.zeros(len(Xu))
    sparse_row[S] = wS
    # --8<-- [end:row]
    # --8<-- [start:evaluate]
    query = np.array([[.25, .3], [.6, .7]])
    # Rebuild neighborhoods at the query locations; no nearest-patch ownership.
    query_source = {
        "u": rbf.Samples(Xu, size=12),
        "boundary": rbf.Samples(Xb, size=8),
        "pde": rbf.Samples(Xf, operator=L, size=11, target="exclude"),
    }
    reconstruction = rbf.LocalApproximation(
        source=query_source, trial=space.representers(query_source),
    ).operators(targets=query, operators={"value": rbf.Identity(2), "dx": rbf.Derivative(0)})
    data = {"u": interior_values, "boundary": gb, "pde": ff}
    predicted = reconstruction.value @ data
    gradient_x = reconstruction.dx @ data
    # --8<-- [end:evaluate]
    result = {"row_difference": float(np.max(np.abs(sparse_row-Su.getrow(row).toarray()[0]))),
              "rhs_difference": float(abs(reduced_rhs-rhs[row])),
              "off_node_error": float(np.max(np.abs(predicted-exact(query)))),
              "center_excluded": bool(row not in F)}
    print(f"centers: solution={nS}, boundary={nB}, PDE={len(F)}; global={Su.shape}")
    print(f"local system={algebra.matrix.shape}; derivative shape={gradient_x.shape}")
    print(result)
    return result


if __name__ == "__main__":
    run()
