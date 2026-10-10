"""Independent solution, PDE and boundary samples in a scalar Hermite construction."""
import numpy as np
from scipy.sparse.linalg import spsolve
import rbflab as rbf


def run():
    cloud = rbf.geometry.unit_box_grid(3)
    Xu = cloud.interior
    Xb = cloud.points[cloud.boundary_indices]
    Xf = np.array([[.2, .3], [.7, .3], [.3, .7], [.7, .7], [.5, .5]])
    L = -rbf.Laplacian(2)
    exact = lambda X: 1 + np.sum(X**2, axis=1)
    source = {
        "u": rbf.Samples(Xu),  # all four solution values
        "wall": rbf.Samples(Xb),
        "forcing": rbf.Samples(Xf, operator=L, size=3, target="exclude"),
    }
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
    approximation = rbf.LocalApproximation(source=source, trial=space.representers(source))
    ops = approximation.operators(targets=Xu, operators={"L": L})
    ff, gb = np.full(len(Xf), -4.), exact(Xb)
    rhs = np.full(len(Xu), -4.) - ops.L["wall"] @ gb - ops.L["forcing"] @ ff
    U = spsolve(ops.L["u"].matrix, rhs)
    query = np.array([[.31, .47], [.68, .57]])
    evaluate = approximation.operators(targets=query, operators={"value": rbf.Identity(2)})
    predicted = evaluate.value @ {"u": U, "wall": gb, "forcing": ff}
    error = float(np.max(np.abs(predicted-exact(query))))
    print("Independent-center error:", error)
    print("First stencil groups:", ops.L.local(0).groups)
    return error


if __name__ == "__main__":
    run()
