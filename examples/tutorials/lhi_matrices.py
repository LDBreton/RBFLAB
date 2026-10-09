"""Construct scalar LHI and advance heat directly with its sparse matrix blocks."""
import argparse
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, splu
import rbflab as rbf


def run(backend="python", steps=5, dt=.001):
    implementation = {"python": rbf.PythonBackend, "cpp": rbf.CppBackend,
                      "torch": rbf.TorchBackend}[backend](compute_condition=False)
    # --8<-- [start:cloud]
    cloud = rbf.geometry.unit_box_grid(6)
    Xu = cloud.points[cloud.interior_indices]
    Xb = cloud.points[cloud.boundary_indices]
    Xf = Xu.copy()  # identical global ordering; local PDE stencils exclude their target
    I = rbf.Identity(2)
    L = -rbf.Laplacian(2)
    # --8<-- [end:cloud]
    # --8<-- [start:construction]
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
    source = {
        "u": rbf.Samples(Xu, operator=I, size=15),
        "pde": rbf.Samples(Xf, operator=L, size=8, target="exclude"),
        "boundary": rbf.Samples(Xb, operator=I, size=6),
    }
    local = rbf.LocalApproximation(
        source=source,
        trial=space.representers(source),
        backend=implementation,
    )
    ops = local.operators(targets=Xu, operators={"L": L})
    Su = ops.L["u"].to_scipy()
    Sf = ops.L["pde"].to_scipy()
    Sb = ops.L["boundary"].to_scipy()
    # --8<-- [end:construction]
    # --8<-- [start:stationary]
    def exact(X):
        return 1 + np.sum(X**2, axis=1)

    fu = np.full(len(Xu), -4.)
    ff = np.full(len(Xf), -4.)
    gb = exact(Xb)
    A = Su
    rhs = fu - Sf @ ff - Sb @ gb
    U = spsolve(A, rhs)
    # --8<-- [end:stationary]
    stationary_error = float(np.max(np.abs(U-exact(Xu))))
    # --8<-- [start:heat]
    # u_t - Laplacian(u) = 0, homogeneous Dirichlet data.
    M = sparse.eye(len(Xu), format="csc") - Sf
    solve_step = splu((M/dt + A).tocsc()).solve
    U = np.sin(np.pi*Xu[:, 0])*np.sin(np.pi*Xu[:, 1])
    for n in range(steps):
        U = solve_step(M @ U/dt)
    # --8<-- [end:heat]
    final_time = steps*dt
    truth = np.exp(-2*np.pi**2*final_time)*np.sin(np.pi*Xu[:, 0])*np.sin(np.pi*Xu[:, 1])
    heat_error = float(np.max(np.abs(U-truth)))
    print(f"solution={len(Xu)}, PDE={len(Xf)}, boundary={len(Xb)}")
    print(f"stationary quadratic error={stationary_error:.3e}")
    print(f"heat at t={final_time:g}: sampled error={heat_error:.3e}")
    return {"stationary_error": stationary_error, "heat_error": heat_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=.001)
    run(**vars(parser.parse_args()))
