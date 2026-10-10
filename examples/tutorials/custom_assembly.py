"""Use RBF-FD maps to assemble variable-coefficient advection-diffusion-reaction."""
import argparse
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import sympy as sp
import rbflab as rbf
from rbflab import geometry


def run(backend="python"):
    # --8<-- [start:cloud]
    cloud = geometry.unit_box_grid(10)  # 121 nodes on the unit square
    X = cloud.points
    x, y = X.T
    I, B = cloud.interior_indices, cloud.boundary_indices
    # --8<-- [end:cloud]
    local = {"python": rbf.PythonBackend, "cpp": rbf.CppBackend,
             "torch": rbf.TorchBackend}[backend](compute_condition=False)
    # --8<-- [start:operators]
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=3)
    source = {"u": rbf.Samples(X, size=35)}
    approximation = rbf.LocalApproximation(
        source=source, trial=space.representers(source), backend=local,
        stencil_policy=rbf.StencilPolicy(scaling="local"),
    )
    ops = approximation.operators(targets=X, operators={"dx": rbf.Derivative(0),
                                  "dy": rbf.Derivative(1), "lap": rbf.Laplacian()})
    # Explicitly export Float64 matrices for the SciPy algorithm below.
    # With Torch this detaches tensors; with MP storage it rounds entries.
    Dx, Dy, Lap = (ops[name]["u"].to_scipy() for name in ("dx", "dy", "lap"))
    # --8<-- [end:operators]
    # --8<-- [start:combine]
    kappa = 1 + .2*x
    bx, by, reaction = .4, -.2, 1.
    A = (-sparse.diags(kappa) @ Lap
         + bx*Dx + by*Dy
         + reaction*sparse.eye(len(X))).tocsr()
    # --8<-- [end:combine]
    # --8<-- [start:data]
    boundary_data = np.sin(x)*np.cos(y)
    forcing = ((2*kappa+reaction)*boundary_data
               + bx*np.cos(x)*np.cos(y) - by*np.sin(x)*np.sin(y))
    # --8<-- [end:data]
    # --8<-- [start:solve]
    A_ii = A[I, :][:, I]
    A_ib = A[I, :][:, B]
    U = np.empty(len(X))
    U[B] = boundary_data[B]
    U[I] = spsolve(A_ii, forcing[I] - A_ib @ U[B])
    # --8<-- [end:solve]
    # --8<-- [start:inspect]
    local_row = ops.lap.local(int(I[0]))
    local_system = ops.lap.reconstruct_local(int(I[0]))
    # Work on a copy if experimenting with a matrix; .local() returns copies too.
    experimental_laplacian = Lap.copy()
    # --8<-- [end:inspect]
    # --8<-- [start:symbolic]
    model = rbf.SymbolicScalar(2)
    u = model.field
    sx, sy = model.coordinates
    truth = sp.sin(sx)*sp.cos(sy)
    lhs = -(1+.2*sx)*model.laplacian(u) + bx*sp.diff(u, sx) + by*sp.diff(u, sy) + reaction*u
    problem = model.stationary(sp.Eq(lhs, lhs.subs(u, truth).doit()),
                               boundary=[model.bc("boundary", sp.Eq(u, truth))])
    # Optional equation-to-system convenience route, using the same numerical recipe.
    method = rbf.RBFFD(rbf.PHS(5), stencil_size=35, polynomial_degree=3,
                       stencil_policy=rbf.StencilPolicy(scaling="local"),
                       local_backend=local)
    symbolic = problem.solve(cloud, method)
    # --8<-- [end:symbolic]
    difference = float(np.max(np.abs(U-symbolic.evaluate(X))))
    error = float(np.max(np.abs(U-boundary_data)))
    print(f"nodal vector={U.shape}, operator={A.shape}, interior solve={A_ii.shape}")
    print(f"symbolic/matrix difference={difference:.3e}, sampled nodal error={error:.3e}")
    return {"route_difference": difference, "error": error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    run(**vars(parser.parse_args()))
