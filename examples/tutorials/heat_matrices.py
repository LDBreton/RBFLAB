"""Heat equation through exposed RBF-FD matrices and the symbolic PDE interface.

Run: python -m examples.tutorials.heat_matrices
Optional: python -m examples.tutorials.heat_matrices --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import expm_multiply, splu
import sympy as sp

import rbflab as rbf


def exact(points, time, kappa=1.0, nonzero_boundary=False):
    """Manufactured temperature at an array of 2D points."""
    x, y = np.asarray(points).T
    if nonzero_boundary:
        return np.exp(-time) * (1 + x + y)
    return np.exp(-2 * kappa * np.pi**2 * time) * np.sin(np.pi*x) * np.sin(np.pi*y)


def forcing(points, time, kappa=1.0, nonzero_boundary=False):
    """Right-hand side of u_t = kappa Laplacian(u) + f."""
    if nonzero_boundary:
        return -exact(points, time, kappa, True)
    return np.zeros(len(points))


# --8<-- [start:rbf_fd_laplacian]
def rbf_fd_laplacian(cloud, stencil_size=20):
    """Assemble reusable RBF-FD Laplacian weights at every node."""
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
    source = {"temperature": rbf.Samples(cloud.points, size=min(stencil_size, len(cloud.points)))}
    approximation = rbf.LocalApproximation(
        source=source, trial=space.representers(source),
        backend=rbf.PythonBackend(compute_condition=False),
    )
    ops = approximation.operators(
        targets=cloud.points,
        operators={"lap": rbf.Laplacian(2)},
    )
    return ops.lap.matrix.tocsr(), ops


# --8<-- [end:rbf_fd_laplacian]


# --8<-- [start:march]
def march(matrix, cloud, *, dt, steps, kappa=1.0, scheme="backward_euler",
          nonzero_boundary=False):
    """March the interior nodal vector with BE or BE-started BDF2."""
    interior = cloud.interior_indices
    boundary = cloud.boundary_indices
    lap_ii = matrix[interior, :][:, interior].tocsc()
    lap_ib = matrix[interior, :][:, boundary].tocsr()
    ident = sparse.eye(len(interior), format="csc")
    be_factor = splu(ident - dt*kappa*lap_ii)
    bdf2_factor = splu(1.5*ident - dt*kappa*lap_ii) if scheme == "bdf2" and steps > 1 else None

    previous = exact(cloud.points[interior], 0, kappa, nonzero_boundary)
    older = None
    states = [exact(cloud.points, 0, kappa, nonzero_boundary)]
    for step in range(1, steps+1):
        time = step*dt
        g = exact(cloud.points[boundary], time, kappa, nonzero_boundary)
        source = kappa*(lap_ib @ g) + forcing(cloud.points[interior], time, kappa, nonzero_boundary)
        if bdf2_factor is None or step == 1:
            current = be_factor.solve(previous + dt*source)
        else:
            current = bdf2_factor.solve(2*previous - 0.5*older + dt*source)
        full = np.empty(len(cloud.points))
        full[interior], full[boundary] = current, g
        states.append(full)
        older, previous = previous, current
    return np.asarray(states), (1 + int(bdf2_factor is not None))


# --8<-- [end:march]


# --8<-- [start:symbolic_solution]
def symbolic_solution(cloud, *, dt, steps, kappa=1.0, scheme="backward_euler",
                      nonzero_boundary=False):
    """Solve the same PDE using the higher-level symbolic interface."""
    model = rbf.SymbolicScalar(2, transient=True)
    u, t = model.field, model.time
    x, y = model.coordinates
    if nonzero_boundary:
        truth = sp.exp(-t)*(1+x+y)
    else:
        truth = sp.exp(-2*kappa*sp.pi**2*t)*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    rhs = sp.diff(truth, t) - kappa*model.laplacian(truth)
    problem = model.evolution(
        sp.Eq(sp.diff(u, t) - kappa*model.laplacian(u), rhs),
        initial=truth.subs(t, 0),
        boundary=[model.bc("boundary", sp.Eq(u, truth))],
    )
    method = rbf.RBFFD(rbf.PHS(5), min(20, len(cloud.points)), polynomial_degree=2)
    return problem.solve(cloud, method, str(dt), steps, scheme=scheme)


# --8<-- [end:symbolic_solution]


def plot_states(cloud, states, dt, output):
    """Save a compact temperature animation (optional plotting extra)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = int(np.sqrt(len(cloud.points)))
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(states[0].reshape(n, n).T, origin="lower", extent=(0, 1, 0, 1),
                      vmin=float(states.min()), vmax=float(states.max()), cmap="inferno")
    ax.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=ax, label="temperature")

    def frame(i):
        image.set_data(states[i].reshape(n, n).T)
        ax.set_title(f"t = {i*dt:.3f}")
        return (image,)

    movie = FuncAnimation(fig, frame, frames=len(states), interval=130, blit=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(str(output), writer=PillowWriter(fps=8))
    plt.close(fig)




def plot_diagnostics(cloud, rbf_matrix, rbf_final, truth, time, output):
    """Save a labeled cloud, matrix pattern, computed field and nodal error."""
    import matplotlib.pyplot as plt

    n = int(np.sqrt(len(cloud.points)))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    interior, boundary = cloud.interior_indices, cloud.boundary_indices
    axes[0, 0].scatter(*cloud.points[interior].T, s=14, label="interior unknown")
    axes[0, 0].scatter(*cloud.points[boundary].T, s=22, label="Dirichlet boundary")
    axes[0, 0].set(xlabel="x", ylabel="y", title="Point cloud / boundary labels",
                   aspect="equal", xlim=(-0.08, 1.08), ylim=(-0.08, 1.08))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].spy(rbf_matrix, markersize=2)
    axes[0, 1].set(xlabel="source node", ylabel="target row",
                   title=f"RBF-FD Laplacian: {rbf_matrix.nnz} nonzeros")
    image = axes[1, 0].imshow(rbf_final.reshape(n, n).T, origin="lower",
        extent=(0, 1, 0, 1), cmap="inferno")
    axes[1, 0].set(xlabel="x", ylabel="y", title=f"RBF-FD temperature at t={time:g}")
    fig.colorbar(image, ax=axes[1, 0], label="temperature", shrink=.85)
    error = np.abs(rbf_final-truth)
    image = axes[1, 1].imshow(error.reshape(n, n).T, origin="lower",
        extent=(0, 1, 0, 1), cmap="magma", vmin=0)
    axes[1, 1].set(xlabel="x", ylabel="y", title="Absolute nodal error")
    fig.colorbar(image, ax=axes[1, 1], label="absolute error", shrink=.85)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def refinement_study(kappa=1.0, final_time=0.05):
    """Separate spatial error from time error using a matrix-exponential reference."""
    print("spatial study: semidiscrete RBF-FD versus continuous exact solution")
    for cells in (4, 6, 8):
        cloud = rbf.geometry.unit_box_grid(cells)
        lap, _ = rbf_fd_laplacian(cloud)
        interior = cloud.interior_indices
        lap_ii = lap[interior, :][:, interior]
        initial = exact(cloud.points[interior], 0, kappa)
        semidiscrete = expm_multiply(final_time*kappa*lap_ii, initial)
        target = exact(cloud.points[interior], final_time, kappa)
        print(f"  cells={cells}, nodes={len(cloud.points)}, "
              f"spatial_max_error={np.max(np.abs(semidiscrete-target)):.6e}")

    cells = 8
    cloud = rbf.geometry.unit_box_grid(cells)
    lap, _ = rbf_fd_laplacian(cloud)
    interior = cloud.interior_indices
    reference = expm_multiply(final_time*kappa*lap[interior, :][:, interior],
                              exact(cloud.points[interior], 0, kappa))
    print("time study: backward Euler versus the fixed-cloud semidiscrete solution")
    for steps in (2, 4, 8):
        dt = final_time/steps
        states, _ = march(lap, cloud, dt=dt, steps=steps, kappa=kappa)
        print(f"  dt={dt:.6f}, steps={steps}, "
              f"time_max_error={np.max(np.abs(states[-1, interior]-reference)):.6e}")


def run(cells=6, dt=0.01, steps=5, *, kappa=1.0, scheme="backward_euler",
        nonzero_boundary=False, plot=None, figure=None):
    """Compare direct RBF-FD time marching with symbolic RBF-FD assembly."""
    cloud = rbf.geometry.unit_box_grid(cells)
    rbf_matrix, ops = rbf_fd_laplacian(cloud)
    rbf_states, rbf_factors = march(rbf_matrix, cloud, dt=dt, steps=steps, kappa=kappa,
                                    scheme=scheme, nonzero_boundary=nonzero_boundary)
    trajectory = symbolic_solution(cloud, dt=dt, steps=steps, kappa=kappa,
                                   scheme=scheme, nonzero_boundary=nonzero_boundary)
    final_truth = exact(cloud.points, dt*steps, kappa, nonzero_boundary)
    symbolic_values = np.asarray(trajectory.final.evaluate(cloud.points), dtype=float).reshape(-1)
    errors = {
        "rbf_fd_matrix": float(np.max(np.abs(rbf_states[-1]-final_truth))),
        "symbolic_rbffd": float(np.max(np.abs(symbolic_values-final_truth))),
    }
    print(f"nodes={len(cloud.points)} interior={len(cloud.interior_indices)} "
          f"boundary={len(cloud.boundary_indices)} stencil={min(20, len(cloud.points))}")
    print(f"L_rbf={rbf_matrix.shape}, "
          f"L_rbf_nnz={rbf_matrix.nnz}, local_factorizations={ops.diagnostics['factorizations']}")
    print(f"time_scheme={scheme}, dt={dt}, steps={steps}, "
          f"matrix_factorizations={rbf_factors}, final_nodal_max_error={errors}")
    if figure is not None:
        plot_diagnostics(cloud, rbf_matrix, rbf_states[-1], final_truth, dt*steps, figure)
        print(f"figure={figure}")
    if plot is not None:
        plot_states(cloud, rbf_states, dt, plot)
        print(f"animation={plot}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--scheme", choices=("backward_euler", "bdf2"), default="backward_euler")
    parser.add_argument("--nonzero-boundary", action="store_true")
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--study", action="store_true")
    args = parser.parse_args()
    run(args.cells, args.dt, args.steps, scheme=args.scheme,
        nonzero_boundary=args.nonzero_boundary, plot=args.plot, figure=args.figure)
    if args.study:
        refinement_study()
