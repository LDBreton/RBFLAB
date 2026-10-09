"""One heat problem expressed symbolically and with explicit RBF-FD matrices."""
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import rbflab as rbf


def run():
    # --8<-- [start:settings]
    cloud = rbf.unit_box_grid(6)  # 49 nodes, including the boundary
    dt, steps, kappa = .01, 5, 1.
    method = rbf.RBFFD(rbf.PHS(5), stencil_size=20, polynomial_degree=2)
    # --8<-- [end:settings]
    # --8<-- [start:symbolic]
    model = rbf.SymbolicScalar(2, transient=True)
    u, t = model.field, model.time
    x, y = model.coordinates
    initial = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    problem = model.evolution(
        sp.Eq(sp.diff(u, t)-kappa*model.laplacian(u), 0),
        initial=initial,
        boundary=[model.bc("boundary", sp.Eq(u, 0))],
    )
    trajectory = problem.solve(cloud, method, str(dt), steps, scheme="bdf2")
    symbolic_values = trajectory.final.evaluate(cloud.points)
    # --8<-- [end:symbolic]
    # --8<-- [start:operators]
    ops = method.operators(source=cloud.points, operators={"lap": rbf.Laplacian()})
    matrix = ops.lap.matrix.tocsr()
    I, B = cloud.interior_indices, cloud.boundary_indices
    lap_ii, lap_ib = matrix[I, :][:, I], matrix[I, :][:, B]
    # --8<-- [end:operators]
    # --8<-- [start:factors]
    identity = sparse.eye(len(I), format="csc")
    first_step = splu((identity-dt*kappa*lap_ii).tocsc())
    later_steps = splu((1.5*identity-dt*kappa*lap_ii).tocsc())
    X = cloud.points
    initial_values = np.sin(np.pi*X[:, 0])*np.sin(np.pi*X[:, 1])
    previous = initial_values[I].copy()
    older = None
    states = [initial_values.copy()]
    # --8<-- [end:factors]
    # --8<-- [start:loop]
    for step in range(1, steps+1):
        time = step*dt
        g = np.zeros(len(B))  # prescribed values at this time
        f = np.zeros(len(I))  # forcing at interior nodes at this time
        source = kappa*(lap_ib @ g) + f
        if step == 1:
            current = first_step.solve(previous + dt*source)
        else:
            current = later_steps.solve(2*previous - .5*older + dt*source)
        full = np.empty(len(X))
        full[I], full[B] = current, g
        states.append(full)
        older, previous = previous, current
    states = np.asarray(states)
    # --8<-- [end:loop]
    result = {"route_difference": float(np.max(np.abs(states[-1]-symbolic_values))),
              "error": float(np.max(np.abs(states[-1]-np.exp(-2*kappa*np.pi**2*steps*dt)*initial_values)))}
    print(f"states={states.shape}, final time={steps*dt}, interior matrix={lap_ii.shape}")
    print(result)
    return result


if __name__ == "__main__":
    run()
