"""Experimental target-frozen Oseen Hermite closure, using public APIs.

The unknown L here is Oseen momentum, unlike the Stokes L in other schemes.
At target i, O_i = -nu*laplacian + b_i dot grad + grad(p).  The public
functional API uses constant coefficients. Its observation at source j is
therefore L_j + (b_i-b_j) dot grad(u_j), not L_j alone. Independent unknowns
X,Y close the derivatives through the SAME rebuilt mixed interpolation.
This costs four vector fields and a public local assembly call per target;
MPI is passed through problem.options, even for those small target batches.
No convergence or stability claim is implied by this experimental closure.
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, eye, vstack
from scipy.sparse.linalg import spsolve

import rbflab as r


def corrected_source(momentum, dx, dy, advector, target_velocity):
    """Evaluate the target-frozen source data from global Oseen momentum."""
    difference = np.asarray(target_velocity)-np.asarray(advector).reshape(-1, 2)
    return (np.asarray(momentum) + np.repeat(difference[:, 0], 2)*dx
            + np.repeat(difference[:, 1], 2)*dy)


def assemble(problem, advector):
    """Rebuild Oseen representers and Oseen/derivative/pressure row maps."""
    points = np.asarray(problem.points)
    velocity = np.asarray(advector, dtype=float).reshape(-1, 2)
    if velocity.shape != points.shape or not np.isfinite(velocity).all():
        raise ValueError("advector must contain one finite vector per interior node")
    names = ("L", "X", "Y", "P")
    rows = {name: [[None]*len(points) for _ in range(5)] for name in names}
    groups = {}
    for i, b in enumerate(velocity):
        groups.setdefault(tuple(b), []).append(i)
    completed = 0
    print(f"Adaptive Hermite rebuilding {len(points)} targets ({len(groups)} frozen operators)", flush=True)
    for frozen_velocity, indices in groups.items():
        b = np.asarray(frozen_velocity)
        stencil = problem.stencil
        if stencil.indices is not None:
            stencil = r.Stencil(stencil.points, indices=[stencil.indices[i] for i in indices])
        momentum = (-problem.nu*r.laplacian(problem.u)
                    + float(b[0])*r.dx(problem.u) + float(b[1])*r.dy(problem.u)
                    + r.gradient(problem.p))
        source = {
            "velocity": r.Samples(points, operator=problem.u, within=stencil),
            "boundary": r.Samples(problem.boundary_points, operator=problem.u,
                                  within=stencil),
            "momentum": r.Samples(points, operator=momentum,
                                  within=stencil, target="exclude"),
        }
        local = r.LocalApproximation(source=source,
                                     trial=problem.space.representers(source),
                                     **problem.options)
        maps = local.operators(targets=points[indices], operators={
            "L": momentum, "X": r.dx(problem.u), "Y": r.dy(problem.u),
            "P": r.gradient(problem.p)})
        cx = diags(np.repeat(b[0]-velocity[:, 0], 2))
        cy = diags(np.repeat(b[1]-velocity[:, 1], 2))
        for name in names:
            op = getattr(maps, name)
            value, boundary, observation = (op[key].to_scipy()
                                            for key in ("velocity", "boundary", "momentum"))
            for destination, block in zip(rows[name], (value, boundary, observation,
                                                       observation@cx, observation@cy)):
                for j, i in enumerate(indices):
                    destination[i] = block[2*j:2*j+2]
        previous_completed = completed
        completed += len(indices)
        if completed//5 > previous_completed//5 or completed == len(points):
            print(f"Adaptive Hermite rebuilt {completed}/{len(points)} targets", flush=True)
    return {name: tuple(vstack(blocks, format="csr") for blocks in row)
            for name, row in rows.items()}


def system(maps, alpha, history_force, boundary):
    """Return the coupled system in unknown order U,L,X,Y."""
    n = maps["L"][0].shape[0]
    identity = eye(n, format="csr")
    zero = csr_matrix((n, n))
    blocks = [[alpha*identity, identity, zero, zero]]
    rhs = [np.asarray(history_force)]
    for name, position in (("L", 1), ("X", 2), ("Y", 3)):
        value, wall, observation, correction_x, correction_y = maps[name]
        row = [-value, -observation, -correction_x, -correction_y]
        row[position] = row[position]+identity
        blocks.append(row)
        rhs.append(wall@boundary)
    return bmat(blocks, format="csc"), np.concatenate(rhs)


def evolve(problem, scheme, dt, steps):
    """BE startup and extrapolated-advector BDF2; rebuild every time step."""
    if scheme not in (7, "adaptive_hermite", "oseen_hermite"):
        raise ValueError(f"Unknown adaptive Hermite scheme: {scheme}")
    if not np.isfinite(dt) or dt <= 0 or steps < 1 or int(steps) != steps:
        raise ValueError("dt must be positive and steps a positive integer")
    U = np.asarray(problem.initial()[0], dtype=float).copy()
    previous = None
    records = []
    for step in range(1, int(steps)+1):
        t = step*dt
        alpha = (1 if step == 1 else 1.5)/dt
        history = U/dt if step == 1 else (2*U-.5*previous)/dt
        advector = U if step == 1 else 2*U-previous
        maps = assemble(problem, advector)
        g = problem.boundary(t)
        matrix, rhs = system(maps, alpha, history+problem.force(t), g)
        state = spsolve(matrix, rhs)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite adaptive Hermite state at step {step}")
        new_u, L, X, Y = np.split(state, 4)
        value, wall, observation, cx, cy = maps["P"]
        gradient = value@new_u + wall@g + observation@L + cx@X + cy@Y
        residual = np.linalg.norm(matrix@state-rhs, np.inf)
        tolerance = 1e-9*max(1., np.linalg.norm(rhs, np.inf))
        if not np.isfinite(gradient).all() or residual > tolerance:
            raise RuntimeError(f"Adaptive Hermite solve failed at step {step}: residual={residual:g}")
        records.append({"step": step, "time": t,
                        "linear_residual_inf": float(residual),
                        "linear_residual_tolerance": float(tolerance),
                        "weight_rebuilds": len(problem.points),
                        "unknown_vector_fields": 4})
        previous, U = U, new_u
    return {"U": U, "L": L, "dx": X, "dy": Y,
            "gradient": np.asarray(gradient), "diagnostics": records,
            "momentum_convention": "Oseen with extrapolated advector"}
