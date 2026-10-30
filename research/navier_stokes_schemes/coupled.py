"""Coupled mixed-LHI time integrators (research, node-major 2D vectors).

L denotes the Stokes observation, not an independently reconstructed velocity.
Every derivative uses the same (U, boundary, L) mixed interpolation. Startup
is backward Euler; subsequent steps use BDF2. Rotational L uses modified
pressure p + |u|²/2; reported gradients recover physical pressure.
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from .linear import LinearSolver


def _derivatives(problem, U, L, g):
    return (problem.DxU @ U + problem.DxB @ g + problem.DxL @ L,
            problem.DyU @ U + problem.DyB @ g + problem.DyL @ L)


def _transport(velocity):
    return diags(np.repeat(velocity[::2], 2)), diags(np.repeat(velocity[1::2], 2))


def _rotation(omega):
    n = len(omega)
    rows = np.arange(2*n)
    cols = rows ^ 1
    data = np.empty(2*n)
    data[::2], data[1::2] = -omega, omega
    return csr_matrix((data, (rows, cols)), shape=(2*n, 2*n))


def convection(problem, U, L, g, rotational=False, *, jacobian=True):
    """Return convection and, when requested, its Jacobian in (U,L)."""
    dx, dy = _derivatives(problem, U, L, g)
    n = len(U)
    if not jacobian:
        if rotational:
            value = _rotation(dx[1::2]-dy[::2]) @ U
        else:
            value = np.repeat(U[::2], 2)*dx + np.repeat(U[1::2], 2)*dy
        return value, None, None
    if rotational:
        omega = dx[1::2] - dy[::2]
        R = _rotation(omega)
        # omega(U,L) and omega cross U, with node-major vector ordering.
        WU = problem.DxU[1::2, :] - problem.DyU[::2, :]
        WL = problem.DxL[1::2, :] - problem.DyL[::2, :]
        rows = np.arange(n)
        cols = rows//2
        values = np.empty(n)
        values[::2], values[1::2] = -U[1::2], U[::2]
        K = csr_matrix((values, (rows, cols)), shape=(n, n//2))
        return R @ U, R + K @ WU, K @ WL
    Vx, Vy = _transport(U)
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack((2*(np.arange(n)//2), 2*(np.arange(n)//2)+1)).ravel()
    values = np.column_stack((dx, dy)).ravel()
    advector_jacobian = csr_matrix((values, (rows, cols)), shape=(n, n))
    return (Vx @ dx + Vy @ dy,
            Vx @ problem.DxU + Vy @ problem.DyU + advector_jacobian,
            Vx @ problem.DxL + Vy @ problem.DyL)


def evolve(problem, scheme, dt, steps, *, rtol=1e-10, atol=1e-12,
           max_iterations=30):
    """Evolve coupled U,L; fail explicitly when nonlinear residual is unresolved.

    Schemes: oseen (extrapolated advector), implicit (Newton),
    oseen_newton (Oseen predictor plus residual-controlled Newton), rotational
    (fully implicit Newton in vorticity form). Sparse solves are Float64.
    problem.initial_rotational(), when present, supplies modified initial L.
    Otherwise initial() must already use the chosen pressure convention.
    """
    aliases = {"bdf2_oseen": "oseen", "fully_implicit": "implicit",
               "oseen_newton_correction": "oseen_newton"}
    scheme = aliases.get(scheme, scheme)
    if scheme not in {"oseen", "implicit", "oseen_newton", "rotational"}:
        raise ValueError(f"Unknown coupled scheme: {scheme}")
    if not np.isfinite(dt) or dt <= 0 or steps < 0 or int(steps) != steps:
        raise ValueError("dt must be positive and steps a nonnegative integer")
    if rtol < 0 or atol < 0 or max_iterations < 1:
        raise ValueError("Invalid nonlinear convergence controls")
    rotational = scheme == "rotational"
    initializer = getattr(problem, "initial_rotational", problem.initial) if rotational else problem.initial
    U, L = (np.asarray(v, dtype=float).copy() for v in initializer())
    n = len(U)
    I, M, A = eye(n, format="csr"), csr_matrix(problem.M), csr_matrix(problem.A)
    records = []
    predictor_solver = LinearSolver(getattr(problem, 'linear_solver', 'direct'),
                                    ordering=getattr(problem, 'global_ordering', 'COLAMD'))
    Uold = Lold = None
    gradient = problem.PY @ U + problem.PB @ problem.boundary(0) + problem.PL @ L
    if rotational:
        dx, dy = _derivatives(problem, U, L, problem.boundary(0))
        v = U.reshape(-1, 2)
        gradient -= np.column_stack((np.sum(v*dx.reshape(-1, 2), axis=1),
                                     np.sum(v*dy.reshape(-1, 2), axis=1))).ravel()
    for step in range(1, int(steps)+1):
        t = step*dt
        alpha = (1 if step == 1 else 1.5)/dt
        history = U/dt if step == 1 else (2*U - .5*Uold)/dt
        g, f = problem.boundary(t), problem.force(t)
        top_rhs, rhs = -problem.B @ g, f + history
        advector = U if step == 1 else 2*U-Uold
        Vx, Vy = _transport(advector)
        OU = Vx @ problem.DxU + Vy @ problem.DyU
        OL = Vx @ problem.DxL + Vy @ problem.DyL
        cb = Vx @ (problem.DxB @ g) + Vy @ (problem.DyB @ g)
        if scheme in {"oseen", "oseen_newton"}:
            predictor_matrix = bmat([[A, -M], [alpha*I+OU, I+OL]], format="csc")
            candidate = predictor_solver.solve(
                predictor_matrix, np.concatenate((top_rhs, rhs-cb)),
                key=alpha if predictor_solver.equilibrate else None)
        else:
            # Fully implicit variants start from extrapolated states; only the
            # correction scheme pays for an additional Oseen predictor solve.
            initial_L = L if step == 1 else 2*L-Lold
            candidate = np.concatenate((advector, initial_L))
        iterations = 0
        scale = max(1., np.linalg.norm(np.concatenate((top_rhs, rhs)), np.inf))
        threshold = atol + rtol*scale

        def residual(z):
            u, ell = z[:n], z[n:]
            c, ju, jl = convection(problem, u, ell, g, rotational,
                                   jacobian=scheme != 'oseen')
            return np.concatenate((A @ u-M @ ell-top_rhs,
                                   alpha*u+ell+c-rhs)), ju, jl

        if scheme != "oseen":
            for iteration in range(max_iterations+1):
                residual_vector, JU, JL = residual(candidate)
                residual_norm = np.linalg.norm(residual_vector, np.inf)
                if np.isfinite(residual_norm) and residual_norm <= threshold:
                    break
                if iteration == max_iterations or not np.isfinite(residual_norm):
                    raise RuntimeError(f"{scheme} failed at step {step}: residual={residual_norm:g}")
                jacobian = bmat([[A, -M], [alpha*I+JU, I+JL]], format="csc")
                correction = spsolve(jacobian, -residual_vector)
                # Backtracking guards against an unconverged full Newton jump.
                damping = 1.
                while damping > 2**-16:
                    trial = candidate + damping*correction
                    trial_norm = np.linalg.norm(residual(trial)[0], np.inf)
                    if np.isfinite(trial_norm) and trial_norm < residual_norm:
                        candidate = trial
                        break
                    damping *= .5
                else:
                    raise RuntimeError(f"{scheme} Newton line search failed at step {step}")
                iterations += 1
        if not np.isfinite(candidate).all():
            raise FloatingPointError(f"Nonfinite coupled state at step {step}")
        direct_check = {}
        if scheme == 'oseen' and predictor_solver.method != 'direct' and step == steps:
            reference = spsolve(predictor_matrix, np.concatenate((top_rhs, rhs-cb)))
            delta_u, delta_l = np.split(candidate-reference, 2)
            delta_gradient = problem.PY @ delta_u + problem.PL @ delta_l
            direct_check = dict(direct_check_velocity_max=float(np.max(abs(delta_u))),
                                direct_check_gradient_max=float(np.max(abs(delta_gradient))))
            if (not np.isfinite(reference).all()
                    or direct_check['direct_check_velocity_max'] > 1e-10
                    or direct_check['direct_check_gradient_max'] > 1e-7):
                raise RuntimeError(f'GMRES/direct final-system parity failed: {direct_check}')
        Uold, U = U, candidate[:n]
        Lold, L = L, candidate[n:]
        nonlinear_residual = np.linalg.norm(residual(candidate)[0], np.inf)
        gradient = problem.PY @ U + problem.PB @ g + problem.PL @ L
        if rotational:
            dx, dy = _derivatives(problem, U, L, g)
            velocity, dx, dy = (v.reshape(-1, 2) for v in (U, dx, dy))
            kinetic_gradient = np.column_stack((np.sum(velocity*dx, axis=1),
                                                np.sum(velocity*dy, axis=1))).ravel()
            gradient = gradient-kinetic_gradient
        solved_residual = (predictor_matrix @ candidate-np.concatenate((top_rhs, rhs-cb))
                           if scheme == "oseen" else residual(candidate)[0])
        records.append({"step": step, "time": t, "newton_iterations": iterations,
                        "solved_residual_inf": float(np.linalg.norm(solved_residual, np.inf)),
                        "nonlinear_residual_inf": float(nonlinear_residual),
                        "constraint_residual_inf": float(np.linalg.norm(A @ U-M @ L-top_rhs, np.inf)),
                        "tolerance": float(threshold), **direct_check})
        if step % 50 == 0:
            print(f'{scheme}: step {step}/{steps}, t={t:g}, '
                  f'max |U|={np.max(np.abs(U)):.6g}', flush=True)
    result = {"U": U, "L": L, "gradient": np.asarray(gradient), "diagnostics": records}
    if scheme in {'oseen', 'oseen_newton'}:
        result['linear_solver'] = dict(method=predictor_solver.method,
                                     ordering=predictor_solver.ordering,
                                     iterations=predictor_solver.iterations,
                                     factorizations=predictor_solver.factorizations,
                                     residual_max=predictor_solver.residual_max,
                                     solves=predictor_solver.solves)
    return result
