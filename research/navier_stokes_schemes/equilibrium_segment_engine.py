"""Segmented exact copy of coupled.evolve; only history export/import added.
Original source SHA256 45dfb1b6417e88a5d84cd3248238fd483af364dd5780791c8bb63deb2034dbde
"""
import numpy as np
from scipy.sparse import bmat,csr_matrix,diags,eye
from scipy.sparse.linalg import spsolve
from .coupled import _derivatives,_transport,_rotation,convection
from .linear import LinearSolver
def evolve(problem, scheme, dt, steps, *, rtol=1e-10, atol=1e-12,
           max_iterations=30, _history=None):
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
    start_step = 1
    if _history is not None:
        Uold, Lold, start_step = _history
    gradient = problem.PY @ U + problem.PB @ problem.boundary(0) + problem.PL @ L
    if rotational:
        dx, dy = _derivatives(problem, U, L, problem.boundary(0))
        v = U.reshape(-1, 2)
        gradient -= np.column_stack((np.sum(v*dx.reshape(-1, 2), axis=1),
                                     np.sum(v*dy.reshape(-1, 2), axis=1))).ravel()
    for step in range(start_step, start_step+int(steps)):
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
        if scheme == 'oseen' and predictor_solver.method != 'direct' and step == start_step+int(steps)-1:
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
                        "perturbation_velocity_inf": float(np.max(abs(U-getattr(problem,"_stability_reference",U)))),
                        "perturbation_kinetic_mean": float(.5*np.mean(np.sum((U-getattr(problem,"_stability_reference",U)).reshape(-1,2)**2,axis=1))),
                        "solved_residual_inf": float(np.linalg.norm(solved_residual, np.inf)),
                        "nonlinear_residual_inf": float(nonlinear_residual),
                        "constraint_residual_inf": float(np.linalg.norm(A @ U-M @ L-top_rhs, np.inf)),
                        "tolerance": float(threshold), **direct_check})
        if step % 50 == 0:
            print(f'{scheme}: step {step}/{start_step+int(steps)-1}, t={t:g}, '
                  f'max |U|={np.max(np.abs(U)):.6g}', flush=True)
    result = {"U": U, "L": L, "_history": (Uold,Lold,start_step+int(steps)), "gradient": np.asarray(gradient), "diagnostics": records}
    if scheme in {'oseen', 'oseen_newton'}:
        result['linear_solver'] = dict(method=predictor_solver.method,
                                     ordering=predictor_solver.ordering,
                                     iterations=predictor_solver.iterations,
                                     factorizations=predictor_solver.factorizations,
                                     residual_max=predictor_solver.residual_max,
                                     solves=predictor_solver.solves)
    return result
