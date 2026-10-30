"""Consistent mixed-LHI IMEX and characteristic time integrators.

The algebraic momentum observation is L, with M L = A U + B g.
Every convection evaluation includes the momentum derivative maps. IMEX
midpoint combines an implicit-Euler half predictor with explicit midpoint
convection and Crank--Nicolson Stokes evolution (second order).
"""
import numpy as np
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import factorized


def evolve(problem, scheme, dt, steps):
    """Advance; characteristic boundary exits need an explicit extension.

    ``problem.evaluate(U,L,t,points)`` must use the same mixed reconstruction.
    Optional ``bounds`` and ``exterior_velocity(points,t)`` support prescribed
    manufactured continuation outside the physical domain. No extrapolated
    boundary value is silently replaced by interior exact solution data.
    """
    if not np.isfinite(dt) or dt <= 0 or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("dt must be finite and positive and steps a nonnegative integer")
    aliases = {4: "imex", 6: "characteristic", "imex_midpoint": "imex",
               "characteristics": "characteristic", "characteristic_bdf2": "characteristic_bdf2"}
    name = aliases.get(scheme, scheme)
    if name not in ("imex", "imex_ars222", "characteristic", "characteristic_bdf2"):
        raise ValueError(f"Unknown transport scheme {scheme!r}")
    p = problem
    M, A, B = p.M, p.A, p.B
    cache = {}

    def solve(alpha, rhs):
        if alpha not in cache:
            cache[alpha] = factorized(bmat([[alpha*eye(A.shape[0]), eye(A.shape[0])], [-A, M]], format="csc"))
        state = cache[alpha](np.asarray(rhs).ravel())
        return np.split(state, 2)

    def convection(u, ell, t):
        g = p.boundary(t)
        dx = (p.DxU @ u + p.DxB @ g + p.DxL @ ell).reshape(-1, 2)
        dy = (p.DyU @ u + p.DyB @ g + p.DyL @ ell).reshape(-1, 2)
        velocity = u.reshape(-1, 2)
        return (velocity[:, :1]*dx + velocity[:, 1:]*dy).ravel()

    outside_count = 0

    def value(u, ell, t, x):
        nonlocal outside_count
        bounds = getattr(p, "bounds", None)
        outside = np.zeros(len(x), dtype=bool)
        if bounds is not None:
            bounds = np.asarray(bounds)
            outside = np.any((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=1)
        result = np.empty((len(x), 2))
        if np.any(~outside):
            result[~outside] = np.asarray(p.evaluate(u, ell, t, x[~outside])).reshape(-1, 2)
        if np.any(outside):
            extension = getattr(p, "exterior_velocity", None)
            if extension is None:
                raise ValueError("Characteristic foot left domain; provide exterior_velocity")
            result[outside] = np.asarray(extension(x[outside], t)).reshape(-1, 2)
            outside_count += int(outside.sum())
        return result

    U, L = (np.asarray(v).copy() for v in p.initial())
    previous = None
    residual = 0.0
    for k in range(steps):
        t, tn = k*dt, (k+1)*dt
        if name == "imex_ars222":
            # ARS(2,2,2): explicit nodes 0,gamma,1; implicit SDIRK
            # nodes gamma,1. The last row is the final state in both
            # tableaux, retaining the algebraic momentum constraint.
            gamma = 1 - 1/np.sqrt(2)
            delta = 1 - 1/(2*gamma)
            alpha = 1/(gamma*dt)
            ts = t + gamma*dt
            c0 = convection(U, L, t)
            stage, stage_l = solve(alpha, np.r_[
                alpha*U + p.force(ts) - c0, B @ p.boundary(ts)])
            cs = convection(stage, stage_l, ts)
            rhs = (alpha*U + ((1-gamma)/gamma)*(p.force(ts)-stage_l)
                   + p.force(tn) - (delta/gamma)*c0
                   - ((1-delta)/gamma)*cs)
            new, new_l = solve(alpha, np.r_[rhs, B @ p.boundary(tn)])
        elif name == "imex":
            half = t + dt/2
            predictor, lp = solve(2/dt, np.r_[2*U/dt + p.force(half) - convection(U, L, t), B @ p.boundary(half)])
            rhs = 2*U/dt - L + 2*(p.force(half) - convection(predictor, lp, half))
            new, new_l = solve(2/dt, np.r_[rhs, B @ p.boundary(tn)])
        else:
            x = np.asarray(p.points)
            v = value(U, L, t, x)
            if name == "characteristic_bdf2" and previous is not None:
                up, lp, tp = previous
                start = 2*v - value(up, lp, tp, x)
                midpoint = x - .5*dt*start
                vm = 1.5*value(U, L, t, midpoint) - .5*value(up, lp, tp, midpoint)
                foot = x - dt*vm
                foot2 = x - 2*dt*value(U, L, t, x-dt*start)
                history = (2*value(U, L, t, foot).ravel() - .5*value(up, lp, tp, foot2).ravel())/dt
                alpha = 1.5/dt
            else:
                foot = x - dt*v
                history = value(U, L, t, foot).ravel()/dt
                alpha = 1/dt
            new, new_l = solve(alpha, np.r_[history + p.force(tn), B @ p.boundary(tn)])
        residual = max(residual, float(np.linalg.norm(M @ new_l - A @ new - B @ p.boundary(tn), np.inf)))
        previous, U, L = (U, L, t), new, new_l
        if not np.isfinite(U).all() or not np.isfinite(L).all():
            raise FloatingPointError(f"Nonfinite transport state at step {k+1}")
        if (k+1) % 100 == 0:
            print(f'{name}: step {k+1}/{steps}, t={tn:g}, '
                  f'max |U|={np.max(np.abs(U)):.6g}', flush=True)
    gradient = p.PY @ U + p.PB @ p.boundary(steps*dt) + p.PL @ L
    return dict(U=U, L=L, gradient=np.asarray(gradient).ravel(), diagnostics={
        "scheme": name, "momentum_residual_max": residual,
        "exterior_evaluations": outside_count, "steps": steps})
