"""Navier--Stokes as a control residual constrained by backward-Euler Stokes.

The control enters the physical momentum equation with a minus sign. For
each control trajectory, one forward Stokes sweep produces (U,L), where L
is the locally eliminated mixed Stokes observation. A reverse-time sweep
with the *transpose* of the same discrete block gives the exact discrete
adjoint. No symmetry of the LHI maps is assumed.

Only the interior nodal velocity is the advector. Its derivatives use the
same mixed reconstruction, including boundary data and momentum samples.
This module is research code, not a new public RBFLAB solver API.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import LinearOperator, gmres, splu


class _StokesBlockFactor:
    """Exact Schur solves of K and K.T using one smaller sparse factor."""
    def __init__(self, A, M, dt, factorization=None):
        self.M, self.dt, self.n = M, dt, A.shape[0]
        self.factorization = (splu((A+M/dt).tocsc(), permc_spec="MMD_AT_PLUS_A")
                              if factorization is None else factorization)

    def solve(self, rhs, trans="N"):
        a, b = np.split(rhs, 2)
        if trans == "N":
            U = self.factorization.solve(a+self.M@b)
            return np.r_[U, b-U/self.dt]
        if trans == "T":
            first = self.factorization.solve(a-b/self.dt, trans="T")
            return np.r_[first, b+self.M.T@first]
        raise ValueError("Only forward and transpose Stokes solves are supported")


class StokesControl:
    """A fixed time window with prescribed boundary values and initial state.

    ``problem`` supplies A, M, B, DxU, DxL, DxB, DyU, DyL, DyB and the
    callables boundary(t), force(t), initial(). Vectors are node-major 2D.
    Controls and residuals have shape (steps, 2*interior_nodes). States have
    shape (steps, 4*interior_nodes), with U then L in each row.

    ``node_weights`` are positive spatial quadrature weights. The default
    is a normalized nodal mean; pass physical quadrature weights for a
    spatial integral. Time quadrature is the backward-Euler rectangle rule.
    """

    def __init__(self, problem, dt, steps, *, start_time=0., initial=None,
                 node_weights=None, factorization=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not np.isfinite(start_time):
            raise ValueError("start_time must be finite")
        self.problem, self.dt, self.steps = problem, float(dt), int(steps)
        self.start_time = float(start_time)
        initial = problem.initial() if initial is None else initial
        self.initial_U, self.initial_L = (np.asarray(v, dtype=float).ravel().copy()
                                          for v in initial)
        self.n = n = len(self.initial_U)
        if n == 0 or n % 2 or len(self.initial_L) != n:
            raise ValueError("Expected equally sized node-major 2D U and L")
        if not np.isfinite(np.r_[self.initial_U, self.initial_L]).all():
            raise ValueError("Initial observations must be finite")
        for name in ("A", "M", "DxU", "DxL", "DyU", "DyL"):
            if getattr(problem, name).shape != (n, n):
                raise ValueError(f"Unexpected {name} shape")
        weights = (np.full(n//2, 1/(n//2)) if node_weights is None
                   else np.asarray(node_weights, dtype=float))
        if weights.shape != (n//2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("node_weights must contain one positive finite weight per node")
        self.node_weights = weights.copy()
        self.weights = self.dt*np.repeat(weights, 2)
        self.sqrt_weights = np.sqrt(self.weights)
        # Positive scaling affects optimizer tolerances, never the minimizer.
        self.measure = self.steps*float(np.sum(self.weights))
        self.times = self.start_time+self.dt*np.arange(1, self.steps+1)
        self.boundaries = [np.asarray(problem.boundary(t), dtype=float).ravel()
                           for t in self.times]
        self.forces = np.asarray([np.asarray(problem.force(t), dtype=float).ravel()
                                  for t in self.times])
        if self.forces.shape != (self.steps, n) or not np.isfinite(self.forces).all():
            raise ValueError("Physical force has invalid dimensions or nonfinite values")
        for g in self.boundaries:
            if len(g) != problem.B.shape[1] or not np.isfinite(g).all():
                raise ValueError("Boundary data have invalid dimensions or nonfinite values")
        self.boundary_rhs = np.asarray([-problem.B@g for g in self.boundaries])
        self.boundary_dx = np.asarray([problem.DxB@g for g in self.boundaries])
        self.boundary_dy = np.asarray([problem.DyB@g for g in self.boundaries])
        I = eye(n, format="csc")
        self.matrix = bmat([[problem.A, -problem.M], [I/self.dt, I]], format="csc")
        self.lu = _StokesBlockFactor(problem.A, problem.M, self.dt, factorization)
        self.factorization = self.lu.factorization
        self.factorizations = int(factorization is None)
        self.forward_sweeps = self.adjoint_sweeps = 0

    def reset_window(self, initial, start_time):
        """Reuse this window and its LU with new initial data and absolute time.

        The mesh, maps, dt, window length and quadrature are unchanged. This
        is useful for per-step control solves on large meshes: no new sparse
        factorization is needed. Boundary and forcing data are refreshed.
        """
        U, L = (np.asarray(v, dtype=float).ravel().copy() for v in initial)
        if len(U) != self.n or len(L) != self.n or not np.isfinite(np.r_[U, L]).all():
            raise ValueError("Invalid new initial observations")
        if not np.isfinite(start_time):
            raise ValueError("start_time must be finite")
        self.initial_U, self.initial_L = U, L
        self.start_time = float(start_time)
        self.times = self.start_time+self.dt*np.arange(1, self.steps+1)
        p = self.problem
        self.boundaries = [np.asarray(p.boundary(t), dtype=float).ravel() for t in self.times]
        self.forces = np.asarray([np.asarray(p.force(t), dtype=float).ravel() for t in self.times])
        if self.forces.shape != (self.steps, self.n) or not np.isfinite(self.forces).all():
            raise ValueError("Invalid new physical force")
        for g in self.boundaries:
            if len(g) != p.B.shape[1] or not np.isfinite(g).all():
                raise ValueError("Invalid new boundary data")
        self.boundary_rhs = np.asarray([-p.B@g for g in self.boundaries])
        self.boundary_dx = np.asarray([p.DxB@g for g in self.boundaries])
        self.boundary_dy = np.asarray([p.DyB@g for g in self.boundaries])
        return self

    def _controls(self, controls):
        value = np.asarray(controls, dtype=float)
        if value.size != self.steps*self.n:
            raise ValueError("Control trajectory has the wrong size")
        value = value.reshape(self.steps, self.n)
        if not np.isfinite(value).all():
            raise ValueError("Controls must be finite")
        return value

    def forward(self, controls):
        """Solve forward Stokes with b-f; reuse the same sparse LU each step."""
        controls = self._controls(controls)
        states = np.empty((self.steps, 2*self.n))
        previous = self.initial_U
        for k in range(self.steps):
            rhs = np.r_[self.boundary_rhs[k],
                        self.forces[k]-controls[k]+previous/self.dt]
            states[k] = self.lu.solve(rhs)
            previous = states[k, :self.n]
        self.forward_sweeps += 1
        if not np.isfinite(states).all():
            raise FloatingPointError("Nonfinite forward Stokes trajectory")
        return states

    def derivatives(self, states):
        """Differentiate the same mixed (U,g,L) field at its arrival nodes."""
        states = np.asarray(states).reshape(self.steps, 2*self.n)
        U, L, p = states[:, :self.n], states[:, self.n:], self.problem
        dx = (p.DxU@U.T+p.DxL@L.T).T+self.boundary_dx
        dy = (p.DyU@U.T+p.DyL@L.T).T+self.boundary_dy
        return dx, dy

    def convection(self, states):
        dx, dy = self.derivatives(states)
        U = np.asarray(states)[:, :self.n]
        return np.repeat(U[:, ::2], 2, axis=1)*dx+np.repeat(U[:, 1::2], 2, axis=1)*dy

    def convection_jvp(self, states, dstates):
        """Exact derivative of convection, including momentum observations."""
        U = np.asarray(states).reshape(self.steps, 2*self.n)[:, :self.n]
        dstates = np.asarray(dstates).reshape(self.steps, 2*self.n)
        dU, dL, p = dstates[:, :self.n], dstates[:, self.n:], self.problem
        dx, dy = self.derivatives(states)
        ddx, ddy = (p.DxU@dU.T+p.DxL@dL.T).T, (p.DyU@dU.T+p.DyL@dL.T).T
        return (np.repeat(dU[:, ::2], 2, axis=1)*dx
                +np.repeat(dU[:, 1::2], 2, axis=1)*dy
                +np.repeat(U[:, ::2], 2, axis=1)*ddx
                +np.repeat(U[:, 1::2], 2, axis=1)*ddy)

    def convection_vjp(self, states, dual):
        """Euclidean transpose of convection_jvp, not a rediscretized PDE."""
        U = np.asarray(states).reshape(self.steps, 2*self.n)[:, :self.n]
        dual = np.asarray(dual).reshape(self.steps, self.n)
        dx, dy = self.derivatives(states)
        p = self.problem
        vx = np.repeat(U[:, ::2], 2, axis=1)*dual
        vy = np.repeat(U[:, 1::2], 2, axis=1)*dual
        barU = (p.DxU.T@vx.T+p.DyU.T@vy.T).T
        barL = (p.DxL.T@vx.T+p.DyL.T@vy.T).T
        barU[:, ::2] += (dual*dx).reshape(self.steps, -1, 2).sum(axis=2)
        barU[:, 1::2] += (dual*dy).reshape(self.steps, -1, 2).sum(axis=2)
        return np.concatenate((barU, barL), axis=1)

    def tangent(self, direction):
        """Control-to-state derivative; boundary and initial variations are zero."""
        direction = self._controls(direction)
        tangent = np.empty((self.steps, 2*self.n))
        previous = np.zeros(self.n)
        for k in range(self.steps):
            tangent[k] = self.lu.solve(np.r_[np.zeros(self.n),
                                           -direction[k]+previous/self.dt])
            previous = tangent[k, :self.n]
        return tangent

    def adjoint(self, state_source):
        """Forward sweep in reversed time, with zero terminal recurrence data.

        K.T lambda[k] = source[k] + H.T lambda[k+1]. Here H transmits
        only old velocity into the next physical momentum equation. Using
        SuperLU's transpose solve is essential for nonsymmetric LHI maps.
        """
        state_source = np.asarray(state_source).reshape(self.steps, 2*self.n)
        multipliers = np.empty_like(state_source)
        next_multiplier = np.zeros(2*self.n)
        for reverse_step in range(self.steps):
            k = self.steps-1-reverse_step
            rhs = state_source[k].copy()
            rhs[:self.n] += next_multiplier[self.n:]/self.dt
            multipliers[k] = self.lu.solve(rhs, trans="T")
            next_multiplier = multipliers[k]
        self.adjoint_sweeps += 1
        return multipliers

    def evaluate(self, controls, *, gradient=True):
        controls = self._controls(controls)
        states = self.forward(controls)
        residual = controls-self.convection(states)
        weighted_residual = self.weights*residual
        result = dict(states=states, residual=residual,
                      cost=float(.5*np.sum(residual*weighted_residual)),
                      residual_inf=float(np.max(abs(residual))))
        if gradient:
            self._add_gradient(result)
        return result

    def _add_gradient(self, result):
        """Complete an accepted forward evaluation without solving it again."""
        weighted_residual = self.weights*result['residual']
        multipliers = self.adjoint(self.convection_vjp(result['states'], weighted_residual))
        derivative = weighted_residual+multipliers[:, self.n:]
        result.update(gradient=derivative, l2_gradient=derivative/self.weights,
                      adjoint=multipliers)

    def residual_jacobian(self, controls, *, states=None, weighted=True, normalized=False):
        """Matrix-free exact Jacobian and transpose of the control residual."""
        states = self.forward(controls) if states is None else np.asarray(states).copy()
        scale = self.sqrt_weights if weighted else np.ones(self.n)
        if normalized:
            scale = scale/np.sqrt(self.measure)
        def matvec(direction):
            direction = np.asarray(direction).reshape(self.steps, self.n)
            value = direction-self.convection_jvp(states, self.tangent(direction))
            return (scale*value).ravel()
        def rmatvec(dual):
            dual = scale*np.asarray(dual).reshape(self.steps, self.n)
            multipliers = self.adjoint(self.convection_vjp(states, dual))
            return (dual+multipliers[:, self.n:]).ravel()
        size = self.steps*self.n
        return LinearOperator((size, size), matvec=matvec, rmatvec=rmatvec, dtype=float)

    def solve(self, initial_controls=None, *, method="lbfgs", maxiter=200,
              gtol=1e-10, residual_tolerance=1e-8, on_iteration=None,
              linear_maxiter=200):
        """Minimize the same objective with a positive constant normalization.

        The normalization is J/measure and is recorded explicitly. Success
        requires the actual unscaled residual, not an optimizer status flag.
        ``gauss_newton`` uses matrix-free trust-region least squares/LSMR;
        ``newton_krylov`` solves the square residual Newton equation with
        GMRES and uses its exact adjoint gradient in an Armijo line search.
        For nonsingular square Jacobians this is the Gauss--Newton direction
        without forming normal equations. ``lbfgs`` also uses the adjoint.
        """
        if method not in ("lbfgs", "gauss_newton", "newton_krylov"):
            raise ValueError("Unknown control optimizer")
        if maxiter < 1 or gtol <= 0 or residual_tolerance <= 0:
            raise ValueError("Invalid optimization controls")
        guess = (np.zeros((self.steps, self.n)) if initial_controls is None
                 else self._controls(initial_controls).copy())
        history = []
        latest = {}
        recorded_controls = None
        def value(x):
            nonlocal latest
            if not latest or not np.array_equal(latest['controls'], x):
                latest = dict(controls=x.copy(), **self.evaluate(x))
            return latest
        def record(x):
            nonlocal recorded_controls
            if recorded_controls is not None and np.array_equal(recorded_controls, x):
                return
            current = value(x)
            row = dict(iteration=len(history), cost=current['cost'],
                       residual_inf=current['residual_inf'],
                       gradient_inf=float(np.max(abs(current['gradient']))))
            for key in ('linear_iterations', 'linear_relative_residual', 'line_search_step',
                        'directional_derivative', 'linear_info'):
                if key in current:
                    row[key] = current[key]
            history.append(row)
            recorded_controls = x.copy()
            if on_iteration is not None:
                on_iteration(row, current)
        record(guess.ravel())
        if history[-1]['residual_inf'] <= residual_tolerance:
            result = dict(optimizer_success=True, optimizer_message="Initial residual satisfies tolerance",
                          optimizer_iterations=0)
            controls = guess
        elif method == "newton_krylov":
            x = guess.ravel()
            iterations = 0
            message = "Maximum control iterations reached"
            for _ in range(maxiter):
                current = value(x)
                if current['residual_inf'] <= residual_tolerance:
                    message = "Physical control residual satisfies tolerance"
                    break
                operator = self.residual_jacobian(x, states=current['states'], weighted=False)
                krylov_steps = []
                direction, info = gmres(operator, -current['residual'].ravel(),
                                        rtol=1e-8, atol=0., restart=min(30, len(x)),
                                        maxiter=linear_maxiter, callback=krylov_steps.append,
                                        callback_type='legacy')
                linear_relative = float(np.linalg.norm(operator@direction+current['residual'].ravel())
                                        /max(np.linalg.norm(current['residual']), np.finfo(float).tiny))
                slope = float(np.dot(current['gradient'].ravel(), direction))
                if info != 0 or not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                    message = f"Residual Newton direction failed: GMRES info={info}, relative residual={linear_relative:g}"
                    break
                accepted = None
                step_length = 1.
                for _backtrack in range(25):
                    trial = x+step_length*direction
                    try:
                        evaluation = self.evaluate(trial, gradient=False)
                    except (FloatingPointError, ValueError):
                        evaluation = None
                    if (evaluation is not None and np.isfinite(evaluation['cost'])
                            and evaluation['cost'] <= current['cost']+1e-4*step_length*slope):
                        accepted = evaluation
                        break
                    step_length *= .5
                if accepted is None:
                    message = "Armijo line search failed"
                    break
                self._add_gradient(accepted)
                x = trial
                latest = dict(controls=x.copy(), **accepted,
                              linear_iterations=len(krylov_steps), linear_info=int(info),
                              linear_relative_residual=linear_relative,
                              line_search_step=step_length, directional_derivative=slope)
                iterations += 1
                record(x)
                if accepted['residual_inf'] <= residual_tolerance:
                    message = "Physical control residual satisfies tolerance"
                    break
            controls = x.reshape(self.steps, self.n)
            result = dict(optimizer_success=value(x)['residual_inf'] <= residual_tolerance,
                          optimizer_message=message, optimizer_iterations=iterations)
        elif method == "lbfgs":
            def objective(x):
                current = value(x)
                return current['cost']/self.measure, current['gradient'].ravel()/self.measure
            optimized = minimize(objective, guess.ravel(), jac=True, method="L-BFGS-B",
                                 callback=record,
                                 options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxls=40))
            controls = optimized.x.reshape(self.steps, self.n)
            result = dict(optimizer_success=bool(optimized.success),
                          optimizer_message=str(optimized.message),
                          optimizer_iterations=int(optimized.nit))
        else:
            def fun(x):
                return (value(x)['residual']*self.sqrt_weights/np.sqrt(self.measure)).ravel()
            def jac(x):
                current = value(x)
                return self.residual_jacobian(x, states=current['states'], normalized=True)
            optimized = least_squares(fun, guess.ravel(), jac=jac, method="trf",
                                      tr_solver="lsmr", max_nfev=maxiter,
                                      gtol=gtol, ftol=1e-13, xtol=1e-13,
                                      tr_options=dict(atol=1e-10, btol=1e-10,
                                                      maxiter=linear_maxiter))
            controls = optimized.x.reshape(self.steps, self.n)
            result = dict(optimizer_success=bool(optimized.success),
                          optimizer_message=str(optimized.message),
                          optimizer_iterations=None, optimizer_function_evaluations=int(optimized.nfev))
        record(controls.ravel())
        final = value(controls.ravel())
        result.update(final)
        result.update(controls=controls, history=history, method=method,
                      converged=final['residual_inf'] <= residual_tolerance,
                      optimizer_objective_scale=1/self.measure,
                      factorizations=self.factorizations, forward_sweeps=self.forward_sweeps,
                      adjoint_sweeps=self.adjoint_sweeps)
        return result
