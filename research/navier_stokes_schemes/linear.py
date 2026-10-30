"""Sparse linear solves; reuse a factorization only as a preconditioner."""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, gmres, splu, spsolve
from time import perf_counter


class LinearSolver:
    def __init__(self, method='direct', *, ordering='COLAMD'):
        if method not in ('direct', 'gmres', 'gmres_scaled'):
            raise ValueError('Choose direct, gmres or gmres_scaled')
        self.method = method
        self.ordering = ordering
        self.equilibrate = method == 'gmres_scaled'
        self.factor = None
        self.key = None
        self.row_scale = self.column_scale = None
        self.iterations = 0
        self.factorizations = 0
        self.residual_max = 0.
        self.solves = []

    def solve(self, matrix, rhs, *, key=None):
        """Solve the current equations; key marks a new preconditioner family.

        In scaled mode solve R A C y = R b and return x=C y. R and C
        are positive diagonal matrices, frozen while key is unchanged.
        This equation scaling is independent of local kernel h-scaling.
        """
        before = perf_counter()
        if self.method == 'direct':
            answer = spsolve(matrix, rhs)
            self.factorizations += 1
            self.residual_max = max(self.residual_max,
                                    float(np.linalg.norm(matrix @ answer-rhs, np.inf)))
            return answer
        matrix = matrix.tocsc()
        if key != self.key:
            self.factor = None
            self.row_scale = self.column_scale = None
            self.key = key
        if self.equilibrate and self.row_scale is None:
            row_max = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
            self.row_scale = 1/np.where(row_max > 0, row_max, 1.)
            scaled = diags(self.row_scale) @ matrix
            column_max = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
            self.column_scale = 1/np.where(column_max > 0, column_max, 1.)
        if self.equilibrate:
            system = (diags(self.row_scale) @ matrix @ diags(self.column_scale)).tocsc()
            load = self.row_scale*rhs
        else:
            system, load = matrix, rhs
        factor_seconds, info, inner_iterations = 0., 0, 0

        def factorize():
            nonlocal factor_seconds
            start = perf_counter()
            self.factor = splu(system, permc_spec=self.ordering)
            factor_seconds += perf_counter()-start
            self.factorizations += 1

        if self.factor is None:
            factorize()
            answer = self.factor.solve(load)
        else:
            preconditioner = LinearOperator(system.shape, matvec=self.factor.solve, dtype=system.dtype)
            def count(_):
                nonlocal inner_iterations
                inner_iterations += 1
                self.iterations += 1
            answer, info = gmres(system, load, M=preconditioner,
                                 rtol=1e-14, atol=1e-14 if self.equilibrate else 1e-11,
                                 restart=40, maxiter=5,
                                 callback=count, callback_type='pr_norm')
            # Refresh and solve directly if the iterative solve is unresolved.
            if info != 0:
                factorize()
                answer = self.factor.solve(load)
        transformed_error = system @ answer-load
        scaled_residual = float(np.linalg.norm(transformed_error, np.inf))
        if self.equilibrate:
            answer = self.column_scale*answer
        original_error = matrix @ answer-rhs
        residual = float(np.linalg.norm(original_error, np.inf))
        if not np.isfinite(answer).all() or not np.isfinite(residual):
            raise FloatingPointError('Nonfinite linear solution')
        self.residual_max = max(self.residual_max, residual)
        self.solves.append(dict(key=key, gmres_info=int(info), iterations=inner_iterations,
                                fallback=bool(info != 0), factor_seconds=factor_seconds,
                                seconds=perf_counter()-before, residual_inf=residual,
                                residual_l2=float(np.linalg.norm(original_error)),
                                transformed_residual_inf=scaled_residual,
                                transformed_residual_l2=float(np.linalg.norm(transformed_error)),
                                transformed_threshold=max(1e-14 if self.equilibrate else 1e-11,
                                                          1e-14*float(np.linalg.norm(load))),
                                factor_nnz=int(self.factor.L.nnz+self.factor.U.nnz)))
        return answer
