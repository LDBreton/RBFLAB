"""Shared functional assembly and equilibrated dense solves."""
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning


def functional_matrix(kernel, x, left, y, right):
    from .operators import bind_operators
    left, right = bind_operators(left,x), bind_operators(right,y)
    out = np.empty((len(x), len(y)))
    lg, rg = {}, {}
    for i, op in enumerate(left):
        lg.setdefault(op, []).append(i)
    for i, op in enumerate(right):
        rg.setdefault(op, []).append(i)
    if len(lg)>8 or len(rg)>8:
        # Point-dependent coefficients otherwise produce one tiny kernel call
        # per point pair. Group derivative indices and multiply coefficient
        # vectors outside the vectorized kernel blocks instead.
        from .operators import Operator
        out.fill(0)
        terms=[]
        for operators in (left,right):
            groups={}
            for i,op in enumerate(operators):
                for alpha,c in op.terms:
                    rows,coefficients=groups.setdefault(alpha,([],[]))
                    rows.append(i);coefficients.append(float(c))
            terms.append(groups)
        for alpha,(rows,ca) in terms[0].items():
            for beta,(cols,cb) in terms[1].items():
                block=kernel.matrix(x[rows],y[cols],Operator(((alpha,1),)),Operator(((beta,1),)))
                out[np.ix_(rows,cols)]+=np.asarray(ca)[:,None]*block*np.asarray(cb)[None,:]
        return out
    for opx, rows in lg.items():
        for opy, cols in rg.items():
            out[np.ix_(rows, cols)] = kernel.matrix(x[rows], y[cols], opx, opy)
    return out


class Factor:
    """Diagonal equilibration; no jitter, pseudoinverse or silent regularization."""
    def __init__(self, matrix, general=False, compute_condition=True):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1] or not np.isfinite(self.matrix).all():
            raise ValueError("Expected a finite square matrix")
        diagonal = np.max(np.abs(self.matrix), axis=1) if general else np.abs(np.diag(self.matrix))
        if np.any(diagonal == 0):
            raise np.linalg.LinAlgError("Zero functional diagonal")
        self.scale = 1 / np.sqrt(diagonal)
        self.scaled = self.scale[:, None] * self.matrix * self.scale[None, :]
        if type(compute_condition) is not bool:raise TypeError("compute_condition must be bool")
        self.condition = float(np.linalg.cond(self.scaled)) if compute_condition else None
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                self.lu = lu_factor(self.scaled)
            except LinAlgWarning as exc:
                raise np.linalg.LinAlgError(str(exc)) from exc
        if self.condition is not None and (not np.isfinite(self.condition) or self.condition > 1e15):
            warnings.warn(f"Ill-conditioned equilibrated system: {self.condition:.3e}",
                          RuntimeWarning, stacklevel=2)

    def solve(self, rhs, transpose=False):
        rhs = np.asarray(rhs)
        scale = self.scale if rhs.ndim == 1 else self.scale[:, None]
        return scale * lu_solve(self.lu, scale * rhs, trans=1 if transpose else 0)


def relative_residual(matrix, solution, rhs):
    error = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
    # RHS-relative residual: sensitive to inaccurate local/global solves.
    return float(error / max(np.linalg.norm(rhs, ord=np.inf), np.finfo(float).tiny))
