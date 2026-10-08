"""Discrete nodal consistency and small-system stability diagnostics."""
import numpy as np


def nodal_diagnostics(system, exact_unknowns, *, condition_limit=200):
    """Diagnose an assembled LHI/RBF-FD system, before reconstruction.

    Pass exact values in matrix-column order: interior values for LHI, all
    values for RBF-FD. This API does not accept global RBF coefficients.
    Row normalization removes equation magnitude, not physical units.
    A fixed alternating perturbation of the row-normalized RHS measures one
    response direction, not worst-case amplification. The system is factored
    lazily if necessary. Float64 small systems additionally report the row-scaled infinity-norm
    condition; extended systems keep consistency arithmetic extended and omit
    that Float64 condition estimate. No claim of a stability certificate.
    """
    matrix=system.matrix
    if hasattr(matrix,"matvec"):
        ctx=matrix.ctx
        from .precision import mp_number
        exact=ctx.matrix([mp_number(ctx,v) for v in exact_unknowns])
        if any(not ctx.isfinite(v) for v in exact):
            raise ValueError("Exact values must be finite")
        if len(exact)!=len(system.rhs):
            raise ValueError("Expected one exact value per unknown")
        defect=matrix.matvec(exact)-system.rhs
        norm=lambda x:ctx.norm(x,"inf")
        scales=[max((abs(v) for v in row.values()),default=ctx.zero) for row in matrix.rows]
        if any(v==0 for v in scales):
            raise ValueError("Cannot normalize an empty equation")
        from .sparse_precision import MPSparseLU
        factor=system.factor or MPSparseLU(matrix)
        response=factor.solve(ctx.matrix([scale*(-1)**i for i,scale in enumerate(scales)]))
        normalized=ctx.matrix([defect[i]/scale for i,scale in enumerate(scales)])
        return {"consistency_inf":float(norm(defect)),
                "row_scaled_consistency_inf":float(norm(normalized)),
                "consistency_decimal":ctx.nstr(norm(defect),16),
                "digits":ctx.dps,"row_scaled_condition_inf":None,
                "alternating_rhs_gain_inf":float(norm(response))}
    exact=np.asarray(exact_unknowns,dtype=float)
    if exact.shape!=(matrix.shape[1],) or not np.isfinite(exact).all():
        raise ValueError("Expected one finite exact value per unknown")
    defect=matrix@exact-system.rhs
    scales=np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(scales==0):
        raise ValueError("Cannot normalize an empty equation")
    from scipy.sparse.linalg import splu
    factor=system.factor or splu(matrix)
    response=factor.solve(scales*(-1.)**np.arange(len(scales)))
    condition=None
    if matrix.shape[0]<=condition_limit:
        condition=float(np.linalg.cond(matrix.toarray()/scales[:,None],np.inf))
    return {"consistency_inf":float(np.max(abs(defect))),
            "row_scaled_consistency_inf":float(np.max(abs(defect)/scales)),
            "digits":None,"row_scaled_condition_inf":condition,
            "alternating_rhs_gain_inf":float(np.max(abs(response)))}
