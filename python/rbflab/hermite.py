"""Polynomial-augmented Hermite local systems shared with nodal assembly."""
import numpy as np
from .nodal import NodalBasis
from .assembly import relative_residual
from .precision import mp_number


def augmented_local_weights(arithmetic,points,operators,target,target_operator,degree,round_weights=True,kernel_scaling="physical"):
    basis=NodalBasis(arithmetic,points,degree,source_operators=operators,kernel_scaling=kernel_scaling)
    matrix=basis.system_matrix(points,operators)
    factor=arithmetic.factor(matrix)
    q=basis.evaluation(target,[target_operator]).T
    n=len(points)
    if arithmetic.ctx:
        ctx=arithmetic.ctx
        complete=factor.solve(q,transpose=True)
        wmp=complete[:n]
        residual=factor.residual(complete,q)
        rounded=ctx.matrix(complete)
        if round_weights:
            for i in range(n):
                rounded[i]=mp_number(ctx,float(complete[i]))
        physical_norm=ctx.norm(ctx.matrix(wmp),"inf")
        diagnostics={"local_digits":ctx.dps,"condition_norm":"infinity",
            "condition_decimal":ctx.nstr(factor.condition_mp,12),
            "local_residual_decimal":ctx.nstr(residual,12),
            "weights_rounded":round_weights,
            "rounded_weight_residual":float(factor.residual(rounded,q)),
            "relative_weight_rounding":float(ctx.norm(complete-rounded,"inf")/physical_norm) if physical_norm else 0.0}
        w=np.array([float(v) for v in wmp]) if round_weights else None
        if w is not None and not np.isfinite(w).all():
            raise FloatingPointError("Weights cannot be represented in Float64")
    else:
        q=q[:,0]
        complete=factor.solve(q,transpose=True)
        residual=relative_residual(matrix.T,complete,q)
        w,wmp=complete[:n],None
        diagnostics={"local_digits":None,"condition_norm":"2"}
    return basis,factor,w,wmp,float(residual),diagnostics
