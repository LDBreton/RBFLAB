"""Private arbitrary-precision kernels, factorizations and data evaluation."""
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal
from functools import lru_cache
import warnings
import numpy as np
import mpmath
import sympy as sp
from .kernels import _X, _Y, _C, _COORDINATES, _base_expression, PHS, Hybrid


@dataclass(frozen=True)
class Precision:
    """Select arithmetic used by the numerical methods.

    Args:
        local_digits: Decimal digits for local weights; `None` uses Float64.
        global_dtype: `"float64"` or `"mpmath"` for supported sparse solves.
        global_digits: Decimal digits for supported dense global solves."""
    local_digits: int | None = None
    global_dtype: str = "float64"
    global_digits: int | None = None

    def __post_init__(self):
        if self.local_digits is not None and (
            type(self.local_digits) is not int or self.local_digits < 16
        ):
            raise ValueError("local_digits must be None or an integer >= 16")
        if self.global_digits is not None and (type(self.global_digits) is not int or self.global_digits < 16):
            raise ValueError("global_digits must be None or an integer >=16")
        if self.global_dtype not in ("float64","mpmath"):
            raise NotImplementedError("global_dtype must be float64 or mpmath")
        if self.global_dtype=="mpmath" and self.local_digits is None:
            raise ValueError("mpmath sparse LHI requires local_digits")


def mp_number(ctx, value):
    """Strings/Fractions retain exact input; floats retain their binary value."""
    if isinstance(value, sp.Expr):
        if value.free_symbols or value.is_real is not True or value.is_finite is not True:
            raise ValueError("Expected a finite real symbolic constant")
        if isinstance(value, (sp.Rational, sp.Float)):
            exact = sp.Rational(value)
            return ctx.mpf(int(exact.p)) / int(exact.q)
        return ctx.mpf(str(value.evalf(ctx.dps + 10)))
    if isinstance(value, Decimal):
        value = str(value)
    if isinstance(value, Fraction):
        return ctx.mpf(value.numerator) / value.denominator
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        value = int(value)
    return ctx.mpf(value)


@lru_cache(maxsize=128)
def _expression(family, *alpha):
    expr = _base_expression(family, len(alpha))
    return sp.diff(expr, *sum(([v,n] for v,n in zip(_COORDINATES,alpha)), []))


class MPBackend:
    def __init__(self, kernel, digits):
        self.ctx = mpmath.mp.clone()
        self.ctx.dps = digits
        self.kernel = kernel
        self.c = mp_number(self.ctx, getattr(kernel,"c",1))
        self.functions = {}

    def derivative(self, *args):
        *coordinates, alpha = args
        return self._kernel_derivative(self.kernel,coordinates,tuple(alpha))

    def _kernel_derivative(self,kernel,coordinates,alpha):
        if len(alpha) not in (2,3) or len(coordinates)!=len(alpha) or any(type(i) is not int or i<0 for i in alpha):
            raise ValueError("Invalid derivative multi-index")
        if sum(alpha)>6:
            raise NotImplementedError("Derivatives through order six are supported")
        from .symbolic_kernel import BoundKernel
        if isinstance(kernel,BoundKernel):return kernel.mp_derivative(self.ctx,coordinates,alpha)
        if isinstance(kernel,Hybrid):
            return (self._kernel_derivative(kernel.smooth,coordinates,alpha)
                    +mp_number(self.ctx,kernel.weight)*self._kernel_derivative(kernel.phs,coordinates,alpha))
        if isinstance(kernel,PHS):
            if sum(alpha)>kernel.power-1:
                raise ValueError(f"PHS({kernel.power}) supports continuous derivatives only through order {kernel.power-1}")
            if all(v==0 for v in coordinates):
                return self.ctx.zero
        key=(kernel.family,alpha)
        if key not in self.functions:
            self.functions[key]=sp.lambdify(
                (*_COORDINATES[:len(alpha)],_C),_expression(kernel.family,*alpha),
                modules=[{"mpf":self.ctx.mpf,"sqrt":self.ctx.sqrt,"exp":self.ctx.exp},"mpmath"],cse=True)
        return self.functions[key](*coordinates,mp_number(self.ctx,getattr(kernel,"c",1)))

    def matrix(self, x, left, y, right):
        from .operators import bind_operators
        left, right = bind_operators(left,x), bind_operators(right,y)
        ctx = self.ctx
        dimensions = {len(point) for point in list(x)+list(y)}
        if len(dimensions) != 1 or next(iter(dimensions)) not in (2,3):
            raise ValueError("Point dimensions must match")
        dimension = next(iter(dimensions))
        if any(op.dimension != dimension for op in list(left)+list(right)):
            raise ValueError("Operator and point dimensions must match")
        xx = [[mp_number(ctx, z) for z in point] for point in x]
        yy = [[mp_number(ctx, z) for z in point] for point in y]
        out = ctx.matrix(len(xx), len(yy))
        for i, point in enumerate(xx):
            for j, center in enumerate(yy):
                # Subtract after conversion: no float64 intermediate displacements.
                displacement = [v-w for v,w in zip(point,center)]
                terms = []
                for a, ca in left[i].terms:
                    for b, cb in right[j].terms:
                        alpha = tuple(v+w for v,w in zip(a,b))
                        terms.append(mp_number(ctx, ca)*mp_number(ctx, cb)
                                     * (-1)**sum(b)*self.derivative(*displacement, alpha))
                out[i, j] = ctx.fsum(terms)
        return out


class MPFactor:
    """Equilibrated pivoted LU, reused for original and transposed local solves."""
    def __init__(self, backend, matrix, general=False, compute_condition=True):
        self.backend, self.ctx, self.matrix = backend, backend.ctx, matrix
        ctx, n = self.ctx, matrix.rows
        if matrix.rows != matrix.cols or any(not ctx.isfinite(v) for v in matrix):
            raise ValueError("Expected a finite square matrix")
        magnitudes = [max(abs(matrix[i,j]) for j in range(n)) if general else abs(matrix[i,i]) for i in range(n)]
        if any(v == 0 for v in magnitudes):
            raise np.linalg.LinAlgError("Zero equation or functional diagonal")
        self.scale = [1/ctx.sqrt(v) for v in magnitudes]
        self.scaled = ctx.matrix(n)
        for i in range(n):
            for j in range(n):
                self.scaled[i, j] = self.scale[i]*matrix[i, j]*self.scale[j]
        try:
            self.P, self.L, self.U = ctx.lu(self.scaled)
        except ZeroDivisionError as exc:
            raise np.linalg.LinAlgError(
                "System singular at requested precision; change discretization or increase digits"
            ) from exc
        # Infinity-norm condition number computed at requested precision.
        if type(compute_condition) is not bool:raise TypeError("compute_condition must be bool")
        self.condition_mp = self.condition = None
        if compute_condition:
            inverse = self._scaled_solve(ctx.eye(n))
            self.condition_mp = ctx.mnorm(self.scaled, "inf")*ctx.mnorm(inverse, "inf")
            self.condition = float(self.condition_mp)
        if self.condition_mp is not None and self.condition_mp * ctx.eps > ctx.mpf("1e-4"):
            warnings.warn("Condition number consumes most requested precision",
                          RuntimeWarning, stacklevel=2)

    def _scaled_solve(self, rhs, transpose=False):
        ctx, n = self.ctx, self.matrix.rows
        out = ctx.matrix(n, rhs.cols)
        if not transpose:
            b = self.P*rhs
            for k in range(rhs.cols):
                z = [ctx.zero]*n
                for i in range(n):
                    z[i] = b[i,k]-ctx.fsum(self.L[i,j]*z[j] for j in range(i))
                for i in range(n-1,-1,-1):
                    out[i,k] = (z[i]-ctx.fsum(self.U[i,j]*out[j,k]
                                            for j in range(i+1,n)))/self.U[i,i]
        else:
            z, v = ctx.matrix(n,rhs.cols), ctx.matrix(n,rhs.cols)
            for k in range(rhs.cols):
                for i in range(n):
                    z[i,k] = (rhs[i,k]-ctx.fsum(self.U[j,i]*z[j,k]
                                               for j in range(i)))/self.U[i,i]
                for i in range(n-1,-1,-1):
                    v[i,k] = z[i,k]-ctx.fsum(self.L[j,i]*v[j,k] for j in range(i+1,n))
            out = self.P.T*v
        return out

    def solve(self, rhs, transpose=False):
        ctx = self.ctx
        if hasattr(rhs, "rows"):
            b = ctx.matrix(rhs)
        else:
            array = np.asarray(rhs, dtype=object)
            b = ctx.matrix([[mp_number(ctx,v) for v in row] for row in array]) if array.ndim == 2 else ctx.matrix([mp_number(ctx,v) for v in array])
        for i in range(b.rows):
            for j in range(b.cols):
                b[i,j] *= self.scale[i]
        out = self._scaled_solve(b, transpose)
        for i in range(out.rows):
            for j in range(out.cols):
                out[i,j] *= self.scale[i]
        return out

    def residual(self, weights, rhs, transpose=True):
        ctx = self.ctx
        matrix = self.matrix.T if transpose else self.matrix
        numerator = ctx.norm(matrix*weights-rhs, "inf")
        denominator = ctx.norm(rhs, "inf")
        return numerator/denominator if denominator else numerator


def local_weights(kernel, points, ops, target, target_operator, precision, backend=None, round_weights=True):
    backend = backend or MPBackend(kernel, precision.local_digits)
    gram = backend.matrix(points, ops, points, ops)
    factor = MPFactor(backend, gram)
    q = backend.matrix(target, [target_operator], points, ops).T
    weights_mp = factor.solve(q, transpose=True)
    if not round_weights:
        ctx=backend.ctx
        residual=factor.residual(weights_mp,q)
        return factor,None,weights_mp,float(residual),{
            "local_digits":precision.local_digits,"condition_norm":"infinity",
            "condition_decimal":ctx.nstr(factor.condition_mp,12),
            "local_residual_decimal":ctx.nstr(residual,12),"weights_rounded":False,
        }
    weights = np.array([float(w) for w in weights_mp])
    if not np.isfinite(weights).all():
        raise FloatingPointError("Local weights cannot be represented in float64")
    rounded = backend.ctx.matrix([mp_number(backend.ctx,w) for w in weights])
    ctx = backend.ctx
    denominator = ctx.norm(weights_mp,"inf")
    delta = ctx.norm(weights_mp-rounded,"inf")
    diagnostics = {
        "local_digits": precision.local_digits,
        "condition_norm": "infinity",
        "condition_decimal": ctx.nstr(factor.condition_mp, 12),
        "local_residual_decimal": ctx.nstr(factor.residual(weights_mp,q), 12),
        "rounded_weight_residual": float(factor.residual(rounded,q)),
        "relative_weight_rounding": float(delta/denominator if denominator else delta),
    }
    return factor, weights, weights_mp, float(factor.residual(weights_mp,q)), diagnostics



@dataclass(frozen=True)
class PrecisionData:
    """Dual data callback: vectorized Float64 and scalar extended(ctx, *coordinates)."""
    float64: object
    extended: object

    def __call__(self, points):
        return self.float64(points)

    def mp_values(self, ctx, points):
        return [self.extended(ctx, *(mp_number(ctx, v) for v in p))
                for p in points]


def mp_values(data, points, ctx):
    """Explicit high-precision callback, exact constant, or documented float fallback."""
    if hasattr(data, "mp_values"):
        out = [mp_number(ctx,v) for v in data.mp_values(ctx,points)]
        if len(out) != len(points):
            raise ValueError("Extended data must return one value per point")
    elif callable(data):
        from .problems import values
        out = [mp_number(ctx,v) for v in values(data,points)]
    else:
        if np.ndim(data) == 0:
            out = [mp_number(ctx,data)]*len(points)
        else:
            out = [mp_number(ctx,v) for v in data]
            if len(out) != len(points):
                raise ValueError("Data must return one value per point")
    if not all(ctx.isfinite(v) for v in out):
        raise ValueError("Nonfinite extended data")
    return out
