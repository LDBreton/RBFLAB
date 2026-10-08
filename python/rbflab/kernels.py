"""Smooth radial kernels and cached symbolic Cartesian derivative evaluators."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import sympy as sp
from .operators import Identity
from fractions import Fraction

_X, _Y, _Z, _C = sp.symbols("x y z c", real=True)
_COORDINATES = (_X, _Y, _Z)


def _base_expression(family, dimension=2):
    s = sum(v**2 for v in _COORDINATES[:dimension])
    if family == "imq":
        return (1 + _C*s)**sp.Rational(-1, 2)
    if family == "gaussian":
        return sp.exp(-_C*s)
    if family.startswith("phs"):
        power = int(family[3:])
        return (-1)**((power+1)//2) * s**sp.Rational(power, 2)
    raise ValueError("Unknown scalar kernel family")


def validate_polynomial_degree(kernel, degree):
    minimum = getattr(kernel, "minimum_degree", -1)
    if minimum >= 0 and (degree is None or degree < minimum):
        raise ValueError(f"This kernel requires polynomial_degree >= {minimum}")


@lru_cache(maxsize=128)
def _derivative(family, *alpha):
    coordinates = _COORDINATES[:len(alpha)]
    expr = _base_expression(family, len(alpha))
    return sp.lambdify((*coordinates, _C), sp.diff(expr, *sum(([v, n] for v, n in zip(coordinates, alpha)), [])),
                      modules="numpy", cse=True)


@dataclass(frozen=True)
class ScalarKernel:
    c: float = 1.0
    family: str = "imq"

    def __post_init__(self):
        if self.family not in ("imq", "gaussian"):
            raise ValueError("Supported kernels: imq, gaussian")
        if not np.isfinite(float(self.c)) or Fraction(self.c) <= 0:
            raise ValueError("c must be finite and positive")

    def derivative(self, displacement, alpha=None):
        z = np.asarray(displacement, dtype=float)
        if z.ndim < 1 or z.shape[-1] not in (2, 3) or not np.isfinite(z).all():
            raise ValueError("Displacements must have finite last dimension 2 or 3")
        alpha = (0,) * z.shape[-1] if alpha is None else tuple(alpha)
        if len(alpha) != z.shape[-1] or any(type(i) is not int or i < 0 for i in alpha):
            raise ValueError("Invalid derivative multi-index")
        if sum(alpha) > 6:
            raise NotImplementedError("This milestone supports derivatives through order six")
        value = _derivative(self.family, *alpha)(*(z[..., i] for i in range(z.shape[-1])), float(self.c))
        return np.broadcast_to(value, z.shape[:-1])

    def matrix(self, x, y, left=None, right=None):
        """left acts on x; right acts on source y (odd derivatives change sign)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] not in (2, 3) or y.shape[1] != x.shape[1]:
            raise ValueError("Points must have matching shapes (N, dimension), dimension 2 or 3")
        left, right = left or Identity(x.shape[1]), right or Identity(x.shape[1])
        if left.dimension != x.shape[1] or right.dimension != x.shape[1]:
            raise ValueError("Operator and point dimensions must match")
        from .operators import SpatialOperator
        if isinstance(left,SpatialOperator) or isinstance(right,SpatialOperator):
            from .assembly import functional_matrix
            return functional_matrix(self,x,[left]*len(x),y,[right]*len(y))
        z = x[:, None, :] - y[None, :, :]
        out = np.zeros((len(x), len(y)))
        for a, ca in left.terms:
            for b, cb in right.terms:
                alpha = tuple(u+v for u,v in zip(a,b))
                out += float(ca) * float(cb) * (-1)**sum(b) * self.derivative(z, alpha)
        return out


def IMQ(c=1.0):
    """Legacy convention: (1 + c*r**2)**(-1/2)."""
    return ScalarKernel(c, "imq")


def Gaussian(c=1.0):
    return ScalarKernel(c, "gaussian")


@dataclass(frozen=True)
class PHS:
    """Signed odd polyharmonic spline, conditionally positive definite.

    phi(r)=(-1)**((power+1)//2) * r**power, power in {3,5,7,9}.
    Requires polynomial degree >= (power-1)//2. No shape parameter.
    """
    power: int = 5

    def __post_init__(self):
        if type(self.power) is not int or self.power not in (3,5,7,9):
            raise ValueError("PHS power must be 3, 5, 7 or 9")

    @property
    def family(self):
        return f"phs{self.power}"

    @property
    def minimum_degree(self):
        return (self.power-1)//2

    def derivative(self, displacement, alpha=None):
        z=np.asarray(displacement,dtype=float)
        if z.ndim<1 or z.shape[-1] not in (2,3) or not np.isfinite(z).all():
            raise ValueError("Expected finite 2D or 3D displacements")
        alpha = (0,) * z.shape[-1] if alpha is None else tuple(alpha)
        if len(alpha)!=z.shape[-1] or any(type(i) is not int or i<0 for i in alpha):
            raise ValueError("Invalid derivative multi-index")
        if sum(alpha)>min(6,self.power-1):
            raise ValueError(f"PHS({self.power}) supports continuous derivatives only through order {min(6,self.power-1)}")
        # D^alpha phi is homogeneous of degree power-|alpha|. Evaluate
        # the symbolic expression at bounded coordinates, then restore scale.
        # Direct evaluation near zero can overflow inverse powers of r even
        # when the final derivative is representable (e.g. D_x^6 r^7).
        scale=np.max(np.abs(z),axis=-1)
        zero=scale==0
        safe_scale=np.where(zero,1.0,scale)
        normalized=z/safe_scale[...,None]
        coordinates=[np.where(zero, float(i==0), normalized[...,i])
                     for i in range(z.shape[-1])]
        value=_derivative(self.family,*alpha)(*coordinates,1.0)
        # A result below Float64 range may legitimately underflow to zero.
        with np.errstate(under="ignore"):
            value=value*safe_scale**(self.power-sum(alpha))
        return np.where(zero,0.0,np.broadcast_to(value,z.shape[:-1]))

    matrix = ScalarKernel.matrix


@dataclass(frozen=True)
class Hybrid:
    """smooth(r) + weight * signed_PHS(r); requires PHS polynomial augmentation.

    Negative weights require explicit allow_indefinite=True for historical
    experiments; positive-definiteness guarantees do not apply to that sum.
    Distances are physical coordinates; weight is dimensional and must be
    chosen consistently when rescaling geometry.
    """
    smooth: ScalarKernel
    phs: PHS
    weight: object = "0.01"
    allow_indefinite: bool = False

    def __post_init__(self):
        if not isinstance(self.smooth,ScalarKernel) or not isinstance(self.phs,PHS):
            raise TypeError("Hybrid requires an IMQ/Gaussian kernel and PHS")
        if type(self.allow_indefinite) is not bool:raise TypeError("allow_indefinite must be bool")
        if not np.isfinite(float(self.weight)) or Fraction(self.weight)==0:
            raise ValueError("Hybrid weight must be finite and nonzero")
        if Fraction(self.weight)<0 and not self.allow_indefinite:
            raise ValueError("Negative hybrid weights require allow_indefinite=True; intended for legacy experiments")

    @property
    def minimum_degree(self):
        return self.phs.minimum_degree

    def derivative(self,displacement,alpha=None):
        return self.smooth.derivative(displacement,alpha)+float(self.weight)*self.phs.derivative(displacement,alpha)

    matrix = ScalarKernel.matrix


@dataclass(frozen=True)
class DivergenceFree:
    """2D Hessian(psi) - I*Laplacian(psi); output (Nx, Ny, 2, 2)."""
    scalar: ScalarKernel

    def matrix(self, x, y):
        if np.asarray(x).ndim != 2 or np.asarray(y).ndim != 2 or np.asarray(x).shape[1] != 2 or np.asarray(y).shape[1] != 2:
            raise NotImplementedError("DivergenceFree vector kernels currently support 2D only")
        z = np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]
        out = np.empty(z.shape[:-1] + (2, 2))
        out[..., 0, 0] = -self.scalar.derivative(z, (0, 2))
        out[..., 1, 1] = -self.scalar.derivative(z, (2, 0))
        out[..., 0, 1] = out[..., 1, 0] = self.scalar.derivative(z, (1, 1))
        return out
