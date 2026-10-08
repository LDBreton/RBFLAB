"""Constant and spatially varying scalar differential operators in 2D/3D."""
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from fractions import Fraction
from decimal import Decimal
import sympy as sp

_SCALARS = (Real, str, Decimal, sp.Expr)

def _coefficient(value):
    if isinstance(value, sp.Expr):
        if value.free_symbols or value.is_real is not True or value.is_finite is not True:
            raise ValueError("Symbolic coefficients must be finite real constants")
        if isinstance(value, (sp.Rational, sp.Float)):
            rational = sp.Rational(value)
            return Fraction(int(rational.p), int(rational.q))
        return value
    if not isinstance(value, _SCALARS):
        raise TypeError('Only finite real constant coefficients are supported')
    try:
        return Fraction(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Coefficient must be finite') from exc


@dataclass(frozen=True)
class Operator:
    """Constant coefficients times Cartesian multi-index derivatives; @ composes."""
    terms: tuple
    dimension: int | None = None

    def __post_init__(self):
        dimension = self.dimension if self.dimension is not None else (len(self.terms[0][0]) if self.terms else 2)
        _dimension(dimension)
        object.__setattr__(self, "dimension", dimension)
        terms = {}
        for alpha, value in self.terms:
            if len(alpha) != dimension or any(type(i) is not int or i < 0 for i in alpha):
                raise ValueError("Derivative indices must match the operator dimension and be nonnegative integers")
            terms[tuple(alpha)] = terms.get(tuple(alpha), Fraction(0)) + _coefficient(value)
        object.__setattr__(self, "terms",
                           tuple(sorted((a, c) for a, c in terms.items() if c != 0)))

    def __add__(self, other):
        if isinstance(other, _SCALARS):
            other = other * Identity(self.dimension)
        if not isinstance(other, Operator):
            return NotImplemented
        self._compatible(other)
        return Operator(self.terms + other.terms, self.dimension)

    __radd__ = __add__

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, scalar):
        if not isinstance(scalar, _SCALARS):
            return NotImplemented
        scalar = _coefficient(scalar)
        return Operator(tuple((a, scalar * c) for a, c in self.terms), self.dimension)

    __rmul__ = __mul__

    def __matmul__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented
        self._compatible(other)
        return Operator(tuple((tuple(x+y for x,y in zip(a,b)), c*d)
                              for a, c in self.terms for b, d in other.terms), self.dimension)

    def _compatible(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Operator dimensions must match")


def _dimension(dimension):
    if type(dimension) is not int or dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")


def Identity(dimension=2):
    """Return the value functional in the selected spatial dimension."""
    _dimension(dimension)
    return Operator((((0,) * dimension, 1.0),))


def Derivative(axis, dimension=2):
    """Return a first Cartesian derivative operator.

    Args:
        axis (int): Zero-based coordinate axis.
        dimension (int): Number of spatial coordinates."""
    _dimension(dimension)
    if type(axis) is not int or not 0 <= axis < dimension:
        raise ValueError("Derivative axis must lie within the spatial dimension")
    return Operator(((tuple(int(i == axis) for i in range(dimension)), 1),))


def Laplacian(dimension=2):
    """Return the sum of second Cartesian derivatives."""
    _dimension(dimension)
    return sum((Derivative(i, dimension) @ Derivative(i, dimension)
                for i in range(dimension)), 0 * Identity(dimension))


@dataclass(frozen=True)
class NormalDerivative:
    """Expanded with the supplied geometric normal, not differentiated."""
    def at(self, normal):
        return sum((value * Derivative(i, len(normal)) for i, value in enumerate(normal)),
                   0 * Identity(len(normal)))


@dataclass(frozen=True)
class Robin:
    """Boundary functional `alpha*u + beta*normal_derivative(u)`.

    The geometric unit normal is supplied when the functional is bound to a
    boundary point."""
    alpha: float = 1.0
    beta: float = 1.0

    def at(self, normal):
        return self.alpha * Identity(len(normal)) + self.beta * NormalDerivative().at(normal)



@dataclass(frozen=True)
class SpatialOperator:
    """Sum a_alpha(x) D^alpha, bound at physical evaluation/source points.

    Compose/differentiate expressions in SymPy before compiling; binding freezes
    the resulting coefficient values and never differentiates them a second time.
    """
    terms: tuple
    coordinates: tuple

    def __post_init__(self):
        _dimension(len(self.coordinates))
        if len(set(self.coordinates)) != len(self.coordinates) or any(not isinstance(x,sp.Symbol) for x in self.coordinates):
            raise ValueError("Expected distinct coordinate symbols")
        terms={}
        for alpha,value in self.terms:
            if len(alpha)!=self.dimension or any(type(i) is not int or i<0 for i in alpha):
                raise ValueError("Invalid spatial derivative multi-index")
            value=sp.sympify(value)
            if value.free_symbols-set(self.coordinates):raise ValueError("Unbound spatial coefficient symbols")
            if value.has(sp.Derivative,sp.Integral,sp.DiracDelta):raise ValueError("Coefficients must be explicit spatial expressions")
            terms[tuple(alpha)]=terms.get(tuple(alpha),0)+value
        object.__setattr__(self,'terms',tuple((alpha,sp.simplify(value)) for alpha,value in sorted(terms.items()) if sp.simplify(value)!=0))

    @property
    def dimension(self):return len(self.coordinates)

    @property
    def requires_normals(self):return False

    def at_point(self,point,normal=None):
        if len(point)!=self.dimension:raise ValueError("Operator and point dimensions must match")
        return self._bind(tuple(float(v) for v in point))

    @lru_cache(maxsize=8192)
    def _bind(self,point):
        values={x:sp.Rational(v) for x,v in zip(self.coordinates,point)}
        return Operator(tuple((alpha,c.xreplace(values)) for alpha,c in self.terms),self.dimension)


def bind_operators(operators,points):
    """Resolve every spatial functional at its own physical point."""
    if len(operators)!=len(points):raise ValueError("One functional is required per point")
    return [op.at_point(point) if isinstance(op,SpatialOperator) else op for op,point in zip(operators,points)]
