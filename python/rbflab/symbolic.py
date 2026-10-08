"""Compile scalar SymPy equations to existing numerical problems.

No string equation parser or dynamic kernel generation is involved. PDE coefficients may depend on spatial coordinates; boundary
coefficients may also depend on supplied outward normals. All unsupported terms fail explicitly.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from .operators import Operator, SpatialOperator
from .precision import mp_number
from .problems import LinearPDE, BoundaryCondition
from .evolution import EvolutionPDE


def _expr(value):
    if isinstance(value,str):
        raise TypeError("Use SymPy expressions, not equation strings; numeric parameter strings are supported")
    result=sp.sympify(value)
    if not isinstance(result,sp.Expr):raise TypeError("Expected a scalar SymPy expression")
    return result


def _number(value):
    result=sp.Rational(value) if isinstance(value,str) else _expr(value)
    if result.free_symbols or result.is_real is not True or result.is_finite is not True:
        raise ValueError("Parameters must be finite real numerical constants")
    return result


@lru_cache(maxsize=128)
def _numpy_function(expression,symbols):
    return sp.lambdify(symbols,expression,modules="numpy",cse=True)


@lru_cache(maxsize=128)
def _mp_function(expression,symbols,ctx):
    # Override the whole mpmath function namespace with the private context.
    # A fallback to global mpmath.sin/exp would silently lose requested digits.
    namespace={name:getattr(ctx,name) for name in dir(ctx) if not name.startswith('_')}
    return sp.lambdify(symbols,expression,modules=[namespace,"mpmath"],cse=True)


@dataclass(frozen=True)
class SymbolicData:
    """Compiled NumPy/MP data; geometric normals are bound by boundary assembly."""
    expression: object
    coordinates: tuple
    normal_symbols: tuple=()
    time_symbol: object=None
    time: object=None
    normals: object=None

    @property
    def requires_normals(self):return bool(self.normal_symbols)

    def bind_normals(self,normals):return replace(self,normals=np.asarray(normals,dtype=float))

    def at_time(self,time):return replace(self,time=time) if self.time_symbol is not None else self

    def _arguments(self,points):
        from .methods import _query
        points=_query(points,len(self.coordinates))
        if self.normal_symbols:
            if self.normals is None or self.normals.shape!=points.shape:
                raise ValueError("Symbolic normal data require matching bound boundary normals")
        if self.time_symbol is not None and self.time is None:
            raise ValueError("Time-dependent symbolic data must be bound with at_time(time)")
        return points

    @property
    def symbols(self):
        return self.coordinates+self.normal_symbols+((self.time_symbol,) if self.time_symbol is not None else ())

    def __call__(self,points):
        points=self._arguments(points)
        args=list(points.T)
        if self.normal_symbols:args.extend(self.normals.T)
        if self.time_symbol is not None:args.append(float(self.time))
        value=_numpy_function(self.expression,self.symbols)(*args)
        return np.broadcast_to(np.asarray(value,dtype=float),(len(points),)).copy()

    def operator_values(self,operator,points,ctx=None):
        """Analytic initial-field derivatives for boundary compatibility checks."""
        expression=0
        for alpha,coefficient in operator.terms:
            term=self.expression
            for coordinate,order in zip(self.coordinates,alpha):
                term=sp.diff(term,coordinate,order)
            expression+=sp.sympify(coefficient)*term
        data=replace(self,expression=expression)
        return data.mp_values(ctx,points) if ctx else data(points)

    def mp_values(self,ctx,points):
        points=self._arguments(points);function=_mp_function(self.expression,self.symbols,ctx)
        result=[]
        for i,point in enumerate(points):
            args=[mp_number(ctx,v) for v in point]
            if self.normal_symbols:args.extend(mp_number(ctx,v) for v in self.normals[i])
            if self.time_symbol is not None:args.append(mp_number(ctx,self.time))
            result.append(function(*args))
        return result


@dataclass(frozen=True)
class SymbolicBoundary:
    on: object
    equation: object


@dataclass(frozen=True)
class _PointBoundaryOperator:
    terms: tuple
    coordinates: tuple
    normal_symbols: tuple

    @property
    def requires_normals(self):
        return any(c.free_symbols.intersection(self.normal_symbols) for _,c in self.terms)

    def at_point(self,point,normal=None):
        return self._bind(tuple(float(v) for v in point),
                          tuple(float(v) for v in normal) if normal is not None else None)

    @lru_cache(maxsize=4096)
    def _bind(self,point,normal):
        mapping={x:sp.Rational(v) for x,v in zip(self.coordinates,point)}
        if self.requires_normals:
            if normal is None:raise ValueError("Boundary normals are required")
            mapping.update({x:sp.Rational(v) for x,v in zip(self.normal_symbols,normal)})
        return Operator(tuple((alpha,coefficient.xreplace(mapping)) for alpha,coefficient in self.terms),len(point))


class SymbolicScalar:
    """A single scalar field in Cartesian 2D or 3D.

    Build equations with SymPy Eq/diff and compile using stationary/evolution.
    Pass transient=True for a field depending on time. The resulting objects
    are ordinary LinearPDE/EvolutionPDE instances accepted by existing methods.
    """
    def __init__(self,dimension=2,*,name="u",transient=False):
        if type(dimension) is not int or dimension not in (2,3):raise ValueError("dimension must be 2 or 3")
        if not isinstance(name,str) or not name:raise ValueError("Field name must be nonempty")
        self.dimension=dimension
        self.coordinates=sp.symbols("x y z",real=True)[:dimension]
        self.normal=sp.symbols("n_x n_y n_z",real=True)[:dimension]
        self.time=sp.Symbol("t",real=True) if transient else None
        args=self.coordinates+((self.time,) if self.time is not None else ())
        self.field=sp.Function(name)(*args)

    def laplacian(self,expression):return sum(sp.diff(expression,x,2) for x in self.coordinates)
    def normal_derivative(self,expression):return sum(n*sp.diff(expression,x) for n,x in zip(self.normal,self.coordinates))
    def bc(self,on,equation):return SymbolicBoundary(on,equation)

    def _parameters(self,parameters):
        mapping={}
        reserved=set(self.coordinates+self.normal+((self.time,) if self.time is not None else ()))
        for symbol,value in (parameters or {}).items():
            if not isinstance(symbol,sp.Symbol) or symbol in reserved:
                raise ValueError("Parameter keys must be symbols distinct from coordinates, normals and time")
            mapping[symbol]=_number(value)
        return mapping

    def _residual(self,equation,mapping):
        if isinstance(equation,sp.Equality):equation=equation.lhs-equation.rhs
        expression=_expr(equation).xreplace(mapping).doit()
        if expression.has(sp.Integral,sp.Sum,sp.Product):
            raise NotImplementedError("Only local differential expressions are supported")
        unknown=expression.atoms(AppliedUndef)-{self.field}
        if unknown:raise NotImplementedError(f"Only the declared scalar field is supported; found {unknown}")
        return sp.expand(expression)

    def _linear(self,expression):
        derivatives=sorted(expression.atoms(sp.Derivative),key=sp.default_sort_key)
        slots=[self.field]+derivatives
        alphas=[((0,)*self.dimension,0)]
        for derivative in derivatives:
            if derivative.expr!=self.field:raise NotImplementedError("Unsupported derivative expression")
            alpha=[0]*self.dimension;time_order=0
            for variable,count in derivative.variable_count:
                if variable in self.coordinates:alpha[self.coordinates.index(variable)]+=int(count)
                elif self.time is not None and variable==self.time:time_order+=int(count)
                else:raise ValueError("Derivative variable is not a model coordinate or time")
            alphas.append((tuple(alpha),time_order))
        placeholders=tuple(sp.Dummy() for _ in slots)
        replaced=expression.xreplace(dict(zip(slots,placeholders)))
        try:polynomial=sp.Poly(replaced,*placeholders)
        except sp.PolynomialError as exc:raise ValueError("Equation must be linear in the field and its derivatives") from exc
        if polynomial.total_degree()>1:raise ValueError("Equation must be linear in the field and its derivatives")
        terms=[]
        for index,slot in enumerate(placeholders):
            coefficient=sp.simplify(polynomial.coeff_monomial(slot))
            if coefficient!=0:terms.append((*alphas[index],coefficient))
        return terms,-polynomial.coeff_monomial(1)

    def _data(self,expression,*,timed=False,normals=False):
        expression=sp.simplify(expression)
        if expression.atoms(AppliedUndef) or expression.has(sp.Derivative):
            raise ValueError("Data must be independent of unknown fields and derivatives")
        allowed=set(self.coordinates)
        if normals:allowed.update(self.normal)
        if timed and self.time is not None:allowed.add(self.time)
        unknown=expression.free_symbols-allowed
        if unknown:raise ValueError(f"Unbound or unsupported data symbols: {unknown}")
        if expression.has(sp.I,sp.oo,-sp.oo,sp.zoo,sp.nan):raise ValueError("Data must be real and finite")
        use_normals=self.normal if expression.free_symbols.intersection(self.normal) else ()
        return SymbolicData(expression,self.coordinates,use_normals,self.time if timed else None)

    def data(self,expression,*,parameters=None):
        """Compile field-independent spatial data for interpolation or initial values."""
        return self._data(_expr(expression).xreplace(self._parameters(parameters)).doit())

    def _operator(self,terms):
        unknown=set().union(*(c.free_symbols for _,c in terms))-set(self.coordinates)
        if unknown:
            raise ValueError(f"Unbound parameters or unsupported coefficient symbols: {unknown}")
        if any(c.free_symbols for _,c in terms):
            return SpatialOperator(tuple(terms),self.coordinates)
        return Operator(tuple(terms),self.dimension)

    def _boundaries(self,boundary,mapping,*,timed):
        result=[]
        for spec in boundary:
            if not isinstance(spec,SymbolicBoundary):raise TypeError("Use model.bc(label, equation) for symbolic boundaries")
            terms,rhs=self._linear(self._residual(spec.equation,mapping))
            if not terms:raise ValueError("Boundary equation has no nonzero field operator")
            if any(order for _,order,_ in terms):raise NotImplementedError("Time derivatives in boundary operators are unsupported")
            spatial=[(alpha,c) for alpha,_,c in terms]
            allowed=set(self.coordinates+self.normal)
            if any(c.free_symbols-allowed for _,c in spatial):
                raise ValueError("Boundary operator has unbound parameters or time-dependent coefficients")
            # Normalize constant weighted value equations, including transient Dirichlet.
            if len(spatial)==1 and not any(spatial[0][0]) and not spatial[0][1].free_symbols:
                coefficient=spatial[0][1]
                _number(coefficient)
                rhs=rhs/coefficient;spatial=[((0,)*self.dimension,sp.Integer(1))]
            if any(c.free_symbols for _,c in spatial):
                operator=_PointBoundaryOperator(tuple(spatial),self.coordinates,self.normal)
            else:operator=self._operator(spatial)
            result.append(BoundaryCondition(spec.on,operator,self._data(rhs,timed=timed,normals=True)))
        return tuple(result)

    def operator(self,expression,*,parameters=None):
        """Compile a homogeneous spatial expression for low-level numerical use."""
        terms,rhs=self._linear(self._residual(expression,self._parameters(parameters)))
        if rhs!=0:raise ValueError("Operator expressions cannot include forcing or independent terms")
        if any(order for _,order,_ in terms):raise NotImplementedError("Use evolution for time derivatives")
        if not terms:raise ValueError("Expected a nonzero operator expression")
        return self._operator([(alpha,c) for alpha,_,c in terms])

    def stationary(self,equation,*,boundary,parameters=None):
        if self.time is not None:raise ValueError("Use a non-transient model for stationary equations")
        mapping=self._parameters(parameters)
        terms,rhs=self._linear(self._residual(equation,mapping))
        if any(order for _,order,_ in terms):raise NotImplementedError("Stationary equations cannot contain time derivatives")
        operator=self._operator([(alpha,c) for alpha,_,c in terms])
        return LinearPDE(operator,self._data(rhs),self._boundaries(boundary,mapping,timed=False))

    def evolution(self,equation,*,initial,boundary,parameters=None):
        if self.time is None:raise ValueError("Evolution needs SymbolicScalar(transient=True)")
        mapping=self._parameters(parameters)
        terms,rhs=self._linear(self._residual(equation,mapping))
        time_coefficient=sp.Integer(0);spatial=[]
        for alpha,order,coefficient in terms:
            if order:
                if order!=1 or any(alpha):raise NotImplementedError("Only a first time derivative of the field is supported")
                time_coefficient+=coefficient
            else:spatial.append((alpha,coefficient))
        if time_coefficient==0:raise ValueError("Evolution equation must contain a nonzero first time derivative")
        _number(time_coefficient)
        operator=self._operator([(alpha,sp.simplify(c/time_coefficient)) for alpha,c in spatial])
        initial_data=self._data(_expr(initial).xreplace(mapping).doit())
        return EvolutionPDE(operator,self._data(rhs/time_coefficient,timed=True),initial_data,
                            self._boundaries(boundary,mapping,timed=True))
