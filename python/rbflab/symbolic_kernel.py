"""Runtime-parameter radial kernel families with verified Cartesian origin jets."""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
import sympy as sp
from .kernels import ScalarKernel, _COORDINATES

@dataclass(frozen=True)
class Kernel:
    """Symbolic radial kernel family with runtime-bound parameters.

    Args:
        expression: SymPy formula in `radial_variable` and declared parameters.
        radial_variable: SymPy symbol representing squared radius `r**2`.
        parameters: Tuple of runtime SymPy parameter symbols.
        minimum_degree: Required minimum polynomial degree, or `-1`.
        dimension: Optional restriction to 2D or 3D.

    Call the family with named parameter values to obtain a bound kernel.
    `compile(...)` caches a validated native implementation when available."""
    expression: object
    radial_variable: object
    parameters: tuple = ()
    minimum_degree: int = -1
    dimension: int | None = None

    def __post_init__(self):
        expr=sp.sympify(self.expression);params=tuple(self.parameters);s=self.radial_variable
        if not isinstance(s,sp.Symbol) or any(not isinstance(p,sp.Symbol) for p in params):raise TypeError('Use SymPy symbols for s and parameters')
        if len(set(params))!=len(params) or s in params:raise ValueError('Parameters must be distinct from the radial variable')
        if len({p.name for p in params})!=len(params):raise ValueError('Parameter names must be unique')
        if expr.free_symbols-set(params)-{s}:raise ValueError('Undeclared kernel symbols')
        if expr.atoms(sp.NumberSymbol):raise ValueError('Named constants must be supplied as runtime parameters')
        interior,_=compact_parts(expr,s)
        if interior.has(sp.Piecewise,sp.Derivative,sp.Integral):raise ValueError('Nested piecewise and unevaluated expressions are not supported')
        if self.dimension not in (None,2,3):raise ValueError('dimension must be 2 or 3')
        if any(f.func not in (sp.exp,sp.sin,sp.cos,sp.log) for f in interior.atoms(sp.Function)):raise ValueError('Supported functions: exp, sin, cos, log, rational powers')
        if type(self.minimum_degree) is not int or self.minimum_degree < -1:raise ValueError('minimum_degree must be -1 or nonnegative; supplied by the author, not inferred')
        object.__setattr__(self,'expression',expr);object.__setattr__(self,'parameters',params)

    def __call__(self,**values):
        if set(values)!={p.name for p in self.parameters}:raise ValueError('Supply exactly the declared kernel parameters')
        ordered=tuple(values[p.name] for p in self.parameters)
        for symbol,value in zip(self.parameters,ordered):
            v=sp.sympify(value)
            if v.free_symbols or v.is_real is not True or v.is_finite is not True:raise ValueError('Parameters must be finite real constants')
            if symbol.is_positive and not v>0:raise ValueError(f'{symbol} must be positive')
            if symbol.is_nonnegative and not v>=0:raise ValueError(f'{symbol} must be nonnegative')
        return BoundKernel(self,ordered)

    def compile(self,*,dimension=2,derivative_order=6,arithmetic='float64',cache_dir=None):
        from .kernel_compiler import compile_kernel
        return compile_kernel(self,dimension,derivative_order,arithmetic,cache_dir)

@lru_cache(maxsize=256)
def origin_polynomial(family,order):
    s=family.radial_variable
    try:series=sp.series(compact_parts(family.expression,s)[0],s,0,order//2+2)
    except Exception as exc:raise ValueError('Cannot certify radial expansion at the origin') from exc
    polynomial=sp.S.Zero
    for term in sp.Add.make_args(sp.expand(series.removeO())):
        power=term.as_powers_dict().get(s,sp.S.Zero);coefficient=sp.simplify(term/s**power)
        if coefficient.has(s) or not power.is_Rational or power<0:raise ValueError('Cannot certify a finite direction-independent origin jet')
        if power.is_Integer:polynomial+=coefficient*s**power
        elif 2*power<=order:raise ValueError(f'Kernel lacks continuous derivatives through order {order} at the origin')
    return polynomial

@lru_cache(maxsize=512)
def expressions(family,alpha):
    if len(alpha) not in (2,3) or any(type(i) is not int or i<0 for i in alpha):raise ValueError('Expected a 2D/3D derivative multi-index')
    if sum(alpha)>6:raise NotImplementedError('Derivatives through order six are supported')
    if family.dimension is not None and len(alpha)!=family.dimension:raise ValueError("Kernel dimension mismatch")
    validate_support(family,sum(alpha))
    coords=_COORDINATES[:len(alpha)];s=sum(x*x for x in coords)
    def diff(expr):
        for x,n in zip(coords,alpha):expr=sp.diff(expr,x,n)
        return expr
    origin=diff(origin_polynomial(family,sum(alpha)).subs(family.radial_variable,s)).subs(dict.fromkeys(coords,0))
    expr=diff(compact_parts(family.expression,family.radial_variable)[0].subs(family.radial_variable,s))
    if origin.has(sp.oo,sp.zoo,sp.nan):raise ValueError('Nonfinite origin derivative')
    return expr,origin

@lru_cache(maxsize=512)
def numpy_functions(family,alpha):
    return tuple(sp.lambdify((*_COORDINATES[:len(alpha)],*family.parameters),e,'numpy',cse=True) for e in expressions(family,alpha))

@dataclass(frozen=True)
class BoundKernel:
    family: Kernel
    values: tuple
    @property
    def minimum_degree(self):return self.family.minimum_degree
    def derivative(self,displacement,alpha=None):
        z=np.asarray(displacement,dtype=float)
        if z.ndim<1 or z.shape[-1] not in (2,3) or not np.isfinite(z).all():raise ValueError('Expected finite 2D/3D displacements')
        alpha=tuple(alpha) if alpha is not None else (0,)*z.shape[-1]
        if len(alpha)!=z.shape[-1]:raise ValueError('Derivative dimension mismatch')
        fn,origin=numpy_functions(self.family,alpha);flat=z.reshape(-1,z.shape[-1]);zero=np.all(flat==0,axis=1);out=np.zeros(len(flat));params=tuple(map(float,self.values))
        support=self.support_squared()
        active=np.ones(len(flat),dtype=bool) if support is None else np.sum(flat*flat,axis=1)<support
        if zero.any():out[zero]=origin(*([0]*len(alpha)),*params)
        near=(active & ~zero & (np.sum(flat*flat,axis=1)<support/16)) if support is not None else np.zeros(len(flat),dtype=bool)
        regular=active & ~zero & ~near
        if regular.any():out[regular]=fn(*flat[regular].T,*params)
        if near.any():out[near]=near_function(self.family,alpha)(*flat[near].T,*params)
        return out.reshape(z.shape[:-1])
    def mp_derivative(self,ctx,coordinates,alpha):
        from .precision import mp_number
        expressions(self.family,tuple(alpha))  # Validate even for exterior queries.
        support=self.support_squared(ctx)
        if support is not None and sum(v*v for v in coordinates)>=support:return ctx.zero
        near=support is not None and not all(v==0 for v in coordinates) and sum(v*v for v in coordinates)<support/16
        fn=mp_functions(self.family,tuple(alpha),self_ctx=ctx)[2 if near else int(all(v==0 for v in coordinates))]
        return fn(*coordinates,*(mp_number(ctx,v) for v in self.values))
    def support_squared(self,ctx=None):
        from .precision import mp_number
        _,bound=compact_parts(self.family.expression,self.family.radial_variable)
        if bound is None:return None
        values={p:sp.sympify(v,rational=True) for p,v in zip(self.family.parameters,self.values)}
        value=bound.subs(values)
        return mp_number(ctx,value) if ctx else float(value)
    def compile(self,**kwargs):
        kwargs.setdefault('dimension',self.family.dimension or 2)
        artifact=self.family.compile(**kwargs)
        return artifact(**{p.name:v for p,v in zip(self.family.parameters,self.values)})
    matrix=ScalarKernel.matrix


@lru_cache(maxsize=512)
def mp_functions(family,alpha,self_ctx):
    ctx=self_ctx
    return tuple(sp.lambdify((*_COORDINATES[:len(alpha)],*family.parameters),expr,modules=[{'mpf':ctx.mpf,'sqrt':ctx.sqrt,'exp':ctx.exp,'sin':ctx.sin,'cos':ctx.cos,'log':ctx.log},'mpmath'],cse=True) for expr in (*expressions(family,alpha),near_expression(family,alpha)))


def compact_parts(expression,s):
    """Only an interior branch s < positive support_squared, then zero."""
    if not isinstance(expression,sp.Piecewise):return expression,None
    if len(expression.args)!=2:raise ValueError('Compact support requires exactly two branches')
    (inside,condition),(outside,otherwise)=expression.args
    if outside!=0 or otherwise!=sp.true or not isinstance(condition,(sp.StrictLessThan,sp.StrictGreaterThan)) or condition.lts!=s:
        raise ValueError('Expected Piecewise((interior, s < support_squared), (0, True))')
    support=condition.gts
    if support.has(s) or support.is_positive is not True:raise ValueError('Support squared must be provably positive and independent of s')
    return inside,support

@lru_cache(maxsize=256)
def validate_support(family,order):
    inside,bound=compact_parts(family.expression,family.radial_variable)
    if bound is None:return
    s=family.radial_variable;derivative=inside
    for n in range(order+1):
        value=sp.simplify(derivative.subs(s,bound))
        if value.has(sp.nan,sp.zoo,sp.oo,-sp.oo):value=sp.simplify(sp.limit(derivative,s,bound,dir='-'))
        if value!=0:raise ValueError(f'Compact-support boundary does not match zero exterior through derivative order {n}')
        derivative=sp.diff(derivative,s)

@lru_cache(maxsize=512)
def near_expression(family,alpha):
    inside,bound=compact_parts(family.expression,family.radial_variable)
    if bound is None:return expressions(family,alpha)[0]
    # Expand the radial profile first: cancelled low odd powers must disappear
    # before Cartesian differentiation, avoiding catastrophic origin cancellation.
    expr=sp.expand(inside).subs(family.radial_variable,sum(x*x for x in _COORDINATES[:len(alpha)]))
    for x,n in zip(_COORDINATES,alpha):expr=sp.diff(expr,x,n)
    return expr

@lru_cache(maxsize=512)
def near_function(family,alpha):
    return sp.lambdify((*_COORDINATES[:len(alpha)],*family.parameters),near_expression(family,alpha),'numpy',cse=True)


def Wendland(*,smoothness=6,dimension=2,radius=1):
    """Normalized Wendland C0/C2/C4/C6 kernels for dimension 2 or 3.

    radius is the physical support radius, a runtime parameter in generated C++.
    """
    if dimension not in (2,3):raise ValueError('Wendland currently supports dimension 2 or 3')
    if type(smoothness) is not int or smoothness not in (0,2,4,6):raise ValueError('smoothness must be 0, 2, 4 or 6')
    s=sp.Symbol('s',nonnegative=True);rho=sp.Symbol('radius',positive=True);q=sp.sqrt(s)/rho
    profiles={0:(1-q)**2,2:(1-q)**4*(4*q+1),4:(1-q)**6*(35*q*q+18*q+3)/3,6:(1-q)**8*(32*q**3+25*q*q+8*q+1)}
    expression=sp.Piecewise((profiles[smoothness],s<rho**2),(0,True))
    return Kernel(expression,s,(rho,),dimension=dimension)(radius=radius)
