"""Gauge-free divergence-free Stokes assembly selected through explicit spaces."""
import warnings
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial
from types import SimpleNamespace
import numpy as np
import sympy as sp
from .spaces import SpaceStokesProblem,DivergenceFreeSpace,PressureSpace,ScalarSpace
from .operators import Identity,Derivative,Laplacian
from .stokes import StokesArithmetic,_blocks
from .stencils import polynomial_powers
from .stokes_polynomials import append_columns,rank_ratio
from .methods import _query
from .time_data import at_time
from .evolution import _mv,_factor
from .unsteady_stokes import StokesTrajectory,UnsteadyStokesProblem


@lru_cache(maxsize=32)
def _polynomials(dimension,velocity_degree,pressure_degree):
    columns=[]
    if velocity_degree is not None:
        powers=polynomial_powers(dimension,velocity_degree)
        modes=[(c,power) for c in range(dimension) for power in powers]
        divergence_powers=polynomial_powers(dimension,max(0,velocity_degree-1))
        D=sp.zeros(len(divergence_powers),len(modes))
        for j,(c,power) in enumerate(modes):
            if power[c]:
                derivative=list(power);derivative[c]-=1
                D[divergence_powers.index(tuple(derivative)),j]=power[c]
        for vector in D.nullspace():
            columns.append(tuple((c,coefficient,power) for (c,power),coefficient in zip(modes,vector) if coefficient))
    if pressure_degree is not None:
        # Work modulo constants; no invisible pressure column enters assembly.
        for power in polynomial_powers(dimension,pressure_degree):
            if any(power):columns.append(((dimension,sp.Integer(1),power),))
    return tuple(columns)


class FlowPolynomials:
    def __init__(self,a,points,vd,pd):
        self.a=a;self.dimension=points.shape[1]
        self.origin=[a.number(v) for v in points[0]]
        self.scale=max(abs(a.number(v)-o) for p in points for v,o in zip(p,self.origin)) or a.number(1)
        self.columns=_polynomials(self.dimension,vd,pd)

    def matrix(self,points,rows):
        a=self.a;out=a.zeros(len(points),len(self.columns))
        for i,(point,row) in enumerate(zip(points,rows)):
            z=[(a.number(v)-o)/self.scale for v,o in zip(point,self.origin)]
            for j,column in enumerate(self.columns):
                for component,c,power in column:
                    if component not in row:continue
                    for alpha,b in row[component].terms:
                        if any(k>p for k,p in zip(alpha,power)):continue
                        v=a.number(c)*a.number(b)/self.scale**sum(alpha)
                        for coord,p,k in zip(z,power,alpha):v*=factorial(p)//factorial(p-k)*coord**(p-k)
                        out[i,j]+=v
        return out


def boundary_values(problem,cloud,a,time=0):
    bd={}
    for on,components in problem.boundary:
        labels=list(cloud.boundary) if on=='boundary' else ([on] if isinstance(on,str) else list(on))
        for label in labels:
            if label not in cloud.boundary:raise ValueError(f"Unknown boundary label: {label}")
            ids=cloud.boundary[label]
            data=[a.data(at_time(c,time),cloud.points[ids]) for c in components]
            for j,node in enumerate(ids):
                if int(node) not in bd:bd[int(node)]=[values[j] for values in data]
    if set(bd)!=set(cloud.boundary_indices):raise ValueError("Every boundary node requires velocity data")
    return a.vector([bd[int(node)][c] for c in range(problem.dimension) for node in cloud.boundary_indices])


def select_spaces(method,problem):
    if not isinstance(method.spaces,dict) or set(method.spaces)!=set(problem.model.fields):
        raise ValueError("Supply spaces={velocity: DivergenceFreeSpace(...), pressure: ScalarSpace(...)}")
    velocity,pressure=(method.spaces[f] for f in problem.model.fields)
    if not isinstance(velocity,DivergenceFreeSpace) or not isinstance(pressure,ScalarSpace) or isinstance(pressure,DivergenceFreeSpace):
        raise TypeError("Momentum-only Stokes requires DivergenceFreeSpace velocity and ScalarSpace pressure")
    return velocity,pressure,velocity.degree(),pressure.degree()


def assemble_spaces(method,problem,cloud):
    from .methods import GlobalCollocation
    from .symbolic_system import BlockPDE
    if method.kernel is not None:raise ValueError("Use either kernel or spaces, not both")
    if isinstance(problem,BlockPDE):
        if getattr(method,'legacy_unaugmented_hybrid',False) or getattr(method,'min_boundary_centers',0) or getattr(method,'local_backend',None) is not None:
            raise ValueError("These stencil options require divergence-free Stokes spaces")
        from .block_methods import BlockGlobal,BlockLHI
        if any(type(s) is not ScalarSpace for s in method.spaces.values()):raise TypeError("Independent scalar block fields require ScalarSpace")
        degrees={s.degree() for s in method.spaces.values()}
        if len(degrees)!=1:raise NotImplementedError("Scalar block spaces currently need a common polynomial degree")
        kernels={f:s.kernel for f,s in method.spaces.items()};degree=degrees.pop()
        target=(BlockGlobal(kernels,degree,method.precision) if isinstance(method,GlobalCollocation) else
                BlockLHI(kernels,method.stencil_size,degree,method.precision,method.stencil_policy))
        return target.assemble(problem,cloud)
    if not isinstance(problem,SpaceStokesProblem):raise TypeError("Space-based methods require a symbolic space or block problem")
    if problem.dimension!=cloud.dimension:raise ValueError("Problem and cloud dimensions differ")
    if method.polynomial_degree is not None:raise ValueError("Specify polynomial degrees on the spaces")
    velocity,pressure,vd,pd=select_spaces(method,problem)
    if isinstance(method,GlobalCollocation):
        if method.scheme!='symmetric':raise NotImplementedError("Space Stokes uses symmetric Hermite collocation")
        if method.precision.local_digits is not None or method.precision.global_dtype!='float64':raise ValueError("Global space Stokes uses global_digits")
        a=StokesArithmetic(velocity.kernel,pressure.kernel,method.precision.global_digits)
        return GlobalFlowSystem(problem,cloud,a,vd,pd)
    return LocalFlowSystem(method,problem,cloud,velocity,pressure,vd,pd)


class GlobalFlowSystem:
    """Dense divergence-free Stokes system for velocity/pressure coefficients.

    matrix contains momentum and boundary rows plus polynomial constraints;
    mass contains velocity-evaluation rows for transient solves and is not
    identity. A transient solve returns a StokesTrajectory.
    """
    def __init__(self,problem,cloud,a,vd,pd):
        self.problem,self.cloud,self.arithmetic=problem,cloud,a
        d=problem.dimension;ii=cloud.interior_indices;bi=cloud.boundary_indices
        if not len(ii) or not len(bi):raise ValueError("Interior and boundary nodes are required")
        self.points=np.vstack([cloud.interior]*d+[cloud.points[bi]]*d)
        self.functionals=[]
        for c in range(d):self.functionals.extend([{c:-problem.viscosity*Laplacian(d),d:Derivative(c,d)}]*len(ii))
        for c in range(d):self.functionals.extend([{c:Identity(d)}]*len(bi))
        self.polynomials=FlowPolynomials(a,self.points,vd,pd)
        self.n=len(self.points);self.size=self.n+len(self.polynomials.columns)
        self.matrix=a.zeros(self.size,self.size)
        self.matrix[:self.n,:]=self.evaluation(self.points,self.functionals)
        P=self.matrix[:self.n,self.n:]
        if self.size>self.n:
            if rank_ratio(P)<1e-12:raise ValueError("Divergence-free polynomial functionals are rank deficient")
            self.matrix[self.n:,:self.n]=P.T
        self.mass=a.zeros(self.size,self.size)
        for c in range(d):
            self.mass[c*len(ii):(c+1)*len(ii),:]=self.evaluation(cloud.interior,[{c:Identity(d)}]*len(ii))
        self.factor=None;self._factors={}
        if not problem.transient:self.rhs=self.data(0)

    def evaluation(self,points,rows):
        a=self.arithmetic
        kernel=_blocks(a,points,rows,self.points,self.functionals)
        return append_columns(a,kernel,self.polynomials.matrix(points,rows))

    def data(self,time):
        a=self.arithmetic
        forcing=[v for f in self.problem.forcing for v in a.data(at_time(f,time),self.cloud.interior)]
        boundary=list(boundary_values(self.problem,self.cloud,a,time))
        return a.vector(forcing+boundary+[0]*(self.size-self.n))

    def solve(self,dt=None,steps=None,*,scheme='bdf2',start_time=0):
        a=self.arithmetic
        if not self.problem.transient:
            if dt is not None or steps is not None:raise ValueError("Steady Stokes does not take time-step arguments")
            if self.factor is None:self.factor=a.factor(self.matrix)
            rhs=self.rhs;z=self.factor.solve(rhs)
            residual=a.norm(_mv(self.matrix,z)-rhs)/(a.norm(rhs) or a.number(1))
            return GlobalFlowSolution(self,z,0,diagnostics={'relative_residual':float(residual),'unknowns':self.size,
                'pressure':'kernel representative modulo constants','divergence':'analytic','scaled_condition':self.factor.condition})
        if dt is None or steps is None:raise ValueError("Unsteady Stokes requires dt and steps")
        dt=Fraction(dt);start_time=Fraction(start_time)
        if dt<=0 or type(steps) is not int or steps<1:raise ValueError("Positive dt and integer steps required")
        if scheme not in ('bdf2','backward_euler'):raise ValueError("Unknown time scheme")
        d=self.problem.dimension;ni=len(self.cloud.interior);nb=len(self.cloud.boundary_indices)
        initial=InitialFlow(self.problem,a,a.number(start_time))
        iv=initial.velocity(self.cloud.points[self.cloud.boundary_indices],extended=bool(a.ctx))
        bd=boundary_values(self.problem,self.cloud,a,a.number(start_time));tol=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        if any(abs(iv[i,c]-bd[c*nb+i])>tol*(1+abs(bd[c*nb+i])) for c in range(d) for i in range(nb)):
            raise ValueError("Initial velocity is incompatible with boundary data at start_time")
        iv=initial.velocity(self.cloud.interior,extended=bool(a.ctx))
        previous=a.vector([iv[i,c] for c in range(d) for i in range(ni)]+[0]*(self.size-d*ni))
        previous2=None;states=[];new=0
        for step in range(1,steps+1):
            second=scheme=='bdf2' and step>1;alpha=(Fraction(3,2) if second else Fraction(1))/dt
            weights=(2/dt,-Fraction(1,2)/dt) if second else (1/dt,)
            time=a.number(start_time+step*dt);rhs=self.data(time)+a.number(weights[0])*previous
            if second:rhs+=a.number(weights[1])*previous2
            if alpha not in self._factors:
                matrix=self.matrix+a.number(alpha)*self.mass
                self._factors[alpha]=(matrix,a.factor(matrix));new+=1
            matrix,factor=self._factors[alpha];z=factor.solve(rhs)
            old=[states[-1] if states else initial]
            if second:old.append(states[-2] if len(states)>1 else initial)
            residual=a.norm(_mv(matrix,z)-rhs)/(a.norm(rhs) or a.number(1))
            states.append(GlobalFlowSolution(self,z,time,a.number(alpha),old,[a.number(w) for w in weights],
                {'relative_residual':float(residual),'divergence':'analytic','pressure':'kernel representative modulo constants'}))
            previous2=previous;previous=_mv(self.mass,z)
        return StokesTrajectory(initial,states,{'steps':steps,'dt':str(dt),'new_factorizations':new,
            'cached_factorizations':len(self._factors),'pressure':'no initial value or gauge imposed','scheme':scheme})


class InitialFlow:
    def __init__(self,problem,a,time):self.problem,self.a,self.time=problem,a,time
    def velocity(self,points,*,extended=False):
        points=_query(points,self.problem.dimension);a=self.a;out=a.zeros(len(points),self.problem.dimension)
        for j,data in enumerate(self.problem.initial):
            for i,v in enumerate(a.data(data,points)):out[i,j]=v
        if extended:return out
        return np.asarray(out.tolist() if a.ctx else out,dtype=float).reshape(len(points),self.problem.dimension)


class GlobalFlowSolution:
    """Global divergence-free velocity and pressure representative.

    velocity, pressure_gradient, divergence, and momentum residuals can
    be sampled at query points. pressure is defined modulo constants;
    pass a reference point/value to obtain a chosen gauge.
    """
    def __init__(self,system,z,time,alpha=0,history=(),weights=(),diagnostics=None):
        self.system,self.coefficients,self.time=system,z,time
        self.alpha,self.history,self.weights=alpha,history,weights
        self.diagnostics=diagnostics or {}

    def _components(self,points,components,operators,extended=False):
        p=self.system.problem;a=self.system.arithmetic;points=_query(points,p.dimension)
        if extended and not a.ctx:raise ValueError("Extended evaluation requires global_digits")
        out=a.zeros(len(points),len(components))
        for j,(c,op) in enumerate(zip(components,operators)):
            values=_mv(self.system.evaluation(points,[{c:op}]*len(points)),self.coefficients)
            for i,v in enumerate(values):out[i,j]=v
        if extended:return out
        return np.asarray(out.tolist() if a.ctx else out,dtype=float).reshape(len(points),len(components))

    def velocity(self,points,*,extended=False):
        d=self.system.problem.dimension
        return self._components(points,range(d),[Identity(d)]*d,extended)

    def pressure_gradient(self,points,*,extended=False):
        d=self.system.problem.dimension
        return self._components(points,[d]*d,[Derivative(j,d) for j in range(d)],extended)

    def pressure(self,points,*,reference=None,extended=False):
        d=self.system.problem.dimension;a=self.system.arithmetic
        values=self._components(points,[d],[Identity(d)],extended)
        if reference is not None:
            point,value=reference
            offset=a.number(value)-self._components(np.asarray(point,dtype=float)[None,:],[d],[Identity(d)],bool(a.ctx))[0,0]
            for i in range(len(points)):values[i,0]+=offset if extended else float(offset)
        return values if extended else values[:,0]

    def divergence(self,points,*,extended=False):
        d=self.system.problem.dimension;a=self.system.arithmetic
        derivatives=self._components(points,range(d),[Derivative(j,d) for j in range(d)],extended)
        if extended:return a.ctx.matrix([sum(derivatives[i,j] for j in range(d)) for i in range(len(points))])
        return derivatives.sum(axis=1)

    def residuals(self,points,*,extended=False):
        d=self.system.problem.dimension;a=self.system.arithmetic
        lap=self._components(points,range(d),[Laplacian(d)]*d,bool(a.ctx));pg=self.pressure_gradient(points,extended=bool(a.ctx))
        out=pg-a.number(self.system.problem.viscosity)*lap
        if self.alpha:
            out+=self.alpha*self.velocity(points,extended=bool(a.ctx))
            for old,w in zip(self.history,self.weights):out-=w*old.velocity(points,extended=bool(a.ctx))
        for c,f in enumerate(self.system.problem.forcing):
            for i,v in enumerate(a.data(at_time(f,self.time),points)):out[i,c]-=v
        if extended:
            if not a.ctx:raise ValueError("Extended evaluation requires global_digits")
            return out
        return np.asarray(out.tolist() if a.ctx else out,dtype=float)


@dataclass(frozen=True)
class TaggedComponent:
    problem: object
    cloud: object
    component: int
    time: object=0
    def at_time(self,time):return TaggedComponent(self.problem,self.cloud,self.component,time)
    def _values(self,points,a):
        values=boundary_values(self.problem,self.cloud,a,self.time);ids=self.cloud.boundary_indices;n=len(ids)
        lookup={tuple(self.cloud.points[node]):values[self.component*n+j] for j,node in enumerate(ids)}
        return [lookup[tuple(point)] for point in points]
    def __call__(self,points):
        from .nodal import Arithmetic
        from .kernels import IMQ
        return np.asarray(self._values(points,Arithmetic(IMQ(1))),dtype=float)
    def mp_values(self,ctx,points):
        from .nodal import Arithmetic
        from .kernels import IMQ
        a=Arithmetic(IMQ(1));a.ctx=ctx
        return self._values(points,a)


class LocalFlowSystem:
    """Two-dimensional divergence-free LHI Stokes wrapper.

    Its global sparse unknowns are local solution-center velocity values.
    Supported pressure output is local pressure gradients; there is no
    globally reconciled pressure scalar field.
    """
    def __init__(self,method,problem,cloud,velocity,pressure,vd,pd):
        from .lhi_stokes import LHIUnsteadyStokes
        if cloud.dimension!=2:raise NotImplementedError("Divergence-free LHI space assembly currently supports 2D; global supports 2D/3D")
        if (vd is None and pd is not None) or (vd is not None and (vd<2 or pd!=vd-1)):
            raise NotImplementedError("LHI currently needs no polynomial tails, or velocity degree 2..4 and pressure degree one lower")
        # Steady data have no time symbol, so binding time does not change them.
        from .legacy_cpp import LegacyCppLHIBackend
        if isinstance(method.local_backend,LegacyCppLHIBackend) and problem.transient:
            raise NotImplementedError('Experimental C++ backend currently supports steady LHI only')
        from .configuration import configured_backend
        native=UnsteadyStokesProblem(problem.forcing,tuple(TaggedComponent(problem,cloud,j) for j in range(2)),
            problem.initial if problem.transient else (0,0),problem.viscosity)
        backend=LHIUnsteadyStokes(velocity.kernel,method.stencil_size,method.precision,method.stencil_policy,
            method.pde_stencil_size,pressure_kernel=pressure.kernel,polynomial_degree=vd,
            legacy_unaugmented_hybrid=False,
            min_boundary_centers=method.min_boundary_centers,local_backend=configured_backend(method))
        self.base=backend.assemble(native,cloud);self.problem=problem
        self.matrix=self.base.SY;self.mass=self.base.mass;self.arithmetic=self.base.arithmetic;self.factor=None
        if not problem.transient:
            f,g=self.base._data(0);a=self.arithmetic
            boundary=a.vector([sum(v*g[j] for j,v in row.items()) for row in self.base.sb_rows])
            self.rhs=_mv(self.mass,f)-boundary

    def solve(self,dt=None,steps=None,**kwargs):
        if self.problem.transient:
            if dt is None or steps is None:raise ValueError("Unsteady Stokes requires dt and steps")
            result=self.base.solve(dt,steps,**kwargs)
            return StokesTrajectory(result.initial,[LocalFlowSolution(s) for s in result.states],result.diagnostics)
        if dt is not None or steps is not None or kwargs:raise ValueError("Steady Stokes does not take time-step arguments")
        from .lhi_stokes import LHIStokesState
        a=self.arithmetic;f,g=self.base._data(0)
        boundary=a.vector([sum(v*g[j] for j,v in row.items()) for row in self.base.sb_rows])
        if self.factor is None:self.factor=_factor(self.matrix,a)
        condition=float(np.linalg.cond(self.matrix.toarray())) if not a.ctx and len(self.rhs)<=300 else None
        if condition is not None and condition>1e12:
            warnings.warn("Ill-conditioned steady divergence-free LHI system; inspect field errors and stencil choices",RuntimeWarning,stacklevel=2)
        values=self.factor.solve(self.rhs)
        residual=float(a.norm(_mv(self.matrix,values)-self.rhs)/(a.norm(self.rhs) or a.number(1)))
        state=LHIStokesState(self.base,values,f,g,0,0,[],[],residual)
        state.diagnostics.update(getattr(self.base,'backend_diagnostics',{}))
        state.diagnostics.update(pressure='local gradients only',divergence='patchwise analytic',steady=True,global_condition=condition)
        return LocalFlowSolution(state)


class LocalFlowSolution:
    """LHI velocity and pressure-gradient reconstruction at one time.

    Off-node evaluation is patchwise. pressure() is unavailable because
    local gradients have not been integrated into a global gauge field.
    """
    def __init__(self,state):self.state=state;self.time=state.time;self.diagnostics=state.diagnostics
    def velocity(self,points,*,extended=False):return self.state.evaluate_velocity(points,extended=extended)
    def pressure_gradient(self,points,*,extended=False):return self.state.pressure_gradient(points,extended=extended)
    def pressure(self,*args,**kwargs):raise NotImplementedError("LHI returns pressure gradients; no globally reconciled pressure reconstruction is available")
    def residuals(self,points,*,extended=False):return self.state.residuals(points,extended=extended)[:,:2]
    def divergence(self,points,*,extended=False):
        a=self.state.system.local_arithmetic
        if extended and not a.ctx:raise ValueError("Extended evaluation requires local_digits")
        derivatives=self.state._fields(points,[0,1],[Derivative(0),Derivative(1)])
        values=[derivatives[i,0]+derivatives[i,1] for i in range(len(points))]
        return a.ctx.matrix(values) if extended else np.asarray(values,dtype=float)
