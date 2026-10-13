"""Native global divergence-free velocity/pressure collocation.

Python describes rows and exact polynomial modes; the worker assembles and
factors every dense block and evaluates the resulting vector expansion.
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np

from .global_backend import CppGlobalSystem, _compiled, _acquire_worker
from .space_stokes import (FlowPolynomials, GlobalFlowSolution, select_spaces,
                           boundary_values, InitialFlow)
from .time_data import at_time
from .unsteady_stokes import StokesTrajectory
from .stokes import StokesArithmetic
from .operators import Identity, Derivative, Laplacian, bind_operators
from .methods import _query


def _field_tokens(row, encode):
    terms = [(component, alpha, value) for component, op in row.items()
             for alpha, value in op.terms]
    return [str(len(terms)), *(token for component, alpha, value in terms
            for token in [str(component), encode(value), *(str(v) for v in alpha)])]


def _artifact(method, problem, dimension):
    if dimension != problem.dimension:
        raise ValueError('Problem and preparation dimensions differ')
    velocity, pressure, _, _ = select_spaces(method, problem)
    # Momentum uses two velocity derivatives on either side plus the
    # two-derivative solenoidal kernel transform. Both pressure sources and
    # public pressure-gradient queries use one derivative. Preparing only two
    # pressure derivatives also permits C2 potentials such as PHS(3).
    pressure_order = 2
    executable, bound = _compiled(velocity.kernel, dimension, 6,
        'mpfr' if method.precision.global_digits else 'float64', method.backend,
        pressure_kernel=pressure.kernel, pressure_order=pressure_order)
    from .kernel_compiler import as_bound
    return executable, bound, as_bound(pressure.kernel), pressure_order


def prepare_cpp_flow(method, problem, dimension, cache_dir):
    if cache_dir is not None:
        method.backend = replace(method.backend, cache_dir=cache_dir)
    method.compiled_artifact = _artifact(method, problem, dimension)[0]
    return method


class _MassMatrix:
    """Read-only explicit export of the native velocity mass matrix."""
    def __init__(self, system):self.system=system
    @property
    def shape(self):
        self.system.worker.check_open()
        return (self.system.size,self.system.size)
    def _export(self):
        reply=self.system.worker.request('MASS_EXPORT');n=self.system.size
        if not reply or int(reply[0])!=n or len(reply)!=1+n*n:
            raise RuntimeError('Invalid native mass export')
        return reply[1:]
    def to_numpy(self):
        return np.asarray(self._export(),dtype=float).reshape(self.shape)
    def to_mpmath(self):
        s=self.system
        if not s.ctx:raise ValueError('to_mpmath requires extended precision')
        values=self._export()
        return s.ctx.matrix([[s.number(values[i*s.size+j]) for j in range(s.size)] for i in range(s.size)])
    def __array__(self,dtype=None,copy=None):
        result=self.to_numpy()
        if dtype is not None:result=result.astype(dtype)
        return result.copy() if copy else result


class CppGlobalFlowSystem(CppGlobalSystem):
    """Native momentum/boundary functionals plus solenoidal polynomial modes."""
    _native_global_flow = True

    @property
    def mass(self):
        self.worker.check_open()
        return self._mass

    def data(self,time):
        a=self.arithmetic
        forcing=[v for f in self.problem.forcing for v in a.data(at_time(f,time),self.cloud.interior)]
        return a.vector(forcing+list(boundary_values(self.problem,self.cloud,a,time))+[0]*(self.size-self.n))

    def _mass_values(self,coefficients):
        reply=self.worker.request('MASS_MV '+' '.join(self.encode(v) for v in coefficients))
        if not reply or int(reply[0])!=self.size or len(reply)!=self.size+1:
            raise RuntimeError('Invalid native mass product')
        return self.arithmetic.vector([self.number(v) for v in reply[1:]])

    def solve(self, dt=None, steps=None, *, scheme='bdf2', start_time=0):
        self.worker.check_open()
        if not self.problem.transient:
            if dt is not None or steps is not None:
                raise ValueError('Native stationary Stokes does not take time-step arguments')
            solution = super().solve()
            solution.diagnostics.update(pressure='kernel representative modulo constants',divergence='analytic')
            return CppGlobalFlowSolution(self, solution.coefficients, 0,diagnostics=solution.diagnostics)
        if dt is None or steps is None:raise ValueError('Unsteady Stokes requires dt and steps')
        dt,start_time=Fraction(dt),Fraction(start_time)
        if dt<=0 or type(steps) is not int or steps<1:raise ValueError('Positive dt and integer steps required')
        if scheme not in ('bdf2','backward_euler'):raise ValueError('Unknown time scheme')
        a=self.arithmetic;d=self.problem.dimension;ni=len(self.cloud.interior);nb=len(self.cloud.boundary_indices)
        initial=InitialFlow(self.problem,a,a.number(start_time))
        iv=initial.velocity(self.cloud.points[self.cloud.boundary_indices],extended=bool(a.ctx))
        bd=boundary_values(self.problem,self.cloud,a,a.number(start_time))
        tol=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        if any(abs(iv[i,c]-bd[c*nb+i])>tol*(1+abs(bd[c*nb+i])) for c in range(d) for i in range(nb)):
            raise ValueError('Initial velocity is incompatible with boundary data at start_time')
        iv=initial.velocity(self.cloud.interior,extended=bool(a.ctx))
        previous=a.vector([iv[i,c] for c in range(d) for i in range(ni)]+[0]*(self.size-d*ni))
        previous2=None;states=[];new=0
        original_rhs=self.rhs
        try:
            for step in range(1,steps+1):
                second=scheme=='bdf2' and step>1
                alpha=(Fraction(3,2) if second else Fraction(1))/dt
                weights=(2/dt,-Fraction(1,2)/dt) if second else (1/dt,)
                time=a.number(start_time+step*dt)
                rhs=self.data(time)+a.number(weights[0])*previous
                if second:rhs+=a.number(weights[1])*previous2
                fresh,cached=map(int,self.worker.request('SHIFT '+self.encode(alpha)))
                new+=fresh
                self.rhs=rhs;solved=super().solve()
                history=[states[-1] if states else initial]
                if second:history.append(states[-2] if len(states)>1 else initial)
                solved.diagnostics.update(divergence='analytic',pressure='kernel representative modulo constants')
                states.append(CppGlobalFlowSolution(self,solved.coefficients,time,a.number(alpha),history,
                    [a.number(w) for w in weights],solved.diagnostics))
                previous2=previous;previous=self._mass_values(solved.coefficients)
        finally:self.rhs=original_rhs
        return StokesTrajectory(initial,states,{'steps':steps,'dt':str(dt),'scheme':scheme,
            'new_factorizations':new,'cached_factorizations':cached,
            'backend':'cpp_global','arithmetic':'mpfr' if a.ctx else 'float64',
            'pressure':'no initial value or gauge imposed'})


class CppGlobalFlowSolution(GlobalFlowSolution):
    """Native velocity, pressure, derivatives and momentum residual evaluation.

    Uses the same field API as the Python global Stokes solution. Pass
    ``extended=True`` to retain multiprecision output.
    """
    def _components(self, points, components, operators, extended=False):
        s = self.system
        s.worker.check_open()
        d = s.problem.dimension
        points = _query(points, d)
        components, operators = list(components), list(operators)
        if extended and s.ctx is None:
            raise ValueError('Extended evaluation requires global_digits')
        if len(components) != len(operators):
            raise ValueError('One operator required per component')
        rows = []
        for component, operator in zip(components, operators):
            if component not in range(d+1):
                raise ValueError('Invalid field component')
            for op in bind_operators([operator]*len(points), points):
                if op.dimension != d:
                    raise ValueError('Evaluation operator and point dimensions must match')
                order = max((sum(alpha) for alpha, _ in op.terms), default=0)
                limit = s.pressure_order-1 if component == d else 2
                if order > limit:
                    raise ValueError('Evaluation derivative exceeds compiled field order')
                rows.append({component: op})
        count = len(rows)
        out = s.arithmetic.zeros(len(points), len(components))
        if count:
            tokens = ['EVAL', str(count)]
            tokens.extend(s.encode(v) for v in np.tile(points, (len(components), 1)).flat)
            for row in rows:
                tokens.extend(_field_tokens(row, s.encode))
            tokens.extend(s.encode(v) for v in self.coefficients)
            reply = s.worker.request(' '.join(tokens))
            if not reply or int(reply[0]) != count or len(reply) != count+1:
                raise RuntimeError('Invalid native field evaluation result')
            for j in range(len(components)):
                for i in range(len(points)):
                    out[i, j] = s.number(reply[1+j*len(points)+i])
        if extended:
            return out
        return np.asarray(out.tolist() if s.ctx else out, dtype=float).reshape(len(points), len(components))


def assemble_cpp_flow(method, problem, cloud):
    d = problem.dimension
    velocity, pressure, vd, pd = select_spaces(method, problem)
    if not len(cloud.interior_indices) or not len(cloud.boundary_indices):
        raise ValueError('Interior and boundary nodes are required')
    a = StokesArithmetic(velocity.kernel, pressure.kernel, method.precision.global_digits)
    points = np.vstack([cloud.interior]*d + [cloud.points[cloud.boundary_indices]]*d)
    rows = []
    for component in range(d):
        rows.extend([{component: -problem.viscosity*Laplacian(d),
                      d: Derivative(component, d)}]*len(cloud.interior_indices))
    for component in range(d):
        rows.extend([{component: Identity(d)}]*len(cloud.boundary_indices))
    polynomials = FlowPolynomials(a, points, vd, pd)
    columns = polynomials.columns
    if columns:
        # Small geometric rank screen only; native arithmetic assembles the
        # actual polynomial blocks from these exact mode descriptors.
        from .stokes_polynomials import rank_ratio
        if rank_ratio(polynomials.matrix(points, rows)) < 1e-12:
            raise ValueError('Divergence-free polynomial functionals are rank deficient')
    if problem.transient:
        # Time data belong to solve(start_time=...), not assembly at t=0.
        rhs = [0]*(len(points)+len(columns))
    else:
        forcing = [v for data in problem.forcing for v in a.data(data, cloud.interior)]
        rhs = forcing + list(boundary_values(problem, cloud, a)) + [0]*len(columns)
    executable, bound, pressure_bound, pressure_order = _artifact(method, problem, d)
    worker = _acquire_worker(executable, method.backend)
    system = CppGlobalFlowSystem(worker, points, rows, columns, rhs, a.ctx,
        method.precision.global_digits or 17, method.backend, 6)
    system.problem, system.cloud, system.arithmetic = problem, cloud, a
    system.points, system.functionals, system.polynomials = points, rows, polynomials
    system.n, system.pressure_order = len(points), pressure_order
    enc = system.encode
    try:
        tokens = ['INIT', str(system.digits), str(method.backend.threads), '1',
                  str(len(points)), str(len(columns)), enc(1)]
        tokens.extend(enc(v) for v in bound.values)
        tokens.extend(enc(v) for v in pressure_bound.values)
        tokens.extend(enc(v) for v in points.flat)
        tokens.extend(enc(v) for v in points.flat)
        tokens.extend(enc(v) for v in polynomials.origin)
        tokens.extend([enc(polynomials.scale)]*d)
        for column in columns:
            tokens.append(str(len(column)))
            for component, coefficient, power in column:
                tokens.extend([str(component), enc(coefficient), *(str(v) for v in power)])
        for row in rows + rows:
            tokens.extend(_field_tokens(row, enc))
        worker.request(' '.join(tokens))
        worker.request('BUILD_MOL' if problem.transient else 'BUILD')
        if problem.transient:
            mass_rows=[{component:Identity(d)} for component in range(d) for _ in cloud.interior_indices]
            tokens=['MASS',str(len(mass_rows))]
            for row in mass_rows:tokens.extend(_field_tokens(row,enc))
            worker.request(' '.join(tokens))
            system._mass=_MassMatrix(system)
    except Exception:
        worker.close(force=True)
        raise
    return system
