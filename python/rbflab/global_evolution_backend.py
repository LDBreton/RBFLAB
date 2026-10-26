"""Scalar method-of-lines through the shared persistent native dense engine."""
from .api_contracts import _UNSET, _keyword_alias
from dataclasses import replace
from fractions import Fraction
import numpy as np
from .global_backend import CppGlobalSystem,CppGlobalSolution,_op_tokens
from .global_scalar_engine import assemble_scalar_native
from .global_data_codegen import native_data_bundle,NativeDataPlan
from .operators import Identity,bind_operators
from .problems import boundary_data
from .nodal import dense_arithmetic
from .evolution import _EvolutionSlice,EvolutionState,EvolutionTrajectory,_sparse


def scalar_data_bundle(problem,dimension,initial=None):
    """Problem-neutral programs; geometry chooses their row mapping later."""
    transient=hasattr(problem,'initial')
    rhs=native_data_bundle((problem.rhs,*(bc.rhs for bc in problem.boundary)),dimension,transient=transient)
    start=native_data_bundle((initial,),dimension,transient=False) if initial is not None else None
    expressions=list(rhs[0].expressions) if rhs else []
    initial_id=None
    if start:
        expression=start[0].expressions[start[1][0]]
        if expression not in expressions:expressions.append(expression)
        initial_id=expressions.index(expression)
    plan=NativeDataPlan(tuple(expressions),(),(),dimension) if expressions else None
    return plan,rhs[1] if rhs else None,initial_id


def scalar_data_rows(problem,cloud,ids,order=None):
    if ids is None:return None
    selected={};conflicting=False
    for bc,program in zip(problem.boundary,ids[1:]):
        labels=list(cloud.boundary) if bc.on=='boundary' else ([bc.on] if isinstance(bc.on,str) else list(bc.on))
        for label in labels:
            if label not in cloud.boundary:raise ValueError(f'Unknown boundary label: {label}')
            for node in cloud.boundary[label]:
                node=int(node)
                if node in selected and selected[node]!=program:conflicting=True
                selected.setdefault(node,program)
    if conflicting:return None
    nodes=range(len(cloud.points)) if order is None else order
    return [selected.get(int(node),ids[0]) for node in nodes]


class NativeMatrixView:
    def __init__(self,system,command):self.system,self.command=system,command
    @property
    def shape(self):self.system.worker.check_open();return (self.system.size,self.system.size)
    def _export(self):
        reply=self.system.worker.request(self.command);n=self.system.size
        if not reply or int(reply[0])!=n or len(reply)!=1+n*n:raise RuntimeError('Invalid native matrix export')
        return reply[1:]
    def to_numpy(self):return np.asarray(self._export(),dtype=float).reshape(self.shape)
    def to_mpmath(self):
        s=self.system
        if not s.ctx:raise ValueError('to_mpmath requires extended precision')
        values=self._export();return s.ctx.matrix([[s.number(values[i*s.size+j]) for j in range(s.size)] for i in range(s.size)])
    def __array__(self,dtype=None,copy=None):
        values=self.to_numpy()
        if dtype is not None:values=values.astype(dtype)
        return values.copy() if copy else values


class CppScalarEvolutionSystem(CppGlobalSystem):
    kind='global'
    @property
    def spatial(self):return self.matrix
    @spatial.setter
    def spatial(self,value):self.matrix=value
    @property
    def mass(self):self.worker.check_open();return self._mass
    @property
    def initial_matrix(self):self.worker.check_open();return self._initial_matrix
    @initial_matrix.setter
    def initial_matrix(self,value):
        array=np.asarray(value.tolist() if hasattr(value,'cols') else value)
        if array.shape!=(self.size,self.size):raise ValueError('Initial matrix dimensions changed')
        self.worker.request('INITIAL_REPLACE '+str(self.size)+' '+' '.join(self.encode(v) for v in array.flat))

    def _native_data_valid(self):
        from .global_flow_backend import _boundary_key
        key=(self.problem.rhs,tuple((bc.on if isinstance(bc.on,str) else tuple(bc.on),bc.rhs) for bc in self.problem.boundary))
        return (self._native_rows is not None and self.problem is self._data_problem and key==self._data_key
            and np.array_equal(self.cloud.points,self._data_points) and _boundary_key(self.cloud)==self._data_boundary)

    def data(self,time):
        a=self.arithmetic
        if self._native_data_valid():
            reply=self.worker.request('RHS_DATA '+self.encode(time))
            if int(reply[0])!=self.size or len(reply)!=self.size+1:raise RuntimeError('Invalid native data result')
            rhs=a.vector([self.number(v) for v in reply[1:]])
        else:
            p=self.problem.at(time);bd=boundary_data(p,self.cloud,ctx=a.ctx)
            forcing=a.data(p.rhs,self.cloud.interior);rhs=a.vector([0]*self.size)
            for node,v in zip(self.cloud.interior_indices,forcing):rhs[int(node)]=v
            for node in self.cloud.boundary_indices:rhs[int(node)]=bd[int(node)][1]
        f=a.vector([rhs[int(node)] for node in self.cloud.interior_indices])
        g=a.vector([rhs[int(node)] for node in self.cloud.boundary_indices])
        return rhs,f,g

    def _initial(self,time):
        a=self.arithmetic;cloud=self.cloud
        if self._initial_program is not None and self.problem.initial is self._prepared_initial and np.array_equal(cloud.points,self._data_points):
            reply=self.worker.request('SAMPLE_DATA '+str(self._initial_program)+' '+self.encode(time))
            values=a.vector([self.number(v) for v in reply[1:]])
        else:values=a.vector(a.data(self.problem.initial,cloud.points))
        _,_,g=self.data(time);mismatch=a.number(0)
        tolerance=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        for node,boundary in zip(cloud.boundary_indices,g):
            op=self.boundary_operators[int(node)];data=self.problem.initial
            if op==Identity(cloud.dimension):value=values[int(node)]
            elif hasattr(data,'operator_values'):value=data.operator_values(op,cloud.points[[int(node)]],ctx=a.ctx)[0]
            elif not callable(data) and np.ndim(data)==0:value=sum(a.number(c)*a.number(data) for alpha,c in op.terms if not any(alpha))
            else:raise TypeError('Derivative boundary initialization needs SymbolicData or InitialData with analytic derivatives')
            if not (a.ctx.isfinite(value) if a.ctx else np.isfinite(value)):
                raise ValueError('Initial boundary derivatives must be finite')
            error=abs(value-boundary)/(1+abs(boundary));mismatch=max(mismatch,error)
            if error>tolerance:raise ValueError('Initial data are incompatible with boundary data at start_time')
        target=list(values)
        for node,v in zip(cloud.boundary_indices,g):target[int(node)]=v
        target += [0]*len(self.powers)
        reply=self.worker.request('EVOLVE_INIT '+' '.join(self.encode(v) for v in target))
        if int(reply[1])!=self.size or len(reply)!=self.size+3:raise RuntimeError('Invalid initial native solve')
        coefficients=a.vector([self.number(v) for v in reply[2:-1]])
        self.initial_compatibility_residual=float(mismatch);self.initial_constraint_residual=float(self.number(reply[-1]))
        return CppGlobalSolution(self,coefficients,{'backend':'cpp_global','initial':True})

    def solve(self,dt=None,steps=None,*,scheme=_UNSET,start_time=0,time_scheme=_UNSET):
        scheme = _keyword_alias(scheme, time_scheme, name="scheme", alias_name="time_scheme", default="bdf2")
        self.worker.check_open()
        if dt is None or steps is None:raise ValueError('Unsteady scalar PDE requires dt and steps')
        dt,start_time=Fraction(dt),Fraction(start_time)
        if dt<=0 or type(steps) is not int or steps<1:raise ValueError('Positive dt and integer steps required')
        if scheme not in ('backward_euler','bdf2'):raise ValueError('Unknown time scheme')
        a=self.arithmetic;initial=self._initial(a.number(start_time));states=[];new=0;cached=0
        for step in range(1,steps+1):
            second=scheme=='bdf2' and step>1
            alpha=(Fraction(3,2) if second else Fraction(1))/dt
            weights=(2/dt,-Fraction(1,2)/dt) if second else (1/dt,)
            time=a.number(start_time+step*dt);native=self._native_data_valid()
            tokens=['TIME_STEP',self.encode(time),self.encode(alpha),str(len(weights)),*(self.encode(w) for w in weights),str(int(native))]
            if not native:tokens.extend(self.encode(v) for v in self.data(time)[0])
            reply=self.worker.request(' '.join(tokens));fresh,cached,n=map(int,reply[:3])
            if n!=self.size or len(reply)!=2*n+4:raise RuntimeError('Invalid native timestep result')
            new+=fresh;coefficients=a.vector([self.number(v) for v in reply[3:3+n]])
            derivative=a.vector([self.number(v) for v in reply[3+n:3+2*n]])
            residual=self.number(reply[-1]);diagnostics={'relative_residual':float(residual),'backend':'cpp_global',
                'linear_solver':self.backend.linear_solver,'data_backend':'cpp_symbolic' if native else 'python',
                'scheme':'bdf2' if second else 'backward_euler'}
            if a.ctx:diagnostics['relative_residual_mp']=residual
            old=[states[-1] if states else initial]
            if second:old.append(states[-2] if len(states)>1 else initial)
            spatial=CppGlobalSolution(self,coefficients,diagnostics)
            states.append(EvolutionState(self,spatial,coefficients,derivative,time,a.number(alpha),old,[a.number(w) for w in weights],diagnostics))
        return EvolutionTrajectory(initial,states,{'scheme':scheme,'steps':steps,'dt':str(dt),'kind':'global',
            'backend':'cpp_global','linear_solver':self.backend.linear_solver,'data_backend':states[-1].diagnostics['data_backend'],
            'unknowns':self.size,'global_digits':self.digits if a.ctx else None,'new_factorizations':new,'cached_factorizations':cached,
            'initial_factorizations':1,'initial_compatibility_residual':self.initial_compatibility_residual,
            'initial_constraint_residual':self.initial_constraint_residual,'shifted_trial_functionals':self.shifted_trial_functionals})


def assemble_cpp_evolution(method,problem,cloud):
    from .global_flow_backend import _boundary_key
    a=dense_arithmetic(method.kernel,method.precision);d=cloud.dimension;identity=Identity(d)
    if not len(cloud.interior):raise ValueError('Evolution requires interior points')
    template=_EvolutionSlice(problem.operator,0,tuple(replace(bc,rhs=0) for bc in problem.boundary))
    boundary=boundary_data(template,cloud)
    target=bind_operators([boundary[i][0] if i in boundary else problem.operator for i in range(len(cloud.points))],cloud.points)
    shifted=not any(any(not any(alpha) for alpha,_ in op.terms) for op in target)
    source=[op+identity if shifted and i not in boundary else op for i,op in enumerate(target)] if method.scheme=='symmetric' else [identity]*len(target)
    plan,ids,initial_id=scalar_data_bundle(problem,d,problem.initial)
    system=assemble_scalar_native(method,cloud.points,target,source,[0]*len(target),data_plan=plan,
        factor_now=False,system_class=CppScalarEvolutionSystem,arithmetic=a)
    system.problem,system.cloud=problem,cloud;system.boundary_operators={i:op for i,(op,_) in boundary.items()}
    system.shifted_trial_functionals=shifted and method.scheme=='symmetric'
    system._data_problem=problem;system._data_points=cloud.points.copy();system._data_boundary=_boundary_key(cloud)
    system._data_key=(problem.rhs,tuple((bc.on if isinstance(bc.on,str) else tuple(bc.on),bc.rhs) for bc in problem.boundary))
    system._native_rows=scalar_data_rows(problem,cloud,ids);system._initial_program=initial_id;system._prepared_initial=problem.initial
    system._mass=NativeMatrixView(system,'MASS_EXPORT');system._initial_matrix=NativeMatrixView(system,'INITIAL_EXPORT')
    rows=[{} for _ in range(system.size)]
    for j,node in enumerate(cloud.boundary_indices):rows[int(node)][j]=a.number(1)
    system.boundary_map=_sparse(rows,a,len(cloud.boundary_indices))
    try:
        if system._native_rows is not None:
            system.worker.request('DATA_INIT '+str(len(target))+' '+' '.join(map(str,system._native_rows)))
        tokens=['MASS_INDEXED',str(len(cloud.interior_indices))]
        for node in cloud.interior_indices:tokens.extend([str(int(node)),*_op_tokens(identity,system.encode)])
        system.worker.request(' '.join(tokens))
        initial_rows=[boundary[i][0] if i in boundary else identity for i in range(len(target))]
        tokens=['INITIAL_MATRIX',str(len(target))]
        for op in initial_rows:tokens.extend(_op_tokens(op,system.encode))
        system.worker.request(' '.join(tokens))
    except Exception:system.worker.close(force=True);raise
    return system
