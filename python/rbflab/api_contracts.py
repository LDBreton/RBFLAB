"""Shared preflight, preparation and assembled-system contracts."""
import copy
from dataclasses import replace, asdict
from functools import wraps
import hashlib
import numpy as np
from scipy.sparse import issparse
from .configuration import LocalSolver

class MethodContract:
    def preflight(self, problem=None, cloud=None, *, operation="assemble", space=None):
        """Validate supported combinations without compiling or assembling matrices.

        Raises for unsupported combinations. Geometry rank is checked when local
        matrices are built. Returns an effective recipe, not an accuracy guarantee.
        """
        from .precision import Precision
        from .spaces import SpaceStokesProblem
        from .symbolic_system import BlockPDE
        from .evolution import EvolutionPDE
        from .lhi_backends import PythonBackend, CppBackend
        from .torch_backend import TorchBackend
        from .stencils import StencilPolicy
        if operation not in ('assemble','operators','prepare'):raise ValueError('Unknown preflight operation')
        from .methods import GlobalCollocation, LHI
        from .rbf_fd import RBFFD
        name = 'GlobalCollocation' if isinstance(self,GlobalCollocation) else 'RBFFD' if isinstance(self,RBFFD) else 'LHI'
        precision=self.precision
        if not isinstance(precision,Precision):raise TypeError('Expected Precision')
        if self.kernel is not None and self.spaces is not None:raise ValueError('Specify kernel or spaces, not both')
        if self.kernel is None and not self.spaces:raise ValueError('Specify kernel or spaces')
        if self.spaces and self.polynomial_degree not in (None,'auto'):
            raise ValueError('Specify polynomial degree on spaces, not on both method and spaces')
        flow=isinstance(problem,SpaceStokesProblem);block=isinstance(problem,BlockPDE)
        evolution=isinstance(problem,EvolutionPDE)
        backend=getattr(self,'local_backend',None)
        if backend is not None and not isinstance(backend,(PythonBackend,CppBackend,TorchBackend)):
            raise TypeError('Stable methods support PythonBackend, CppBackend and TorchBackend only')
        if isinstance(backend,TorchBackend) and precision.local_digits is not None:
            raise NotImplementedError('TorchBackend requires Float64 local arithmetic')
        if operation=='prepare' and block:
            raise NotImplementedError('Block symbolic kernel preparation is not implemented; bind/compile each scalar kernel explicitly')
        if name=='GlobalCollocation':
            if self.scheme not in ('symmetric','asymmetric'):raise ValueError('Unknown global scheme')
            if precision.local_digits is not None or precision.global_dtype!='float64':raise ValueError('GlobalCollocation uses global_digits')
            if (flow or block) and self.scheme!='symmetric':raise NotImplementedError('Space/block global assembly currently requires symmetric collocation')
        else:
            if not isinstance(self.stencil_policy,StencilPolicy):raise TypeError('Expected StencilPolicy')
            if not isinstance(self.local_solver,LocalSolver):raise TypeError('Expected LocalSolver')
            if precision.global_digits is not None:raise ValueError('Local methods use local_digits and global_dtype')
            if self.local_solver.method!='lu' and not (flow and isinstance(backend,CppBackend) and precision.local_digits is None):
                raise NotImplementedError('SVD currently supports C++ Float64 Stokes LHI only')
            if self.stencil_policy.shape_rule!='fixed' and not flow:
                raise NotImplementedError('Per-stencil shape policies currently support Stokes LHI only; scalar kernels use fixed parameters/local scaling')
        if name=='RBFFD':
            if self.scheme not in ('standard','symmetric','boundary_hermite'):raise ValueError('Unknown RBF-FD scheme')
            if flow or block:raise NotImplementedError('RBFFD mixed PDE assembly is not implemented; use source-target operators')
            if operation=='operators' and self.scheme!='standard':raise NotImplementedError('operators/weights currently require standard nodal RBF-FD')
            from .discrete_operators import resolve_space
            descriptor=resolve_space(self,space)
            descriptor.degree()
            from .spaces import DivergenceFreeSpace
            if operation!='operators' and isinstance(descriptor,DivergenceFreeSpace):raise NotImplementedError('DivergenceFreeSpace uses RBFFD.operators, not scalar PDE assembly')
        if name=='LHI':
            if self.centers is not None:
                if flow or block or evolution:raise NotImplementedError('Independent center groups support stationary scalar LHI; transient PDE data require additional mass maps')
                if getattr(problem,'_evolution_slice',False):raise NotImplementedError('Independent transient LHI groups are not implemented')
                if self.pde_stencil_size is not None:raise ValueError('Use group sizes instead of pde_stencil_size with explicit centers')
                from .centers import CenterGroup
                if not isinstance(self.centers,dict) or not self.centers or any(not isinstance(g,CenterGroup) for g in self.centers.values()):raise TypeError('centers must map names to CenterGroup')
                if cloud is not None:
                    for g in self.centers.values():
                        if g.points.shape[1]!=cloud.dimension:raise ValueError('Group/cloud dimensions differ')
                        if g.indices is not None and len(g.indices)!=len(cloud.interior):raise ValueError('One explicit membership row is required per interior target')
            if not flow and backend is not None:
                if not isinstance(backend,PythonBackend):raise NotImplementedError('Scalar LHI currently supports PythonBackend only')
                if backend.workers!=1 or not backend.compute_condition:raise NotImplementedError('Scalar LHI currently uses one worker and condition estimation')
            if flow:
                from .space_stokes import select_spaces
                _,_,vd,pd=select_spaces(self,problem)
                if cloud is not None and cloud.dimension!=2:raise NotImplementedError('Stokes LHI currently supports 2D only')
                if (vd is None and pd is not None) or (vd is not None and (vd<2 or vd>4 or pd!=vd-1)):raise NotImplementedError('Stokes LHI requires no tails or velocity degree 2..4 and pressure degree one lower')
                if isinstance(backend,TorchBackend) and vd is not None:raise NotImplementedError('Torch Stokes LHI currently supports unaugmented spaces only')
                if isinstance(backend,CppBackend) and self.stencil_policy.scaling!='physical':raise NotImplementedError('C++ Stokes LHI requires physical kernel coordinates')
                if self.stencil_policy.shape_rule=='hardy' and (vd is not None or self.stencil_policy.scaling!='physical' or self.stencil_policy.selection!='nearest' or self.min_boundary_centers or self.pde_stencil_size is not None):raise NotImplementedError('Hardy Stokes scaling needs unaugmented physical nearest stencils with the default PDE selection')
                if isinstance(backend,TorchBackend) and self.stencil_policy.shape_rule not in ('fixed','hardy'):raise NotImplementedError('Torch Stokes supports fixed or Hardy shape policies')
        if name!='RBFFD' and not flow and not block:
            from .spaces import ScalarSpace, DivergenceFreeSpace
            from .kernels import validate_polynomial_degree
            if self.spaces:
                if len(self.spaces)!=1:raise ValueError('Scalar problems require one ScalarSpace')
                descriptor=next(iter(self.spaces.values()))
                if not isinstance(descriptor,ScalarSpace) or isinstance(descriptor,DivergenceFreeSpace):raise TypeError('Scalar problems require ScalarSpace')
                descriptor.degree()
            else:validate_polynomial_degree(self.kernel,self.polynomial_degree)
        if block:
            from .spaces import ScalarSpace
            if not self.spaces or set(self.spaces)!=set(problem.fields):raise ValueError('Block problems require one space for each symbolic field')
            if any(type(s) is not ScalarSpace for s in self.spaces.values()):raise TypeError('Block fields require ScalarSpace')
            if len({s.degree() for s in self.spaces.values()})!=1:raise NotImplementedError('Block spaces currently require a common polynomial degree')
            if backend is not None:raise NotImplementedError('Block assembly currently uses its Python implementation; omit local_backend')
        if cloud is not None and problem is not None:
            d=getattr(problem,'dimension',getattr(getattr(problem,'operator',None),'dimension',None))
            if d is not None and d!=cloud.dimension:raise ValueError('Problem and cloud dimensions differ')
        return recipe(self,problem)

def recipe(method,problem=None):
    p=method.precision
    return dict(method=type(method).__name__,scheme=getattr(method,'scheme','Hermite'),
        kernel=repr(method.kernel),spaces={str(k):repr(v) for k,v in (method.spaces or {}).items()},
        polynomial_degree=method.polynomial_degree,independent_centers=getattr(method,'centers',None) is not None,
        stencil_policy=asdict(method.stencil_policy) if hasattr(method,'stencil_policy') else None,
        local_solver=asdict(method.local_solver) if hasattr(method,'local_solver') else None,
        backend=repr(getattr(method,'local_backend',None)) if getattr(method,'local_backend',None) is not None else 'Python (default)',precision=asdict(p),
        arithmetic=dict(coordinates='float64',local_digits=p.local_digits,global_digits=p.global_digits,
                        sparse_storage=p.global_dtype,global_solve_digits=p.global_digits if type(method).__name__=='GlobalCollocation' else p.local_digits if p.global_dtype=='mpmath' else None,default_output='float64'),
        reconstruction='nearest-stencil' if type(method).__name__ in ('LHI','RBFFD') else 'global expansion')

def prepared(fn):
    @wraps(fn)
    def prepare(self,problem,*args,**kwargs):
        self.preflight(problem,operation='prepare')
        method=copy.deepcopy(self)
        from .problems import LinearPDE
        from .evolution import EvolutionPDE
        if method.spaces and isinstance(problem,(LinearPDE,EvolutionPDE)):
            from .discrete_operators import scalar_method
            method=scalar_method(method)
        return fn(method,problem,*args,**kwargs)
    return prepare

def fingerprint(matrix):
    h=hashlib.sha256()
    if issparse(matrix):
        m=matrix.tocsr();h.update(repr(m.shape).encode())
        for x in (m.data,m.indices,m.indptr):h.update(x.tobytes())
    elif isinstance(matrix,np.ndarray):h.update(repr(matrix.shape).encode());h.update(matrix.tobytes())
    elif isinstance(getattr(matrix,'rows',None),list):
        h.update(repr(matrix.shape).encode())
        for row in matrix.rows:
            h.update(repr([(j,getattr(v,'_mpf_',v)) for j,v in sorted(row.items())]).encode())
    elif hasattr(matrix,'cols'):
        h.update(repr((matrix.rows,matrix.cols)).encode())
        h.update(repr([getattr(v,'_mpf_',v) for v in matrix]).encode())
    else:h.update(repr(matrix).encode())
    return h.digest()

def expose_system(system,method,cloud,problem):
    system.recipe=recipe(method,problem)
    owner=getattr(system,'base',system)
    stencils=getattr(owner,'stencils',None)
    if hasattr(owner,'backend_diagnostics'):system.recipe['backend_diagnostics']=copy.deepcopy(owner.backend_diagnostics)
    if stencils is not None:
        system.recipe['stencils']=[dict(groups={k:np.asarray(v).tolist() for k,v in getattr(s,'groups',{}).items()},
              functionals=len(getattr(s,'points',getattr(getattr(s,'basis',None),'centers',[]))),
              polynomial_terms=len(getattr(getattr(s,'basis',None),'powers',getattr(getattr(s,'polynomial_basis',None),'columns',[]))),
              slots=list(getattr(s,'slots',[])),
              kernel=repr(getattr(getattr(s,'local_arithmetic',None),'kernel',method.kernel)),
              pressure_kernel=repr(getattr(getattr(getattr(s,'local_arithmetic',None),'pressure_arithmetic',None),'kernel',None))) for s in stencils]
        if hasattr(owner,'indices'):
            for record,indices in zip(system.recipe['stencils'],owner.indices):record['source_indices']=np.asarray(indices).tolist()
    if not hasattr(system,'matrix'):return system
    n=system.matrix.shape[1] if hasattr(system.matrix,'shape') else system.matrix.cols
    kind=type(system).__name__
    if kind in ('LHISystem','MPSparseLHISystem'): meaning='interior solution values';nodes=cloud.interior_indices.copy()
    elif kind=='FDSystem':meaning='nodal functionals';nodes=np.arange(len(cloud.points))
    elif kind=='LocalFlowSystem':meaning='component-major interior velocity values';nodes=np.tile(cloud.interior_indices,cloud.dimension)
    else:meaning='expansion coefficients or method-specific block unknowns';nodes=None
    system.dof_map=dict(kind=meaning,size=n,point_indices=nodes,
        operators=getattr(system,'dof_operators',getattr(system,'operators',None)))
    if kind in ('GlobalSystem','MPGlobalSystem'):
        system.dof_map.update(kind='functional expansion coefficients',points=system.centers.copy(),operators=system.operators)
    elif kind=='DenseNodalSystem':
        system.dof_map.update(kind='kernel coefficients followed by polynomial coefficients',points=system.basis.centers.copy(),operators=system.basis.source_operators,polynomial_powers=system.basis.powers)
    elif kind=='GlobalFlowSystem':
        system.dof_map.update(kind='vector functional coefficients followed by polynomial coefficients',points=system.points.copy(),operators=system.functionals,polynomial_columns=system.polynomials.columns)
    elif kind in ('BlockGlobalSystem','BlockLHISystem'):
        system.dof_map.update(fields=system.problem.fields,ordering='field-major; constraints follow physical unknowns')
    elif kind=='EvolutionSystem':
        system.dof_map.update(kind='global coefficients' if system.kind=='global' else 'interior values' if system.kind=='lhi' else 'nodal values',point_indices=cloud.interior_indices.copy() if system.kind=='lhi' else np.arange(len(cloud.points)))
    system.reconstruction=dict(kind=system.recipe['reconstruction'],
        pressure='gradient only' if kind=='LocalFlowSystem' else 'method-dependent')
    original=system.solve
    watched=['matrix']+[name for name in ('mass','initial_matrix','boundary_map') if hasattr(system,name)]
    previous=tuple(fingerprint(getattr(system,name)) for name in watched)
    shape=system.matrix.shape if hasattr(system.matrix,'shape') else (system.matrix.rows,system.matrix.cols)
    @wraps(original)
    def solve(*args,**kwargs):
        nonlocal previous
        current=tuple(fingerprint(getattr(system,name)) for name in watched)
        if current!=previous:
            current_shape=system.matrix.shape if hasattr(system.matrix,'shape') else (system.matrix.rows,system.matrix.cols)
            if current_shape!=shape:raise ValueError('Matrix dimensions changed; reassemble the method instead')
            if kind=='LocalFlowSystem':
                raise ValueError('Stokes LHI uses dependent row maps; reassemble after changes or export matrices to an external solver')
            if hasattr(system,'_factors'):system._factors.clear()
            if hasattr(system,'_initial_factor'):system._initial_factor=None
            if kind=='GlobalSystem':
                from .assembly import Factor
                system.factor=Factor(system.matrix)
            elif kind=='DenseNodalSystem':system.factor=system.basis.arithmetic.factor(system.matrix)
            elif kind=='MPGlobalSystem':
                from .precision import MPFactor
                system.factor=MPFactor(system.backend,system.matrix)
            elif hasattr(system,'factor'):system.factor=None
            previous=current
        result=original(*args,**kwargs)
        if not hasattr(result,'system'):result.system=system
        return result
    system.solve=solve
    return system

def assembled(fn):
    @wraps(fn)
    def assemble(self,problem,cloud):
        self.preflight(problem,cloud)
        from .problems import LinearPDE
        from .evolution import EvolutionPDE
        method=self
        if self.spaces and isinstance(problem,(LinearPDE,EvolutionPDE)):
            from .discrete_operators import scalar_method
            method=scalar_method(self)
        from .rbf_fd import RBFFD
        if isinstance(method,RBFFD) and method.polynomial_degree=='auto':method=replace(method,polynomial_degree=2)
        system=fn(method,problem,cloud)
        return expose_system(system,method,cloud,problem)
    return assemble
