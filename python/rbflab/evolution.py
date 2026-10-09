"""Scalar u_t + L u = f with fixed spatial and boundary operators.

Exposes M, A, and a boundary map C in M z_dot + A z = forcing(t)+C g(t).
Local Hermite unknowns have M=I-SL, not an identity mass matrix.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from .operators import Identity, bind_operators
from .problems import LinearPDE, boundary_data
from .precision import MPFactor
from .nodal import Arithmetic, NodalBasis, NodalSolution, dense_arithmetic
from .sparse_precision import MPSparseMatrix, MPSparseLU
from .time_data import at_time


class _EvolutionSlice(LinearPDE):
    # Constant modes are determined by the initial condition and time derivative.
    # This marker bypasses only the stationary constant-nullspace rejection.
    _evolution_slice = True


@dataclass(frozen=True)
class EvolutionPDE:
    """u_t + operator(u) = rhs(points,t); initial(points) is at start_time.

    Spatial coefficients and boundary operators are fixed. Boundary values may
    depend on time. Extended callbacks use TimeData and PrecisionData.
    """
    operator: object
    rhs: object
    initial: object
    boundary: tuple

    def __post_init__(self):
        object.__setattr__(self,"boundary",tuple(self.boundary))
        LinearPDE(self.operator,0,self.boundary)

    def at(self,time):
        return _EvolutionSlice(self.operator,at_time(self.rhs,time),
            tuple(replace(bc,rhs=at_time(bc.rhs,time)) for bc in self.boundary))

    def solve(self,cloud,method,dt,steps,**kwargs):
        return method.assemble(self,cloud).solve(dt,steps,**kwargs)


class BoundaryMap:
    """Rectangular sparse MP boundary map; never converts entries to Float64."""
    def __init__(self,rows,arithmetic,ncols):
        self.rows,self.arithmetic=rows,arithmetic
        self.shape=(len(rows),ncols)
        self.nnz=sum(len(row) for row in rows)

    def matvec(self,vector):
        a=self.arithmetic
        return a.vector([a.ctx.fsum(v*vector[j] for j,v in row.items()) for row in self.rows])


def _sparse(rows,a,ncols=None):
    ncols=len(rows) if ncols is None else ncols
    if a.ctx:
        return MPSparseMatrix(a.ctx,rows) if ncols==len(rows) else BoundaryMap(rows,a,ncols)
    entries=[(i,j,v) for i,row in enumerate(rows) for j,v in row.items() if v!=0]
    return coo_matrix(([v for i,j,v in entries],([i for i,j,v in entries],
        [j for i,j,v in entries])),shape=(len(rows),ncols)).tocsc()


def _size(matrix):
    return matrix.shape[0] if hasattr(matrix,"shape") else matrix.rows


def _mv(matrix,vector):
    if hasattr(matrix,"matvec"):
        return matrix.matvec(vector)
    return matrix@vector if issparse(matrix) or isinstance(matrix,np.ndarray) else matrix*vector


def _shift(spatial,mass,alpha,a):
    if isinstance(spatial,MPSparseMatrix):
        rows=[dict(row) for row in spatial.rows]
        for i,row in enumerate(mass.rows):
            for j,v in row.items():rows[i][j]=rows[i].get(j,a.ctx.zero)+alpha*v
        return MPSparseMatrix(a.ctx,rows)
    return spatial+alpha*mass


def _factor(matrix,a):
    if isinstance(matrix,MPSparseMatrix):return MPSparseLU(matrix)
    if issparse(matrix):return splu(matrix.tocsc())
    return a.factor(matrix)


def assemble_evolution(method,problem,cloud):
    from .methods import GlobalCollocation,LHI
    from .rbf_fd import RBFFD
    # Validate coverage and fixed boundary functionals without evaluating timed data.
    template=_EvolutionSlice(problem.operator,0,tuple(replace(bc,rhs=0) for bc in problem.boundary))
    boundary=boundary_data(template,cloud)
    if not len(cloud.interior):raise ValueError("Evolution requires interior points")
    identity=Identity(cloud.dimension)
    bi=cloud.boundary_indices;ii=cloud.interior_indices
    if isinstance(method,GlobalCollocation):
        if method.scheme not in ("symmetric","asymmetric"):raise ValueError("Unknown global scheme")
        a=dense_arithmetic(method.kernel,method.precision)
        ops=[boundary[i][0] if i in boundary else problem.operator for i in range(len(cloud.points))]
        source_ops=bind_operators(ops,cloud.points)
        # Pure derivative functionals annihilate constants. A fixed shifted
        # interior trial functional retains that mode in the Hermite basis;
        # the assembled PDE rows remain L, not L+I.
        shifted=not any(any(not any(alpha) for alpha,_ in op.terms) for op in source_ops)
        if shifted:
            source_ops=[op if i in boundary else op+identity for i,op in enumerate(source_ops)]
        basis=NodalBasis(a,cloud.points,method.polynomial_degree,
                         source_operators=source_ops if method.scheme=="symmetric" else None)
        spatial=basis.system_matrix(cloud.points,ops)
        n=len(cloud.points)+len(basis.powers)
        mass=a.zeros(n,n)
        values=basis.evaluation(cloud.interior,[identity]*len(ii))
        for row,node in enumerate(ii):
            for j in range(n):mass[int(node),j]=values[row,j]
        boundary_rows=[{} for _ in range(n)]
        for j,node in enumerate(bi):boundary_rows[int(node)][j]=a.number(1)
        system=EvolutionSystem(problem,cloud,a,spatial,mass,_sparse(boundary_rows,a,len(bi)),"global")
        system.basis=basis
        system.initial_matrix=basis.system_matrix(cloud.points,
            [boundary[i][0] if i in boundary else identity for i in range(len(cloud.points))])
        system.shifted_trial_functionals=shifted and method.scheme=="symmetric"
    elif isinstance(method,(LHI,RBFFD)):
        base=method.assemble(template,cloud)
        if isinstance(method,RBFFD):
            a=base.arithmetic;n=len(cloud.points)
            mass_rows=[{i:a.number(1)} if i not in boundary else {} for i in range(n)]
            boundary_rows=[{} for _ in range(n)]
            for j,node in enumerate(bi):boundary_rows[int(node)][j]=a.number(1)
            kind="rbffd"
        else:
            a=Arithmetic(method.kernel)
            if method.precision.global_dtype=="mpmath":
                a.backend=base.stencils[0].factor.backend;a.ctx=a.backend.ctx
            n=len(ii);interior_map={int(node):i for i,node in enumerate(ii)}
            boundary_map={int(node):i for i,node in enumerate(bi)}
            mass_rows=[{i:a.number(1)} for i in range(n)]
            boundary_rows=[{} for _ in range(n)]
            sl_rows=[{} for _ in range(n)]
            for row,s in enumerate(base.stencils):
                weights=s.weights_mp if a.ctx else s.weights
                ns,nb=len(s.solution_indices),len(s.boundary_indices)
                for node,w in zip(s.boundary_indices,weights[ns:ns+nb]):
                    boundary_rows[row][boundary_map[int(node)]]=-a.number(w)
                for node,w in zip(s.pde_indices,weights[ns+nb:]):
                    j=interior_map[int(node)];w=a.number(w)
                    sl_rows[row][j]=sl_rows[row].get(j,a.number(0))+w
                    mass_rows[row][j]=mass_rows[row].get(j,a.number(0))-w
            kind="lhi"
        system=EvolutionSystem(problem,cloud,a,base.matrix,_sparse(mass_rows,a),
                               _sparse(boundary_rows,a,len(bi)),kind)
        system.base=base
        if kind=="rbffd":
            # Preserve initial interior values and solve the algebraic boundary
            # rows for boundary nodal values in the selected arithmetic.
            if a.ctx:
                rows=[dict(base.matrix.rows[i]) if i in boundary else {i:a.number(1)} for i in range(n)]
            else:
                csr=base.matrix.tocsr()
                rows=[dict(zip(csr.getrow(i).indices,csr.getrow(i).data)) if i in boundary else {i:1.0} for i in range(n)]
            system.initial_matrix=_sparse(rows,a)
        if kind=="lhi":
            system.SL=_sparse(sl_rows,a)
            system.tree=cKDTree(cloud.interior)
    else:
        raise TypeError("Unsupported scalar evolution discretization")
    system.boundary_operators={i:op for i,(op,_) in boundary.items()}
    return system


class EvolutionSystem:
    """Assembled system M z_dot + A z = forcing(t) + C g(t).

    spatial is A, mass is M, and boundary_map is C. Global unknowns
    differ by method. In LHI, M = I - SL, where SL contains weights of
    nearby PDE centers; it is not the identity mass of nodal RBF-FD.
    Create with method.assemble(problem, cloud).
    """
    def __init__(self,problem,cloud,arithmetic,spatial,mass,boundary_map,kind):
        self.problem,self.cloud,self.arithmetic=problem,cloud,arithmetic
        self.spatial,self.mass,self.boundary_map=spatial,mass,boundary_map
        self.kind=kind;self._factors={};self._initial_factor=None

    @property
    def matrix(self):
        """Spatial operator A in M*z_dot + A*z = rhs(t)."""
        return self.spatial

    @matrix.setter
    def matrix(self,value):
        self.spatial=value

    def data(self,time):
        a=self.arithmetic;p=self.problem.at(time)
        if a.ctx:
            for item in [p.rhs]+[bc.rhs for bc in p.boundary]:
                if callable(item) and not hasattr(item,"mp_values"):
                    raise TypeError("Extended evolution requires TimeData callbacks")
        bd=boundary_data(p,self.cloud,ctx=a.ctx)
        g=a.vector([bd[int(j)][1] for j in self.cloud.boundary_indices])
        f=a.vector(a.data(p.rhs,self.cloud.interior))
        if self.kind=="lhi":forcing=_mv(self.mass,f)
        else:
            forcing=a.vector([0]*_size(self.mass))
            for row,node in enumerate(self.cloud.interior_indices):forcing[int(node)]=f[row]
        return forcing+_mv(self.boundary_map,g),f,g

    def _initial(self,time):
        a=self.arithmetic;cloud=self.cloud
        data=a.vector(a.data(self.problem.initial,cloud.points))
        _,_,g=self.data(time)
        tolerance=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        initial_data=self.problem.initial
        mismatch=a.number(0)
        for node,v in zip(cloud.boundary_indices,g):
            op=self.boundary_operators[int(node)]
            point=cloud.points[[int(node)]]
            if op==Identity(cloud.dimension):value=data[int(node)]
            elif hasattr(initial_data,"operator_values"):
                value=initial_data.operator_values(op,point,ctx=a.ctx)[0]
            elif not callable(initial_data) and np.ndim(initial_data)==0:
                value=sum(a.number(c)*a.number(initial_data) for alpha,c in op.terms if not any(alpha))
            else:
                raise TypeError("Derivative boundary initialization needs SymbolicData or InitialData with analytic derivatives")
            if not (a.ctx.isfinite(value) if a.ctx else np.isfinite(value)):
                raise ValueError("Initial boundary derivatives must be finite")
            error=abs(value-v)/(1+abs(v));mismatch=max(mismatch,error)
            if error>tolerance:
                raise ValueError("Initial data are incompatible with boundary data at start_time")
        self.initial_compatibility_residual=float(mismatch)
        initial=_InitialData(initial_data,a,cloud.dimension,time)
        if self.kind in ("global","rbffd"):
            target=data.copy()
            for node,v in zip(cloud.boundary_indices,g):target[int(node)]=v
            if self.kind=="global":target=self.basis.padded(target)
            if self._initial_factor is None:self._initial_factor=_factor(self.initial_matrix,a)
            z=self._initial_factor.solve(target)
            self.initial_constraint_residual=float(a.norm(_mv(self.initial_matrix,z)-target)/(a.norm(target) or a.number(1)))
            if self.kind=="global":initial=NodalSolution(SimpleNamespace(basis=self.basis),z,{})
            else:
                from .rbf_fd import FDSolution
                initial=FDSolution(self.base,z,{})
        else:
            z=a.vector([data[int(j)] for j in cloud.interior_indices])
            self.initial_constraint_residual=None
        return z,initial

    def solve(self,dt,steps,*,scheme="bdf2",start_time=0) -> "EvolutionTrajectory":
        """Advance at fixed dt with backward Euler or BE-started BDF2.

        Args:
            dt (float, str, or Fraction): Positive time step; exact inputs retain decimal
                intent on conversion to the selected arithmetic.
            steps (int): Positive number of steps.
            scheme (str): "backward_euler" or "bdf2".
            start_time (float, str, or Fraction): Time of the supplied initial field.

        Returns:
            EvolutionTrajectory: Initial field, saved states, and diagnostics.
                A + alpha*M is factored once per distinct alpha.
        """
        dt=Fraction(dt);start_time=Fraction(start_time)
        if dt<=0 or type(steps) is not int or steps<1:raise ValueError("Positive dt and integer steps required")
        if scheme not in ("backward_euler","bdf2"):raise ValueError("Unknown time scheme")
        a=self.arithmetic
        previous,initial=self._initial(a.number(start_time));previous2=None
        states=[];new_factors=0
        for step in range(1,steps+1):
            second=scheme=="bdf2" and step>1
            alpha=(Fraction(3,2) if second else Fraction(1))/dt
            weights=(2/dt,-Fraction(1,2)/dt) if second else (1/dt,)
            time=a.number(start_time+step*dt)
            history=a.number(weights[0])*previous
            if second:history+=a.number(weights[1])*previous2
            rhs,f,g=self.data(time)
            # History acts on global unknowns through M. Using identity
            # here would change the LHI semidiscrete method.
            rhs=rhs+_mv(self.mass,history)
            if alpha not in self._factors:
                matrix=_shift(self.spatial,self.mass,a.number(alpha),a)
                self._factors[alpha]=(matrix,_factor(matrix,a));new_factors+=1
            matrix,factor=self._factors[alpha]
            z=factor.solve(rhs);zdot=a.number(alpha)*z-history
            residual=a.norm(_mv(matrix,z)-rhs)/(a.norm(rhs) or a.number(1))
            old=[states[-1] if states else initial]
            if second:old.append(states[-2] if len(states)>1 else initial)
            diagnostics={"relative_residual":float(residual),
                         "scheme":"bdf2" if second else "backward_euler"}
            if self.kind=="global":
                spatial=NodalSolution(SimpleNamespace(basis=self.basis),z,diagnostics)
            elif self.kind=="rbffd":
                from .rbf_fd import FDSolution
                spatial=lambda z=z,diagnostics=diagnostics:FDSolution(self.base,z,diagnostics)
            else:spatial=lambda z=z,f=f,zdot=zdot,g=g:_LHIField(self,z,f-zdot,g)
            states.append(EvolutionState(self,spatial,z,zdot,time,a.number(alpha),old,
                                         [a.number(w) for w in weights],diagnostics))
            previous2=previous;previous=z
        return EvolutionTrajectory(initial,states,{"scheme":scheme,"steps":steps,"dt":str(dt),
            "new_factorizations":new_factors,"cached_factorizations":len(self._factors),
            "initial_factorizations":int(self._initial_factor is not None),
            "global_digits":a.ctx.dps if a.ctx else None,"kind":self.kind,
            "unknowns":_size(self.mass),
            "initial_compatibility_residual":self.initial_compatibility_residual,
            "initial_constraint_residual":self.initial_constraint_residual,
            "shifted_trial_functionals":getattr(self,"shifted_trial_functionals",False)})


class _InitialData:
    def __init__(self,data,arithmetic,dimension,time):
        self.data,self.arithmetic,self.dimension,self.time=data,arithmetic,dimension,time

    def evaluate(self,points,operator=None):
        return np.array([float(v) for v in self._values(points,operator)])

    def _values(self,points,operator):
        from .methods import _query
        points=_query(points,self.dimension)
        if operator is not None and operator!=Identity(self.dimension):
            raise NotImplementedError("Initial LHI data provide values only, not a spatial reconstruction")
        return self.arithmetic.vector(self.arithmetic.data(self.data,points))

    def evaluate_mp(self,points,operator=None):
        if not self.arithmetic.ctx:raise ValueError("Extended arithmetic is not enabled")
        return self._values(points,operator)


class _LHIField:
    def __init__(self,system,values,pde_values,boundary_values):
        self.system=system;self._coefficients=[]
        cloud=system.cloud
        interior=dict(zip(cloud.interior_indices,values))
        pde=dict(zip(cloud.interior_indices,pde_values))
        boundary=dict(zip(cloud.boundary_indices,boundary_values))
        self.tree=system.tree
        for s in system.base.stencils:
            data=[interior[j] for j in s.solution_indices]+[boundary[j] for j in s.boundary_indices]+[pde[j] for j in s.pde_indices]
            if s.basis:data=s.basis.padded(data)
            self._coefficients.append(s.factor.solve(data))

    def _evaluate(self,points,operator=None):
        from .methods import _query
        points=_query(points,self.system.cloud.dimension)
        if not len(points):return []
        _,owners=self.tree.query(points);out=[]
        op=operator or Identity(self.system.cloud.dimension)
        for point,owner in zip(points,owners):
            s=self.system.base.stencils[int(owner)];q=point[None,:]
            if s.basis:matrix=s.basis.evaluation(q,[op])
            elif isinstance(s.factor,MPFactor):matrix=s.factor.backend.matrix(q,[op],s.points,s.operators)
            else:
                from .assembly import functional_matrix
                matrix=functional_matrix(self.system.base.kernel,q,[op],s.points,s.operators)
            out.append(_mv(matrix,self._coefficients[int(owner)])[0])
        return out

    def evaluate(self,points,operator=None):return np.array([float(v) for v in self._evaluate(points,operator)])

    def evaluate_mp(self,points,operator=None):
        factor=self.system.base.stencils[0].factor
        if not isinstance(factor,MPFactor):raise ValueError("Extended local arithmetic is not enabled")
        return factor.ctx.matrix(self._evaluate(points,operator))


class EvolutionState:
    """One time level and its reconstructable spatial field.

    unknown_derivative is the BE/BDF2 difference quotient, not the
    continuous time derivative. evaluate() delegates to the method's
    spatial reconstruction.
    """
    def __init__(self,system,spatial,unknowns,derivative,time,alpha,history,weights,diagnostics):
        self.system,self._spatial,self.unknowns,self.unknown_derivative=system,spatial,unknowns,derivative
        self.time,self.alpha,self.history,self.history_weights=time,alpha,history,weights
        self.diagnostics=diagnostics

    @property
    def spatial(self):
        if callable(self._spatial):self._spatial=self._spatial()
        return self._spatial

    def evaluate(self,points,operator=None):return self.spatial.evaluate(points,operator)
    def evaluate_mp(self,points,operator=None):return self.spatial.evaluate_mp(points,operator)

    def residuals(self,points,*,extended=False):
        """Off-node BE/BDF2 time difference + L(reconstruction) - f.

        This is not an exact continuous-time derivative or an error estimator.
        """
        a=self.system.arithmetic
        if extended and not a.ctx:raise ValueError("Extended residuals require global MP arithmetic")
        evaluate=lambda field,op=None: field.evaluate_mp(points,op) if a.ctx else field.evaluate(points,op)
        residual=self.alpha*evaluate(self)+evaluate(self,self.system.problem.operator)
        for old,w in zip(self.history,self.history_weights):residual-=w*evaluate(old)
        residual-=a.vector(a.data(at_time(self.system.problem.rhs,self.time),points))
        return residual if extended else np.array([float(v) for v in residual])


@dataclass
class EvolutionTrajectory:
    """Initial field, saved time states, and evolution diagnostics.

    final returns the last saved state; states excludes the initial field.
    Diagnostics include scheme, step count, and factorization reuse.
    """
    initial: object
    states: list
    diagnostics: dict

    @property
    def final(self):return self.states[-1]
