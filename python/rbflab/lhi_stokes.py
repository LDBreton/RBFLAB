"""Compact local Hermite unsteady Stokes: (I-SL) Udot + SY U = (I-SL) f - SB g.

Six groups: solution velocity, boundary velocity, Stokes momentum, each in two
components. Pressure gradients are reconstructed locally; a global pressure
field/gauge is not defined. Nearest-patch velocity is only patchwise solenoidal.
"""
from .symbolic_kernel import BoundKernel
from dataclasses import dataclass,field
from fractions import Fraction
from math import fsum
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .precision import Precision
from .nodal import Arithmetic
from .kernels import ScalarKernel,Hybrid
from .operators import Identity,Derivative,Laplacian
from .methods import _query
from .stencils import StencilPolicy,neighbors,geometry_quality
from .stokes import _blocks,StokesArithmetic
from .stokes_polynomials import StokesPolynomialBasis,append_columns,rank_ratio
from .unsteady_stokes import UnsteadyStokesProblem,_at,_InitialVelocity,StokesTrajectory
from .sparse_precision import MPSparseMatrix,MPSparseLU


def _matrix(rows,a):
    if a.ctx:return MPSparseMatrix(a.ctx,rows)
    entries=[(i,j,v) for i,row in enumerate(rows) for j,v in row.items() if v!=0]
    return coo_matrix(([v for i,j,v in entries],([i for i,j,v in entries],
        [j for i,j,v in entries])),shape=(len(rows),len(rows))).tocsc()


def _mv(matrix,vector):
    return matrix.matvec(vector) if isinstance(matrix,MPSparseMatrix) else matrix@vector


def _add(row,col,value):row[col]=row.get(col,0)+value


@dataclass
class VectorHermiteStencil:
    center: int
    points: object
    functionals: list
    slots: list
    factor: object
    scale: object
    geometry_diagnostics: dict
    polynomial_basis: object = None

    def evaluation_matrix(self,a,points,functionals):
        a=getattr(self,'local_arithmetic',a)
        kernel=_blocks(a,points,functionals,self.points,self.functionals,self.scale)
        if self.polynomial_basis is None:return kernel
        return append_columns(a,kernel,self.polynomial_basis.matrix(points,functionals))


@dataclass
class LHIUnsteadyStokes:
    """Experimental compact Stokes method; check spatial stability and accuracy.

    Small solve residuals or patchwise zero divergence are not validation.
    pressure_kernel defaults to the velocity scalar potential. Hybrid potentials
    require polynomial_degree or explicit legacy_unaugmented_hybrid=True.
    Polynomial augmentation uses solenoidal velocity degree d and pressure degree
    d-1 modulo constants. It is experimental: reproduction does not imply stable
    time evolution. Velocity requires six continuous derivatives, pressure two.
    min_boundary_centers adds nearest boundary data only to stencils already
    touching the boundary. Existing solution centers are retained.
    """
    kernel: object
    stencil_size: int = 15
    precision: Precision = field(default_factory=Precision)
    stencil_policy: StencilPolicy = field(default_factory=StencilPolicy)
    pde_stencil_size: int | None = None
    pressure_kernel: object | None = None
    legacy_unaugmented_hybrid: bool = False
    polynomial_degree: int | None = None
    min_boundary_centers: int = 0
    local_backend: object = None

    def assemble(self,problem,cloud):
        if cloud.dimension != 2:
            raise NotImplementedError("Stokes solvers currently support 2D only")
        if not isinstance(problem,UnsteadyStokesProblem):raise TypeError("Expected UnsteadyStokesProblem")
        if self.polynomial_degree is not None and (type(self.polynomial_degree) is not int or not 2<=self.polynomial_degree<=4):
            raise ValueError("polynomial_degree must be None or an integer from 2 through 4")
        if type(self.min_boundary_centers) is not int or self.min_boundary_centers<0:
            raise ValueError("min_boundary_centers must be a nonnegative integer")
        pressure_kernel=self.kernel if self.pressure_kernel is None else self.pressure_kernel
        for kernel,order in ((self.kernel,6),(pressure_kernel,2)):
            if isinstance(kernel,Hybrid):
                if not self.legacy_unaugmented_hybrid and self.polynomial_degree is None:
                    raise ValueError("Set legacy_unaugmented_hybrid=True to use the experimental legacy hybrid without polynomial tails")
                if kernel.phs.power-1<order:
                    raise ValueError(f"This block requires continuous derivatives through order {order}")
            elif not isinstance(kernel,(ScalarKernel,BoundKernel)):
                raise NotImplementedError("Vector PHS polynomial augmentation is not implemented")
        if isinstance(pressure_kernel,Hybrid) and self.polynomial_degree is not None and self.polynomial_degree-1<pressure_kernel.phs.minimum_degree:
            raise ValueError("Pressure PHS requires a higher polynomial_degree")
        if not isinstance(self.precision,Precision) or self.precision.global_digits is not None:
            raise ValueError("LHI uses local_digits and global_dtype")
        if not isinstance(self.stencil_policy,StencilPolicy):raise TypeError("Expected StencilPolicy")
        try:valid_gauge=Fraction(problem.pressure_value)==0
        except (TypeError,ValueError):valid_gauge=False
        if not valid_gauge:raise NotImplementedError("LHI reconstructs pressure gradients only; no pressure-value gauge is supported")
        if self.local_backend is not None:
            from .legacy_cpp import LegacyCppLHIBackend
            from .lhi_backends import PythonBackend,CppBackend
            from .cuda_backend import CudaLHIBackend
            from .torch_backend import TorchBackend
            if not isinstance(self.local_backend,(LegacyCppLHIBackend,PythonBackend,CppBackend,CudaLHIBackend,TorchBackend)):raise TypeError('Expected an LHI numerical backend')
            return self.local_backend.assemble(self,problem,cloud)
        a=StokesArithmetic(self.kernel,self.pressure_kernel,self.precision.local_digits)
        g=a if self.precision.global_dtype=='mpmath' else Arithmetic(self.kernel)
        ii,bi=cloud.interior_indices,cloud.boundary_indices;ni,nb=len(ii),len(bi)
        if not ni or not nb:raise ValueError("Interior and boundary nodes are required")
        if type(self.stencil_size) is not int or not 3<=self.stencil_size<=len(cloud.points):
            raise ValueError("Invalid stencil_size")
        if self.pde_stencil_size is not None and (type(self.pde_stencil_size) is not int or not 0<=self.pde_stencil_size<ni):
            raise ValueError("Invalid pde_stencil_size")
        imap={int(j):i for i,j in enumerate(ii)};bmap={int(j):i for i,j in enumerate(bi)}
        if self.min_boundary_centers>nb:raise ValueError("min_boundary_centers exceeds boundary count")
        tree=cKDTree(cloud.points);ptree=cKDTree(cloud.interior);btree=cKDTree(cloud.points[bi])
        momentum=[{c:-(Laplacian()*problem.viscosity),2:Derivative(c)} for c in (0,1)]
        sy=[{} for _ in range(2*ni)];sl=[{} for _ in range(2*ni)];sb=[{} for _ in range(2*ni)];stencils=[]
        for owner,center in enumerate(ii):
            selected=self.stencil_policy.select(tree,cloud.points[center],self.stencil_size)
            sc=[int(j) for j in selected if int(j) in imap]
            bc=[int(j) for j in selected if int(j) in bmap]
            if bc and len(bc)<self.min_boundary_centers:
                extra=bi[neighbors(btree,cloud.points[center],self.min_boundary_centers)]
                bc=sorted(set(bc)|set(map(int,extra)))
            pc=[j for j in sc if j!=center]
            if self.pde_stencil_size is not None:
                candidate=ii[neighbors(ptree,cloud.points[center],self.pde_stencil_size+1)]
                pc=[int(j) for j in candidate if j!=center][:self.pde_stencil_size]
            if not pc:
                raise ValueError("Each Stokes stencil needs auxiliary PDE centers; enlarge the stencil")
            points=[];ops=[];slots=[]
            for group,ids,mapping,size in [('solution',sc,imap,ni),('boundary',bc,bmap,nb),('pde',pc,imap,ni)]:
                for c in (0,1):
                    for j in ids:
                        points.append(cloud.points[j]);ops.append(momentum[c] if group=='pde' else {c:Identity()})
                        slots.append((group,mapping[j]+c*size))
            points=np.asarray(points);scale=None
            if self.stencil_policy.scaling=='local':
                sqrt=a.ctx.sqrt if a.ctx else np.sqrt
                scale=max(sqrt(sum((a.number(v)-a.number(w))**2 for v,w in zip(point,cloud.points[center]))) for point in points)
            gram=_blocks(a,points,ops,points,ops,scale)
            basis=None
            diagnostics=geometry_quality(cloud.points[np.unique(sc+bc+pc)],cloud.points[center],1)
            if self.polynomial_degree is not None:
                sqrt=a.ctx.sqrt if a.ctx else np.sqrt
                radius=max(sqrt(sum((a.number(v)-a.number(w))**2 for v,w in zip(point,cloud.points[center]))) for point in points)
                basis=StokesPolynomialBasis(a,cloud.points[center],radius,self.polynomial_degree)
                poly=basis.matrix(points,ops);ratio=rank_ratio(poly)
                diagnostics['functional_polynomial_ratio']=ratio
                if ratio<1e-12:raise ValueError(f"Polynomial functionals are rank deficient at center {center}; enlarge or rebalance the stencil")
                n=len(points);m=len(basis.columns);aug=a.zeros(n+m,n+m)
                aug[:n,:n]=gram;aug[:n,n:]=poly;aug[n:,:n]=poly.T;gram=aug
            factor=a.factor(gram)
            target=np.repeat(cloud.points[[center]],2,axis=0)
            q=_blocks(a,target,momentum,points,ops,scale)
            if basis is not None:q=append_columns(a,q,basis.matrix(target,momentum))
            weights=factor.solve(q.T,transpose=True)
            for c in (0,1):
                row=owner+c*ni
                for j,(group,col) in enumerate(slots):
                    _add({'solution':sy,'boundary':sb,'pde':sl}[group][row],col,g.number(weights[j,c]))
            stencils.append(VectorHermiteStencil(int(center),points,ops,slots,factor,scale,
                diagnostics,basis))
        mass=[{i:g.number(1)} for i in range(2*ni)]
        for i,row in enumerate(sl):
            for j,v in row.items():_add(mass[i],j,-v)
        return LHIStokesSystem(problem,cloud,a,g,stencils,sy,sb,sl,mass)


class LHIStokesSystem:
    def __init__(self,problem,cloud,a,g,stencils,sy,sb,sl,mass):
        self.problem,self.cloud,self.local_arithmetic,self.arithmetic=problem,cloud,a,g
        self.stencils,self.sy_rows,self.sb_rows,self.sl_rows,self.mass_rows=stencils,sy,sb,sl,mass
        self.SY,self.SL,self.mass=_matrix(sy,g),_matrix(sl,g),_matrix(mass,g)
        self.tree=cKDTree(cloud.interior);self._factors={}

    def polynomial_consistency_diagnostics(self,degree=2):
        """Compact momentum defects on normalized solenoidal polynomials.

        Uses assembled weights in their global precision, including any cast
        from MP local solves. Reports relative componentwise moment defects.
        Geometry/rank screening remains Float64.
        """
        if type(degree) is not int or not 2<=degree<=4:raise ValueError("degree must be 2 through 4")
        a=self.arithmetic;rows=[];ni=len(self.cloud.interior)
        momentum=[{c:-(Laplacian()*self.problem.viscosity),2:Derivative(c)} for c in (0,1)]
        for owner,s in enumerate(self.stencils):
            basis=StokesPolynomialBasis(a,self.cloud.points[s.center],s.geometry_diagnostics['radius'],degree)
            poly=basis.matrix(s.points,s.functionals)
            target=basis.matrix(np.repeat(self.cloud.points[[s.center]],2,axis=0),momentum)
            defect=0.
            for c in (0,1):
                row=owner+c*ni
                weights=[{'solution':self.sy_rows,'boundary':self.sb_rows,'pde':self.sl_rows}[group][row].get(j,a.number(0)) for group,j in s.slots]
                for k in range(len(basis.columns)):
                    terms=[v*poly[j,k] for j,v in enumerate(weights)]
                    value=sum(terms,a.number(0));scale=max(a.number(1),abs(target[c,k]),sum(map(abs,terms)))
                    defect=max(defect,float(abs(value-target[c,k])/scale))
            rows.append({'center':s.center,'touches_boundary':any(g=='boundary' for g,j in s.slots),'relative_moment_defect':defect})
        return {'degree':degree,'max_relative_moment_defect':max(r['relative_moment_defect'] for r in rows),'stencils':rows}

    def stability_diagnostics(self,max_unknowns=300):
        """Small-system Float64 diagnostic, even for an MP-assembled system.

        Positive finite growth eigenvalues indicate semidiscrete instability.
        Singular mass matrices use an index-one descriptor reduction, reporting
        estimated algebraic modes separately. Rank decisions use Float64 SVD.
        Negative values alone do not certify stability for a nonnormal system.
        """
        from .descriptor import descriptor_spectrum
        n=self.mass.shape[0]
        if n>max_unknowns:return {"status":"skipped","unknowns":n,"limit":max_unknowns}
        def dense(matrix):
            if not isinstance(matrix,MPSparseMatrix):return matrix.toarray()
            out=np.zeros(matrix.shape)
            for i,row in enumerate(matrix.rows):
                for j,v in row.items():out[i,j]=float(v)
            return out
        mass,sy=dense(self.mass),dense(self.SY)
        try:spectrum,descriptor=descriptor_spectrum(sy,mass)
        except ValueError as exc:
            return {"status":"unresolved","arithmetic":"Float64 estimate","reason":str(exc)}
        finite=np.isfinite(spectrum)
        growth=float(spectrum[finite].real.max()) if finite.any() else None
        return {"status":"computed","arithmetic":"Float64 estimate",**descriptor,
            "mass_condition_inf":float(np.linalg.cond(mass,np.inf)),
            "largest_growth_real":growth,"nonfinite_eigenvalues":int(np.sum(~finite)),
            "positive_growth_detected":bool(growth is not None and growth>0)}

    def _data(self,time):
        a=self.arithmetic
        f=a.vector([v for data in self.problem.forcing for v in a.data(_at(data,time),self.cloud.interior)])
        boundary=self.cloud.points[self.cloud.boundary_indices]
        g=a.vector([v for data in self.problem.boundary_velocity for v in a.data(_at(data,time),boundary)])
        return f,g

    def solve(self,dt,steps,*,scheme='bdf2',start_time=0):
        if hasattr(self,'cpp_weights'):raise NotImplementedError('Experimental C++ backend currently supports steady LHI only')
        dt=Fraction(dt);start_time=Fraction(start_time)
        if dt<=0 or type(steps) is not int or steps<1:raise ValueError("Positive dt and integer steps required")
        if scheme not in ('backward_euler','bdf2'):raise ValueError("Unknown time scheme")
        a=self.arithmetic;cloud=self.cloud;ni=len(cloud.interior)
        initial=_InitialVelocity(a,self.problem.initial_velocity,a.number(start_time))
        _,boundary=self._data(a.number(start_time));nb=len(cloud.boundary_indices)
        iv=initial.velocity(cloud.points[cloud.boundary_indices])
        tol=a.ctx.sqrt(a.ctx.eps) if a.ctx else 1e-10
        if any(abs(iv[i,c]-boundary[i+c*nb])>tol*(1+abs(boundary[i+c*nb])) for c in (0,1) for i in range(nb)):
            raise ValueError("Initial velocity is incompatible with boundary data")
        iv=initial.velocity(cloud.interior)
        previous=a.vector([iv[i,c] for c in (0,1) for i in range(ni)]);previous2=None
        states=[];new_factors=0
        for step in range(1,steps+1):
            bdf2=scheme=='bdf2' and step>1
            alpha=Fraction(3,2)/dt if bdf2 else 1/dt
            weights=(2/dt,-Fraction(1,2)/dt) if bdf2 else (1/dt,)
            time=a.number(start_time+step*dt);forcing,boundary=self._data(time)
            history=a.number(weights[0])*previous
            if bdf2:history+=a.number(weights[1])*previous2
            boundary_term=a.vector([sum(v*boundary[j] for j,v in row.items()) for row in self.sb_rows])
            rhs=_mv(self.mass,forcing+history)-boundary_term
            if alpha not in self._factors:
                rows=[dict(row) for row in self.sy_rows]
                for i,row in enumerate(self.mass_rows):
                    for j,v in row.items():_add(rows[i],j,a.number(alpha)*v)
                matrix=_matrix(rows,a)
                factor=MPSparseLU(matrix) if a.ctx else splu(matrix)
                self._factors[alpha]=(matrix,factor);new_factors+=1
            matrix,factor=self._factors[alpha];u=factor.solve(rhs)
            udot=a.number(alpha)*u-history
            defect=_mv(matrix,u)-rhs
            residual=a.norm(defect)/(a.norm(rhs) or a.number(1))
            old=([states[-1] if states else initial] if not bdf2 else [states[-1],states[-2] if len(states)>1 else initial])
            state=LHIStokesState(self,u,forcing-udot,boundary,time,a.number(alpha),old,
                [a.number(w) for w in weights],float(residual))
            states.append(state);previous2=previous;previous=u
        return StokesTrajectory(initial,states,{'scheme':scheme,'steps':steps,'dt':str(dt),
            'new_factorizations':new_factors,'cached_factorizations':len(self._factors),
            'global_digits':a.ctx.dps if a.ctx else None,
            'local_digits':self.local_arithmetic.ctx.dps if self.local_arithmetic.ctx else None,
            'local_condition_norm':'infinity' if self.local_arithmetic.ctx else '2',
            'max_local_condition':max((s.factor.condition for s in self.stencils if s.factor.condition is not None),default=None),
            'pressure':'local gradients only','unknowns':2*ni,'mass_nnz':self.mass.nnz})


class LHIStokesState:
    def __init__(self,system,u,pde,boundary,time,alpha,history,weights,residual):
        self.system,self.nodal_velocity,self.pde_values,self.boundary_values=system,u,pde,boundary
        self.time,self.alpha,self.history,self.history_weights=time,alpha,history,weights
        self.diagnostics={**getattr(system,'backend_diagnostics',{}),'relative_residual':residual};self.coefficients=[]
        a=system.local_arithmetic
        for s in system.stencils:
            if getattr(system,'lazy_reconstruction',False):
                self.coefficients.append(None);continue
            values=[a.number({'solution':u,'boundary':boundary,'pde':pde}[group][j]) for group,j in s.slots]
            if s.polynomial_basis is not None:values += [a.number(0)]*len(s.polynomial_basis.columns)
            self.coefficients.append(s.factor.solve(a.vector(values)))

    def _fields(self,points,components,operators):
        points=_query(points, dimension=2);a=self.system.local_arithmetic
        out=a.zeros(len(points),len(components))
        if not len(points):return out
        _,owners=self.system.tree.query(points)
        for owner in np.unique(owners):
            rows=np.flatnonzero(owners==owner);s=self.system.stencils[int(owner)]
            if hasattr(self.system,'reconstruction_weights') and all(np.array_equal(points[row],self.system.cloud.points[s.center]) for row in rows):
                supported=all((c in (0,1) and op==Identity()) or (c==2 and op in (Derivative(0),Derivative(1))) for c,op in zip(components,operators))
                if supported:
                    ni=len(self.system.stencils)
                    values=[a.number({'solution':self.nodal_velocity,'boundary':self.boundary_values,'pde':self.pde_values}[group][j]) for group,j in s.slots]
                    for k,(c,op) in enumerate(zip(components,operators)):
                        if c in (0,1):value=a.number(self.nodal_velocity[int(owner)+c*ni])
                        else:
                            target=2+(0 if op==Derivative(0) else 1)
                            value=(a.ctx.fsum if a.ctx else fsum)(w*v for w,v in zip(self.system.reconstruction_weights[target*ni+int(owner)],values))
                        for row in rows:out[row,k]=value
                    continue
            if self.coefficients[int(owner)] is None:
                values=[a.number({'solution':self.nodal_velocity,'boundary':self.boundary_values,'pde':self.pde_values}[group][j]) for group,j in s.slots]
                if s.polynomial_basis is not None:values += [a.number(0)]*len(s.polynomial_basis.columns)
                self.coefficients[int(owner)]=s.factor.solve(a.vector(values))
            for k,(c,op) in enumerate(zip(components,operators)):
                matrix=s.evaluation_matrix(a,points[rows],[{c:op} for _ in rows])
                value=matrix*self.coefficients[int(owner)] if a.ctx else matrix@self.coefficients[int(owner)]
                for i,row in enumerate(rows):out[row,k]=value[i]
        return out

    def velocity(self,points):return self._fields(points,[0,1],[Identity(),Identity()])

    def _output(self,value,extended):
        a=self.system.local_arithmetic
        if extended:
            if not a.ctx:raise ValueError("extended requires local_digits")
            return value
        return np.asarray(value.tolist(),dtype=float) if a.ctx else value

    def evaluate_velocity(self,points,*,extended=False):return self._output(self.velocity(points),extended)

    def pressure_gradient(self,points,*,extended=False):
        return self._output(self._fields(points,[2,2],[Derivative(0),Derivative(1)]),extended)

    def residuals(self,points,*,extended=False):
        """BE/BDF2 momentum and patchwise divergence; excludes interface jumps."""
        points=_query(points, dimension=2);a=self.system.local_arithmetic
        lap=self._fields(points,[0,1],[Laplacian(),Laplacian()])
        pg=self._fields(points,[2,2],[Derivative(0),Derivative(1)])
        div=self._fields(points,[0,1],[Derivative(0),Derivative(1)])
        derivative=a.number(self.alpha)*self.velocity(points)
        for state,w in zip(self.history,self.history_weights):
            old=state.velocity(points)
            for i in range(len(points)):
                for c in (0,1):derivative[i,c]-=a.number(w)*a.number(old[i,c])
        out=a.zeros(len(points),3)
        for c in (0,1):
            f=a.data(_at(self.system.problem.forcing[c],self.time),points)
            for i in range(len(points)):out[i,c]=derivative[i,c]-a.number(self.system.problem.viscosity)*lap[i,c]+pg[i,c]-f[i]
        for i in range(len(points)):out[i,2]=div[i,0]+div[i,1]
        return self._output(out,extended)
