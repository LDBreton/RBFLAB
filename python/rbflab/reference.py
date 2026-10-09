"""Small dense arbitrary-precision reference solves; not scalable sparse solvers."""
import numpy as np
from .operators import Identity
from .precision import MPBackend, MPFactor, mp_number, mp_values
from .problems import boundary_data
from .methods import _query


class MPGlobalSystem:
    def __init__(self, kernel, problem, cloud, digits):
        self.backend=MPBackend(kernel,digits)
        ctx=self.backend.ctx
        bd=boundary_data(problem,cloud,ctx=ctx)
        ii,bi=cloud.interior_indices,cloud.boundary_indices
        self.centers=cloud.points[np.r_[ii,bi]]
        self.operators=[problem.operator]*len(ii)+[bd[int(j)][0] for j in bi]
        self.rhs=ctx.matrix(mp_values(problem.rhs,cloud.points[ii],ctx)+
                            [bd[int(j)][1] for j in bi])
        self.matrix=self.backend.matrix(self.centers,self.operators,self.centers,self.operators)
        self.factor=MPFactor(self.backend,self.matrix)

    def solve(self):
        coefficients=self.factor.solve(self.rhs)
        residual=self.factor.residual(coefficients,self.rhs,transpose=False)
        diagnostics={"global_digits":self.backend.ctx.dps,
                     "condition_norm":"infinity",
                     "scaled_gram_condition":float(self.factor.condition_mp),
                     "condition_decimal":self.backend.ctx.nstr(self.factor.condition_mp,12),
                     "relative_residual":float(residual),
                     "residual_decimal":self.backend.ctx.nstr(residual,12),
                     "unknowns":len(coefficients)}
        return MPGlobalSolution(self,coefficients,diagnostics)


class MPGlobalSolution:
    def __init__(self,system,coefficients,diagnostics):
        self.system,self.coefficients,self.diagnostics=system,coefficients,diagnostics

    def evaluate_mp(self,points,operator=None):
        points=_query(points, self.system.centers.shape[1])
        if len(points)==0:
            return self.system.backend.ctx.matrix(0,1)
        matrix=self.system.backend.matrix(points,[operator or Identity(points.shape[1])]*len(points),
                                           self.system.centers,self.system.operators)
        return matrix*self.coefficients

    def evaluate(self,points,operator=None):
        return np.array([float(v) for v in self.evaluate_mp(points,operator)])


def solve_lhi_reference(system, weight_source="unrounded", rounded_system=False,
                        extended_data=False):
    """Reassemble in MP from stored weights. No recomputation of Float64 matrices.

    weight_source='rounded' uses Float64 weights but MP RHS assembly.
    rounded_system=True uses the actual Float64 matrix/RHS, isolating solve error.
    extended_data=True requests PrecisionData callbacks; default promotes exactly
    the same Float64 forcing/boundary data as the production solve.
    """
    if weight_source not in ("unrounded","rounded"):
        raise ValueError("weight_source must be unrounded or rounded")
    if rounded_system and (weight_source!="rounded" or extended_data):
        raise ValueError("rounded_system requires rounded weights and Float64 data")
    if any(s.weights_mp is None for s in system.stencils):
        raise ValueError("Reference solve requires extended local weights")
    if len(system.stencils)>200:
        raise ValueError("Dense LHI reference is limited to 200 interior nodes")
    backend=system.stencils[0].factor.backend
    ctx=backend.ctx
    ii=system.cloud.interior_indices
    mapping={int(j):i for i,j in enumerate(ii)}
    if extended_data and getattr(system,'recipe',{}).get('independent_centers',False):
        raise NotImplementedError('For independent groups, assemble with full sparse extended precision instead of resampling reference data')
    if extended_data:
        bd=boundary_data(system.problem,system.cloud,ctx=ctx)
        forcing=mp_values(system.problem.rhs,system.cloud.points,ctx)
    else:
        from .problems import values
        bd={j:(op,mp_number(ctx,v)) for j,(op,v) in system.boundary_data.items()}
        forcing=[mp_number(ctx,v) for v in values(system.problem.rhs,system.cloud.points)]
    a=ctx.matrix(len(ii))
    rhs=ctx.matrix([forcing[j] for j in ii])
    for row,s in enumerate(system.stencils):
        weights=s.weights_mp if weight_source=="unrounded" else [
            mp_number(ctx,v) for v in s.weights]
        for j,w in zip(s.solution_indices,weights[:len(s.solution_indices)]):
            a[row,mapping[int(j)]]+=w
        known=[mp_number(ctx,v) for v in s.known_data] if s.known_data is not None and not extended_data else [bd[int(j)][1] for j in s.boundary_indices]+[forcing[j] for j in s.pde_indices]
        rhs[row]-=ctx.fsum(weights[len(s.solution_indices)+j]*v for j,v in enumerate(known))
    if rounded_system:
        a=ctx.matrix(system.matrix.toarray().tolist())
        rhs=ctx.matrix(system.rhs.tolist())
    factor=MPFactor(backend,a)
    u=factor.solve(rhs)
    residual=factor.residual(u,rhs,transpose=False)
    diagnostics={"global_digits":ctx.dps,"weight_source":weight_source,
                 "rounded_system":rounded_system,"extended_data":extended_data,
                 "relative_residual":float(residual),"residual_decimal":ctx.nstr(residual,12),
                 "scaled_system_condition":float(factor.condition_mp),"condition_norm":"infinity"}
    return MPLHIReferenceSolution(system,u,bd,forcing,diagnostics)


class MPLHIReferenceSolution:
    def __init__(self,system,u,bd,forcing,diagnostics):
        from scipy.spatial import cKDTree
        self.system,self.interior_values,self.diagnostics=system,u,diagnostics
        self.ctx=system.stencils[0].factor.ctx
        self._tree=cKDTree(system.cloud.interior)
        nodal=dict(zip(system.cloud.interior_indices,u))
        self._coefficients=[]
        for s in system.stencils:
            data=[nodal[j] for j in s.solution_indices]+(list(s.known_data) if s.known_data is not None and not diagnostics.get('extended_data',False) else [bd[int(j)][1] for j in s.boundary_indices]+[forcing[j] for j in s.pde_indices])
            self._coefficients.append(s.factor.solve(s.basis.padded(data) if s.basis else self.ctx.matrix(data)))

    def evaluate_mp(self,points,operator=None):
        points=_query(points, self.system.cloud.dimension)
        out=self.ctx.matrix(len(points),1)
        if len(points)==0:
            return out
        _,owners=self._tree.query(points)
        for owner in np.unique(owners):
            rows=np.flatnonzero(owners==owner)
            s=self.system.stencils[int(owner)]
            m=(s.basis.evaluation(points[rows],[operator or Identity(points.shape[1])]*len(rows)) if s.basis else
               s.factor.backend.matrix(points[rows],[operator or Identity(points.shape[1])]*len(rows),s.points,s.operators))
            v=m*self._coefficients[int(owner)]
            for row,value in zip(rows,v):
                out[int(row)]=value
        return out

    def evaluate(self,points,operator=None):
        return np.array([float(v) for v in self.evaluate_mp(points,operator)])
