"""Compatibility assembly for scalar LHI using LocalApproximation weights.

Only historical data binding and result adapters live here. There is no
independent kernel differentiation or weight solver. New applications should
use Samples and named maps directly.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from .samples import Samples
from .spaces import ScalarSpace
from .local_approximation import LocalApproximation
from .nodal import Arithmetic, NodalBasis
from .operators import bind_operators
from .stencils import neighbors, geometry_quality
from .problems import boundary_data, values
from .centers import select_groups
from .lhi_backends import PythonBackend


class _ReconstructionFactor:
    """Historical factor interface, allocated only when reconstruction needs it."""
    def __init__(self, basis, condition):
        self.basis = basis
        self.backend = basis.arithmetic.backend
        self.ctx = basis.arithmetic.ctx
        self.condition = condition
        self._factor = None

    def _get(self):
        if self._factor is None:
            self._factor = self.basis.arithmetic.factor(
                self.basis.system_matrix(self.basis.centers,self.basis.source_operators),
                compute_condition=False)
        return self._factor

    def solve(self, rhs, transpose=False):
        return self._get().solve(rhs,transpose=transpose)

    def residual(self, *args, **kwargs):
        return self._get().residual(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get(), name)


def assemble_scalar_lhi(method, problem, cloud):
    from .methods import Stencil, LHISystem
    if method.centers is None and (type(method.stencil_size) is not int or not 3 <= method.stencil_size <= len(cloud.points)):
        raise ValueError('stencil_size must be between 3 and the point count')
    a = Arithmetic(method.kernel,method.precision.local_digits)
    full = method.precision.global_dtype == 'mpmath'
    if full:
        for data in [problem.rhs]+[bc.rhs for bc in problem.boundary]:
            if callable(data) and not hasattr(data,'mp_values'):
                raise TypeError('Full-precision LHI data callbacks must use PrecisionData')
    bd = boundary_data(problem,cloud,ctx=a.ctx if full else None)
    ii = cloud.interior_indices
    if not len(ii):
        raise ValueError('LHI needs interior solution centers')
    if method.pde_stencil_size is not None and (type(method.pde_stencil_size) is not int or not 0 <= method.pde_stencil_size < len(ii)):
        raise ValueError('pde_stencil_size must be between 0 and interior_count-1')
    pde_tree = cKDTree(cloud.interior) if method.pde_stencil_size is not None else None
    tree = cKDTree(cloud.points)
    inside = set(map(int,ii))
    records, memberships, unique_points, unique_ops, lookup = [], [], [], [], {}
    for row, center in enumerate(ii):
        ids = method.stencil_policy.select(tree,cloud.points[center],method.stencil_size,method.polynomial_degree) if method.centers is None else np.array([],int)
        pc = np.array([int(j) for j in ids if int(j) in inside and j != center],int)
        if pde_tree is not None:
            candidates = ii[neighbors(pde_tree,cloud.points[center],method.pde_stencil_size+1)]
            pc = candidates[candidates != center][:method.pde_stencil_size]
        record = select_groups(method,problem,cloud,bd,row,int(center),ids,pc,a.ctx if full else None)
        sc,bc,pc,points,ops,known,groups = record
        ops = bind_operators(ops,points)
        record = (sc,bc,pc,points,ops,known,groups)
        membership = []
        for point,op in zip(points,ops):
            key = (tuple(point),op.terms)
            if key not in lookup:
                lookup[key] = len(unique_points)
                unique_points.append(point.copy()); unique_ops.append(op)
            membership.append(lookup[key])
        records.append(record); memberships.append(membership)
    samples = Samples(np.asarray(unique_points),operator=unique_ops,indices=memberships)
    source = {'data':samples}
    space = ScalarSpace(method.kernel,method.polynomial_degree)
    maps = LocalApproximation(source=source,trial=space.representers(source),
        backend=method.local_backend or PythonBackend(),precision=method.precision,
        stencil_policy=method.stencil_policy,local_solver=method.local_solver).operators(
            targets=cloud.interior,operators={'L':problem.operator})
    op = maps.L
    stencils, rows, cols, data = [], [], [], []
    rhs = None if full else values(problem.rhs,cloud.points)[ii].copy()
    interior_map = {int(node):j for j,node in enumerate(ii)}
    for row,(center,record) in enumerate(zip(ii,records)):
        sc,bc,pc,points,ops,known,groups = record
        local = op.local(row)
        basis = NodalBasis(a,points,method.polynomial_degree,ops,method.stencil_policy.scaling)
        factor = _ReconstructionFactor(basis,local.diagnostics['scaled_condition'])
        wmp = a.ctx.matrix(local.weights[:,0].tolist()) if a.ctx else None
        w = None if full else np.array([float(v) for v in (wmp if a.ctx else local.weights[:,0])])
        pd = dict(local_digits=method.precision.local_digits,condition_norm='infinity' if a.ctx else '2')
        residual = local.diagnostics['weight_residual']
        if a.ctx:
            ctx = a.ctx
            complete = ctx.matrix(op._owner.weights[row][:,0].tolist())
            rounded = ctx.matrix(complete)
            if not full:
                for j in range(len(points)):
                    rounded[j] = a.number(float(complete[j]))
            G = basis.system_matrix(points,ops)
            q = basis.evaluation(cloud.points[[center]],[problem.operator]).T
            den = ctx.norm(q,'inf') or ctx.one
            pd.update(condition_decimal=str(factor.condition),local_residual_decimal=str(residual),
                weights_rounded=not full,rounded_weight_residual=float(ctx.norm(G.T*rounded-q,'inf')/den),
                relative_weight_rounding=float(ctx.norm(complete-rounded,'inf')/(ctx.norm(complete,'inf') or ctx.one)))
        quality = geometry_quality(np.unique(points,axis=0),cloud.points[center],method.polynomial_degree if method.polynomial_degree is not None else 1)
        stencils.append(Stencil(int(center),sc,bc,pc,points,ops,factor,w,residual,wmp,pd,basis,quality,known,groups))
        if not full:
            for j,weight in zip(sc,w[:len(sc)]):
                rows.append(row); cols.append(interior_map[int(j)]); data.append(weight)
            rhs[row] -= w[len(sc):] @ np.asarray(known)
    if full:
        from .sparse_precision import assemble_sparse_system
        result = assemble_sparse_system(method.kernel,cloud,problem,stencils)
    else:
        matrix = coo_matrix((data,(rows,cols)),shape=(len(ii),len(ii))).tocsc()
        result = LHISystem(method.kernel,cloud,problem,bd,stencils,matrix,rhs)
    result.backend_diagnostics = maps.diagnostics
    return result
