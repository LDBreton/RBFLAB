"""Scalar local ansatz variants using shared functional assembly/backends."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from .nodal import Arithmetic,NodalBasis,collocation_data
from .operators import Identity,bind_operators
from .problems import boundary_data
from .stencils import geometry_quality
from .scalar_backends import ScalarExecutor


def assemble_scalar_fd(method,problem,cloud):
    from .rbf_fd import FDSystem
    from .evolution import _sparse
    from .precision import Precision
    from .stencils import StencilPolicy
    if not isinstance(method.precision,Precision):raise TypeError('precision must be Precision')
    if method.precision.global_digits is not None:raise ValueError('RBF-FD uses local_digits and global_dtype')
    if not isinstance(method.stencil_policy,StencilPolicy):raise TypeError('stencil_policy must be StencilPolicy')
    if type(method.stencil_size) is not int or not 3<=method.stencil_size<=len(cloud.points):raise ValueError('stencil_size must be between 3 and point count')
    if method.scheme not in ('standard','symmetric','boundary_hermite'):raise ValueError('scheme must be standard, symmetric or boundary_hermite')
    local=Arithmetic(method.kernel,method.precision.local_digits)
    a=local if method.precision.global_dtype=='mpmath' else Arithmetic(method.kernel)
    operators,targets=collocation_data(problem,cloud,a);operators=bind_operators(operators,cloud.points)
    bd=boundary_data(problem,cloud,ctx=a.ctx)
    identity=Identity(cloud.dimension);ids=[];jobs=[];stencils=[];tree=cKDTree(cloud.points)
    data_ops=[bd[i][0] if method.scheme=='boundary_hermite' and i in bd else identity for i in range(len(cloud.points))]
    for i,point in enumerate(cloud.points):
        selected=method.stencil_policy.select(tree,point,method.stencil_size,method.polynomial_degree)
        left=[data_ops[int(j)] for j in selected]
        right=[operators[int(j)] for j in selected] if method.scheme=='symmetric' else left
        # Nodal moments for the transformed trial basis retain harmonic polynomial
        # modes even when L annihilates them on an entirely interior stencil.
        moments=left
        basis=NodalBasis(local,cloud.points[selected],method.polynomial_degree,right,method.stencil_policy.scaling)
        P=basis.polynomials(basis.centers,moments)
        check=np.array(P.tolist() if local.ctx else P,dtype=float)
        if check.shape[1]:
            rows=np.max(abs(check),axis=1);check/=np.where(rows>0,rows,1)[:,None]
            columns=np.max(abs(check),axis=0);check/=np.where(columns>0,columns,1)[None,:]
            if np.linalg.matrix_rank(check)<len(basis.powers):raise np.linalg.LinAlgError('Polynomial functionals are not unisolvent; enlarge/change stencil')
        job=dict(points=basis.centers,left=left,right=right,moments=moments,degree=method.polynomial_degree,scaling=method.stencil_policy.scaling,target=point,op=operators[i])
        jobs.append(job);ids.append(selected)
        stencils.append(SimpleNamespace(basis=basis,factor=None,geometry_diagnostics=geometry_quality(basis.centers,point,method.polynomial_degree or 1)))
    from .configuration import configured_backend
    executor=ScalarExecutor(method.kernel,method.precision.local_digits,configured_backend(method),jobs)
    results=executor.solve();rows=[]
    for i,(result,stencil,selected) in enumerate(zip(results,stencils,ids)):
        stencil.factor=SimpleNamespace(condition=result['condition']);stencil.weight_residual=result['residual']
        w=[a.number(v) for v in result['weights'][:len(selected)]]
        # Boundary-Hermite DOFs are B(u), not u: impose their supplied data exactly.
        rows.append({i:a.number(1)} if method.scheme=='boundary_hermite' and i in bd else {int(j):v for j,v in zip(selected,w)})
    system=FDSystem(cloud,a,_sparse(rows,a),a.vector(targets),stencils,ids,tree)
    system.executor=executor;system.scheme=method.scheme;system.dof_operators=data_ops
    system.backend_diagnostics=dict(backend=type(executor.backend).__name__,scheme=method.scheme,local_digits=method.precision.local_digits,weight_seconds=executor.last_seconds,max_local_weight_residual=max(r['residual'] for r in results),local_condition_norm='infinity' if type(executor.backend).__name__=='CppBackend' or local.ctx else '2')
    return system
