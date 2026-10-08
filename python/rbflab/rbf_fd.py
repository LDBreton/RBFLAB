"""Polynomial-augmented RBF-FD with shared nodal stencils and sparse solvers."""
from dataclasses import dataclass,field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .precision import Precision
from .operators import Identity
from .assembly import relative_residual
from .nodal import Arithmetic,LocalNodalStencil,collocation_data
from .stencils import StencilPolicy, geometry_quality
from .sparse_precision import MPSparseMatrix,MPSparseLU


@dataclass
class RBFFD:
    """Polynomial-augmented local finite differences on a point cloud.

    The standard scheme uses value-based local ansatz functions. Alternative
    schemes and supported backends can be selected through constructor fields.

    Args:
        kernel: Radial basis function used for stencil weights.
        stencil_size: Number of points in each local stencil.
        polynomial_degree: Highest total polynomial degree.
        precision: Local-weight and sparse-solve arithmetic policy.
        stencil_policy: Selection and scaling of local points.
        scheme: Local ansatz scheme; `"standard"` is the default.
        local_backend: Optional local-weight implementation.
        spaces: Optional source/target approximation-space mapping.

    Use `operators(...)` for reusable discrete matrices or `assemble(...)`
    for a scalar PDE system."""
    kernel: object = None
    stencil_size: int = 15
    polynomial_degree: int = 2
    precision: Precision = field(default_factory=Precision)
    stencil_policy: StencilPolicy = field(default_factory=StencilPolicy)

    scheme: str = 'standard'
    local_backend: object = None
    spaces: object = None

    def operators(self, *, source, targets=None, operators, space=None) -> "OperatorSet":
        """Return reusable named sparse differentiation operators.

        Args:
            source (PointCloud or ndarray): Finite (N, 2) or (N, 3) nodes.
            targets (PointCloud or ndarray, optional): Targets; defaults to source.
            operators (Mapping[str, Operator]): Named differential operators.
            space (str, optional): Field when multiple spaces are declared.

        Returns:
            OperatorSet: Entries expose matrix, local(i), and
                reconstruct_local(i). Only standard nodal RBF-FD is supported.
        """
        from .discrete_operators import build_operators
        return build_operators(self, source=source, targets=targets, operators=operators, space=space)

    def prepare(self,problem,*,dimension=2,cache_dir=None):
        """Prepare supported symbolic kernel derivatives for a problem.

        May compile and cache native derivative code when the selected
        backend and kernel require it. Numeric kernel parameters remain
        runtime inputs; compilation is an optional source workflow.
        """
        if self.spaces is not None:
            from .discrete_operators import scalar_method
            return scalar_method(self).prepare(problem,dimension=dimension,cache_dir=cache_dir)
        if self.local_backend is not None or self.scheme!='standard':
            from .scalar_backends import prepare_scalar_method
            return prepare_scalar_method(self,problem,dimension,cache_dir)
        from .kernel_compiler import prepare_method
        return prepare_method(self,problem,dimension,cache_dir)

    def assemble(self,problem,cloud):
        """Assemble scalar PDE/boundary rows or dispatch an evolution system.

        Stationary scalar rows apply PDE or boundary functionals to local
        value-interpolation bases and form a sparse nodal FDSystem.
        Evolution, space-based, and optional-backend configurations dispatch
        to their corresponding assemblers.
        """
        if self.spaces is not None:
            from .discrete_operators import scalar_method
            return scalar_method(self).assemble(problem,cloud)
        from .evolution import EvolutionPDE, assemble_evolution
        if isinstance(problem, EvolutionPDE):
            return assemble_evolution(self, problem, cloud)
        if self.scheme!='standard' or self.local_backend is not None:
            from .scalar_fd import assemble_scalar_fd
            return assemble_scalar_fd(self,problem,cloud)
        if not isinstance(self.precision,Precision):
            raise TypeError("precision must be Precision")
        if self.precision.global_digits is not None:
            raise ValueError("RBF-FD uses local_digits and global_dtype")
        if not isinstance(self.stencil_policy,StencilPolicy):
            raise TypeError("stencil_policy must be StencilPolicy")
        local=Arithmetic(self.kernel,self.precision.local_digits)
        full=self.precision.global_dtype=="mpmath"
        global_arithmetic=local if full else Arithmetic(self.kernel)
        a=global_arithmetic
        operators,targets=collocation_data(problem,cloud,a)
        tree=cKDTree(cloud.points)
        rows=[];rhs=[];stencils=[];ids=[]
        for i,point in enumerate(cloud.points):
            selected=self.stencil_policy.select(tree,point,self.stencil_size,self.polynomial_degree)
            stencil=LocalNodalStencil(local,cloud.points[selected],self.polynomial_degree,self.stencil_policy.scaling)
            stencil.geometry_diagnostics=geometry_quality(cloud.points[selected],point,
                self.polynomial_degree if self.polynomial_degree is not None else 1)
            op,target=operators[i],targets[i]
            w=stencil.weights(point,op)
            rows.append({int(j):a.number(v) for j,v in zip(selected,w)})
            rhs.append(target)
            stencils.append(stencil);ids.append(selected)
        if full:
            matrix=MPSparseMatrix(a.ctx,rows)
        else:
            entries=[(i,j,v) for i,row in enumerate(rows) for j,v in row.items()]
            matrix=coo_matrix(([v for i,j,v in entries],
                ([i for i,j,v in entries],[j for i,j,v in entries])),
                shape=(len(rows),len(rows))).tocsc()
        return FDSystem(cloud,a,matrix,a.vector(rhs),stencils,ids,tree)


class FDSystem:
    """Assembled scalar RBF-FD linear system.

    matrix maps nodal unknowns to PDE and boundary row values; rhs holds
    sampled forcing and boundary data. solve() returns FDSolution.
    """
    def __init__(self,cloud,arithmetic,matrix,rhs,stencils,indices,tree):
        self.cloud,self.arithmetic,self.matrix,self.rhs=cloud,arithmetic,matrix,rhs
        self.stencils,self.indices,self.tree=stencils,indices,tree
        self.factor=None

    def solve(self):
        """Solve and report the sparse-system residual.

        The factorization is reused on later calls. Float64 uses SciPy
        sparse LU; configured extended arithmetic uses MPSparseLU.
        """
        a=self.arithmetic
        if self.factor is None:
            self.factor=MPSparseLU(self.matrix) if a.ctx else splu(self.matrix)
        nodal=self.factor.solve(self.rhs)
        residual=self.factor.residual(nodal,self.rhs) if a.ctx else relative_residual(self.matrix,nodal,self.rhs)
        return FDSolution(self,nodal,{"relative_residual":float(residual),
            "unknowns":len(nodal),"nnz":self.matrix.nnz,
            "global_digits":a.ctx.dps if a.ctx else None,
            "max_scaled_local_condition":max((s.factor.condition for s in self.stencils if s.factor.condition is not None),default=None),
            "local_condition_norm":getattr(self,"backend_diagnostics",{}).get("local_condition_norm","infinity" if self.stencils[0].basis.arithmetic.ctx else "2"),
            "local_digits":self.stencils[0].basis.arithmetic.ctx.dps if self.stencils[0].basis.arithmetic.ctx else None})


class FDSolution:
    """Nodal result with nearest-stencil off-node reconstruction.

    dof_values are global solved degrees of freedom. evaluate(points,
    operator) samples values or derivatives in Float64; evaluate_mp
    retains extended local results when available.
    """
    def __init__(self,system,nodal,diagnostics):
        self.system,self.dof_values,self.diagnostics=system,nodal,diagnostics
        self._coefficients=[]
        if hasattr(system,'executor'):
            data=[[system.stencils[i].basis.arithmetic.number(nodal[int(j)]) for j in ids]+[0]*len(system.stencils[i].basis.powers) for i,ids in enumerate(system.indices)]
            results=system.executor.solve(data)
            self._coefficients=[s.basis.arithmetic.vector(row['weights']) for s,row in zip(system.stencils,results)]
            return
        for stencil,ids in zip(system.stencils,system.indices):
            self._coefficients.append(stencil.factor.solve(
                stencil.basis.padded([nodal[int(i)] for i in ids])))

    @property
    def nodal_values(self):
        if getattr(self.system,'scheme','standard')=='boundary_hermite':
            values=self._evaluate(self.system.cloud.points)
            return self.system.arithmetic.vector(values)
        return self.dof_values

    def _evaluate(self,points,operator=None):
        from .methods import _query
        points=_query(points, self.system.cloud.dimension)
        if not len(points):
            return []
        _,owners=self.system.tree.query(points)
        out=[]
        for point,owner in zip(points,owners):
            stencil=self.system.stencils[int(owner)]
            a=stencil.basis.arithmetic
            row=stencil.basis.evaluation(point.reshape(1,-1),[operator or Identity(points.shape[1])])
            coefficients=self._coefficients[int(owner)]
            v=row*coefficients if a.ctx else row@coefficients
            out.append(v[0])
        return out

    def evaluate(self,points,operator=None):
        """Evaluate at finite (N, dimension) query points.

        Uses the nearest local stencil. The patchwise field is not
        guaranteed continuous across stencil-owner changes.
        """
        return np.array([float(v) for v in self._evaluate(points,operator)])

    def evaluate_mp(self,points,operator=None):
        a=self.system.stencils[0].basis.arithmetic
        if not a.ctx:
            raise ValueError("evaluate_mp requires extended local precision")
        return a.ctx.matrix(self._evaluate(points,operator))
