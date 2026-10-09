"""Dense global Hermite collocation and genuine local Hermite interpolation."""
from .api_contracts import MethodContract, assembled, prepared
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .stencils import neighbors as stencil_neighbors, StencilPolicy, geometry_quality
from .operators import Identity
from .kernels import validate_polynomial_degree
from .configuration import LocalSolver
from .precision import Precision, MPBackend, MPFactor, local_weights
from .problems import values, boundary_data
from .assembly import functional_matrix, Factor, relative_residual


@dataclass
class GlobalCollocation(MethodContract):
    """Dense scalar collocation using a radial kernel.

    Args:
        kernel: Scalar radial kernel or bound symbolic kernel.
        precision: Arithmetic policy for assembly and solve.
        scheme: `"symmetric"` Hermite or `"asymmetric"` nodal collocation.
        polynomial_degree: Optional polynomial augmentation degree.
        spaces: Optional field-to-space mapping for coupled Stokes problems.

    Use `assemble(problem, cloud)` to obtain a system with a matrix and RHS;
    then call `system.solve()`."""
    kernel: object = None
    precision: Precision = field(default_factory=Precision)
    scheme: str = "symmetric"
    polynomial_degree: int | None = None
    spaces: dict | None = None

    @prepared
    def prepare(self,problem,*,dimension=2,cache_dir=None):
        """Prepare and cache required derivatives of a symbolic kernel.

        Returns a method using the prepared kernel. Native compilation
        needs an optional source build and is not part of core pip install.
        """
        from .kernel_compiler import prepare_method
        return prepare_method(self,problem,dimension,cache_dir)

    @assembled
    def assemble(self, problem, cloud):
        """Build a dense global collocation system.

        Symmetric assembly applies functionals to both kernel arguments;
        asymmetric assembly uses ordinary kernel translates as its trial
        basis. Both solve for expansion coefficients, not nodal values.
        Coupled spaces and evolution problems dispatch to their systems.
        """
        if self.spaces is not None:
            from .space_stokes import assemble_spaces
            return assemble_spaces(self,problem,cloud)
        from .spaces import SpaceStokesProblem
        if isinstance(problem,SpaceStokesProblem):raise ValueError("Space Stokes requires a spaces mapping on the method")
        from .evolution import EvolutionPDE, assemble_evolution
        if isinstance(problem, EvolutionPDE):
            return assemble_evolution(self, problem, cloud)
        if self.scheme not in ("symmetric", "asymmetric"):
            raise ValueError("scheme must be symmetric or asymmetric")
        validate_polynomial_degree(self.kernel,self.polynomial_degree)
        if self.scheme == "asymmetric" or self.polynomial_degree is not None:
            from .nodal import assemble_global
            return assemble_global(self.kernel, problem, cloud, self.precision,self.polynomial_degree,self.scheme)
        if not isinstance(self.precision, Precision):
            raise TypeError("precision must be a Precision object")
        if self.precision.global_dtype=="mpmath":
            raise ValueError("GlobalCollocation uses global_digits; global_dtype is for LHI")
        if self.precision.local_digits is not None:
            raise ValueError("GlobalCollocation uses global_digits, not local_digits")
        if self.precision.global_digits is not None:
            from .reference import MPGlobalSystem
            return MPGlobalSystem(self.kernel, problem, cloud, self.precision.global_digits)
        bd = boundary_data(problem, cloud)
        ii, bi = cloud.interior_indices, cloud.boundary_indices
        centers = cloud.points[np.r_[ii, bi]]
        ops = [problem.operator] * len(ii) + [bd[int(i)][0] for i in bi]
        rhs = np.r_[values(problem.rhs, cloud.points[ii]), [bd[int(i)][1] for i in bi]]
        matrix = functional_matrix(self.kernel, centers, ops, centers, ops)
        return GlobalSystem(self.kernel, centers, ops, matrix, rhs)


class GlobalSystem:
    """Dense collocation matrix and sampled PDE/boundary right side.

    Rows and columns correspond to the stored functional centers.
    solve() returns expansion coefficients and algebraic diagnostics.
    """
    def __init__(self, kernel, centers, ops, matrix, rhs):
        self.kernel, self.centers, self.operators = kernel, centers, ops
        self.matrix, self.rhs = matrix, rhs
        self.factor = Factor(matrix)

    def solve(self):
        coefficients = self.factor.solve(self.rhs)
        diagnostics = {
            "scaled_gram_condition": self.factor.condition,
            "relative_residual": relative_residual(self.matrix, coefficients, self.rhs),
            "unknowns": len(coefficients),
        }
        return GlobalSolution(self, coefficients, diagnostics)


class GlobalSolution:
    """Global kernel expansion determined by solved coefficients.

    evaluate(points, operator) applies a value or differential functional
    to the expansion at finite query points of matching dimension.
    """
    def __init__(self, system, coefficients, diagnostics):
        self.system, self.coefficients, self.diagnostics = system, coefficients, diagnostics

    def evaluate(self, points, operator=None):
        points = _query(points, self.system.centers.shape[1])
        matrix = functional_matrix(self.system.kernel, points,
                                   [operator or Identity(points.shape[1])] * len(points),
                                   self.system.centers, self.system.operators)
        return matrix @ self.coefficients


@dataclass
class Stencil:
    center: int
    solution_indices: np.ndarray
    boundary_indices: np.ndarray
    pde_indices: np.ndarray
    points: np.ndarray
    operators: list
    factor: Factor
    weights: np.ndarray
    weight_residual: float
    weights_mp: object = None
    precision_diagnostics: dict = field(default_factory=dict)
    basis: object = None
    geometry_diagnostics: dict = field(default_factory=dict)
    known_data: object = None
    groups: dict = field(default_factory=dict)


@dataclass
class LHI(MethodContract):
    """Sparse local Hermite interpolation for scalar PDEs or Stokes spaces.

    Local systems mix solution, boundary, and PDE functionals. By default a
    stencil excludes its target from PDE centers; explicit groups select this rule. The sparse solve returns nodal
    values; off-node evaluation uses a nearby local stencil.

    Args:
        kernel: Radial kernel used in each local Hermite system.
        stencil_size: Number of nearby solution/boundary candidates.
        precision: Local and sparse-solve arithmetic policy.
        centers: Optional named CenterGroup mapping for independent scalar
            stationary data layouts. None preserves combined-neighbor selection.
        pde_stencil_size: Optional number of additional PDE centers in the default layout.
        polynomial_degree: Optional polynomial reproduction degree.
        stencil_policy: Neighbor selection and local scaling policy.
        spaces: Optional field-to-space mapping for divergence-free Stokes.
        local_backend: Optional supported local-weight backend.
        local_solver: Backend-independent local factorization configuration."""
    kernel: object = None
    stencil_size: int = 15
    precision: Precision = field(default_factory=Precision)
    centers: dict | None = None
    pde_stencil_size: int | None = None
    polynomial_degree: int | None = None
    stencil_policy: StencilPolicy = field(default_factory=StencilPolicy)
    spaces: dict | None = None
    min_boundary_centers: int = 0
    local_backend: object = None
    local_solver: LocalSolver = field(default_factory=LocalSolver)

    @prepared
    def prepare(self,problem,*,dimension=2,cache_dir=None):
        """Prepare and cache required derivatives of a symbolic kernel.

        Returns a method using the prepared kernel. Native compilation
        needs an optional source build and is not part of core pip install.
        """
        from .kernel_compiler import prepare_method
        return prepare_method(self,problem,dimension,cache_dir)

    @assembled
    def assemble(self, problem, cloud):
        """Construct local Hermite weights and the global sparse PDE system."""
        if self.spaces is not None:
            from .space_stokes import assemble_spaces
            return assemble_spaces(self,problem,cloud)
        if self.min_boundary_centers:
            raise ValueError("These stencil options require divergence-free Stokes spaces")
        from .spaces import SpaceStokesProblem
        if isinstance(problem,SpaceStokesProblem):raise ValueError("Space Stokes requires a spaces mapping on the method")
        from .evolution import EvolutionPDE, assemble_evolution
        if isinstance(problem, EvolutionPDE):
            return assemble_evolution(self, problem, cloud)
        if self.centers is None and (type(self.stencil_size) is not int or not 3 <= self.stencil_size <= len(cloud.points)):
            raise ValueError("stencil_size must be between 3 and the point count")
        if not isinstance(self.precision, Precision):
            raise TypeError("precision must be a Precision object")
        if self.precision.global_digits is not None:
            raise ValueError('LHI uses local_digits; use system.solve_reference for a dense global reference')
        validate_polynomial_degree(self.kernel,self.polynomial_degree)
        if not isinstance(self.stencil_policy,StencilPolicy):
            raise TypeError("stencil_policy must be StencilPolicy")
        arithmetic = None
        if self.polynomial_degree is not None or self.stencil_policy.scaling=="local":
            from .nodal import Arithmetic
            arithmetic = Arithmetic(self.kernel,self.precision.local_digits)
        backend = arithmetic.backend if arithmetic else (
            MPBackend(self.kernel, self.precision.local_digits) if self.precision.local_digits else None)
        full_mp=self.precision.global_dtype=="mpmath"
        if full_mp:
            for data_item in [problem.rhs]+[bc.rhs for bc in problem.boundary]:
                if callable(data_item) and not hasattr(data_item,"mp_values"):
                    raise TypeError("Full-precision LHI data callbacks must use PrecisionData")
        bd = boundary_data(problem, cloud,ctx=backend.ctx if full_mp else None)
        ii = cloud.interior_indices
        if len(ii) == 0:
            raise ValueError("LHI needs interior solution centers")
        if self.pde_stencil_size is not None and (
            type(self.pde_stencil_size) is not int or not 0 <= self.pde_stencil_size < len(ii)
        ):
            raise ValueError("pde_stencil_size must be None or between 0 and interior_count-1")
        pde_tree = cKDTree(cloud.interior) if self.pde_stencil_size is not None else None
        interior_set = set(ii.tolist())
        interior_map = {int(node): j for j, node in enumerate(ii)}
        forcing = None if full_mp else values(problem.rhs, cloud.points)
        tree = cKDTree(cloud.points)
        rows, cols, data, stencils = [], [], [], []
        rhs = None if full_mp else forcing[ii].copy()
        for row, center in enumerate(ii):
            neighbors = self.stencil_policy.select(tree, cloud.points[center], self.stencil_size,self.polynomial_degree) if self.centers is None else np.array([],int)
            sc = np.array([int(j) for j in neighbors if int(j) in interior_set], dtype=int)
            fc = np.array([int(j) for j in neighbors if int(j) in bd], dtype=int)
            pc = sc[sc != center]  # Default preserves the original selection.
            if pde_tree is not None:
                candidates = ii[stencil_neighbors(pde_tree, cloud.points[center], self.pde_stencil_size+1)]
                pc = candidates[candidates != center][:self.pde_stencil_size]
            from .centers import select_groups
            sc,fc,pc,points,ops,known,groups = select_groups(
                self,problem,cloud,bd,row,int(center),neighbors,pc,
                backend.ctx if full_mp else None)
            basis = None
            if arithmetic is not None:
                from .hermite import augmented_local_weights
                basis,factor,w,wmp,residual,pd = augmented_local_weights(
                    arithmetic,points,ops,cloud.points[[center]],problem.operator,
                    self.polynomial_degree,round_weights=not full_mp,
                    kernel_scaling=self.stencil_policy.scaling)
            elif backend is None:
                gram = functional_matrix(self.kernel, points, ops, points, ops)
                factor = Factor(gram)
                q = functional_matrix(self.kernel, cloud.points[[center]], [problem.operator],
                                      points, ops)[0]
                w = factor.solve(q, transpose=True)
                residual = relative_residual(gram.T, w, q)
                wmp, pd = None, {"local_digits": None, "condition_norm": "2"}
            else:
                factor, w, wmp, residual, pd = local_weights(
                    self.kernel, points, ops, cloud.points[[center]], problem.operator,
                    self.precision, backend, round_weights=not full_mp)
            quality=geometry_quality(np.unique(points,axis=0),cloud.points[center],
                                     self.polynomial_degree if self.polynomial_degree is not None else 1)
            if full_mp:
                stencils.append(Stencil(int(center), sc, fc, pc, points, ops, factor, w,
                                        residual, wmp, pd, basis, quality, known, groups))
                continue
            for j, weight in zip(sc, w[:len(sc)]):
                rows.append(row)
                cols.append(interior_map[int(j)])
                data.append(weight)
            # Boundary and PDE-center data are prescribed. Move their
            # contribution right; only solution-center weights become
            # columns of the global interior matrix.
            rhs[row] -= w[len(sc):] @ np.asarray(known)
            stencils.append(Stencil(int(center), sc, fc, pc, points, ops, factor, w,
                                    residual, wmp, pd, basis, quality, known, groups))
        if full_mp:
            from .sparse_precision import assemble_sparse_system
            return assemble_sparse_system(self.kernel,cloud,problem,stencils)
        matrix = coo_matrix((data, (rows, cols)), shape=(len(ii), len(ii))).tocsc()
        return LHISystem(self.kernel, cloud, problem, bd, stencils, matrix, rhs)


class LHISystem:
    """Sparse system for interior solution-center values.

    Each stencil holds solution, boundary, and PDE center indices and a
    local Hermite factorization. Known boundary and forcing contributions
    have already been moved into rhs.
    """
    def __init__(self, kernel, cloud, problem, bd, stencils, matrix, rhs):
        self.kernel, self.cloud, self.problem = kernel, cloud, problem
        self.boundary_data, self.stencils = bd, stencils
        self.matrix, self.rhs = matrix, rhs
        self.factor = None

    def solve_reference(self, **kwargs):
        from .reference import solve_lhi_reference
        return solve_lhi_reference(self, **kwargs)

    def solve(self):
        """Factor and solve the interior sparse system.

        Returns an LHISolution with local/rounded-weight diagnostics and
        an algebraic sparse-system residual. Reuses the sparse factor.
        """
        if self.factor is None:
            self.factor = splu(self.matrix)
        u = self.factor.solve(self.rhs)
        diagnostics = {
            "relative_residual": relative_residual(self.matrix, u, self.rhs),
            "max_scaled_local_condition": max(s.factor.condition for s in self.stencils),
            "max_local_weight_residual": max(s.weight_residual for s in self.stencils),
            "unknowns": len(u), "nnz": self.matrix.nnz,
            "max_local_constraints": max(len(s.points) for s in self.stencils),
            "local_digits": self.stencils[0].precision_diagnostics["local_digits"],
            "condition_norm": self.stencils[0].precision_diagnostics["condition_norm"],
            "max_rounded_weight_residual": max(
                s.precision_diagnostics.get("rounded_weight_residual", s.weight_residual)
                for s in self.stencils),
            "max_relative_weight_rounding": max(
                s.precision_diagnostics.get("relative_weight_rounding", 0.0)
                for s in self.stencils),
        }
        return LHISolution(self, u, diagnostics)


class LHISolution:
    """Interior LHI nodal values and local Hermite reconstructions.

    evaluate(points, operator) selects the nearest interior stencil for
    each off-node point; the patchwise field can jump across boundaries
    between stencil owners.
    """
    def __init__(self, system, interior_values, diagnostics):
        self.system, self.interior_values, self.diagnostics = system, interior_values, diagnostics
        self._tree = cKDTree(system.cloud.interior)
        nodal = dict(zip(system.cloud.interior_indices, interior_values))
        self._coefficients = []
        for stencil in system.stencils:
            data = np.r_[[nodal[j] for j in stencil.solution_indices],
                         stencil.known_data if stencil.known_data is not None else
                         [system.boundary_data[int(j)][1] for j in stencil.boundary_indices]+list(values(system.problem.rhs, system.cloud.points[stencil.pde_indices]))]
            self._coefficients.append(stencil.factor.solve(stencil.basis.padded(data) if stencil.basis else data))

    def evaluate(self, points, operator=None):
        """Nearest-stencil reconstruction; not guaranteed continuous across patches."""
        points = _query(points, self.system.cloud.dimension)
        if len(points) == 0:
            return np.empty(0)
        _, owners = self._tree.query(points)
        out = np.empty(len(points))
        for owner in np.unique(owners):
            rows = np.flatnonzero(owners == owner)
            s = self.system.stencils[int(owner)]
            if s.basis is not None:
                matrix=s.basis.evaluation(points[rows],[operator or Identity(points.shape[1])]*len(rows))
                v=matrix*self._coefficients[int(owner)] if s.basis.arithmetic.ctx else matrix@self._coefficients[int(owner)]
                out[rows]=[float(value) for value in v]
            elif isinstance(s.factor, MPFactor):
                matrix = s.factor.backend.matrix(
                    points[rows], [operator or Identity(points.shape[1])] * len(rows), s.points, s.operators)
                out[rows] = [float(v) for v in matrix * self._coefficients[int(owner)]]
            else:
                matrix = functional_matrix(self.system.kernel, points[rows],
                                           [operator or Identity(points.shape[1])] * len(rows),
                                           s.points, s.operators)
                out[rows] = matrix @ self._coefficients[int(owner)]
        return out


def _query(points, dimension=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or not np.isfinite(points).all():
        raise ValueError("Query points must be finite with shape (N, 2) or (N, 3)")
    if dimension is not None and points.shape[1] != dimension:
        raise ValueError("Query dimension must match the solution dimension")
    return points


def interpolate(kernel, centers, data, precision=None, polynomial_degree=None):
    """Fit a scalar RBF interpolant to supplied values at distinct centers.

    Args:
        kernel (object): Built-in radial kernel or bound symbolic kernel.
        centers (numpy.ndarray): Finite coordinate array of shape (N, 2) or (N, 3).
        data (numpy.ndarray or callable): Scalar samples of shape (N,), or a callback evaluated at centers.
            Row i corresponds to centers[i]. Extended callbacks require
            PrecisionData.
        precision (Precision or None): Optional Precision(global_digits=...) for dense arithmetic.
        polynomial_degree (int or None): Highest total polynomial degree. Must meet
            the kernel's minimum augmentation requirement.

    Returns:
        result (NodalSolution): Fitted expansion with coefficients, diagnostics, and
            evaluate(points, operator=None). Evaluation returns one value per
            query point; an operator differentiates the fitted expansion.

    This is exact interpolation, not noise-aware smoothing. Coefficients are
    expansion coefficients, not the input nodal values. Assembly is dense.
    """
    from .nodal import interpolation
    return interpolation(kernel, centers, data, precision or Precision(), polynomial_degree)
