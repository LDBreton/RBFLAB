"""Dense global Hermite collocation and genuine local Hermite interpolation."""
from .api_contracts import MethodContract, assembled, prepared, _UNSET, _keyword_alias
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .stencils import neighbors as stencil_neighbors, StencilPolicy, geometry_quality
from .operators import Identity
from .kernels import resolve_polynomial_degree
from .configuration import LocalSolver
from .precision import Precision, MPBackend, MPFactor, local_weights
from .problems import values, boundary_data
from .assembly import functional_matrix, Factor, relative_residual


@dataclass(init=False)
class GlobalCollocation(MethodContract):
    """Dense scalar or field collocation using radial kernels.

    Args:
        kernel (object): Scalar radial kernel or bound symbolic kernel.
        precision (Precision): Arithmetic policy for assembly and solve.
        scheme (str): `"symmetric"` Hermite or `"asymmetric"` nodal collocation.
        polynomial_degree (int, None, or "auto"): `None` disables polynomial
            augmentation; `"auto"` selects the kernel's CPD minimum. Explicit
            degrees below that minimum are retained with a warning.
        space (ScalarSpace): Optional ScalarSpace shorthand for scalar problems.
        collocation_scheme (str): Alias for scheme; specify only one spelling.
        spaces (dict or None): Optional field-to-space mapping for coupled Stokes problems.
        backend (object): Optional CppGlobalBackend for native stationary/unsteady scalar or
            steady/unsteady symmetric divergence-free Stokes solves in 2D/3D.

    Use `assemble(problem, cloud)` to obtain a system with a matrix and RHS;
    then call `system.solve()`, or `system.solve(dt=..., steps=...)` for evolution."""
    kernel: object = None
    precision: Precision = field(default_factory=Precision)
    scheme: str = "symmetric"
    polynomial_degree: object = None
    spaces: dict | None = None
    backend: object = None

    def __init__(self, kernel=_UNSET, precision=_UNSET, scheme=_UNSET,
                 polynomial_degree=None, spaces=_UNSET, backend=None, *,
                 space=_UNSET, collocation_scheme=_UNSET):
        if space is not _UNSET:
            from .spaces import ScalarSpace
            if kernel is not _UNSET or spaces is not _UNSET:
                raise ValueError("Specify space, kernel or spaces, not more than one")
            if type(space) is not ScalarSpace:
                raise TypeError("GlobalCollocation space must be a ScalarSpace; use spaces for fields")
            if polynomial_degree is not None:
                raise ValueError("Specify polynomial_degree in ScalarSpace only")
            kernel, polynomial_degree = space.kernel, space.degree()
        self.kernel = None if kernel is _UNSET else kernel
        self.precision = Precision() if precision is _UNSET else precision
        self.scheme = _keyword_alias(scheme, collocation_scheme, name="scheme",
                                     alias_name="collocation_scheme", default="symmetric")
        self.polynomial_degree = polynomial_degree
        self.spaces = None if spaces is _UNSET else spaces
        self.backend = backend

    @prepared
    def prepare(self,problem,*,dimension=2,cache_dir=None):
        """Prepare and cache required derivatives of a symbolic kernel.

        Returns a method with compiled kernel support. With CppGlobalBackend,
        this warms the native worker executable and retains the bound kernel.
        """
        if self.backend is not None:
            from .global_backend import prepare_cpp_global
            return prepare_cpp_global(self,problem,dimension,cache_dir)
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
        if self.backend is not None:
            from .global_backend import assemble_cpp_global
            return assemble_cpp_global(self, problem, cloud)
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
        local_solver: Backend-independent local factorization configuration.
        execution: Optional LocalExecution scheduling of LU Stokes weights.
            MPI requires CppBackend; global sparse solves remain separate."""
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

    # This schedules local Stokes weights, not the global sparse solve.
    execution: object = None

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
        from ._compat_lhi import assemble_scalar_lhi
        return assemble_scalar_lhi(self, problem, cloud)


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
            "max_scaled_local_condition": max(
                (s.factor.condition for s in self.stencils if s.factor.condition is not None),
                default=None),
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


def interpolate(kernel=_UNSET, centers=_UNSET, data=_UNSET, precision=None, polynomial_degree=None, *, backend=None, space=_UNSET, patches=None):
    """Fit scalar, componentwise or divergence-free global interpolants.

    Args:
        kernel (object): Built-in radial kernel, bound symbolic kernel, or
            ScalarSpace specifying its polynomial augmentation. DivergenceFreeSpace
            selects a coupled solenoidal vector interpolant in 2D/3D.
        space (object): Alias for kernel; specify only one spelling.
        centers (numpy.ndarray): Finite coordinate array of shape (N, 2) or (N, 3).
        data (numpy.ndarray or callable): Scalar samples of shape (N,), independent
            outputs of shape (N, C), or a scalar callback evaluated at centers.
            DivergenceFreeSpace takes (N, dimension) samples or a tuple of
            dimension scalar data callbacks/expressions.
            Row i corresponds to centers[i]. Extended callbacks require
            PrecisionData.
        precision (Precision or None): Optional Precision(global_digits=...) for dense arithmetic.
        polynomial_degree (int, None, or "auto"): Highest total polynomial degree.
            `None` disables augmentation; `"auto"` selects the kernel's CPD
            minimum. An explicit degree below that minimum is retained with a warning.
        backend (CppGlobalBackend or None): Optional native global assembly,
            factorization and evaluation. Native interpolants own a live worker;
            close them explicitly or use a context manager.
        patches (PatchCover or None): Explicit compact-support balls for
            partition-of-unity interpolation. Local factors persist for refit.
            Scalar and independent outputs support derivatives through order two.
            Planar DivergenceFreeSpace uses aligned local streamfunctions and
            supports velocity and first derivatives.

    Returns:
        result (object): Fitted expansion with coefficients, diagnostics, and
            evaluate(points, operator=None). Scalar output is (M,); independent
            outputs are (M, C); solenoidal outputs are (M, dimension). An operator
            differentiates each component. evaluate_mp retains extended arithmetic.

    Scalar and global vector fits interpolate nodes exactly. The planar
    divergence-free partition-of-unity fit is an approximant and generally
    differs at nodes because the blended potential has weight derivatives.
    Coefficients are
    expansion coefficients, not the input nodal values. Assembly is dense.
    """
    kernel = _keyword_alias(kernel, space, name="kernel", alias_name="space", default=_UNSET)
    missing = [name for name, value in (("kernel or space", kernel), ("centers", centers), ("data", data)) if value is _UNSET]
    if missing:
        raise TypeError("interpolate missing required argument(s): " + ", ".join(missing))
    from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace
    if patches is not None and isinstance(kernel, PressureSpace):
        raise NotImplementedError('PUM does not support PressureSpace gauge constraints')
    if isinstance(kernel, PressureSpace):
        raise ValueError('PressureSpace represents pressure modulo constants; '
                         'use ScalarSpace to interpolate absolute pressure values')
    if isinstance(kernel, DivergenceFreeSpace):
        if polynomial_degree is not None:
            raise ValueError('Specify polynomial_degree in DivergenceFreeSpace only')
        if patches is not None:
            from .dfpu import interpolate_dfpu
            return interpolate_dfpu(kernel, centers, data, precision or Precision(), patches, backend)
        from .global_vector_interpolation_backend import interpolate_vector
        return interpolate_vector(kernel, centers, data, precision or Precision(), backend)
    if isinstance(kernel, ScalarSpace):
        if polynomial_degree is not None:
            raise ValueError('Specify polynomial_degree in ScalarSpace only')
        polynomial_degree=kernel.degree()
        kernel=kernel.kernel
    else:
        polynomial_degree=resolve_polynomial_degree(kernel,polynomial_degree)
    if patches is not None:
        from .patches import interpolate_patches
        return interpolate_patches(kernel,centers,data,precision or Precision(),polynomial_degree,patches,backend)
    if not callable(data):
        array=np.asarray(data.tolist() if hasattr(data,'rows') else data,dtype=object)
        if array.ndim==2 and array.shape[1]>1:
            from .global_vector_interpolation_backend import interpolate_componentwise
            return interpolate_componentwise(kernel,centers,data,precision or Precision(),polynomial_degree,backend)
    if backend is not None:
        from .global_interpolation_backend import interpolate_cpp
        return interpolate_cpp(kernel, centers, data, precision or Precision(), polynomial_degree, backend)
    from .nodal import interpolation
    return interpolation(kernel, centers, data, precision or Precision(), polynomial_degree)
