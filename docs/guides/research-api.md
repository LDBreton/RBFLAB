# Construct and modify a numerical method

RBFLAB separates a mathematical problem, its approximation space, the selected
method, and the implementation used to compute weights. The principal methods
are `GlobalCollocation`, `RBFFD`, and `LHI`; interpolation uses `interpolate`.

## One explicit stencil

For interpolation data $d_i=u(x_i)$ and a target functional $q$, an augmented
local interpolation system $A$ gives weights by

$$A^T\begin{bmatrix}w\\\eta\end{bmatrix}=q_{\mathrm{basis}},
\qquad q(u)\approx w^T d.$$

The polynomial multipliers $\eta$ enforce reproduction; they are not additional
nodal values. The supplied center order is the weight order.

```python
import numpy as np
import rbflab as rbf

cloud = rbf.geometry.unit_box_grid(3)
method = rbf.RBFFD(spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)})
local = method.weights(centers=cloud.points, target=[.4, .5],
                       operator=rbf.Laplacian())
values = np.sum(cloud.points**2, axis=1)
print(local.weights[:, 0] @ values)  # 4
reconstruction = local.reconstruct_local()
augmented = np.vstack((local.weights, local.multipliers))
print(np.max(np.abs(reconstruction.matrix.T @ augmented - reconstruction.rhs)))
```

For many targets, use `method.operators(source=..., targets=..., operators=...)`.
Each named operator exposes `@`, `.matrix`, `.local(i)` and
`.reconstruct_local(i)`. Multiple operators share a factorization per stencil.
Single-stencil weights use that same backend path. For vector spaces, source
DOFs are node-major; local weights have shape `(source DOFs, output components)`.

## Numerical choices and execution

Choose kernels/polynomials in spaces, neighbor/scaling/shape rules in
`StencilPolicy`, arithmetic in `Precision`, and factorization in `LocalSolver`.
Choose workers/threads and implementation in a backend. Backend changes do not
choose another shape rule. Each implemented combination remains explicit:

```python
method = rbf.RBFFD(
    spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)},
    stencil_size=12,
    stencil_policy=rbf.StencilPolicy(scaling="local"),
    local_backend=rbf.PythonBackend(compute_condition=False),
)
print(method.preflight(operation="operators"))
```

`preflight(problem, cloud)` checks supported configuration without compiling or
assembling matrices. Geometry rank and data-dependent checks still occur during
assembly. It raises an actionable exception instead of silently changing the
method. At present, SVD is available for C++ Float64 Stokes LHI; this is not a
promise of SVD support for scalar RBF-FD.

Coordinate normalization, kernel shape selection and algebraic equilibration
are distinct. `reconstruct_local().scale` describes algebraic equilibration;
`StencilPolicy.scaling` controls kernel coordinates. See
[conditioning](../theory/conditioning.md).

## Systems and reconstruction

Use `system = method.assemble(problem, cloud)`, inspect `system.matrix` and
`system.rhs`, then call `system.solve()`. `system.recipe` records configuration;
`system.dof_map` describes unknowns, and `system.reconstruction` describes the
available representation. Local center membership is recorded in each stencil.

Sparse matrices remain compatible with external SciPy solvers. Before a repeated
solve, stable stationary system wrappers check matrix contents and invalidate
cached factors after edits. This costs a linear scan of the matrix entries.
Stokes LHI uses dependent row maps and rejects matrix edits; export its matrices
for custom solves or reassemble the method. This contract concerns matrix entries with unchanged dimensions; changing the
unknown space requires reassembly. Evolution matrices must retain their
boundary/initial-data interpretation; do not modify a coupled time system by
editing only one of its dependent matrices.

Global unknowns are expansion coefficients. RBF-FD unknowns are nodal values or
boundary functionals, depending on the scheme. Scalar LHI unknowns are interior
solution values. Nearest-stencil off-node reconstruction can jump between
patches. Divergence-free LHI returns pressure gradients, not a globally
reconciled scalar pressure.

`method.prepare(problem)` returns a prepared copy. Keep the returned method;
the original method and kernels are unchanged.

## Extending the local construction

The internal Hermite machinery evaluates
$A_{ij}=\ell_i^x\lambda_j^y K(x_i,y_j)$ and polynomial constraints. LHI exposes
[functional center groups](../tutorials/lhi-centers.md) for changing its data
layout. Standard RBF-FD exposes explicit nodal stencils. Arbitrary independent
trial/data pairings and arbitrary matrix-kernel callbacks are not yet a stable
public extension protocol. Reuse exported matrices and weights for custom
algorithms; avoid depending on private factorization objects.

## Public surface

- Scalar expressions: `SymbolicScalar`.
- Coupled scalar stationary expressions: `SymbolicSystem`.
- Specialized divergence-free Stokes: `SymbolicStokes` plus spaces.
- Geometry and sampling: `rbflab.geometry`, `rbflab.meshgen`.
- Diagnostics: `rbflab.diagnostics`.
- Research experiments: `rbflab.experimental`, without stable API guarantees.

The old root `rbf_fd_weights` is replaced by `RBFFD.weights`. Specialized
Stokes/block method classes are internal assembly routes; use the principal
methods and spaces. There are no compatibility aliases in the root namespace.
