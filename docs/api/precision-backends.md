# Precision and backends

Extended precision can be selected for local weight construction, global dense solves, or the supported sparse LHI path. The optional backends have method-specific limits; see their docstrings and [capabilities](../CAPABILITIES.md).

::: rbflab.Precision

::: rbflab.PrecisionData

For a worked explanation of `scaling="local"`, its default, and its effect on
shape parameters, see [local stencil scaling](../theory/conditioning.md#local-stencil-scaling).

::: rbflab.StencilPolicy

::: rbflab.PythonBackend

::: rbflab.CppBackend

::: rbflab.CppGlobalBackend

`CppGlobalBackend` selects native dense assembly, factorization and evaluation
for scalar stationary `GlobalCollocation` and steady/unsteady symmetric
`SymbolicStokes` with `DivergenceFreeSpace` velocity and scalar pressure spaces
in 2D/3D. Use `Precision()` for double or
`Precision(global_digits=80)` for MPFR. This setting is independent of the
local-weight backend used by RBF-FD and LHI. See the
[global collocation tutorial](../tutorials/global-collocation.md#native-double-and-multiprecision).
The [Stokes tutorial](../tutorials/stokes.md#native-global-fields) shows the field
API: `velocity`, `pressure`, `pressure_gradient`, `divergence`, and `residuals`
accept `extended=True` to retain multiprecision output.
Unsteady Stokes supports backward Euler and BDF2 with native stiffness/mass
assembly, shifted LU caching and mass products. Prescribed forcing and boundary
data are evaluated by Python in the selected arithmetic.

::: rbflab.TorchBackend

::: rbflab.LocalSolver
