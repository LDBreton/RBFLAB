# Precision and backends

Extended precision can be selected for local weight construction, global dense solves, or the supported sparse LHI path. The optional backends have method-specific limits; see their docstrings and [capabilities](../CAPABILITIES.md).

::: rbflab.Precision

::: rbflab.PrecisionData

For a worked explanation of `scaling="local"`, its default, and its effect on
shape parameters, see [local stencil scaling](../theory/conditioning.md#local-stencil-scaling).

::: rbflab.StencilPolicy

::: rbflab.PythonBackend

::: rbflab.CppBackend

::: rbflab.LocalExecution

::: rbflab.CppGlobalBackend

`CppGlobalBackend` selects native dense assembly, factorization and evaluation
for scalar stationary/unsteady `GlobalCollocation`, global `interpolate`, and steady/unsteady symmetric
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
data use generated C++ evaluators when all components are supported symbolic
expressions or numeric constants. The problem declarations and time-stepping
API are unchanged. Initial velocity evaluation remains in Python.

Scalar `EvolutionPDE` problems use the same `assemble(...).solve(dt=..., steps=...)`
contract with backward Euler or BDF2. C++ assembles indexed mass rows, projects
initial data to coefficients using a separately cached initial factor, combines
coefficient histories, evaluates supported symbolic data and solves each shifted
system. Python schedules steps and constructs the existing trajectory objects.
Initial derivative-boundary compatibility and arbitrary data callbacks retain
Python evaluation. The spatial and boundary operators must be linear and fixed
in time. `.matrix`/`.spatial` export stiffness A; `.mass` exports M; neither
silently becomes the last shifted system. `.initial_matrix` exposes the initial
projection separately.

`interpolate(kernel_or_space, points, data, backend=CppGlobalBackend(...))` shares
the dense engine and factor options. Scalar samples return scalar predictions.
An `(N, C)` array with an ordinary kernel or `ScalarSpace` fits independent
components, sharing one native LU. `DivergenceFreeSpace` instead builds a coupled
solenoidal interpolant in 2D/3D with `(N, dimension)` predictions. This choice
enforces analytic divergence freedom; it is not automatic for vector data.
Native interpolation supports derivatives and `evaluate_mp`, and owns its worker:
use a context manager or call `.close()`. Supplied Float64 measurements retain
their original accuracy even when the solve uses MPFR.

Native symbolic data supports arithmetic, integer/rational powers, `sin`, `cos`,
`exp`, `log`, absolute values, and exact rational/float constants including native
MPFR evaluation of pi and e. Expressions participate in executable cache
identities; runtime row mappings preserve component ordering and boundary tags.
Arbitrary callbacks, normal-dependent data, unsupported expressions, and
conflicting overlapping boundary groups retain Python evaluation for the whole
data vector. Stationary time-bound symbolic data also retain Python semantics.
Compilation and native evaluation failures are reported rather than hidden by
fallback. Diagnostics expose `data_backend` and `data_fallback_reason`.

For an optional MPLAPACK MPFR/OpenMP factor engine, build an explicit dependency
prefix using `scripts/build_mplapack_windows.py` on Windows or
`scripts/build_mplapack.sh` on Linux as described in
[native setup](../INSTALL.md), then select:

```python
backend = CppGlobalBackend(
    linear_solver="mplapack", mplapack_dir="path/to/mplapack-prefix", threads=4,
)
# Pass backend together with Precision(global_digits=60) to GlobalCollocation.
```

MPLAPACK requires extended precision; Float64 requests fail during preflight.
Eigen remains the default. Both engines retain the same assembly, balancing,
residual, field evaluation, condition estimation and shifted-factor caching.
MPLAPACK transfers scalars directly through MPFR without conversion to double.
Preparation never downloads or installs dependencies and never silently falls
back to another engine. Solution and trajectory diagnostics report
`linear_solver`. The explicit dependency prefix must remain available while
workers run because it supplies the MPFR/GMP/MPC runtime libraries.

`threads` also controls eligible Eigen global dense operations. Local stencil
solvers retain their separate outer-parallel scheduling. MPFR worker threads
share the precision selected for that worker's current assembly.
Global MPFR matrix updates reuse scratch storage for multiplication and addition,
avoiding temporary allocations while retaining separate rounding operations.
An unchanged time-step shift reuses its active matrix and factorization.
Repeated field queries cache their evaluation operators in a per-system LRU
cache bounded to 512 MiB of estimated storage; larger queries use uncached
evaluation. Interior velocity queries reuse the existing mass matrix.
Closing or reinitializing a system releases its cached operators. Threaded
factorization can change arithmetic ordering, so expect precision-level
agreement rather than bitwise-identical coefficients across thread counts.

Compiled global workers persist in `~/.cache/rbflab`, shared across examples
and Python sessions. Set `CppGlobalBackend(cache_dir="path/to/shared/cache")`
to choose another directory. SHA-256 artifact identifiers cover generated
source, native headers, compiler version, Eigen, build flags, and arithmetic
mode, selected factor engine and optional dependency content. Completed builds are published from unique temporary directories;
different builds have different identifiers. Integrity manifests are checked
before reuse. Bound kernel parameters and MPFR digit counts are runtime inputs,
so they can reuse a compatible compiled kernel family. `method.prepare(problem)`
warms this cache before a benchmark; it does not cache a particular mesh or LU.

::: rbflab.TorchBackend

::: rbflab.LocalSolver

On Windows, current CPU C++ backends compile and run native MinGW-w64 executables
with Float64 or MPFR; see [native Windows setup](../INSTALL.md#5-enable-native-c-on-windows).
The normal Python backend API is unchanged. WSL is retained only by the archived
legacy bridge and experimental CUDA adapter. Optional MPLAPACK uses an explicitly built native static prefix on Windows;
see the MPLAPACK setup instructions.

## Inspecting execution

`execution_info(obj)` returns a JSON-compatible description of covered global
methods, systems, scalar/componentwise/vector fits, global evolution states and
trajectories, and `LocalApproximation`. `evidence="configured"` describes a
requested route; `evidence="implementation"` describes the object's code path.
`recorded` includes selected diagnostics already attached to the object. No
timings, worker checks, matrix exports, callbacks, or solves are triggered.

The report includes `backend`, `arithmetic`, decimal `digits` (null for
Float64), `linear_solver`, Float64 `geometry`, and `locations` for assembly,
solve, evaluation, data and time stepping. Null means unknown or inapplicable;
`supported=false` means the helper does not cover that type, rather than that
the numerical feature is unsupported. A native data route is reported only
when recorded. In evolution reports this is the recorded forcing/boundary
route for the final saved step; initial data may have a different, unknown
route. Arbitrary callbacks and symbolic fallback routes run in Python.
Native scalar evolution uses Python orchestration and C++ step algebra;
native Stokes retains Python history orchestration with C++ mass and shifted
solves. `components` preserves independent fit reports, including mixed
metadata if present. Native componentwise interpolation shares one LU.

Local reports cover weight/map construction only and leave the later solve
unknown. Extended arithmetic does not increase precision of existing inputs:
Python floats retain their binary values, and geometry/query storage is
Float64. Use exact constants and precision-aware data where supported.

::: rbflab.execution_info
