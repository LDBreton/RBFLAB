# Verified capabilities

The method and backend combinations below are intentionally precise. A local
backend choice does not automatically change a separate global sparse solve.

Public global entry points accept `interpolate(space=..., centers=..., data=...)`
and `GlobalCollocation(space=..., collocation_scheme=...)`; evolution uses
`solve(time_scheme=...)`. Earlier positional and keyword spellings remain
compatible. `ScalarSpace` fits `(N, C)` independent scalar outputs, whereas
`DivergenceFreeSpace` fits `(N, dimension)` coupled vector outputs.
`execution_info(obj)` reports configured or implemented execution and selected
existing diagnostics without dispatching computation; unknown routes remain
null. Geometry remains Float64 even with extended arithmetic.

Polynomial degree selection is per space. `None` (the default) omits polynomial
columns, an integer selects exactly that degree even below a kernel's CPD
minimum, and `"auto"` selects the kernel minimum. A below-minimum choice may
affect solvability and accuracy; the minimum is diagnostic guidance, not a
rejection threshold. Velocity and pressure augmentations are independent.
`PressureSpace` always omits the additive pressure constant modulo its gauge;
`polynomial_degree=None` additionally requests no pressure polynomial tail.
These choices do not establish a general stability guarantee.

| Capability | Python | C++ double | C++ MPFR | PyTorch |
|---|---|---|---|---|
| Scalar stationary global collocation, 2D/3D | Yes | Yes, `CppGlobalBackend` from source checkout | Yes, `Precision(global_digits=...)` | — |
| Scalar unsteady global collocation, 2D/3D | Yes, backward Euler/BDF2 | Yes, native initial/mass/shifted solves | Yes, `Precision(global_digits=...)` | — |
| Global scalar/vector interpolation, 2D/3D | Yes | Yes, `interpolate(..., backend=...)` | Yes, including optional MPLAPACK | — |
| Explicit-cover RBF partition-of-unity interpolation, scalar or independent outputs, 2D/3D | Yes, `PatchCover` | Yes, one native worker with retained patch factors | Yes, `Precision(global_digits=...)` | — |
| Planar divergence-free RBF partition of unity, 2D | Yes, aligned streamfunction blend | Yes, retained native patch factors and evaluation | Yes, mpmath glue alignment and optional native MPFR local solves/evaluation | Velocity and first derivatives; approximates rather than exactly interpolates nodes |
| Scalar LHI convenience assembly (`LHI`) | Yes | — | — | — |
| Scalar RBF-FD operators, 2D/3D | Yes | Yes, from source checkout | Supported; installed archive test pending | Yes, CPU Float64 |
| Divergence-free steady Stokes, global 2D/3D | Yes | Yes, symmetric `CppGlobalBackend` | Yes, `Precision(global_digits=...)` | — |
| Divergence-free unsteady Stokes, global 2D/3D | Yes, backward Euler/BDF2 | Yes, native mass and shifted solves | Yes, `Precision(global_digits=...)` | — |
| Divergence-free Stokes LHI, 2D | Yes | Yes, from source archive | 50 digit local weights verified from source archive | Unaugmented stencils only; not the curated example |
| Custom symbolic kernel evaluation | Yes | Compilation from source checkout | Compilation from source checkout | Supported where the local backend accepts the kernel |
| Extended mpmath global sparse solve | Supported by selected Python methods | Separate from local C++ backend | Separate setting | — |

The curated [3D operator example](https://github.com/LDBreton/RBFLAB/blob/main/examples/operators_3d.py) ran with Python,
C++ double, and PyTorch Float64 on Windows/WSL. The
[Stokes example](https://github.com/LDBreton/RBFLAB/blob/main/examples/stokes_spaces.py) ran with Python, C++ double,
and C++ MPFR local weights.

Scalar explicit-cover partition-of-unity interpolation supports derivatives
through order two. Planar divergence-free partition of unity supports velocity
and first derivatives. Both exclude pressure spaces.
The six default Python examples ran from an installed wheel and source archive
outside the checkout.

`CppGlobalBackend` supports symmetric Hermite and asymmetric scalar collocation,
with optional polynomial augmentation. Kernel/polynomial assembly, dense LU,
residual calculation and expansion evaluation execute in C++. A persistent
worker retains the factors for repeated right-hand sides. Scalar input geometry
remains Float64; MPFR preserves extended arithmetic after input conversion.
Stationary `SymbolicStokes` also accepts this backend with symmetric collocation,
`DivergenceFreeSpace` velocity and scalar pressure spaces, including solenoidal
polynomial augmentation in 2D/3D. The Python and C++ MPFR routes also support a
degree-zero velocity polynomial space together with
`PressureSpace(..., polynomial_degree=None)`. Velocity, pressure (modulo constants), pressure
gradient, divergence and momentum residuals have native expansion evaluation.
Supported symbolic forcing and velocity boundary data also evaluate in the
native worker, with Python fallback for arbitrary callbacks or unsupported
expressions. Initial velocity evaluation and time stepping remain Python-owned.
The optional `CppGlobalBackend(linear_solver="mplapack", mplapack_dir=...)`
uses an explicitly installed MPFR/OpenMP dense LU engine with
`Precision(global_digits=...)`. Eigen remains the default; kernel assembly,
evaluation, balancing and method APIs are shared by both engines.
Scalar evolution supports linear, fixed spatial and boundary operators with
time-dependent data. Both symmetric Hermite and asymmetric scalar schemes are
supported; pure derivative Hermite source functionals retain the existing
constant-mode adjustment. Native interpolation supports independent scalar
outputs and explicit divergence-free vector spaces, with deterministic worker
cleanup. Asymmetric Stokes and coupled scalar blocks remain unsupported by this
global backend. Nonlinear or time-varying scalar operators are outside this path.
See the [native scalar example](https://github.com/LDBreton/RBFLAB/blob/main/examples/global_cpp.py)
and [global tutorial](tutorials/global-collocation.md#native-double-and-multiprecision).

RBFMeshGen generation is integrated into `rbflab.geometry` and `rbflab.meshgen`; Gmsh remains optional. The built-in
`unit_box_grid` supplies a small 2D/3D deterministic cloud for tutorials.


Curved-domain scalar examples were compared on identical clouds with Python,
C++ Float64 and PyTorch CPU Float64. See [the validation record](guides/curved-validation.md).
Core geometry includes 2D parametric domains/holes, 3D implicit regions and
parametric surfaces. Staggering remains explicit 2D triangle-based geometry.

## Explicit functional approximation (available in 0.5)

`LocalApproximation` exposes sampled source functionals and an explicit trial.
Use its named matrix blocks to construct RBF-FD and scalar LHI algorithms with
your own equation assembly and time integration. The existing method table above
refers to packaged PDE assembly; it does not restrict manual assembly from the
new functional maps. See [the construction tutorial](tutorials/local-approximation.md)
and [LHI from matrices](tutorials/lhi-matrices.md).

The canonical examples exercise value samples and scalar Hermite value/PDE/
boundary samples. Both new tutorials were run with Python Float64, C++ Float64,
and Torch CPU Float64 on the same 49-point cloud. Their stationary quadratic
LHI solve reproduces the field to approximately $2\times10^{-15}$; the short
heat example reports sampled error $2.63\times10^{-3}$ at $t=0.005$. These are
algebraic/short-run checks, not a general stability claim. Backend selection controls local construction, not the separate
SciPy solve. Unsupported spaces, trial layouts, operators and precision choices
must raise explicitly; the interface is not a claim of arbitrary mixed-vector
Hermite assembly or end-to-end GPU/autograd support.

The common engine supports scalar functionals and componentwise functionals on
`DivergenceFreeSpace` in 2D/3D. Backend parity tests cover Python, C++ Float64,
C++ MPFR and Torch CPU Float64; supported extended Python arithmetic is separate
from Torch. These kernel/block tests do not establish arbitrary mixed-field PDE
assembly, pressure constraints, or a general 3D Stokes solver.

### Generic mixed local approximation

`MixedSpace` and `FieldOperator` extend `LocalApproximation` to named direct sums
of scalar and DF fields. Field blocks specify coupled linear functionals;
sample groups and targets have independent component counts. Matching Hermite
representers, named sparse maps, multi-RHS local solves and local matrix
inspection are supported. This is approximation algebra, not automatic mixed
PDE assembly or a new Stokes solver. Existing Stokes LU assembly now uses these
maps for its weights; scalar block convenience assembly remains separate.

| Mixed local capability | Python | C++ | MPI |
|---|---|---|---|
| Scalar/DF direct-sum kernel contraction, 2D/3D | Float64/mpmath | Float64/MPFR from source toolchain | C++ jobs across real MPI ranks |
| Named maps with variable source/target arities | Yes | Yes | Deterministic job restoration |
| Target-centered local kernel scaling with physical derivative factors | Yes | Yes | Same native requests |
| Identical-request reuse | No | `LocalExecution(reuse="exact")` | Same exact-job policy |
| Extended sparse weight storage | `global_dtype="mpmath"` | Same explicit storage policy | Preserved through rank output |
| Arbitrary mixed source/trial pairing or translates | No | No | No |
| Quotient `PressureSpace` factors with constant-invariant functionals | Yes | Yes | Same native requests |
| Mixed Torch | No | No | No |

Mixed maps return `(Ntargets, target_components)`, including `(Ntargets,1)` for
scalar targets. Use `.matrix` for flat algebra. Polynomials are degree-generated
per field, with target-centered radius normalization. Explicit polynomial degrees
and CPD warnings retain their existing meaning. Field names do not imply PDE
roles. Source/trial samples must be explicit field functionals; zero and repeated
target rows are valid, while invalid source rows/rank are rejected.
For mixed local scaling, the radius is the maximum selected-source distance
from the target; the single-field approximation measures selected source and
trial centers from its first selected source center.
See [scaling conventions](guides/research-api.md#numerical-execution).
Mixed quality selection grows eligible sized groups using the joint functional
polynomial rank/ratio and per-group separation checks. Explicit memberships
remain fixed; exhausted size limits raise an error.

`LocalExecution(scheduler="mpi", ranks=...)` requires a native MPI launcher or
the Windows/WSL bridge and `CppBackend`. It only distributes local weight jobs.
The common single-field paths reject this execution option explicitly; the
Stokes LHI LU adapter accepts it on `LHI(..., execution=...)`. Exact
reuse matches full serialized requests; translated or approximate reuse is not
implemented. See [the workflow and prerequisites](guides/research-api.md#mixed-field-hermite-maps).

Direct single-field `PressureSpace` now consistently omits constant polynomial
columns and accepts only constant-invariant observations, representers and
targets. Absolute pressure values use `ScalarSpace` with an externally selected
gauge. The previous direct-local behavior that inserted a constant at degree
`None` is corrected; specialized Stokes output rules are unchanged.

Standard `RBFFD.operators`/`.weights` and scalar `LHI` weight construction now
reuse this engine. The RBF-FD compatibility adapter still exports SciPy matrices
for Torch; direct `LocalApproximation` keeps tensors. Scalar LHI preserves its
historical data/solution interpretation. Stokes LHI Python/C++ LU weights now
use the same engine, retaining exact stencil/observation memberships, quotient
pressure, compact matrices and pressure-gradient reconstruction. The Stokes
SVD/experimental routes and coupled scalar block assembly remain separate.

`LHI(..., execution=LocalExecution(scheduler="mpi", ranks=...))` schedules
standard C++ LU Stokes weights. It does not parallelize the global sparse solve.
Python process workers also retain the common local weight path. Numerical
shape policies remain explicit and are bound before batching generic requests;
no new global pressure gauge or pressure-value unknowns are introduced.
Fixed-parameter Stokes LU supports both physical and target-centered local
kernel coordinates in Python, C++ and native MPI execution. Native SVD and
historical custom executables retain their physical-coordinate restriction;
the `hardy`, `radius_scaled` and `separation_scaled` shape rules also require
physical coordinates.

`Samples.target="exclude"` removes coincident source points from a target's
neighborhood. It is useful for PDE samples that would otherwise make a local PDE
row tautological. Named source blocks retain their full global group column
layout after local selection.

Independent PDE centers in an evolution algorithm need an explicitly chosen
map for their time-derivative data. The [matrix tutorial](tutorials/lhi-matrices.md)
uses coincident solution/PDE groups with identical global ordering, so that its
mass matrix is exactly `I - Sf` for the stated formulation.

## Configuration preflight

Use `method.preflight(problem, cloud)` or
`method.preflight(operation="operators")` before an expensive run. The same
validation is used by assembly. This checks configuration support, not geometric
rank or expected accuracy.

Independent `CenterGroup` layouts support scalar stationary Python LHI in 2D/3D,
including supported extended arithmetic. Default-layout transient LHI remains
supported; independent transient groups require additional unknown-data maps and
are explicitly rejected. `SymbolicStokes` describes constant-viscosity momentum
with prescribed velocity boundaries. Coupled scalar stationary problems use
`SymbolicSystem`; arbitrary mixed vector equations are not advertised.

`RBFFD.weights` and `RBFFD.operators` currently support the standard nodal
construction. PDE assembly retains `standard`, `symmetric`, and
`boundary_hermite` variants. Solver availability is checked separately:
`LocalSolver("svd")` currently requires C++ Float64 Stokes LHI.
