# C++ numerical backends

For the step-by-step Ubuntu or native Windows setup, start with the
[installation guide](https://ldbreton.github.io/RBFLAB/INSTALL/). This reference
describes solver internals after that setup. Generated scalar kernels compile
on demand; built-in vector LHI also has the dedicated executables below.

## Scalar and field global collocation

Global interpolation, stationary scalar PDEs, scalar evolution and Stokes share
the persistent dense worker. `interpolate(..., backend=CppGlobalBackend(...))`
fits scalar or independent multi-output data; `DivergenceFreeSpace` selects
coupled vector interpolation with a vector-only executable and no pressure
channel. Scalar evolution preserves original cloud-node ordering through
`MASS_INDEXED`, retains a separate `INITIAL_MATRIX`/initial factor, and advances
numerical histories through `TIME_STEP`. Python chooses BE/BDF2 and retains
public trajectory objects; C++ evaluates supported data, combines coefficients,
applies mass, solves and checks residuals. Callback and initial derivative checks
retain Python fallbacks. Spatial/boundary operators must remain fixed and linear.

`EVOLVE_INIT` projects initial values through the initial matrix and resets
history without discarding shifted factors. Initial and stiffness matrix edits
invalidate their respective caches. Pure-derivative Hermite trials retain their
constant-mode adjustment; unequal source/target rows never use symmetric
assembly shortcuts. Scalar shifted matrices use general row-max scaling.

### Optional MPLAPACK MPFR factorization

The MPLAPACK solver is an explicit optional dependency. Prepare a separate
prefix with MPLAPACK 3.0.1 and gmpfrxx_mkII 1.4.0, then pass that prefix as
`mplapack_dir` when selecting the MPLAPACK linear solver. Backend preparation
does not download or install it. The default Eigen solver needs no MPLAPACK.

On Ubuntu (or Ubuntu WSL), provide CMake, a C++ compiler with OpenMP, curl,
Python 3, and GMP/MPFR/MPC development headers and libraries first. The helper
downloads the official pinned release tag and builds only the optimized MPFR
archive; it never changes system packages:

```sh
bash scripts/build_mplapack.sh /absolute/path/mplapack-prefix /usr 2
```

The second argument may instead point to a locally extracted dependency
prefix with `include` and `lib` directories, including a multiarch library
subdirectory. The third argument limits build parallelism. This script builds
Linux libraries. For native Windows, use the matching helper:

```powershell
python -m pip install cmake ninja
python scripts/build_mplapack_windows.py cpp/deps/mplapack-windows --jobs 4
```

It uses MinGW-w64 GCC/OpenMP, verifies the same pinned MPLAPACK source archive,
and stages `libmplapack_mpfr_opt.a`, `libmpc.a`, `libmpfr.a`, `libgmp.a` and complete
headers. Pass that Windows prefix to `mplapack_dir`. Linux shared libraries cannot
be linked into the Windows worker. See the installation guide for integration tests.

The prepared layout contains `include/mplapack`, GMP/MPFR/MPC headers in
`include`, `lib/libmplapack_mpfr_opt.a`, bundled GMP/MPFR/MPC libraries
(static archives on Windows; shared libraries and SONAME files on Linux)
in `lib`, and `rbflab-mplapack.json`. That manifest
records the release, downloaded source hash, wrapper version, dependency
prefix and build options. All headers, archive, runtime libraries and the
manifest contribute to the executable cache identity; runtime lookup uses
the explicit prefix. Keep the prefix in place while its workers run.

The helper retains upstream MPLAPACK license files in `licenses`. Retain the
license and copyright notices for GMP, MPFR, MPC and the bundled wrapper when
redistributing a prefix, and follow their respective redistribution terms.
Dependency package selection remains a user decision. Building the archive
can take substantial time; existing matching prefixes can be reused.

`GlobalCollocation(..., backend=CppGlobalBackend(...))` compiles
`global_collocation.hpp` together with the requested radial derivatives.
`Precision()` selects double; `Precision(global_digits=80)` selects MPFR.
Both symmetric Hermite and asymmetric scalar stationary collocation support
2D/3D and polynomial augmentation. The Python frontend resolves PDE/boundary
functionals and samples user data; the worker assembles kernel and polynomial
blocks and performs equilibration, pivoted LU, solves and evaluation.

Symmetric divergence-free Stokes also supports unsteady method-of-lines systems
in 2D/3D. The worker assembles stiffness and velocity mass blocks, caches up to
four shifted LU factors for backward Euler/BDF2, and performs native mass
products. Python samples prescribed time-dependent forcing and boundary data.
The mass matrix has zero boundary/constraint rows and is exported read-only.

Unlike the batch local runtime, a global worker persists for the lifetime of
its system. Dense matrices and factors remain in contiguous Eigen storage.
Repeated RHS solves reuse the factorization; native matrix edits invalidate it.
The worker communicates over standard input/output, including bulk export for
explicit matrix inspection. Each worker uses one arithmetic precision. On
Windows it runs directly as a native MinGW-w64 executable (Float64 or MPFR).

Supported symbolic Stokes forcing and velocity boundary expressions are compiled
into the same worker. `DATA_INIT` supplies per-row expression indices at assembly;
`RHS_DATA time` evaluates the forcing/boundary vector in the selected arithmetic,
with zero polynomial constraints. Expressions enter all executable cache keys.
Native data runs in parallel for sufficiently large systems and refreshes
per-thread precision before evaluating cached constants. Arbitrary callbacks,
normal-dependent/unsupported expressions and conflicting boundary overlaps use
the existing Python route. Python still evaluates initial velocity and owns
time integration and history combinations.

Use `with method.assemble(problem, cloud) as system:` or `system.close()` for
deterministic cleanup. Solutions require the live worker for evaluation.
`evaluate()` exports Float64; `evaluate_mp()` retains the selected precision.
MPFR geometry starts from the supplied Float64 coordinates; higher arithmetic
precision cannot restore input digits.

`compute_condition=True` estimates the equilibrated matrix's 1-norm condition
using LU solves. It does not form a dense inverse. The estimate's norm differs
from the existing Python Float64 2-norm diagnostic. Assembly/evaluation can use
OpenMP; eligible dense Eigen updates also use the configured global thread limit.
The optional MPLAPACK MPFR engine uses its optimized OpenMP factorization.
Dense storage and
factorization still cost O(N²) and O(N³), respectively.

```sh
python -m examples.global_cpp --scheme asymmetric --digits 60
RBFLAB_CPP_TESTS=1 python -m pytest tests/test_global_cpp.py tests/test_global_cpp_contracts.py -q
```

In PowerShell, set `$env:RBFLAB_CPP_TESTS='1'` before the pytest command.
Stationary symmetric `SymbolicStokes` supports `CppGlobalBackend` in 2D/3D.
The field runtime applies functionals to the block kernel
`diag(Hessian(phi_velocity) - I*Laplacian(phi_velocity), phi_pressure)`.
Diagonal velocity entries evaluate only the other coordinate derivatives:
`K_cc = -sum(j != c) derivative_jj(phi_velocity)`. This removes canceling
radial evaluations in both native assembly and field prediction.
The global worker prepares repeated operator pairs once, including derivative
dispatch indices, source signs, coefficient products and scale powers. It
reuses them across point pairs and bypasses division for a unit kernel scale.
Preparation is bounded to 32 distinct operators on each side and 65,536 term
pairs; more varied operators retain direct evaluation without a large cache.
Generated global MPFR kernels cache numeric constants separately in each
thread and rebuild them when the working precision changes. No binary64
conversion is introduced for those constants.
Generated global evaluators also use a lazy cache of common expressions and
derivatives for each point pair. Assembly reuses it within an operator pair;
prediction additionally shares it across velocity components and source
functionals at identical coordinates. Target groups are capped at eight rows
and scratch storage at eight times the number of source functionals per
thread. Coefficient summation retains its original order. Caches are local to
one pair/request, so changes in parameters or precision cannot reuse stale
values. Compact kernels retain their support and stable near-origin formulas;
their derivative values can be reused without combining those formulas.
Symbolic common-expression elimination can change the last rounding digits.
Velocity and pressure can use different radial families and parameters. The
worker receives exact solenoidal polynomial modes, omits the invisible constant
pressure mode, and builds both polynomial blocks natively. A pressure reference
is applied at evaluation time, not as an extra collocation constraint.
The field solution API uses `velocity(..., extended=True)`,
`pressure_gradient(..., extended=True)` and analogous field methods for MPFR
output. Scalar `evaluate_mp` remains unchanged. Run:

```sh
python -m examples.global_cpp_fields --dimension 3 --digits 60
RBFLAB_CPP_TESTS=1 python -m pytest tests/test_global_cpp_fields.py -q
```

Closing systems releases their matrix state; the bounded idle worker pool also
serves subsequent field systems. Active systems remain independent, and scalar
and field executables have distinct cache identities. Asymmetric Stokes,
coupled scalar blocks and evolution remain unsupported by this global backend.

### Internal evaluation profiling

For global step diagnostics, `GLOBAL_PROFILE 1` enables aggregate timers and
clears counters; `GLOBAL_PROFILE 0` disables them and clears counters. `INIT`
and `RESET` also clear this state. Retrieve `GLOBAL_TIMING` after a measured
batch rather than querying after every command. It returns the number of
records, followed by each command name, count and ten aggregate seconds:
parse, matrix setup/balancing, factorization, RHS scaling, triangular solve,
coefficient scaling, residual, mass product, response formatting, total.
Records cover `SHIFT`, `SOLVE`, `MASS_MV`, and supported `RHS_DATA`. For `RHS_DATA`,
evaluation is included in total command time, with parsing/formatting separated.
Factorization includes the solver
adapter's transfers and checks. Total excludes stdout flushing, transport and
Python work. Round-trip wall time minus native total is an unattributed
remainder, including scheduling and pipe overhead. Ordinary responses and
arithmetic are unchanged; these are internal opt-in diagnostics.

For native-worker diagnostics, `EVAL_PROFILE 1` enables opt-in phase timers;
`EVAL_PROFILE 0` disables them. After a successful profiled `EVAL`,
`EVAL_TIMING` returns three wall-clock seconds: parsing/operator and point-group preparation, kernel and
polynomial evaluation, and decimal/response formatting. Standard `EVAL`
responses remain unchanged. Profiling retains results in an additional array
and separates parallel computation from parallel formatting. Transport and
stdout flushing are outside these native phase timers. `RESET` clears profiling
state. These commands are internal diagnostics, not a public fitting API.

## Local factorization

The production LHI local factorization is Eigen `PartialPivLU`. Float64 uses
`Eigen::Matrix<double,...>`; arbitrary precision uses the existing MPFR `Real`
wrapper with `Eigen::NumTraits<Real>`. No conversion through double occurs in
MPFR factorization or triangular solves. Shape policy, kernel assembly,
symmetric row-max equilibration and reconstruction formulas are unchanged.

OpenMP parallelizes stencils. Eigen internal parallelism is disabled. MPFR
precision is thread-local and set in each worker before allocating matrices.
A factorization serves all four reconstruction RHS and the optional inverse
used to estimate the infinity-norm condition number. Exact zero pivots raise
an error; LU is not a rank-revealing solver and does not regularize ill-conditioned
matrices. Eigen cannot recover digits lost to an excessively flat Float64 kernel.

## Dependencies and build

GCC, OpenMP, MPFR/GMP runtime libraries and development headers are required.
Eigen 3.4.0 was validated. On Ubuntu/WSL, extract the distribution's
development packages into the ignored project dependency directory:

```sh
bash cpp/prepare_dependencies.sh
```

From the repository root, using its Python environment:

```sh
python -m examples.build_cpp_backend
RBFLAB_CPP_TESTS=1 python -m pytest tests/test_discrete_operators.py tests/test_symbolic_kernel.py -q
```

On PowerShell set `$env:RBFLAB_CPP_TESTS='1'` before invoking pytest.
The public `CppBackend(threads=..., compute_condition=...)` API is unchanged.
`Precision(local_digits=None)` selects Float64; `Precision(local_digits=80)`
selects MPFR for local assembly/solves. Global precision remains separately
controlled by `global_dtype`.

Custom symbolic Stokes kernels compile against the same Eigen solver.
The adapter and Eigen header content hashes participate in cache identity;
previous custom binaries cannot be reused accidentally. Build manifests record
the Eigen header hash. Dependencies and generated executables remain untracked.
Off-node lazy Python reconstruction and global sparse solvers are unchanged.

## Optional Float64 SVD

`CppBackend(local_solver="svd", svd_rcond=None)` selects Eigen JacobiSVD.
`svd_rcond` is the relative singular-value cutoff; `None` uses Eigen's default
(matrix dimension times machine epsilon). Set an explicit value such as
`1e-12` when testing regularization. MPFR currently supports LU only.
SVD solves the equilibrated system with a truncated pseudoinverse; this can
change the discretization substantially. It is not an automatic fallback.

Diagnostics include each stencil's retained rank, dimension, relative cutoff,
and condition number from the untruncated computed singular values. If condition
reporting is enabled, SVD reports the 2-norm condition, whereas LU reports the
infinity-norm estimate. Tiny singular values of an unresolved Float64 matrix
cannot certify its mathematical condition number. Lazy off-node reconstruction
uses NumPy SVD with the same scaling and cutoff; kernel evaluation/rounding may
differ from C++, as in the existing Python reconstruction path.
