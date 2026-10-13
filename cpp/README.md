# C++ numerical backends

For the step-by-step Ubuntu or Windows/WSL setup, start with the
[installation guide](https://ldbreton.github.io/RBFLAB/INSTALL/). This reference
describes solver internals after that setup. Generated scalar kernels compile
on demand; built-in vector LHI also has the dedicated executables below.

## Scalar and field global collocation

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
Windows it runs through Ubuntu WSL using the existing compiler/dependency setup.

Use `with method.assemble(problem, cloud) as system:` or `system.close()` for
deterministic cleanup. Solutions require the live worker for evaluation.
`evaluate()` exports Float64; `evaluate_mp()` retains the selected precision.
MPFR geometry starts from the supplied Float64 coordinates; higher arithmetic
precision cannot restore input digits.

`compute_condition=True` estimates the equilibrated matrix's 1-norm condition
using LU solves. It does not form a dense inverse. The estimate's norm differs
from the existing Python Float64 2-norm diagnostic. Assembly/evaluation can use
OpenMP; LU retains the current serial Eigen policy. Dense storage and
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
