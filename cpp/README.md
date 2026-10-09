# C++ local numerical backend

For the step-by-step Ubuntu or Windows/WSL setup, start with the
[installation guide](https://ldbreton.github.io/RBFLAB/INSTALL/). This reference
describes solver internals after that setup. Generated scalar kernels compile
on demand; built-in vector LHI also has the dedicated executables below.

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
