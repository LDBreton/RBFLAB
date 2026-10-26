# Matched Windows / WSL backend benchmark

Run from the repository root using Windows Python after preparing the native
Windows compiler, Ubuntu WSL GCC, and the shared Eigen/MPFR development headers.
The scripts use the installed toolchains and never install dependencies.

```powershell
python -m experiments.windows_wsl_native.benchmark
python -m experiments.windows_wsl_native.local
python -m experiments.windows_wsl_native.diagnose outputs/windows-wsl-native
python -m experiments.windows_wsl_native.summarize outputs/windows-wsl-native
```

Run benchmarks sequentially with no competing numerical jobs. Default global
cases cover scalar Poisson and divergence-free Stokes, 49/121 nodes, one/four
threads, Float64/60-digit MPFR, and 256 deterministic query points. The local
comparison covers 81 Stokes LHI stencils on 121 nodes. Each host gets an excluded
warmup and five timed samples, alternating host order. Compilation is separate.
The shared generated source gets benchmark-only C++ command timers; production
sources and backend routing are not changed.

`outputs/windows-wsl-native/report.md` contains all results; `results.json` and
`local-results.json` retain raw timing samples. Inputs, full-precision outputs,
instrumented source, native executables, compilation logs, and artifact SHA256
hashes are retained there too. These output files are ignored by Git.

The original scale-aware comparison thresholds remain fixed at 1e-8 for Float64
and 1e-48 for 60-digit MPFR. Field and coefficient agreement are reported
separately; passing field checks never hides a coefficient check failure.
The untimed diagnostic exports the larger Poisson matrices and estimates their
conditioning with Float64 SVD. Compiler and arithmetic-library versions differ
between hosts, so results describe the installed configurations rather than an
isolated operating-system effect.

The baseline run passed every field and local-weight comparison. Two Float64
Poisson coefficient checks exceeded the preset threshold slightly; see the
preserved matrix/conditioning diagnostic and full report for exact differences.
