# Backends for curved-domain examples

The annulus, ellipse, flower heat and 3D ball examples expose the same choice:

```sh
python -m examples.annulus --backend python
python -m examples.annulus --backend cpp
python -m examples.annulus --backend torch
```

Their equations, nodes, PHS5 kernel and polynomial degree stay fixed. Backend
selection changes local numerical assembly and solves; the global sparse solver
remains SciPy Float64. Local condition-number calculation is disabled for these
examples. Geometric quality can be inspected separately with `meshgen.quality`.

## Python and PyTorch

From a checkout, install `python -m pip install -e ".[examples]"` for Python, or
`python -m pip install -e ".[examples,torch]"` for the CPU Float64 PyTorch backend.
Four local threads are requested for C++ and PyTorch. More threads do not always
help small systems. The documented Torch path does not use CUDA or MPFR.

## C++ source setup

Use the step-by-step [Ubuntu](../INSTALL.md#4-enable-c-on-ubuntu-linux) or
[Windows/WSL](../INSTALL.md#5-enable-c-from-windows-through-wsl) installation route.
The guide distinguishes generated scalar kernels from dedicated vector LHI
executables and explains dependencies, cache reuse and matching native sources.

## Read timings honestly

Each example reports `prepare_s`, `assembly_s`, and `solve_s` separately. Preparation
includes cache lookup and, on first use, compilation or tensor import. The solve
stage can include reconstruction setup. Error evaluation and plotting occur
after these measurements. Warm caches and thread contention affect timing.

At these small sizes C++ or PyTorch may not improve end-to-end runtime; Python
orchestration, transfer and reconstruction still cost time. Use a larger cloud
and repeat in isolation before making speed claims. Changing a backend should
first pass the matched-cloud numerical checks:

```sh
python -m examples.curved_validation --backends python cpp torch
```
