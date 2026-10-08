# Installation

RBFLAB requires Python 3.11 or newer. The Python core needs no compiler or
mesh generator.

Install the package from PyPI:

```sh
python -m pip install rbflab
python -c "import rbflab; print(rbflab.__file__)"
```

To install from a source checkout instead, run `python -m pip install .` at
the repository root. To build an installable wheel from that checkout:

```sh
python -m pip install build
python -m build --wheel
python -m pip install dist/rbflab-0.1.0-py3-none-any.whl
```

Copy the symbolic Poisson example from the README, or run
`python examples/symbolic_poisson.py` from a source download. The default
example uses only core dependencies and prints a numerical error.

## Dependencies

| Install route | Packages or tools | Purpose |
|---|---|---|
| `rbflab` | NumPy, SciPy, SymPy, mpmath, threadpoolctl | Core array, sparse, symbolic, extended precision, and thread support |
| `rbflab[torch]` | PyTorch | Optional CPU Float64 tensor backend |
| `rbflab[mesh]` | Gmsh | Optional Gmsh cloud generation |
| `rbflab[examples]` | Matplotlib, Pillow | Optional plots and GIF animations |
| `rbflab[fast]` | gmpy2 | Optional arithmetic acceleration |
| `rbflab[test]` | pytest | Contributor tests |

Install an optional extra with, for example,
`python -m pip install "rbflab[torch]"`. PyTorch wheel availability varies
by platform; select its CPU or accelerator build using the PyTorch installer
when needed. The current Torch backend is CPU Float64 for the documented
local methods; selecting it does not imply GPU execution.

After `python -m pip install "rbflab[examples]"`, use
`from rbflab import viz` for `plot_scalar`, `plot_velocity`, and
`animate_scalar` or `animate_velocity_samples`. From a source checkout, use
`python -m pip install ".[examples]"`. The plotting packages are loaded only
when a plotting function runs.

## Native C++ backend

The C++ double and MPFR backends are currently **source-build options**, not
part of the Python wheel. Built-in Stokes executables and custom symbolic
kernel compilation need GCC, OpenMP, Eigen, and (for MPFR) MPFR/GMP. On Windows,
the verified path uses WSL Ubuntu. Follow [the native build guide](../cpp/README.md)
from a source checkout or source archive. Use `r.CppBackend(...)` after building
the executables.

For custom kernels, the first compilation creates a content-addressed artifact
in a user cache; parameter values can change without recompilation. The
`--compile` option in [the kernel example](../examples/custom_kernel.py)
demonstrates cache reuse. A compiler is needed for that option.

`Precision(local_digits=80)` requests MPFR local weights in supported C++
methods. The sparse global solve remains Float64 unless `global_dtype="mpmath"`
is chosen and supported. Always state both stages when reporting precision.

## Troubleshooting

- `ModuleNotFoundError: gmsh`: install `rbflab[mesh]` only when using Gmsh helpers.
- `ModuleNotFoundError: torch`: install `rbflab[torch]` for the tensor backend.
- `Eigen headers missing` or `Build the C++ backend first`: follow the source
  build guide; the Python wheel does not contain native build assets.
- Importing a local checkout after installing a wheel can hide packaging
  problems. Test imports from a directory outside the repository.
