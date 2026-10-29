# Installation

Start with the Python package. Add plotting, tensor assembly, or a native compiler
when you need them. **The PyPI wheel contains the Python library; C++ is a source-build option.**

## 1. Create an environment

Python **3.11 or newer** is required. Use a dedicated environment so the package
and its optional dependencies do not change another project's installation.

On Linux/macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activation is optional; call its interpreter directly:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "rbflab[examples]"
```

In the commands below, `python` means the interpreter of that environment. In
PowerShell, substitute `.\.venv\Scripts\python.exe` if it is not activated.

## 2. Install and check the Python package

```sh
python -m pip install --upgrade pip
python -m pip install --upgrade "rbflab[examples]>=0.5"
python -c "from importlib.metadata import version; import rbflab; print(version('rbflab')); print(rbflab.__file__)"
```

Use `rbflab` instead of `"rbflab[examples]"` if you do not need figures or animations.
The [first problem](getting-started.md) runs without a compiler or Gmsh.

| Install | Adds | Used for |
|---|---|---|
| `rbflab` | NumPy, SciPy, SymPy, mpmath, threadpoolctl, Shapely | Core geometry, kernels, operators and solves |
| `rbflab[examples]` | Matplotlib, Pillow | Plots and GIFs |
| `rbflab[torch]` | PyTorch | Supported local methods on CPU in Float64 |
| `rbflab[mesh]` | Gmsh | Optional connected meshes and Gmsh helpers |
| `rbflab[fast]` | gmpy2 | Optional extended-arithmetic acceleration |
| `rbflab[test]` / `rbflab[docs]` | pytest / documentation tools | Contributor workflows |

Extras combine: `python -m pip install "rbflab[examples,torch]"`.
There is no `rbflab[cpp]` extra that installs a compiler or prebuilt native backend.
PyTorch is an optional local-weight backend; it does not turn the global SciPy
solver into a GPU solver. The documented path is CPU Float64.

### Check that the new local API is installed

The current source manual targets **RBFLAB 0.6.0**. Mixed-field local maps,
MPI scheduling and the shared Stokes LU weight path require this source version.
Install the checkout with the same Python interpreter that runs your code:

```sh
python -m pip install -e .
python -c "from rbflab import Samples, LocalApproximation, MixedSpace, FieldOperator, LocalExecution; from importlib.metadata import version; print(version('rbflab'))"
```

The 0.6.0 changelog is marked unreleased; installing the released package alone
does not imply that these source features are present. Release 0.5 also includes
breaking research-API cleanup from 0.4. Consult the
[changelog](https://github.com/LDBreton/RBFLAB/blob/v0.5.0/CHANGELOG.md) when adapting
older scripts. The supported symbolic PDE examples still use the explicit
`RBFFD`/`LHI` adapters; these are distinct from the matrix-building interface.

## 3. Get the runnable examples

Library snippets can run anywhere after installation. The online manual's
tutorial scripts follow the current repository; use its main checkout below
for the latest lessons (the library API requires 0.5 or newer).
For an archived release, check out its matching tag instead.
Commands beginning with
`python -m examples...` need the repository's example files:

```sh
git clone https://github.com/LDBreton/RBFLAB.git
cd RBFLAB
python -m pip install -e ".[examples]"
python -m examples.tutorials.first_problem
```

Run module commands **from this repository root**. On Windows, if your earlier
environment was created in a different folder and is not activated, create one
inside this checkout and use its interpreter:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[examples]"
.\.venv\Scripts\python.exe -m examples.tutorials.first_problem
```

Keep using that full interpreter path for subsequent `python` commands, or use
an activated environment. The editable install keeps Python pointed at the source
being used. A GitHub source ZIP also works: extract
it, enter its top-level folder and run the same install command. Some examples
share helpers, so keep the `examples` directory together.

## 4. Enable C++ on Ubuntu Linux

The verified native route uses **GCC with OpenMP, Eigen, MPFR and GMP**, on
Ubuntu x86-64. Native Windows with MinGW-w64 is covered below; MSVC and macOS
builds are not covered by this guide. Python-only installation remains independent of these tools.

From the checkout root, with your Python environment selected:

```sh
sudo apt-get update
sudo apt-get install -y g++ libmpfr6 libgmp10
bash cpp/prepare_dependencies.sh
python -m pip install -e ".[examples,test]"
```

The helper downloads the distribution's Eigen/MPFR/GMP development packages and
extracts headers into `cpp/deps`. It does not install them system-wide. The build
expects this local layout, so installing system Eigen alone is not sufficient.
The compiler and MPFR/GMP runtime libraries must still be available to Ubuntu.

### First native scalar calculation

```sh
python -m examples.annulus --backend cpp
```

This compiles the scalar kernel/operator artifact on first use and caches it.
It then prints preparation, assembly and solve times, plus solution errors.
Repeat the command to reuse the cache. Preparation can dominate a tiny cold run.

### Build the dedicated vector LHI executables

For the built-in vector LHI path, also run:

```sh
python -m examples.build_cpp_backend
RBFLAB_CPP_TESTS=1 python -m pytest tests/test_discrete_operators.py tests/test_symbolic_kernel.py -q
```

The build should report **both** `cpp/lhi_double` and `cpp/lhi_mpfr`, with their
manifests. The scalar example above uses generated artifacts; these dedicated
executables serve the built-in vector LHI path. They are different build routes.

## 5. Enable native C++ on Windows

Windows Python compiles and launches native MinGW-w64 executables directly.
From an ordinary PowerShell in the checkout, run:

```powershell
python -m pip install -e ".[examples,test]"
python scripts/setup_windows_cpp.py
python -m examples.annulus --backend cpp
python -m examples.build_cpp_backend
$env:RBFLAB_CPP_TESTS = "1"
python -m pytest tests/test_global_cpp.py tests/test_discrete_operators.py tests/test_symbolic_kernel.py -q
Remove-Item Env:RBFLAB_CPP_TESTS
```

The setup script downloads official MSYS2 UCRT64 GCC, OpenMP, Eigen, MPFR and GMP
packages and their dependencies into ignored `cpp/deps/windows`. It checks each
package against the SHA256 recorded in the repository database and saves package
versions/checksums in `packages.json`. No administrator rights or WSL installation
are needed. The first setup needs network access and Windows `tar` with zstd support.

Generated kernels, local functional maps and persistent global workers support
both Float64 and MPFR. Built-in vector LHI produces `cpp/lhi_double.exe` and
`cpp/lhi_mpfr.exe` with separate Windows manifests. Executables are statically
linked, so runtime DLL directories need not be added to PATH. Compiler helpers
supply their own child-process PATH without changing your environment.

An existing MinGW-w64 GCC can instead be selected with `RBFLAB_CXX` (the full
`g++.exe` path); `RBFLAB_NATIVE_PREFIX` optionally selects its matching MPFR/GMP
include/lib prefix. Eigen can live in that compiler prefix's `include/eigen3`
or the existing `cpp/deps/usr/include/eigen3`. Linux headers/libraries for MPFR/GMP
must not be reused for Windows. Compiler target and host platform separate the
compiled cache from old WSL artifacts. The `distribution="Ubuntu"` field remains
for API compatibility; current CPU backends launch directly on the host.

The archived `LegacyCppLHIBackend` and experimental CUDA adapter retain their
existing Linux/WSL execution paths.

## 6. Select the backend in Python

For local matrices, the backend belongs to `LocalApproximation`:

```python
import rbflab as rbf

cloud = rbf.geometry.unit_box_grid(6)
space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
source = {"u": rbf.Samples(cloud.points, size=20)}
local = rbf.LocalApproximation(
    source=source, trial=space.representers(source),
    backend=rbf.CppBackend(threads=4, compute_condition=False),
    stencil_policy=rbf.StencilPolicy(scaling="local"),
)
ops = local.operators(targets=cloud.points, operators={"lap": rbf.Laplacian(2)})
D = ops.lap["u"].to_scipy()
```

Substitute `PythonBackend(...)` or `TorchBackend(...)` to keep the same
mathematical construction. C++ prepares the required compiled kernel derivatives
and local solves; the first invocation may include compilation.
`scaling="local"` uses the stencil radius as the kernel's distance unit, with
derivatives returned in physical units. See [local scaling](theory/conditioning.md#local-stencil-scaling).

The `.matrix` property retains backend storage. `to_scipy()` explicitly exports
Float64 CSR matrices and detaches Torch tensors. Your global SciPy solve remains
separate from local weight construction.

For extended-precision local construction, pass
`precision=rbf.Precision(local_digits=80)` to `LocalApproximation` with
`CppBackend` (MPFR) or `PythonBackend`. To retain those digits in sparse maps
also set `global_dtype="mpmath"`; exporting them with `to_scipy()` rounds them.
Geometry is still Float64. The new functional engine currently supports LU
local solves and fixed kernel parameters; Torch supports Float64.
See [precision](theory/conditioning.md) and [capabilities](CAPABILITIES.md).

The optional symbolic equation adapter instead accepts
`RBFFD(..., local_backend=rbf.CppBackend(...))`. It builds and solves the PDE
system for you.

For scalar stationary dense collocation, select the separate global backend:

```python
method = rbf.GlobalCollocation(
    rbf.PHS(5), scheme="symmetric", polynomial_degree=2,
    backend=rbf.CppGlobalBackend(threads=4, compute_condition=False),
    precision=rbf.Precision(global_digits=80),
)
```

`Precision()` selects double; `global_digits` selects MPFR for native assembly,
factorization and evaluation. Both symmetric and asymmetric scalar collocation
support 2D/3D and polynomial augmentation. The existing compiler dependencies
above are sufficient; no separate global executable build command is needed.
See [global collocation](tutorials/global-collocation.md#native-double-and-multiprecision)
for the solution and precision contracts.

## Optional MPLAPACK global MPFR solver

Eigen is the default factor engine. MPLAPACK 3.0.1 provides an optional optimized
MPFR/OpenMP factor engine with the same global API.

On native Windows, first prepare the toolchain in step 5, then run:

```powershell
python -m pip install cmake ninja
python scripts/build_mplapack_windows.py cpp/deps/mplapack-windows --jobs 4
$env:RBFLAB_CPP_TESTS = "1"
$env:RBFLAB_MPLAPACK_DIR = (Resolve-Path cpp/deps/mplapack-windows).Path
python -m pytest tests/test_global_cpp_mplapack.py -q
```

The helper verifies the pinned official source archive, builds the full optimized
MPFR target with native MinGW-w64 GCC and OpenMP, and stages its archive, wrapper
headers, matching static GMP/MPFR/MPC libraries, available licenses and build
manifest into the specified prefix. No WSL or administrator rights are needed.
The first build compiles over a thousand source files. Repeating the command
reuses the verified source and incremental build. `--deps` optionally selects a
matching native dependency prefix; `--archive` reuses a downloaded release archive.

On Linux, use the separate helper with an existing GMP/MPFR/MPC development prefix:

```sh
bash scripts/build_mplapack.sh /absolute/path/to/mplapack-prefix /usr 4
```

Select the prepared prefix explicitly:

```python
backend = rbf.CppGlobalBackend(
    linear_solver="mplapack",
    mplapack_dir="cpp/deps/mplapack-windows",  # use your Linux prefix on Linux
    threads=4,
)
# Pass backend with Precision(global_digits=60) to GlobalCollocation or interpolate.
```

MPLAPACK requires MPFR precision; Float64 keeps the Eigen solver. Preparation
checks and hashes the prefix without installing dependencies or silently falling
back. Windows uses static native libraries; Linux shared libraries cannot be
reused. Keep the prefix in place while workers run. Tests cover precision changes,
worker reuse, normal/transposed solves, condition estimation and Stokes stepping.
See [precision and backends](api/precision-backends.md).

## Installed wheel with a separate native checkout

The simplest native setup is the editable install above. If you must keep a wheel
installation, obtain matching native sources and set `RBFLAB_SOURCE_ROOT` to that
checkout before compilation. Use the release tag matching your installed version.

```sh
export RBFLAB_SOURCE_ROOT=/absolute/path/to/RBFLAB
```

PowerShell equivalent: `$env:RBFLAB_SOURCE_ROOT = "C:\path\to\RBFLAB"`.
The root must contain `cpp/` and its prepared dependencies. Do not mix unrelated
source versions with an installed wheel. Rebuild dedicated executables when their
source or build manifest changes. Do not reuse Linux binaries across incompatible
machines. For solver/cache details, see the [native reference](https://github.com/LDBreton/RBFLAB/blob/main/cpp/README.md).

## Troubleshooting by stage

| Message or symptom | Check |
|---|---|
| `No module named examples...` | Run from the source root; examples are not a wheel package |
| `Eigen headers missing` | Windows: run `python scripts/setup_windows_cpp.py`; Linux: run `bash cpp/prepare_dependencies.sh` |
| Native `g++` missing | Windows: run `python scripts/setup_windows_cpp.py` or set `RBFLAB_CXX`; Linux: install GCC |
| `libmpfr.so.6` / `libgmp.so.10` missing | Install Ubuntu runtime libraries; Windows DLLs cannot satisfy a Linux executable |
| `Build the C++ backend first` / manifest missing | Run `python -m examples.build_cpp_backend` from matching sources |
| Native sources changed | Rebuild; the content hashes intentionally reject stale binaries |
| Native preparation is slow on first use | Compare a second cached run; report cold and warm timings separately |
| Import uses the wrong installation | Check `python -m pip --version` and `rbflab.__file__` with the same interpreter |
| Plotting or Gmsh import is missing | Install the corresponding optional extra |

Geometry generation itself does not require Gmsh. See [backend comparisons](guides/curved-backends.md)
for matched-cloud tests and [contributor documentation](MAINTAINING_DOCS.md) for building this manual.
