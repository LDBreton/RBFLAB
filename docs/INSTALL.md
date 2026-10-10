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

The current manual targets **RBFLAB 0.5.0**. If an older environment is active,
upgrade using the same Python interpreter that runs your code:

```sh
python -m pip install --upgrade "rbflab>=0.5"
python -c "from rbflab import Samples, LocalApproximation; from importlib.metadata import version; print(version('rbflab'))"
```

Release 0.5 also includes breaking research-API cleanup from 0.4. Consult the
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
Ubuntu x86-64 or Ubuntu under WSL. Native Windows/MSVC and macOS builds are not
covered by this guide. Python-only installation remains independent of these tools.

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

## 5. Enable C++ from Windows through WSL

The tested arrangement keeps **Python on Windows** and runs **GCC/native executables
inside WSL Ubuntu**. This does not require moving the Python environment into Linux.
Keep the checkout on a Windows drive accessible to WSL. Do not reuse a Windows
virtual environment from a Linux Python interpreter, or vice versa.

If Ubuntu WSL is not installed, install it with `wsl --install -d Ubuntu` from
an administrator PowerShell, finish Ubuntu's first-run setup, and reopen the terminal.
Check the distribution name with:

```powershell
wsl --list --verbose
```

From the RBFLAB checkout in ordinary PowerShell:

```powershell
wsl -d Ubuntu -- sudo apt-get update
wsl -d Ubuntu -- sudo apt-get install -y g++ libmpfr6 libgmp10
$windowsRepo = (Get-Location).Path.Replace('\', '/')
$linuxRepo = (wsl -d Ubuntu --exec wslpath -a $windowsRepo).Trim()
wsl -d Ubuntu --exec bash "$linuxRepo/cpp/prepare_dependencies.sh"
python -m pip install -e ".[examples,test]"
python -m examples.annulus --backend cpp
```

Use your environment's Python command from step 1. `wslpath` derives the Linux path
from your checkout; no personal drive path is required.

For built-in vector LHI:

```powershell
python -m examples.build_cpp_backend
$env:RBFLAB_CPP_TESTS = "1"
python -m pytest tests/test_discrete_operators.py tests/test_symbolic_kernel.py -q
Remove-Item Env:RBFLAB_CPP_TESTS
```

The supplied build helper and generated-kernel compiler currently target a WSL
distribution named **Ubuntu**. Use that name for the documented route; changing
one backend field does not reconfigure every compiler helper.

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
system for you. C++ does not accelerate the global dense collocation assembler.

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
| `Eigen headers missing` | Run `cpp/prepare_dependencies.sh`; check `cpp/deps/usr/include/eigen3/Eigen/Core` |
| `g++` missing / WSL cannot find Ubuntu | Install the compiler in Ubuntu and verify `wsl --list --verbose` |
| `libmpfr.so.6` / `libgmp.so.10` missing | Install Ubuntu runtime libraries; Windows DLLs cannot satisfy a Linux executable |
| `Build the C++ backend first` / manifest missing | Run `python -m examples.build_cpp_backend` from matching sources |
| Native sources changed | Rebuild; the content hashes intentionally reject stale binaries |
| Native preparation is slow on first use | Compare a second cached run; report cold and warm timings separately |
| Import uses the wrong installation | Check `python -m pip --version` and `rbflab.__file__` with the same interpreter |
| Plotting or Gmsh import is missing | Install the corresponding optional extra |

Geometry generation itself does not require Gmsh. See [backend comparisons](guides/curved-backends.md)
for matched-cloud tests and [contributor documentation](MAINTAINING_DOCS.md) for building this manual.
