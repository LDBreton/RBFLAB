# Verified capabilities

The method and backend combinations below are intentionally precise. A local
backend choice does not automatically change a separate global sparse solve.

| Capability | Python | C++ double | C++ MPFR | PyTorch |
|---|---|---|---|---|
| Scalar global collocation | Yes | — | — | — |
| Scalar LHI convenience assembly (`LHI`) | Yes | — | — | — |
| Scalar RBF-FD operators, 2D/3D | Yes | Yes, from source checkout | Supported; installed archive test pending | Yes, CPU Float64 |
| Divergence-free steady Stokes, global | Yes | — | — | — |
| Divergence-free Stokes LHI, 2D | Yes | Yes, from source archive | 50 digit local weights verified from source archive | Unaugmented stencils only; not the curated example |
| Custom symbolic kernel evaluation | Yes | Compilation from source checkout | Compilation from source checkout | Supported where the local backend accepts the kernel |
| Extended mpmath global sparse solve | Supported by selected Python methods | Separate from local C++ backend | Separate setting | — |

The curated [3D operator example](https://github.com/LDBreton/RBFLAB/blob/main/examples/operators_3d.py) ran with Python,
C++ double, and PyTorch Float64 on Windows/WSL. The
[Stokes example](https://github.com/LDBreton/RBFLAB/blob/main/examples/stokes_spaces.py) ran with Python, C++ double,
and C++ MPFR local weights.
The six default Python examples ran from an installed wheel and source archive
outside the checkout.

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

Standard `RBFFD.operators`/`.weights` and scalar `LHI` weight construction now
reuse this engine. The RBF-FD compatibility adapter still exports SciPy matrices
for Torch; direct `LocalApproximation` keeps tensors. Scalar LHI preserves its
historical data/solution interpretation. Existing specialized Stokes and coupled
assembly paths remain separate.

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
