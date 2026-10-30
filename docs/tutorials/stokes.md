# Steady and unsteady Stokes in a divergence-free space

**You will learn:** to declare coupled velocity and pressure fields, assign a divergence-free velocity space, and inspect pressure-gradient error. Prerequisites: [divergence-free kernels](../theory/divergence-free.md) and [time discretization](../theory/time-discretization.md).

Steady Stokes momentum is

$$
-\mu\Delta\boldsymbol u+\nabla p=\boldsymbol f.
$$

Unsteady momentum adds \(\partial_t\boldsymbol u\). The manufactured example uses \(\mu=1\), \(\boldsymbol u=(e^{-t}y(1-y),0)\), and \(p=-2x\); the steady version omits the exponential. Boundary values and forcing are derived from these fields.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/stokes_spaces.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/stokes_spaces.py).

Run `python -m examples.tutorials.stokes_spaces`. Velocity uses the matrix kernel \(\nabla\nabla^{\mathsf T}\phi-I\Delta\phi\), which is divergence-free by construction. The symbolic system supplies momentum equations but no redundant divergence equation. Pressure is defined up to a constant, so the example checks \(\nabla p\), not an absolute pressure value.

For the 25-node cloud, both global systems were \(61\times61\). A verified steady query had velocity error \(1.482\times10^{-17}\), pressure-gradient error \(4.521\times10^{-16}\), and reported divergence zero. At \(t=0.03\), the short BDF2 run had velocity error \(1.271\times10^{-7}\), pressure-gradient error \(3.296\times10^{-6}\), and reported divergence zero. These are **single-query** errors, not domain norms.

BDF2 starts with one backward Euler step. The initial condition sets velocity, not pressure. Try changing \(\mu\) and rederive forcing. See [equations and spaces](../api/equations-spaces.md).

## A simple local Stokes example

`examples/stokes_lhi_baseline.py` follows five steps: choose an exact solution,
derive its forcing, declare the kernels and spaces, assemble `LHI`, then compare
the nodal solution with the exact fields. It uses homogeneous walls, 37-point
stencils, velocity/pressure augmentation 6/1 and the fully mesh-scaled kernels
with pressure coefficient `cbar_p=1`.

```bash
python -m examples.stokes_lhi_baseline
python -m examples.stokes_lhi_baseline --ranks 2
```

The default is a small 81-node example with C++ weights. MPI requires a launcher
as described in [installation](../INSTALL.md). Both time steps run to `T=0.1`;
the printed errors are absolute vector RMS values at interior mesh nodes.
The coarse default demonstrates the workflow; use the fine meshes to study
the accuracy of the baseline.
For the larger meshes, use `--cells 89` (8,100 nodes) or `--cells 126`
(16,129 nodes). This direct teaching route constructs every stencil and can
take substantially longer than a specialized batched reproduction driver.

??? example "Complete local Stokes script"

    ```python
    --8<-- "examples/stokes_lhi_baseline.py"
    ```

To inspect a single local weight construction, start with
`python -m examples.stokes_lhi_weights`. It shows `MixedSpace`, `FieldOperator`,
`Samples` and `LocalApproximation` directly. Add `--backend cpp --digits 80 --mpi 2`
for native MPI weights. The simpler two-scalar-field example is
`python -m examples.mixed_local_maps`.

## Navier-Stokes with explicit local assembly

`examples/navier_stokes_lhi.py` builds the matrices directly with the public
`LocalApproximation` API and advances the solution with SciPy:

```bash
python -m examples.navier_stokes_lhi --cells 8 --nu 1 --ranks 2
```

The example declares `MixedSpace` and unpacks `u, p = space.fields`.
One `Stencil(points, size=37)` is shared by interior velocity, boundary
velocity and neighboring momentum samples; momentum excludes the target.
The mixed maps give `A`, `B`, `SL`, `PY`, `PB` and `PL`, with `M = I - SL`.
A separate degree-5 divergence-free value reconstruction gives interior
`Dx`, `Dy` and boundary `Dxb`, `Dyb` blocks on the same 37 sites.
All vector blocks use node-major ordering. Prescribed boundary data enter
momentum as `-B @ g(t)`, pressure recovery as `PB @ g(t)`, and convection
derivatives as `Dx @ U + Dxb @ g(t)` and `Dy @ U + Dyb @ g(t)`.

The example uses 6/1 mixed augmentation, full h-scaled kernels and 80-digit
native weights with MPI. Backward Euler starts BDF2 with explicitly
extrapolated numerical convection. Both time steps reach `T=0.1`, and only
interior-nodal absolute vector RMS and maximum errors are measured. The small
default mesh illustrates the workflow; it is not a fine-grid accuracy claim.
Changing viscosity rebuilds the mixed weights. The global SciPy solve is
serial, and locally divergence-free reconstructions do not impose global
divergence conformity. The manufactured velocity includes a constant spatial
offset multiplied by `exp(-t)`, giving nonzero time-dependent boundary data.
Use `--offset 0` to recover homogeneous walls. This example does not implement
the cavity wall and corner treatment.

The 81-node viscosity-1 example with the default offset completed both time
steps with velocity RMS about 0.009930 and pressure-gradient RMS about 0.6587.
The previous homogeneous-wall case gave about 0.01025 and 0.6585 respectively.
A viscosity-0.1 homogeneous-wall trial on this coarse mesh became nonfinite
at step 109 with `dt=0.00025`; do not use
that configuration as a stable baseline. Fine-grid results require separate
runs and are not inferred from this example.

??? example "Complete local Navier-Stokes script"

    ```python
    --8<-- "examples/navier_stokes_lhi.py"
    ```

For readable research implementations of alternative Navier–Stokes time schemes,
see `research/navier_stokes_schemes/README.md` in the repository. The comparison
includes coupled Oseen/Newton BDF2, midpoint and ARS IMEX variants, rotational
convection, characteristics, and an experimental rebuilt Oseen Hermite closure.
These are research drivers, not additional stable package solver APIs. Their
coarse verification runs must not be confused with the fine-grid baseline.

## Native global fields

For the local Stokes method in source version 0.6.0, see
[shared local weight construction](../guides/research-api.md#existing-stokes-lhi-through-the-common-engine).
Python/C++ LU `LHI(spaces=...)` now uses `LocalApproximation` for weights, while
retaining the compact mass matrix and local pressure-gradient output. Add
`execution=LocalExecution(scheduler="mpi", ranks=2)` with `CppBackend` to
distribute those local weights. This does not select a native global solver or
introduce a scalar pressure gauge field.
For fixed kernel parameters, this C++ LU path also accepts
`stencil_policy=StencilPolicy(scaling="local")`, including with MPI. Its radius
is measured from each target to the selected source points, matching Python
Stokes LHI. Returned derivatives already include their physical scaling factors.
Keep physical coordinates when reproducing a recipe whose kernel parameters
already include the desired mesh scaling. Native SVD and historical custom
executables retain their physical-coordinate requirement.

The stationary problem above can use C++ for kernel and polynomial assembly,
LU, repeated solves, and field evaluation. Select the backend on the same method:

```python
from examples.tutorials.stokes_spaces import make_problem
import rbflab as rbf

problem, spaces = make_problem(transient=False)
method = rbf.GlobalCollocation(
    spaces=rbf.MixedSpace(spaces),
    precision=rbf.Precision(global_digits=60),  # omit global_digits for double
    backend=rbf.CppGlobalBackend(threads=2),
)
cloud = rbf.geometry.unit_box_grid(4)
with method.assemble(problem, cloud) as system:
    solution = system.solve()
    velocity = solution.velocity([[.3, .4]], extended=True)
    gradient = solution.pressure_gradient([[.3, .4]], extended=True)
    divergence = solution.divergence([[.3, .4]], extended=True)
    pressure = solution.pressure(
        [[.3, .4]], reference=([0., 0.], 0), extended=True,
    )
    momentum_error = solution.residuals([[.3, .4]], extended=True)
```

`DivergenceFreeSpace` applies the matrix-valued kernel transform to the velocity
potential; pressure has its own scalar potential. Each potential can use a
different supported kernel family and parameter values. Velocity polynomial
columns lie in the exact divergence-free nullspace. Pressure polynomials omit
the constant mode, so prescribing a reference shifts the evaluated pressure
without adding a gauge equation to the matrix.

This native path supports **steady and unsteady symmetric Stokes in 2D and 3D** with
constant viscosity and prescribed velocity boundaries, with or without
polynomial augmentation. Asymmetric field collocation, general mixed-vector
equations and coupled scalar blocks are not supported. Kernel derivatives through total
order six cover native velocity values, first/second derivatives and momentum
residuals; `derivative_order` above two is rejected for native Stokes.

Field methods return NumPy arrays by default. With `global_digits` set,
`extended=True` retains mpmath output converted directly from MPFR decimal
results. Geometry still starts as Float64. Evaluate before closing the system;
RHS edits reuse the native factors, and matrix edits invalidate them.

For unsteady Stokes, use `make_problem(transient=True)` and the same backend:

```python
problem, spaces = make_problem(transient=True)
method = rbf.GlobalCollocation(
    spaces=rbf.MixedSpace(spaces),
    backend=rbf.CppGlobalBackend(threads=2),
)
with method.assemble(problem, cloud) as system:
    trajectory = system.solve(dt="0.01", steps=100, scheme="bdf2")
    velocity = trajectory.final.velocity([[.3, .4]])
    gradient = trajectory.final.pressure_gradient([[.3, .4]])
    A = system.matrix.to_numpy()  # explicit export of native stiffness
    M = system.mass.to_numpy()    # explicit export; read-only native mass
    b = system.data(1.0)
```

The native worker assembles `A` and `M` for `M c_dot + A c = b(t)`. Only
interior velocity rows have mass; boundary and polynomial constraint rows have
zero mass. Python evaluates forcing and boundary data at each time in the
selected precision. Backward Euler and BDF2 perform native LU solves and mass
products without exporting either dense matrix. BDF2 starts with backward
Euler; fixed time steps reuse two shifted factorizations. Up to four shifts
are cached per system. Editing `system.matrix` invalidates these factors;
`system.mass` is read-only. `system.matrix` remains the stiffness matrix after
time stepping. Use `.to_mpmath()` for explicit extended-precision matrix export.
The trajectory and all its field predictions require the system to remain open.

Run `python -m examples.global_cpp_unsteady --power 9 --digits 40` for a
manufactured IMQ + PHS r9 velocity / r5 pressure example. Its three-query errors
include both spatial and time discretization error; they are not domain norms.

Run `python -m examples.global_cpp_fields` for a double check, or
`python -m examples.global_cpp_fields --dimension 3 --digits 60` for extended
arithmetic. This example measures reproduction of a quadratic divergence-free
field and linear pressure at five off-node points, not general convergence.
