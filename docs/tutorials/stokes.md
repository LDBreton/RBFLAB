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

## Native global fields

The stationary problem above can use C++ for kernel and polynomial assembly,
LU, repeated solves, and field evaluation. Select the backend on the same method:

```python
from examples.tutorials.stokes_spaces import make_problem
import rbflab as rbf

problem, spaces = make_problem(transient=False)
method = rbf.GlobalCollocation(
    spaces=spaces,
    precision=rbf.Precision(global_digits=60),  # omit global_digits for double
    backend=rbf.CppGlobalBackend(threads=2, compute_condition=False),
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
    spaces=spaces,
    backend=rbf.CppGlobalBackend(threads=2, compute_condition=False),
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
