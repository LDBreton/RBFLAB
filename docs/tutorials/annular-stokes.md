# Couple velocity and pressure in Stokes flow

**Goal:** express coupled fields and choose a divergence-free velocity space.
Read [symbolic PDEs](symbolic-pde.md) and [global collocation](global-collocation.md) first.

## 1. Model creeping flow between rotating cylinders

In an annulus with radii $a=0.5$ and $b=1$, solve

$$-\mu\Delta\boldsymbol u+\nabla p=0,\qquad \nabla\cdot\boldsymbol u=0,\qquad\mu=1.$$

The inner wall rotates with angular speed one, so $\boldsymbol u=(-y,x)$ there.
The outer wall is fixed. There is no convective term in steady Stokes flow.

```python
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
```

```python
--8<-- "examples/tutorials/stokes_construction.py:geometry"
```

The two labels connect wall pieces to different vector boundary equations. The
cloud has 70 interior and 80 boundary nodes.

![The annular cloud and computed velocity](../assets/annular_stokes.png)

## 2. Declare the fields and momentum equation

```python
--8<-- "examples/tutorials/stokes_construction.py:fields"
```

`U` is a two-component symbolic vector; `p` is a scalar symbolic function.
`laplacian(U)` acts component by component, while `gradient(p)` is a vector.
Their sum expresses the two momentum equations.

```python
--8<-- "examples/tutorials/stokes_construction.py:boundary"
```

`sp.zeros(2, 1)` is a vector right-hand side. Boundary equations also compare
vectors. This velocity-Dirichlet problem does not prescribe pressure on the wall.

## 3. Choose the approximation spaces

```python
--8<-- "examples/tutorials/stokes_construction.py:spaces"
```

The dictionary keys are the symbolic fields. Each value describes how that
field is approximated. For velocity, the scalar radial kernel generates

$$\Phi_{\rm div}(x-y)=\bigl(\nabla\nabla^T-I\Delta\bigr)\phi(\|x-y\|).$$

Every column has zero divergence. Velocity polynomials are restricted to
divergence-free modes as well. Thus the equation list contains momentum, while
incompressibility belongs to the trial space. Adding an explicit
`Eq(model.divergence(U), 0)` is not how this space-based route is specified.

The velocity polynomial degree is two; the pressure degree is one. Pressure is
represented modulo additive constants. This formulation handles the nullspace
in its construction; it is not an arbitrary extra pressure row in a scalar
block system. The [space derivation](../theory/divergence-free.md) explains the
coupled kernel and its pressure representation.

## 4. Recover physical fields

```python
--8<-- "examples/tutorials/stokes_construction.py:evaluate"
```

Velocity and pressure gradient have shape `(M, 2)`; pressure has shape `(M,)`.
The reference fixes the displayed pressure constant at $(0.75,0)$. It does not
alter velocity or the pressure gradient.

Here the analytic solution is

$$\boldsymbol u=(-y,x)\frac{1/r^2-1}{3},\qquad\nabla p=0.$$

Constant pressure belongs to this creeping-flow model. Finite-inertia circular
flow has a different momentum balance.

## 5. Display and modify the problem

```python
from rbflab import viz
fig, ax = viz.plot_velocity(solution, domain=domain, title="Rotating inner wall")
```

- Change wall expressions to prescribe another velocity.
- Replace the zero momentum right-hand side with a vector body force.
- Change kernels independently in velocity and pressure spaces.
- For another domain, supply its corresponding boundary labels and data.

This lesson uses the Python global space assembler. Selecting a local C++ backend
does not accelerate this dense system. LHI Stokes has different capabilities,
including pressure-gradient reconstruction rather than a globally reconciled
pressure field; see [capabilities](../CAPABILITIES.md) before changing routes.

### A quick check

Compare velocity with the azimuthal formula at a few points. Pressure-gradient
accuracy is a separate question; the existing [validation record](../guides/curved-validation.md)
reports it independently.

## Complete example

Run `python -m examples.tutorials.stokes_construction` from a source checkout.
Add `--plot stokes.png` to save a velocity figure.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/stokes_construction.py"
    ```

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/stokes_construction.py).

**Next:** [Build a cavity algorithm](cavity.md), using a different scalar-space
velocity-pressure discretization and explicit sparse blocks.
