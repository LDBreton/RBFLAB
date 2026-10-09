# Write a PDE as a symbolic problem

**Goal:** translate an equation, boundary data, and an approximation choice into
separate Python objects. The [first problem](../getting-started.md) is a shorter introduction.

## 1. State the problem

On an ellipse, solve

$$-\Delta u+\alpha u=f\quad\text{in }\Omega,\qquad u=g\quad\text{on }\Gamma_{\rm wall}.$$

Choose $\alpha=2$, $g=\sin x\cos y$, and $f=(2+\alpha)\sin x\cos y$.
A known solution makes this example concrete; ordinary applications supply their
own forcing and boundary expressions without knowing the solution.

```python
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
```

## 2. Give the geometry a named boundary

```python
--8<-- "examples/tutorials/symbolic_pde.py:geometry"
```

The cloud contains 200 interior and 80 boundary nodes. `wall` connects the
geometric subset to its boundary equation; it is not an unknown field name.
Generation also supplies the outward normals needed for flux conditions.

![Ellipse nodes and the introductory field](../assets/first_problem.png)

The left panel shows the same 280-node cloud. The right panel illustrates the
same analytic field in the introductory Poisson example. See
[mesh generation](../geometry/index.md) for composing shapes and boundary groups.

## 3. Create the unknown and coordinate symbols

```python
--8<-- "examples/tutorials/symbolic_pde.py:field"
```

`u` is a symbolic function, not a numerical vector. `x` and `y` are its
coordinates. `alpha` is a coefficient in the equation, independent of the
kernel's shape parameter. NumPy arrays enter when evaluating the solution.

## 4. Write the equations

```python
--8<-- "examples/tutorials/symbolic_pde.py:boundary"
```
```python
--8<-- "examples/tutorials/symbolic_pde.py:problem"
```

`sp.Eq(left, right)` constructs an equation; use it instead of Python `==`.
`model.laplacian(u)` expresses $u_{xx}+u_{yy}$. Ordinary SymPy derivatives such as
`sp.diff(u, x)` can appear in the same linear differential expression.

At this point `problem` specifies what to solve. It does not choose an RBF
kernel, a stencil size, or a numerical backend.

## 5. Choose the approximation and assemble

```python
--8<-- "examples/tutorials/symbolic_pde.py:solve"
```

| Object | Mathematical role |
|---|---|
| `problem` | $\mathcal Lu=f$, $\mathcal Bu=g$ |
| `cloud` | Interior/boundary points and geometric data |
| `method` | Approximation and stencil rules |
| `system` | Assembled equations $A_hU=b_h$ |
| `solution` | Computed values and field reconstruction |

PHS5, cubic augmentation, and 35-node neighborhoods define this example's recipe.
`scaling="local"` normalizes kernel distances by each stencil radius and restores
physical derivative factors automatically; see [local scaling](../theory/conditioning.md#local-stencil-scaling).
`problem.solve(cloud, method)` combines assembly and solution; separate steps
let you inspect `system.matrix` and `system.rhs` first.

## 6. Evaluate and display

```python
--8<-- "examples/tutorials/symbolic_pde.py:evaluate"
```

Both arrays contain one scalar per query. The derivative query differentiates
the method's reconstruction.

```python
from rbflab import viz
fig, ax = viz.plot_scalar(solution, domain=domain, title="Reaction-diffusion")
```

Plotting needs `rbflab[examples]`. Its display grid does not change the numerical cloud.

## 7. Change the mathematics

**Reaction or transport:** change `alpha`, or add `sp.diff(u, x)` to the left-hand
side. Supply the forcing for your new problem; changing an operator does not
automatically change an independently supplied forcing.

**Robin data:** replace `wall` before constructing `problem`:

```python
--8<-- "examples/tutorials/symbolic_pde.py:robin"
```

This imposes $\partial_nu+u=h$ using the cloud's outward normals, with $h$ chosen
from a known field. Run the script with `--robin`. To prescribe your own flux,
replace the right-hand side. Multiple labels accept different `model.bc(...)` entries.

**Approximation:** keep `problem` and replace the method with a compatible
`GlobalCollocation(...)` or `LHI(...)`. Their [global](global-collocation.md) and
[LHI](lhi.md) lessons explain why their matrices and unknown vectors differ.

### A quick check

Evaluate a few points against $\sin x\cos y$ to check the entered equation and data.

## Complete example

Run `python -m examples.tutorials.symbolic_pde` from a [source checkout](../INSTALL.md).

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/symbolic_pde.py"
    ```

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/symbolic_pde.py).

**Next:** [Construct the sparse equation yourself](custom-assembly.md), or use [mixed boundaries](ellipse.md).
