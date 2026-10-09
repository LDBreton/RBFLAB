# Your first RBF problem

RBFLAB connects four things: **geometry, equations, approximation, and solution**.
You can let the symbolic interface assemble a PDE, or work with the local weights
and sparse matrices yourself. This first example follows the symbolic route;
the [RBF-FD heat tutorial](tutorials/heat-equation.md) opens up the matrix route.

## Install and import

```sh
python -m pip install rbflab
```

The calculation below needs only the core package. Plotting is optional:
install `"rbflab[examples]"` for the last step. See [installation](INSTALL.md)
for virtual environments, C++ and PyTorch.

```python
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
```

## 1. Describe the domain and its boundary

```python
--8<-- "examples/tutorials/first_problem.py:geometry"
```

The ellipse has semi-axes 1.3 and 0.8. `wall` labels its perimeter; the result
contains 200 interior nodes and 80 boundary nodes. No triangles are needed.
Geometry construction does not choose an equation or a numerical method.
Learn to compose other shapes in [Mesh generation](geometry/index.md).

## 2. Write the equation

We solve a Poisson problem with a known answer:

$$
-\Delta u=2\sin x\cos y\quad\text{in }\Omega,
\qquad u=\sin x\cos y\quad\text{on }\partial\Omega.
$$

Thus $u_{\mathrm{exact}}=\sin x\cos y$. Using a known solution lets us check the
numerical result; ordinary applications supply their own forcing and boundary data.

```python
--8<-- "examples/tutorials/first_problem.py:equation"
```

`SymbolicScalar(2)` supplies a scalar field and two spatial coordinates.
`stationary` declares a time-independent equation. The string `wall` connects
the boundary equation to the nodes generated in step 1.

## 3. Choose RBF-FD and solve

```python
--8<-- "examples/tutorials/first_problem.py:solve"
```

`PHS(5)` selects a polyharmonic spline kernel. Each local stencil uses 35 nodes
and polynomials through degree three. Local coordinate scaling normalizes each
stencil before its weights are computed. The global matrix is sparse.

- **`problem`** describes the PDE and boundary equations.
- **`method`** chooses the approximation and its numerical settings.
- **`system`** holds the assembled algebraic problem.
- **`solution`** evaluates the computed field at requested coordinates.

The local stencil size must fit the cloud and be large enough for the polynomial
basis. These settings are a small example recipe, not universal accuracy defaults.

## 4. Check points that were not used in the solve

```python
--8<-- "examples/tutorials/first_problem.py:check"
```

The default run gives about $1.02\times10^{-4}$ in Float64.
This reports a maximum over 100 independently sampled interior points. It is a
sampled solution error, not a bound over the whole ellipse and not the residual
of the linear solver. See [errors and residuals](theory/errors.md).

![Ellipse nodes with their wall label and the computed Poisson solution](assets/first_problem.png)

## 5. Plot with one library call

```python
from rbflab import viz
fig, ax = viz.plot_scalar(solution, domain=domain, title="Poisson on an ellipse")
fig.savefig("poisson.png", dpi=160)
```

`domain` masks the exterior. The plotting grid only displays the solution;
it does not change the numerical nodes.

??? example "Complete runnable source"

    ```python
    --8<-- "examples/tutorials/first_problem.py"
    ```

[Download this self-contained script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/first_problem.py),
or run `python -m examples.tutorials.first_problem --plot poisson.png` from a
source checkout. `pip install rbflab` installs the library; it does not install
the repository's `examples` directory as a Python package.

## Choose the next level of control

| You want to… | Continue with… |
|---|---|
| Add holes or label curved boundary pieces | [Mesh generation](geometry/index.md) |
| Understand how local derivative weights arise | [One RBF-FD stencil](tutorials/one-stencil.md) |
| Assemble and time-step sparse operators yourself | [Heat with RBF-FD matrices](tutorials/heat-equation.md) |
| Compare global collocation, LHI and RBF-FD | [Compare numerical methods](tutorials/global-lhi.md) |
| Build a nonlinear flow algorithm | [Navier–Stokes cavity](tutorials/cavity.md) |

Changing methods or backends is a separate decision from changing the equation.
Compare errors whenever you change kernels, stencil rules or precision.
