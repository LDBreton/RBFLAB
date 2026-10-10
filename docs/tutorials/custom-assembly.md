# Assemble your own PDE from RBF operators

**Goal:** use RBFLAB for differentiation while owning equation assembly and the
sparse solve. Read [from samples to operators](local-approximation.md) first;
[one stencil](one-stencil.md) opens a single weight row if you want that detail.

## 1. Choose the mathematical expression

Consider a non-divergence-form equation:

$$-\kappa(x,y)\Delta u+b_xu_x+b_yu_y+\alpha u=f,\qquad u|_\Gamma=g,$$

where $\kappa=1+0.2x$, $b_x=0.4$, $b_y=-0.2$, and $\alpha=1$.
We will write every matrix operation explicitly.

```python
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
```

```python
--8<-- "examples/tutorials/custom_assembly.py:cloud"
```

`I` and `B` select interior and boundary rows of `X`. Every full matrix below
uses the original cloud ordering.

![Square nodes and their sparse Laplacian](../assets/teaching_operators.png)

This is the square cloud and 35-node stencil recipe used by the script.
The sparsity plot identifies which source values contribute at each target.

## 2. Construct reusable differential maps

```python
local = rbf.PythonBackend(compute_condition=False)
```

```python
--8<-- "examples/tutorials/custom_assembly.py:operators"
```

We explicitly set `targets=X`, so all three matrices have shape $(N,N)$.
`Samples` describes the supplied values and selects 35 neighbors per target;
`space.representers(source)` makes the trial functions visible.
`to_scipy()` exports each source block as Float64 for the SciPy algebra below.
Their rows approximate $u_x$, $u_y$, and $\Delta u$. The recipe uses PHS5 and cubic
polynomials. [Local scaling](../theory/conditioning.md#local-stencil-scaling)
normalizes kernel distances while preserving physical derivative units.
`compute_condition=False` skips an optional diagnostic, not part of the equation.

## 3. Translate the equation into sparse algebra

For nodal samples $U$,

$$A_h=-\operatorname{diag}(\kappa(X))L+b_xD_x+b_yD_y+\alpha I_N.$$

```python
--8<-- "examples/tutorials/custom_assembly.py:combine"
```

Left multiplication by the diagonal samples $\kappa$ at the target of each row.
This is $-\kappa\Delta u$, not $-\nabla\cdot(\kappa\nabla u)$. The latter also
contains $-\nabla\kappa\cdot\nabla u$ by the product rule. Keep the continuous
equation clear when combining matrices.

Boundary rows of `A` still represent the differential expression. Next we apply
the boundary condition separately.

## 4. Supply the data and eliminate prescribed values

For $u=\sin x\cos y$, take
$f=(2\kappa+\alpha)u+b_x\cos x\cos y-b_y\sin x\sin y$ and $g=u|_\Gamma$.

```python
--8<-- "examples/tutorials/custom_assembly.py:data"
```

The interior equations split into

$$A_{II}U_I+A_{IB}g_B=f_I,\qquad A_{II}U_I=f_I-A_{IB}g_B.$$

```python
--8<-- "examples/tutorials/custom_assembly.py:solve"
```

`U` contains nodal values in cloud order, not expansion coefficients. This is
your SciPy solve: you can replace its solver, reuse a factorization, or embed it
in a larger algorithm. Neumann/Robin conditions do not prescribe $U_B$ directly;
assemble boundary-functional rows instead of reusing this elimination unchanged.

For this 121-node square example, $A_h$ is $121\times121$ and the interior
solve is $81\times81$. The maximum nodal error against the manufactured
$\sin x\cos y$ field is about $2.10\times10^{-5}$.

## 5. Inspect the local construction

```python
--8<-- "examples/tutorials/custom_assembly.py:inspect"
```

`local_row` contains copies of indices, weights and polynomial multipliers.
`local_system` reconstructs the augmented local weight equation on demand.
Editing these inspection outputs does not update the operator.

Here `Lap` is an explicit SciPy export. Copy matrices before experimenting;
changing a matrix does not update cached local inspection records.

For custom neighborhoods, use `Samples(points, indices=rows)`, where each row
contains that target's selected group-local indices. Alternatively create an
approximation from `Samples(selected_points)` and call
`approximation.weights(target=target, operator=operator)`.
The [one-stencil lesson](one-stencil.md) shows this complete construction.
Choose `space.translates(...)` or `space.representers(...)` explicitly when
experimenting with the trial basis.

??? info "Optional: compare symbolic equation assembly"

    ```python
    --8<-- "examples/tutorials/custom_assembly.py:symbolic"
    ```

    This uses the `RBFFD` equation-assembly adapter with the same cloud,
    kernel, polynomial degree, stencil size, and scaling. Its boundary
    unknowns need not be eliminated in the same form. The script compares
    `U` with `symbolic.evaluate(X)`.

## Where to put your own idea

| Change | Relevant construction |
|---|---|
| New radial family | Kernel supplied before constructing `ops` |
| New differential expression | Requested operators and matrix combination |
| New boundary model | Boundary rows or elimination |
| New solver/preconditioner | The `spsolve` call |
| New time integrator | A loop around the spatial matrices |
| New neighbor rule | Memberships in `Samples`, then `approximation.weights` |

New values on fixed nodes can reuse fixed weights. Changing nodes, kernel or
stencil settings requires rebuilding. With [backend setup](../INSTALL.md), the
script accepts `--backend cpp` or `--backend torch` for local assembly. The global
solve here remains SciPy Float64.

## Complete example

Run `python -m examples.tutorials.custom_assembly` from a source checkout.

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/custom_assembly.py).

**Next:** [Compare global collocation](global-collocation.md), then
[construct LHI from functional data](lhi.md).
