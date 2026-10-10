# Solve a PDE with global collocation

**Goal:** follow a global trial expansion through PDE rows, boundary rows,
coefficients, and evaluation. After the [local stationary equation](custom-assembly.md),
this lesson shows what changes when one expansion spans the whole domain.

## 1. State the equation and choose centers

On the unit square, solve

$$-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y),\qquad u|_{\partial\Omega}=0.$$

```python
import numpy as np
import sympy as sp
import rbflab as rbf
```

```python
--8<-- "examples/tutorials/global_collocation.py:problem"
```

`unit_box_grid(5)` divides each axis into five intervals: six points per axis,
36 total. Sixteen are interior. This small cloud makes a dense matrix easy to
inspect; no element connectivity is used.

![The global centers and one LHI neighborhood on the same cloud](../assets/teaching_centers.png)

The left panel is the global cloud used here. The right anticipates the
[LHI lesson](lhi.md), which reuses the same PDE and nodes.

## 2. Choose an ordinary radial trial expansion

For asymmetric collocation, use

$$s(x)=\sum_{j=1}^N a_jK(x,x_j),\qquad
A_{ij}=\begin{cases}-\Delta_xK(x_i,x_j),&x_i\in\Omega,\\
K(x_i,x_j),&x_i\in\partial\Omega.\end{cases}$$

Every row imposes the requested equation on the trial function. Solve $Aa=b$,
where $b_i=f(x_i)$ in the interior and $b_i=g(x_i)$ on the boundary.

```python
--8<-- "examples/tutorials/global_collocation.py:asymmetric"
```

`system.matrix` and `system.rhs` are the dense collocation system.
`solution.coefficients` contains $a$, not $u(X)$. `solution.evaluate(query)`
forms the kernel expansion at the query locations.
The method has no local stencil-size parameter: all 36 centers contribute.

## 3. Connect the matrix to kernel evaluations

We can build those same rows explicitly:

```python
--8<-- "examples/tutorials/global_collocation.py:rows"
```

`left=-Laplacian()` applies the PDE to the evaluation argument. Omitting it gives
value rows for Dirichlet data. This unaugmented asymmetric example stores rows
and columns in `cloud.points` order, so `A` matches `system.matrix` directly.
`direct_values` shows exactly how coefficients become field values.

## 4. Change the trial space: symmetric collocation

Let $\lambda_i$ be the PDE or boundary functional at the $i$th functional center.
The symmetric construction uses

$$s(x)=\sum_j a_j\lambda_j^yK(x,y),\qquad
G_{ij}=\lambda_i^x\lambda_j^yK(x,y).$$

```python
--8<-- "examples/tutorials/global_collocation.py:symmetric"
```

The same problem is imposed, but the trial functions and coefficients change.
Use `hermite_solution.evaluate` to apply that expansion; multiplying its
coefficients by an ordinary kernel-value matrix would evaluate the wrong basis.
The unaugmented symmetric implementation orders interior functional centers
before boundary functional centers; `hermite_system.centers` and `.operators`
record that ordering. Do not assume array layouts from a different scheme.

## Modify the construction

- Change the symbolic equation or boundary data to change the imposed rows.
- Change `kernel` to change the trial functions.
- Add `polynomial_degree=...` when polynomial augmentation is desired or required;
  the resulting system also includes polynomial coefficients and side conditions.
- Use `solution.evaluate(query, rbf.Derivative(0))` for a derivative of the solution.

Global collocation uses dense storage. The local
[RBF-FD stencil](one-stencil.md) instead contributes a row of a sparse operator;
[LHI](lhi.md) adds PDE and boundary functionals to each local patch.

### A quick check

The script verifies that the explicitly constructed asymmetric matrix and
expansion agree with their assembled counterparts, then evaluates both methods.
Both manual comparisons are zero to the script's reported precision.
At its two off-node queries, the maximum errors are about
$1.10\times10^{-2}$ for asymmetric and $1.38\times10^{-2}$ for symmetric
collocation. The 36-node exercise teaches matrix meaning, not a ranking
of the methods or a convergence claim.

## Complete example

Run `python -m examples.tutorials.global_collocation` from a [source checkout](../INSTALL.md).
The short API fragments above also work with an installed package when combined
with their imports and preceding steps.

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/global_collocation.py).

**Next:** [Construct local Hermite interpolation](lhi.md).
