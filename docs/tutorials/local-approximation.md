# From sampled functionals to sparse operators

**Goal:** describe a local approximation explicitly, then use its matrices in your
own numerical algorithm. No PDE or evolution object is required.

This interface is included in **RBFLAB 0.5.0 and newer**. Existing `RBFFD` and
`LHI` convenience interfaces remain supported. Standard RBF-FD operator/weight
construction and scalar LHI weight assembly already delegate to this shared
engine; specialized coupled/Stokes routes remain separate.

## 1. Separate approximation, data, and evaluation

On a local stencil, write

$$s(x)=\sum_j a_j m_j^yK(x,z_j)+\sum_k b_kp_k(x),\qquad \ell_i s=d_i.$$

The **space** supplies $K$ and the polynomial tail. The **trial** supplies $m_j$.
The **source samples** supply $\ell_i$ and their locations. A **target** supplies
$\tau s$, the quantity we want to evaluate.

For ordinary RBF-FD, $m_j$ and $\ell_j$ are point evaluations. For Hermite
interpolation, they can be derivative or PDE functionals. A space's
`representers(source)` applies each source functional to the second kernel
argument. `space.translates(points)` instead describes ordinary kernel translates.
The trial choice is visible, rather than hidden behind a scheme name.

## 2. Declare a space and value samples

```python
import numpy as np
import rbflab as rbf
implementation = rbf.PythonBackend(compute_condition=False)
```

```python
--8<-- "examples/tutorials/local_approximation.py:construction"
```

`Samples(X, size=20)` selects 20 source points around each target. The default
functional is value evaluation. `scaling="local"` normalizes kernel coordinates
using a stencil scale and includes derivative chain-rule factors; a kernel shape
parameter therefore has a dimensionless interpretation. The default is physical
coordinates. See [conditioning](../theory/conditioning.md).

The space uses $\phi(r)=r^5$ and polynomials of total degree at most two. These
choices determine the approximation. The requested `value` and `lap` operators
only determine what we evaluate from it.

## 3. Apply or export the maps

For source values $U_j=u(x_j)$ and target points $y_i$,

$$(\Delta u)(y_i)\approx\sum_{j\in S_i}w_{ij}U_j,
\qquad D_\Delta\in\mathbb R^{N_{\mathrm{target}}\times N_{\mathrm{source}}}.$$

```python
--8<-- "examples/tutorials/local_approximation.py:apply"
```

`ops.lap["u"]` selects the block acting on the named `u` samples. With several
sample groups, `ops.lap @ {"u": U, "pde": F, ...}` sums their contributions.
`ops.lap.matrix` concatenates blocks in source declaration order. Native matrices
retain backend storage, including Torch sparse tensors. Call
`ops.lap["u"].to_scipy()` for an explicit Float64 SciPy export; this detaches Torch
data and does not preserve extended arithmetic. Keep named blocks
when building equations: their columns represent different mathematical data.

Application also retains native backend output: Torch returns tensors. For
NumPy-only reporting, explicitly use `values.detach().cpu().numpy()`; keep tensors
when continuing a Torch calculation.

The quadratic example reconstructs a field and its constant Laplacian. The
script prints the errors as a brief check, not as a convergence study.

## 4. Inspect the local algebra

For a square augmented local system $\mathcal A$, the target weights satisfy

$$\mathcal A^T\begin{bmatrix}w\\\eta\end{bmatrix}=q_\tau.$$

`.local(i)` exposes stored weights and source membership. `.reconstruct_local(i)`
builds the dense local matrix and right-hand side on demand; dense matrices are
not retained for every target. Polynomial multipliers $\eta$ are not global
solution unknowns. See [the weight derivation](../theory/local-weights.md).

A chosen trial/data pairing must produce a square, unisolvent local system.
Kernel smoothness must support the combined derivative orders. Arbitrary pairings
are not automatically stable merely because their matrices can be assembled.

## 5. Bring a symbolic operator into the same construction

A symbolic equation object is optional; the symbolic compiler can supply just
its spatial differential expression:

```python
import sympy as sp
model = rbf.SymbolicScalar(2)
u = model.field
x, y = model.coordinates
L = model.operator(-model.laplacian(u) + 2*sp.diff(u, x) + u)
transport = local.operators(targets=Y, operators={"L": L})
LU = transport.L["u"] @ U
```

Forcing is separate data, not part of the homogeneous operator expression.
The same `L` can appear in `Samples(Xf, operator=L)` when fitting PDE data in a
Hermite construction. Compiling the expression does not assemble boundary rows
or choose a time integrator.

## 6. Change the construction

- Replace `Y` to interpolate or differentiate on another cloud.
- Add named operators to reuse a local factorization for multiple evaluations.
- Replace value samples by PDE or boundary functionals: [LHI from matrices](lhi-matrices.md).
- Replace the space's kernel, including a [symbolic kernel](custom-kernel.md).
- Select a supported backend; the global sparse solver remains your choice.

```sh
python -m examples.tutorials.local_approximation
python -m examples.tutorials.local_approximation --backend cpp
python -m examples.tutorials.local_approximation --backend torch
```

Run these commands from a source checkout. C++ requires its optional source-build
toolchain; Torch is optional and currently CPU Float64. See
[installation](../INSTALL.md) and [capabilities](../CAPABILITIES.md).
