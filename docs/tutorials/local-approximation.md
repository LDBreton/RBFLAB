# From sampled functionals to sparse operators

**Goal:** describe a local approximation explicitly, then use its matrices in your
own numerical algorithm. No PDE or evolution object is required.

This interface is included in **RBFLAB 0.5.0 and newer**. It constructs local
maps for RBF-FD and scalar Hermite methods. The [symbolic PDE adapter](symbolic-pde.md)
and [global collocation](global-collocation.md) have different roles.

## 1. Separate approximation, data, and evaluation

On a local stencil, write

$$s(x)=\sum_j a_j m_j^yK(x,z_j)+\sum_k b_kp_k(x),\qquad \ell_i s=d_i.$$

Polynomial side conditions complete this construction:

$$\sum_j a_j(m_jp_k)=0\qquad\text{for every polynomial }p_k.$$

With $n$ source functionals, $n$ trial representers, and $q$ polynomial
terms, the local interpolation system is square of size $(n+q)\times(n+q)$:

$$
\mathcal A=
\begin{bmatrix}
(\ell_i^x m_j^yK) & (\ell_i p_k)\\
(m_j p_k)^T & 0
\end{bmatrix}.
$$

The source functionals \(\ell_i\) sample data; the trial functionals
\(m_j\) build kernel representers. They coincide in the ordinary and
symmetric Hermite constructions below, but need not in general.

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
functional is value evaluation. `scaling="local"` evaluates the radial kernel
using a stencil distance unit, schematically $\phi(\lVert x-z\rVert/R)$,
and returns requested derivatives in physical units. The default uses
physical distances. Shape parameters require care; see
[local scaling](../theory/conditioning.md#local-stencil-scaling).

The space uses the signed PHS5 kernel $\phi(r)=-r^5$ and polynomials of
total degree at most two. These choices determine the approximation. The
requested `value` and `lap` operators determine what we evaluate from it.

## 3. Apply or export the maps

For source values $U_j=u(x_j)$ and target points $y_i$,

$$(\Delta u)(y_i)\approx\sum_{j\in S_i}w_{ij}U_j,
\qquad D_\Delta\in\mathbb R^{N_{\mathrm{target}}\times N_{\mathrm{source}}}.$$

```python
--8<-- "examples/tutorials/local_approximation.py:apply"
```

`ops.lap["u"]` selects the block acting on the named `u` samples. With several
sample groups, `ops.lap @ {"u": U, "pde": F, ...}` sums their contributions.
Keep named blocks when building equations: their columns represent different
mathematical data. For SciPy algebra, `ops.lap["u"].to_scipy()` exports a
Float64 matrix; [capabilities](../CAPABILITIES.md) explains backend limits.

Here $X$ has 49 points, $Y$ has three targets, and each local solve uses
20 value samples plus six quadratic polynomial terms: a $26\times26$
system. The resulting Laplacian map is $3\times49$ and gives 4 for
$1+x^2+y^2$ to within about $6\times10^{-14}$. This is a reproduction
check, not a convergence study.

## 4. Inspect the local algebra

For a target functional $\tau$, the local weights satisfy

$$\mathcal A^T\begin{bmatrix}w\\\eta\end{bmatrix}
=\begin{bmatrix}(\tau^x m_j^yK)\\(\tau p_k)\end{bmatrix}
=q_\tau.$$

`.local(i)` exposes stored weights and source membership. `.reconstruct_local(i)`
builds the dense local matrix and right-hand side on demand; dense matrices are
not retained for every target. Polynomial multipliers $\eta$ are not global
solution unknowns. Repeating the local solve scatters weights into an
$N_{\rm target}\times N_{\rm source}$ map, which may be rectangular.
There is no inverse of that global map in the weight construction.
See [one stencil](one-stencil.md) and the
[weight derivation](../theory/local-weights.md).

A chosen trial/data pairing must produce a square, unisolvent local system.
Kernel smoothness must support the combined derivative orders. Arbitrary pairings
are not automatically stable merely because their matrices can be assembled.

??? info "Optional: bring in a symbolic differential expression"

    A symbolic model can supply only its spatial operator:

    ```python
    import sympy as sp
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    L = model.operator(-model.laplacian(u) + 2*sp.diff(u, x) + u)
    transport = local.operators(targets=Y, operators={"L": L})
    LU = transport.L["u"] @ U
    ```

    Forcing remains separate data. Compiling this expression does not
    assemble boundary rows or choose a time integrator.

## Change the construction

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

**Next:** [Assemble a stationary PDE](custom-assembly.md). For the full
local equation behind one row, use [one stencil](one-stencil.md).
