# Solve a PDE with local Hermite interpolation

For the canonical explicit local API, see [functionals to operators](local-approximation.md)
and [LHI and heat from matrices](lhi-matrices.md). This page explains the retained
PDE convenience interface.

**Goal:** understand which data enter an LHI patch and which weights enter the
sparse PDE equation. Read [one stencil](one-stencil.md) and the
[global construction](global-collocation.md) first.

## 1. Reuse the equation, change the local information

Use the same unit-square Poisson problem as the global lesson:

```python
import numpy as np
import sympy as sp
import rbflab as rbf
```

```python
--8<-- "examples/tutorials/lhi_construction.py:problem"
```

Unlike nodal RBF-FD, LHI incorporates nearby PDE and boundary data into the local
approximation. For each target $x_i$, distinguish three groups:

| Group | Functional | Data |
|---|---|---|
| $S_i$: solution centers | $s(x_j)$ | Unknown interior values $U_j$ |
| $B_i$: boundary centers | $\mathcal Bs(x_j)$ | Prescribed $g_j$ |
| $F_i$: PDE centers | $\mathcal Ls(x_j)$ | Prescribed $f_j$ |

A physical location may appear in $S_i$ and $F_i$ because these are different
functionals. The target remains a solution center and is excluded from its own
PDE-center group.

![Global cloud and actual LHI center groups for the first interior row](../assets/teaching_centers.png)

The right panel shows the patch used below: 12 solution centers, 8 boundary
centers and 11 PDE centers. Open triangles overlay solution points that also
carry PDE data. The target has no triangle.

## 2. Assemble and solve with LHI

```python
--8<-- "examples/tutorials/lhi_construction.py:solve"
```

`stencil_size=20` selects solution/boundary candidates, not the final Hermite
matrix dimension. By default, the other interior candidates also supply PDE
functionals. The first patch has 31 data functionals plus six quadratic
polynomial terms. The global system has only 16 interior value unknowns.

`interior_values[k]` belongs to `cloud.interior_indices[k]`. It is not an RBF
expansion coefficient. This scalar lesson uses the Python LHI assembler; do not
assume the scalar RBF-FD backend options apply to it.

## 3. Identify the local approximation

Stack the local functionals as $\lambda=(I_S,\mathcal B_B,\mathcal L_F)$ and write

$$s_i(x)=\sum_j a_j\lambda_j^yK(x,y)+\sum_m b_mp_m(x),$$

$$H_i=\begin{bmatrix}G_i&P_i\\P_i^T&0\end{bmatrix},\quad
(G_i)_{kj}=\lambda_k^x\lambda_j^yK,\quad (P_i)_{km}=\lambda_kp_m.$$

Applying $\mathcal L$ at the target yields weights by the transpose solve
$H_i^T[w_i;\eta_i]=q_i$. The polynomial rows of $q_i$ contain
$\mathcal Lp_m(x_i)$. See the [full derivation](../theory/lhi.md) for the blocks.
The library performs this construction during assembly.

The current scalar result exposes the center groups and functional weights:

```python
--8<-- "examples/tutorials/lhi_construction.py:patch"
```

`row` indexes an interior equation. `patch.center` is its original cloud index.
The weights follow solution, boundary, PDE order. Polynomial multipliers are
not included in `patch.weights` and do not become global unknowns.
These are inspection records; editing them is not a supported way to rebuild a
solved system. There is no generic scalar-LHI `reconstruct_local()` interface
matching RBF-FD's. Reassemble after changing method settings.

## 4. Form one sparse equation

The local identity is

$$\mathcal Ls_i(x_i)=(w_i^S)^TU_{S_i}+(w_i^B)^Tg_{B_i}+(w_i^F)^Tf_{F_i}.$$

Enforcing $\mathcal Ls_i(x_i)=f(x_i)$ gives

$$\boxed{(w_i^S)^TU_{S_i}=f(x_i)-(w_i^B)^Tg_{B_i}-(w_i^F)^Tf_{F_i}.}$$

Only the solution weights enter the sparse matrix. Known terms move right:

```python
--8<-- "examples/tutorials/lhi_construction.py:row"
```

The index dictionary maps original cloud indices into the shorter vector of
interior unknowns. Here $g=0$, but its contribution is shown explicitly. For
Neumann/Robin boundaries, $g$ means the prescribed boundary-functional value,
not necessarily the solution value itself.

## 5. Reconstruct the field

```python
--8<-- "examples/tutorials/lhi_construction.py:evaluate"
```

The solution fills each local data vector with computed $U$ and prescribed
$g,f$, obtains its expansion, and uses the nearest interior target's patch at a
query point. That patchwise reconstruction need not be continuous across patch
ownership boundaries. A derivative query differentiates the selected local
expansion, not the sparse solution vector directly.

## Modify the idea

- Change the equation and boundary objects to change the local functionals.
- Set `pde_stencil_size` to control the number of nearby PDE centers separately.
- Change the kernel and augmentation degree together.
- Keep center counts distinct from functional counts when designing a new patch.

### A quick check

The script reconstructs the first sparse row and its right-hand side from the
three weight groups and compares them with `system.matrix` and `system.rhs`.

## Complete example

Run `python -m examples.tutorials.lhi_construction` from a [source checkout](../INSTALL.md).
The short API fragments above also work with an installed package when combined
with their imports and preceding steps.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/lhi_construction.py"
    ```

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/lhi_construction.py).

## Independent functional centers

Use [named center groups](lhi-centers.md) to choose solution, boundary, PDE, and derivative-data points separately.
