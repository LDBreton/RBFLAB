# Heat diffusion with RBF-FD

![Computed heat diffusion on a flower domain](../assets/flower_heat.gif)

**Learn two levels of control:** declare a transient PDE symbolically, then build
the RBF-FD Laplacian and time step its sparse matrix yourself. Both routes use RBF
weights. The square below is only a convenient domain with a known decaying mode;
the stencil construction also works on irregular clouds.

## 1. Start with the equation and a curved domain

For a diffusivity $\kappa>0$,

$$u_t-\kappa\Delta u=f,\qquad u|_{\partial\Omega}=g(t),\qquad u(\cdot,0)=u_0.$$

The flower example uses $\kappa=0.15$ and the known solution

$$u_{\rm exact}(x,y,t)=e^{-2t}\left(\cos x\cos y+\tfrac14\sin(2x)\right).$$

The forcing is $f=\partial_tu_{\rm exact}-\kappa\Delta u_{\rm exact}$, and the boundary
values are its trace. This is a forced verification case, not an insulated heat pulse.
Here is the complete core calculation, without example-helper functions:

```python
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen

domain = geometry.Flower()
cloud = meshgen.generate(domain, interior=250, boundary=120, seed=42)
model = rbf.SymbolicScalar(2, transient=True)
u, t = model.field, model.time
x, y = model.coordinates
exact = sp.exp(-2*t)*(sp.cos(x)*sp.cos(y) + sp.sin(2*x)/4)
lhs = sp.diff(u,t) - sp.Rational(15,100)*model.laplacian(u)
problem = model.evolution(
    sp.Eq(lhs, lhs.subs(u,exact).doit()), initial=exact.subs(t,0),
    boundary=[model.bc("boundary", sp.Eq(u,exact))],
)
method = rbf.RBFFD(rbf.PHS(5), 35, polynomial_degree=3,
    stencil_policy=rbf.StencilPolicy(scaling="local"))
trajectory = problem.solve(cloud, method, "0.025", 20, scheme="bdf2")
```

The 370-node recipe uses PHS5, degree-three polynomials, 35-node stencils and
Float64. Here `scaling="local"` divides kernel distances by each stencil's
radius; RBFLAB includes the inverse-square factor for the physical Laplacian.
See [local stencil scaling](../theory/conditioning.md#local-stencil-scaling).
Backward Euler starts BDF2. To animate the computed solution:

```python
from rbflab import viz
viz.animate_scalar(trajectory, "flower.gif", domain=domain)
```

The display grid does not enter the PDE discretization. The recorded final
sampled off-node error at $t=0.5$ is about $1.1\times10^{-3}$; it contains both
space and time error. Reproduce the error report with
`python -m examples.heat_flower` from a source checkout.

## 2. Open up the spatial operator

At each target node $x_i$, local RBF interpolation determines Laplacian weights:

$$ (L\boldsymbol u)_i=\sum_{j\in S_i}w_{ij}^{\Delta}u_j\approx\Delta u(x_i). $$

Rows use overlapping stencils $S_i$. Assemble the reusable operator once:

```python
--8<-- "examples/tutorials/heat_matrices.py:rbf_fd_laplacian"
```

This function returns the full $N\times N$ sparse Laplacian and its operator
collection. `ops.lap @ values` applies it; `ops.lap.local(i)` exposes a row's local
weights, and `ops.lap.reconstruct_local(i)` rebuilds its local interpolation system.
See [one stencil](one-stencil.md) for the derivation.

The following matrix demonstration uses `unit_box_grid(6)`: 49 nodes, 25 interior
unknowns, PHS5, degree-two polynomials and 20-node stencils. Its parameters differ
from the flower recipe so it can run quickly and be checked against a simple exact mode:

$$u(x,y,t)=e^{-2\kappa\pi^2t}\sin(\pi x)\sin(\pi y),\quad f=g=0.$$

## 3. Eliminate prescribed boundary values

Separate interior indices $I$ and boundary indices $B$. The full field is
$(\boldsymbol u_I,\boldsymbol g_B)$, so the interior equations become

$$\dot{\boldsymbol u}_I=\kappa L_{II}\boldsymbol u_I
+\kappa L_{IB}\boldsymbol g_B(t)+\boldsymbol f_I(t).$$

The matrix slices are:

```python
interior, boundary = cloud.interior_indices, cloud.boundary_indices
lap_ii = matrix[interior, :][:, interior]
lap_ib = matrix[interior, :][:, boundary]
```

This is a fragment of the time marcher below; `matrix` is the Laplacian returned
by `rbf_fd_laplacian`. Boundary contributions are on the right-hand side. Dropping
$L_{IB}g_B$ is only valid when those prescribed values are zero.

## 4. Take implicit time steps

Backward Euler gives

$$
(I-\Delta t\,\kappa L_{II})\boldsymbol u_I^{n+1}
=\boldsymbol u_I^n+\Delta t(\kappa L_{IB}\boldsymbol g_B^{n+1}+\boldsymbol f_I^{n+1}).
$$

After one startup step, BDF2 gives

$$
(\tfrac32 I-\Delta t\,\kappa L_{II})\boldsymbol u_I^{n+1}
=2\boldsymbol u_I^n-\tfrac12\boldsymbol u_I^{n-1}
+\Delta t(\kappa L_{IB}\boldsymbol g_B^{n+1}+\boldsymbol f_I^{n+1}).
$$

The fixed left matrix is factored once per time formula. Changing boundary values
or forcing only changes the right-hand side; changing the time step would require
new factors. The actual implementation is:

```python
--8<-- "examples/tutorials/heat_matrices.py:march"
```

`exact` and `forcing` below provide the verification data. An application can
replace these callbacks with its own initial state, boundary values and source.
This loop is ordinary SciPy code acting on RBFLAB's spatial matrix.

## 5. Compare matrix and symbolic assembly

The example also declares the same square-domain problem through `model.evolution`:

??? example "Symbolic version of the same matrix verification problem"

    ```python
    --8<-- "examples/tutorials/heat_matrices.py:symbolic_solution"
    ```

The two RBF-FD routes use identical nodes, kernels, stencils and time steps.
Their agreement tests assembly and boundary handling; it is not an independent
check of spatial accuracy. Compare both against the known continuous solution.

```sh
python -m examples.tutorials.heat_matrices
python -m examples.tutorials.heat_matrices --scheme bdf2 --nonzero-boundary
python -m examples.tutorials.heat_matrices --study
python -m examples.tutorials.heat_matrices --figure outputs/heat_matrices.png
```

The default five backward-Euler steps use $\Delta t=0.01$ and reach $t=0.05$.
The nodal maximum error is about $0.03767$ for both routes. This deliberately
small test has substantial time error. `--nonzero-boundary` switches to
$u=e^{-t}(1+x+y)$, $f=-u$, and $g=u|_{\partial\Omega}$ to test the boundary term.

![RBF-FD nodes, sparse matrix, computed temperature and absolute nodal error](../assets/heat_matrices.png)

## 6. Separate the sources of error

`--study` uses `expm_multiply` to evolve the fixed semidiscrete system without the
BE/BDF2 time formula. First it compares that reference with the continuous solution
on progressively finer clouds; then it holds the cloud fixed and varies $\Delta t$.

| Measurement | What changes? | What it isolates |
|---|---|---|
| Semidiscrete versus exact PDE solution | Node count | Spatial error for this recipe |
| Time-stepped versus semidiscrete solution | Time step | Time-integration error |
| Matrix versus symbolic route | Assembly route only | Implementation agreement |

An implicit time formula cannot repair an unstable spatial operator. Check errors,
cloud quality and refinement together; see [time discretization](../theory/time-discretization.md)
and [error measures](../theory/errors.md).

## Complete matrix example and backend choices

??? example "Complete runnable matrix and symbolic comparison"

    ```python
    --8<-- "examples/tutorials/heat_matrices.py"
    ```

[Download the standalone matrix example](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/heat_matrices.py).
The plotting options require `rbflab[examples]`. For the flower, use
`python -m examples.heat_flower --backend cpp --animation flower.gif` or
`--backend torch` after [installing the backend](../INSTALL.md). C++/PyTorch change
local weight construction; the sparse time solves remain Float64 SciPy solves.
