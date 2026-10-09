# Conditioning, scaling, and precision

A local interpolation solve, its weight solve, and the global PDE solve have different conditioning. Diagnose each stage separately.

## Local solve sensitivity

For the augmented weight system \(H_i^{\mathsf T}\widetilde w_i=q_i\), write the computed residual as

$$
r_i=H_i^{\mathsf T}\widehat{\widetilde w}_i-q_i,\qquad
\rho_i=\frac{\|r_i\|}{\|H_i^{\mathsf T}\|\,\|\widehat{\widetilde w}_i\|+\|q_i\|}.
$$

This normalized residual measures how well the represented equation was solved. With \(H_i\) fixed and nonzero \(q_i\),

$$
\frac{\|\widehat{\widetilde w}_i-\widetilde w_i\|}{\|\widetilde w_i\|}
\leq\kappa(H_i^{\mathsf T})\frac{\|r_i\|}{\|q_i\|}.
$$

A tiny residual can coexist with inaccurate weights when the system is ill-conditioned. Matrix-entry errors, kernel evaluation errors, and weight rounding introduce further perturbations. Norms must be consistent; do not compare a 2-norm condition estimate and an infinity-norm estimate as identical quantities.

## Local stencil scaling

The examples often choose:

```python
stencil_policy = rbf.StencilPolicy(scaling="local")
method = rbf.RBFFD(rbf.PHS(5), stencil_size=35, polynomial_degree=3,
                  stencil_policy=stencil_policy)
```

Think of this as **using the size of each neighborhood as its distance unit**.
A separation of $0.006$ in a stencil of radius $0.02$ becomes $0.3$ when evaluating
the kernel. The physical nodes, domain, boundary data and requested PDE stay in
their original coordinates. RBFLAB includes the derivative conversion factors;
you do not rescale the returned weights yourself.

### Which radius and which default?

For source centers $X_i=\{x_{i1},\ldots,x_{in_i}\}$, the current implementation uses

$$
h_i=\max_j\|x_{ij}-x_{i1}\|_2,\qquad
\widehat x=\frac{x-x_{i1}}{h_i},\qquad
\widehat r=\frac{\|x-y\|_2}{h_i}.
$$

The anchor is the **first source center**. For a nearest-neighbor stencil evaluated
at one of its own nodes, this is normally the target node. For an off-node target,
it need not be; the kernel radius is not generally the diagnostic radius measured
from that target. A zero-radius source set uses a fallback scale of one, though a
useful derivative stencil needs distinct nodes and sufficient polynomial rank.

| Setting | Kernel evaluated on a stencil |
|---|---|
| `StencilPolicy()` or `scaling="physical"` (default) | $\phi(\|x-y\|_2)$ |
| `StencilPolicy(scaling="local")` | $\phi(\|x-y\|_2/h_i)$ |

Local scaling alone keeps `selection="nearest"`: it neither grows a 35-node
stencil nor enables geometric screening. `selection="quality"` is a separate
option. It also leaves arithmetic precision unchanged. See the
[`StencilPolicy` API](../api/precision-backends.md#rbflab.StencilPolicy).

### The weights still differentiate physical coordinates

For a multi-index $\alpha$ of total order $s=|\alpha|$, the chain rule gives

$$
\partial_x^\alpha=h_i^{-s}\partial_{\widehat x}^\alpha.
$$

Thus a first derivative contributes $h_i^{-1}$ and a Laplacian $h_i^{-2}$.
RBFLAB applies these factors to each kernel derivative during assembly, including
source-side derivatives in Hermite systems. For example,

$$
(aI+b\partial_x+\nu\Delta_x)\phi(\widehat r)
=a\phi(\widehat r)+\frac{b}{h_i}\partial_{\widehat x}\phi(\widehat r)
+\frac{\nu}{h_i^2}\Delta_{\widehat x}\phi(\widehat r).
$$

Each term has its own factor; multiplying the whole operator by $h_i^{-2}$ would
be incorrect. The resulting row still approximates the original physical PDE.
For a Laplacian row, a useful consistency check is
$\sum_jw_{ij}^{\Delta}x_j^2\approx2$ when the polynomial degree is at least two.

### PHS, IMQ and hybrid parameters

**Pure PHS.** For the signed odd-power kernel used by `PHS(m)`,
$\phi(r/h_i)=h_i^{-m}\phi(r)$. This is a uniform kernel factor on that stencil.
For standard nodal RBF-FD with the same polynomial space, the physical weights
are unchanged in exact arithmetic. The matrix representation and floating-point
roundoff can change. This is why the PHS examples can normalize kernel distances
without introducing a shape parameter or a different approximation order.

**IMQ and Gaussian.** Their parameter `c` multiplies squared distance. With
`IMQ(gamma)` and local scaling, the physical kernel is

$$
\left(1+\gamma\frac{r^2}{h_i^2}\right)^{-1/2}
=\left(1+c_i r^2\right)^{-1/2},\qquad c_i=\frac{\gamma}{h_i^2}.
$$

`gamma` is now dimensionless. Holding it fixed as the cloud is refined is a
specific shape-parameter policy. Switching from physical to local scaling while
keeping the same numeric `c` generally changes the kernel and its weights.
To reproduce a fixed physical parameter $c$ on one stencil, use
$\gamma_i=c h_i^2$; varying stencil radii require varying dimensionless parameters.
`StencilPolicy(scaling="local")` does not perform that parameter adjustment.

**Hybrids and compact support.** Normalizing the argument of an IMQ + PHS sum
changes its two terms differently; a fixed mixing coefficient does not in general
preserve the physical hybrid kernel. A dimensionless support radius $\rho$ becomes
$h_i\rho$ in physical coordinates. Keep the scaling policy explicit when comparing
kernels or reusing a numerical recipe.

### What scaling can and cannot improve

Normalization makes kernel distances comparable between small and large
neighborhoods and can improve their numerical representation. It does not
promise a smaller condition number, a smaller PDE error, or stable time evolution.
Check polynomial reproduction, solution errors and the assembled operator as well.

The polynomial basis is already shifted and scaled internally under **both**
policies. Its normalization uses the largest absolute coordinate displacement
from the first source center, rather than the Euclidean kernel radius above;
both preserve the same polynomial space and include physical derivative factors.

## Coordinate and functional scaling

Coordinate normalization is distinct from **algebraic equilibration**: multiplying
matrix rows and columns by factors to balance their sizes. Hermite rows mix
values, first derivatives and second derivatives, so their units differ.
Consistent row/column scaling may materially change the reported condition number
while leaving the exact approximation unchanged. Always state which matrix was
measured: physical-kernel, locally normalized, or algebraically equilibrated.
The `scaled_matrix` returned by `reconstruct_local(i)` refers to this algebraic
equilibration, not a request to switch `StencilPolicy.scaling`.

Polynomial degree, node separation, and geometric rank also matter. Adding more
almost-collinear points does not repair missing two-dimensional polynomial information.

## Precision is a pipeline

$$
\text{coordinates/data}\longrightarrow\text{kernel derivatives}
\longrightarrow H_i\longrightarrow w_i
\longrightarrow A_h,b_h\longrightarrow U.
$$

MPFR local solves cannot restore coordinate digits absent from Float64 inputs, and casting high-precision weights to Float64 loses digits before sparse assembly. Record coordinate precision, local arithmetic, stored-weight precision, and global solver precision separately.

SVD truncation replaces inversion of small singular values with a regularized solve. It can reduce sensitivity, but may weaken exact reproduction; report the cutoff and reproduction defects. Optional condition estimation can be disabled for production runs after a recipe has been checked.

## Conditioning is not dynamical stability

For \(M\dot U+A_hU=0\), the evolution generator is \(-M^{-1}A_h\) when \(M\) is invertible. A well-conditioned local system does not rule out growing global modes. Eigenvalues diagnose asymptotic behavior; nonnormality can also create transient amplification. For singular \(M\), use the constrained/generalized system rather than forming an inverse.

**Continue:** [errors and residuals](errors.md), [backend choices](../api/precision-backends.md).
**Background:** [RBF-QR](references.md#fornberg-2011) and [RBF-RA](references.md#wright-fornberg-2017) change the numerical representation. Selecting MPFR or SVD is not an implementation of either algorithm.
