# Divergence-free kernels and pressure

From a sufficiently smooth scalar potential \(\phi\), RBFLAB forms the matrix kernel

$$
K_{ij}(z)=\partial_i\partial_j\phi(z)-\delta_{ij}\Delta\phi(z),\qquad z=x-y.
$$

The sign is **Hessian minus Laplacian times identity**. Its column divergence is

$$
\sum_i\partial_iK_{ij}
=\partial_j\Delta\phi-\partial_j\Delta\phi=0.
$$

A velocity assembled from its columns is therefore divergence-free at the function-space level, up to numerical evaluation errors. The same construction works in two and three dimensions when the underlying derivatives are available.

Steady Stokes momentum is \(-\mu\Delta\boldsymbol u+\nabla p=\boldsymbol f\). The unsteady problem adds \(\partial_t\boldsymbol u\). A `DivergenceFreeSpace` already represents incompressible velocity; its symbolic momentum formulation does not add a redundant divergence equation.

Pressure is determined up to a constant. The [Stokes tutorial](../tutorials/stokes.md) uses a pressure space modulo constants and reports \(\nabla p\); a displayed pressure value needs an explicit gauge or reference. Pressure-value, pressure-gradient, and velocity errors can differ substantially. See [equations and spaces](../api/equations-spaces.md).

## Reading and mathematical context

[Narcowich & Ward (1994)](references.md#narcowich-ward-1994) develops the
matrix-valued Hermite interpolation framework.
[Wendland (2009)](references.md#wendland-2009) applies divergence-free kernels
to Stokes collocation. Keep RBFLAB's signed scalar-potential convention in mind
when comparing formulas. That global analysis is background, not an automatic
error estimate for our local pressure reconstruction.
