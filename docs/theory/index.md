# Mathematical foundations

The central question is: **what do we approximate locally, and what do we solve for globally?**
Start with a function approximation, apply an operator to it, then eliminate local coefficients when a sparse method is desired. The chapters below follow that route.

## Reading route

1. [Notation and the model problem](notation.md): nodes, functionals, operators, and unknowns.
2. [Kernel interpolation](interpolation.md): the approximation space and polynomial side conditions.
3. [Global collocation](global.md): asymmetric and symmetric systems for one global expansion.
4. [From a local approximation to weights](local-weights.md): the derivation of the transpose solve.
5. [RBF finite differences](rbf-fd.md): weights on nodal values and sparse PDE assembly.
6. [Local Hermite interpolation](lhi.md): weights on values, boundary data, and PDE data.
7. [Divergence-free spaces and Stokes](divergence-free.md): vector kernels and pressure.
8. [Time discretization](time-discretization.md): mass matrices, boundary data, and BDF2.
9. [Conditioning and precision](conditioning.md), then [errors and residuals](errors.md): how to judge the computation.
10. [Papers and further reading](references.md): the mathematical literature behind these constructions.

## Keep these three matrices separate

| Object | What its rows mean | What the solve produces |
|---|---|---|
| Interpolation matrix \(H\) | Match values or Hermite data | Expansion coefficients \(c\) |
| Transposed local matrix \(H_i^{\mathsf T}\) | Reproduce a target functional on the local space | Local weights \(\widetilde w_i\) |
| Assembled PDE matrix \(A_h\) | Enforce equations across the domain | Global unknowns \(U\) |

For global collocation, \(U\) consists of expansion coefficients. For RBF-FD and scalar LHI, it consists of nodal values (after the relevant boundary elimination). A local matrix may be symmetric while the assembled PDE matrix is not.

!!! tip "A short route with code"
    Read [local weights](local-weights.md), work through [one stencil](../tutorials/one-stencil.md), then build the [heat equation from matrices](../tutorials/heat-equation.md). Return to [LHI](lhi.md) to see what changes when local data include derivatives.

## Work through the constructions in code

The [practical global lesson](../tutorials/global-collocation.md) builds PDE rows
from kernel evaluations. The [LHI lesson](../tutorials/lhi.md) follows the three
center groups into one sparse equation. For your own differential expression,
use [custom operator assembly](../tutorials/custom-assembly.md). These lessons
translate the notation above into concrete arrays and API calls.
