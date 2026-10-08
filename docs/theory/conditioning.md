# Conditioning, scaling, precision, and solvers

A small residual \(Gz-r\) means a local linear system was solved accurately **for that matrix**. It does not imply small weight error when \(G\) is ill-conditioned. For a rough scale, perturbations can be amplified by the condition number \(\kappa(G)\). Stencil geometry, polynomial degree, kernel shape, and dimensional scaling all affect it. A cloud that is nearly collinear may be unusable even with many points.

Scale coordinates by a local stencil radius when comparing a shape parameter across refinements; record the resulting dimensionless parameter. A flat IMQ kernel can improve approximation while making the local basis poorly conditioned. PHS plus polynomials avoids a tunable flatness parameter, but its degree and stencil size still govern accuracy and stability.

RBFLAB exposes local solver/precision choices and optional condition estimation. Condition estimates are useful for diagnosis but can be expensive at every stencil. Extended arithmetic can improve the **local weights**; it does not make a later Float64 sparse solve extended precision. Record both stages. SVD may regularize an ill-conditioned local solve but changes the effective approximation when singular values are discarded.

For evolution, also inspect the assembled operator: a local matrix can be well solved while the global semidiscrete dynamics contain growing modes. See [RBF-FD weights](rbf-fd.md), [time stepping](time-discretization.md), and the [precision/backend API](../api/precision-backends.md).

## Reading and mathematical context

[Fornberg, Larsson & Flyer (2011)](references.md#fornberg-2011) uses a stable
basis for Gaussian RBFs, illustrating the difference between approximation
quality and the conditioning of its direct representation.
[Wright & Fornberg (2017)](references.md#wright-fornberg-2017) describes RBF-RA,
including Hermite weights. Neither method is implied by selecting MPFR or SVD
in RBFLAB; they require different numerical algorithms.
