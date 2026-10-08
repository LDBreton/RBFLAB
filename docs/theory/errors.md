# Error and residual measures

For known exact values at test points \(x_k\), the **sampled maximum error** is
\(E_\infty=\max_k|u_h(x_k)-u(x_k)|\), and the **sampled RMS error** is

$$
E_{\mathrm{RMS}}=\sqrt{\frac1N\sum_{k=1}^N|u_h(x_k)-u(x_k)|^2}.
$$

Neither is a continuous \(L^2\) norm without a suitable quadrature rule.

The **algebraic residual** \(r=A U-b\) checks whether an assembled linear system was solved. The **local weight residual** checks a small stencil equation \(G^{\mathsf T}w=r_{\mathrm{local}}\). A small value of either can coexist with solution error because discretization is approximate. In a manufactured-solution test, inserting exact nodal values into \(A U-b\) probes consistency of the full discretization.

An **off-node PDE residual** evaluates the PDE operator on a reconstructed field at points not used as collocation rows. It depends on how that field is reconstructed; a nearest-stencil choice can create jumps. Do not interpret it as a norm of the nodal sparse-system residual. For Stokes, report velocity, pressure or pressure-gradient errors, and divergence separately, with gauge and sample locations stated.

The [heat tutorial](../tutorials/heat-equation.md) isolates spatial and time errors with a semidiscrete reference. The [global/LHI tutorial](../tutorials/global-lhi.md) distinguishes nodal and off-node evaluations.
