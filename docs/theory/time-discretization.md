# Semidiscrete systems and time stepping

After spatial assembly, write a linear problem as

$$
M\dot U+A U=b(t).
$$

Here \(U\) is the vector of **global unknowns**. \(M\) depends on the method. For a direct RBF-FD nodal heat solve, \(M=I\) on interior nodes. A global coefficient formulation or LHI evolution can have a nonidentity, possibly singular, mass matrix. Do not infer \(M=I\) merely from the symbolic PDE.

Backward Euler gives

$$
(M+\Delta t A)U^{n+1}=M U^n+\Delta t\,b(t_{n+1}).
$$

BDF2 gives

$$
(\tfrac32 M+\Delta t A)U^{n+1}
=2M U^n-\tfrac12 M U^{n-1}+\Delta t\,b(t_{n+1}).
$$

A backward Euler startup step supplies \(U^1\). The [heat tutorial](../tutorials/heat-equation.md) shows the \(M=I\), \(A=-\kappa L_{II}\) case, including the boundary term in \(b(t)\).

If the coefficients and time step are fixed, factor each distinct left matrix once and reuse it. A time-implicit formula does not ensure that the spatial discretization has no growing modes: inspect the relevant generalized operator or a measured energy response. See [conditioning](conditioning.md).

## Reading and mathematical context

[Hairer & Wanner (1996)](references.md#hairer-wanner-1996), chapters V–VI,
covers multistep stability and differential-algebraic systems.
The spatial RBF context is surveyed in
[Fornberg & Flyer (2015)](references.md#fornberg-flyer-2015).
Temporal order and spatial convergence should be measured separately.
