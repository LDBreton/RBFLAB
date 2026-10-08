# Global collocation and local Hermite interpolation

For \(\mathcal Lu=f\) inside \(\Omega\) and \(\mathcal Bu=g\) on its boundary, let \(F_i\) denote the PDE or boundary functional applied at row \(i\).

**Asymmetric global collocation** uses
\(u_h(x)=\sum_j a_jK(x,x_j)\), so
\(A_{ij}=F_i^xK(x_i,x_j)\).
**Symmetric Hermite collocation** uses source-applied basis functions
\(u_h(x)=\sum_j a_jF_j^yK(x,x_j)\), so
\(G_{ij}=F_i^xF_j^yK(x_i,x_j)\).
The source derivative has a sign for translation-invariant kernels because differentiating \(x-y\) with respect to \(y\) reverses direction. Both global unknown vectors are expansion coefficients, not sampled \(u(x_i)\).

LHI repeats a Hermite construction on small stencils. An interior target has three kinds of centers:

- solution centers \(S_i\), supplying unknown nodal values;
- boundary centers \(B_i\), supplying prescribed boundary functionals;
- PDE centers \(P_i\), supplying known forcing.

The target is in \(S_i\) and excluded from \(P_i\). If local weights are \((w_S,w_B,w_P)\), the assembled sparse row is

$$
w_S^{\mathsf T}u_{S_i}
=f(x_i)-w_B^{\mathsf T}g_{B_i}-w_P^{\mathsf T}f_{P_i}.
$$

Thus the global LHI solve uses interior solution values, while each local weight system contains more functionals. For time-dependent LHI, the resulting mass matrix need not be identity. An off-node value is reconstructed from a local stencil and should be assessed separately from nodal error.

The [global/LHI tutorial](../tutorials/global-lhi.md) builds the three methods. [Error measures](errors.md) distinguishes the relevant diagnostics.
