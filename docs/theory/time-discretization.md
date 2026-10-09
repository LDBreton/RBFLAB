# Spatial systems and time discretization

Consider fixed spatial and boundary operators with time-dependent data:

$$
\partial_tu+\mathcal Lu=f(x,t),\qquad \mathcal Bu=g(x,t),\qquad u(x,0)=u_0(x).
$$

After spatial assembly write

$$
\boxed{M\dot U+A_hU=b_h(t).}
$$

The meaning of \(U\) and the matrix \(M\) depend on the method. The symbolic equation alone does not imply \(M=I\).

## RBF-FD: interior nodal values

For \(u_t=\kappa\Delta u+f\) with Dirichlet boundary data and interior values \(U_I\),

$$
\dot U_I-\kappa D_{II}U_I=f_I+\kappa D_{IB}g_B.
$$

Thus \(M=I\), \(A_h=-\kappa D_{II}\). If boundary unknowns are retained, their algebraic boundary rows have zero mass rows instead. Derivative boundary conditions need their assembled equations; they cannot be substituted as Dirichlet values.

## Global collocation: expansion coefficients

With a fixed trial basis, write \(u_h(x,t)=\sum_jc_j(t)\psi_j(x)\), including any polynomial basis functions in the index. Interior rows satisfy

$$
\sum_j\psi_j(x_i)\dot c_j+
\sum_j(\mathcal L\psi_j)(x_i)c_j=f(x_i,t).
$$

The mass rows are **basis evaluations**, not identity rows. Boundary and polynomial side-condition equations are algebraic and have zero time-derivative rows. The resulting system may be differential-algebraic. Stationary Gram-matrix symmetry does not imply symmetry of this evolution system.

## LHI: why the mass is I minus PDE-data weights

Reuse the stationary spatial LHI construction from [the LHI chapter](lhi.md). At a PDE center the supplied spatial datum is now

$$
\mathcal Lu=f-\partial_tu.
$$

The target approximation is therefore

$$
(\mathcal Lu)_I\approx W_SU_I+W_Bg_B+W_F(f_I-\dot U_I).
$$

Substitute it into the target PDE and collect terms:

$$
\dot U_I+W_SU_I+W_Bg_B+W_Ff_I-W_F\dot U_I=f_I,
$$

$$
\boxed{(I-W_F)\dot U_I+W_SU_I=(I-W_F)f_I-W_Bg_B.}
$$

So \(M=I-W_F\), \(A_h=W_S\). RBFLAB calls the assembled PDE-data matrix `SL`, hence `mass = I - SL`. The forcing is also multiplied by \(M\). Replacing only the history term by an identity mass changes the method.

This derivation assumes fixed spatial weights and PDE centers drawn from the interior solution cloud. Other choices of centers, evolving geometry, or time-dependent trial spaces require additional mappings or terms.

## Backward Euler and BDF2

Let \(t_n=t_0+n\Delta t\). Backward Euler gives

$$
(M+\Delta t A_h)U^{n+1}=MU^n+\Delta t\,b_h(t_{n+1}).
$$

BDF2 replaces the time derivative by

$$
\dot U(t_{n+1})\approx\frac{3U^{n+1}-4U^n+U^{n-1}}{2\Delta t},
$$

and therefore

$$
(\tfrac32M+\Delta t A_h)U^{n+1}
=2MU^n-\tfrac12MU^{n-1}+\Delta t\,b_h(t_{n+1}).
$$

RBFLAB uses a backward Euler startup step. For a smooth, stable problem with consistent initial data, its one-step error is sufficient for second-order BDF2 convergence as \(\Delta t\to0\). Boundary and algebraic constraints must also be satisfied. Reuse each fixed left-hand matrix factorization; BE and BDF2 have different coefficients.

Crank–Nicolson averages the spatial/RHS terms at adjacent times; BDF2 uses a two-step backward derivative with new-time spatial terms. They are both formally second order under appropriate assumptions, but have different amplification and damping properties. An implicit time scheme does not remove the need to check the spatial operator for growing modes.

**Try it:** [heat from matrices](../tutorials/heat-equation.md) and [heat on a flower](../tutorials/flower-heat.md).
**Background:** [Hairer & Wanner](references.md#hairer-wanner-1996).
