# Divergence-free spaces and Stokes

For incompressible Stokes flow, approximate the velocity in a space whose basis functions already have zero divergence. Pressure remains a separate scalar field.

## Build a matrix-valued kernel

Let \(z=x-y\) and choose a scalar potential \(\phi(z)\) with the derivatives required by the construction. RBFLAB uses

$$
\boldsymbol K_{\mathrm{div}}(z)=\nabla\nabla^{\mathsf T}\phi(z)-\Delta\phi(z)I_d,
\qquad
(K_{\mathrm{div}})_{ij}=\partial_i\partial_j\phi-\delta_{ij}\Delta\phi.
$$

This is **Hessian minus Laplacian times identity**, with the library's signed scalar-potential convention. Its column divergence is

$$
\sum_{i=1}^d\partial_i(K_{\mathrm{div}})_{ij}
=\partial_j\Delta\phi-\partial_j\Delta\phi=0.
$$

Consequently the expansion

$$
\boldsymbol u_h(x)=\sum_j\boldsymbol K_{\mathrm{div}}(x-x_j)\boldsymbol a_j
+\boldsymbol p_{\mathrm{div}}(x),\qquad
\nabla\cdot\boldsymbol p_{\mathrm{div}}=0,
$$

is divergence-free wherever the required derivatives exist, up to arithmetic/evaluation error. The construction applies in 2D and 3D. For local reconstruction, each local field has this property; stitching patches by nearest-stencil selection does not establish global continuity or a distributionally divergence-free field across patch jumps.

## Couple velocity and pressure through momentum

The steady and unsteady momentum equations are

$$
-\mu\Delta\boldsymbol u+\nabla p=\boldsymbol f,
\qquad
\partial_t\boldsymbol u-\mu\Delta\boldsymbol u+\nabla p=\boldsymbol f.
$$

A `DivergenceFreeSpace` supplies velocity columns; a `PressureSpace` supplies scalar pressure columns. Schematically, the combined kernel is

$$
\boldsymbol\Psi(x,y)=
\begin{bmatrix}\boldsymbol K_{\mathrm{div}}(x-y)&0\\0&K_p(x,y)\end{bmatrix}.
$$

Componentwise momentum and velocity-boundary functionals act on this combined space. In a Hermite construction, applying them on both sides produces coupled velocity/pressure blocks. The pressure kernel can differ from the scalar potential used to form the velocity kernel.

The divergence constraint is built into this trial space; the symbolic momentum formulation does not add a redundant `divergence(U) = 0` equation. This differs from approximating every velocity component by independent scalar RBFs and imposing a separate continuity row. Analytical incompressibility alone is not a proof of stability or pressure accuracy for every cloud.

## Pressure and its constant mode

If the equations and boundary data only involve \(\nabla p\), then \(p\) and \(p+C\) are indistinguishable. A pressure gauge selects one representative, for example

$$
\int_\Omega p\,dx=0\quad\text{or}\quad p(x_*)=0.
$$

A pressure space modulo constants instead represents that equivalence directly. RBFLAB's divergence-free Stokes route uses this pressure-space interpretation and reports pressure gradients. Do not add an arbitrary pressure boundary condition to remove the constant mode; a gauge is not an extra physical boundary condition.

## Reconstruction is another functional

Let \(s_i=(\boldsymbol u_i,p_i)\) be a local coupled interpolant with matrix \(H_i\) and data \(d_i\). Direct pressure-gradient reconstruction uses the target
\(\tau_{k,z}s_i=\partial_kp_i(z)\). As in [local weights](local-weights.md),

$$
H_i^{\mathsf T}\widetilde w_{k,z}=q_{k,z},\qquad
\partial_kp_i(z)=w_{k,z}^{\mathsf T}d_i.
$$

These weights act on the complete local data vector. Coupling through the coefficient solve is expected even though the output functional selects pressure.

A momentum-based estimate is different:

$$
\nabla p\approx\boldsymbol f-\partial_t\boldsymbol u_h+\mu\Delta\boldsymbol u_h.
$$

The two reconstructions agree only to the extent that the reconstructed momentum equation holds at the evaluation point. Report the chosen reconstruction and distinguish velocity, pressure-gradient, and divergence errors. Higher derivatives may amplify approximation and rounding errors.

**Try it:** [spaces-based Stokes](../tutorials/stokes.md), [annular Stokes](../tutorials/annular-stokes.md).
**Background:** [Narcowich & Ward](references.md#narcowich-ward-1994) and [Wendland's global Stokes analysis](references.md#wendland-2009). The latter is not automatically a local LHI error theorem.
