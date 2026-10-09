# RBF finite differences

RBF-FD converts local interpolation into differentiation formulas on nodal values. The interpolation coefficients are eliminated before the PDE solve.

<figure class="stencil-figure">
<a href="../../assets/rbf_fd_stencil.svg"><img src="../../assets/rbf_fd_stencil.svg" alt="Twenty neighbors around a target and their computed signed Laplacian weights in global column order"></a>
<figcaption>A computed 20-node Laplacian stencil in a seeded 121-node cloud, using PHS5 and degree-two polynomials. The radius R is the stencil radius, not kernel support. The right panel displays R² times the weights.</figcaption>
</figure>

## Ordinary value-based RBF-FD

At target \(\xi_i\), select nodes \(X_i=\{x_j:j\in S_i\}\) and fit

$$
s_i(x)=\sum_{j\in S_i}a_jK(x,x_j)+\sum_{m=1}^Qb_mp_m(x),
\qquad s_i(x_j)=U_j.
$$

With \((\Phi_i)_{jk}=K(x_j,x_k)\) and \((P_i)_{jm}=p_m(x_j)\), the [coefficient-elimination derivation](local-weights.md) gives

$$
\begin{bmatrix}\Phi_i&P_i\\P_i^{\mathsf T}&0\end{bmatrix}^{\mathsf T}
\begin{bmatrix}w_i^{\mathcal D}\\\eta_i\end{bmatrix}
=
\begin{bmatrix}
[(\mathcal D_xK)(\xi_i,x_j)]_{j\in S_i}\\
[(\mathcal Dp_m)(\xi_i)]_{m=1}^Q
\end{bmatrix}.
$$

Consequently,

$$
(\mathcal Du)(\xi_i)\approx\sum_{j\in S_i}w_{ij}^{\mathcal D}U_j.
$$

Each target has its own stencil and local coefficients. The global unknowns are the shared nodal values \(U_j\), not those coefficients.

## Assemble the sparse operator

Insert each weight into its source node's global column:

$$
(D_h)_{ij}=\begin{cases}w_{ij}^{\mathcal D},&j\in S_i,\\0,&j\notin S_i.\end{cases}
\qquad (\mathcal Du)(Z)\approx D_hU.
$$

If source and target clouds differ, \(D_h\) is rectangular. This is useful for staggered operators: pressure-to-velocity gradients and velocity-to-pressure divergence need not use the same points.

In the API, `ops.lap @ U` or `ops["lap"] @ U` applies an operator, `.matrix` exposes its sparse matrix, `.local(i)` exposes weights, and `.reconstruct_local(i)` reconstructs the generating local system.

## Enforce PDE and boundary equations

For a scalar PDE, assemble interior rows using \(\mathcal D=\mathcal L\), and boundary rows using \(\mathcal D=\mathcal B\). For Dirichlet data the boundary row simply fixes \(U_B=g_B\). For Neumann or Robin data, construct the corresponding normal-derivative or combined weights.

For \(-\kappa\Delta u=f\) with Dirichlet values, partition the Laplacian matrix into interior and boundary columns:

$$
-\kappa D_{II}U_I=f_I+\kappa D_{IB}g_B.
$$

This is the matrix that the PDE solver factors. It is different from every small interpolation matrix \(H_i\).

## What does “symmetric” mean here?

The ordinary value-interpolation matrix is already symmetric for a symmetric kernel. Nevertheless, the assembled differentiation matrix generally is not: different targets select different neighborhoods and weights.

The API's `RBFFD(scheme="symmetric")` selects a **trial basis inspired by symmetric global collocation**:

$$
\psi_j(x)=\lambda_j^yK(x,y),
$$

where \(\lambda_j\) is the PDE or boundary functional at source node \(j\). But the local data remain **values**, so the kernel block is \(C_{kj}=\psi_j(x_k)\), not \(\lambda_k^x\lambda_j^yK\). Its implemented augmented system is

$$
H_i=\begin{bmatrix}C_i&P_i\\P_i^{\mathsf T}&0\end{bmatrix},\qquad
(P_i)_{km}=p_m(x_k).
$$

The nodal polynomial side conditions retain polynomial modes even when the PDE source functionals annihilate them. Neither this local matrix nor the assembled PDE matrix is generally symmetric. The scheme name describes the source-transformed ansatz, not a matrix-symmetry guarantee.

`scheme="boundary_hermite"` instead uses value functionals in the interior and boundary functionals on the boundary for both data and trial construction. Its boundary degrees of freedom represent \(\mathcal Bu\), not always \(u\). [LHI](lhi.md) additionally includes nearby PDE data and eliminates their known contributions explicitly. The reusable `.operators(...)` interface currently supports the standard nodal scheme.

## Scaling and consistency

With \(\widehat x=(x-\xi_i)/R_i\), a pure derivative of order \(s\) obeys

$$
\partial_x^\alpha=R_i^{-s}\partial_{\widehat x}^\alpha,\qquad |\alpha|=s.
$$

A mixed-order operator must scale each term separately. After mapping back to physical coordinates, weights must reproduce the physical polynomial derivatives. The [conditioning chapter](conditioning.md) explains why coordinate scaling helps numerical representation without guaranteeing PDE stability.

**Work through:** [one stencil](../tutorials/one-stencil.md) → [heat matrices](../tutorials/heat-equation.md).
**Background:** [Fornberg & Flyer](references.md#fornberg-flyer-2015), [Flyer et al.](references.md#flyer-2016), [Bayona et al.](references.md#bayona-2017).
