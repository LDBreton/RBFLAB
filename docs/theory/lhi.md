# Local Hermite interpolation (LHI)

LHI builds a local approximation using three kinds of information: solution values, boundary equations, and the PDE itself at nearby points. The unknown solution values are then coupled through a sparse global system.

This chapter derives the stationary **scalar** construction implemented by `LHI`. Vector Stokes uses the same functional principle with matrix-valued kernels; its pressure reconstruction is a separate operator.

## 1. Choose centers by their role

For an interior target \(\xi_i=x_i\), choose ordered sets

- \(S_i\): solution centers, carrying unknown values \(U_{S_i}\);
- \(B_i\): boundary centers, carrying known data \(g_{B_i}\);
- \(F_i\): PDE centers, carrying known forcing \(f_{F_i}\).

We use \(F_i\) for the PDE-center set so that \(P_i\) can consistently denote the polynomial matrix. A location can belong to both \(S_i\) and \(F_i\): a value and a PDE derivative are different functionals at that location.

<figure class="stencil-figure">
<a href="../../assets/lhi_centers.svg"><img src="../../assets/lhi_centers.svg" alt="Solution circles, PDE-data triangles, and boundary squares in an LHI neighborhood; the target has no PDE-data triangle"></a>
<figcaption>Roles in a local Hermite neighborhood. Marker counts are illustrative, not a prescribed stencil size. The target is a solution center and is excluded from the PDE-data centers.</figcaption>
</figure>

Stack the functionals and data in the same order:

$$
\lambda=\left(\{\delta_{x_j}\}_{j\in S_i},\;
\{\delta_{x_j}\mathcal B_j\}_{j\in B_i},\;
\{\delta_{x_j}\mathcal L\}_{j\in F_i}\right),
\qquad
 d_i=\begin{bmatrix}U_{S_i}\\g_{B_i}\\f_{F_i}\end{bmatrix}.
$$

The local constraint count is \(n_i=|S_i|+|B_i|+|F_i|\), not the number of distinct physical locations. Polynomial augmentation adds another \(Q\) rows and columns. In the scalar API, `stencil_size` selects value/boundary candidates; additional PDE functionals can make the local matrix larger.

## 2. Build the Hermite approximation

Use the data functionals on the source side:

$$
s_i(x)=\sum_{j=1}^{n_i}a_j\lambda_j^yK(x,y)
       +\sum_{m=1}^Qb_mp_m(x).
$$

Enforce \(\lambda_k s_i=(d_i)_k\) and polynomial side conditions:

$$
H_i\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d_i\\0\end{bmatrix},\qquad
H_i=\begin{bmatrix}G_i&P_i\\P_i^{\mathsf T}&0\end{bmatrix},
$$

$$
(G_i)_{kj}=\lambda_k^x\lambda_j^yK(x,y),\qquad
(P_i)_{km}=\lambda_kp_m.
$$

In particular, boundary/PDE rows of \(P_i\) contain **boundary/PDE operators applied to the polynomials**, not merely polynomial values. This is how augmentation stays consistent with Hermite data.

The kernel block has the structure

$$
G_i=\begin{bmatrix}
K_{SS}&(\mathcal B_yK)_{SB}&(\mathcal L_yK)_{SF}\\
(\mathcal B_xK)_{BS}&(\mathcal B_x\mathcal B_yK)_{BB}&(\mathcal B_x\mathcal L_yK)_{BF}\\
(\mathcal L_xK)_{FS}&(\mathcal L_x\mathcal B_yK)_{FB}&(\mathcal L_x\mathcal L_yK)_{FF}
\end{bmatrix}.
$$

Under the symmetry and regularity assumptions of [global Hermite collocation](global.md), \(G_i\) and \(H_i\) are symmetric in exact arithmetic. Repeated identical functionals or insufficient polynomial rank can still make them singular.

## 3. Evaluate the PDE at the target

Set \(\tau_i=\delta_{\xi_i}\mathcal L\). Form

$$
q_i=\begin{bmatrix}
[\tau_i^x\lambda_j^yK]_{j=1}^{n_i}\\
[\tau_i p_m]_{m=1}^Q
\end{bmatrix},\qquad
H_i^{\mathsf T}\begin{bmatrix}w_i\\\eta_i\end{bmatrix}=q_i.
$$

By [eliminating the coefficients](local-weights.md),

$$
(\mathcal Ls_i)(\xi_i)
=(w_i^S)^{\mathsf T}U_{S_i}
 +(w_i^B)^{\mathsf T}g_{B_i}
 +(w_i^F)^{\mathsf T}f_{F_i}.
$$

Enforce \((\mathcal Ls_i)(\xi_i)=f(\xi_i)\). Only the value data are global unknowns:

$$
\boxed{
(w_i^S)^{\mathsf T}U_{S_i}
=f(\xi_i)-(w_i^B)^{\mathsf T}g_{B_i}
          -(w_i^F)^{\mathsf T}f_{F_i}.}
$$

For example, a patch containing three solution values, one boundary datum, and two PDE data produces **three sparse coefficients** in this global row, although it used six local functionals (plus polynomials).

## 4. Assemble the global system

Scatter the three weight groups into matrices \(W_S,W_B,W_F\), preserving their global index maps. With targets covering interior solution centers,

$$
\underbrace{W_S}_{A_h}U_I
=\underbrace{f_I-W_Bg_B-W_Ff_I}_{b_h}.
$$

Here \(W_F\) has columns for all interior forcing samples, with zeros outside each patch's PDE set. \(W_B\) acts on sampled boundary **functional data**, which are not always boundary values.

Although every \(H_i\) is symmetric, \(A_h=W_S\) is generally nonsymmetric: its rows come from different local problems and retain only one group of weights.

## Why exclude the target from PDE centers?

If \(\delta_{\xi_i}\mathcal L\) were already a data functional, the interpolant would satisfy that equation by construction. In exact arithmetic its evaluation weights would simply select the same forcing datum. Assembling it again would give \(f(\xi_i)=f(\xi_i)\), not a new equation for the solution values.

RBFLAB therefore includes the target among solution centers and excludes it from PDE centers. This is an algebraic requirement of this collocation row, not a rule against using multiple distinct functionals at other locations.

## Reconstruct after solving

After obtaining \(U_I\), insert the solved values into every \(d_i\) and recover local coefficients. Alternatively, construct weights for another target functional \(\tau\):

$$
H_i^{\mathsf T}\widetilde w_i^{\tau}=q_i^{\tau},\qquad
\tau s_i=(w_i^{\tau})^{\mathsf T}d_i.
$$

Value, gradient, Laplacian, and pressure-gradient reconstructions have different \(q_i^\tau\). Reusing the PDE weights for a different output is not valid. The scalar `LHISolution.evaluate` selects the nearest interior stencil; neighboring reconstructions need not agree on patch interfaces. Off-node errors and PDE residuals must therefore be reported with the reconstruction policy.

**Try it:** [global/LHI comparison](../tutorials/global-lhi.md) and [annular Laplace](../tutorials/annulus.md).
**Read next:** [time-dependent systems](time-discretization.md), where PDE-center data can contain unknown time derivatives.
**Background:** [Stevens et al.](references.md#stevens-2009) and [Narcowich & Ward](references.md#narcowich-ward-1994).
