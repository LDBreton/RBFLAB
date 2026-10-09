# From a local approximation to weights

The executable [functional approximation tutorial](../tutorials/local-approximation.md)
maps these trial, data and target functionals directly to the public API.

This is the algebra shared by local differentiation methods. It explains **why a transpose solve appears**, and why local expansion coefficients disappear from the global PDE solve.

## 1. Match the local data

On stencil \(i\), take \(n_i\) data functionals \(\lambda_1,\ldots,\lambda_{n_i}\), trial functions \(\psi_1,\ldots,\psi_{n_i}\), and \(Q\) polynomials:

$$
s_i(x)=\sum_{j=1}^{n_i}a_j\psi_j(x)+\sum_{m=1}^Qb_mp_m(x).
$$

For ordinary RBF-FD, \(\lambda_j=\delta_{x_j}\) and \(\psi_j(x)=K(x,x_j)\). For Hermite interpolation, \(\psi_j(x)=\lambda_j^yK(x,y)\), and the data may be values or derivatives.

Interpolation plus side conditions gives

$$
H_i c_i=\widetilde d_i,\qquad
c_i=\begin{bmatrix}a\\b\end{bmatrix},\qquad
\widetilde d_i=\begin{bmatrix}d_i\\0_Q\end{bmatrix},
\qquad H_i\in\mathbb R^{(n_i+Q)\times(n_i+Q)}.
$$

For the ordinary or symmetric Hermite constructions,

$$
H_i=\begin{bmatrix}G_i&P_i\\P_i^{\mathsf T}&0\end{bmatrix},\qquad
(G_i)_{kj}=\lambda_k\psi_j,\quad (P_i)_{km}=\lambda_kp_m.
$$

When trial and data functionals differ, write

$$\psi_j=m_j^yK(\cdot,z_j),\quad
G_{ij}=\ell_i^xm_j^yK(x_i,z_j),\quad
(P_\ell)_{ik}=\ell_ip_k,\quad(P_m)_{jk}=m_jp_k.$$

The explicit representer construction in `LocalApproximation` uses

$$H=\begin{bmatrix}G&P_\ell\\P_m^T&0\end{bmatrix}.$$

The lower block comes from the **trial** functionals. Replacing it silently by
$P_\ell^T$ would change the method. A symmetric kernel and matching point/functionals
give a symmetric Hermite block; source-transformed trial functions with nodal
data generally do not. Both polynomial functional blocks must have the required
rank. For example, Laplacian-only trial functionals annihilate constants, so they
cannot support an unconstrained constant polynomial tail in this formulation.

Assume this system is nonsingular. The derivation below also works with other invertible interpolation systems that are not symmetric.

## 2. Apply the desired operator

Let \(\tau_i v=(\mathcal Dv)(\xi_i)\) be the target functional. Apply it to every trial function:

$$
q_i=\begin{bmatrix}
\tau_i\psi_1\\\vdots\\\tau_i\psi_{n_i}\\
\tau_i p_1\\\vdots\\\tau_i p_Q
\end{bmatrix},\qquad
\tau_i s_i=q_i^{\mathsf T}c_i.
$$

For example, \(\tau_i=\delta_{\xi_i}\Delta\) constructs Laplacian weights, while \(\tau_i=\delta_{\xi_i}\) constructs interpolation weights. The data functionals \(\lambda_j\) describe what is supplied; \(\tau_i\) describes what is requested.

## 3. Eliminate the expansion coefficients

Substitute the interpolation solution into the evaluation:

$$
\begin{aligned}
\tau_i s_i
&=q_i^{\mathsf T}H_i^{-1}\widetilde d_i\\
&=(H_i^{-\mathsf T}q_i)^{\mathsf T}\widetilde d_i\\
&=\begin{bmatrix}w_i\\\eta_i\end{bmatrix}^{\mathsf T}
  \begin{bmatrix}d_i\\0\end{bmatrix}
=w_i^{\mathsf T}d_i.
\end{aligned}
$$

Therefore **compute weights with**

$$
\boxed{H_i^{\mathsf T}\begin{bmatrix}w_i\\\eta_i\end{bmatrix}=q_i},
\qquad
\boxed{(\mathcal Du)(\xi_i)\approx w_i^{\mathsf T}d_i}.
$$

The equality is exact for the local interpolant \(s_i\); replacing \(u\) by \(s_i\) is the approximation. The \(Q\) auxiliary entries \(\eta_i\) multiply zero side-condition data, so only the first \(n_i\) entries are applied to observations. Do not discard these auxiliary unknowns while solving the augmented system.

**No inverse needs to be formed.** Factor \(H_i\) and solve its transposed system. Reuse the factorization for several target operators. For fixed geometry, kernels, parameters, and operators, the weights do not depend on the data values.

## 4. Read the polynomial equations

The lower block of the transpose solve is

$$
P_i^{\mathsf T}w_i=
\begin{bmatrix}\tau_i p_1&\cdots&\tau_i p_Q\end{bmatrix}^{\mathsf T}.
$$

Thus \(\sum_j w_{ij}\lambda_jp=\tau_i p\) for every \(p\in\Pi_q\). For value data and a 2D Laplacian, this includes

$$
\sum_jw_{ij}=0,\quad
\sum_jw_{ij}x_{j,1}=\sum_jw_{ij}x_{j,2}=0,\quad
\sum_jw_{ij}(x_{j,1}^2+x_{j,2}^2)=4.
$$

These are useful consistency checks; they do not prove stability of the assembled PDE operator.

## A concrete one-dimensional check

On \((-h,0,h)\), requiring exact second derivatives of \(1,x,x^2\) gives

$$
\begin{aligned}
w_-+w_0+w_+&=0,\\
-hw_-+hw_+&=0,\\
h^2w_-+h^2w_+&=2.
\end{aligned}
\qquad\Longrightarrow\qquad
w=\frac1{h^2}\begin{bmatrix}1&-2&1\end{bmatrix}^{\mathsf T}.
$$

Here polynomial reproduction alone determines the weights. This is an algebraic illustration; RBFLAB's documented cloud API works in 2D and 3D. On larger irregular stencils the kernel equations supply the additional information.

## Where the methods separate

| Method | Local data \(d_i\) | What is assembled globally |
|---|---|---|
| [RBF-FD](rbf-fd.md) | Nodal values \(U_j\) | All differentiation weights become sparse columns |
| [LHI](lhi.md) | Unknown values, known boundary data, known PDE data | Only value weights become unknown columns; known contributions move to the RHS |

In the discrete-operator API, `op.local(i)` returns the weight row and `op.reconstruct_local(i)` rebuilds its generating system. Follow the [one-stencil tutorial](../tutorials/one-stencil.md) to inspect the actual arrays.
