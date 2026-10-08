# From interpolation to RBF-FD weights

At target \(x_i\), a differential functional \(\mathcal L\) is replaced by

$$
\mathcal L u(x_i)\approx\sum_{j\in S_i}w_{ij}u(x_j).
$$

For kernel translates and polynomial augmentation, enforce this identity on every basis function. With \(\Phi_{jk}=\phi(\lVert x_j-x_k\rVert)\) and \(P_{jm}=p_m(x_j)\), solve

$$
\begin{bmatrix}\Phi&P\\P^{\mathsf T}&0\end{bmatrix}^{\mathsf T}
\begin{bmatrix}w\\\lambda\end{bmatrix}
=
\begin{bmatrix}
\mathcal L_x\phi(\lVert x-x_j\rVert)|_{x=x_i}\\
\mathcal L p_m(x_i)
\end{bmatrix}.
$$

The upper-left block belongs to a **local interpolation system**. The first \(|S_i|\) entries of \(w\) become a row of the **assembled differentiation matrix**. Polynomial multipliers remain local; they are not PDE unknowns.

For a translation-invariant \(K(x,y)=\phi(x-y)\), \(\partial_{y_k}K=-\partial_{x_k}K\). This sign matters when constructing Hermite functionals at source points. `reconstruct_local(i)` exposes the augmented system and uses `matrix.T @ augmented_weights = rhs` as its check.

For degree-two polynomials, a Laplacian row should annihilate \(1\) and produce \(4\) on \(x^2+y^2\). That tests consistency. The local condition number and spectrum of the assembled diffusion matrix address different questions; a tiny local solve residual alone does not guarantee stable time evolution.

Continue with the [one-stencil calculation](../tutorials/one-stencil.md) and [heat matrices](../tutorials/heat-equation.md).
