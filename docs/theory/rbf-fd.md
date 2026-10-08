# From interpolation to RBF-FD weights

<figure class="stencil-figure">
<a href="../../assets/rbf_fd_stencil.svg" aria-label="Open the stencil diagram at full size"><img src="../../assets/rbf_fd_stencil.svg" alt="An irregular 121-node cloud with 20 selected neighbors around a target, followed by their signed Laplacian weights in a sparse matrix row"></a>
<figcaption>One computed RBF-FD row on an irregular cloud. Blue nodes form the 20-point stencil; the orange star is the target. The dashed circle marks its radius R, not compact kernel support. The right panel shows RÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â² times the Laplacian weights in global node order; unselected columns are zero. The figure uses PHS5, degree-two polynomials, and a seeded 121-node cloud; the runnable one-stencil tutorial defaults to 36 nodes.</figcaption>
</figure>

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

The illustration is generated with `python -m examples.make_stencil_figures`
from a source checkout. It checks that the computed row annihilates constants
and maps \(x^2+y^2\) to 4.

## Reading and mathematical context

For the connection to finite differences, see section 5 of
[Fornberg & Flyer (2015)](references.md#fornberg-flyer-2015).
[Flyer et al. (2016)](references.md#flyer-2016) studies polynomial reproduction
and accuracy; [Bayona et al. (2017)](references.md#bayona-2017) extends the
discussion to elliptic PDEs. These papers motivate the polynomial and stencil
choices; a particular cloud still needs its own consistency and stability checks.
