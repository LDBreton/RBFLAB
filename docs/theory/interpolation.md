# RBF interpolation and kernel conventions

A scalar RBF interpolant at centers \(x_j\) is

$$
s(x)=\sum_{j=1}^{N}a_j\phi(\lVert x-x_j\rVert)+\sum_{m=1}^{Q}b_m p_m(x).
$$

For data \(d_i=s(x_i)\), coefficients solve

$$
\begin{bmatrix}\Phi&P\\P^{\mathsf T}&0\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d\\0\end{bmatrix}.
$$

The lower row is the polynomial side condition \(P^{\mathsf T}a=0\). The coefficients \(a,b\) are **not** nodal PDE unknowns. In local RBF-FD, this augmented matrix is used to find weights; in global interpolation, it finds the expansion coefficients.

RBFLAB's signed PHS convention for odd order \(m\) is \(\phi(r)=(-1)^{(m+1)/2}r^m\); thus PHS5 is \(-r^5\). The IMQ kernel used in these examples is \((1+c r^2)^{-1/2}\). Parameter \(c\), stencil radius, and polynomial degree must be recorded with accuracy results. Kernel derivatives at \(r=0\) are handled by the kernel's origin limits; replacing \(r=0\) with an arbitrary small number changes the mathematics.

The [interpolation tutorial](../tutorials/interpolation.md) compares IMQ and polynomially augmented PHS. See [conditioning](conditioning.md) before changing kernel shape or precision.
