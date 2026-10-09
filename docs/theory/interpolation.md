# Kernel interpolation

Before differentiating or solving a PDE, construct a function that matches sampled values. Given distinct centers \(X=\{x_j\}_{j=1}^N\) and data \(d_j\), use

$$
s(x)=\sum_{j=1}^N a_jK(x,x_j)+\sum_{m=1}^Q b_mp_m(x),\qquad p_m\in\Pi_q.
$$

The first term supplies radial kernel translates. The polynomial term makes low-degree polynomial reproduction explicit. Neither \(a_j\) nor \(b_m\) is a nodal value.

## Derive the augmented system

Impose \(s(x_j)=d_j\) and the side conditions
\(\sum_j a_jp_m(x_j)=0\). Define

$$
\Phi_{jk}=K(x_j,x_k),\qquad P_{jm}=p_m(x_j).
$$

Then

$$
\underbrace{\begin{bmatrix}\Phi&P\\P^{\mathsf T}&0\end{bmatrix}}_{H}
\underbrace{\begin{bmatrix}a\\b\end{bmatrix}}_c
=\begin{bmatrix}d\\0\end{bmatrix}.
$$

There are \(N+Q\) equations for \(N+Q\) coefficients. The lower equations constrain the kernel coefficients; they are not extra observations of \(u\). Without polynomial augmentation, omit the polynomial blocks.

If \(d_j=p(x_j)\) for \(p\in\Pi_q\), the choice \(a=0\) and the coefficients of \(p\) solves the system. When \(H\) is nonsingular, uniqueness therefore gives \(s=p\): **polynomial reproduction**.

For conditionally positive definite kernels, the kernel's required polynomial space and polynomial unisolvency matter. Unisolvency means \(P\) has full column rank: the only polynomial in \(\Pi_q\) vanishing at all centers is zero. Having \(N\ge Q\) is necessary, but not sufficient. A symmetric augmented matrix can be indefinite even when the kernel block is positive definite.

## Kernel parameters in RBFLAB

| Kernel | Convention | Main choices |
|---|---|---|
| IMQ | \(\phi(r)=(1+c r^2)^{-1/2}\) | \(c>0\); equivalent to shape \(\varepsilon=\sqrt c\) |
| Odd PHS | \(\phi(r)=(-1)^{(m+1)/2}r^m\) | Power \(m\), polynomial degree \(q\); PHS5 is \(-r^5\) |
| Hybrid | Weighted sum of constituent kernels | Every component parameter and mixing coefficient |
| Wendland | A validated compact-support radial function | Smoothness, dimension, support radius |

Do not identify the PHS power with an automatic convergence order. Accuracy also depends on polynomial degree, differentiation order, smoothness, cloud geometry, and stability. Kernel support radius, stencil radius, and point spacing are different quantities.

## Differentiate the interpolant

For a linear differential operator \(\mathcal D\), differentiate the basis functions:

$$
(\mathcal Ds)(\xi)=\sum_j a_j(\mathcal D_xK)(\xi,x_j)
+\sum_m b_m(\mathcal Dp_m)(\xi).
$$

Coincident centers require the correct derivative limits at \(r=0\). An arbitrary replacement \(r\mapsto10^{-k}\) changes the operator. Sufficient derivative regularity is especially important in Hermite systems, which differentiate both kernel arguments.

The next two routes are [global PDE collocation](global.md) and [eliminating coefficients to obtain local weights](local-weights.md).

**Try it:** [interpolation and derivatives](../tutorials/interpolation.md).
**Background:** [Fornberg & Flyer](references.md#fornberg-flyer-2015), [Flyer et al.](references.md#flyer-2016), and [Wendland's compact kernels](references.md#wendland-1995).
