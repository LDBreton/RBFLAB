# Notation and the model problem

We use one notation throughout the foundations. The scalar stationary model is

$$
\begin{aligned}
\mathcal L u(x)&=f(x), &&x\in\Omega\subset\mathbb R^d,\\
\mathcal B_\ell u(x)&=g_\ell(x), &&x\in\Gamma_\ell\subset\partial\Omega.
\end{aligned}
$$

The boundary pieces \(\Gamma_\ell\) carry labels in a `PointCloud`. Examples are
\(\mathcal L=-\kappa\Delta+\boldsymbol\beta\cdot\nabla+\sigma\) and

$$
\mathcal B_Du=u,\qquad
\mathcal B_Nu=\boldsymbol n\cdot\nabla u,\qquad
\mathcal B_Ru=a u+b\boldsymbol n\cdot\nabla u.
$$

Here \(\boldsymbol n\) is the outward unit normal, including on hole boundaries. The functions, coefficients, and data must have the regularity required by the derivatives used in the method.

## Coordinates, nodes, and functionals

| Symbol | Meaning |
|---|---|
| \(x,y\in\mathbb R^d\) | Evaluation and source coordinates |
| \(X=\{x_j\}_{j=1}^N\) | Global cloud; \(I\) and \(B\) index interior and boundary nodes |
| \(\xi_i\) | Target of local approximation \(i\); it need not be a source node |
| \(S_i\) | Ordered list of value-node indices in stencil \(i\) |
| \(K(x,y)=\phi(\|x-y\|)\) | Scalar radial kernel, with parameters left implicit |
| \(\Pi_q\) | Polynomials of total degree at most \(q\) |
| \(p_1,\ldots,p_Q\) | Basis of \(\Pi_q\), with \(Q=\binom{q+d}{d}\) |
| \(a,b\) | Kernel and polynomial expansion coefficients |
| \(U\) | Global numerical unknown vector; its meaning depends on the method |
| \(d_i\) | Local data vector, which may contain unknown values and prescribed data |

An operator returns a function. A **functional** returns one number. Evaluation is
\(\delta_\xi v=v(\xi)\), and a differential functional is

$$
\lambda v=(\mathcal Dv)(\xi),\qquad
\lambda^x K(x,y)=\left.\mathcal D_xK(x,y)\right|_{x=\xi}.
$$

The superscript identifies which kernel argument is differentiated. It is not a power. For a translation-invariant kernel,

$$
\partial_{y_k}K(x,y)=-\partial_{x_k}K(x,y),\qquad
\Delta_yK(x,y)=\Delta_xK(x,y).
$$

For example, \(\lambda^x\mu^yK\) means apply both functionals, evaluating each at its own center. A source normal derivative uses the normal at the **source** point. Source differentiation is not an integration-by-parts or weak-form operation.

## Matrix conventions

All data and coefficient vectors are columns. \(H^{-\mathsf T}=(H^{\mathsf T})^{-1}\). Inverses in derivations express an identity; implementations use linear solves.

We write \(H_i\) for a local augmented interpolation matrix, \(G_i\) for its kernel/Hermite block, \(A_h\) for a global PDE matrix, and \(D_h\) for a nodal differentiation matrix. This avoids using one letter for several different systems.

Next: [build the approximation space](interpolation.md).
