# Global RBF collocation

Global collocation represents the entire solution by one expansion. For
\(\mathcal Lu=f\) in \(\Omega\) and \(\mathcal Bu=g\) on \(\partial\Omega\), define one data functional per row:

$$
\lambda_i v=\begin{cases}
(\mathcal Lv)(x_i),&i\in I,\\
(\mathcal Bv)(x_i),&i\in B,
\end{cases}
\qquad
 d_i=\begin{cases}f(x_i),&i\in I,\\g(x_i),&i\in B.\end{cases}
$$

For multiple boundary labels, use the appropriate \(\mathcal B_\ell\) at each row. The scalar implementation assigns a boundary equation to each boundary node; conflicting corner conditions require a deliberate choice.

## 1. Asymmetric collocation

Use ordinary kernel translates as the trial functions:

$$
u_h(x)=\sum_{j=1}^N a_jK(x,x_j)+\sum_{m=1}^Qb_mp_m(x).
$$

Applying \(\lambda_i\) yields

$$
A_{ij}=\lambda_i^xK(x,x_j),\qquad
Q_{im}=\lambda_i p_m,\qquad P_{jm}=p_m(x_j),
$$

$$
\begin{bmatrix}A&Q\\P^{\mathsf T}&0\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d\\0\end{bmatrix}.
$$

Rows contain PDE or boundary derivatives; columns remain value-centered kernel translates. This matrix is generally asymmetric even for a symmetric radial kernel.

For Dirichlet Poisson, \(\mathcal L=-\Delta\) and \(\mathcal B=I\): interior rows contain \(-\Delta_xK\), while boundary rows contain \(K\).

## 2. Symmetric Hermite collocation

Instead, put the data functionals into the trial basis:

$$
u_h(x)=\sum_{j=1}^N a_j\lambda_j^yK(x,y)+\sum_{m=1}^Qb_mp_m(x).
$$

Here each source functional includes evaluation at its own center. Applying the row functional gives

$$
G_{ij}=\lambda_i^x\lambda_j^yK(x,y),\qquad Q_{im}=\lambda_i p_m,
$$

$$
\begin{bmatrix}G&Q\\Q^{\mathsf T}&0\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d\\0\end{bmatrix}.
$$

For a sufficiently smooth symmetric kernel and the same ordered functionals on rows and columns, \(G=G^{\mathsf T}\) in exact arithmetic. The statement concerns matrix symmetry; it does not require a self-adjoint PDE operator and does not establish positive definiteness or invertibility for an arbitrary setup.

For Dirichlet Poisson, the kernel block is

$$
G=\begin{bmatrix}
[\Delta_x\Delta_yK]_{II}&[-\Delta_xK]_{IB}\\
[-\Delta_yK]_{BI}&[K]_{BB}
\end{bmatrix}.
$$

The two minus signs in the interior-interior block cancel. This block needs fourth-order derivatives of the scalar kernel; ordinary asymmetric Poisson collocation needs only second-order derivatives.

## What the global solve returns

Both schemes solve for **expansion coefficients** \((a,b)\). At query points \(Z\), evaluation gives

$$
u_h(Z)=E(Z)\begin{bmatrix}a\\b\end{bmatrix},\qquad
(\mathcal Du_h)(Z)=E_{\mathcal D}(Z)\begin{bmatrix}a\\b\end{bmatrix}.
$$

These evaluation matrices use the same trial basis as assembly. Solving the coefficient system and then treating \(a\) as sampled values is incorrect.

Global kernels usually produce dense systems. Compact support can introduce zeros, but this alone does not turn the current global assembler into a dedicated sparse solver.

## API connection

```python
asymmetric = rbf.GlobalCollocation(rbf.IMQ(2), scheme="asymmetric")
symmetric = rbf.GlobalCollocation(rbf.IMQ(2), scheme="symmetric")
system = symmetric.assemble(problem, cloud)
solution = system.solve()
values = solution.evaluate(query_points)
```

See the [runnable comparison](../tutorials/global-lhi.md). For a sparse method, continue to [local coefficient elimination](local-weights.md); LHI reuses the Hermite construction locally.

**Background:** [Fornberg & Flyer](references.md#fornberg-flyer-2015) and the [generalized Hermite framework](references.md#narcowich-ward-1994).
