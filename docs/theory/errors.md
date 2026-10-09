# Errors, residuals, and validation

An accurate linear solve is only one part of an accurate PDE solution. Use the notation from [the model problem](notation.md) to distinguish the following quantities.

## 1. Sampled solution error

At test points \(Z=\{z_k\}_{k=1}^{N_Z}\), define

$$
e_k=u_h(z_k)-u_{\mathrm{exact}}(z_k),\qquad
E_{\infty,Z}=\max_k|e_k|,\qquad
E_{\mathrm{RMS},Z}=\left(\frac1{N_Z}\sum_k|e_k|^2\right)^{1/2}.
$$

These are sampled errors. RMS is not a continuous \(L^2(\Omega)\) norm; that requires integration or quadrature weights. For vectors, state whether the pointwise norm is Euclidean or componentwise maximum. Use independent off-node samples as well as solution nodes.

## 2. Algebraic residual

For the assembled system,

$$
r_{\mathrm{alg}}=A_h\widehat U-b_h.
$$

This tests the numerical linear solve. A tiny residual says nothing by itself about the approximation of the continuous PDE.

For a method with nodal unknowns, insert the sampled exact solution \(U_*\) instead:

$$
\tau_h=A_hU_*-b_h,\qquad
A_h(\widehat U-U_*)=r_{\mathrm{alg}}-\tau_h.
$$

Hence, when \(A_h\) is invertible,

$$
\|\widehat U-U_*\|\leq\|A_h^{-1}\|
\bigl(\|r_{\mathrm{alg}}\|+\|\tau_h\|\bigr).
$$

The exact-data defect \(\tau_h\) probes consistency; the inverse factor shows why consistency alone does not guarantee accuracy. For global coefficient methods, sampled exact values are not a valid coefficient vector: first define a compatible representation or test the reconstructed field instead.

## 3. Local reproduction and solve residuals

The local algebraic residual is

$$
r_i=H_i^{\mathsf T}\widetilde w_i-q_i.
$$

Polynomial reproduction separately checks
\(\sum_jw_{ij}\lambda_jp-\tau_i p\) on \(p\in\Pi_q\).
Record normalization and matrix scaling. Neither diagnostic measures the stability of the global system.

## 4. Off-node PDE residual

For a reconstructed field,

$$
r_\Omega(z)=(\mathcal Lu_h)(z)-f(z),\qquad
r_\Gamma(z)=(\mathcal Bu_h)(z)-g(z).
$$

These test the reconstructed function, not the nodal sparse equation. For nearest-stencil LHI, evaluate the derivatives of the selected local approximation; a jump between two stencil owners is not differentiated by that operation. Thus it is a patchwise residual, not a distributional residual of one globally smooth function.

For Stokes, report separately

$$
\boldsymbol r_m=\partial_t\boldsymbol u_h-\mu\Delta\boldsymbol u_h+\nabla p_h-\boldsymbol f,
\qquad r_d=\nabla\cdot\boldsymbol u_h.
$$

An analytically divergence-free trial space controls \(r_d\) within each reconstructed field but does not ensure an accurate pressure gradient. Pressure values need a consistent gauge before comparison.

## A useful validation sequence

1. Check polynomial reproduction and derivative signs on a small stencil.
2. Check exact-data consistency and the linear solve residual independently.
3. Compare nodal and independent off-node solution errors.
4. Refine multiple clouds with a fixed parameter policy.
5. For evolution, refine time separately and inspect divergence, energy, and growing modes as appropriate.

The [curved validation report](../guides/curved-validation.md) applies several of these checks. The [heat tutorial](../tutorials/heat-equation.md) separates temporal error from the spatial discretization using a semidiscrete reference.

**Background:** [Flyer et al.](references.md#flyer-2016) and [Bayona et al.](references.md#bayona-2017). Match sample locations and norm definitions before comparing results across studies.
