# Conditioning, scaling, and precision

A local interpolation solve, its weight solve, and the global PDE solve have different conditioning. Diagnose each stage separately.

## Local solve sensitivity

For the augmented weight system \(H_i^{\mathsf T}\widetilde w_i=q_i\), write the computed residual as

$$
r_i=H_i^{\mathsf T}\widehat{\widetilde w}_i-q_i,\qquad
\rho_i=\frac{\|r_i\|}{\|H_i^{\mathsf T}\|\,\|\widehat{\widetilde w}_i\|+\|q_i\|}.
$$

This normalized residual measures how well the represented equation was solved. With \(H_i\) fixed and nonzero \(q_i\),

$$
\frac{\|\widehat{\widetilde w}_i-\widetilde w_i\|}{\|\widetilde w_i\|}
\leq\kappa(H_i^{\mathsf T})\frac{\|r_i\|}{\|q_i\|}.
$$

A tiny residual can coexist with inaccurate weights when the system is ill-conditioned. Matrix-entry errors, kernel evaluation errors, and weight rounding introduce further perturbations. Norms must be consistent; do not compare a 2-norm condition estimate and an infinity-norm estimate as identical quantities.

## Coordinate and functional scaling

Set \(\widehat x=(x-\xi_i)/R_i\), where \(R_i\) is a stencil radius. An IMQ in physical coordinates becomes

$$
(1+c\|x-y\|^2)^{-1/2}
=(1+\widehat c\|\widehat x-\widehat y\|^2)^{-1/2},\qquad
\widehat c=cR_i^2.
$$

Thus a fixed dimensionless choice \(\widehat c=\gamma\) means \(c_i=\gamma/R_i^2\). It is not the same policy as holding the physical parameter \(c\) fixed through refinement.

Pure order-\(s\) derivatives acquire \(R_i^{-s}\). Hermite rows mix values, first derivatives, and second derivatives, so their units differ; consistent row/column scaling may materially change the reported condition number while leaving the exact approximation unchanged. Always state which matrix was measured: raw, coordinate-scaled, or algebraically equilibrated.

Polynomial degree, node separation, and geometric rank also matter. Adding more almost-collinear points does not repair missing two-dimensional polynomial information.

## Precision is a pipeline

$$
\text{coordinates/data}\longrightarrow\text{kernel derivatives}
\longrightarrow H_i\longrightarrow w_i
\longrightarrow A_h,b_h\longrightarrow U.
$$

MPFR local solves cannot restore coordinate digits absent from Float64 inputs, and casting high-precision weights to Float64 loses digits before sparse assembly. Record coordinate precision, local arithmetic, stored-weight precision, and global solver precision separately.

SVD truncation replaces inversion of small singular values with a regularized solve. It can reduce sensitivity, but may weaken exact reproduction; report the cutoff and reproduction defects. Optional condition estimation can be disabled for production runs after a recipe has been checked.

## Conditioning is not dynamical stability

For \(M\dot U+A_hU=0\), the evolution generator is \(-M^{-1}A_h\) when \(M\) is invertible. A well-conditioned local system does not rule out growing global modes. Eigenvalues diagnose asymptotic behavior; nonnormality can also create transient amplification. For singular \(M\), use the constrained/generalized system rather than forming an inverse.

**Continue:** [errors and residuals](errors.md), [backend choices](../api/precision-backends.md).
**Background:** [RBF-QR](references.md#fornberg-2011) and [RBF-RA](references.md#wright-fornberg-2017) change the numerical representation. Selecting MPFR or SVD is not an implementation of either algorithm.
