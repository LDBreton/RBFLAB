# Pressure-point variational LHI experiment

Implemented and measured 2026-10-30. This is an isolated research implementation,
not a public RBFLAB solver and not a change to the Dropbox article.

## Method and exact benchmark

Domain [0,1]^2; viscosity mu=0.1; uniform n-by-n grid including boundary.
All (n-2)^2 interior sites carry two velocity unknowns and one shared pressure
unknown. Every interior site is a patch target. Each patch selects its nearest
17 grid sites (stable index tie breaking), with velocity/pressure samples at
selected interior sites and exact Dirichlet samples at selected boundary sites.
Auxiliary momentum sites are the selected interior sites excluding the target;
their two momentum data are patch-local unknowns. Pressure sample sets include
the target: no pressure self-interpolation equation is used.

Scalar potential psi(r)=exp(-epsilon^2 r^2); velocity kernel
(-Delta I+grad grad^T)psi; scalar pressure kernel psi. Only augmentation is the
exact additive pressure constant with the Hermite side condition q^T alpha=0.
No velocity polynomial augmentation. Fix pressure at the first interior point
to zero; compare the exact pressure in that same gauge.

Manufactured fields:

    u = exp(-t) (sin(pi*x) cos(pi*y), -cos(pi*x) sin(pi*y))
    p = exp(-t) sin(pi*x) sin(pi*y), modulo gauge
    f = (2*pi^2*mu - 1)*u + grad(p)

Boundary velocity is nonzero prescribed exact velocity, not a no-slip-zero
benchmark. Time integration: backward Euler startup, then BDF2 with the time
derivative moved to the RHS of stationary L. Initial values are analytical.
Previous velocities at auxiliary sites come from their own patch's reconstruction.

Minimize squared momentum residuals plus lambda times native interpolation
energy. Auxiliary residual block is divided by sqrt(number of scalar auxiliary
data); the two target residuals are divided by sqrt(2). Energy has no additional
normalization. All quantities are dimensionless. Local QR eliminates auxiliary
data; sparse normal equations couple shared velocity and pressure unknowns.
Diagonal scaling and SuperLU solve these equations in Float64. Local Gram
matrices use diagonal scaling and Cholesky. No jitter or eigenvalue clipping.
Dense eigendecompositions are diagnostic only, on these small global systems.

The final local implementation computes target weights by factor solves, and
the native energy by a whitened orthogonal projection, avoiding explicit inverse
multiplication in those operations. This materially improves numerical behavior.

## Findings

Primary final-code results are in
`../../outputs/pressure_point_variational_20261030_stable/results.json`.
For epsilon=2, lambda=1e-6, dt=.02, T=.1:

| Grid | Global unknowns | Velocity relative nodal RMS | Pressure relative nodal RMS | Velocity off-node RMS | Pressure off-node RMS |
|---|---:|---:|---:|---:|---:|
| 5 x 5 | 26 | 3.33% | 14.46% | .0320 | .0641 |
| 7 x 7 | 74 | about .38% | about 3.64% | about .00223 | about .0155 |
| 9 x 9 | 146 | .0640% | .4993% | .000352 | .002347 |
| 13 x 13 | 362 | .0220% | .0448% | .0000982 | .000211 |

The last row is NOT a reliable accuracy certification: its local scaled Gram
condition reaches 1.45e16 and its worst interpolation matrix identity defect is
0.289. At 9 x 9 these diagnostics are 3.38e12 and 7.20e-5 respectively; arithmetic
sensitivity remains relevant there too. Higher precision or a stable flat-kernel
basis is required for a convincing fine-grid study. The apparent improvement
for this smooth target does not remove the interpolation defect.

Errors are pointwise RMS, not domain-integral L2. Velocity RMS averages over
both components. Relative errors divide by exact nodal RMS (pressure after point
gauge alignment). Off-node measurements use 200 uniform pseudorandom points in
[.03,.97]^2, seed 20261030, and the nearest patch target's reconstruction; no
partition-of-unity blending or globally smooth reconstruction is claimed.

All 12 final-code spatial/regularization cases had positive numerical minimum
eigenvalues of the scaled global Hessian and its pressure Schur complement.
At 9 x 9, epsilon=2, lambda=1e-6, the pressure Schur minimum is 4.176e-6 and
scaled global condition is 1.526e6. This supports pressure independence for these
assembled configurations, not a mesh-uniform inf-sup bound. The 146-unknown
matrix has 13,684 stored nonzeros; the 362-unknown matrix has 43,114. Tiny grids
are relatively dense because patches overlap broadly.

Regularization bias matters. On the 9 x 9 grid, increasing lambda from 1e-6 to
1 raises pressure relative error from about 0.50% to 14.21%, and momentum
residual RMS from about 2.02e-4 to .335. An invertible system is not necessarily
an accurate PDE solution.

## Time checks and limitations

Final-code additional results:

- `../../outputs/pressure_point_variational_20261030_stable_half/results.json`
- `../../outputs/pressure_point_variational_20261030_stable_quarter/results.json`
- `../../outputs/pressure_point_variational_20261030_stable_long/results.json`

At 9 x 9, epsilon=2, lambda=1e-6, T=.1:

| dt | Relative velocity RMS | Relative pressure RMS |
|---|---:|---:|
| .02 | .0640% | .4993% |
| .01 | .0614% | .6486% |
| .005 | .0745% | .9060% |

Thus this does NOT demonstrate temporal convergence. Spatial approximation,
regularization/time-scaling effects, and finite arithmetic have not been
separated. With dt=.02 through T=1, relative velocity/pressure errors are .0874%
and .4998%. This forced trajectory remained accurate; it is not a perturbation
stability or long-time stability proof.

## Verification and reproduction

Run from repository root with the existing virtual environment:

    .venv/Scripts/python.exe experiments/pressure_point_variational/verify.py
    .venv/Scripts/python.exe experiments/pressure_point_variational/run.py --output outputs/NEW_UNIQUE_DIRECTORY --n 5 7 9 --eps 2 --lam 0.000001 0.001 1

The runner refuses an existing output directory. Add `--dt .01`, `--T 1`, or
`--stencil 17` as required. No prior data is overwritten.

Independent checks passed:

- Symbolic SymPy differentiation versus Hermite-polynomial kernel contraction:
  maximum difference 9.77e-15 over all 25 functional block types.
- Small-patch interpolation identity defect: 5.41e-11.
- Native-energy quadratic form agrees with alpha^T A alpha; constant mode has
  zero energy to numerical tolerance.
- Direct full least-squares solve versus local QR condensation: solution
  difference 2.88e-12.

Earlier output directories with prefixes `pilot`, `sweep`, `verified`,
`dt_half`, `dt_quarter`, `long`, and `shape3` preserve preliminary calculations
before the factor-solve/whitened-energy correction. Do not mix them with the
final `stable*` results. The preliminary epsilon=4 sweep was much less accurate
on coarse grids; no universal kernel or regularization optimum was established.

Conclusion: the formulation is implementable, its pressure closure works in
these tests, and useful accuracy is possible on this manufactured field.
It is not yet a validated convergent or generally stable Stokes method.
