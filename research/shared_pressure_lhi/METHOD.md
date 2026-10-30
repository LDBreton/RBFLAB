# Shared-pressure mixed LHI: local reconstruction and global closure

This research formulation adds shared nodal pressure unknowns to the mixed
Local Hermite Interpolation (LHI) reconstruction. It uses the notation of the
LHI manuscript: velocity **y**, pressure p, viscosity mu, local disks D^k,
local matrices A^(k), basis rows H^(k), and weights W^(k). The global closure
below is a candidate discretization. Square matrices, polynomial reproduction,
and small residuals do not establish stability or convergence.

**Precision correction, 2026-10-30:** the original Float64 global assembly
fails its numerical rank check on 49 and 81 nodes. Those measurements are
preserved in [RESULTS.md](RESULTS.md), but they do not prove exact missing
closure equations. The subsequent [retained-80-digit audit](../../outputs/shared_pressure_precision_20261030/audit.json)
on 49 nodes with 6/2 augmentation finds only the pressure gauge below a
relative singular-value cutoff of 1e-30. The smallest nongauge relative
singular values are approximately 1.87e-22 for stationary Stokes, 1.96e-24
for dt=0.00025, and 9.88e-25 for dt=0.000125. This corresponds to effective
nongauge condition estimates of order 1e22--1e24 in the audited scaling.
The Float64 solve rejection therefore exposes severe numerical conditioning;
it is not evidence that one pressure gauge is mathematically insufficient.
Geometry remains Float64, and this small high-precision audit does not prove
convergence or evolution stability.

The local construction follows `examples/stokes_pressure_samples.py`. That
example alone builds local maps; `solver.py` adds the global pressure
consistency equations derived here. Pressure observations are unknown values,
not measured data or prescribed pressure boundary conditions.

## 1. Problem and notation

For Stokes on the unit square, write

\[
\partial_t\mathbf y+\boldsymbol{\mathcal L}\mathbf u=\mathbf F,
\qquad \nabla\cdot\mathbf y=0,
\qquad \mathbf y|_{\partial\Omega}=\mathbf g,
\qquad \mathbf u=(\mathbf y,p),
\]
\[
\boldsymbol{\mathcal L}\mathbf u=-\mu\Delta\mathbf y+\nabla p.
\]

Stationary Stokes omits the time derivative. Denote the momentum field by
\(\boldsymbol\ell=\boldsymbol{\mathcal L}\mathbf u\); hence
\(\boldsymbol\ell=\mathbf F\) in the stationary problem. Define scalar
observation operators

\[
I_a\mathbf u=y_a,\quad I_{d+1}\mathbf u=p,\quad
L_a\mathbf u=-\mu\Delta y_a+\partial_a p,
\qquad a=1,\ldots,d.
\]

For Dirichlet boundary conditions, \(B_a\mathbf u=y_a\) at boundary sites.
The following derivation uses \(d=2\), but the block notation is general.

## 2. Local sites and observations

Each target \(x_1^{(k)}\) is an interior mesh node. Its stencil contains 37
nearest physical sites, drawn from the complete cloud including the boundary.
Within that shared stencil, define:

| Manuscript-style set | Observations | Local data |
| --- | --- | --- |
| \(\Omega_{sc}^k\) | Interior velocity \(I_a\mathbf u\) | \(\mathbf Y^{(k)}\) |
| \(\Omega_{pc}^k\) | Interior pressure \(I_{d+1}\mathbf u\) | \(P^{(k)}\) |
| \(\Omega_{fc}^k\) | Boundary velocity \(B_a\mathbf u\) | \(\mathbf g^{(k)}\) |
| \(\Omega_{pdec}^k\) | Interior momentum \(L_a\mathbf u\) | \(\boldsymbol\ell^{(k)}\) |

The target belongs to the velocity samples. Exclude the target from pressure
and momentum samples. Otherwise evaluating either sampled functional at that
same target gives an interpolation identity, not an independent closure row.
These sets can overlap in physical location: 37 sites does not mean 37 scalar
observations. No velocity boundary observation is duplicated as an interior
velocity observation.

For N interior nodes, the shared global unknowns are
\(\mathbf Y\in\mathbb R^{2N}\) and \(P\in\mathbb R^N\).
Velocity components are stored node-major, as `(y1, y2)` per node. Every patch
refers to the same global pressure value at a shared pressure site.

## 3. Kernel, scaling, and polynomial space

The mixed matrix kernel is

\[
\boldsymbol\Phi(x,\xi)=
\begin{pmatrix}\boldsymbol\Phi_{Div}(x,\xi)&0\\0&\psi_p(x,\xi)\end{pmatrix},
\qquad
\boldsymbol\Phi_{Div}=(-\Delta_x I+\nabla_x\nabla_x^T)\psi_v.
\]

For grid spacing \(h=1/\mathrm{cells}\), use full scaling
\(q=(1/44)/h=\mathrm{cells}/44\), with scalar potentials

\[
\psi_v(r)=e^{-10q^2r^2}+q^7r^7,
\qquad
\psi_p(r)=\mu^2q^4e^{-q^2r^2}.
\]

The pressure Gaussian coefficient is \(\bar c_p=1\). The factor \(\mu^2\)
is part of this pressure-observation prototype's kernel normalization; it
must be recorded when comparing with other saved runs. In the local example,
\(\mu=0.01\). Local weights use 80 decimal digits; geometry and assembled
global solves remain Float64.

The polynomial augmentation is the direct sum of divergence-free vector
polynomials of degree at most 6 and scalar pressure polynomials of degree
at most 1 or 2. Polynomial observations, including momentum observations,
enter the augmented matrix. These polynomials need not lie in the nullspace
of the momentum operator.

Absolute pressure observations require `ScalarSpace`, including its constant
polynomial mode. A pressure space defined modulo constants cannot directly
represent absolute pressure samples. Keeping that local constant mode does
not remove the global additive pressure gauge.

## 4. Local Hermite reconstruction

List every scalar observation as a functional \(\lambda_r\), including its
site. With a basis \(\boldsymbol\pi_s\) of the augmentation space, write

\[
\widehat{\mathbf u}^{(k)}(x)
=\sum_{r=1}^{n_o^{(k)}}a_r^{(k)}
  \lambda_r^\xi\boldsymbol\Phi(x,\xi)
+\sum_{s=1}^{m}c_s^{(k)}\boldsymbol\pi_s(x).
\]

Source operators act rowwise on the matrix kernel, producing vector-valued
representers. In the manuscript's expanded ansatz these are the velocity
terms \(I_a^\xi\boldsymbol\Phi\), boundary terms
\(B_a^\xi\boldsymbol\Phi\), and momentum terms
\(L_a^\xi\boldsymbol\Phi\), together with the new pressure terms
\(I_{d+1}^\xi\boldsymbol\Phi\).

Define

\[
G^{(k)}_{rs}=\lambda_r^x\lambda_s^\xi\boldsymbol\Phi(x,\xi),
\qquad \Pi^{(k)}_{rs}=\lambda_r\boldsymbol\pi_s,
\]
\[
A^{(k)}=
\begin{pmatrix}G^{(k)}&\Pi^{(k)}\\(\Pi^{(k)})^T&0\end{pmatrix},
\qquad
A^{(k)}\binom{a^{(k)}}{c^{(k)}}=
d^{(k)}=
\begin{pmatrix}\mathbf Y^{(k)}\\P^{(k)}\\\mathbf g^{(k)}\\
\boldsymbol\ell^{(k)}\\0\end{pmatrix}.
\]

Each pressure observation adds both one equation and its matching representer.
The local system is square. This construction still requires independent
observations and polynomial unisolvency; square does not imply invertible.

## 5. Local weights and sparse maps

Collect the representers and polynomials into \(H^{(k)}(x)\). For a target
functional \(\mathcal J\),

\[
W_{\mathcal J}^{(k)}=
\left.\mathcal J H^{(k)}(x)\right|_{x=x_1^{(k)}}(A^{(k)})^{-1}.
\]

Do not form the inverse. Solve the transposed local system

\[
(A^{(k)})^T(W_{\mathcal J}^{(k)})^T
=\left[\left.\mathcal JH^{(k)}(x)\right|_{x=x_1^{(k)}}\right]^T.
\]

The weights depend on sites and approximation choices, not the unknown
velocity or pressure values. MPI can distribute independent target patches;
global assembly and solves are separate from this local-weight parallelism.

Scatter the momentum and pressure evaluation rows into sparse global maps:

\[
\widehat{\boldsymbol\ell}=A\mathbf Y+GP+B\mathbf g+C\boldsymbol\ell,
\qquad
\widehat P=R\mathbf Y+QP+E\mathbf g+T\boldsymbol\ell.
\]

Here A without a superscript is a global velocity-to-momentum map; it is not
the local Hermite matrix \(A^{(k)}\). The block dimensions are

| Map | Dimension |
| --- | --- |
| A, C | \(2N\times2N\) |
| G | \(2N\times N\) |
| R, T | \(N\times2N\) |
| Q | \(N\times N\) |
| B | \(2N\times2N_b\) |
| E | \(N\times2N_b\) |

## 6. Stationary global closure

Enforce both momentum and pressure consistency:
\(\widehat{\boldsymbol\ell}=\mathbf F\) and \(\widehat P=P\).
With \(M=I-C\), the system is

\[
K\binom{\mathbf Y}{P}=b,
\qquad
K=\begin{pmatrix}A&G\\-R&I-Q\end{pmatrix},
\qquad
b=\binom{M\mathbf F-B\mathbf g}{E\mathbf g+T\mathbf F}.
\]

Momentum alone gives 2N equations for 3N unknowns. The pressure reconstruction
adds N candidate equations. They are not pressure boundary data or the
continuous incompressibility equation. Their independence and numerical
adequacy must be measured rather than assumed.

## 7. Backward Euler

Before eliminating momentum, the three relations are

\[
A\mathbf Y^{n+1}+GP^{n+1}-M\boldsymbol\ell^{n+1}
=-B\mathbf g^{n+1},
\]
\[
-R\mathbf Y^{n+1}+(I-Q)P^{n+1}-T\boldsymbol\ell^{n+1}
=E\mathbf g^{n+1},
\]
\[
\mathbf Y^{n+1}/\Delta t+\boldsymbol\ell^{n+1}
=r^{n+1}:=\mathbf F^{n+1}+\mathbf Y^n/\Delta t.
\]

Substitution gives the sparse system

\[
K_{\Delta t}\binom{\mathbf Y^{n+1}}{P^{n+1}}
=\binom{Mr^{n+1}-B\mathbf g^{n+1}}
{E\mathbf g^{n+1}+Tr^{n+1}},
\]
\[
K_{\Delta t}=
\begin{pmatrix}
A+M/\Delta t&G\\
-R+T/\Delta t&I-Q
\end{pmatrix}.
\]

The pressure gauge must be handled at every time level. For fixed geometry,
viscosity, and time step, the global factorization can be reused.

## 8. Pressure gauge, rank, and compatibility

Constant reproduction predicts the right null vector
\(v=(0_{2N},1_N)^T\). Check \(Kv\), and likewise
\(K_{\Delta t}v\), instead of assuming exact Float64 cancellation.
On a small diagnostic mesh, inspect all singular values. The expected
nullity is one; additional numerical null directions must be reported.

The implementation equilibrates each row by its absolute row sum. With
\(D_{ii}=1/\max(\sum_j|K_{ij}|,\mathrm{tiny})\), it audits
\(\widetilde K=DK\) and uses the numerical rank threshold
\(\sigma_{\max}(\widetilde K)(3N)\epsilon_{64}\). More than one numerical
null mode raises an error; no least-squares fallback conceals numerical rank
loss. Such loss may reflect severe conditioning rather than exact missing
equations, as the retained-precision audit above demonstrates.
The dense SVD is deliberately restricted to at most 3000 unknowns and is not
a scalable production rank test.

For a scaled matrix with precisely one null direction, let z be a unit left
null vector, obtained from its smallest left singular vector. Let
\(w=(0_{2N},1_N/N)^T\) prescribe the arithmetic nodal pressure mean. Solve

\[
\begin{pmatrix}\widetilde K&z\\w^T&0\end{pmatrix}
\binom{x}{\lambda}=\binom{Db}{0},
\qquad x=(\mathbf Y,P)^T.
\]

The border fixes the pressure mean without arbitrarily discarding a physical
row. In exact arithmetic, compatibility requires \(z^TDb=0\), and the border
multiplier satisfies \(\lambda=z^TDb\). A nonzero multiplier means that the
solution satisfies \(Kx=b-D^{-1}z\lambda\), not the original equations exactly.
Report both the original residual and the multiplier. In a nearly singular
Float64 matrix, z is an approximate left null vector, and the same quantities
remain diagnostics rather than an exact compatibility theorem. Numerical
nullity zero is permitted because roundoff may weakly lift the expected
constant mode, but the constant-pressure defect and raw residual must still
be inspected. The implementation reports both the scaled and original
maximum absolute residuals, the six smallest singular values, the rank
threshold, and the scaled constant-pressure defect.

## 9. What divergence-free means here

Every local reconstructed velocity is divergence-free by construction,
including the polynomial part. Applying divergence to that same local
reconstruction produces zero rows, so it cannot supply an additional global
pressure equation. Independently reconstructed overlapping patches need not
define one globally continuous velocity function. Local divergence
cancellation is distinct from consistency, global flux conservation,
stability, and nodal solution error.

## 10. Corner subtraction for a later cavity experiment

For a known divergence-free singular field \((\mathbf y_s,p_s)\), define
\(\mathbf y=\mathbf y_s+\mathbf v\) and \(p=p_s+\pi\). The remainder solves

\[
\partial_t\mathbf v-\mu\Delta\mathbf v+\nabla\pi
=\mathbf F-\partial_t\mathbf y_s+\mu\Delta\mathbf y_s-\nabla p_s,
\qquad \mathbf v|_{\partial\Omega}=\mathbf g-\mathbf y_s.
\]

Use these remainder source and boundary values in exactly the same blocks,
then add the singular field back at evaluation nodes. A cutoff applied to a
singular velocity can destroy its divergence-free property; any cutoff must
be constructed compatibly (for example through a streamfunction), with all
resulting derivatives included. The initial smooth pressure-patch experiment
does not implement or validate corner subtraction or a cavity solution.

## 11. Validation and limits

For the pressure-only equilibrium, prescribe only
\(\mathbf g=0\) and \(\mathbf F=(y,x)\), and solve for velocity and pressure.
The exact fields are \(\mathbf y=0\), \(p=xy\), with the exact nodal mean
subtracted for the chosen gauge. Do not feed exact pressure into the global
solve. Degree-2 pressure augmentation includes xy; degree 1 does not.

Report absolute velocity and pressure-gradient errors at mesh nodes, and
gauge-aligned pressure errors separately. Evaluate the reconstructed gradient
using the solved velocity and pressure, boundary data, and the actual
momentum data. For backward Euler, use
\(\boldsymbol\ell^{n+1}=r^{n+1}-\mathbf Y^{n+1}/\Delta t\).

The second manufactured test uses
\[
\mathbf y=(1+t)(x^2,-2xy),\qquad p=xy,
\]
\[
\boldsymbol\ell=(y-2\mu(1+t),x),\qquad
\mathbf F=\boldsymbol\ell+(x^2,-2xy).
\]
It has nonzero Dirichlet boundary data and a velocity linear in time, so
backward Euler introduces no temporal truncation error for the exact nodal
solution. This separates assembly and coupling errors from time accuracy.
For its stationary variant, fix the amplitude at one and use
\(\mathbf F=\boldsymbol\ell\); the time-derivative term must not be included.

Compare the local backend/MPI weights independently of global accuracy.
Inspect global rank, gauge defects, compatibility, and original equation
residuals. Repeating backward Euler at two time steps tests those specific
runs only. A successful polynomial patch test does not establish convergence
for a cavity, and backward Euler boundedness does not rule out positive
semidiscrete modes.

## 12. Running the research prototype

From the repository root, with an environment containing RBFLAB:

```powershell
python -m research.shared_pressure_lhi.solver --ranks 2 --pressure-degree 2 --output outputs/shared_pressure_lhi_trial/p2.json
python -m research.shared_pressure_lhi.solver --ranks 2 --pressure-degree 1 --output outputs/shared_pressure_lhi_trial/p1.json
```

The default mesh has 6 cells per direction, 49 total nodes, and 25 interior
targets. It uses nearest-37 stencils and the C++ backend; `--backend python`
selects the Python backend without MPI. Existing output files are rejected
to preserve evidence. Both pressure degrees use the same geometry and all
interior nodes as targets.

Each experiment prepares stationary and backward-Euler versions of both
manufactured fields. The planned time steps are 0.00025 and 0.000125, with 4 and 8
steps respectively, reaching the same final time 0.001 from exact initial
velocity. Reported velocity maximum and pressure-gradient maximum are maxima
over absolute components at interior nodes. Velocity vector RMS is
\(\sqrt{N^{-1}\sum_i\|\mathbf Y_i-\mathbf y(x_i)\|_2^2}\). Pressure maximum
uses the exact arithmetic nodal mean gauge. These are absolute nodal errors,
not relative or off-node errors. The saved JSON records the complete tested
parameters and diagnostics. This document derives the implementation; it
does not imply that these tests produce a unique velocity/pressure solution.
The initial recorded runs stop at the rank audit, before time stepping or
nodal solution errors can be measured.
The current runner catches `UnresolvedNullspace` and saves a per-case
`rank_deficient` record. It can additionally apply the assembled matrix to
the known manufactured nodal fields and report each block's exact-field
residual. Such residuals test consistency without solving the singular
system; they are not computed-solution errors. Rank-deficient cases skip all
planned time integration.
