# Fixed-patch PDE and overlap coupling study

Measured status: 2026-10-30. This research study compares additional global
residual equations while retaining the original mixed LHI reconstruction.
It does not change the public API or establish a convergent Stokes method.

The owning implementation is [coupling_study.py](coupling_study.py), with
the original square formulation derived in [METHOD.md](METHOD.md). The
completed results are in
[cells6/results.json](../../outputs/shared_pressure_coupling_20261030/cells6/results.json)
and [cells8/results.json](../../outputs/shared_pressure_coupling_20261030/cells8/results.json).
These two meshes provide a limited comparison, not a convergence study.

## 1. What remains fixed and what is added

The domain is the unit square. The 49-node grid has 25 interior nodes and
24 boundary nodes, with spacing h=1/6. Each interior owner selects its nearest
37 physical grid sites once. Interior velocity observations include the owner;
interior pressure and momentum observations exclude that owner. Boundary
velocity observations are the boundary sites in the same selection.

For every new evaluation location belonging to owner k, the code explicitly
reuses all four source membership lists of owner k. It does not select a new
nearest-37 stencil around the new location. Thus the local Hermite matrix,
ansatz, observations, and polynomial augmentation remain the same; only the
evaluation functional changes. In particular, an auxiliary PDE test point is
never inserted as an exact momentum observation.

The kernel and normalization match the degree-2 baseline: viscosity 0.01,
q=cells/44, velocity potential exp(-10 q^2 r^2)+q^7 r^7, pressure kernel
mu^2 q^4 exp(-q^2 r^2), divergence-free velocity polynomial degree 6, and
scalar pressure polynomial degree 2. The pressure constant is retained.
Local C++ weights use 80 decimal digits and two MPI ranks; saved sparse maps,
global assembly, SVD, and least-squares solves use Float64. Geometry is Float64.

There are four auxiliary evaluation points per owner, at offsets
h(0.3,0.2), h(-0.3,-0.2), h(0.2,-0.3), and h(-0.2,0.3).
`pde2` uses the first two; `pde4` uses all four. Adjacent interior grid owners
also evaluate at their shared midpoint, once using each owner's reconstruction.
For this mesh there are 40 such pairs. The weight build therefore evaluates
25 owner targets, 100 auxiliary targets, and 80 paired midpoint targets:
205 target jobs, without adding any global unknowns. The global unknowns
remain 50 velocity components and 25 pressure values.

The 81-node grid has spacing h=1/8, 49 interior nodes, and 32 boundary nodes.
Its 84 adjacent-owner pairs yield 49 owner evaluations, 196 auxiliary
evaluations, and 168 paired midpoint evaluations: 413 target jobs. Source
support remains nearest-37 with the same owner exclusions and h-scaled
kernel convention; q is now 8/44. There are 147 global unknowns, or 146 after
the pressure gauge.

## 2. Baseline equations and momentum elimination

Write ell=-mu Delta y+grad p and alpha=0 for stationary Stokes or 1/dt
for backward Euler. Let r=F for the stationary problem and
r=F^(n+1)+alpha Y^n at the interior nodes for backward Euler. Then

\[
\ell=r-\alpha Y.
\]

At owner targets the maps are

\[
\widehat\ell=A Y+G P+B g+C\ell,\qquad
\widehat p=R Y+Q P+E g+T\ell.
\]

With M=I-C, requiring momentum and pressure consistency gives

\[
K_\alpha\binom{Y}{P}=
\binom{Mr-Bg}{Eg+Tr},\qquad
K_\alpha=
\begin{pmatrix} A+\alpha M&G\\-R+\alpha T&I-Q\end{pmatrix}.
\]

The [saved validation](../../outputs/shared_pressure_coupling_20261030/validation.json)
found maximum absolute difference zero
between this baseline matrix and `SharedPressureStokes.matrix` for the same
owner maps, in both the stationary and tested backward-Euler assemblies.
This is an assembly parity check, not an accuracy or stability result.

## 3. Independent PDE test rows

At an auxiliary point z, let V_z evaluate velocity and L_z evaluate momentum
from the fixed owner reconstruction. Stationary extra equations enforce
L_z d=F(z), where d=(Y,P,g,ell) is the local data scattered from shared arrays.
Backward Euler instead enforces

\[
(L_z+\alpha V_z)d^{n+1}
=F(z,t^{n+1})+\alpha V_z d^n.
\]

Thus the old velocity at an auxiliary point comes from the previous solved
reconstruction, rather than from exact manufactured velocity after the first
step. Its data contain the previous solved velocity and pressure, previous
boundary values, and previous momentum ell^n=r^n-alpha Y^n. The initial data
are exact and pressure is aligned to the arithmetic nodal mean gauge.

For B_z=L_z+alpha V_z, substituting ell=r-alpha Y yields the actual row

\[
\bigl[B_{z,Y}-\alpha B_{z,\ell}\quad B_{z,P}\bigr]
\binom{Y}{P}
=F(z)+\alpha V_zd^n-B_{z,g}g-B_{z,\ell}r.
\]

For stationary Stokes the alpha terms vanish. These rows add observations
of the PDE residual, without introducing unknown auxiliary velocities or
pressures. Evaluating a sampled source functional at its own source point
would instead reproduce the prescribed datum and provide no new residual.

## 4. Overlap velocity, pressure, and traction rows

For adjacent owners k and l and their midpoint z, candidate overlap equations
are yhat^k(z)-yhat^l(z)=0 or phat^k(z)-phat^l(z)=0. The shared source values
alone do not already enforce these off-node agreements.

Traction uses the same normal n, directed from owner k to owner l, in both
evaluations:

\[
\bigl(\widehat\sigma^k(z)-\widehat\sigma^l(z)\bigr)n=0,
\qquad \sigma=-pI+\mu(\nabla y+\nabla y^T).
\]

The component rows use sigma_xx=2 mu dx(y_1)-p,
sigma_yy=2 mu dy(y_2)-p, and
sigma_xy=mu(dy(y_1)+dx(y_2)). Their two traction components are stored
node-major. Every overlap row also eliminates ell=r-alpha Y, giving
unknown block [B_Y-alpha B_ell, B_P] and right-hand side
-B_g g-B_ell r. Pressure constants cancel between the two reconstructions.

Finite midpoint agreement is a discrete coupling condition. It does not
prove that the patches form one globally continuous or globally solenoidal
field. Integrating flux around a cell entirely within one divergence-free
patch would be an identity and is not an independent closure test.

## 5. Near-nullspace detection and the actual least-squares objective

Let H denote the additional rows and stack S=[K_alpha; H]. Pressure mean zero
is imposed by x=Z_g c, with Z_g the identity on velocity and an orthonormal
basis of mean-zero pressure vectors on pressure. No original equation is
discarded and no pressure penalty is added.

The code minimizes

\[
\min_c\|D(SZ_gc-b)\|_2,\qquad
D_{ii}=1/\max(\|S_{i,:}\|_1,\mathrm{tiny}).
\]

Reduced columns are scaled to unit 2-norm before the dense SVD. This is a
change of variables, not an additional residual weight. A singular value
must exceed sigma_max max(rows,columns) eps64 to count toward numerical rank.
A deficient reduced system is rejected; the runner does not silently apply
a truncated pseudoinverse. The row weighting is an explicit experimental
choice, not an established optimal balance between momentum, velocity,
pressure, and traction units.

The diagnostic also extracts a numerical weak basis Z from the row-scaled,
gauge-reduced baseline matrix and reports the singular values of D_H H Z.
These show whether the new rows detect the baseline's weak directions.
Their interpretation depends on scaling and threshold: they are not a
theorem about an exact missing-equation count. Stacking rows and minimizing
their residuals deliberately changes the discrete objective; it need not
preserve every original equation exactly on a nonpolynomial field.

The [full-precision audit](../../outputs/shared_pressure_precision_20261030/REPORT.md)
retains native 80-digit local weights through global assembly and SVD. On
the 49-node baseline only the pressure gauge lies below relative 1e-30;
the smallest nongauge relative singular values are approximately 1.87e-22
steady, 1.96e-24 at dt=0.00025, and 9.88e-25 at dt=0.000125. This supports
severe numerical conditioning as the explanation for the Float64 rank loss,
rather than nine demonstrated exact missing equations. It remains a
finite-precision audit, with Float64 geometry, rather than an exact-rank proof.

A separate [local-observation study](../../outputs/shared_pressure_observations_20261030/REPORT.md),
implemented in [observation_study.py](observation_study.py), adds two or four
off-node PDE source observations with their matching representers, or replaces
the original PDE source observations with four off-node ones. Those are
changes to the local ansatz, rather than the independent global residual rows
studied here. Every tested stationary variant on 49 or 81 nodes retains
Float64 numerical nullity 10; these changes have not cured the measured
Float64 conditioning problem. Their own fully multiprecision global solves
were not tested. That study uses a different smooth pressure, sin(pi x)
sin(pi y), and reports exact-field consistency residuals rather than solved
errors, so its numbers should not be read as entries in the tables below.

## 6. Measured 49-node results

All successful augmented variants reach full reduced numerical rank 74.
The baseline has reduced rank 65 steady and 67 at both time steps under this
study's scaling. Pressure overlap alone reaches rank 67 and remains rejected
in all three cases. Other variants solve all three manufactured cases.

The smooth case uses streamfunction
(1+t) sin(pi x)^2 sin(pi y)^2 and pressure sin(pi x) cos(pi y).
Backward Euler advances computed history to T=0.01: 40 steps for dt=0.00025
and 80 steps for dt=0.000125. The fields are linear in time, so backward Euler
has no temporal truncation error on the exact manufactured fields; observed
time-step differences still include reconstruction and evolution effects.

The table reports maximum absolute velocity-component and reconstructed
pressure-gradient-component errors at the original 25 interior nodes.

| Variant | Rows | Steady velocity / gradient | dt=0.00025 velocity / gradient | dt=0.000125 velocity / gradient |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 75 | Rejected | Rejected | Rejected |
| PDE, 2 points | 175 | 0.03233 / 0.1342 | 0.08572 / 1.526 | 0.1028 / 0.5676 |
| PDE, 4 points | 275 | 0.06146 / 0.06764 | 0.06311 / 1.559 | 0.08368 / 1.517 |
| Velocity overlap | 155 | 0.4625 / 0.5546 | 0.08897 / 0.4586 | 0.1030 / 0.5821 |
| Pressure overlap | 115 | Rejected | Rejected | Rejected |
| Traction overlap | 155 | 0.2035 / 0.5264 | 0.03356 / 0.1762 | 0.04487 / 0.2293 |
| PDE4 + velocity | 355 | 0.07634 / 0.1897 | 0.05012 / 1.299 | 0.05495 / 1.850 |
| PDE4 + velocity + pressure | 395 | 0.07644 / 0.1797 | 0.04936 / 1.387 | 0.05048 / 1.735 |
| PDE4 + traction | 355 | 0.05964 / 0.06593 | 0.02534 / 0.9992 | 0.02102 / 0.9799 |

The smallest velocity error and smallest pressure-gradient error need not
come from the same variant. For example, PDE4 plus traction gives the
smallest final-time smooth velocity errors among these candidates, while
traction alone gives smaller final-time pressure-gradient errors. Reducing
dt does not uniformly improve errors. The post-scaling condition numbers
of successful variants range from approximately 232 to 1.09e4 steady and
4.17e6 to 8.15e7 for the tested time-dependent matrices.

Both polynomial cases also solve accurately in every full-rank variant.
Across all successful variants and all three equations/time-step settings,
the worst velocity and pressure-gradient errors are respectively 1.69e-12
and 1.96e-10 for y=0,p=xy, and 2.10e-10 and 3.12e-8 for
y=(1+t)(x^2,-2xy),p=xy. These patch tests verify reproduction and assembly;
they do not establish nonpolynomial convergence.

For comparison, the original square closure solved at full 80-digit precision
on the same smooth case gives steady velocity/gradient errors 1.96572/2.57549,
and final-time errors 0.0696248/5.23754 and 0.0696416/5.23745 at the two time
steps. See the owning
[matched-smooth audit](../../outputs/shared_pressure_precision_20261030/matched_smooth/audit.json).
Recovering the baseline's numerical rank alone therefore does not ensure
field accuracy, and the augmented least-squares systems are different
discretizations rather than merely more precise baseline solves.

## 7. Measured 81-node results and two-mesh comparison

The manufactured fields, final time, time steps, and error definitions remain
the same. Errors are now measured at the original 49 interior nodes. Every
previously successful augmented variant reaches full reduced numerical rank
146. The baseline remains rejected, with reduced rank 137 steady and 138 at
both time steps; pressure overlap alone remains rejected at rank 141.

| Variant | Rows | Steady velocity / gradient | dt=0.00025 velocity / gradient | dt=0.000125 velocity / gradient |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 147 | Rejected | Rejected | Rejected |
| PDE, 2 points | 343 | 0.009230 / 0.1376 | 0.007107 / 0.2199 | 0.006677 / 0.2609 |
| PDE, 4 points | 539 | 0.01108 / 0.1869 | 0.007729 / 0.3021 | 0.007893 / 0.3084 |
| Velocity overlap | 315 | 0.07307 / 0.7336 | 0.02672 / 1.251 | 0.02933 / 1.143 |
| Pressure overlap | 231 | Rejected | Rejected | Rejected |
| Traction overlap | 315 | 0.08130 / 0.1266 | 0.005562 / 0.2451 | 0.003310 / 0.2463 |
| PDE4 + velocity | 707 | 0.01377 / 0.2135 | 0.01590 / 1.184 | 0.02154 / 1.274 |
| PDE4 + velocity + pressure | 791 | 0.01347 / 0.2048 | 0.01242 / 0.9055 | 0.01672 / 0.8546 |
| PDE4 + traction | 707 | 0.01315 / 0.1459 | 0.005409 / 0.2474 | 0.003776 / 0.2162 |

Velocity errors decrease from 49 to 81 total nodes for all successful
variants in all three settings. Pressure-gradient errors do not improve
uniformly. For example, stationary PDE4 changes from 0.06764 to 0.1869;
stationary PDE2 changes from 0.1342 to 0.1376. Traction-only final-time
gradient errors change from 0.1762/0.2293 to 0.2451/0.2463, despite its smaller
velocity errors. These counterexamples preclude a claim of uniform field
improvement or an inferred convergence rate.

PDE4 plus traction offers a useful velocity/gradient compromise in the tested
81-node evolution: velocity errors are 0.00540915 and 0.00377624, with gradient
errors 0.247447 and 0.216212. Traction alone gives velocity errors 0.00556186
and 0.00331018, with gradient errors 0.245089 and 0.246287. Neither dominates
the other at both time steps and in both fields. Additional residual rows
and smaller time steps are therefore not automatic accuracy improvements.

Successful 81-node stationary matrices have post-scaling condition numbers
approximately 201 to 7042; PDE2 is about 400. Successful time-dependent
matrices range from 2.37e6 to 4.43e7. These measures describe the explicitly
scaled least-squares matrices, not the continuous Stokes problem. Both
polynomial cases again solve accurately: across successful variants and all
three settings, velocity/gradient error maxima are approximately
4.48e-12/5.59e-10 for y=0,p=xy, and 8.08e-11/1.30e-8 for the nonzero-boundary
polynomial velocity. Polynomial reproduction still does not establish
nonpolynomial convergence.

The saved `overlap_velocity_jump` is a separate maximum component difference
between neighboring patch reconstructions at their shared midpoints. On the
smooth case it ranges approximately 0.00582 to 0.0207 on 49 total nodes, and
0.00184 to 0.0122 on 81 total nodes, across successful variants and settings.
It is not an error against exact velocity and is not a domain-wide continuity
norm. Auxiliary test points are used to impose residuals; neither table
reports solution error at those points or a dense off-node validation grid.
No growth or nonfinite abort occurred over the tested T=0.01 interval on
either mesh. This limited observation does not establish evolution stability,
and the two grids do not establish convergence for these formulations.

## 8. Verification and reproduction

[tests/test_shared_pressure_coupling.py](../../tests/test_shared_pressure_coupling.py)
passed 23 tests in 4.46 seconds in the saved coordinating validation. The tests use synthetic maps
and a mocked weight build, so they do not repeat the expensive MPI experiment.
They independently check direct patch residuals against assembled equations
for all nine variants, stationary and backward Euler; symmetric traction
with an oblique normal; the weighted least-squares objective and gauge;
rejection and detection of an added weak direction; smooth manufactured
derivatives; and frozen owner memberships at all auxiliary/overlap targets.
The deliberately singular synthetic case reports infinite condition number.

For a new output directory:

```powershell
.venv/Scripts/python.exe -m research.shared_pressure_lhi.coupling_study --cells 6 --ranks 2 --output outputs/new_coupling_study
```

The runner preserves completed result files by refusing an existing
`results.json`, stores reusable sparse maps below `weights/`, and records
rank rejection rather than fabricating a solution. Different geometries,
test-center placements, residual weights, and stencil choices need their own
evidence. The 81-node experiment uses `--cells 8` in another new output
directory. The completed two-mesh comparison does not establish a general
convergence rate or long-time stability.
