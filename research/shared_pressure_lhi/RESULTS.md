# Shared-pressure mixed LHI: initial Float64 rank failure

The initial Float64 implementation does **not reliably resolve** velocity and
pressure on either tested mesh: its global matrices have ten numerical null
directions at the reported threshold. This is a failed Float64 solve test,
not a successful cavity or Stokes accuracy result.

**Precision correction, 2026-10-30:** the subsequent
[retained-80-digit audit](../../outputs/shared_pressure_precision_20261030/audit.json)
keeps native MPFR weights through global assembly and an 80-digit SVD, rather
than promoting an already rounded Float64 matrix. On 49 nodes with 6/2
augmentation, only the pressure gauge falls below a relative cutoff of
1e-30. The smallest nongauge relative singular values are 1.87e-22
(stationary), 1.96e-24 (dt=0.00025), and 9.88e-25 (dt=0.000125), implying
effective nongauge condition estimates of order 1e22--1e24 in that scaling.
Thus the earlier numerical nullity is not proof of exact structural
dependence or missing closure equations. The original measurements below
remain valid historical evidence of severe Float64 conditioning. Geometry
remains Float64; higher-precision rank resolution alone does not establish
accuracy, convergence, or evolution stability.

## Configuration and evidence

These runs use the unit square with 6 cells per coordinate: 49 total nodes,
25 interior velocity/pressure nodes, and 75 global unknowns. The stencil
contains the nearest 37 physical sites. Velocity augmentation is degree 6;
pressure augmentation is degree 1 or 2. The full h-scaled kernels, viscosity
0.01, pressure normalization, and exact observation exclusions are specified
in [METHOD.md](METHOD.md).

Local weights were computed by the C++ backend using 80 decimal digits and
two MPI ranks. Each worker completed successfully. Geometry, global assembly,
and the dense SVD rank audit use Float64. The singular-value audit is applied
after row 1-norm equilibration, with threshold
\(\sigma_{\max}(K_{scaled})\,75\,\epsilon_{64}\).

| Pressure degree | Equation | Global size | Numerical nullity | Rank threshold |
| --- | --- | --- | --- | --- |
| 1 | Stationary | 75 | 10 | 3.052e-14 |
| 1 | Backward Euler, dt=0.00025 | 75 | 10 | 2.297e-14 |
| 1 | Backward Euler, dt=0.000125 | 75 | 10 | 2.284e-14 |
| 2 | Stationary | 75 | 10 | 3.052e-14 |
| 2 | Backward Euler, dt=0.00025 | 75 | 10 | 2.297e-14 |
| 2 | Backward Euler, dt=0.000125 | 75 | 10 | 2.285e-14 |

Primary artifacts:

- [Degree-1 MPI JSON](../../outputs/shared_pressure_closure_20261030/degree1_mpi.json)
- [Degree-2 MPI JSON](../../outputs/shared_pressure_closure_20261030/degree2_mpi.json)

Each artifact records both the pressure-only xy case and the nonzero-boundary
manufactured case. They use the same operator matrices; the duplicate rank
results for different forcing cases are not independent evidence about rank.

The expected constant-pressure vector is indeed nearly annihilated: its
scaled maximum defect ranges from 3.47e-17 to 7.11e-17. However, nine additional
numerical null directions remain. The six smallest reported singular values
are between approximately 7.8e-18 and 6.1e-16. The reported nullity uses the
complete singular-value spectrum, not only those six displayed values.

The MPI execution metadata records 25 distinct target jobs, split 13 and 12
across two workers, with zero cache hits. The recorded MPI execution interval
is about 20.1 seconds for pressure degree 1 and 22.6 seconds for pressure
degree 2. These intervals are not an end-to-end solver timing or a speedup
comparison.

## Consequences

A finer degree-2 run uses 8 cells per direction: 81 total nodes, 49 interior
nodes, and 147 global unknowns, retaining the 37-point stencil. Its numerical
nullity is also 10 for stationary Stokes and both backward-Euler matrices.
The corresponding rank thresholds are 5.366e-14, 4.211e-14, and 4.097e-14.
The scaled constant-pressure defects range from 4.21e-17 to 4.97e-17. Thus the
observed defect is not confined to the 49-node cloud, although these two small
meshes do not establish its behavior on arbitrary clouds.

Additional primary artifacts:

- [Degree-2 49-node exact-field diagnostics](../../outputs/shared_pressure_closure_20261030/degree2_patch_mpi.json)
- [Degree-2 81-node MPI diagnostics](../../outputs/shared_pressure_closure_20261030/degree2_cells8_mpi.json)

The 81-node MPI run completes 49 distinct target jobs, split 25 and 24, with
zero cache hits and an MPI execution interval of about 47.2 seconds.

The solver reports `rank_deficient` and does not use least squares or discard
equations to fabricate a unique solution. No nodal velocity, pressure, or
pressure-gradient errors are available from these rejected solves. No time
trajectory was advanced, so these runs provide no stability observation.

Increasing pressure augmentation from 1 to 2 does not cure this rank failure.
Changing the time step between the two tested values also does not cure it.
Refining from 49 to 81 total nodes with degree-2 pressure also does not cure it.
The Float64 matrix fails the numerical rank threshold before any forcing
values or exact pressure values are applied. This issue is distinct from the local xy polynomial reproduction
test: a locally accurate map need not yield independent global closure rows.

## Exact-field consistency without solving

The newer degree-2 artifacts substitute the known manufactured nodal fields
into the assembled equations. The following are **maximum absolute block
residuals**, not velocity/pressure errors from a computed solution. The
nonzero-boundary backward-Euler check uses exact previous and current fields
for one step; no numerical time trajectory is advanced.

| Total nodes | Exact field | Equation | Momentum block residual | Pressure block residual |
| --- | --- | --- | --- | --- |
| 49 | y=0, p=xy | Stationary and both dt | 4.668e-14 | 5.127e-16 |
| 81 | y=0, p=xy | Stationary and both dt | 6.656e-14 | 1.511e-16 |
| 49 | Nonzero boundary polynomial | Stationary | 4.885e-14 | 5.481e-16 |
| 49 | Nonzero boundary polynomial | dt=0.00025 | 1.364e-12 | 6.163e-16 |
| 49 | Nonzero boundary polynomial | dt=0.000125 | 5.457e-12 | 6.799e-16 |
| 81 | Nonzero boundary polynomial | Stationary | 9.670e-14 | 1.554e-16 |
| 81 | Nonzero boundary polynomial | dt=0.00025 | 2.728e-12 | 1.319e-16 |
| 81 | Nonzero boundary polynomial | dt=0.000125 | 3.638e-12 | 2.440e-16 |

These results support polynomial consistency of the assembled rows for the
tested fields while exposing the separate failure of Float64 numerical rank.
They do not show that the matrix selects the exact field among its possible
solutions. The larger backward-Euler momentum residuals should be interpreted
with the 1/dt matrix scaling; they are not evidence of accumulated time error.

## Possible mechanisms to investigate

The following are hypotheses, not established explanations:

1. The pressure prediction equations may be algebraically dependent on
   momentum reconstruction and polynomial reproduction identities. Excluding
   the target pressure avoids an immediate identity row but does not prove
   independence from the other rows.
2. The degree-6 divergence-free polynomial space is large relative to this
   small cloud, and nearest-37 patches overlap over much of the 49-node grid.
   Structured sites and extensive overlap could make polynomial relations
   unusually prominent. This does not prove that polynomial augmentation
   causes the extra null modes or that a finer mesh will fix them.
3. Float64 global assembly can expose numerical rank loss even with
   multiprecision local weights. The repeated nullity and very small singular
   values warrant inspecting the actual null vectors and their pressure and
   velocity components, rather than attributing the result to conditioning
   alone.

The focused suite `tests/test_shared_pressure_closure.py` passed all three
tests in 440.26 seconds. Its independent Python-backend, 49-node degree-2
assembly also reports extra null modes for the steady matrix and both time
steps. This checks the failure through a second backend; it is not a full
entrywise matrix-parity test. The other tests check a compatible gauge solve,
expose an incompatible right-hand side through its residual and border
multiplier, and reject additional null modes.

A useful next diagnostic is to evaluate block residuals on each null vector and
project its velocity and pressure samples onto polynomial bases. Only that
evidence could distinguish a structural closure defect from a small-cloud
degeneracy or a numerical implementation issue. This report does not propose
adjusting the time step as a remedy for the measured conditioning problem.
