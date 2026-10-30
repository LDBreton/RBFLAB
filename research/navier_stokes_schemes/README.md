# Mixed LHI Navier–Stokes time schemes

This directory contains research implementations for comparing time discretizations
of the same mixed velocity/pressure reconstruction. These modules are outside the
stable `rbflab` namespace. Algebraic unit checks validate individual formulas;
they do not establish spatial accuracy, convergence on a node cloud, or stability.

## Shared equations and observations

Consider `u_t - nu Δu + (u·∇)u + ∇p = f`, with prescribed velocity `g(t)`
on the boundary. Vectors use node-major component order
`[ux_0, uy_0, ux_1, uy_1, ...]`. For schemes 1–6, `L` is the interior
Stokes observation `-nu Δu + ∇p`. Mixed interpolation supplies the constraint

```
A U + B g + SL L = L,       M = I - SL,
U_t + L + C(U,L,g) = f.
```

Velocity derivatives always include all three observation groups:
`DxU @ U + DxB @ g + DxL @ L`, and likewise for `Dy`.
Pressure gradient is `PY @ U + PB @ g + PL @ L`.
No separate divergence-free velocity reconstruction supplies these derivatives.
The coupled formulations avoid assuming that `M` is invertible.

## Methods

| Number | Module / selector | Time discretization and nonlinear treatment |
| --- | --- | --- |
| 1 | `coupled.py`, `oseen` | Backward-Euler startup, then BDF2; extrapolate the advector and solve jointly for current `U,L`, including current mixed derivatives. |
| 2 | `coupled.py`, `implicit` | Same time derivative; full advective nonlinearity solved by damped Newton from extrapolated states. |
| 3 | `coupled.py`, `oseen_newton` | Scheme 1 supplies the initial iterate; Newton corrections continue until the full nonlinear residual meets tolerance. The converged equation is the same as scheme 2. |
| 4 | `transport.py`, `imex` | Implicit-Euler half-step predictor, explicit midpoint convection, and Crank–Nicolson Stokes evolution, with algebraic constraints at both stages. |
| 4 variant | `transport.py`, `imex_ars222` | Globally stiffly accurate ARS(2,2,2): two coupled implicit Stokes stages and explicit convection evaluated with stage-consistent mixed observations. |
| 5 | `coupled.py`, `rotational` | Fully implicit BDF2 with convection `omega × u`, `omega=dx(uy)-dy(ux)`; solve for modified pressure `p+|u|²/2`. |
| 6 | `transport.py`, `characteristic` or `characteristic_bdf2` | First-order characteristic backward Euler, or BDF2 with midpoint departure trajectories and extrapolated velocity. Values at departure points use the mixed reconstruction. |
| 7 | `adaptive_hermite.py`, `adaptive_hermite` | Experimental target-frozen Oseen Hermite representers rebuilt each step, with independent derivative unknowns and source corrections. |

The formal second-order methods require smooth solutions, consistent initial
observations, and sufficiently accurate spatial maps and nonlinear solves.
Backward-Euler startup introduces a single local error of order `dt²`.
The first-order `characteristic` variant is a separate baseline.

The ARS variant uses `gamma=1-1/sqrt(2)` and
`delta=1-1/(2*gamma)`. Its final state is the last stage of both tableaux,
so the final algebraic momentum constraint is enforced directly. Each implicit
stage solves jointly for velocity and momentum; no inverse of `M` is assumed.
This is a distinct method from the retained midpoint/CN implementation.

The current native comparison reported a midpoint/CN pressure-gradient failure
on its 49-node run at `T=0.1`: gradient RMS error was about 67 at
`dt=0.00025` and about 134 after halving the step, while velocity remained
bounded. Treat this as a failed configuration pending the owning run metadata;
it does not establish failure for all mixed reconstructions. Scalar DAE order
tests alone do not validate pressure accuracy on that node cloud. The ARS
variant was added to compare stiffly accurate stages against this retained
failure; its accuracy on that configuration requires a separate measured run.

## Rotational pressure convention

The rotational identity is
`(u·∇)u = omega × u + ∇(|u|²/2)`.
Scheme 5 therefore stores **modified-pressure** Stokes observations in `L`.
Its returned `gradient` subtracts the kinetic-energy gradient computed from the
same mixed derivatives, and represents physical pressure gradient.
`initial_rotational()`, when provided, must supply modified initial `L`.
Otherwise `initial()` must already use that convention for this scheme.
Discrete derivative maps need not satisfy an exact product rule, so rotational
and advective discretizations can differ even after nonlinear convergence.

## Characteristic evaluation and boundary continuation

The problem supplies `evaluate(U,L,t,points)` using the mixed reconstruction.
Departure-point interpolation is a distinct error source from the time scheme.
When domain bounds are supplied, points outside those bounds require an explicit
`exterior_velocity(points,t)` continuation. A manufactured analytic continuation
is benchmark information; it is not a general inflow boundary algorithm.
Rectangular bounds alone cannot detect holes or other nonrectangular exclusions.
Compare exterior-evaluation counts as well as field errors.

## Experimental Oseen Hermite closure

Scheme 7 has a different observation convention: `L` is Oseen momentum,
`-nu Δu + b·∇u + ∇p`, with extrapolated nodal advector `b`.
At target `i`, the public functional API permits a frozen coefficient `b_i`.
The matching source observation at `j` is consequently

```
L_j + (b_i - b_j)_x X_j + (b_i - b_j)_y Y_j,
```

where `X,Y` are mixed-reconstructed derivatives. The assembled four-field
system closes `U,L,X,Y` using those corrected sources. Omitting the correction
would identify different differential functionals at target and source.
This construction rebuilds local maps per target and time step; its cost can
substantially exceed the fixed-map schemes. It remains experimental, with no
claimed stability, accuracy, or performance advantage. Static formula review
is separate from executing a native assembly or an accuracy benchmark.

## Problem protocol and diagnostics

The fixed-map modules accept `A,B,SL,M,PY,PB,PL` and
`DxU,DxB,DxL,DyU,DyB,DyL`, together with `force(t)`, `boundary(t)` and
`initial()` returning `(U,L)`. Transport additionally requires interior `points`
and its mixed `evaluate` callback for characteristic variants. Scheme 7 needs
the public-space, stencil, geometry, and assembly options used to rebuild maps.

Each module exposes `evolve(problem, scheme, dt, steps)` and returns `U,L`,
physical pressure `gradient`, and diagnostics. Nonlinear coupled schemes accept
`rtol`, `atol`, and `max_iterations`; unresolved nonlinear residuals raise errors.
For Oseen, the residual of its linear equation and the residual of the fully
nonlinear equation measure different things. A small constraint or solve
residual is not a field-accuracy estimate.

Sparse global solves use Float64 even if local weights were computed with MPFR.
Record geometry, kernel convention, augmentation, shared stencil, local precision,
boundary data, forcing, pressure convention, nonlinear tolerance, `dt`, final
time, and error definition for every comparison. Compare velocity and physical
pressure-gradient errors separately. Time refinement at fixed spatial maps can
reach a spatial-error floor; passing algebraic tests cannot establish a spatial
convergence rate or exclude nonnormal transient growth.

## Running the comparisons

For larger fixed-map comparisons, use a longer native timeout and the optional
preconditioned Oseen solver:

```powershell
.venv/Scripts/python.exe -m research.navier_stokes_schemes.run --cells 89 126 --ranks 8 --schemes oseen imex_ars222 --end-time 0.1 --native-timeout 14400 --linear-solver gmres --global-threads 4 --reuse-stencils --output outputs/ns_large_new
```

`gmres` solves the same coupled Oseen equations with an LU preconditioner,
refreshing it if iteration fails. The final linear system is also solved
directly to check velocity and pressure-gradient differences. This check
compares one system; `validate_linear.py --output <new.json>` compares complete
small-mesh trajectories with identical MPI maps. Direct solves remain the
default. ARS IMEX uses its fixed stage factorization independently of this flag.

For the fine-grid Float64 system, `--linear-solver gmres_scaled` also equilibrates
the global equations: it solves `R A C y = R b` and returns `x=C y`. Positive
diagonal row and column scales normalize the maximum entry in each row, then
each column. Scales and the LU preconditioner stay fixed within each time-step
coefficient family; backward Euler and BDF2 get separate factorizations. The
kernel h-scaling and mixed equations are unchanged. The stopping norm is that
of the transformed system (`rtol=1e-14`, `atol=1e-14`); original residuals and
per-solve iteration/factorization timings are saved too. The final physical
velocity and pressure-gradient checks against a direct solve remain enabled.
The extra final direct factorization is not included in the preconditioner LU
count, but its time is included in the case's total integration time.
`--global-ordering MMD_AT_PLUS_A` optionally changes the preconditioner's sparse
ordering (default `COLAMD`). The final comparison still uses the original
direct ordering. Lower fill in a measured benchmark is a performance result;
physical field parity is checked separately.

`--reuse-stencils` restores the earlier research builder's exact translation
and row-permutation reuse. It retains the same nearest 37 physical sites and
falls back to absolute coordinates whenever a Float64 displacement is not
exactly representable. No coordinates are rounded. Each unique template is
solved through public `LocalApproximation`, in MPI batches of 64, then scattered
to the original node-major layout. The kernels, full h-scaling and augmentation
are unchanged. This research optimization currently supports fixed-map schemes;
moving-point characteristic reconstruction keeps the original construction.

The driver saves sparse matrices and node coordinates in `maps_cells*/` before
time integration. Fixed-map runs then release the local reconstruction objects.
For this manufactured viscosity-1 study, `--maps-from <saved-run>` reloads these
maps and checks their node coordinates before time integration. Use a new
`--output` directory; no MPI weights are rebuilt. This allows global-solver
comparisons without repeating local work.
Local preparation includes serial work before MPI execution; increasing MPI
ranks does not parallelize that preparation. `--global-threads` controls BLAS
threads only during time integration. Local native weights use one thread per
MPI rank.

Run the fixed-map schemes first, using a new output directory:

```powershell
.venv/Scripts/python.exe -m research.navier_stokes_schemes.run --cells 6 8 --dt 0.00025 0.000125 --end-time 0.001 --ranks 2 --schemes oseen implicit oseen_newton imex imex_ars222 rotational characteristic characteristic_bdf2 --output outputs/ns_schemes_first6
```

Request scheme 7 separately after those runs finish:

```powershell
.venv/Scripts/python.exe -m research.navier_stokes_schemes.run --cells 6 8 --dt 0.00025 0.000125 --end-time 0.001 --ranks 2 --schemes adaptive_hermite --output outputs/ns_schemes_last7
```

These small meshes check implementation behavior. They are not the 8100-node
fine-grid study. The benchmark uses offset 0 and homogeneous velocity boundaries.
Its exact analytic initial observations need not satisfy the discrete mixed
constraint exactly; assessing startup consistency is separate from supplying
exact continuous initial data.

Generate a report from the saved directories without rerunning solvers:

```powershell
.venv/Scripts/python.exe -m research.navier_stokes_schemes.report --inputs outputs/ns_schemes_first6 outputs/ns_schemes_last7 --output outputs/ns_schemes_report
```
