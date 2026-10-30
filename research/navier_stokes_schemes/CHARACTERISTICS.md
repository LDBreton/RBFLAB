# First-order characteristics with corrected mixed LHI

This research implementation keeps the existing nearest-37 mixed divergence-free LHI reconstruction and local pressure elimination. It changes the time treatment of convection. It is not a new public RBFLAB solver or a claim that the cavity spatial discretization is stable.

For the old nodal velocity `U` and mixed momentum observation `L`, each interior arrival node traces one explicit-Euler foot:

```python
feet = points - dt * U.reshape(-1, 2)
transported = evaluator.evaluate(U, L, boundary(t), feet).ravel()

# A Unew + B gnew - M Lnew = 0
# (Unew - transported)/dt + Lnew = force(t + dt)
matrix = bmat([[A, -M], [I / dt, I]], format="csc")
rhs = concatenate((-B @ boundary(t + dt), force(t + dt) + transported / dt))
Unew, Lnew = split(lu.solve(rhs), 2)
```

The LU factorization is reused while the time step and spatial maps are fixed. The implementation does not form a convection derivative matrix or perform Newton iterations for this characteristic step. The `be_oseen` comparison uses backward Euler at every step and freezes the advector at the old nodal velocity. Historical BDF2-Oseen comparisons are labeled separately.

## The departure evaluator

The evaluator uses the exact same arrival-owner observations as the saved spatial maps. It does not select a new stencil around the foot. Interior momentum observations still exclude the original arrival target; prescribed boundary values are separate observations.

The actual corrected local field is

    F = F0 + (Q - F0 C) E,     C = S Q,     E C = I.

For data `d`, evaluate it as `F0(d - C E d) + Q E d`. The common correction reproduces divergence-free velocity polynomials through degree 6 and pressure polynomials through degree 2. The underlying kernel system uses DF6/P1. Local pressure constants remain unobservable.

`characteristic_reconstruction.py` builds reusable, scaled MPFR80 LU factors through eight MPI ranks, using RBFLAB's mixed kernel machinery. A persistent native worker then solves local coefficients and evaluates the analytic kernel representers in MPFR80 with eight OpenMP threads. Neither the coefficients nor the cancellation-sensitive contraction is reduced to Float64; only the observations supplied by the global solver and the final field values are Float64. No Taylor approximation, independent interpolation kernel or exact cavity solution is substituted for the field.

The fitted-field cache/evaluator is research code using RBFLAB internals. Public `LocalApproximation` independently validates selected off-node evaluations, but the public API does not currently expose this reusable native fitted-field worker.

## Walls, diagnostics and checkpoints

Only interior nodes are traced. New wall data enter `B @ gnew` directly. A foot outside the physical square rejects the step with its owner and coordinates recorded; there is no clipping or silent exterior extension. A numerical speed bound stops unstable runs and preserves the triggering finite state. This bound is a diagnostic stopping rule, not a blowup theorem.

All reported field errors and Ghia comparisons are nodal. Off-node values are used to advance the method and to validate its reconstruction, not to redefine the requested solution-error norm. Reference Ghia values are linearly interpolated to the existing centerline mesh nodes.

A semidiscrete steady root is generally not an exact fixed point of the characteristic time step. Therefore a perturbation run is paired with an unperturbed run of the same scheme and time step. Their difference measures perturbation amplification; distance from the semidiscrete root also includes time-discretization drift. Both quantities are retained. Physical-rest startup is a separate test, and its transient Ghia differences are not steady benchmark errors.

Saved configurations hash the raw/corrected maps, coefficients and initial fields. Restarts reject changed inputs. Checkpoints preserve the state and absolute step; trajectories store every accepted velocity and momentum vector for matched-time comparisons.

## Files

- `characteristic_cavity.py`: first-order time step, comparisons and checkpointed experiment CLI.
- `characteristic_reconstruction.py`: MPI-built native factors and persistent exact mixed-field evaluation.
- `tests/test_characteristic_cavity.py`: algebra, boundary, restart and factor-reuse checks.
- `tests/test_characteristic_reconstruction.py`: evaluator contracts and opt-in native parity.
- `tests/test_characteristic_audit.py`: independent transport/pressure/consistency checks.
- `outputs/ns_characteristic_audit_20261030/RESULTS.md`: actual two-grid reconstruction validation.
- `outputs/ns_characteristic_comparison_20261030/`: experiment outputs and comparisons.

Use `python -m research.navier_stokes_schemes.characteristic_cavity --help` for the driver options. Run from the repository root with the configured scientific Python environment. The raw baseline, corrected maps, correction parameters and initial fields must belong to the same mesh and configuration; the driver checks their consistency before characteristic evolution.
