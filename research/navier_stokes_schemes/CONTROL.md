# Navier–Stokes through Stokes control solves

This research experiment uses the same nearest-37 divergence-free mixed LHI
maps for velocity, Stokes momentum and convection derivatives. Pressure is
eliminated locally. No finite-element velocity/pressure representation is added.

For a prescribed control f, solve the linear state equation

    u_t - nu*Delta(u) + grad(p) = b - f,
    div(u) = 0,  u|boundary = g,  u(0) = u0.

The control objective is

    J(f) = 0.5 * integral |f - (u(f).grad)u(f)|^2 dx dt.

The Dirichlet and initial data are fixed. Their variations are zero; an
equivalent unrestricted Lagrangian includes boundary and initial multipliers.
The control itself is not constrained to be divergence-free.

## Discrete state and objective

Let U contain interior nodal velocities and L contain the local Stokes momentum
observations. Both are node-major vectors, with x/y components interleaved.
The saved mixed maps impose

    A U + B g - M L = 0,     M = I - SL.

Backward Euler gives

    K x[n] = H x[n-1] + d[n] - E f[n],

where x=(U,L), K=[[A,-M],[I/dt,I]], H=[[0,0],[I/dt,0]],
E=[[0],[I]], and d=(-B g,b). Only the old velocity is a time-history variable.

Compute nodal gradients from this same mixed reconstruction:

    dx = DxU @ U + DxB @ g + DxL @ L
    dy = DyU @ U + DyB @ g + DyL @ L
    C  = repeat(U[::2], 2)*dx + repeat(U[1::2], 2)*dy

For residual r=f-C and positive nodal weights W, the discrete objective is

    J_h = 0.5 * dt * sum_n r[n].T @ W @ r[n].

The cavity runs use h^2 per interior node, repeated for the two components.
This is an interior nodal quadrature, not an exact domain integral.

## The actual discrete adjoint

The reverse-time recurrence is

    K.T lambda[n] = C'[n].T @ (dt*W*r[n]) + H.T lambda[n+1],
    lambda[N+1] = 0,
    gradient_Euclidean[n] = dt*W*r[n] + E.T lambda[n].

The weighted L2 gradient divides this derivative by dt*W. Iterating n from N
down to 1 is a forward loop in reversed time. The transpose is essential:
the LHI matrices are not presumed symmetric. The convection Jacobian includes
both the changing advector U and the changing reconstructed derivatives,
including their dependence on L. Boundary data are held fixed during variations.

One sparse factorization of A+M/dt solves both K and K.T by exact block
elimination. The factorization is reused across control iterations and time
windows. The forward and adjoint solves remain Float64 global algebra.

## Optimization and time windows

`StokesControl` supports a full chosen time window. Matrix-free Gauss–Newton,
L-BFGS and a damped Newton–Krylov residual solve use the same objective. For
a nonsingular square residual Jacobian, solving J_r*delta=-r gives the
Gauss–Newton direction without forming normal equations. Its line search uses
the objective and the actual adjoint gradient. No dense control Jacobian is built.

The production cavity driver uses one time step per control window, warm-started
from the last control. When each residual vanishes, the assembled trajectory
solves the fully implicit backward-Euler Navier–Stokes equations. This does not
approximate Oseen's frozen convection. Four-step windows separately validate
the genuine reverse-time recurrence by finite differences on both large grids.

The driver requires the physical residual infinity norm to be at most 1e-8.
Optimizer status alone is insufficient. A speed bound of 10 stops a run and
preserves the candidate; it is a diagnostic bound, not a blowup theorem.

## Spatial baseline and MPI

Meshes have 8,100 and 16,129 total nodes (89 and 126 cells per side), with
degree-6 divergence-free velocity and degree-1 pressure augmentation.
With q=cells/44, velocity uses exp(-10*q^2*r^2)+q^7*r^7. For nu=.01 the
pressure potential is nu^2*q^4*exp(-q^2*r^2), with cbar_p=1.

These maps are obtained by an exact viscosity rescaling of saved public-API
MPI8/MPFR80 viscosity-one maps. A,B,PY,PB multiply by nu; DxL,DyL divide by nu;
M,SL,PL and velocity/boundary differentiation blocks remain unchanged. Fresh
MPI8/MPFR80 builds of 25 selected owners on each large mesh check this identity,
including wall, corner-adjacent and interior stencils. This is reuse plus new
MPI validation, not a claim that every large-grid weight was rebuilt in this run.

The cavity has a unit tangential lid, stationary corner nodes and other walls,
zero physical body force, and initial rest. No corner subtraction or quadratic
pressure-consistency correction is added. Thus this study preserves the raw
Stokes baseline and must separately report its pressure-consistency limitations.

## Interpretation and reproduction

A small objective means a small nonlinear equation residual. It does not prove
spatial accuracy, eliminate a spurious spatial mode, or establish long-time
stability. Homogeneous Stokes probes, gradient checks and physical cavity
evolution are reported separately.

All solution comparisons are nodal. The 8,100-node grid contains no exact Ghia
centerline nodes; its Ghia metric is marked unavailable. On 16,129 nodes, the
reference table is interpolated onto existing centerline nodes, while the
computed field is left untouched. Short-time differences from steady Ghia data
are not steady-solution errors.

Use `python -m research.navier_stokes_schemes.control_cavity --help` for runs.
Each output records its source hashes, time step, objective, optimizer settings,
accepted-step history, checkpoint, endpoint fields and failure if present.
The experiment lives in `outputs/ns_stokes_control_20261030/`.

This implementation is research orchestration using public-API-built LHI maps;
it does not add a public RBFLAB optimal-control solver API.
