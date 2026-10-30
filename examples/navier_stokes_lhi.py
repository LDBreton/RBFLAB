"""Mixed LHI Navier-Stokes with prescribed time-dependent boundary velocity, using only public APIs.

    python -m examples.navier_stokes_lhi --cells 8 --ranks 2

Requires the optional C++/MPFR toolchain and MPI launcher. The small default
mesh demonstrates the method; it is not the fine-grid accuracy benchmark.
All vectors use node-major order: [ux_0, uy_0, ux_1, uy_1, ...].
"""
import argparse

import numpy as np
import sympy as sp
from scipy.sparse import eye
from scipy.sparse.linalg import factorized
from threadpoolctl import threadpool_limits

import rbflab as r


def assemble(cells, nu, ranks):
    # 1. One shared 37-site physical stencil, partitioned by sample group.
    cloud = r.geometry.unit_box_grid(cells)
    points, targets = cloud.points, cloud.interior
    boundary = points[cloud.boundary_indices]
    stencil = r.Stencil(points, size=37)

    # 2. Full h-scaling; the kernels act on physical coordinates.
    q = sp.Rational(cells, 44)
    velocity_kernel = r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7)
    s = sp.Symbol("s", nonnegative=True)  # squared distance
    pressure_kernel = r.Kernel(q**4*sp.exp(-q*q*s), s)()
    space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=6),
        "p": r.PressureSpace(pressure_kernel, polynomial_degree=1),
    })
    u, p = space.fields
    value = u
    momentum = -nu*r.laplacian(u) + r.gradient(p)
    pressure_gradient = r.gradient(p)
    options = dict(backend=r.CppBackend(threads=1, timeout=3600),
                   execution=r.LocalExecution(scheduler="mpi", ranks=ranks),
                   precision=r.Precision(local_digits=80))

    # 3. Separate interior, boundary and neighboring momentum observations.
    source = {
        "velocity": r.Samples(targets, operator=u, within=stencil),
        "boundary": r.Samples(boundary, operator=u, within=stencil),
        "momentum": r.Samples(targets, operator=momentum,
                              within=stencil, target="exclude"),
    }
    local = r.LocalApproximation(source=source, trial=space.representers(source), **options)
    maps = local.operators(targets=targets,
                           operators={"L": momentum, "P": pressure_gradient})

    # M U_t + A U = M(f - C) - B g(t).
    A = maps.L["velocity"].to_scipy()
    B = maps.L["boundary"].to_scipy()
    SL = maps.L["momentum"].to_scipy()
    M = eye(A.shape[0], format="csr") - SL
    PY = maps.P["velocity"].to_scipy()
    PB = maps.P["boundary"].to_scipy()
    PL = maps.P["momentum"].to_scipy()

    # 4. Separate degree-5 divergence-free, value-only reconstruction.
    # Both components are coupled; these are not scalar differentiation maps.
    convection_space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=5)})
    values = {
        "velocity": r.Samples(targets, operator=value, within=stencil),
        "boundary": r.Samples(boundary, operator=value, within=stencil),
    }
    reconstruction = r.LocalApproximation(
        source=values, trial=convection_space.representers(values), **options)
    derivatives = reconstruction.operators(targets=targets, operators={
        "dx": r.dx(u),
        "dy": r.dy(u),
    })
    Dx = derivatives.dx["velocity"].to_scipy()
    Dxb = derivatives.dx["boundary"].to_scipy()
    Dy = derivatives.dy["velocity"].to_scipy()
    Dyb = derivatives.dy["boundary"].to_scipy()
    print(f"Built mixed and convection maps for {len(points)} nodes with {ranks} MPI ranks.")
    return targets, boundary, M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb


def manufactured_solution(points, boundary, nu, offset=1):
    # 5. Exact velocity, pressure gradient and Navier-Stokes forcing.
    x, y = sp.symbols("x y")
    psi = 256*x**2*(1-x)**2*y**2*(1-y)**2
    v = sp.Matrix([sp.diff(psi, y) + offset, -sp.diff(psi, x) - offset])
    lap = v.diff(x, 2) + v.diff(y, 2)
    nonlinear = v.jacobian([x, y])*v

    def sample(expressions, nodes=points):
        return np.column_stack([np.broadcast_to(
            sp.lambdify((x, y), e, "numpy")(*nodes.T), (len(nodes),))
            for e in expressions]).ravel()

    U0 = sample(v)
    G0 = sample(sp.Matrix([1, -1]))
    F1 = sample(-v - nu*lap + sp.Matrix([1, -1]))
    F2 = sample(nonlinear)
    forcing = lambda t: np.exp(-t)*F1 + np.exp(-2*t)*F2
    boundary_initial = sample(v, boundary)
    boundary_velocity = lambda t: np.exp(-t)*boundary_initial
    return U0, G0, forcing, boundary_velocity


def evolve(M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb,
           U0, forcing, boundary_velocity, dt, steps):
    # 6. Implicit diffusion/pressure, explicit extrapolated convection.
    def convection(U, t):
        v = U.reshape(-1, 2)
        g = boundary_velocity(t)
        ux = (Dx @ U + Dxb @ g).reshape(-1, 2)
        uy = (Dy @ U + Dyb @ g).reshape(-1, 2)
        return (v[:, :1]*ux + v[:, 1:]*uy).ravel()

    be = factorized((M/dt + A).tocsc())
    bdf2 = factorized((1.5*M/dt + A).tocsc())
    U = U0.copy()
    C = convection(U, 0)
    for k in range(1, steps+1):
        if k == 1:
            Cstar, history, alpha, solve = C, U/dt, 1/dt, be
        else:
            Cstar = 2*C - C_previous
            history, alpha, solve = (2*U - .5*U_previous)/dt, 1.5/dt, bdf2
        force = forcing(k*dt) - Cstar
        g = boundary_velocity(k*dt)
        U_new = solve(M @ (history + force) - B @ g)
        U_t = alpha*U_new - history
        grad_p = PY @ U_new + PB @ g + PL @ (force - U_t)
        if not (np.isfinite(U_new).all() and np.isfinite(grad_p).all()):
            raise FloatingPointError(f"Nonfinite field at step {k}")
        U_previous, U = U, U_new
        C_previous, C = C, convection(U_new, k*dt)
    return U, grad_p


def run(cells=8, nu=1.0, ranks=2, end_time="0.1", offset=1):
    # 7. Reuse weights for both time steps; errors only at interior mesh nodes.
    duration = sp.Rational(end_time)
    dts = (sp.Rational("0.00025"), sp.Rational("0.000125"))
    if any(duration/dt < 1 or (duration/dt).denominator != 1 for dt in dts):
        raise ValueError("end_time must be a positive multiple of both time steps")
    points, boundary, M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb = assemble(cells, nu, ranks)
    U0, G0, forcing, boundary_velocity = manufactured_solution(points, boundary, nu, offset)
    results = []
    for dt in dts:
        U, G = evolve(M, A, B, PY, PB, PL, Dx, Dxb, Dy, Dyb,
                      U0, forcing, boundary_velocity, float(dt), int(duration/dt))
        row = {"dt": float(dt)}
        for name, field, exact in (("velocity", U, U0), ("gradient", G, G0)):
            error = (field - np.exp(-float(duration))*exact).reshape(-1, 2)
            lengths = np.linalg.norm(error, axis=1)
            row[name + "_rms"] = float(np.sqrt(np.mean(lengths**2)))
            row[name + "_max"] = float(lengths.max())
        print(row, flush=True)
        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--end-time", default="0.1")
    parser.add_argument("--offset", type=float, default=1,
                        help="Constant velocity offset; 0 recovers homogeneous walls")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells, args.nu, args.ranks, args.end_time, args.offset)
