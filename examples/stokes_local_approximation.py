"""Unsteady Stokes with LocalApproximation: a small example for API review.

Run: python -m examples.stokes_local_approximation
Requires the C++/MPFR toolchain and MPI. Errors are measured only at nodes.
"""
import numpy as np
import sympy as sp
from scipy.sparse import eye
from scipy.sparse.linalg import factorized
from threadpoolctl import threadpool_limits

import rbflab as r


def main():
    # 1. Settings: a small demonstration mesh, not an accuracy benchmark.
    cells, ranks = 8, 2
    nu, dt, steps = 1, 0.00025, 400
    cloud = r.geometry.unit_box_grid(cells)
    points, targets = cloud.points, cloud.interior
    boundary = cloud.points[cloud.boundary_indices]

    # 2. Define velocity and pressure spaces; kernels are already h-scaled,
    #    matching LocalApproximation's default physical kernel scaling.
    q = sp.Rational(cells, 44)
    s = sp.Symbol("s", nonnegative=True)  # squared distance
    velocity_kernel = r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7)
    pressure_kernel = r.Kernel(q**4*sp.exp(-q*q*s), s)()
    space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=6),
        "p": r.PressureSpace(pressure_kernel, polynomial_degree=1),
    })

    # 3. Describe velocity values, momentum, and pressure gradient.
    u, p = space.fields
    velocity = u
    momentum = -nu * r.laplacian(u) + r.gradient(p)
    gradient = r.gradient(p)

    # One geometric stencil, shared by all three observation groups.
    stencil = r.Stencil(points, size=37)
    source = {
        "velocity": r.Samples(targets, operator=u, within=stencil),
        "boundary": r.Samples(boundary, operator=u, within=stencil),
        "momentum": r.Samples(targets, operator=momentum,
                              within=stencil, target="exclude"),
    }
    local = r.LocalApproximation(
        source=source,
        trial=space.representers(source),
        precision=r.Precision(local_digits=80),
        backend=r.CppBackend(threads=1, timeout=3600),
        execution=r.LocalExecution(scheduler="mpi", ranks=ranks),
    )
    maps = local.operators(targets=targets, operators={
        "momentum": momentum, "gradient": gradient,
    })

    # 4. Assemble M U_t + A U = M f - B g(t).
    # Pressure gradient: PY U + PB g(t) + PL (f - U_t).
    # Vectors are node-major: [ux_0, uy_0, ux_1, uy_1, ...].
    A = maps.momentum["velocity"].to_scipy()
    B = maps.momentum["boundary"].to_scipy()
    SL = maps.momentum["momentum"].to_scipy()
    M = eye(A.shape[0], format="csr") - SL
    PY = maps.gradient["velocity"].to_scipy()
    PB = maps.gradient["boundary"].to_scipy()
    PL = maps.gradient["momentum"].to_scipy()

    # 5. u = exp(-t) * (curl(psi) + [1, -1]), p = exp(-t)*(x-y).
    # The constant spatial offset gives a nonzero, time-dependent boundary.
    x, y = sp.symbols("x y")
    psi = 256*x**2*(1-x)**2*y**2*(1-y)**2
    v = sp.Matrix([sp.diff(psi, y) + 1, -sp.diff(psi, x) - 1])
    force = -v - nu*(v.diff(x, 2) + v.diff(y, 2)) + sp.Matrix([1, -1])

    def sample(expressions, nodes=targets):
        return np.column_stack([np.broadcast_to(
            sp.lambdify((x, y), e, "numpy")(*nodes.T), (len(nodes),))
            for e in expressions]).ravel()

    U0, F0 = sample(v), sample(force)
    G0 = sample(sp.Matrix([1, -1]))
    boundary_initial = sample(v, boundary)

    def boundary_velocity(t):
        # Modify this along with forcing/initial data for a different problem.
        return np.exp(-t)*boundary_initial

    # 6. Backward-Euler startup, then BDF2. The sparse solve is serial.
    be = factorized((M/dt + A).tocsc())
    bdf2 = factorized((1.5*M/dt + A).tocsc())
    U = U0.copy()
    for step in range(1, steps + 1):
        F = np.exp(-step*dt)*F0
        g = boundary_velocity(step*dt)
        if step == 1:
            history, alpha, solve = U/dt, 1/dt, be
        else:
            history = (2*U - 0.5*U_previous)/dt
            alpha, solve = 1.5/dt, bdf2
        U_new = solve(M @ (F + history) - B @ g)
        U_t = alpha*U_new - history
        grad_p = PY @ U_new + PB @ g + PL @ (F - U_t)
        U_previous, U = U, U_new

    # 7. Absolute vector RMS errors, only at interior mesh nodes.
    for name, numerical, initial in (("velocity", U, U0), ("gradient", grad_p, G0)):
        error = (numerical - np.exp(-steps*dt)*initial).reshape(-1, 2)
        rms = np.sqrt(np.mean(np.sum(error**2, axis=1)))
        print(f"{name}: absolute nodal vector RMS = {rms:.6e}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
