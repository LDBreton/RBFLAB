"""Unsteady Stokes: homogeneous walls, exact solution, and nodal errors.

Small example (requires the C++ toolchain):
    python -m examples.stokes_lhi_baseline

The two baseline meshes, with MPI local weights:
    python -m examples.stokes_lhi_baseline --cells 89 --ranks 8
    python -m examples.stokes_lhi_baseline --cells 126 --ranks 8

This teaching script calls LHI directly. Large runs construct every stencil;
the measured batch/checkpoint driver is in research/divfree_lhi_energy/.
"""
import argparse
from fractions import Fraction

import numpy as np
import sympy as sp
from threadpoolctl import threadpool_limits

import rbflab as r


def run(cells=8, ranks=1, end_time="0.1", backend="cpp", digits=80):
    # 1. Choose an exact divergence-free velocity and its pressure.
    model = r.SymbolicStokes(2, transient=True)
    u, p = model.fields
    x, y = model.coordinates
    t = model.time
    psi = 256*x**2*(1-x)**2*y**2*(1-y)**2
    exact_u = sp.exp(-t)*sp.Matrix([sp.diff(psi, y), -sp.diff(psi, x)])
    exact_p = sp.exp(-t)*(x-y)

    # 2. Derive the forcing for u_t - Laplacian(u) + grad(p) = f.
    momentum = sp.diff(u, t)-model.laplacian(u)+model.gradient(p)
    forcing = momentum.subs(dict(zip(u, exact_u)) | {p: exact_p}).doit()
    problem = model.evolution(
        [sp.Eq(momentum, forcing)],
        initial={u: exact_u.subs(t, 0)},
        boundary=[model.bc("boundary", sp.Eq(u, sp.zeros(2, 1)))],
    )

    # 3. Full h-scaling, with cbar_p = 1. Here s denotes r**2.
    q = Fraction(cells, 44)  # q = (1/44)/h, where h = 1/cells
    velocity_kernel = r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7)
    s = sp.Symbol("s", nonnegative=True)
    pressure_kernel = r.Kernel(q**4*sp.exp(-q*q*s), s)()
    spaces = {
        u: r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=6),
        p: r.PressureSpace(pressure_kernel, polynomial_degree=1),
    }

    # 4. Assemble once; reuse the same matrices for both time steps.
    numerical_backend = (r.CppBackend(threads=1, compute_condition=False, timeout=3600)
                         if backend == "cpp" else r.PythonBackend(compute_condition=False))
    execution = r.LocalExecution(scheduler="mpi", ranks=ranks) if ranks > 1 else None
    method = r.LHI(
        spaces=spaces, stencil_size=37,
        precision=r.Precision(local_digits=digits),
        stencil_policy=r.StencilPolicy(scaling="physical"),
        local_backend=numerical_backend, execution=execution,
    )
    cloud = r.geometry.unit_box_grid(cells)
    system = method.assemble(problem, cloud)

    # 5. Measure absolute vector errors only at interior mesh nodes.
    points = cloud.interior
    final_time = float(Fraction(end_time))
    truth = sp.lambdify((x, y), exact_u.subs(t, final_time), "numpy")
    velocity_exact = np.asarray(truth(points[:, 0], points[:, 1]), dtype=float).reshape(2, -1).T
    gradient_exact = np.tile(np.exp(-final_time)*np.array([1., -1.]), (len(points), 1))
    results = []
    for dt in ("0.00025", "0.000125"):
        steps = Fraction(end_time)/Fraction(dt)
        if steps.denominator != 1 or steps < 1:
            raise ValueError("end_time must be a positive multiple of both time steps")
        solution = system.solve(dt=dt, steps=int(steps), time_scheme="bdf2").states[-1]
        velocity_error = np.linalg.norm(solution.velocity(points)-velocity_exact, axis=1)
        gradient_error = np.linalg.norm(solution.pressure_gradient(points)-gradient_exact, axis=1)
        result = {
            "dt": float(dt),
            "velocity_rms": float(np.sqrt(np.mean(velocity_error**2))),
            "gradient_rms": float(np.sqrt(np.mean(gradient_error**2))),
            "velocity_max": float(velocity_error.max()),
            "gradient_max": float(gradient_error.max()),
        }
        results.append(result)
        print(f"nodes={len(cloud.points)}, dt={dt}: "
              f"velocity RMS={result['velocity_rms']:.6e}, "
              f"pressure-gradient RMS={result['gradient_rms']:.6e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--backend", choices=("python", "cpp"), default="cpp")
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--end-time", default="0.1")
    args = parser.parse_args()
    if args.ranks > 1 and args.backend != "cpp":
        parser.error("MPI local weights require --backend cpp")
    with threadpool_limits(limits=1):
        run(args.cells, args.ranks, args.end_time, args.backend, args.digits)
