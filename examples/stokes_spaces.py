"""Steady Stokes with a divergence-free velocity space and symbolic momentum.

Run: python examples/stokes_spaces.py --method global
     python examples/stokes_spaces.py --method lhi --backend python
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as r


def make_solution(method_name="global", backend_name="python", local_digits=None):
    model = r.SymbolicStokes(2,vector_fields=("U",), scalar_fields=("p",))
    U, p = model.fields
    x, y = model.coordinates
    velocity = sp.ImmutableMatrix([y*(1-y), 0])
    pressure = -2*x
    momentum = -model.laplacian(U) + model.gradient(p)
    forcing = momentum.subs(dict(zip(U, velocity)) | {p: pressure}).doit()
    problem = model.stationary(
        [sp.Eq(momentum, forcing)],
        boundary=[model.bc("boundary", sp.Eq(U, velocity))],
    )
    spaces = {
        U: r.DivergenceFreeSpace(r.IMQ(2), 2),
        p: r.PressureSpace(r.IMQ(2), 1),
    }
    if method_name == "global":
        if local_digits is not None:
            raise ValueError("--digits configures LHI local weights; global precision uses global_digits")
        if backend_name != "python":
            raise ValueError("This global space method currently uses its Python assembler")
        method = r.GlobalCollocation(spaces=spaces)
    else:
        backends = {
            "python": lambda: r.PythonBackend(compute_condition=False),
            "cpp": lambda: r.CppBackend(threads=2, compute_condition=False),
        }
        method = r.LHI(spaces=spaces, stencil_size=15,
                       precision=r.Precision(local_digits=local_digits),
                       local_backend=backends[backend_name]())
    return problem.solve(r.geometry.unit_box_grid(4), method)


def run(method_name="global", backend_name="python", local_digits=None):
    solution = make_solution(method_name, backend_name, local_digits)
    query = np.array([[.3, .4]])
    velocity_error = float(np.max(np.abs(solution.velocity(query) - [0.24, 0.])))
    pressure_gradient_error = float(np.max(np.abs(solution.pressure_gradient(query) - [-2., 0.])))
    divergence = float(np.max(np.abs(solution.divergence(query))))
    precision = f"local MPFR {local_digits} digits; global Float64" if local_digits else "Float64"
    print(f"{method_name}/{backend_name} ({precision}): velocity error {velocity_error:.3e}; "
          f"pressure-gradient error {pressure_gradient_error:.3e}; divergence {divergence:.3e}")
    return velocity_error, pressure_gradient_error, divergence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("global", "lhi"), default="global")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--digits", type=int, help="LHI local decimal digits (global sparse solve stays Float64)")
    args = parser.parse_args()
    run(args.method, args.backend, args.digits)
