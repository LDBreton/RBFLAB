"""Steady and unsteady Stokes with an explicit divergence-free velocity space.

Run: python -m examples.tutorials.stokes_spaces
"""
import numpy as np
import sympy as sp
import rbflab as rbf


def make_problem(transient):
    model = rbf.SymbolicSystem(
        2, vector_fields=("U",), scalar_fields=("p",), transient=transient,
    )
    U, p = model.fields
    x, y = model.coordinates
    velocity = sp.ImmutableMatrix([y*(1-y), 0])
    if transient:
        velocity = sp.exp(-model.time)*velocity
    pressure = -2*x
    momentum = -model.laplacian(U) + model.gradient(p)
    if transient:
        momentum += sp.diff(U, model.time)
    forcing = momentum.subs(dict(zip(U, velocity)) | {p: pressure}).doit()
    boundary = [model.bc("boundary", sp.Eq(U, velocity))]
    if transient:
        problem = model.evolution(
            [sp.Eq(momentum, forcing)],
            initial={U: velocity.subs(model.time, 0)},
            boundary=boundary,
        )
    else:
        problem = model.stationary([sp.Eq(momentum, forcing)], boundary=boundary)
    spaces = {
        U: rbf.DivergenceFreeSpace(rbf.IMQ(2), 2),
        p: rbf.PressureSpace(rbf.IMQ(2), 1),
    }
    return problem, spaces


def run(cells=4):
    cloud = rbf.unit_box_grid(cells)
    query = np.array([[0.3, 0.4]])
    results = {}
    for transient in (False, True):
        problem, spaces = make_problem(transient)
        method = rbf.GlobalCollocation(spaces=spaces)
        system = method.assemble(problem, cloud)
        solution = (system.solve("0.01", 3, scheme="bdf2").final
                    if transient else system.solve())
        time = 0.03 if transient else 0.0
        expected_velocity = np.array([[np.exp(-time)*0.4*0.6, 0.]])
        velocity_error = float(np.max(np.abs(solution.velocity(query)-expected_velocity)))
        gradient_error = float(np.max(np.abs(solution.pressure_gradient(query)-[-2., 0.])))
        divergence = float(np.max(np.abs(solution.divergence(query))))
        name = "unsteady" if transient else "steady"
        results[name] = (velocity_error, gradient_error, divergence)
        print(f"{name}: system={system.matrix.shape}, velocity_error={velocity_error:.3e}, "
              f"pressure_gradient_error={gradient_error:.3e}, divergence={divergence:.3e}")
    return results


if __name__ == "__main__":
    run()
