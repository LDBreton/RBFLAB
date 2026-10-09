"""Unforced heat equation with a manufactured, time-dependent solution."""
import numpy as np
import sympy as sp
import rbflab as r


def run(dt="0.01", steps=5):
    model = r.SymbolicScalar(2, transient=True)
    u, t = model.field, model.time
    x, y = model.coordinates
    exact = sp.exp(-2*sp.pi**2*t)*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
    lhs = sp.diff(u, t) - model.laplacian(u)
    problem = model.evolution(
        sp.Eq(lhs, 0), initial=exact.subs(t, 0),
        boundary=[model.bc("boundary", sp.Eq(u, 0))],
    )
    cloud = r.geometry.unit_box_grid(5)
    trajectory = problem.solve(cloud, r.GlobalCollocation(r.IMQ(2), scheme="asymmetric"), dt, steps)
    query = np.random.default_rng(4).uniform(.1, .9, (30, 2))
    final_time = float(sp.Rational(dt) * steps)
    expected = np.exp(-2*np.pi**2*final_time)*np.sin(np.pi*query[:, 0])*np.sin(np.pi*query[:, 1])
    error = float(np.max(np.abs(trajectory.final.evaluate(query)-expected)))
    print(f"heat at t={final_time:g}: off-node max error {error:.3e}")
    return error


if __name__ == "__main__":
    run()
