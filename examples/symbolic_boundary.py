"""Spatially varying diffusion and a Robin condition on the top face."""
import numpy as np
import sympy as sp
import rbflab as r


def run():
    model = r.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.exp(x) * (1 + y)
    diffusivity = 1 + x*x
    lhs = -sum(sp.diff(diffusivity * sp.diff(u, q), q) for q in (x, y)) + u
    robin = (2 + x) * u + model.normal_derivative(u)
    problem = model.stationary(
        sp.Eq(lhs, lhs.subs(u, exact).doit()),
        boundary=[
            model.bc("top", sp.Eq(robin, robin.subs(u, exact).doit())),
            model.bc("boundary", sp.Eq(u, exact)),
        ],
    )
    cloud = r.unit_box_grid(5)
    solution = problem.solve(cloud, r.GlobalCollocation(r.IMQ(2), scheme="asymmetric"))
    query = np.random.default_rng(12).uniform(.1, .9, (30, 2))
    expected = np.exp(query[:, 0]) * (1 + query[:, 1])
    error = float(np.max(np.abs(solution.evaluate(query) - expected)))
    print(f"variable diffusion + Robin: off-node max error {error:.3e}")
    return error


if __name__ == "__main__":
    run()
