"""Native scalar global collocation: python -m examples.global_cpp --digits 60."""
import argparse

import numpy as np
import sympy as sp

import rbflab as rbf


def run(*, scheme="symmetric", digits=None, dimension=2, threads=2):
    cloud = rbf.geometry.unit_box_grid(2, dimension)
    model = rbf.SymbolicScalar(dimension)
    u = model.field
    exact = 1 + sum((i + 1) * x**2 for i, x in enumerate(model.coordinates))
    equation = -model.laplacian(u) + u
    problem = model.stationary(
        sp.Eq(equation, equation.subs(u, exact).doit()),
        boundary=[model.bc("boundary", sp.Eq(u, exact))],
    )
    method = rbf.GlobalCollocation(
        rbf.PHS(5), scheme=scheme, polynomial_degree=2,
        precision=rbf.Precision(global_digits=digits),
        backend=rbf.CppGlobalBackend(threads=threads),
    )
    query = np.full((1, dimension), .375)
    truth = model.data(exact)
    with method.assemble(problem, cloud) as system:
        solution = system.solve()
        if digits is None:
            error = float(np.max(np.abs(solution.evaluate(query) - truth(query))))
            assert error < 1e-9
            display = str(error)
        else:
            values = solution.evaluate_mp(query)
            ctx = values.ctx
            error = max(abs(a - b) for a, b in zip(values, truth.mp_values(ctx, query)))
            assert error < ctx.mpf(10) ** (-digits + 12)
            display = ctx.nstr(error, 8)
        print(f"{scheme}, dimension={dimension}, digits={digits or 'double'}")
        print(f"quadratic reproduction error: {display}")
        print(solution.diagnostics)
    return display


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("symmetric", "asymmetric"), default="symmetric")
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--threads", type=int, default=2)
    run(**vars(parser.parse_args()))
