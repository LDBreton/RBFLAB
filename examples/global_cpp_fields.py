"""Native global divergence-free Stokes with velocity and pressure fields.

Run: python -m examples.global_cpp_fields --dimension 3 --digits 60
The quadratic solution checks polynomial reproduction, not general PDE accuracy.
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as r


def run(dimension=2, digits=None, cells=2):
    model = r.SymbolicStokes(dimension)
    U, p = model.fields
    x = model.coordinates
    velocity = sp.ImmutableMatrix([x[(j+1) % dimension]**2 for j in range(dimension)])
    pressure = sum((j+1)*coordinate for j, coordinate in enumerate(x))
    momentum = -model.laplacian(U)+model.gradient(p)
    forcing = momentum.subs(dict(zip(U, velocity)) | {p: pressure}).doit()
    problem = model.stationary(
        [sp.Eq(momentum, forcing)],
        boundary=[model.bc('boundary', sp.Eq(U, velocity))],
    )
    method = r.GlobalCollocation(
        spaces={U: r.DivergenceFreeSpace(r.IMQ('.7'), polynomial_degree=2),
                p: r.PressureSpace(r.Gaussian('1.3'), polynomial_degree=1)},
        precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(threads=2, compute_condition=False),
    )
    cloud = r.geometry.unit_box_grid(cells, dimension)
    queries = np.random.default_rng(37).uniform(.1, .9, (5, dimension))
    with method.assemble(problem, cloud) as system:
        solution = system.solve()
        extended = digits is not None
        actual = solution.velocity(queries, extended=extended)
        gradient = solution.pressure_gradient(queries, extended=extended)
        gauged = solution.pressure(queries, reference=(np.zeros(dimension), 0), extended=extended)
        divergence = solution.divergence(queries, extended=extended)
        residual = solution.residuals(queries, extended=extended)
        a = system.arithmetic
        expected = [a.data(model.data(v), queries) for v in velocity]
        expected_p = a.data(model.data(pressure), queries)
        errors = dict(
            velocity=max(abs(actual[i, j]-expected[j][i]) for i in range(len(queries)) for j in range(dimension)),
            pressure_gradient=max(abs(gradient[i, j]-(j+1)) for i in range(len(queries)) for j in range(dimension)),
            pressure=max(abs((gauged[i, 0] if extended else gauged[i])-expected_p[i]) for i in range(len(queries))),
            divergence=max(abs(v) for v in divergence),
            momentum=max(abs(residual[i, j]) for i in range(len(queries)) for j in range(dimension)),
        )
        print(f'{dimension}D, {digits or "double"}, {len(cloud.points)} nodes, {system.size} unknowns')
        for name, error in errors.items():
            print(f'{name}: {a.ctx.nstr(error, 8) if extended else format(error, ".3e")}')
        return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, choices=(2, 3), default=2)
    parser.add_argument('--digits', type=int, default=None)
    parser.add_argument('--cells', type=int, default=2)
    args = parser.parse_args()
    run(args.dimension, args.digits, args.cells)
