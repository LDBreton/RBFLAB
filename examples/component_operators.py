"""Component expressions and explicit blocks describe the same local operator.

    python -m examples.component_operators
    python -m examples.component_operators --backend cpp --ranks 2
"""
import argparse
import numpy as np
import rbflab as r


def run(backend="python", ranks=0):
    # 1. Two scalar fields; the syntax is not specific to Stokes.
    points = r.geometry.unit_box_grid(2).points
    space = r.MixedSpace({
        "a": r.ScalarSpace(r.Gaussian(1), polynomial_degree=2),
        "b": r.ScalarSpace(r.Gaussian(2), polynomial_degree=2),
    })
    a, b = space.fields
    fields = r.vector(a, b)
    operator = r.vector(
        r.dx(fields[0]) + 2*r.dy(fields[1]),
        r.derivative(a, axis=0, order=2) - r.dx(r.dy(b)),
    )

    # 2. Equivalent explicit blocks: rows are outputs, columns are inputs.
    Dx, Dy = r.Derivative(0), r.Derivative(1)
    blocks = r.FieldOperator({"a": [[Dx], [Dx @ Dx]],
                              "b": [[2*Dy], [-(Dx @ Dy)]]})

    # 3. Both targets reuse the same local Hermite factorization.
    source = {"values": r.Samples(points, operator=fields, size=9)}
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=40),
        backend=(r.CppBackend() if backend == "cpp"
                 else r.PythonBackend()),
        execution=r.LocalExecution(scheduler="mpi", ranks=ranks) if ranks else None,
    )
    maps = local.operators(targets=points, operators={"expression": operator, "blocks": blocks})
    difference = maps.expression.to_scipy() - maps.blocks.to_scipy()
    print("Maximum weight difference:", abs(difference).max())

    # 4. Check only at mesh nodes: a=x^2+y, b=x*y.
    x, y = points.T
    data = {"values": np.c_[x*x+y, x*y]}
    numerical = np.asarray((maps.expression @ data).tolist(), dtype=float)
    exact = np.c_[4*x, np.ones(len(points))]
    print("Maximum absolute nodal component error:", abs(numerical-exact).max())
    return maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--ranks", type=int, default=0)
    args = parser.parse_args()
    if args.ranks and args.backend != "cpp":
        parser.error("MPI requires --backend cpp")
    run(args.backend, args.ranks)
