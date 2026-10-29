"""One 37-point mixed LHI stencil, using LocalApproximation directly.

    python -m examples.stokes_lhi_weights
    python -m examples.stokes_lhi_weights --backend cpp --digits 80 --mpi 2

This example builds momentum and pressure-gradient weights. For a complete
time-dependent solve, see examples/stokes_lhi_baseline.py.
"""
import argparse
from fractions import Fraction

import numpy as np
import sympy as sp

import rbflab as r


def run(backend="python", digits=40, ranks=0):
    # 1. Select 37 physical sites around the central mesh node.
    cells = 8
    cloud = r.geometry.unit_box_grid(cells)
    center = np.array([.5, .5])
    distances = np.linalg.norm(cloud.points-center, axis=1)
    stencil = cloud.points[np.argsort(distances, kind="stable")[:37]]

    # 2. Use the degree-6 divergence-free velocity and degree-1 pressure spaces.
    q = Fraction(cells, 44)
    velocity_kernel = r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7)
    s = sp.Symbol("s", nonnegative=True)  # squared distance
    pressure_kernel = r.Kernel(q**4*sp.exp(-q*q*s), s)()
    space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=6),
        "p": r.PressureSpace(pressure_kernel, polynomial_degree=1),
    })

    # 3. Velocity values and neighboring momentum values are the observations.
    I, Lap = r.Identity(), r.Laplacian()
    Dx, Dy = r.Derivative(0), r.Derivative(1)
    velocity = r.FieldOperator({"u": [[I, 0], [0, I]]})
    momentum = r.FieldOperator({"u": [[-Lap, 0], [0, -Lap]], "p": [[Dx], [Dy]]})
    gradient = r.FieldOperator({"p": [[Dx], [Dy]]})
    source = {
        "velocity": r.Samples(stencil, operator=velocity),
        "momentum": r.Samples(stencil, operator=momentum, target="exclude"),
    }

    # 4. Construct both weight maps with one local factorization.
    numerical_backend = (r.CppBackend(compute_condition=False) if backend == "cpp"
                         else r.PythonBackend(compute_condition=False))
    execution = r.LocalExecution(scheduler="mpi", ranks=ranks) if ranks else None
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        backend=numerical_backend, execution=execution,
        precision=r.Precision(local_digits=digits),
    )
    maps = local.operators(targets=[center], operators={"momentum": momentum, "gradient": gradient})

    # Rows follow the selected observations; columns are the two target outputs.
    weights = maps.gradient.local(0)
    print(f"Physical sites: {len(stencil)}; scalar observations: {len(weights.indices)}")
    print(f"Pressure-gradient weights: {weights.weights.rows} x {weights.weights.cols}")
    print(f"Local weight residual: {weights.diagnostics['weight_residual']:.3e}")
    return maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--digits", type=int, default=40)
    parser.add_argument("--mpi", type=int, default=0)
    args = parser.parse_args()
    if args.mpi and args.backend != "cpp":
        parser.error("MPI local weights require --backend cpp")
    run(args.backend, args.digits, args.mpi)
