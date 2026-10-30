"""Recover two fields from their sum and contrast, then differentiate them.

Run from an installed checkout:
    python -m examples.mixed_local_maps
    python -m examples.mixed_local_maps --backend cpp --digits 50
    python -m examples.mixed_local_maps --backend cpp --mpi 2

Native execution requires the optional source toolchain. MPI additionally
requires an MPI launcher (or the documented Windows/WSL bridge).
"""
import argparse
import numpy as np
import rbflab as r


def run(backend='python', digits=None, ranks=0):
    # 1. Give each field its own kernel and polynomial space.
    points = r.geometry.unit_box_grid(2).points
    I, Dx, Lap = r.Identity(), r.Derivative(0), r.Laplacian()
    space = r.MixedSpace({
        'a': r.ScalarSpace(r.IMQ(2), polynomial_degree=2),
        'b': r.ScalarSpace(r.Gaussian(1), polynomial_degree=2),
    })
    # 2. Describe what is measured: total = a+b and contrast = a-2*b.
    total = r.FieldOperator({'a': [[I]], 'b': [[I]]})
    contrast = r.FieldOperator({'a': [[I]], 'b': [[-2*I]]})
    source = {'total': r.Samples(points, operator=total),
              'contrast': r.Samples(points, operator=contrast)}
    execution = r.LocalExecution(scheduler='mpi', ranks=ranks) if ranks else None
    numerical_backend = (r.CppBackend(threads=1)
                         if backend == 'cpp' else r.PythonBackend())
    # 3. Build the common Hermite approximation.
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        backend=numerical_backend, execution=execution,
        precision=r.Precision(local_digits=digits),
    )
    # 4. Ask for both fields and for Laplacian(a) + derivative_x(b).
    targets = points[[0, 4, 8]]
    maps = local.operators(targets=targets, operators={
        'equation': r.FieldOperator({'a': [[Lap]], 'b': [[Dx]]}),
        'fields': r.FieldOperator({'a': [[I], [0]], 'b': [[0], [I]]}),
    })
    # 5. Apply the maps to measured data and check only at these mesh nodes.
    x, y = points.T
    a, b = x**2+2*y, 1+3*x-y**2
    data = {'total': a+b, 'contrast': a-2*b}
    exact = np.c_[targets[:, 0]**2+2*targets[:, 1],
                  1+3*targets[:, 0]-targets[:, 1]**2]
    field_error = float(np.max(abs(maps.fields @ data-exact)))
    equation_error = float(np.max(abs(maps.equation @ data-5)))
    print(f"Field values: maximum absolute nodal error = {field_error:.3e}")
    print(f"Laplacian(a) + derivative_x(b): error = {equation_error:.3e}")
    return {'field_max_absolute_nodal_error': field_error,
            'equation_max_absolute_nodal_error': equation_error,
            'diagnostics': maps.diagnostics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['python', 'cpp'], default='python')
    parser.add_argument('--digits', type=int)
    parser.add_argument('--mpi', type=int, default=0)
    args = parser.parse_args()
    if args.mpi and args.backend != 'cpp':
        parser.error('--mpi requires --backend cpp')
    run(args.backend, args.digits, args.mpi)
