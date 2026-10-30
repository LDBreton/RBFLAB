"""Mixed Hermite reconstruction from (U, P, g, L), using the public API.

This builds local maps, not a closed global Stokes solver. Pressure samples
must use one consistent gauge. Run with --backend cpp --ranks 2 for MPI.
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as r


def build(cells=6, pressure_degree=1, backend='cpp', ranks=0, *, all_targets=False):
    cloud = r.geometry.unit_box_grid(cells)
    interior = cloud.interior
    boundary = cloud.points[cloud.boundary_indices]
    # An interior patch and a wall-adjacent patch; both targets are mesh nodes.
    targets = interior if all_targets else interior[[0, len(interior)//2]]
    nu = sp.Rational(1, 100)
    q = sp.Rational(cells, 44)
    s = sp.Symbol('s', nonnegative=True)
    space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(
            r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), polynomial_degree=6),
        # Absolute pressure observations require the scalar constant mode.
        'p': r.ScalarSpace(r.Kernel(nu**2*q**4*sp.exp(-q*q*s), s)(),
                           polynomial_degree=pressure_degree),
    })
    u, p = space.fields
    momentum = -nu*r.laplacian(u)+r.gradient(p)
    stencil = r.Stencil(cloud.points, size=37)
    # Excluding P_i makes the pressure output a prediction from neighbors.
    # Keeping it would make pressure evaluation at this site an identity row.
    source = {
        'velocity': r.Samples(interior, operator=u, within=stencil),
        'pressure': r.Samples(interior, operator=p, within=stencil, target='exclude'),
        'boundary': r.Samples(boundary, operator=u, within=stencil),
        'momentum': r.Samples(interior, operator=momentum, within=stencil, target='exclude'),
    }
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=80),
        backend=r.PythonBackend() if backend == 'python' else r.CppBackend(threads=1, timeout=3600),
        execution=r.LocalExecution(scheduler='mpi', ranks=ranks) if ranks else None,
    )
    maps = local.operators(targets=targets, operators={
        'velocity': u, 'pressure': p, 'momentum': momentum,
        'gradient': r.gradient(p), 'dx': r.dx(u), 'dy': r.dy(u),
    })
    return maps, interior, boundary, targets


def pressure_test(maps, interior, boundary, targets, shift=0.):
    # Exact equilibrium: u=0, p=xy+shift, L=grad(p)=(y,x).
    data = {'velocity': np.zeros_like(interior),
            'pressure': interior[:,0]*interior[:,1]+shift,
            'boundary': np.zeros_like(boundary),
            'momentum': interior[:, ::-1]}
    exact = {'velocity': np.zeros_like(targets),
             'pressure': (targets[:,0]*targets[:,1]+shift)[:,None],
             'momentum': targets[:, ::-1], 'gradient': targets[:, ::-1],
             'dx': np.zeros_like(targets), 'dy': np.zeros_like(targets)}
    # These diagnostics concern local reproduction, not global evolution.
    return {name: float(np.max(abs(np.asarray((maps[name] @ data).tolist(), dtype=float)-value)))
            for name, value in exact.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['python', 'cpp'], default='cpp')
    parser.add_argument('--ranks', type=int, default=0)
    parser.add_argument('--pressure-degree', type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    if args.ranks and args.backend != 'cpp':
        parser.error('MPI requires the C++ backend')
    maps, interior, boundary, targets = build(pressure_degree=args.pressure_degree,
                                             backend=args.backend, ranks=args.ranks)
    for name, error in pressure_test(maps, interior, boundary, targets).items():
        print(f'{name}: maximum absolute nodal reproduction error = {error:.6e}')
