"""Stationary mixed LHI with additional known PDE observations.

These are Hermite source observations, not independent PDE test equations.
Run as a module from the repository. This study does not advance in time:
off-node momentum is known from the forcing only for stationary Stokes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.sparse import bmat, eye, save_npz
import rbflab as r

from research.shared_pressure_lhi.solver import GaugedSystem, UnresolvedNullspace


def build_observations(cells=6, extra_count=2, replace=False, ranks=2):
    cloud = r.geometry.unit_box_grid(cells)
    nodes = cloud.interior
    boundary = cloud.points[cloud.boundary_indices]
    offsets = np.array([[.23, .17], [-.21, .13], [.11, -.19], [-.17, -.23]])
    extras = (nodes[:, None, :] + offsets[None, :extra_count, :]/cells).reshape(-1, 2)
    extra_indices = np.arange(len(extras)).reshape(len(nodes), extra_count)
    nu, q = sp.Rational(1, 100), sp.Rational(cells, 44)
    s = sp.Symbol('s', nonnegative=True)
    space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(
            r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), polynomial_degree=6),
        'p': r.ScalarSpace(r.Kernel(nu**2*q**4*sp.exp(-q*q*s), s)(), polynomial_degree=2),
    })
    u, p = space.fields
    momentum = -nu*r.laplacian(u) + r.gradient(p)
    stencil = r.Stencil(cloud.points, size=37)
    source = {
        'velocity': r.Samples(nodes, operator=u, within=stencil),
        'pressure': r.Samples(nodes, operator=p, within=stencil, target='exclude'),
        'boundary': r.Samples(boundary, operator=u, within=stencil),
    }
    if not replace:
        source['momentum'] = r.Samples(nodes, operator=momentum, within=stencil,
                                       target='exclude')
    source['extra_momentum'] = r.Samples(extras, operator=momentum, indices=extra_indices)
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=80), backend=r.CppBackend(threads=1, timeout=3600),
        execution=r.LocalExecution(scheduler='mpi', ranks=ranks) if ranks else None,
    )
    maps = local.operators(targets=nodes, operators={
        'momentum': momentum, 'pressure': p, 'gradient': r.gradient(p),
    })
    return maps, nodes, boundary, extras


def exact_fields(points, case):
    x, y = points.T
    if case == 'pressure_xy':
        return np.zeros_like(points), x*y, np.column_stack((y, x)), np.column_stack((y, x))
    if case == 'polynomial':
        velocity = np.column_stack((x*x, -2*x*y))
        return velocity, x*y, np.column_stack((y-.02, x)), np.column_stack((y, x))
    # Smooth nonpolynomial equilibrium; streamfunction ensures div(u)=0.
    sx, cx, sy, cy = np.sin(np.pi*x), np.cos(np.pi*x), np.sin(np.pi*y), np.cos(np.pi*y)
    velocity = np.column_stack((2*np.pi*sx*sx*sy*cy, -2*np.pi*sx*cx*sy*sy))
    laplace = np.column_stack((4*np.pi**3*sy*cy*(cx*cx-3*sx*sx),
                              -4*np.pi**3*sx*cx*(cy*cy-3*sy*sy)))
    pressure = sx*sy
    gradient = np.pi*np.column_stack((cx*sy, sx*cy))
    return velocity, pressure, -.01*laplace+gradient, gradient


def run(cells, extra_count, replace, ranks, output):
    maps, nodes, boundary, extras = build_observations(cells, extra_count, replace, ranks)
    n = len(nodes)
    momentum, pressure = maps.momentum, maps.pressure
    A, G = momentum['velocity'].to_scipy(), momentum['pressure'].to_scipy()
    R, Q = pressure['velocity'].to_scipy(), pressure['pressure'].to_scipy()
    matrix = bmat([[A, G], [-R, eye(n)-Q]], format='csr')
    save_npz(output.with_suffix('.npz'), matrix)
    try:
        system = GaugedSystem(matrix, n)
        diagnostics = system.diagnostics
    except UnresolvedNullspace as error:
        system, diagnostics = None, error.diagnostics
    records = []
    for case in ('pressure_xy', 'polynomial', 'smooth'):
        u, p, force, gradient = exact_fields(nodes, case)
        g = exact_fields(boundary, case)[0]
        extra_force = exact_fields(extras, case)[2]
        known = {'boundary': g, 'extra_momentum': extra_force}
        if not replace:
            known['momentum'] = force
        momentum_rhs = force.ravel().copy()
        pressure_rhs = np.zeros(n)
        for name, values in known.items():
            momentum_rhs -= momentum[name].to_scipy()@values.ravel()
            pressure_rhs += pressure[name].to_scipy()@values.ravel()
        rhs = np.r_[momentum_rhs, pressure_rhs]
        exact_values = np.r_[u.ravel(), p-p.mean()]
        residual = matrix@exact_values-rhs
        result = dict(case=case, exact_momentum_defect=float(max(abs(residual[:2*n]))),
                      exact_pressure_defect=float(max(abs(residual[2*n:]))),
                      status='rank_deficient' if system is None else 'solved')
        if system is not None:
            values, info = system.solve(rhs)
            velocity, computed_pressure = values[:2*n].reshape(n, 2), values[2*n:]
            data = dict(known, velocity=velocity, pressure=computed_pressure)
            computed_gradient = np.asarray((maps.gradient@data).tolist(), dtype=float)
            result.update(info, velocity_max=float(max(abs(velocity-u).ravel())),
                pressure_max=float(max(abs(computed_pressure-(p-p.mean())))),
                gradient_max=float(max(abs(computed_gradient-gradient).ravel())))
        records.append(result)
    result = dict(cells=cells, total_nodes=(cells+1)**2, interior_nodes=n,
        base_stencil=37, extra_momentum_sites_per_patch=extra_count,
        replace_original_momentum=replace, pressure_degree=2, velocity_degree=6,
        viscosity=.01, q=cells/44, local_digits=80, global_arithmetic='Float64',
        scope='Stationary only; errors at original interior nodes',
        diagnostics=diagnostics, execution=maps.diagnostics.get('execution'), results=records)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=6)
    parser.add_argument('--extra-count', type=int, choices=[2, 4], default=2)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('Choose new output names to preserve evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.cells, args.extra_count, args.replace, args.ranks, args.output)


if __name__ == '__main__':
    main()
