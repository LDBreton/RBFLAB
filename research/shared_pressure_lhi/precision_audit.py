"""Retain native MPFR weights through assembly and an mpmath singular-value audit.

This deliberately small diagnostic never recovers precision from rounded SciPy
matrices. Geometry is still Float64, as in the baseline experiment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp
import rbflab as r


def build_maps(cells=6, digits=80, ranks=2):
    cloud = r.geometry.unit_box_grid(cells)
    inside = cloud.interior
    boundary = cloud.points[cloud.boundary_indices]
    nu, q = sp.Rational(1, 100), sp.Rational(cells, 44)
    s = sp.Symbol('s', nonnegative=True)
    space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(
            r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), polynomial_degree=6),
        'p': r.ScalarSpace(r.Kernel(nu**2*q**4*sp.exp(-q*q*s), s)(),
                           polynomial_degree=2),
    })
    u, p = space.fields
    momentum = -nu*r.laplacian(u)+r.gradient(p)
    stencil = r.Stencil(cloud.points, size=37)
    source = {
        'velocity': r.Samples(inside, operator=u, within=stencil),
        'pressure': r.Samples(inside, operator=p, within=stencil, target='exclude'),
        'boundary': r.Samples(boundary, operator=u, within=stencil),
        'momentum': r.Samples(inside, operator=momentum, within=stencil, target='exclude'),
    }
    local = r.LocalApproximation(
        source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=digits, global_dtype='mpmath'),
        backend=r.CppBackend(threads=1, timeout=3600),
        execution=r.LocalExecution(scheduler='mpi', ranks=ranks) if ranks else None,
    )
    return local.operators(targets=inside, operators={
        'momentum': momentum, 'pressure': p, 'gradient': r.gradient(p)})


def dense(block):
    """Copy MP sparse entries into a dense matrix without float conversion."""
    matrix = block.matrix
    result = matrix.ctx.matrix(*matrix.shape)
    for i, row in enumerate(matrix.rows):
        for j, value in row.items():
            result[i, j] = value
    return result


def assemble(maps, dt=None):
    momentum, pressure = maps.momentum, maps.pressure
    ctx = momentum.matrix.ctx
    n = pressure.shape[0]
    rate = ctx.zero if dt is None else 1/ctx.mpf(dt)
    A, G = dense(momentum['velocity']), dense(momentum['pressure'])
    M = ctx.eye(2*n)-dense(momentum['momentum'])
    R, Q, T = (dense(pressure[group]) for group in ('velocity', 'pressure', 'momentum'))
    matrix = ctx.matrix(3*n)
    matrix[:2*n, :2*n] = A+rate*M
    matrix[:2*n, 2*n:] = G
    matrix[2*n:, :2*n] = -R+rate*T
    matrix[2*n:, 2*n:] = ctx.eye(n)-Q
    return matrix, n


def audit(matrix, pressure_count):
    ctx = matrix.ctx
    size = matrix.rows
    scaled = matrix.copy()
    for i in range(size):
        norm = ctx.fsum(abs(matrix[i, j]) for j in range(size))
        for j in range(size):
            scaled[i, j] /= norm
    started = time.perf_counter()
    singular = ctx.svd(scaled, compute_uv=False)
    float_matrix = np.array(scaled.tolist(), dtype=float)
    float_singular = np.linalg.svd(float_matrix, compute_uv=False)
    float_tolerance = float_singular[0]*size*np.finfo(float).eps
    gauge = ctx.matrix([0]*(size-pressure_count)+[1]*pressure_count)
    # A cutoff family shows sensitivity rather than declaring exact rank.
    return {
        'svd_seconds': time.perf_counter()-started,
        'mp_singular_values': [ctx.nstr(value, ctx.dps) for value in singular],
        'float_singular_values': float_singular.tolist(),
        'float_numerical_nullity': int(np.count_nonzero(float_singular < float_tolerance)),
        'relative_cutoff_nullities': {
            str(exponent): sum(value < singular[0]*ctx.power(10, -exponent) for value in singular)
            for exponent in (12, 16, 20, 30, 40, 50, 60, 70)},
        'constant_pressure_defect': ctx.nstr(max(abs(value) for value in scaled*gauge), ctx.dps),
        'smallest_nongauge_relative_singular': ctx.nstr(singular[size-2]/singular[0], ctx.dps),
    }


def manufactured_solve(maps, matrix, dt, moving):
    """One exact-history step (or steady solve), not a stability trajectory."""
    ctx = matrix.ctx
    nodes = maps.pressure['pressure'].points
    boundary = maps.pressure['boundary'].points
    n = len(nodes)
    time = ctx.zero if dt is None else ctx.mpf(dt)
    rate = ctx.zero if dt is None else 1/time

    def field(points, amplitude):
        values = []
        for x, y in points:
            x, y = ctx.mpf(float(x)), ctx.mpf(float(y))
            values.extend((amplitude*x*x, -2*amplitude*x*y))
        return ctx.matrix(values)

    amplitude = 1+time if moving else ctx.zero
    exact_velocity = field(nodes, amplitude)
    previous = field(nodes, ctx.one if moving else ctx.zero)
    g = field(boundary, amplitude)
    force = ctx.matrix(2*n, 1)
    exact_pressure = ctx.matrix(n, 1)
    exact_gradient = ctx.matrix(2*n, 1)
    for i, (x, y) in enumerate(nodes):
        x, y = ctx.mpf(float(x)), ctx.mpf(float(y))
        force[2*i], force[2*i+1] = y-ctx.mpf('0.02')*amplitude, x
        exact_pressure[i] = x*y
        exact_gradient[2*i], exact_gradient[2*i+1] = y, x
    if moving and dt is not None:
        force += field(nodes, ctx.one)
    mean = ctx.fsum(exact_pressure)/n
    exact_pressure = ctx.matrix([value-mean for value in exact_pressure])
    forcing = force+rate*previous
    M = ctx.eye(2*n)-dense(maps.momentum['momentum'])
    B = dense(maps.momentum['boundary'])
    E, T = dense(maps.pressure['boundary']), dense(maps.pressure['momentum'])
    rhs = ctx.matrix(list(M*forcing-B*g)+list(E*g+T*forcing))
    gauged, data = matrix.copy(), rhs.copy()
    for i in range(3*n):
        scale = ctx.fsum(abs(gauged[i,j]) for j in range(3*n))
        for j in range(3*n):
            gauged[i,j] /= scale
        data[i] /= scale
    # Replace one row only after the spectrum has identified one null mode.
    # Check ALL original equations afterwards to expose compatibility loss.
    for j in range(3*n):
        gauged[3*n-1,j] = ctx.zero if j < 2*n else ctx.one/n
    data[3*n-1] = ctx.zero
    solution = ctx.lu_solve(gauged, data)
    velocity, pressure = solution[:2*n,:], solution[2*n:,:]
    momentum = force-rate*(velocity-previous)
    gradient = sum((dense(maps.gradient[group])*values for group, values in (
        ('velocity',velocity), ('pressure',pressure), ('boundary',g),
        ('momentum',momentum))), ctx.matrix(2*n,1))
    return dict(case='nonzero_boundary' if moving else 'pressure_xy',
        interpretation='steady solve' if dt is None else 'one step from exact history',
        velocity_max=ctx.nstr(max(abs(value) for value in velocity-exact_velocity), ctx.dps),
        pressure_max=ctx.nstr(max(abs(value) for value in pressure-exact_pressure), ctx.dps),
        gradient_max=ctx.nstr(max(abs(value) for value in gradient-exact_gradient), ctx.dps),
        original_residual_max=ctx.nstr(max(abs(value) for value in matrix*solution-rhs),ctx.dps))


def smooth_evolution(maps, matrix, dt, final_time='0.01', pressure_kind='trigonometric'):
    """Nonpolynomial spatial field; actual BE history, no exact-history reset."""
    ctx = matrix.ctx
    nodes = maps.pressure['pressure'].points
    boundary = maps.pressure['boundary'].points
    n = len(nodes)
    rate = ctx.zero if dt is None else 1/ctx.mpf(dt)

    def spatial(points):
        velocity, laplacian, pressure, gradient = [], [], [], []
        pi = ctx.pi
        for x, y in points:
            x, y = ctx.mpf(float(x)), ctx.mpf(float(y))
            sx, sy = ctx.sin(pi*x)**2, ctx.sin(pi*y)**2
            tx, ty = ctx.sin(2*pi*x), ctx.sin(2*pi*y)
            cx, cy = ctx.cos(2*pi*x), ctx.cos(2*pi*y)
            velocity.extend((pi*sx*ty, -pi*tx*sy))
            laplacian.extend((pi**3*(2*cx-4*sx)*ty, pi**3*tx*(4*sy-2*cy)))
            if pressure_kind == 'trigonometric':
                pressure.append(ctx.sin(pi*x)*ctx.cos(pi*y))
                gradient.extend((pi*ctx.cos(pi*x)*ctx.cos(pi*y), -pi*ctx.sin(pi*x)*ctx.sin(pi*y)))
            else:
                pressure.append(x*y)
                gradient.extend((y,x))
        return tuple(ctx.matrix(value) for value in (velocity,laplacian,pressure,gradient))

    base, lap, exact_pressure, exact_gradient = spatial(nodes)
    boundary_base = spatial(boundary)[0]
    exact_pressure -= ctx.matrix([ctx.fsum(exact_pressure)/n]*n)
    M = ctx.eye(2*n)-dense(maps.momentum['momentum'])
    B = dense(maps.momentum['boundary'])
    E, T = dense(maps.pressure['boundary']), dense(maps.pressure['momentum'])
    gradient_maps = {group:dense(maps.gradient[group])
                     for group in ('velocity','pressure','boundary','momentum')}
    scales = [1/ctx.fsum(abs(matrix[i,j]) for j in range(3*n)) for i in range(3*n)]
    gauged = ctx.matrix([[scales[i]*matrix[i,j] for j in range(3*n)] for i in range(3*n)])
    for j in range(3*n):
        gauged[3*n-1,j] = ctx.zero if j < 2*n else ctx.one/n
    permutation, lower, upper = ctx.lu(gauged)

    def solve(rhs):
        data = ctx.matrix([scales[i]*rhs[i] for i in range(3*n)])
        data[3*n-1] = ctx.zero
        permuted = permutation*data
        intermediate, values = ctx.matrix(3*n,1), ctx.matrix(3*n,1)
        for i in range(3*n):
            intermediate[i] = (permuted[i]-ctx.fsum(lower[i,j]*intermediate[j] for j in range(i)))/lower[i,i]
        for i in reversed(range(3*n)):
            values[i] = (intermediate[i]-ctx.fsum(upper[i,j]*values[j] for j in range(i+1,3*n)))/upper[i,i]
        return values

    velocity = base.copy()
    count = 1 if dt is None else int(ctx.nint(ctx.mpf(final_time)/ctx.mpf(dt)))
    history = []
    for step in range(1,count+1):
        time = ctx.zero if dt is None else step*ctx.mpf(dt)
        exact_velocity = (1+time)*base
        g = (1+time)*boundary_base
        force = -ctx.mpf('0.01')*(1+time)*lap+exact_gradient
        if dt is not None:
            force += base
        forcing = force+rate*velocity
        rhs = ctx.matrix(list(M*forcing-B*g)+list(E*g+T*forcing))
        solution = solve(rhs)
        previous = velocity
        velocity, pressure = solution[:2*n,:], solution[2*n:,:]
        momentum = force-rate*(velocity-previous)
        gradient = sum((gradient_maps[group]*values for group,values in (
            ('velocity',velocity),('pressure',pressure),('boundary',g),('momentum',momentum))),ctx.matrix(2*n,1))
        history.append(dict(time=ctx.nstr(time,20),
            velocity_max=ctx.nstr(max(abs(value) for value in velocity-exact_velocity),ctx.dps),
            pressure_max=ctx.nstr(max(abs(value) for value in pressure-exact_pressure),ctx.dps),
            gradient_max=ctx.nstr(max(abs(value) for value in gradient-exact_gradient),ctx.dps),
            original_residual_max=ctx.nstr(max(abs(value) for value in matrix*solution-rhs),ctx.dps)))
    return dict(streamfunction='(1+t)*sin(pi*x)^2*sin(pi*y)^2',
                pressure=('sin(pi*x)*cos(pi*y)' if pressure_kind == 'trigonometric' else 'x*y')+', nodal mean removed',
                steps=count, initial_data='exact velocity, then computed history', history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=6)
    parser.add_argument('--digits', type=int, default=80)
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--final-time', default='0.01')
    parser.add_argument('--smooth-pressure', choices=['trigonometric', 'xy'], default='trigonometric')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory to preserve evidence')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    maps = build_maps(args.cells, args.digits, args.ranks)
    result = dict(cells=args.cells, total_nodes=(args.cells+1)**2,
                  local_digits=args.digits, global_digits=args.digits,
                  geometry='Float64', stencil=37, augmentation='6/2',
                  viscosity='0.01', q=f'{args.cells}/44',
                  weight_seconds=time.perf_counter()-started,
                  execution=maps.diagnostics.get('execution'), records=[])
    for dt in (None, '0.00025', '0.000125'):
        matrix, n = assemble(maps, dt)
        label = 'steady' if dt is None else dt
        # Save native-weight assembled entries before any diagnostic rounding.
        (args.output/f'matrix_{label}.json').write_text(json.dumps({
            'digits': args.digits,
            'entries': [[matrix.ctx.nstr(matrix[i, j], args.digits)
                         for j in range(matrix.cols)] for i in range(matrix.rows)]}), encoding='utf-8')
        record = dict(dt=dt, **audit(matrix, n))
        record['manufactured_solves'] = [manufactured_solve(maps, matrix, dt, moving)
                                         for moving in (False, True)]
        record['smooth_evolution'] = smooth_evolution(maps, matrix, dt, args.final_time, args.smooth_pressure)
        result['records'].append(record)
        (args.output/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(label, record['relative_cutoff_nullities'], flush=True)


if __name__ == '__main__':
    main()
