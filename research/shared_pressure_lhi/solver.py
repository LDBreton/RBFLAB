"""Experimental shared-pressure closure; public API weights, sparse global solve.

Run from the repository: python -m research.shared_pressure_lhi.solver --ranks 2
The dense SVD is a small-mesh diagnostic, not a scalable production solver.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import splu

from examples.stokes_pressure_samples import build


class UnresolvedNullspace(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f'Additional unresolved null modes: {diagnostics}')


class SharedPressureStokes:
    """Keep pressure values globally; eliminate only local Hermite coefficients."""

    def __init__(self, maps, nodes):
        self.maps = maps
        self.nodes = nodes
        self.n = len(nodes)
        momentum, pressure = maps.momentum, maps.pressure
        self.A = momentum['velocity'].to_scipy()
        self.G = momentum['pressure'].to_scipy()
        self.B = momentum['boundary'].to_scipy()
        self.M = eye(2*self.n, format='csr') - momentum['momentum'].to_scipy()
        self.R = pressure['velocity'].to_scipy()
        self.Q = pressure['pressure'].to_scipy()
        self.E = pressure['boundary'].to_scipy()
        self.T = pressure['momentum'].to_scipy()

    def matrix(self, dt=None):
        rate = 0. if dt is None else 1./dt
        return bmat([[self.A + rate*self.M, self.G],
                     [-self.R + rate*self.T, eye(self.n)-self.Q]], format='csr')

    def rhs(self, force, boundary, previous=None, dt=None):
        forcing = np.asarray(force).reshape(-1)
        if dt is not None:
            if previous is None:
                raise ValueError('Backward Euler requires the previous velocity')
            forcing = forcing + np.asarray(previous).reshape(-1)/dt
        g = np.asarray(boundary).reshape(-1)
        return np.concatenate((self.M@forcing-self.B@g, self.E@g+self.T@forcing))

    def factor(self, dt=None):
        return GaugedSystem(self.matrix(dt), self.n)

    def gradient(self, velocity, pressure, boundary, momentum):
        data = dict(velocity=velocity, pressure=pressure, boundary=boundary,
                    momentum=momentum)
        return np.asarray((self.maps.gradient@data).tolist(), dtype=float)


class GaugedSystem:
    """Audit rank, then border with a left-null direction and mean-pressure gauge.

    Row equilibration makes the diagnostic less sensitive to units. The border
    multiplier measures incompatible data; it must not be mistaken for a PDE
    residual. More than one numerical null mode is reported, never hidden by
    least squares or by silently discarding an equation.
    """

    def __init__(self, matrix, pressure_count):
        self.matrix = matrix.tocsr()
        size = matrix.shape[0]
        if size > 3000:
            raise ValueError('Dense rank audit limited to 3000 unknowns')
        row_norm = np.asarray(abs(matrix).sum(axis=1)).ravel()
        self.scale = 1./np.maximum(row_norm, np.finfo(float).tiny)
        scaled = matrix.multiply(self.scale[:, None]).tocsr()
        left, singular, _ = svd(scaled.toarray(), full_matrices=False)
        threshold = singular[0]*size*np.finfo(float).eps
        nullity = int(np.count_nonzero(singular <= threshold))
        gauge_mode = np.r_[np.zeros(size-pressure_count), np.ones(pressure_count)]
        self.diagnostics = {
            'size': size, 'nnz': matrix.nnz, 'numerical_nullity': nullity,
            'rank_threshold': float(threshold),
            'smallest_singular_values': singular[-6:].tolist(),
            'constant_pressure_defect': float(np.max(abs(scaled@gauge_mode))),
        }
        if nullity > 1:
            raise UnresolvedNullspace(self.diagnostics)
        # A Float64 matrix may weakly lift the exact constant-pressure null mode.
        self.border = left[:, -1]
        mean = gauge_mode/pressure_count
        bordered = bmat([[scaled, csr_matrix(self.border[:, None])],
                         [csr_matrix(mean[None, :]), csr_matrix((1, 1))]], format='csc')
        self.factorization = splu(bordered)

    def solve(self, rhs, pressure_mean=0.):
        solution = self.factorization.solve(np.r_[self.scale*rhs, pressure_mean])
        values, multiplier = solution[:-1], solution[-1]
        diagnostics = dict(self.diagnostics)
        diagnostics.update(
            compatibility_multiplier=float(multiplier),
            max_original_residual=float(np.max(abs(self.matrix@values-rhs))),
            max_scaled_residual=float(np.max(abs(self.scale*(self.matrix@values-rhs)))),
        )
        return values, diagnostics


def manufactured(nodes, time, moving):
    """Divergence-free polynomial, nonzero wall data, p=xy in a nodal mean gauge."""
    x, y = nodes.T
    amplitude = 1.+time if moving else 0.
    base = np.column_stack((x*x, -2*x*y))
    velocity = amplitude*base
    momentum = np.column_stack((y-.02*amplitude, x))
    force = momentum + (base if moving else 0.)
    return velocity, x*y, momentum, force


def experiment(cells=6, pressure_degree=2, backend='cpp', ranks=0, steps=4):
    maps, nodes, boundary, _ = build(cells, pressure_degree, backend, ranks,
                                     all_targets=True)
    solver = SharedPressureStokes(maps, nodes)
    records = []
    systems = {}
    for dt in (None, 2.5e-4, 1.25e-4):
        try:
            systems[dt] = solver.factor(dt)
        except UnresolvedNullspace as error:
            systems[dt] = error
    for moving in (False, True):
        for dt in (None, 2.5e-4, 1.25e-4):
            system = systems[dt]
            if isinstance(system, UnresolvedNullspace):
                time = 0. if dt is None else dt
                exact, pressure, momentum, force = manufactured(nodes, time, moving)
                if dt is None:
                    force = momentum
                g = manufactured(boundary, time, moving)[0]
                previous = manufactured(nodes, 0., moving)[0]
                rhs = solver.rhs(force, g, previous, dt)
                values = np.r_[exact.ravel(), pressure-pressure.mean()]
                defect = solver.matrix(dt)@values-rhs
                records.append(dict(system.diagnostics, status='rank_deficient',
                    case='nonzero_boundary' if moving else 'pressure_xy', dt=dt,
                    exact_momentum_equation_defect=float(np.max(abs(defect[:2*len(nodes)]))),
                    exact_pressure_equation_defect=float(np.max(abs(defect[2*len(nodes):])))))
                continue
            velocity = manufactured(nodes, 0., moving)[0]
            # Same final time at both time steps.
            count = 1 if dt is None else round(steps*2.5e-4/dt)
            for step in range(1, count+1):
                time = 0. if dt is None else step*dt
                exact, pressure, momentum, force = manufactured(nodes, time, moving)
                if dt is None:
                    force = momentum
                g = manufactured(boundary, time, moving)[0]
                rhs = solver.rhs(force, g, velocity, dt)
                values, diagnostic = system.solve(rhs)
                previous = velocity
                velocity = values[:2*len(nodes)].reshape(-1, 2)
                p = values[2*len(nodes):]
                ell = force if dt is None else force-(velocity-previous)/dt
            gradient = solver.gradient(velocity, p, g, ell)
            diagnostic.update(case='nonzero_boundary' if moving else 'pressure_xy',
                status='solved',
                dt=dt, final_time=time, steps=count,
                velocity_max=float(np.max(abs(velocity-exact))),
                velocity_vector_rms=float(np.sqrt(np.mean(np.sum((velocity-exact)**2, axis=1)))),
                pressure_max=float(np.max(abs(p-(pressure-pressure.mean())))),
                pressure_gradient_max=float(np.max(abs(gradient-nodes[:, ::-1]))))
            records.append(diagnostic)
    return dict(cells=cells, total_nodes=(cells+1)**2, interior_nodes=len(nodes),
                stencil=37, velocity_degree=6, pressure_degree=pressure_degree,
                viscosity=.01, q=cells/44, local_digits=80,
                geometry_and_global_arithmetic='Float64', backend=backend, ranks=ranks,
                error_scope='interior mesh nodes only',
                execution=maps.diagnostics.get('execution'), results=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=6)
    parser.add_argument('--pressure-degree', type=int, choices=[1, 2], default=2)
    parser.add_argument('--backend', choices=['python', 'cpp'], default='cpp')
    parser.add_argument('--ranks', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.ranks and args.backend != 'cpp':
        parser.error('MPI requires the C++ backend')
    if args.output.exists():
        parser.error('Choose a new output path to preserve previous evidence')
    result = experiment(args.cells, args.pressure_degree, args.backend, args.ranks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for row in result['results']:
        print(row['case'], row['dt'], row['status'], 'velocity', row.get('velocity_max'),
              'gradient', row.get('pressure_gradient_max'), 'nullity', row['numerical_nullity'])


if __name__ == '__main__':
    main()
