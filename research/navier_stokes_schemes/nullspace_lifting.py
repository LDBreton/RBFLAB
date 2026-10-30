"""Bounded common-field nullspace optimization, not a stability guarantee.

E=E0+J Z (I-B B+) D^-1, B=D^-1 C.  J selects the three quadratic
pressure coefficients.  Thus EC=I for arbitrary Z; all eight output maps
remain evaluations of one cardinal analytic DF reconstruction on 37 sites.
Only saved MPFR/MPI source weights are reused. Optimization is Float64.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eig, lstsq, qr, null_space, svd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, eye
from threadpoolctl import threadpool_limits

from .metric_lifting import prepare, save_maps, spectrum
from .pressure_lifting import NAMES, load_problem, all_pressure_controls


class NullLifting:
    def __init__(self, p, alpha=100., beta=0.):
        self.p = p
        self.data, self.selection = prepare(p)
        self.n = len(p.U0)
        self.offsets = [0]
        for d in self.data:
            D = d['D'] * np.where(d['momentum'], alpha, 1.) * (1+d['r2'])**(beta/2)
            B = d['C']/D[:, None]
            inverse, _, rank, _ = lstsq(B, np.eye(len(B)), cond=1e-13, lapack_driver='gelsd')
            if rank != 40:
                raise RuntimeError(f'polynomial rank {rank}')
            d['scale'] = D
            d['E0'] = inverse/D[None, :]
            # Form the projector from orthogonal QR columns. Subtracting B B+
            # suffers cancellation here and spoils high-degree moments.
            orthogonal, _ = qr(B, mode='full')
            null = orthogonal[:, 40:]
            d['P'] = null@null.T
            d['base'] = d['W']+d['R']@d['E0']
            self.offsets.append(self.offsets[-1]+3*len(B))
        self.size = self.offsets[-1]

    def local(self, x, owner):
        d = self.data[owner]
        Z = x[self.offsets[owner]:self.offsets[owner+1]].reshape(3, -1)
        E = d['E0'].copy()
        E[37:40] += (Z@d['P'])/d['scale'][None, :]
        return d['W']+d['R']@E, E

    def matrices(self, x, full=False):
        result = copy.copy(self.p)
        entries = {(o, g): ([], [], []) for o in range(4 if full else 1) for g in range(3)}
        for owner, d in enumerate(self.data):
            V, _ = self.local(x, owner)
            for group, (mask, cols) in enumerate(d['groups']):
                for o in range(4 if full else 1):
                    rr, cc, vv = entries[o, group]
                    for comp in range(2):
                        rr.extend([2*owner+comp]*len(cols))
                        cc.extend(cols.tolist())
                        vv.extend(V[2*o+comp, mask].tolist())
        for (o, group), (rr, cc, vv) in entries.items():
            shape = (self.n, 2*len(self.p.boundary_points) if group == 1 else self.n)
            setattr(result, NAMES[o][group], coo_matrix((vv, (rr, cc)), shape=shape).tocsr())
        result.M = eye(self.n, format='csr')-result.SL
        return result

    def objective(self, x, margin=.01):
        p = self.matrices(x)
        a, m = p.A.toarray(), p.M.toarray()
        values, left, right = eig(-a, m, left=True, right=True)
        finite = np.isfinite(values)
        av, mv = a@right, m@right
        mass = np.linalg.norm(mv, axis=0)/np.linalg.norm(right, axis=0)
        residual = np.full(len(values), np.inf)
        residual[finite] = np.linalg.norm(-av[:, finite]-mv[:, finite]*values[finite], axis=0)/(np.linalg.norm(av[:, finite], axis=0)+abs(values[finite])*np.linalg.norm(mv[:, finite], axis=0))
        resolved = finite & (mass > 1e-8) & (residual < 1e-7)
        if resolved.sum() != self.n-5:
            raise RuntimeError(f'expected {self.n-5} resolved modes, got {resolved.sum()}')
        active = np.flatnonzero(resolved & (values.real > -margin))
        objective = float(np.sum((values[active].real+margin)**2))
        gradient = np.zeros_like(x)
        for k in active:
            lam, v, w = values[k], right[:, k], left[:, k]
            denominator = np.vdot(w, m@v)
            if abs(denominator) < 1e-12:
                raise RuntimeError('ill-conditioned active eigenvalue')
            factor = -2*(lam.real+margin)/denominator
            for owner, d in enumerate(self.data):
                q = np.zeros(len(d['C']), dtype=complex)
                for group in (0, 2):
                    mask, cols = d['groups'][group]
                    q[mask] = v[cols]*(1 if group == 0 else -lam)
                source = d['P']@(q/d['scale'])
                target = w[2*owner:2*owner+2].conj()@d['R'][:2, 37:40]
                gradient[self.offsets[owner]:self.offsets[owner+1]] += (factor*np.outer(target, source)).real.ravel()
        self.last = dict(objective=objective, real=float(np.max(values[resolved].real)),
                         positive_count=int(np.sum(values[resolved].real > 1e-7)),
                         resolved_count=int(resolved.sum()), active_count=len(active),
                         max_parameter=float(np.max(abs(x))), parameter_norm=float(np.linalg.norm(x)))
        return objective, gradient

    def checks(self, x):
        result = []
        for owner, d in enumerate(self.data):
            V, E = self.local(x, owner)
            error = V.astype(np.longdouble)@d['C'].astype(np.longdouble)-d['T'].astype(np.longdouble)
            observed_scale = np.linalg.norm(d['C'], axis=0)
            action_scale = np.linalg.norm(V, axis=1)[:, None]*observed_scale[None, :]+abs(d['T'])
            result.append(dict(owner=owner,
                polynomial_error_inf=float(np.max(abs(V@d['C']-d['T']))),
                left_inverse_error_inf=float(np.max(abs(E@d['C']-np.eye(40)))),
                polynomial_error_longdouble_inf=float(np.max(abs(V.astype(np.longdouble)@d['C'].astype(np.longdouble)-d['T'].astype(np.longdouble)))),
                longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
                polynomial_column_errors_inf=np.max(abs(error), axis=0).astype(float).tolist(),
                polynomial_column_errors_observation_normalized=(np.max(abs(error), axis=0)/observed_scale).astype(float).tolist(),
                polynomial_backward_error_inf=float(np.max(abs(error)/np.maximum(action_scale, 1e-300))),
                normalized_change=float(np.linalg.norm((V-d['W'])*d['D'])/np.linalg.norm(d['W']*d['D']))))
        return result


def descriptor_audit(p):
    """Independent index-one reduction; no eigenvalues are modified."""
    a, m = p.A.toarray(), p.M.toarray()
    left, singular, _ = svd(m)
    rank = len(m)-5
    yrange, ynull = left[:, :rank], left[:, rank:]
    consistent = null_space(ynull.T@a, rcond=1e-12)
    if consistent.shape[1] != rank:
        raise RuntimeError('descriptor constraint rank changed')
    reduced_mass = yrange.T@m@consistent
    reduced_a = yrange.T@a@consistent
    values = eig(-reduced_a, reduced_mass, right=False)
    return dict(mass_singular_values=singular.tolist(),
        min_nonzero_mass_singular_value=float(singular[rank-1]),
        max_null_mass_singular_value=float(singular[rank]),
        reduced_mass_condition=float(np.linalg.cond(reduced_mass)),
        consistent_dimension=int(consistent.shape[1]),
        reduced_spectral_abscissa=float(np.max(values.real)),
        reduced_positive_count=int(np.sum(values.real > 1e-7)))


def gradient_check(model):
    x = np.zeros(model.size)
    rng = np.random.default_rng(20261029)
    direction = rng.standard_normal(model.size)
    direction /= np.linalg.norm(direction)
    value, grad = model.objective(x)
    exact = float(grad@direction)
    rows = []
    for step in (1e-4, 3e-5, 1e-5):
        plus = model.objective(x+step*direction)[0]
        minus = model.objective(x-step*direction)[0]
        difference = (plus-minus)/(2*step)
        rows.append(dict(step=step, analytic=exact, finite_difference=difference,
            relative_error=abs(exact-difference)/max(1., abs(exact), abs(difference))))
    if min(row['relative_error'] for row in rows) > 2e-4:
        raise RuntimeError(f'eigen sensitivity check failed: {rows}')
    return dict(value=value, directions=rows)


def run(cells, source, output, iterations=15, bound=.25, initial=None):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = NullLifting(load_problem(cells, source))
    configuration = dict(cells=cells, source=str(source), selection=model.selection,
        reconstruction=__doc__, alpha=100., beta=0., seed=20261029,
        optimizer='L-BFGS-B, squared positive spectral real parts with margin .01',
        bound=bound, max_accepted_iterations=iterations, parameters=model.size,
        parameter_sharing='owner-specific scaled-coordinate null projector, last3 E rows only',
        acceptance_thresholds=dict(moment_absolute=1e-7,left_inverse_absolute=1e-7,
            mass_nonzero_min=1e-8,reduced_mass_condition_max=1e8,
            resolved_mass_action_min=1e-8,resolved_eigenresidual_max=1e-7),
        precision='saved MPFR80 MPI8 local source; all corrections and eigensolves Float64',
        initial_parameters=str(initial) if initial else None)
    (output/'configuration.json').write_text(json.dumps(configuration, indent=2))
    check = gradient_check(model)
    (output/'gradient_check.json').write_text(json.dumps(check, indent=2))
    x0 = np.load(initial) if initial else np.zeros(model.size)
    if x0.shape != (model.size,) or np.max(abs(x0)) > bound+1e-12:
        raise RuntimeError('invalid initial parameters')
    initial_descriptor = descriptor_audit(model.matrices(x0))
    history = []
    def callback(x):
        model.objective(x)
        row = dict(iteration=len(history)+1, **model.last)
        history.append(row)
        np.save(output/'parameters_latest.npy', x)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(row, flush=True)
    result = minimize(model.objective, x0, method='L-BFGS-B', jac=True,
        bounds=[(-bound, bound)]*model.size, callback=callback,
        options=dict(maxiter=iterations, maxls=12, ftol=1e-12, gtol=1e-7, maxcor=7))
    x = result.x
    np.save(output/'parameters.npy', x)
    p = model.matrices(x, full=True)
    save_maps(p, output/'maps')
    checks = model.checks(x)
    (output/'checks.json').write_text(json.dumps(checks, indent=2))
    final_descriptor = descriptor_audit(p)
    consistent = max(d['polynomial_error_inf'] for d in checks) < 1e-7 and max(d['left_inverse_error_inf'] for d in checks) < 1e-7
    regular = final_descriptor['min_nonzero_mass_singular_value'] > 1e-8 and final_descriptor['reduced_mass_condition'] < 1e8
    summary = dict(status='bounded_search_complete' if consistent and regular else 'rejected_consistency_or_mass', optimizer_message=str(result.message),
        accepted_iterations=int(result.nit), function_evaluations=int(result.nfev),
        objective=float(result.fun), spectrum=spectrum(p),
        initial_descriptor=initial_descriptor, final_descriptor=final_descriptor,
        pressure_controls=all_pressure_controls(p), seconds=time.perf_counter()-start,
        max_polynomial_error=max(d['polynomial_error_inf'] for d in checks),
        max_left_inverse_error=max(d['left_inverse_error_inf'] for d in checks))
    (output/'results.json').write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', type=int, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=15)
    ap.add_argument('--bound', type=float, default=.25)
    ap.add_argument('--initial', type=Path)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells, args.source, args.output, args.iterations, args.bound, args.initial)
