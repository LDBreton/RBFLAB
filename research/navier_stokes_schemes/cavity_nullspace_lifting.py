"""Three bounded frozen-root optimization cycles of the common DF/P2 lifting.

The inner objective uses frozen U,L. Only outer recomputed roots and spectra
test the changed nonlinear problem. No PDE, support or time scheme is changed.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.sparse import bmat, eye, csr_matrix
from threadpoolctl import threadpool_limits
from .nullspace_lifting import NullLifting, descriptor_audit
from .metric_lifting import save_maps, spectrum
from .pressure_lifting import load_problem, all_pressure_controls
from .coupled import convection
from .cavity_candidates import steady, reference_at_nodal_lines


def coupled_spectrum(p, U, L, vectors=False):
    n = len(U)
    _, JU, JL = convection(p, U, L, p.boundary(0))
    K = bmat([[p.A, -p.M], [JU, eye(n)+JL]], format='csr').toarray()
    B = np.zeros_like(K)
    B[n:, :n] = np.eye(n)
    values, left, right = eig(-K, B, left=True, right=True)
    finite = np.isfinite(values)
    kv, bv = K@right, B@right
    mass = np.linalg.norm(bv, axis=0)/np.linalg.norm(right, axis=0)
    residual = np.full(len(values), np.inf)
    residual[finite] = np.linalg.norm(-kv[:, finite]-bv[:, finite]*values[finite], axis=0)/(np.linalg.norm(kv[:, finite], axis=0)+abs(values[finite])*np.linalg.norm(bv[:, finite], axis=0))
    resolved = finite & (mass > 1e-8) & (residual < 1e-7)
    if resolved.sum() != n-5:
        raise RuntimeError(f'coupled expected {n-5} resolved modes; got {resolved.sum()}')
    k = np.flatnonzero(resolved)[np.argmax(values[resolved].real)]
    report = dict(real=float(values[k].real), imag=float(values[k].imag),
        positive_count=int(np.sum(values[resolved].real > 1e-7)),
        resolved_count=int(resolved.sum()), leading_eigenresidual=float(residual[k]),
        leading_mass_action=float(mass[k]))
    if vectors:
        return report, values, left, right, B, resolved
    return report


class CavityLifting(NullLifting):
    def freeze(self, U, L):
        self.frozen_U, self.frozen_L = U.copy(), L.copy()
        g = self.p.boundary(0)
        for d in self.data:
            q = np.zeros(len(d['C']))
            for group, field in enumerate((U, g, L)):
                mask, cols = d['groups'][group]
                q[mask] = field[cols]
            d['frozen_source'] = d['P']@(q/d['scale'])

    def objective(self, x, margin=.01):
        value, gradient = super().objective(x, margin)
        rest = self.last.copy()
        p = self.matrices(x, full=True)
        report, values, left, right, B, resolved = coupled_spectrum(p, self.frozen_U, self.frozen_L, True)
        active = np.flatnonzero(resolved & (values.real > -margin))
        cavity_value = float(np.sum((values[active].real+margin)**2))
        n = self.n
        for k in active:
            lam, v, w = values[k], right[:, k], left[:, k]
            du, dl = v[:n], v[n:]
            denominator = np.vdot(w, B@v)
            if abs(denominator) < 1e-12:
                raise RuntimeError('ill-conditioned coupled active mode')
            factor = -2*(lam.real+margin)/denominator
            for owner, d in enumerate(self.data):
                q = np.zeros(len(d['C']), dtype=complex)
                for group, field in ((0, du), (2, dl)):
                    mask, cols = d['groups'][group]
                    q[mask] = field[cols]
                source = d['P']@(q/d['scale'])
                base = d['frozen_source']
                mom = w[2*owner:2*owner+2].conj()@d['R'][:2, 37:40]
                dx = w[n+2*owner:n+2*owner+2].conj()@d['R'][4:6, 37:40]
                dy = w[n+2*owner:n+2*owner+2].conj()@d['R'][6:8, 37:40]
                change = np.outer(mom, source)
                change += np.outer(dx, self.frozen_U[2*owner]*source+du[2*owner]*base)
                change += np.outer(dy, self.frozen_U[2*owner+1]*source+du[2*owner+1]*base)
                gradient[self.offsets[owner]:self.offsets[owner+1]] += (factor*change).real.ravel()
        self.last = dict(objective=value+cavity_value, rest_objective=value,
                         frozen_cavity_objective=cavity_value, rest_real=rest['real'],
                         rest_positive=rest['positive_count'], cavity=report,
                         max_parameter=float(np.max(abs(x))), parameter_norm=float(np.linalg.norm(x)))
        return value+cavity_value, gradient


def sensitivity_check(model, x):
    rng = np.random.default_rng(20261029)
    direction = rng.standard_normal(len(x))
    direction /= np.linalg.norm(direction)
    value, gradient = model.objective(x)
    analytic = float(gradient@direction)
    rows = []
    for h in (1e-4, 3e-5, 1e-5):
        numerical = (model.objective(x+h*direction)[0]-model.objective(x-h*direction)[0])/(2*h)
        rows.append(dict(step=h, analytic=analytic, central_difference=numerical,
            relative_error=abs(analytic-numerical)/max(1., abs(analytic), abs(numerical))))
    if min(row['relative_error'] for row in rows) > 2e-4:
        raise RuntimeError(f'coupled sensitivity failed {rows}')
    return dict(objective=value, directions=rows)


def continue_root(model, old, new, U, L, folder, initial_fraction=0., initial_step=.2, max_trials=15):
    """Follow the previous root along common-E coefficients; no optimization."""
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    fraction, step = initial_fraction, initial_step
    for trial in range(max_trials):
        target = min(1., fraction+step)
        p = model.matrices(old+(new-old)*target, full=True)
        p.initial = lambda: (U.copy(), L.copy())
        report, next_U, next_L = steady(p)
        rows.append(dict(trial=trial+1, fraction=float(target), step=float(step), steady=report))
        (folder/'results.json').write_text(json.dumps(rows, indent=2))
        if report['status'] != 'converged':
            step /= 2
            if step < 1e-3:
                break
            continue
        fraction = target
        U, L = next_U, next_L
        np.savez(folder/'steady_latest.npz', points=p.points, U=U, L=L)
        if fraction == 1.:
            return report, U, L
        step = min(.2, step*1.5)
    failure = dict(status='continuation_failed', last_converged_fraction=float(fraction),
                   attempted_solves=len(rows), last_attempt=report)
    (folder/'failure.json').write_text(json.dumps(failure, indent=2))
    return failure, U, L


def run(source, initial, root, output, cycles=3):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = CavityLifting(load_problem(12, source))
    x = np.load(initial)
    saved = np.load(root)
    U, L = saved['U'].copy(), saved['L'].copy()
    configuration = dict(cells=12, source=str(source), initial=str(initial), root=str(root),
        outer_cycles=cycles, inner_iterations=5, coefficient_bound=.25, spectral_margin=.01,
        objective='sum squared positive Re(lambda)+margin, rest and frozen full coupled cavity',
        gradient='commonE changes A,SL,DxU/B/L,DyU/B/L; advectorJac dependence on frozen U,L,g included',
        reconstruction='same full40 DF6/P2, EC=I, nearest37 sites, no PDE changes',
        warning='frozen inner objective is not moving-root stability; inspect outer recomputed spectra',
        precision='saved MPFR80 MPI8 source; optimization/corrections/global algebra Float64')
    (output/'configuration.json').write_text(json.dumps(configuration, indent=2))
    rows = []
    for cycle in range(1, cycles+1):
        folder = output/f'cycle{cycle}'
        folder.mkdir()
        model.freeze(U, L)
        check = sensitivity_check(model, x)
        (folder/'gradient_check.json').write_text(json.dumps(check, indent=2))
        history = []
        def callback(current):
            model.objective(current)
            history.append(dict(iteration=len(history)+1, **model.last))
            np.save(folder/'parameters_latest.npy', current)
            (folder/'history.json').write_text(json.dumps(history, indent=2))
            print(dict(cycle=cycle, **history[-1]), flush=True)
        old = x.copy()
        old_U, old_L = U.copy(), L.copy()
        result = minimize(model.objective, x, method='L-BFGS-B', jac=True,
            bounds=[(-.25, .25)]*len(x), callback=callback,
            options=dict(maxiter=5, maxls=12, ftol=1e-12, gtol=1e-7, maxcor=7))
        x = result.x
        np.save(folder/'parameters.npy', x)
        p = model.matrices(x, full=True)
        save_maps(p, folder/'maps')
        checks = model.checks(x)
        (folder/'checks.json').write_text(json.dumps(checks, indent=2))
        p.initial = lambda: (U.copy(), L.copy())
        root_report, U, L = steady(p)
        if root_report['status'] != 'converged':
            (folder/'direct_root_failure.json').write_text(json.dumps(root_report, indent=2))
            root_report, U, L = continue_root(model, old, x, old_U, old_L, folder/'root_continuation')
        np.savez(folder/'steady.npz', points=p.points, U=U, L=L)
        row = dict(cycle=cycle, accepted_iterations=int(result.nit), optimizer_message=str(result.message),
            frozen_objective=float(result.fun), steady=root_report, rest=spectrum(p),
            descriptor=descriptor_audit(p), pressure_controls=all_pressure_controls(p),
            reference=reference_at_nodal_lines(p.points, U),
            max_polynomial_error=max(d['polynomial_error_inf'] for d in checks),
            max_left_inverse_error=max(d['left_inverse_error_inf'] for d in checks))
        if root_report['status'] == 'converged':
            row['moving_root_spectrum'] = coupled_spectrum(p, U, L)
        rows.append(row)
        (output/'results.json').write_text(json.dumps(dict(cycles=rows, seconds=time.perf_counter()-start), indent=2))
        print(dict(cycle=cycle, root_status=root_report['status'], rest=row['rest'], moving=row.get('moving_root_spectrum')), flush=True)
        if root_report['status'] != 'converged':
            break
        if row['rest']['positive_count'] == 0 and row['moving_root_spectrum']['positive_count'] == 0:
            break
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'initial', 'root', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--cycles', type=int, choices=(1, 2, 3), default=3)
    args = ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.initial, args.root, args.output, args.cycles)
