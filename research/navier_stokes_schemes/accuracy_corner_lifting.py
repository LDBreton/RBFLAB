"""Corner singular-field continuation with true-root common-E stabilization.

Physical U and L remain global states. The corner field enters through affine
maps; all spectral derivatives include the x dependence of those affine maps.
No reference cavity data enters the optimization or amplitude continuation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from .corner_accuracy_continuation import transformed
from .equilibrium_nullspace_lifting import root_at, total_gradient
from .cavity_candidates import reference_at_nodal_lines
from .metric_lifting import save_maps, spectrum
from .pressure_lifting import load_problem, all_pressure_controls
from research.divfree_lhi_energy.cavity_corner_field import evaluate


class CornerLifting(CavityLifting):
    def __init__(self, p, amplitude=.01, radius=.25):
        super().__init__(p)
        self.amplitude, self.radius = amplitude, radius
        j = evaluate(p.points, nu=p.nu, radius=radius)
        b = evaluate(p.boundary_points, nu=p.nu, radius=radius)
        self.corner_U = j['velocity'].ravel()
        self.corner_L = j['stokes_force'].ravel()
        self.corner_g = b['velocity'].ravel()

    def matrices(self, x, full=False):
        # transformed requires pressure and both derivative maps.
        base = super().matrices(x, full=True)
        return transformed(base, self.amplitude, self.radius)[0]

    def freeze(self, U, L):
        self.frozen_U, self.frozen_L = U.copy(), L.copy()
        g = self.p.boundary(0)
        remainder = (U-self.amplitude*self.corner_U,
                     g-self.amplitude*self.corner_g,
                     L-self.amplitude*self.corner_L)
        for d in self.data:
            q = np.zeros(len(d['C']))
            for group, field in enumerate(remainder):
                mask, cols = d['groups'][group]
                q[mask] = field[cols]
            d['frozen_source'] = d['P']@(q/d['scale'])


def run(output, steps=25, amplitude=.01, radius=.25, initial=None, root=None, lbfgs=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    saved = Path('outputs/ns_equilibrium_nullspace_lifting_20261029')
    x = np.load(initial or saved/'parameters.npy')
    z = np.load(root or saved/'steady.npz'); U, L = z['U'], z['L']
    model = CornerLifting(load_problem(12, Path('outputs/ns_cavity_candidates_20261029_native/nu2')),
                          amplitude=0., radius=radius)
    roots = []
    for a in ([amplitude] if initial else np.linspace(0., amplitude, 6)):
        model.amplitude = float(a)
        p, rr, U, L = root_at(model, x, U, L)
        roots.append(dict(amplitude=float(a), root=rr))
        if rr['status'] != 'converged':
            (output/'initial_root_failure.json').write_text(json.dumps(roots, indent=2))
            raise RuntimeError(roots[-1])
    value, grad, info, p = total_gradient(model, x, U, L)
    rng = np.random.default_rng(20261029)
    direction = rng.normal(size=len(x)); direction /= np.linalg.norm(direction)
    fd = []
    for h in (1e-5, 3e-6):
        vals = []
        for sign in (1, -1):
            xp = x+sign*h*direction
            pp, rp, up, lp = root_at(model, xp, U, L)
            if rp['status'] != 'converged': raise RuntimeError(rp)
            model.freeze(up, lp)
            vals.append(model.objective(xp)[0])
        numerical = (vals[0]-vals[1])/(2*h)
        fd.append(dict(h=h, analytic=float(grad@direction), numerical=float(numerical),
                       relative=float(abs(grad@direction-numerical)/max(1., abs(numerical)))))
    (output/'gradient_check.json').write_text(json.dumps(fd, indent=2))
    if min(row['relative'] for row in fd) > 2e-4:
        raise RuntimeError(fd)
    config = dict(amplitude=amplitude, radius=radius, cells=12, steps=steps,
        initial_parameters=str(initial or saved/'parameters.npy'), initial_root=str(root or saved/'steady.npz'),
        reconstruction=__doc__, derivative='frozen_source uses U-a*s,g-a*sb,L-a*fs; actual-root adjoint',
        objective='squared positive real part + .01 margin for rest and actual corner-cavity root',
        scope='same37 sites,DF6/P2,local pressure elimination; coefficient-bound .25; Float64 correction/global algebra',
        reference_usage='diagnostic after accepted steps only, no Ghia fitting',
        search_direction='bounded L-BFGS inverse Hessian, seven pairs' if lbfgs else 'scaled negative true-root gradient')
    (output/'configuration.json').write_text(json.dumps(config, indent=2))
    history = [dict(iteration=0, amplitude=amplitude, root=roots[-1], **info)]
    trials = []
    step_radius = .003
    pairs = []
    previous_x = previous_gradient = None
    for iteration in range(1, steps+1):
        value, grad, info, p = total_gradient(model, x, U, L)
        if info['rest_positive'] == 0 and info['cavity']['positive_count'] == 0:
            break
        if lbfgs and previous_x is not None:
            s, y = x-previous_x, grad-previous_gradient
            curvature = float(s@y)
            if curvature > 1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                pairs.append((s, y, 1/curvature))
                pairs = pairs[-7:]
        direction = grad.copy()
        alphas = []
        for s, y, rho in reversed(pairs):
            a = rho*float(s@direction)
            alphas.append(a)
            direction -= a*y
        if pairs:
            s, y, _ = pairs[-1]
            direction *= float(s@y)/float(y@y)
        for (s, y, rho), a in zip(pairs, reversed(alphas)):
            direction += s*(a-rho*float(y@direction))
        direction = -direction
        if float(direction@grad) >= 0:
            direction = -grad
            pairs = []
        direction /= max(np.max(abs(direction)), 1e-300)
        previous_x, previous_gradient = x.copy(), grad.copy()
        accepted = False
        for attempt in range(9):
            xp = np.clip(x+step_radius*direction, -.25, .25)
            pp, rp, up, lp = root_at(model, xp, U, L)
            row = dict(iteration=iteration, radius=step_radius, root=rp)
            if rp['status'] == 'converged':
                model.freeze(up, lp)
                next_value, _ = model.objective(xp)
                row.update(details=model.last)
                if next_value < value and np.max(abs(up-U)) < .1:
                    x, U, L = xp, up, lp
                    accepted = True
                    history.append(dict(iteration=iteration, amplitude=amplitude, root=rp, **model.last))
                    np.save(output/'parameters.npy', x)
                    np.savez(output/'steady.npz', points=pp.points, U=U, L=L)
                    step_radius = min(.01, step_radius*1.5)
            trials.append(row)
            (output/'trials.json').write_text(json.dumps(trials, indent=2))
            if accepted: break
            step_radius *= .5
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(dict(iteration=iteration, accepted=accepted, radius=step_radius, latest=history[-1]), flush=True)
        if not accepted: break
    p = model.matrices(x, full=True)
    np.save(output/'parameters.npy', x)
    np.savez(output/'steady.npz', points=p.points, U=U, L=L)
    save_maps(p, output/'maps')
    checks = model.checks(x)
    result = dict(amplitude=amplitude, radius=radius, root_residual_inf=float(np.max(abs(p.A@U-p.M@L+p.B@p.boundary(0)))),
                  cavity=coupled_spectrum(p, U, L), rest=spectrum(p),
                  reference=reference_at_nodal_lines(p.points, U), pressure=all_pressure_controls(p),
                  max_polynomial_defect=max(row['polynomial_error_inf'] for row in checks),
                  max_left_inverse_defect=max(row['left_inverse_error_inf'] for row in checks),
                  seconds=time.perf_counter()-start)
    (output/'checks.json').write_text(json.dumps(checks, indent=2))
    (output/'results.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--amplitude', type=float, default=.01)
    parser.add_argument('--radius', type=float, default=.25)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--lbfgs', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.steps, args.amplitude, args.radius, args.initial, args.root, args.lbfgs)
