"""Independent high-order local consistency objective with actual-root screening.

No Ghia data enters this objective. All corrections share the existing common E,
so cardinality, analytic local DF velocity and exact DF6/P2 remain intact.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from .cavity_candidates import reference_at_nodal_lines
from .equilibrium_nullspace_lifting import root_at, total_gradient
from .metric_lifting import enriched_moments, save_maps, spectrum
from .pressure_lifting import load_problem, monomial, all_pressure_controls
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry
import rbflab as r


def prepare_accuracy(model):
    """Centered, dimensionless degree 7/8 velocity and degree 3/4 pressure.

    Each test field has unit RMS nondimensional input observation. Outputs use
    h^2/nu for momentum/pressure gradient and h for velocity derivatives. This
    measures local defects, not a global PDE error or accuracy theorem.
    """
    p = model.p
    unique, records, _ = geometry(r.geometry.unit_box_grid(p.cells))
    for owner, rec in enumerate(records):
        d = model.data[owner]
        job = unique[rec['job']]
        xy = (job['points']-job['target'])*p.cells
        codes = np.asarray(job['codes'])
        cv, tv = enriched_moments(xy, codes, 1/p.cells, p.nu, 8)
        # P8 velocity space contains 54 velocity fields; P6 contains 35.
        cv, tv = cv[:, 35:-5], tv[:, 35:-5]
        powers = [(a, degree-a) for degree in (3, 4) for a in range(degree+1)]
        cp = np.zeros((len(codes), len(powers)))
        tp = np.zeros((8, len(powers)))
        for j, (a, b) in enumerate(powers):
            for i, (site, code) in enumerate(zip(xy, codes)):
                if 2 <= code < 4:
                    component = code % 2
                    cp[i, j] = monomial(a, b, *site, 1-component, component)
            for i, code in enumerate((2, 3, 4, 5, 6, 7, 8, 9)):
                if 2 <= code < 6:
                    component = code % 2
                    tp[i, j] = monomial(a, b, 0., 0., 1-component, component)
        c, t = np.column_stack((cv, cp)), np.column_stack((tv, tp))
        normalization = np.sqrt(np.mean((c/d['D'][:, None])**2, axis=0))
        c, t = c/normalization, t/normalization
        output_scale = np.r_[np.full(4, 1/(p.nu*p.cells**2)), np.full(4, 1/p.cells)]
        d['accuracy_C'] = c
        d['accuracy_T'] = t
        d['accuracy_output_scale'] = output_scale
        d['accuracy_source'] = d['P']@(c/d['scale'][:, None])
        d['accuracy_response'] = output_scale[:, None]*d['R'][:, 37:40]


def objective(model, x, gradient=True):
    value = 0.
    grad = np.zeros_like(x)
    velocity, pressure = [], []
    for owner, d in enumerate(model.data):
        v, _ = model.local(x, owner)
        defect = d['accuracy_output_scale'][:, None]*(v@d['accuracy_C']-d['accuracy_T'])
        value += np.sum(defect**2)
        velocity.append(float(np.mean(defect[:, :19]**2)))
        pressure.append(float(np.mean(defect[:, 19:]**2)))
        if gradient:
            block = 2*d['accuracy_response'].T@defect@d['accuracy_source'].T
            grad[model.offsets[owner]:model.offsets[owner+1]] = block.ravel()
    norm = len(model.data)*8*28
    report = dict(objective=float(value/norm), velocity_high_moment_rms=float(np.sqrt(np.mean(velocity))),
                  pressure_high_moment_rms=float(np.sqrt(np.mean(pressure))))
    return value/norm, grad/norm, report


def run(output, steps=12, resume=None):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = CavityLifting(load_problem(12, Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
    prepare_accuracy(model)
    saved = resume or Path('outputs/ns_equilibrium_nullspace_lifting_20261029')
    x = np.load(saved/'parameters.npy')
    root = np.load(saved/'steady.npz')
    U, L = root['U'], root['L']
    value, grad, info = objective(model, x)
    rng = np.random.default_rng(20261029)
    direction = rng.normal(size=len(x)); direction /= np.linalg.norm(direction)
    fd = []
    for h in (1e-4, 3e-5):
        numerical = (objective(model, x+h*direction)[0]-objective(model, x-h*direction)[0])/(2*h)
        fd.append(dict(h=h, analytic=float(grad@direction), numerical=float(numerical),
                       relative=float(abs(grad@direction-numerical)/max(abs(numerical), 1e-12))))
    (output/'gradient_check.json').write_text(json.dumps(fd, indent=2))
    (output/'configuration.json').write_text(json.dumps(dict(cells=12, source=str(saved),
        objective=__doc__, normalization=prepare_accuracy.__doc__, steps=steps,
        input_fields='19 DF degree7/8 plus 9 pure pressure degree3/4 centered monomials',
        acceptance='actual-root converged, no positive rest/cavity resolved modes, accuracy decreases',
        precision='saved native MPFR80 MPI8; correction, optimization and global algebra Float64',
        reference_usage='Ghia reported only after acceptance; never used in objective or direction'), indent=2))
    if min(row['relative'] for row in fd) > 1e-4:
        raise RuntimeError(fd)
    history = [dict(iteration=0, accuracy=info, reference=reference_at_nodal_lines(model.p.points, U))]
    trials = []
    radius = .003
    for iteration in range(1, steps+1):
        value, grad, info = objective(model, x)
        _, spectral_grad, spectral_info, _ = total_gradient(model, x, U, L, margin=.01)
        # Orthogonal projection onto a nonincreasing spectral penalty halfspace.
        direction = -grad
        dot = float(direction@spectral_grad)
        if dot > 0:
            direction -= dot/max(float(spectral_grad@spectral_grad), 1e-300)*spectral_grad
        direction /= max(np.max(abs(direction)), 1e-300)
        accepted = False
        for attempt in range(9):
            trial = np.clip(x+radius*direction, -.25, .25)
            pp, rr, up, lp = root_at(model, trial, U, L)
            row = dict(iteration=iteration, attempt=attempt, radius=radius, root=rr)
            if rr['status'] == 'converged':
                moving = coupled_spectrum(pp, up, lp)
                rest = spectrum(pp)
                next_value, _, accuracy = objective(model, trial, gradient=False)
                row.update(cavity=moving, rest=rest, accuracy=accuracy)
                if next_value < value and moving['real'] < -1e-5 and rest['real'] < -1e-5 and np.max(abs(up-U)) < .05:
                    x, U, L = trial, up, lp
                    accepted = True
                    row['reference'] = reference_at_nodal_lines(pp.points, U)
                    history.append(row)
                    np.save(output/'parameters.npy', x)
                    np.savez(output/'steady.npz', points=pp.points, U=U, L=L)
                    radius = min(.01, radius*1.5)
            trials.append(row)
            (output/'trials.json').write_text(json.dumps(trials, indent=2))
            if accepted:
                break
            radius *= .5
        (output/'history.json').write_text(json.dumps(history, indent=2))
        print(dict(iteration=iteration, accepted=accepted, radius=radius, latest=history[-1]), flush=True)
        if not accepted:
            break
    p = model.matrices(x, full=True)
    np.save(output/'parameters.npy', x)
    np.savez(output/'steady.npz', points=p.points, U=U, L=L)
    save_maps(p, output/'maps')
    checks = model.checks(x)
    result = dict(accepted_steps=len(history)-1, accuracy=objective(model, x, False)[2],
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
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.steps, args.resume)
