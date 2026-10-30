"""Interior nodal-consistency objective, exact root adjoint, stable mode tangent.

FD4 divergence on the central interior is a consistency diagnostic, not an exact
continuous divergence error. The objective uses no reference cavity data. FD2,
full-interior FD4 and bilinear cell flux are independent reported holdouts.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from .equilibrium_nullspace_lifting import root_at
from .pressure_lifting import load_problem, all_pressure_controls
from .metric_lifting import save_maps, spectrum
from .cavity_candidates import reference_at_nodal_lines
from .coupled import convection
from .corner_accuracy_global_audit import nodal_divergence_operator, derivative, grid, fd, cell_flux, stats


def root_adjoint_action(model, p, U, L, state_gradient):
    n = len(U)
    _, JU, JL = convection(p, U, L, p.boundary(0))
    J = bmat([[p.A, -p.M], [JU, eye(n)+JL]], format='csc')
    adj = splu(J.T.tocsc()).solve(state_gradient)
    result = np.zeros(model.size)
    for owner, d in enumerate(model.data):
        upper = adj[2*owner:2*owner+2]@d['R'][:2, 37:40]
        lower = adj[n+2*owner:n+2*owner+2]
        action = upper+U[2*owner]*(lower@d['R'][4:6, 37:40])+U[2*owner+1]*(lower@d['R'][6:8, 37:40])
        result[model.offsets[owner]:model.offsets[owner+1]] = -np.outer(action, d['frozen_source']).ravel()
    return result


def consistency_objective(model, x, U, L, D, boundary, gradient=True):
    p = model.matrices(x, full=True)
    defect = D@U+boundary
    value = float(np.mean(defect**2))
    if not gradient:
        return value, None
    model.freeze(U, L)
    state_gradient = np.r_[2*(D.T@defect)/len(defect), np.zeros(len(U))]
    return value, root_adjoint_action(model, p, U, L, state_gradient)


def modal_gradients(model, x, U, L, cutoff=-.15):
    """Real-part sensitivities of each resolved mode, including root motion."""
    p = model.matrices(x, full=True)
    model.freeze(U, L)
    report, values, left, right, B, resolved = coupled_spectrum(p, U, L, True)
    active = np.flatnonzero(resolved & (values.real > cutoff) & (values.imag >= -1e-10))
    n = len(U)
    rows, modes = [], []
    for k in active:
        lam, v, w = values[k], right[:, k], left[:, k]
        du, dl = v[:n], v[n:]
        factor = -1/np.vdot(w, B@v)
        dxv, dyv = p.DxU@du+p.DxL@dl, p.DyU@du+p.DyL@dl
        wl = w[n:].conj()
        direct = np.zeros(n, dtype=complex)
        direct[::2] = (wl*dxv).reshape(-1, 2).sum(axis=1)
        direct[1::2] = (wl*dyv).reshape(-1, 2).sum(axis=1)
        zx, zy = np.repeat(du[::2], 2)*wl, np.repeat(du[1::2], 2)*wl
        state_gradient = (factor*np.r_[direct+p.DxU.T@zx+p.DyU.T@zy,
                                        p.DxL.T@zx+p.DyL.T@zy]).real
        gradient = root_adjoint_action(model, p, U, L, state_gradient)
        for owner, d in enumerate(model.data):
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
            change += np.outer(dx, U[2*owner]*source+du[2*owner]*base)
            change += np.outer(dy, U[2*owner+1]*source+du[2*owner+1]*base)
            gradient[model.offsets[owner]:model.offsets[owner+1]] += (factor*change).real.ravel()
        rows.append(gradient)
        modes.append(dict(real=float(lam.real), imag=float(lam.imag)))
    return np.asarray(rows), modes, report


def tangent_direction(gradient, constraints, modes=None, bias=.1):
    """Euclidean projection onto small intersection of modal halfspaces."""
    d0 = -gradient
    G = constraints/np.linalg.norm(constraints, axis=1)[:, None]
    bounds = np.zeros(len(G)) if modes is None else np.asarray([
        -bias*np.linalg.norm(d0)*max(0., min(1., (row['real']+.01)/.01)) for row in modes])
    best, best_distance = None, np.inf
    for count in range(len(G)+1):
        for subset in itertools.combinations(range(len(G)), count):
            if count:
                H = G[list(subset)]
                multiplier = np.linalg.lstsq(H@H.T, H@d0-bounds[list(subset)], rcond=1e-12)[0]
                if np.min(multiplier) < -1e-9: continue
                d = d0-H.T@multiplier
            else:
                d = d0.copy()
            if np.max(G@d-bounds) > 1e-9*max(np.linalg.norm(d0), 1e-20): continue
            distance = float((d-d0)@(d-d0))
            if distance < best_distance:
                best, best_distance = d, distance
    if best is None: raise RuntimeError('infeasible modal direction constraints')
    return best


def diagnostics(p, U, L):
    V = grid(p, U)
    result = {}
    for order in (2, 4):
        div = fd(V, derivative(p.cells, order))
        result[f'fd{order}_full'] = stats(div[1:-1, 1:-1])
        result[f'fd{order}_central'] = stats(div[3:-3, 3:-3])
    result['bilinear_cell_divergence'] = stats(cell_flux(V, 1/p.cells)*p.cells**2)
    g = p.boundary(0)
    dx = p.DxU@U+p.DxB@g+p.DxL@L
    dy = p.DyU@U+p.DyB@g+p.DyL@L
    result['local_analytic_divergence'] = stats(dx[::2]+dy[1::2])
    return result


def run(output, steps=15, initial=None, root=None, skip_gradient_check=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    saved = Path('outputs/ns_equilibrium_nullspace_lifting_20261029')
    model = CavityLifting(load_problem(12, Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
    x = np.load(initial or saved/'parameters.npy')
    z = np.load(root or saved/'steady.npz'); U, L = z['U'], z['L']
    D, boundary = nodal_divergence_operator(model.p, order=4, exclude_layers=2)
    value, gradient = consistency_objective(model, x, U, L, D, boundary)
    G, modes, _ = modal_gradients(model, x, U, L)
    rng = np.random.default_rng(20261029)
    direction = rng.normal(size=len(x)); direction /= np.linalg.norm(direction)
    checks = []
    leading = int(np.argmax([row['real'] for row in modes]))
    for h in (() if skip_gradient_check else (1e-5, 3e-6)):
        vals, eigvals = [], []
        for sign in (1, -1):
            xp = x+sign*h*direction
            pp, rp, up, lp = root_at(model, xp, U, L)
            if rp['status'] != 'converged': raise RuntimeError(rp)
            vals.append(consistency_objective(model, xp, up, lp, D, boundary, False)[0])
            eigvals.append(coupled_spectrum(pp, up, lp)['real'])
        numerical = (vals[0]-vals[1])/(2*h)
        eigen_numerical = (eigvals[0]-eigvals[1])/(2*h)
        checks.append(dict(h=h, objective_analytic=float(gradient@direction), objective_numerical=float(numerical),
                           objective_relative=float(abs(gradient@direction-numerical)/max(abs(numerical), 1e-10)),
                           eigen_analytic=float(G[leading]@direction), eigen_numerical=float(eigen_numerical),
                           eigen_relative=float(abs(G[leading]@direction-eigen_numerical)/max(1., abs(eigen_numerical)))))
    (output/'gradient_check.json').write_text(json.dumps(checks, indent=2))
    if checks and (min(row['objective_relative'] for row in checks) > 1e-3 or min(row['eigen_relative'] for row in checks) > 2e-4):
        raise RuntimeError(checks)
    (output/'configuration.json').write_text(json.dumps(dict(cells=12,steps=steps,objective=__doc__,
        minimized='mean squared FD4 divergence on central7x7 nodes; no boundary samples in objective stencil',
        constraints='project gradient onto nonincreasing realparts of all modes Re(lambda)>-.15, restore weak modes toward-.01 with 10percent gradient-normal bias, actual-root stable backtracking',
        source=str(saved),initial_parameters=str(initial or saved/'parameters.npy'),initial_root=str(root or saved/'steady.npz'),
        gradient_check='previously verified analytic expression; skipped this continuation batch' if skip_gradient_check else 'fresh directional differences',
        precision='saved native MPFR80 MPI8 source; Float64 common-E correction/global algebra',
        reference_usage='report after acceptance only; never fit'), indent=2))
    p = model.matrices(x, full=True)
    history = [dict(iteration=0,objective=value,diagnostics=diagnostics(p,U,L),reference=reference_at_nodal_lines(p.points,U))]
    trials=[]; radius=.003
    for iteration in range(1,steps+1):
        value, gradient = consistency_objective(model,x,U,L,D,boundary)
        G,modes,report = modal_gradients(model,x,U,L)
        for bias in (.1,.01,.001,.0001,0.):
            direction=tangent_direction(gradient,G,modes,bias)
            if float(direction@gradient)<-1e-5*np.linalg.norm(direction)*np.linalg.norm(gradient):break
        direction/=max(np.max(abs(direction)),1e-300)
        accepted=False
        for attempt in range(10):
            trial=np.clip(x+radius*direction,-.25,.25)
            pp,rp,up,lp=root_at(model,trial,U,L)
            row=dict(iteration=iteration,attempt=attempt,radius=radius,root=rp)
            if rp['status']=='converged':
                next_value,_=consistency_objective(model,trial,up,lp,D,boundary,False)
                moving,rest=coupled_spectrum(pp,up,lp),spectrum(pp)
                row.update(objective=next_value,cavity=moving,rest=rest)
                if next_value<value and moving['real']<-1e-5 and rest['real']<-1e-5 and np.max(abs(up-U))<.05:
                    x,U,L=trial,up,lp;accepted=True
                    row.update(diagnostics=diagnostics(pp,U,L),reference=reference_at_nodal_lines(pp.points,U))
                    history.append(row)
                    np.save(output/'parameters.npy',x);np.savez(output/'steady.npz',points=pp.points,U=U,L=L)
                    radius=min(.02,radius*1.5)
            trials.append(row)
            (output/'trials.json').write_text(json.dumps(trials,indent=2))
            if accepted:break
            radius*=.5
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(dict(iteration=iteration,accepted=accepted,objective=history[-1]['objective'],radius=radius,
                   real=history[-1].get('cavity',{}).get('real')),flush=True)
        if not accepted:break
    p=model.matrices(x,full=True)
    save_maps(p,output/'maps')
    np.save(output/'parameters.npy',x);np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    poly=model.checks(x)
    (output/'checks.json').write_text(json.dumps(poly,indent=2))
    c=convection(p,U,L,p.boundary(0),jacobian=False)[0]
    result=dict(accepted_steps=len(history)-1,objective=consistency_objective(model,x,U,L,D,boundary,False)[0],
                full_steady_residual_inf=float(np.max(abs(np.r_[p.A@U-p.M@L+p.B@p.boundary(0),L+c]))),
                diagnostics=diagnostics(p,U,L),cavity=coupled_spectrum(p,U,L),rest=spectrum(p),
                pressure=all_pressure_controls(p),max_polynomial_defect=max(row['polynomial_error_inf'] for row in poly),
                reference=reference_at_nodal_lines(p.points,U),seconds=time.perf_counter()-start)
    (output/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=15)
    parser.add_argument('--initial',type=Path);parser.add_argument('--root',type=Path)
    parser.add_argument('--skip-gradient-check',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.steps,args.initial,args.root,args.skip_gradient_check)
