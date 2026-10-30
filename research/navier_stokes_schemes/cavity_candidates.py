"""Bounded public mixed-LHI cavity candidates; isolated from historical evidence."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
import rbflab as r
from .cavity import Cavity
from .coupled import convection, evolve
from .reused import assemble


def candidate(cells, pressure_scale, ranks, folder):
    problem = Cavity(cells=cells, nu=.01, ranks=ranks, digits=80, build_maps=False,
                     native_timeout=900, linear_solver='gmres_scaled', global_ordering='MMD_AT_PLUS_A')
    q = sp.Rational(cells, 44)
    s = sp.Symbol('s', nonnegative=True)
    amplitude = q**4*(sp.Rational(1, 10000) if pressure_scale == 'nu2' else 1)
    problem.space = r.MixedSpace({'u': r.DivergenceFreeSpace(
        r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), polynomial_degree=6),
        'p': r.PressureSpace(r.Kernel(amplitude*sp.exp(-q*q*s), s)(), polynomial_degree=1)})
    problem.u, problem.p = problem.space.fields
    # No weight cache is used: the existing reuse helper's manifest describes
    # the fixed-amplitude baseline only. Save our exact recipe separately.
    cloud = r.geometry.unit_box_grid(cells)
    blocks, info = assemble(problem, cloud)
    for name, matrix in blocks.items():
        setattr(problem, name, matrix)
    from scipy.sparse import save_npz
    for name, matrix in blocks.items():
        save_npz(folder/(name+'.npz'), matrix)
    (folder/'assembly.json').write_text(json.dumps(info, indent=2))
    return problem


def diagnose(p):
    xygrad = np.column_stack((p.points[:, 1], p.points[:, 0])).ravel()
    xy = dict(constraint_inf=float(np.max(abs(p.M@xygrad))),
              gradient_error_inf=float(np.max(abs(p.PL@xygrad-xygrad))))
    leaked_u = spsolve(p.A.tocsc(), p.M@xygrad)
    xy.update(stationary_velocity_rms=float(np.sqrt(np.mean(leaked_u**2))),
              stationary_velocity_inf=float(np.max(abs(leaked_u))),
              stationary_gradient_error_inf=float(np.max(abs(p.PY@leaked_u+p.PL@xygrad-xygrad))))
    values, vectors = eig(-p.A.toarray(), p.M.toarray())
    finite = np.isfinite(values)
    k = np.flatnonzero(finite)[np.argmax(values[finite].real)]
    lam, v = values[k], vectors[:, k]
    residual = -p.A@v-lam*(p.M@v)
    mode = dict(real=float(lam.real), imag=float(lam.imag),
                relative_residual=float(np.linalg.norm(residual)/(np.linalg.norm(p.A.toarray())+abs(lam)*np.linalg.norm(p.M.toarray()))/np.linalg.norm(v)),
                mass_action_norm_ratio=float(np.linalg.norm(p.M@v)/np.linalg.norm(v)),
                action_relative_residual=float(np.linalg.norm(residual)/(np.linalg.norm(p.A@v)+abs(lam)*np.linalg.norm(p.M@v))),
                positive_count=int(np.sum(values[finite].real>1e-7)), finite_count=int(finite.sum()))
    return dict(pressure_xy=xy, rest_stokes_spectrum=mode)


def steady(p):
    n = len(p.U0)
    U, L = p.initial()
    g = p.boundary(0)
    history = []
    for iteration in range(35):
        C, JU, JL = convection(p, U, L, g)
        residual = np.r_[p.A@U-p.M@L+p.B@g, L+C]
        norm = float(np.max(abs(residual)))
        history.append(norm)
        if norm < 1e-9:
            return dict(status='converged', residual_inf=norm, iterations=iteration, history=history), U, L
        J = bmat([[p.A, -p.M], [JU, eye(n)+JL]], format='csc')
        delta = spsolve(J, -residual)
        damping = 1.
        while damping >= 2**-16:
            u, ell = U+damping*delta[:n], L+damping*delta[n:]
            c = convection(p, u, ell, g, jacobian=False)[0]
            trial = np.r_[p.A@u-p.M@ell+p.B@g, ell+c]
            if np.isfinite(trial).all() and np.max(abs(trial)) < norm:
                U, L = u, ell
                break
            damping *= .5
        else:
            return dict(status='line_search_failed', residual_inf=norm, history=history), U, L
    return dict(status='iteration_limit', history=history), U, L



def with_corner(p):
    """Affine remainder transformation, retaining physical nodal U and L.

    u=w+s, L=Lregular+fs. Appending a unit boundary datum carries
    every affine correction through the existing coupled equations.
    """
    import copy
    from scipy.sparse import hstack, csr_matrix
    from research.divfree_lhi_energy.cavity_corner_field import evaluate
    result = copy.copy(p)
    jets = evaluate(p.points, nu=p.nu)
    boundary_jets = evaluate(p.boundary_points, nu=p.nu)
    s = jets['velocity'].ravel()
    sb = boundary_jets['velocity'].ravel()
    fs = jets['stokes_force'].ravel()
    def append(matrix, column):
        return hstack((matrix, csr_matrix(np.asarray(column).reshape(-1, 1))), format='csr')
    result.B = append(p.B, -p.A@s-p.B@sb+p.M@fs)
    for direction in ('Dx', 'Dy'):
        jet = jets[direction.lower()].ravel()
        block = getattr(p, direction+'B')
        correction = jet-getattr(p, direction+'U')@s-block@sb-getattr(p, direction+'L')@fs
        setattr(result, direction+'B', append(block, correction))
    result.PB = append(p.PB, jets['gradpressure'].ravel()-p.PY@s-p.PB@sb-p.PL@fs)
    result.boundary = lambda t: np.r_[p.boundary(t), 1.]
    return result




def reference_at_nodal_lines(points, U):
    """Reference is linearly interpolated to computed nodes; U is untouched."""
    reference_path = Path(__file__).resolve().parents[2]/'outputs/lhi_df_cavity_diagnosis_20261027/ghia_re100.json'
    reference = json.loads(reference_path.read_text())
    velocity = U.reshape(-1, 2)
    result = dict(scope='nodal centerlines against linearly interpolated Ghia table; not true Ghia-location error', reference=str(reference_path))
    center = np.flatnonzero(np.all(np.isclose(points, .5, rtol=0, atol=1e-14), axis=1))
    if len(center):
        result['exact_table_center'] = dict(point=[.5,.5], computed=velocity[center[0]].tolist(), reference=[-.20581,.05454], absolute_difference=abs(velocity[center[0]]-[-.20581,.05454]).tolist())
    for name, axis, coordinate, component, key in [('u',0,1,0,'u_vertical_y'),('v',1,0,1,'v_horizontal_x')]:
        mask = np.isclose(points[:,axis], .5, rtol=0, atol=1e-14)
        sites = points[mask,coordinate]
        order = np.argsort(reference[key])
        target = np.interp(sites, np.asarray(reference[key])[order], np.asarray(reference[name])[order])
        computed = velocity[mask,component]
        result[name] = dict(nodes=sites.tolist(), computed=computed.tolist(), interpolated_reference=target.tolist(),
                            rms_difference=float(np.sqrt(np.mean((computed-target)**2))), max_difference=float(np.max(abs(computed-target))))
    return result


def followup(saved_root, output, cells, scale, corner=False):
    """Run corner/steady checks from exact saved matrices without native reruns."""
    from scipy.sparse import load_npz
    folder = Path(output); folder.mkdir(parents=True, exist_ok=False)
    p = Cavity(cells=cells, nu=.01, build_maps=False, linear_solver='gmres_scaled', global_ordering='MMD_AT_PLUS_A')
    source = Path(saved_root)/scale
    for name in ('A','B','M','SL','PY','PB','PL','DxU','DxB','DxL','DyU','DyB','DyL'):
        setattr(p, name, load_npz(source/(name+'.npz')))
    if corner:
        p = with_corner(p)
    elif (source/'T10_dt0.005.npz').exists():
        saved = np.load(source/'T10_dt0.005.npz')
        p.initial = lambda: (saved['U'].copy(), saved['L'].copy())
    result = dict(cells=cells, scale=scale, corner=corner, source=str(source), diagnostics=diagnose(p))
    result['steady'], U, L = steady(p)
    result['steady']['max_nodal_speed'] = float(np.linalg.norm(U.reshape(-1, 2), axis=1).max())
    np.savez(folder/'steady.npz', points=p.points, U=U, L=L)
    result['dynamics'] = []
    if corner:
        # Newton from rest can fail even for a reachable stable branch.
        # Follow the impulsive physical problem before trying Newton again.
        for dt in (.01, .005):
            try:
                state = evolve(p, 'oseen', dt, round(10/dt))
                speed = np.linalg.norm(state['U'].reshape(-1, 2), axis=1)
                result['dynamics'].append(dict(dt=dt,T=10,max_nodal_speed=float(speed.max()),diagnostics=state['diagnostics']))
                np.savez(folder/f'T10_dt{dt}.npz', points=p.points, U=state['U'], L=state['L'], gradient=state['gradient'])
            except Exception as exc:
                result['dynamics'].append(dict(dt=dt,status='failed',error=str(exc)))
                break
        if result['dynamics'] and 'state' in locals():
            p.initial = lambda: (state['U'].copy(), state['L'].copy())
            result['steady_after_dynamics'], U, L = steady(p)
    result['holds'] = []
    final_steady = result.get('steady_after_dynamics', result['steady'])
    result['final_reference'] = reference_at_nodal_lines(p.points, U)
    np.savez(folder/'final_steady.npz', points=p.points, U=U, L=L)
    if final_steady['status'] == 'converged':
        p.initial = lambda: (U.copy(), L.copy())
        for dt in (.00025, .000125):
            state = evolve(p, 'oseen', dt, round(.05/dt))
            result['holds'].append(dict(dt=dt,T=.05,initial='computed nonlinear steady state',
                velocity_change_inf=float(np.max(abs(state['U']-U))),
                gradient_change_inf=float(np.max(abs(state['gradient']-(p.PY@U+p.PB@p.boundary(0)+p.PL@L)))),
                diagnostics=state['diagnostics']))
    (folder/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('holds','dynamics','final_reference')}, indent=2), flush=True)



def saved_steady_hold(saved_root, field, output, cells, scale, corner=False):
    """Verify a saved converged steady state at both user-requested steps."""
    from scipy.sparse import load_npz
    folder = Path(output); folder.mkdir(parents=True, exist_ok=False)
    p = Cavity(cells=cells, nu=.01, build_maps=False, linear_solver='gmres_scaled', global_ordering='MMD_AT_PLUS_A')
    for name in ('A','B','M','SL','PY','PB','PL','DxU','DxB','DxL','DyU','DyB','DyL'):
        setattr(p,name,load_npz(Path(saved_root)/scale/(name+'.npz')))
    if corner:
        p = with_corner(p)
    z = np.load(field); U,L=z['U'],z['L']
    p.initial=lambda:(U.copy(),L.copy())
    C=convection(p,U,L,p.boundary(0),jacobian=False)[0]
    result=dict(cells=cells, scale=scale, corner=corner, steady_residual_inf=float(np.max(abs(np.r_[p.A@U-p.M@L+p.B@p.boundary(0),L+C]))),
                reference=reference_at_nodal_lines(p.points,U),holds=[])
    for dt in (.00025,.000125):
        state=evolve(p,'oseen',dt,round(.05/dt))
        result['holds'].append(dict(dt=dt,T=.05,velocity_change_inf=float(np.max(abs(state['U']-U))),
                                  constraint_residual_inf=max(row['constraint_residual_inf'] for row in state['diagnostics']),
                                  solved_residual_inf=max(row['solved_residual_inf'] for row in state['diagnostics'])))
    (folder/'results.json').write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!='reference'},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=12)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scales', nargs='+', default=['fixed', 'nu2'], choices=['fixed', 'nu2'])
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    recipe = dict(cells=args.cells, nu=.01, Re=100, stencil=37, velocity_degree=6, pressure_degree=1,
                  velocity='exp(-10*q^2*r^2)+q^7*r^7', q=f'{args.cells}/44', pressure='amplitude*exp(-q^2*r^2)',
                  amplitudes=dict(fixed='q^4', nu2='nu^2*q^4'), precision='local MPFR80; Float64 geometry and sparse global solves',
                  domain='[0,1]^2 uniform unit_box_grid', initial='rest', lid='unit tangential; stationary corner nodes',
                  scheme='BE/BDF2 Oseen', ranks=args.ranks, rounding='none', corner_subtraction=False)
    (root/'configuration.json').write_text(json.dumps(recipe, indent=2))
    rows = []
    with threadpool_limits(limits=1):
        for scale in args.scales:
            folder = root/scale; folder.mkdir()
            row = dict(scale=scale)
            try:
                start = time.perf_counter()
                p = candidate(args.cells, scale, args.ranks, folder)
                row['assembly_seconds'] = time.perf_counter()-start
                row.update(diagnose(p))
                row['steady'], U, L = steady(p)
                row['steady']['max_nodal_speed'] = float(np.linalg.norm(U.reshape(-1, 2), axis=1).max())
                np.savez(folder/'steady.npz', points=p.points, U=U, L=L)
                # A positive semidiscrete mode rejects this spatial candidate;
                # a small time step cannot cure that instability.
                if row['rest_stokes_spectrum']['real'] > 1e-7:
                    row['dynamics'] = dict(status='rejected_positive_rest_stokes_mode')
                else:
                    row['dynamics'] = []
                    for dt in (.01, .005):
                        state = evolve(p, 'oseen', dt, round(10/dt))
                        speed = np.linalg.norm(state['U'].reshape(-1, 2), axis=1)
                        result = dict(dt=dt, T=10, max_speed=float(speed.max()), kinetic_nodal_mean=float(.5*np.mean(speed**2)), diagnostics=state['diagnostics'])
                        row['dynamics'].append(result)
                        np.savez(folder/f'T10_dt{dt}.npz', points=p.points, U=state['U'], L=state['L'], gradient=state['gradient'])
                row['status'] = 'completed'
            except Exception as exc:
                row.update(status='failed', error=str(exc), traceback=traceback.format_exc())
            rows.append(row)
            (root/'results.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps({k:v for k,v in row.items() if k not in ('traceback','dynamics')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
