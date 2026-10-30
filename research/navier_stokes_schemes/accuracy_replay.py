"""Reconstruct an earlier accepted checkpoint from the saved deterministic path.

No search or new parameter selection. Accepted radii are read from saved history;
each recomputed root/objective must match the historical accepted record.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .accuracy_divergence import consistency_objective, modal_gradients, tangent_direction, diagnostics
from .cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from .equilibrium_nullspace_lifting import root_at
from .metric_lifting import save_maps, spectrum
from .pressure_lifting import load_problem, all_pressure_controls
from .corner_accuracy_global_audit import nodal_divergence_operator
from .cavity_candidates import reference_at_nodal_lines
from .coupled import convection


def run(output, count=4):
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path('outputs/ns_equilibrium_nullspace_lifting_20261029')
    trajectory = Path('outputs/ns_accuracy_divergence_20261029')
    history = json.loads((trajectory/'history.json').read_text())
    model = CavityLifting(load_problem(12, Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
    x = np.load(baseline/'parameters.npy')
    z = np.load(baseline/'steady.npz'); U, L = z['U'], z['L']
    D, b = nodal_divergence_operator(model.p, order=4, exclude_layers=2)
    rows = []
    for row in history[1:count+1]:
        _, gradient = consistency_objective(model, x, U, L, D, b)
        G, _, _ = modal_gradients(model, x, U, L)
        # The original trajectory used zero normal bias, before the separate
        # modal-restoration continuation was introduced.
        direction = tangent_direction(gradient, G)
        direction /= max(np.max(abs(direction)), 1e-300)
        x = np.clip(x+row['radius']*direction, -.25, .25)
        p, rr, U, L = root_at(model, x, U, L)
        if rr['status'] != 'converged': raise RuntimeError(rr)
        value, _ = consistency_objective(model, x, U, L, D, b, False)
        discrepancy = abs(value-row['objective'])
        if discrepancy > 1e-10: raise RuntimeError(dict(row=row, actual=value, discrepancy=discrepancy))
        rows.append(dict(iteration=row['iteration'], radius=row['radius'],
                         recorded_objective=row['objective'], replayed_objective=value,
                         objective_discrepancy=discrepancy, root=rr))
    p = model.matrices(x, full=True)
    np.save(output/'parameters.npy', x)
    np.savez(output/'steady.npz', points=p.points, U=U, L=L)
    save_maps(p, output/'maps')
    poly = model.checks(x)
    (output/'checks.json').write_text(json.dumps(poly, indent=2))
    c = convection(p, U, L, p.boundary(0), jacobian=False)[0]
    result = dict(status='deterministic accepted-path replay', replayed_steps=count,
        objective=consistency_objective(model, x, U, L, D, b, False)[0],
        full_steady_residual_inf=float(np.max(abs(np.r_[p.A@U-p.M@L+p.B@p.boundary(0), L+c]))),
        cavity=coupled_spectrum(p, U, L), rest=spectrum(p), pressure=all_pressure_controls(p),
        diagnostics=diagnostics(p, U, L), reference=reference_at_nodal_lines(p.points, U),
        max_polynomial_defect=max(row['polynomial_error_inf'] for row in poly),
        max_left_inverse_defect=max(row['left_inverse_error_inf'] for row in poly))
    (output/'results.json').write_text(json.dumps(result, indent=2))
    (output/'replay.json').write_text(json.dumps(rows, indent=2))
    (output/'configuration.json').write_text(json.dumps(dict(baseline=str(baseline), history=str(trajectory/'history.json'),
        replay_steps=count, purpose=__doc__, source='same saved public native MPFR80/MPI8 maps; Float64 replay',
        selection='early checkpoint chosen after observing Ghia diagnostic improvement; no Ghia enters optimizer; not unbiased validation'), indent=2))
    print(dict(replayed_steps=count, objective=result['objective'], cavity=result['cavity'],
               reference_rms={k:result['reference'][k]['rms_difference'] for k in ('u', 'v')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1):run(args.output, args.steps)
