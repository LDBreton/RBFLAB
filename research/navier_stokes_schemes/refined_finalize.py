"""Preserve a rejected actual unit-lid fine root and test last stage coefficients."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting,coupled_spectrum
from .pressure_lifting import load_problem,all_pressure_controls
from .metric_lifting import save_maps,spectrum
from .cavity_candidates import steady,reference_at_nodal_lines

def run():
    output=Path('outputs/ns_refined_rejected_unitlid_20261029');output.mkdir(parents=True,exist_ok=True)
    model=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    p=model.matrices(np.zeros(model.size),full=True)
    saved=np.load('outputs/ns_refined_transfer_20261029/state_0p0.npz');U,L=saved['U'],saved['L']
    p.initial=lambda:(U.copy(),L.copy());rr,U,L=steady(p)
    if rr['status']!='converged':raise RuntimeError(rr)
    np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    np.save(output/'parameters.npy',np.zeros(model.size))
    save_maps(p,output/'maps')
    checks=model.checks(np.zeros(model.size))
    result=dict(status='rejected_positive_unitlid_modes',lid_amplitude=1.,cells=16,total_nodes=289,interior_nodes=225,root=rr,rest=spectrum(p),cavity=coupled_spectrum(p,U,L),reference=reference_at_nodal_lines(p.points,U),pressure=all_pressure_controls(p),polynomial_error_inf=max(d['polynomial_error_inf'] for d in checks))
    (output/'results.json').write_text(json.dumps(result,indent=2));print(dict(baseline=result['cavity']),flush=True)
    q=np.load('outputs/ns_refined_lid_equilibrium_20261029/checkpoint.npz');x=q['x'];pp=model.matrices(x,full=True)
    pp.initial=lambda:(q['U'].copy(),q['L'].copy());rr,u,l=steady(pp)
    trial=dict(source_lid=.325,target_lid=1.,root=rr,rest=spectrum(pp))
    if rr['status']=='converged':trial.update(cavity=coupled_spectrum(pp,u,l),reference=reference_at_nodal_lines(pp.points,u))
    np.save(output/'partial_lid_coefficients_full_lid.npy',x)
    np.savez(output/'partial_lid_coefficients_full_lid_state.npz',points=pp.points,U=u,L=l)
    (output/'partial_lid_coefficients_full_lid.json').write_text(json.dumps(trial,indent=2));print(trial,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
