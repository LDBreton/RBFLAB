"""Fine-grid corner-field/root probes for three saved common-E parameter choices."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting,coupled_spectrum
from .corner_accuracy_continuation import transformed
from .pressure_lifting import load_problem
from .metric_lifting import spectrum
from .cavity_candidates import steady,reference_at_nodal_lines

def run(output,candidate,amplitude):
    output.mkdir(parents=True,exist_ok=True)
    model=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    paths=dict(transfer='outputs/ns_refined_transfer_20261029/parameters_transfer.npy',reststable='outputs/ns_nullspace_lifting_20261029/cells16_qr_resume/parameters.npy')
    x=np.zeros(model.size) if candidate=='zero' else np.load(paths[candidate])
    base=model.matrices(x,full=True)
    p,_=transformed(base,amplitude,.25)
    q=np.load('outputs/ns_refined_transfer_20261029/initial.npz')
    p.initial=lambda:(q['U'].copy(),q['L'].copy())
    rr,U,L=steady(p)
    row=dict(candidate=candidate,amplitude=amplitude,radius=.25,root=rr,rest=spectrum(p))
    if rr['status']=='converged':row.update(cavity=coupled_spectrum(p,U,L),reference=reference_at_nodal_lines(p.points,U))
    tag=candidate+'_a'+str(amplitude).replace('.','p')
    (output/f'{tag}.json').write_text(json.dumps(row,indent=2))
    np.savez(output/f'{tag}.npz',points=p.points,U=U,L=L)
    print(row,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--candidate',choices=('zero','transfer','reststable'),required=True);ap.add_argument('--amplitude',type=float,default=1.)
    a=ap.parse_args()
    with threadpool_limits(limits=1):run(a.output,a.candidate,a.amplitude)
