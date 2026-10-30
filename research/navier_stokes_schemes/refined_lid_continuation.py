"""Lid-amplitude continuation of fine saved reconstructions; exact root checks."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting,coupled_spectrum
from .corner_accuracy_continuation import transformed
from .pressure_lifting import load_problem
from .cavity_candidates import steady,reference_at_nodal_lines

def run(output,candidate,corner,attempts):
    output.mkdir(parents=True,exist_ok=True)
    model=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    path='outputs/ns_refined_transfer_20261029/parameters_transfer.npy' if candidate=='transfer' else 'outputs/ns_nullspace_lifting_20261029/cells16_qr_resume/parameters.npy'
    p=model.matrices(np.load(path),full=True)
    if corner:p,_=transformed(p,1.,.25)
    fullg=p.boundary(0).copy();U=np.zeros(model.n);L=np.zeros(model.n);alpha=0.;step=.1
    rows=[]
    if (output/'checkpoint.npz').exists():
        q=np.load(output/'checkpoint.npz');U,L=q['U'],q['L'];alpha=float(q['alpha']);step=float(q['step'])
        rows=json.loads((output/'history.json').read_text())
    for k in range(attempts):
        a=min(1.,alpha+step);p.boundary=lambda t:a*fullg
        p.initial=lambda:(U.copy(),L.copy())
        rr,u,l=steady(p);row=dict(alpha=a,step=step,root=rr)
        if rr['status']=='converged':
            U,L=u,l;alpha=a;step=min(.15,step*1.5)
            row['spectrum']=coupled_spectrum(p,U,L)
            row['max_speed']=float(np.linalg.norm(U.reshape(-1,2),axis=1).max())
            if a==1.:row['reference']=reference_at_nodal_lines(p.points,U)
        else:step*=.5
        rows.append(row)
        (output/'history.json').write_text(json.dumps(rows,indent=2))
        np.savez(output/'checkpoint.npz',U=U,L=L,alpha=alpha,step=step,points=p.points)
        print(dict(alpha=a,status=rr['status'],residual=rr.get('residual_inf'),spectrum=row.get('spectrum')),flush=True)
        if alpha==1. or step<1e-4:break
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--candidate',choices=('transfer','reststable'),required=True);ap.add_argument('--corner',action='store_true');ap.add_argument('--attempts',type=int,default=4);a=ap.parse_args()
    with threadpool_limits(limits=1):run(a.output,a.candidate,a.corner,a.attempts)
