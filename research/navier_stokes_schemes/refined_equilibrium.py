"""One bounded true-root spectral-descent step on a fine common-E reconstruction."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting,coupled_spectrum
from .pressure_lifting import load_problem
from .equilibrium_nullspace_lifting import root_at,total_gradient
from .metric_lifting import spectrum,save_maps
from .cavity_candidates import reference_at_nodal_lines

def run(output,steps,lid_checkpoint=None,initial=None):
    output.mkdir(parents=True,exist_ok=True)
    model=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    if lid_checkpoint:
        lq=np.load(lid_checkpoint);alpha=float(lq['alpha']);fullg=model.p.boundary(0).copy();model.p.boundary=lambda t:alpha*fullg
    checkpoint=output/'checkpoint.npz'
    if checkpoint.exists():
        q=np.load(checkpoint);x,U,L,radius=q['x'],q['U'],q['L'],float(q['radius'])
    else:
        q=np.load(lid_checkpoint or 'outputs/ns_refined_transfer_20261029/state_0p0.npz');U,L=q['U'],q['L'];x=np.load('outputs/ns_nullspace_lifting_20261029/cells16_qr_resume/parameters.npy') if lid_checkpoint else np.zeros(model.size);radius=.0005 if lid_checkpoint else .002
        if initial:x=np.load(initial)
        _,rr,U,L=root_at(model,x,U,L)
        (output/'initial_root.json').write_text(json.dumps(rr,indent=2))
        if rr['status']!='converged':raise RuntimeError(rr)
    history=json.loads((output/'history.json').read_text()) if (output/'history.json').exists() else []
    trials=json.loads((output/'trials.json').read_text()) if (output/'trials.json').exists() else []
    for _ in range(steps):
        value,grad,info,p=total_gradient(model,x,U,L)
        print(dict(base=info,radius=radius),flush=True)
        if info['rest_positive']==0 and info['cavity']['positive_count']==0:break
        direction=-grad.copy()
        direction[((x>=.25-1e-14)&(direction>0))|((x<=-.25+1e-14)&(direction<0))]=0.
        direction/=max(np.max(abs(direction)),1e-300)
        accepted=False
        for attempt in range(10):
            trial=np.clip(x+radius*direction,-.25,.25)
            pp,rr,up,lp=root_at(model,trial,U,L)
            row=dict(iteration=len(history)+1,radius=radius,root=rr)
            if rr['status']=='converged':
                model.freeze(up,lp);tv,_=model.objective(trial)
                row.update(objective=tv,details=dict(model.last),root_change=float(np.max(abs(up-U))))
                if tv<value and np.max(abs(up-U))<.1:
                    x,U,L=trial,up,lp;accepted=True
                    history.append(row);radius=min(.01,radius*1.5)
            trials.append(row)
            (output/'trials.json').write_text(json.dumps(trials,indent=2))
            if accepted:break
            radius*=.5
        np.savez(checkpoint,x=x,U=U,L=L,radius=radius)
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(dict(accepted=accepted,latest=history[-1] if history else None),flush=True)
        if not accepted:break
    p=model.matrices(x,full=True)
    final=dict(rest=spectrum(p),cavity=coupled_spectrum(p,U,L),lid_amplitude=alpha if lid_checkpoint else 1.)
    if not lid_checkpoint:final['reference']=reference_at_nodal_lines(p.points,U)
    (output/'results.json').write_text(json.dumps(final,indent=2))
    np.save(output/'parameters.npy',x);np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    mapfolder=output/f'maps_iter{len(history):03d}'
    if not mapfolder.exists():save_maps(p,mapfolder)
    print(final,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=2);ap.add_argument('--lid-checkpoint',type=Path);ap.add_argument('--initial',type=Path);args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.steps,args.lid_checkpoint,args.initial)






