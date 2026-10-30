"""Independent moving-root directional derivative on a scaled-lid fine candidate."""
import json,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting
from .pressure_lifting import load_problem
from .equilibrium_nullspace_lifting import total_gradient,root_at

def run(folder,lid):
    q=np.load(folder/'checkpoint.npz');x,U,L=q['x'],q['U'],q['L']
    p=load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2'));g=p.boundary(0);p.boundary=lambda t:lid*g
    model=CavityLifting(p);value,grad,info,_=total_gradient(model,x,U,L)
    rng=np.random.default_rng(20261029);d=rng.normal(size=len(x));d/=np.linalg.norm(d)
    rows=[]
    for h in (1e-5,3e-6):
        vals=[]
        for sign in (1,-1):
            xp=x+sign*h*d;pp,rr,u,l=root_at(model,xp,U,L)
            if rr['status']!='converged':raise RuntimeError(rr)
            model.freeze(u,l);vals.append(model.objective(xp)[0])
        fd=(vals[0]-vals[1])/(2*h);analytic=float(grad@d)
        rows.append(dict(h=h,analytic=analytic,numerical=float(fd),relative_error=abs(fd-analytic)/max(1.,abs(fd),abs(analytic))))
    (folder/'gradient_check_stage.json').write_text(json.dumps(dict(lid_amplitude=lid,objective=value,rows=rows),indent=2));print(rows,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--folder',type=Path,required=True);a.add_argument('--lid',type=float,required=True);q=a.parse_args()
    with threadpool_limits(limits=1):run(q.folder,q.lid)
