from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from research.navier_stokes_schemes.equilibrium_nullspace_lifting import CavityLifting,load_problem,root_at,total_gradient
from research.navier_stokes_schemes.cavity_nullspace_lifting import coupled_spectrum
from research.navier_stokes_schemes.metric_lifting import save_maps,spectrum
from research.navier_stokes_schemes.pressure_lifting import all_pressure_controls
from research.navier_stokes_schemes.cavity_candidates import reference_at_nodal_lines
with threadpool_limits(limits=1):
 source=Path('outputs/ns_equilibrium_nullspace_lifting_20261029');out=Path('outputs/ns_equilibrium_margin_lifting_20261029');out.mkdir(exist_ok=False)
 model=CavityLifting(load_problem(12,Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
 x=np.load(source/'parameters.npy');s=np.load(source/'steady.npz');U,L=s['U'],s['L'];history=[];radius=2e-5
 for it in range(5):
  value,grad,info,p=total_gradient(model,x,U,L,margin=.05)
  print(dict(iteration=it,**info),flush=True)
  if info['cavity']['real'] < -.05 and info['rest_real'] < -.05:break
  direction=-grad/max(abs(grad));accepted=False
  for attempt in range(8):
   xp=np.clip(x+radius*direction,-.25,.25);pp,rp,up,lp=root_at(model,xp,U,L)
   row=dict(iteration=it,attempt=attempt,radius=radius,root=rp)
   if rp['status']=='converged':
    model.freeze(up,lp);v,g=model.objective(xp,margin=.05);row.update(objective=v,details=model.last)
    if v < value and max(abs(up-U)) < .05:
     x,U,L=xp,up,lp;accepted=True;radius=min(1e-4,radius*1.5)
   history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
   if accepted:break
   radius*=.5
  if not accepted:break
 p,rr,U,L=root_at(model,x,U,L)
 np.save(out/'parameters.npy',x);np.savez(out/'steady.npz',U=U,L=L,points=p.points);save_maps(p,out/'maps')
 checks=model.checks(x)
 result=dict(root=rr,moving=coupled_spectrum(p,U,L),rest=spectrum(p),pressure=all_pressure_controls(p),reference=reference_at_nodal_lines(p.points,U),max_moment=max(d['polynomial_error_inf'] for d in checks),max_left_inverse=max(d['left_inverse_error_inf'] for d in checks))
 (out/'results.json').write_text(json.dumps(result,indent=2));(out/'checks.json').write_text(json.dumps(checks,indent=2));print(result,flush=True)
