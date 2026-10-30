"""Four bounded physical-rest startup checks; exact imported BE/BDF2 mathematics."""
import inspect,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import research.navier_stokes_schemes.equilibrium_segment_engine as coupled
from research.navier_stokes_schemes.pressure_lifting import load_problem
OUT=Path('outputs/ns_equilibrium_cavity_startup_20261029')
SOURCE=Path('outputs/ns_equilibrium_nullspace_lifting_20261029')
text=inspect.getsource(coupled.evolve)
needle='        Uold, U = U, candidate[:n]'
text=text.replace(needle,'        startup_monitor(step, t, candidate[:n], candidate[n:])\n'+needle)
scope=dict(coupled.__dict__)
current={}
def startup_monitor(step,t,U,L):
 speed=float(np.linalg.norm(U.reshape(-1,2),axis=1).max())
 row=dict(step=step,time=t,max_speed=speed,velocity_rms=float(np.sqrt(np.mean(U**2))))
 current['latest']=row
 if step==1 or step%100==0:
  current['history'].append(row)
  (current['folder']/'progress.json').write_text(json.dumps(current['history'],indent=2))
 if any(abs(t-s)<current['dt']/4 for s in (.05,.1,.25,.5,1.)):
  np.savez(current['folder']/f'fields_T{t:g}.npz',points=current['p'].points,U=U,L=L)
 if speed>10:
  np.savez(current['folder']/'speed_bound_state.npz',points=current['p'].points,U=U,L=L)
  raise RuntimeError(f'safety_speed_bound_exceeded at step={step},time={t},max_nodal_speed={speed}; numerical bound, not physical blowup theorem')
scope['startup_monitor']=startup_monitor
exec(text,scope)
evolve=scope['evolve']

import argparse
ap=argparse.ArgumentParser();ap.add_argument("--scheme");ap.add_argument("--dt",type=float);args=ap.parse_args()
started=time.perf_counter()
rows=json.loads((OUT/'results.json').read_text())['cases'] if (OUT/'results.json').exists() else []
completed={(r['scheme'],r['dt']) for r in rows}
with threadpool_limits(limits=1):
 for scheme in ('oseen','oseen_newton'):
  for dt in (.00025,.000125):
   if args.scheme and (scheme!=args.scheme or dt!=args.dt):continue
   existing=OUT/f'segmented_{scheme}_dt{dt}'/'result.json'
   if existing.exists():continue
   folder=OUT/f'segmented_{scheme}_dt{dt}';folder.mkdir(exist_ok=True)
   p=load_problem(12,SOURCE/'maps')
   current.update(folder=folder,p=p,dt=dt,history=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else [],latest={})
   cpfile=folder/'checkpoint.npz';history=None;startstep=1
   if cpfile.exists():
    cp=np.load(cpfile);history=(cp['Uold'],cp['Lold'],int(cp['next_step']));startstep=int(cp['next_step'])
    p.initial=lambda:(cp['U'].copy(),cp['L'].copy())
   else:p.initial=lambda:(np.zeros_like(p.U0),np.zeros_like(p.L0))
   count=min(1200 if scheme=='oseen' else (900 if dt==.000125 else 600),round(1/dt)-startstep+1)
   row=dict(scheme=scheme,dt=dt,requested_end=1.)
   try:
    import contextlib
    with (folder/'run.log').open('a') as log,contextlib.redirect_stdout(log):
     result=evolve(p,scheme,dt,count,_history=history)
    uold,lold,nextstep=result['_history']
    np.savez(cpfile,U=result['U'],L=result['L'],Uold=uold,Lold=lold,next_step=nextstep)
    diagnostics=json.loads((folder/'integrator_diagnostics.json').read_text()) if (folder/'integrator_diagnostics.json').exists() else []
    diagnostics+=result['diagnostics'];(folder/'integrator_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    if nextstep<=round(1/dt):
     print(dict(scheme=scheme,dt=dt,next_step=nextstep,time=(nextstep-1)*dt,seconds=time.perf_counter()-started),flush=True)
     raise SystemExit(0)
    speed=np.linalg.norm(result['U'].reshape(-1,2),axis=1);root=np.load(SOURCE/'steady.npz')
    row.update(status='completed',end=1.,max_nodal_speed=float(speed.max()),distance_to_equilibrium_U_inf=float(abs(result['U']-root['U']).max()),distance_to_equilibrium_L_inf=float(abs(result['L']-root['L']).max()),max_constraint_residual=max(d['constraint_residual_inf'] for d in diagnostics),max_nonlinear_residual=max(d['nonlinear_residual_inf'] for d in diagnostics))
    np.savez(folder/'final.npz',points=p.points,U=result['U'],L=result['L'],gradient=result['gradient'])
   except Exception as exc:
    row.update(status='speed_bound_triggered' if 'safety_speed_bound' in str(exc) else 'failed',error=str(exc),last_accepted=current['latest'],traceback=traceback.format_exc())
   row['last_chunk_seconds']=time.perf_counter()-started;rows.append(row)
   (folder/'result.json').write_text(json.dumps(row,indent=2));print(row,flush=True)
   raise SystemExit(0)
 rows=[json.loads(p.read_text()) for p in OUT.glob('segmented_*/result.json')]
 (OUT/'results.json').write_text(json.dumps(dict(cases=rows),indent=2))
 if len(rows)!=4:raise SystemExit(0)
 comparisons=[]
 for scheme in ('oseen','oseen_newton'):
  for t in (.05,.1,.25,.5,1.):
   a=OUT/f'segmented_{scheme}_dt0.00025'/f'fields_T{t:g}.npz';b=OUT/f'segmented_{scheme}_dt0.000125'/f'fields_T{t:g}.npz'
   if a.exists() and b.exists():
    aa=np.load(a);bb=np.load(b);comparisons.append(dict(scheme=scheme,time=t,velocity_difference_inf=float(abs(aa['U']-bb['U']).max()),velocity_difference_rms=float(np.sqrt(np.mean((aa['U']-bb['U'])**2))),L_difference_inf=float(abs(aa['L']-bb['L']).max())))
 (OUT/'timestep_comparison.json').write_text(json.dumps(comparisons,indent=2));(OUT/'complete.json').write_text(json.dumps(dict(status='complete',cases=len(rows)),indent=2));print('all4complete',flush=True)
