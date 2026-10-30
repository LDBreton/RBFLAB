"""Four bounded physical-rest startup checks; exact imported BE/BDF2 mathematics."""
import inspect,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import research.navier_stokes_schemes.coupled as coupled
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
config=dict(cells=12,total_nodes=169,nu=.01,Re=100,stencil=37,source=str(SOURCE),initial='U=L=0 before impulsive unit lid; stationary corner nodes',force='zero',requested_end=1.,schemes=['oseen','oseen_newton'],steps=[.00025,.000125],speed_bound=10,precision='saved MPFR80 MPI8 sources; optimized corrections and global solves Float64',scope='nodal numerical startup evidence only',integrator_source_sha256=hashlib.sha256(inspect.getsource(coupled.evolve).encode()).hexdigest(),instrumentation='one callback inserted before state acceptance; equations unchanged')
(OUT/'configuration.json').write_text(json.dumps(config,indent=2))
rows=[];started=time.perf_counter()
with threadpool_limits(limits=1):
 for scheme in ('oseen','oseen_newton'):
  for dt in (.00025,.000125):
   folder=OUT/f'{scheme}_dt{dt}';folder.mkdir(exist_ok=False)
   p=load_problem(12,SOURCE/'maps');current.update(folder=folder,p=p,dt=dt,history=[],latest={})
   p.initial=lambda:(np.zeros_like(p.U0),np.zeros_like(p.L0))
   row=dict(scheme=scheme,dt=dt,requested_end=1.);case_start=time.perf_counter()
   try:
    result=evolve(p,scheme,dt,round(1/dt))
    speed=np.linalg.norm(result['U'].reshape(-1,2),axis=1)
    root=np.load(SOURCE/'steady.npz')
    row.update(status='completed',end=1.,max_nodal_speed=float(speed.max()),distance_to_equilibrium_U_inf=float(abs(result['U']-root['U']).max()),distance_to_equilibrium_L_inf=float(abs(result['L']-root['L']).max()),max_constraint_residual=max(d['constraint_residual_inf'] for d in result['diagnostics']),max_nonlinear_residual=max(d['nonlinear_residual_inf'] for d in result['diagnostics']))
    np.savez(folder/'final.npz',points=p.points,U=result['U'],L=result['L'],gradient=result['gradient'])
    (folder/'integrator_diagnostics.json').write_text(json.dumps(result['diagnostics'],indent=2))
   except Exception as exc:
    row.update(status='speed_bound_triggered' if 'safety_speed_bound' in str(exc) else 'failed',error=str(exc),last_accepted=current['latest'],traceback=traceback.format_exc())
   row['seconds']=time.perf_counter()-case_start;rows.append(row)
   (OUT/'results.json').write_text(json.dumps(dict(cases=rows,seconds=time.perf_counter()-started),indent=2));print(row,flush=True)
 # Compare shared saved output times without interpolating fields.
 comparisons=[]
 for scheme in ('oseen','oseen_newton'):
  for t in (.05,.1,.25,.5,1.):
   a=OUT/f'{scheme}_dt0.00025'/f'fields_T{t:g}.npz';b=OUT/f'{scheme}_dt0.000125'/f'fields_T{t:g}.npz'
   if a.exists() and b.exists():
    aa=np.load(a);bb=np.load(b)
    comparisons.append(dict(scheme=scheme,time=t,velocity_difference_inf=float(abs(aa['U']-bb['U']).max()),velocity_difference_rms=float(np.sqrt(np.mean((aa['U']-bb['U'])**2))),L_difference_inf=float(abs(aa['L']-bb['L']).max())))
 (OUT/'timestep_comparison.json').write_text(json.dumps(comparisons,indent=2))
 (OUT/'complete.json').write_text(json.dumps(dict(status='complete',cases=len(rows),seconds=time.perf_counter()-started),indent=2))
