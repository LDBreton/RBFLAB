"""Saved-root local nonlinear dynamics validation; no startup/convergence claim."""
import contextlib, copy, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from scipy.sparse import bmat,eye
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem,pressure_problem
from .coupled import evolve,convection


def run(source,output,end=1.):
 output.mkdir(parents=True,exist_ok=True)
 p=load_problem(12,source/'maps');saved=np.load(source/'steady.npz');U,L=saved['U'],saved['L'];n=len(U)
 _,JU,JL=convection(p,U,L,p.boundary(0));K=bmat([[p.A,-p.M],[JU,eye(n)+JL]],format='csr').toarray();B=np.zeros_like(K);B[n:,:n]=np.eye(n)
 vals,left,right=eig(-K,B,left=True,right=True);mass=np.linalg.norm(B@right,axis=0)/np.linalg.norm(right,axis=0);resolved=np.isfinite(vals)&(mass>1e-8)
 active=np.flatnonzero(resolved);lead=active[np.argmax(vals[active].real)];lam=vals[lead]
 v=right[:,lead]*np.exp(-1j*np.angle(right[np.argmax(abs(right[:n,lead])),lead]));v/=np.max(abs(v[:n].real))
 rng=np.random.default_rng(20261029);coeff=np.zeros(len(vals),complex)
 # Real random combination of all finite eigenmodes and its exact linear propagator.
 coeff[active]=rng.normal(size=len(active))
 rv=(right@coeff).real;coeff/=np.max(abs(rv[:n]));rv=(right@coeff).real
 specs=[('hold',None,0.,.05),('pressure_xy',None,0.,.05)]
 specs += [(kind,vector,amp,end) for kind,vector in [('leading',v.real),('random',rv)] for amp in (1e-7,1e-3)]
 rows=json.loads((output/'results.json').read_text())['cases'] if (output/'results.json').exists() else [];start=time.perf_counter()
 for kind,vector,amplitude,T in specs:
  for scheme in ('oseen','oseen_newton'):
   for dt in (.00025,.000125):
    key=f'{kind}_{amplitude}_{scheme}_{dt}'
    if (output/(key+'.npz')).exists():continue
    q=pressure_problem(p) if kind=='pressure_xy' else copy.copy(p)
    if kind!='pressure_xy':
     initial=np.r_[U,L] if vector is None else np.r_[U,L]+amplitude*vector
     q.initial=lambda initial=initial:(initial[:n].copy(),initial[n:].copy())
    row=dict(control=kind,amplitude=amplitude,scheme=scheme,dt=dt,T=T)
    try:
     with (output/(key+'.log')).open('w') as log,contextlib.redirect_stdout(log):
      result=evolve(q,scheme,dt,round(T/dt),rtol=1e-12,atol=1e-13)
     row.update(status='completed',velocity_inf=float(max(abs(result['U']))),seconds=time.perf_counter()-start,
       max_constraint=max(d['constraint_residual_inf'] for d in result['diagnostics']),max_nonlinear_residual=max(d['nonlinear_residual_inf'] for d in result['diagnostics']),max_solved_residual=max(d['solved_residual_inf'] for d in result['diagnostics']))
     if kind=='pressure_xy':row['gradient_error_inf']=float(max(abs(result['gradient']-q.force(0))))
     else:
      delta=result['U']-U;row['velocity_change_inf']=float(max(abs(delta)))
      if vector is not None:
       linear=(np.exp(lam*T)*v[:n]).real if kind=='leading' else (right[:n]@(np.where(resolved,np.exp(np.where(resolved,vals,0)*T),0)*coeff)).real
       row.update(velocity_amplification_inf=float(max(abs(delta))/amplitude),linear_phase_aware_amplification=float(max(abs(linear))),linear_prediction_error_inf=float(max(abs(delta-amplitude*linear))),perturbation_kinetic_mean=float(.5*np.mean(np.sum(delta.reshape(-1,2)**2,axis=1))))
     np.savez(output/(key+'.npz'),U=result['U'],L=result['L'],gradient=result['gradient'])
    except Exception as exc:row.update(status='failed',error=str(exc))
    rows.append(row);(output/'results.json').write_text(json.dumps(dict(leading=dict(real=lam.real,imag=lam.imag),cases=rows,seconds=time.perf_counter()-start),indent=2));print(row,flush=True)
 return rows

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--end',type=float,default=1.)
 args=ap.parse_args()
 with threadpool_limits(limits=1):run(args.source,args.output,args.end)

