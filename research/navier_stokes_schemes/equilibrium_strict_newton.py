"""Strict absolute-residual implicit Newton checks, saved equilibrium random perturbation."""
import argparse,contextlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from scipy.sparse import bmat,eye
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem
from .coupled import evolve,convection

def run(output,dt):
 output.mkdir(parents=True,exist_ok=True)
 p=load_problem(12,Path('outputs/ns_equilibrium_nullspace_lifting_20261029/maps'));s=np.load('outputs/ns_equilibrium_nullspace_lifting_20261029/steady.npz');U,L=s['U'],s['L'];n=len(U)
 _,JU,JL=convection(p,U,L,p.boundary(0));K=bmat([[p.A,-p.M],[JU,eye(n)+JL]],format='csr').toarray();B=np.zeros_like(K);B[n:,:n]=np.eye(n)
 vals,right=eig(-K,B);mass=np.linalg.norm(B@right,axis=0)/np.linalg.norm(right,axis=0);active=np.flatnonzero(np.isfinite(vals)&(mass>1e-8));rng=np.random.default_rng(20261029);co=np.zeros(len(vals),complex);co[active]=rng.normal(size=len(active));v=(right@co).real;v/=max(abs(v[:n]));p.initial=lambda:(U+.001*v[:n],L+.001*v[n:])
 rows=[]
 for atol in (2e-11,5e-11):
  row=dict(control='random',amplitude=.001,scheme='oseen_newton',dt=dt,T=.1,rtol=0,atol=atol)
  try:
   with (output/f'{dt}_{atol}.log').open('w') as log,contextlib.redirect_stdout(log):res=evolve(p,'oseen_newton',dt,round(.1/dt),rtol=0,atol=atol)
   diag=res['diagnostics'];row.update(status='completed',total_newton_corrections=sum(d['newton_iterations'] for d in diag),corrected_steps=sum(d['newton_iterations']>0 for d in diag),max_iterations_per_step=max(d['newton_iterations'] for d in diag),max_nonlinear_residual=max(d['nonlinear_residual_inf'] for d in diag),max_constraint=max(d['constraint_residual_inf'] for d in diag),max_velocity_change=float(max(abs(res['U']-U))))
   np.savez(output/f'{dt}_{atol}.npz',U=res['U'],L=res['L'],gradient=res['gradient']);(output/f'{dt}_{atol}_steps.json').write_text(json.dumps(diag,indent=2))
  except Exception as exc:row.update(status='failed',error=str(exc))
  rows.append(row);(output/f'{dt}_results.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
  if row['status']=='completed':break

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dt',type=float,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 with threadpool_limits(limits=1):run(args.output,args.dt)
