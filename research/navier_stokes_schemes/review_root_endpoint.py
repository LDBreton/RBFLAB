"""Independent bounded diagnostic of the saved cavity continuation endpoint."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import svdvals, eigvals
from scipy.sparse import bmat, eye
from threadpoolctl import threadpool_limits
from research.navier_stokes_schemes.cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from research.navier_stokes_schemes.pressure_lifting import load_problem
from research.navier_stokes_schemes.coupled import convection
from research.navier_stokes_schemes.cavity_candidates import steady

out=Path('outputs/ns_moving_root_review_20261029');out.mkdir(exist_ok=False)
base=Path('outputs/ns_cavity_nullspace_lifting_20261029')
with threadpool_limits(limits=1):
 model=CavityLifting(load_problem(12,Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
 old=np.load('outputs/ns_nullspace_lifting_20261029/cells12_qr/parameters.npy')
 new=np.load(base/'cycle1/parameters.npy'); direction=new-old
 z=np.load(base/'cycle1/adaptive_root_continuation/steady_latest.npz')
 U,L=z['U'],z['L']; rows=[]
 for fraction in (.3194580078125,.3196,.3198,.3200):
  p=model.matrices(old+fraction*direction,full=True);p.initial=lambda:(U.copy(),L.copy())
  root,U1,L1=steady(p)
  row={'fraction':fraction,'root':root}
  if root['status']=='converged':
   U,L=U1,L1;n=len(U);g=p.boundary(0)
   C,JU,JL=convection(p,U,L,g)
   J=bmat([[p.A,-p.M],[JU,eye(n)+JL]],format='csr').toarray()
   singular=svdvals(J); row['J_sigma_min']=float(singular[-1]);row['J_cond2']=float(singular[0]/singular[-1])
   scaled=J/np.linalg.norm(J,axis=1)[:,None];scaled/=np.linalg.norm(scaled,axis=0)[None,:]
   ss=svdvals(scaled);row['equilibrated_cond2']=float(ss[0]/ss[-1])
   B=np.zeros_like(J);B[n:,:n]=np.eye(n)
   vals=eigvals(-J,B); vals=vals[np.isfinite(vals)]
   row['smallest_modulus_modes']=[[float(v.real),float(v.imag)] for v in vals[np.argsort(abs(vals))[:10]]]
   row['leading']=coupled_spectrum(p,U,L)
   h=1e-5
   def residual(fr):
    pp=model.matrices(old+fr*direction,full=True)
    return np.r_[pp.A@U-pp.M@L+pp.B@g,L+convection(pp,U,L,g,jacobian=False)[0]]
   dR=(residual(fraction+h)-residual(fraction-h))/(2*h)
   tangent=np.linalg.solve(J,-dR)
   row['dU_df_inf']=float(abs(tangent[:n]).max());row['dL_df_inf']=float(abs(tangent[n:]).max())
  rows.append(row);print(json.dumps(row),flush=True)
  (out/'diagnostics.json').write_text(json.dumps(rows,indent=2))
