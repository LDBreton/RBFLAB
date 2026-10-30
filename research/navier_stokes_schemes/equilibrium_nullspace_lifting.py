"""Equilibrium-aware common-E spectral optimization with an implicit root adjoint."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import bmat, eye
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits
from .cavity_nullspace_lifting import CavityLifting, coupled_spectrum
from .cavity_candidates import steady, reference_at_nodal_lines
from .coupled import convection
from .pressure_lifting import load_problem, all_pressure_controls
from .metric_lifting import save_maps, spectrum
from .nullspace_lifting import descriptor_audit


def root_at(model,x,U,L):
    p=model.matrices(x,full=True)
    p.initial=lambda:(U.copy(),L.copy())
    report,u,ell=steady(p)
    return p,report,u,ell


def total_gradient(model,x,U,L,margin=.01):
    model.freeze(U,L)
    value,grad=model.objective(x,margin)
    p=model.matrices(x,full=True)
    report,values,left,right,B,resolved=coupled_spectrum(p,U,L,True)
    n=len(U)
    state_gradient=np.zeros(2*n)
    for k in np.flatnonzero(resolved & (values.real > -margin)):
        lam,v,w=values[k],right[:,k],left[:,k]
        a,b=v[:n],v[n:]
        dxv=p.DxU@a+p.DxL@b
        dyv=p.DyU@a+p.DyL@b
        wl=w[n:].conj()
        direct=np.zeros(n,dtype=complex)
        direct[::2]=(wl*dxv).reshape(-1,2).sum(axis=1)
        direct[1::2]=(wl*dyv).reshape(-1,2).sum(axis=1)
        zx=np.repeat(a[::2],2)*wl
        zy=np.repeat(a[1::2],2)*wl
        gu=direct+p.DxU.T@zx+p.DyU.T@zy
        gl=p.DxL.T@zx+p.DyL.T@zy
        factor=-2*(lam.real+margin)/np.vdot(w,B@v)
        state_gradient+=(factor*np.r_[gu,gl]).real
    _,JU,JL=convection(p,U,L,p.boundary(0))
    J=bmat([[p.A,-p.M],[JU,eye(n)+JL]],format='csc')
    adj=spsolve(J.T,state_gradient)
    for owner,d in enumerate(model.data):
        base=d['frozen_source']
        upper=adj[2*owner:2*owner+2]@d['R'][:2,37:40]
        lower=adj[n+2*owner:n+2*owner+2]
        action=upper+U[2*owner]*(lower@d['R'][4:6,37:40])+U[2*owner+1]*(lower@d['R'][6:8,37:40])
        grad[model.offsets[owner]:model.offsets[owner+1]]-=np.outer(action,base).ravel()
    return value,grad,dict(model.last),p


def run(output,steps=15):
    output.mkdir(parents=True,exist_ok=False)
    base=Path('outputs/ns_nullspace_lifting_20261029/cells12_qr/parameters.npy')
    target=Path('outputs/ns_cavity_nullspace_lifting_20261029/cycle1/parameters.npy')
    root=Path('outputs/ns_cavity_nullspace_lifting_20261029/cycle1/adaptive_root_continuation/steady_latest.npz')
    model=CavityLifting(load_problem(12,Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
    x0=np.load(base);x=x0+.3194580078125*(np.load(target)-x0)
    saved=np.load(root);U,L=saved['U'],saved['L']
    p,rr,U,L=root_at(model,x,U,L)
    if rr['status']!='converged':raise RuntimeError(rr)
    value,grad,info,p=total_gradient(model,x,U,L)
    rng=np.random.default_rng(20261029);direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction)
    checks=[]
    for h in (1e-5,3e-6):
        vals=[]
        for sign in (1,-1):
            xp=x+sign*h*direction
            pp,rp,up,lp=root_at(model,xp,U,L)
            if rp['status']!='converged':raise RuntimeError(rp)
            model.freeze(up,lp);vals.append(model.objective(xp)[0])
        numerical=(vals[0]-vals[1])/(2*h)
        checks.append(dict(h=h,analytic=float(grad@direction),numerical=float(numerical),relative=abs(grad@direction-numerical)/max(1,abs(numerical))))
    (output/'gradient_check.json').write_text(json.dumps(checks,indent=2))
    print(dict(gradient_check=checks),flush=True)
    if min(c['relative'] for c in checks)>2e-4:raise RuntimeError('total derivative failed')
    history=[dict(iteration=0,root=rr,**info)]
    trials=[]
    radius=.002
    for iteration in range(1,steps+1):
        value,grad,info,p=total_gradient(model,x,U,L)
        if info['rest_positive']==0 and info['cavity']['positive_count']==0:break
        direction=-grad/max(np.max(abs(grad)),1e-300)
        accepted=False
        for attempt in range(min(10,60-len(trials))):
            trial=np.clip(x+radius*direction,-.25,.25)
            pp,rp,up,lp=root_at(model,trial,U,L)
            row=dict(iteration=iteration,radius=radius,root=rp)
            if rp['status']=='converged':
                model.freeze(up,lp)
                tv,tg=model.objective(trial)
                row.update(objective=tv,details=model.last)
                if tv < value and np.max(abs(up-U))<.05:
                    x,U,L=trial,up,lp
                    accepted=True
                    history.append(dict(iteration=iteration,radius=radius,root=rp,**model.last))
                    np.save(output/'parameters.npy',x)
                    np.savez(output/'steady.npz',points=pp.points,U=U,L=L)
                    radius=min(.005,radius*1.5)
            trials.append(row)
            (output/'trials.json').write_text(json.dumps(trials,indent=2))
            if accepted:break
            radius*=.5
            if radius<1e-9:break
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print(history[-1] if accepted else dict(status='no accepted step',iteration=iteration,radius=radius),flush=True)
        if not accepted:break
    p=model.matrices(x,full=True)
    np.save(output/'parameters.npy',x);np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    save_maps(p,output/'maps')
    checks=model.checks(x)
    _,final_root,U,L=root_at(model,x,U,L)
    result=dict(root=final_root,rest=spectrum(p),moving=coupled_spectrum(p,U,L),descriptor=descriptor_audit(p),pressure_controls=all_pressure_controls(p),reference=reference_at_nodal_lines(p.points,U),max_polynomial_error=max(d['polynomial_error_inf'] for d in checks),max_left_inverse_error=max(d['left_inverse_error_inf'] for d in checks),accepted_steps=len(history)-1,trials=len(trials))
    (output/'checks.json').write_text(json.dumps(checks,indent=2))
    (output/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=15)
    args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.steps)
