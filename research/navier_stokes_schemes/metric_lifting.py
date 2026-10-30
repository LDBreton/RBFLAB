"""Bounded common-field observation-metric search; no PDE modification.

All maps are evaluations of F0+(Q-F0 C)E, EC=I.  Only the norm defining
the common left inverse changes.  Saved MPFR80/MPI8 maps are the input;
the correction, geometry and generalized eigensolves remain Float64.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eig, lstsq
from scipy.sparse import coo_matrix, eye, save_npz
from threadpoolctl import threadpool_limits
import rbflab as r
from .pressure_lifting import NAMES, load_problem, moments, monomial, all_pressure_controls
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry


def enriched_moments(xy,codes,h,nu,degree):
    """Same physical fields/scaling as original moments, enriched DF velocity."""
    if degree==6:
        return moments(xy,codes,h,nu)
    powers=[(a,d-a) for d in range(1,degree+2) for a in range(d+1)]
    pressure=[(a,d-a) for d in range(1,3) for a in range(d+1)]
    def evaluate(x,y,code):
        row=[]; component=code%2
        for a,b in powers:
            def v(dx=0,dy=0):
                return (b*monomial(a,b-1,x,y,dx,dy) if component==0 and b
                    else -a*monomial(a-1,b,x,y,dx,dy) if component==1 and a else 0.)
            row.append(v() if code<2 else -nu/h**2*(v(2,0)+v(0,2)) if code<4
                else 0. if code<6 else v(1,0)/h if code<8 else v(0,1)/h)
        for a,b in pressure:
            row.append(monomial(a,b,x,y,1-component,component) if 2<=code<6 else 0.)
        return row
    return (np.asarray([evaluate(x,y,int(c)) for (x,y),c in zip(xy,codes)],float),
        np.asarray([evaluate(0.,0.,c) for c in (2,3,4,5,6,7,8,9)],float))


def prepare(p,degree=6):
    unique, records, selection = geometry(r.geometry.unit_box_grid(p.cells))
    ni, nb = len(p.points), len(p.boundary_points)
    prepared = []
    for owner, rec in enumerate(records):
        job = unique[rec['job']]
        xy = (job['points']-job['target'])*p.cells
        C, T = enriched_moments(xy, job['codes'], 1/p.cells, p.nu,degree)
        W = np.zeros((8,len(C)))
        groups=[]
        for group, size in enumerate((ni,nb,ni)):
            mask=rec['groups']==group
            cols=rec['columns'][mask]
            cols=2*(cols%size)+cols//size
            groups.append((mask,cols))
            for o,names in enumerate(NAMES):
                for comp in range(2):
                    W[2*o+comp,mask]=getattr(p,names[group])[2*owner+comp,cols].toarray().ravel()
        prepared.append(dict(C=C,T=T,W=W,R=T-W@C,
            D=np.where(np.asarray(job['codes'])>=2,p.nu*p.cells**2,1.),
            momentum=np.asarray(job['codes'])>=2,r2=np.sum(xy**2,axis=1),groups=groups))
    return prepared, selection


def construct(p, prepared, alpha, beta):
    ni,nb=len(p.points),len(p.boundary_points)
    entries={(o,g):([],[],[]) for o in range(4) for g in range(3)}
    checks=[]
    for owner,d in enumerate(prepared):
        C,T,W=d['C'],d['T'],d['W']
        # A larger D penalizes the corresponding weight correction more.
        D=d['D']*np.where(d['momentum'],alpha,1.)*(1+d['r2'])**(beta/2)
        nf=C.shape[1]
        inverse,_,rank,sv=lstsq((C/D[:,None]).T,np.eye(nf),cond=1e-13,lapack_driver='gelsd')
        E=inverse.T/D[None,:]
        V=W+d['R']@E
        error=float(np.max(abs(V@C-T)))
        if rank!=nf or error>1e-7:
            raise RuntimeError(f'owner={owner}, rank={rank}, moment_error={error}')
        checks.append(dict(owner=owner,rank=int(rank),moment_error_inf=error,
            left_inverse_error_inf=float(np.max(abs(E@C-np.eye(nf)))),
            singular_ratio=float(sv[-1]/sv[0]),
            normalized_change=float(np.linalg.norm((V-W)*D)/np.linalg.norm(W*D))))
        for group,(mask,cols) in enumerate(d['groups']):
            for o in range(4):
                rr,cc,vv=entries[o,group]
                for comp in range(2):
                    rr.extend([2*owner+comp]*len(cols));cc.extend(cols.tolist())
                    vv.extend(V[2*o+comp,mask].tolist())
    result=copy.copy(p)
    for (o,group),(rr,cc,vv) in entries.items():
        setattr(result,NAMES[o][group],coo_matrix((vv,(rr,cc)),shape=(2*ni,2*(nb if group==1 else ni))).tocsr())
    result.M=eye(2*ni,format='csr')-result.SL
    return result, checks


def spectrum(p,raw_path=None):
    a,m=p.A.toarray(),p.M.toarray()
    homogeneous,vectors=eig(-a,m,homogeneous_eigvals=True)
    values=np.full(len(homogeneous[0]),complex(np.inf),dtype=complex)
    np.divide(homogeneous[0],homogeneous[1],out=values,where=homogeneous[1]!=0)
    finite=np.isfinite(values)
    av_all=a@vectors;mv_all=m@vectors
    mass=np.linalg.norm(mv_all,axis=0)/np.linalg.norm(vectors,axis=0)
    residual=np.full(len(values),np.inf)
    residual[finite]=np.linalg.norm(-av_all[:,finite]-mv_all[:,finite]*values[finite],axis=0)/(np.linalg.norm(av_all[:,finite],axis=0)+abs(values[finite])*np.linalg.norm(mv_all[:,finite],axis=0))
    resolved=finite&(mass>1e-8)&(residual<1e-7)
    pair_scale=np.maximum(abs(homogeneous[0]),abs(homogeneous[1]))
    near_infinite=abs(homogeneous[1])<1e-12*pair_scale
    if raw_path is not None:
        np.savez(raw_path,alpha=homogeneous[0],beta=homogeneous[1],eigenvalues=values,
            mass_action_ratio=mass,action_relative_residual=residual,
            resolved=resolved,near_infinite=near_infinite)
    k=np.flatnonzero(resolved)[np.argmax(values[resolved].real)]
    lam,v=values[k],vectors[:,k]
    av,mv=a@v,m@v
    return dict(real=float(lam.real),imag=float(lam.imag),
        positive_count=int(np.sum(values[resolved].real>1e-7)),finite_count=int(finite.sum()),
        resolved_count=int(resolved.sum()),unresolved_finite_count=int(np.sum(finite&~resolved)),
        raw_positive_count=int(np.sum(values[finite].real>1e-7)),
        homogeneous_near_infinite_count=int(near_infinite.sum()),
        resolution_rule='mass_action>1e-8 and action_relative_residual<1e-7; retain raw eig separately; does not change any operator',
        mass_action_norm_ratio=float(np.linalg.norm(mv)/np.linalg.norm(v)),
        action_relative_residual=float(np.linalg.norm(-av-lam*mv)/(np.linalg.norm(av)+abs(lam)*np.linalg.norm(mv))))


def save_maps(p,folder):
    folder.mkdir(parents=True,exist_ok=False)
    for name in sum((list(x) for x in NAMES),[])+['M']:
        save_npz(folder/(name+'.npz'),getattr(p,name))


def run(cells,source,output,degree=6,alphas=None,betas=None):
    output.mkdir(parents=True,exist_ok=False)
    p=load_problem(cells,source)
    start=time.perf_counter()
    prepared,selection=prepare(p,degree)
    configuration=dict(cells=cells,source=str(source),selection=selection,nu=p.nu,
        reconstruction=f'F0+(Q-F0 C) E; full {prepared[0]["C"].shape[1]} fields, DF velocity P{degree} and pressure P2',
        metric='D(momentum)=alpha*nu/h^2; D(velocity)=1; all D *= (1+r_local^2)^(beta/2)',
        precision='source native C++ MPFR80 MPI8; correction and global algebra Float64',
        stencil_sites=37,alpha_values=alphas or [.01,.03,.1,.3,1.,3.,10.,30.,100.],beta_values=betas if betas is not None else [0.,1.,2.])
    (output/'configuration.json').write_text(json.dumps(configuration,indent=2))
    rows=[]
    # Stop at first stable metric; cross-mesh validation must freeze this choice.
    for beta in configuration['beta_values']:
        for alpha in configuration['alpha_values']:
            row=dict(alpha=alpha,beta=beta)
            try:
                q,checks=construct(p,prepared,alpha,beta)
                name=f'alpha{alpha:g}_beta{beta:g}'
                save_maps(q,output/name)
                row.update(status='screened',spectrum=spectrum(q,output/name/'spectrum.npz'),
                    max_polynomial_defect=max(c['moment_error_inf'] for c in checks),
                    max_left_inverse_defect=max(c['left_inverse_error_inf'] for c in checks),
                    min_singular_ratio=min(c['singular_ratio'] for c in checks),
                    min_rank=min(c['rank'] for c in checks))
                (output/name/'checks.json').write_text(json.dumps(checks,indent=2))
                # Every failed candidate is retained, not just its largest mode.
                row['pressure_controls']=all_pressure_controls(q)
            except Exception as exc:
                row.update(status='failed',error=str(exc))
            rows.append(row)
            (output/'results.json').write_text(json.dumps(dict(candidates=rows,seconds=time.perf_counter()-start),indent=2))
            print({k:v for k,v in row.items() if k!='pressure_controls'},flush=True)
            if row.get('spectrum',{}).get('positive_count')==0:
                return rows
    return rows


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells',type=int,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--velocity-degree',type=int,default=6)
    ap.add_argument('--alphas',type=float,nargs='+')
    ap.add_argument('--betas',type=float,nargs='+')
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells,args.source,args.output,args.velocity_degree,args.alphas,args.betas)
