"""Common cardinal DF reconstruction lifting, isolated pressure-consistency study.

Fnew = F0 + (Q-F0 C) E, C=S Q, E C=I.  Therefore S Fnew=I,
Fnew C=Q and its velocity is divergence-free for arbitrary observations.
All output maps use the SAME E.  E minimizes the norm of the change in
weights after nondimensionalizing momentum by nu/h**2, unlike the historical
nu/radius**2 metric.  No energy-stability conclusion follows from this identity.
"""
import argparse
import copy
import json
from math import factorial
from pathlib import Path
import time

import numpy as np
import sympy as sp
from scipy.linalg import eig, lstsq
from scipy.sparse import coo_matrix, eye, load_npz, save_npz
from threadpoolctl import threadpool_limits

import rbflab as r
from .cavity import Cavity
from .cavity_candidates import diagnose, reference_at_nodal_lines, steady
from .coupled import evolve
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry

NAMES = [('A','B','SL'), ('PY','PB','PL'),
         ('DxU','DxB','DxL'), ('DyU','DyB','DyL')]


def monomial(a, b, x, y, dx=0, dy=0):
    if a < dx or b < dy:
        return np.zeros_like(np.asarray(x), dtype=float)
    return (factorial(a)/factorial(a-dx)*factorial(b)/factorial(b-dy)
            *np.asarray(x)**(a-dx)*np.asarray(y)**(b-dy))


def moments(xy, codes, h, nu):
    """Evaluate mixed observations and outputs on the same 40 exact fields."""
    powers = [(a,d-a) for d in range(1,8) for a in range(d+1)]
    pressure = [(a,d-a) for d in range(1,3) for a in range(d+1)]
    def evaluate(x,y,code):
        row=[]
        component=code % 2
        for a,b in powers:
            def v(dx=0,dy=0):
                return (b*monomial(a,b-1,x,y,dx,dy) if component==0 and b
                        else -a*monomial(a-1,b,x,y,dx,dy) if component==1 and a
                        else np.zeros_like(np.asarray(x),dtype=float))
            row.append(v() if code<2 else -nu/h**2*(v(2,0)+v(0,2)) if code<4
                       else 0. if code<6 else v(1,0)/h if code<8 else v(0,1)/h)
        for a,b in pressure:
            row.append(monomial(a,b,x,y,1-component,component) if 2<=code<6 else 0.)
        return row
    C=np.asarray([evaluate(x,y,int(c)) for (x,y),c in zip(xy,codes)],float)
    T=np.asarray([evaluate(0.,0.,c) for c in (2,3,4,5,6,7,8,9)],float)
    return C,T


def load_problem(cells, source):
    p=Cavity(cells=cells,nu=.01,build_maps=False,linear_solver='direct')
    # These experiments only load the viscosity-normalized baseline.
    # Q pressure polynomials are h*xi^a*eta^b, explaining the unit gradient
    # scale in moments(); the velocity streamfunctions use the same factor h.
    q=sp.Rational(cells,44);s=sp.Symbol('s',nonnegative=True)
    p.space=r.MixedSpace({'u':r.DivergenceFreeSpace(
        r.Hybrid(r.Gaussian(10*q*q),r.PHS(7),q**7),polynomial_degree=6),
        'p':r.PressureSpace(r.Kernel(sp.Rational(1,10000)*q**4*sp.exp(-q*q*s),s)(),polynomial_degree=1)})
    p.u,p.p=p.space.fields
    for name in sum((list(x) for x in NAMES),[]):
        setattr(p,name,load_npz(source/(name+'.npz')))
    p.M=eye(len(p.U0),format='csr')-p.SL
    return p


def all_pressure_controls(p):
    x,y=p.points.T
    controls={'x':np.column_stack((np.ones_like(x),0*x)),
              'y':np.column_stack((0*x,np.ones_like(x))),
              'x2':np.column_stack((2*x,0*y)),
              'xy':np.column_stack((y,x)),
              'y2':np.column_stack((0*x,2*y))}
    from scipy.sparse.linalg import spsolve
    result={}
    for name,g in controls.items():
        g=g.ravel();u=spsolve(p.A.tocsc(),p.M@g)
        result[name]=dict(constraint_inf=float(np.max(abs(p.M@g))),
            gradient_error_inf=float(np.max(abs(p.PL@g-g))),
            dx_velocity_inf=float(np.max(abs(p.DxL@g))),
            dy_velocity_inf=float(np.max(abs(p.DyL@g))),
            stationary_velocity_vector_rms=float(np.sqrt(np.mean(np.sum(u.reshape(-1,2)**2,axis=1)))))
    return result


def correct(p, destination):
    cloud=r.geometry.unit_box_grid(p.cells)
    unique,records,selection=geometry(cloud)
    ni,nb=len(p.points),len(p.boundary_points)
    entries={(o,g):([],[],[]) for o in range(4) for g in range(3)}
    checks=[]
    # Saved maps include every local source column, including explicit zeros.
    for owner,rec in enumerate(records):
        job=unique[rec['job']]
        xy=(job['points']-job['target'])*p.cells
        C,T=moments(xy,job['codes'],1/p.cells,p.nu)
        W=np.zeros((8,len(C)))
        nodecols=[]
        for group,size in enumerate((ni,nb,ni)):
            mask=rec['groups']==group
            cols=rec['columns'][mask]
            cols=2*(cols%size)+cols//size
            nodecols.append((mask,cols))
            for o,names in enumerate(NAMES):
                for comp in range(2):
                    W[2*o+comp,mask]=getattr(p,names[group])[2*owner+comp,cols].toarray().ravel()
        # weight metric ||delta W D||, D(momentum)=nu/h².
        D=np.where(np.asarray(job['codes'])>=2,p.nu*p.cells**2,1.)
        scaled=C/D[:,None]
        inverse,_,rank,sv=lstsq(scaled.T,np.eye(40),cond=1e-13,lapack_driver='gelsd')
        E=inverse.T/D[None,:]
        V=W+(T-W@C)@E
        error=float(np.max(abs(V@C-T)))
        if rank!=40 or error>1e-7:
            raise RuntimeError((owner,rank,error))
        checks.append(dict(owner=owner,rank=int(rank),moment_error_inf=error,
            left_inverse_error_inf=float(np.max(abs(E@C-np.eye(40)))),
            singular_ratio=float(sv[-1]/sv[0]),
            normalized_change=float(np.linalg.norm((V-W)*D)/np.linalg.norm(W*D))))
        for group,(mask,cols) in enumerate(nodecols):
            for o in range(4):
                rr,cc,vv=entries[o,group]
                for comp in range(2):
                    rr.extend([2*owner+comp]*len(cols));cc.extend(cols.tolist())
                    vv.extend(V[2*o+comp,mask].tolist())
    result=copy.copy(p)
    for (o,group),(rr,cc,vv) in entries.items():
        matrix=coo_matrix((vv,(rr,cc)),shape=(2*ni,2*(nb if group==1 else ni))).tocsr()
        name=NAMES[o][group]
        setattr(result,name,matrix);save_npz(destination/(name+'.npz'),matrix)
    result.M=eye(2*ni,format='csr')-result.SL
    save_npz(destination/'M.npz',result.M)
    (destination/'correction.json').write_text(json.dumps(dict(metric='D(momentum)=nu/h^2; D(velocity)=1',
         source_precision='saved public LocalApproximation MPFR80 MPI8; correction and global algebra Float64',
         reconstruction='F0+(Q-F0 C) E; same E for all outputs',selection=selection,checks=checks),indent=2))
    return result


def pressure_problem(p):
    p=copy.copy(p)
    force=np.column_stack((p.points[:,1],p.points[:,0])).ravel()
    p.boundary=lambda t: np.zeros(2*len(p.boundary_points))
    p.force=lambda t: force.copy()
    p.initial=lambda: (np.zeros_like(force),force.copy())
    return p


def rejection_audit(p, output, results, end=.05, controls=None):
    """Finite diagnostic controls on a rejected spatial candidate, not acceptance."""
    results['steady'],U,L=steady(p)
    results['reference']=reference_at_nodal_lines(p.points,U)
    np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    values,vectors=eig(-p.A.toarray(),p.M.toarray())
    finite=np.flatnonzero(np.isfinite(values))
    k=finite[np.argmax(values[finite].real)]
    lam=values[k]
    complex_mode=vectors[:,k]*(1e-7/np.max(abs(vectors[:,k].real)))
    perturb=complex_mode.real
    np.savez(output/'positive_mode.npz',eigenvalue=lam,vector=vectors[:,k])
    for control in (controls or ('pressure_xy','rest_mode','cavity_startup','cavity_hold')):
        if control=='cavity_hold' and results['steady']['status']!='converged':
            continue
        problem=pressure_problem(p) if control=='pressure_xy' else copy.copy(p)
        duration=end if control in ('pressure_xy','rest_mode','cavity_hold') else .001
        if control=='rest_mode':
            problem.boundary=lambda t:np.zeros(2*len(p.boundary_points))
            problem.initial=lambda:(perturb.copy(),(-lam*complex_mode).real.copy())
        elif control=='cavity_hold':
            problem.initial=lambda:(U.copy(),L.copy())
        for scheme in ('oseen','oseen_newton'):
            for dt in (.00025,.000125):
                row=dict(control=control,scheme=scheme,dt=dt,T=duration,
                         scope='bounded diagnostic of rejected spatial candidate')
                try:
                    state=evolve(problem,scheme,dt,round(duration/dt))
                    row.update(status='completed',velocity_inf=float(np.max(abs(state['U']))),
                               diagnostics=state['diagnostics'])
                    if control=='pressure_xy':
                        row['gradient_error_inf']=float(np.max(abs(state['gradient']-problem.force(0))))
                    if control=='rest_mode':
                        row['velocity_amplification_inf']=float(np.max(abs(state['U']))/np.max(abs(perturb)))
                        row['linear_exact_amplification']=float(np.max(abs((np.exp(lam*duration)*complex_mode).real))/np.max(abs(perturb)))
                    if control=='cavity_hold':
                        row['velocity_change_inf']=float(np.max(abs(state['U']-U)))
                    np.savez(output/f'{control}_{scheme}_{dt}.npz',points=p.points,U=state['U'],L=state['L'],gradient=state['gradient'])
                except Exception as exc:
                    row.update(status='failed',error=str(exc))
                results['dynamics'].append(row)
                print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
                (output/'results.json').write_text(json.dumps(results,indent=2))


def run(cells,source,output,end):
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    p=correct(load_problem(cells,source),output)
    results=dict(cells=cells,source=str(source),diagnostics=diagnose(p),
                 all_pressure_controls=all_pressure_controls(p),dynamics=[])
    results['diagnostics']['pressure_xy']['rms_convention']='scalar component RMS sqrt(mean(u_components**2)); vector RMS in all_pressure_controls'
    (output/'results.json').write_text(json.dumps(results,indent=2))
    print('SCREEN',cells,results['diagnostics'],flush=True)
    # A positive finite mode is a spatial rejection. Do not hide it with dt.
    if results['diagnostics']['rest_stokes_spectrum']['positive_count']:
        results['status']='rejected_positive_rest_mode'
        rejection_audit(p,output,results,end)
    else:
        results['steady'],U,L=steady(p)
        results['reference']=reference_at_nodal_lines(p.points,U)
        np.savez(output/'steady.npz',points=p.points,U=U,L=L)
        for control in ('pressure_xy','cavity_startup','cavity_hold'):
            if control=='cavity_hold' and results['steady']['status']!='converged':
                continue
            problem=pressure_problem(p) if control=='pressure_xy' else copy.copy(p)
            if control=='cavity_hold':
                problem.initial=lambda:(U.copy(),L.copy())
            for scheme in ('oseen','oseen_newton'):
                for dt in (.00025,.000125):
                    row=dict(control=control,scheme=scheme,dt=dt,T=end)
                    try:
                        state=evolve(problem,scheme,dt,round(end/dt))
                        row.update(status='completed',velocity_inf=float(np.max(abs(state['U']))),
                            diagnostics=state['diagnostics'])
                        if control=='pressure_xy':
                            row['gradient_error_inf']=float(np.max(abs(state['gradient']-problem.force(0))))
                        if control=='cavity_hold':
                            row['velocity_change_inf']=float(np.max(abs(state['U']-U)))
                        np.savez(output/f'{control}_{scheme}_{dt}.npz',points=p.points,U=state['U'],L=state['L'],gradient=state['gradient'])
                    except Exception as exc:
                        row.update(status='failed',error=str(exc))
                    results['dynamics'].append(row)
                    print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
                    (output/'results.json').write_text(json.dumps(results,indent=2))
        results['status']='completed_bounded_screen'
    results['seconds']=time.perf_counter()-start
    (output/'results.json').write_text(json.dumps(results,indent=2))
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells',type=int,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--end',type=float,default=.05)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells,args.source,args.output,args.end)
