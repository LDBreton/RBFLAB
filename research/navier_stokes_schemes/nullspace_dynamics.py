"""Bounded nodal validation of common-field optimized LHI; not convergence proof."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eig
from scipy.sparse import bmat,eye,csr_matrix
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem,pressure_problem,all_pressure_controls
from .cavity_candidates import diagnose,steady,reference_at_nodal_lines
from .coupled import convection,evolve


def linearized(p,U,L,folder):
    n=len(U)
    _,JU,JL=convection(p,U,L,p.boundary(0))
    K=bmat([[p.A,-p.M],[JU,eye(n)+JL]],format='csr').toarray()
    Z=csr_matrix((n,n))
    B=bmat([[Z,Z],[eye(n),Z]],format='csr').toarray()
    homogeneous,vectors=eig(-K,B,homogeneous_eigvals=True)
    values=np.full(2*n,complex(np.inf))
    np.divide(homogeneous[0],homogeneous[1],out=values,where=homogeneous[1]!=0)
    finite=np.isfinite(values)
    kv,bv=K@vectors,B@vectors
    mass=np.linalg.norm(bv,axis=0)/np.linalg.norm(vectors,axis=0)
    residual=np.full(2*n,np.inf)
    residual[finite]=np.linalg.norm(-kv[:,finite]-bv[:,finite]*values[finite],axis=0)/(np.linalg.norm(kv[:,finite],axis=0)+abs(values[finite])*np.linalg.norm(bv[:,finite],axis=0))
    resolved=finite&(mass>1e-10)&(residual<1e-7)
    k=np.flatnonzero(resolved)[np.argmax(values[resolved].real)]
    lam=values[k];mode=vectors[:,k]
    # Rotate the phase so the largest velocity entry is real and positive.
    phase=np.angle(mode[np.argmax(abs(mode[:n]))])
    mode=mode*np.exp(-1j*phase)
    mode*=1e-7/np.max(abs(mode[:n].real))
    np.savez(folder/'linearized_spectrum.npz',alpha=homogeneous[0],beta=homogeneous[1],
        eigenvalues=values,mass_action_ratio=mass,action_relative_residual=residual,
        resolved=resolved,leading_mode=mode,leading_eigenvalue=lam)
    return dict(real=float(lam.real),imag=float(lam.imag),resolved_count=int(resolved.sum()),
        positive_count=int(np.sum(values[resolved].real>1e-7)),
        mass_action_norm_ratio=float(mass[k]),action_relative_residual=float(residual[k]),
        linearization='K=[[A,-M],[JU,I+JL]], B=[[0,0],[I,0]], -Kz=lambda Bz'),mode,lam


def run(cells,source,output,end=.05):
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    p=load_problem(cells,source)
    report=dict(cells=cells,source=str(source),nu=.01,stencil_sites=37,
        scope='bounded nodal dynamics; no arbitrary-pressure, off-node or mesh-uniform stability claim',
        correction='common-field nullspace-optimized E; source native MPFR80/MPI8, correction/global Float64',
        time_scheme='BE startup, then BDF2; Oseen or Oseen predictor plus residual-controlled Newton',
        rest=diagnose(p),pressure_controls=all_pressure_controls(p),dynamics=[])
    def checkpoint():
        report['seconds']=time.perf_counter()-start
        (output/'results.json').write_text(json.dumps(report,indent=2))
    report['steady'],U,L=steady(p)
    report['reference']=reference_at_nodal_lines(p.points,U)
    report['reference']['state_status']=report['steady']['status']
    if report['steady']['status']!='converged':
        report['reference']['scope']+='; nonconverged Newton iterate, not a cavity root comparison'
    np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    checkpoint()
    mode=None;lam=None
    if report['steady']['status']=='converged':
        report['cavity_linearization'],mode,lam=linearized(p,U,L,output)
        checkpoint()
    for control in ('pressure_xy','cavity_startup','cavity_hold','cavity_perturbed_hold'):
        if 'hold' in control and mode is None:
            continue
        q=pressure_problem(p) if control=='pressure_xy' else copy.copy(p)
        if control=='cavity_hold':
            q.initial=lambda:(U.copy(),L.copy())
        if control=='cavity_perturbed_hold':
            q.initial=lambda:(U+mode[:len(U)].real,L+mode[len(U):].real)
        for scheme in ('oseen','oseen_newton'):
            for dt in (.00025,.000125):
                row=dict(control=control,scheme=scheme,dt=dt,T=end)
                try:
                    state=evolve(q,scheme,dt,round(end/dt))
                    row.update(status='completed',velocity_inf=float(np.max(abs(state['U']))),
                        kinetic_mean=float(.5*np.mean(np.sum(state['U'].reshape(-1,2)**2,axis=1))),
                        diagnostics=state['diagnostics'])
                    if control=='pressure_xy':
                        row['gradient_error_inf']=float(np.max(abs(state['gradient']-q.force(0))))
                    if 'hold' in control:
                        row['velocity_change_inf']=float(np.max(abs(state['U']-U)))
                    if control=='cavity_perturbed_hold':
                        row['velocity_amplification_inf']=row['velocity_change_inf']/1e-7
                        row['linear_phase_aware_amplification']=float(np.max(abs((np.exp(lam*end)*mode[:len(U)]).real))/1e-7)
                    np.savez(output/f'{control}_{scheme}_{dt}.npz',points=p.points,
                        U=state['U'],L=state['L'],gradient=state['gradient'])
                except Exception as exc:
                    row.update(status='failed',error=str(exc))
                report['dynamics'].append(row)
                checkpoint()
                print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
    report['status']='completed_bounded_validation'
    checkpoint()
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells',type=int,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--end',type=float,default=.05)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells,args.source,args.output,args.end)
