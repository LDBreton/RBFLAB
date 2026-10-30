"""Corner affine-amplitude continuation, same fixed selected LHI weights."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem, all_pressure_controls
from .cavity_candidates import with_corner, steady, reference_at_nodal_lines
from .cavity_nullspace_lifting import coupled_spectrum
from research.divfree_lhi_energy.cavity_corner_field import evaluate


SOURCE=Path('outputs/ns_equilibrium_nullspace_lifting_20261029')


def transformed(base, amplitude, radius=.25):
    # Reproduce with_corner directly to expose the radius as a research option.
    p=copy.copy(base)
    j=evaluate(p.points,nu=p.nu,radius=radius)
    b=evaluate(p.boundary_points,nu=p.nu,radius=radius)
    s,sb,f=j['velocity'].ravel(),b['velocity'].ravel(),j['stokes_force'].ravel()
    corrections={'B':-p.A@s-p.B@sb+p.M@f,
                 'PB':j['gradpressure'].ravel()-p.PY@s-p.PB@sb-p.PL@f}
    for d in ('Dx','Dy'):
        corrections[d+'B']=j[d.lower()].ravel()-getattr(p,d+'U')@s-getattr(p,d+'B')@sb-getattr(p,d+'L')@f
    for name,col in corrections.items():
        setattr(p,name,hstack((getattr(base,name),csr_matrix(col[:,None])),format='csr'))
    p.boundary=lambda t:np.r_[base.boundary(t),amplitude]
    return p,j


def run(out,radius,maximum=1.,steps=20):
    out.mkdir(parents=True,exist_ok=True)
    p=load_problem(12,SOURCE/'maps')
    z=np.load(SOURCE/'steady.npz');U,L=z['U'],z['L']
    if (out/'latest.npz').exists():
        z=np.load(out/'latest.npz');U,L=z['U'],z['L']
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    start=rows[-1]['amplitude'] if rows else -1
    for a in np.linspace(0,maximum,steps+1):
        if a<=start+1e-12:continue
        q,j=transformed(p,float(a),radius)
        q.initial=lambda:(U.copy(),L.copy())
        report,u,l=steady(q)
        row=dict(amplitude=float(a),radius=radius,root=report)
        if report['status']=='converged':
            U,L=u,l
            row.update(spectrum=coupled_spectrum(q,U,L),reference=reference_at_nodal_lines(q.points,U),max_speed=float(np.linalg.norm(U.reshape(-1,2),axis=1).max()))
            np.savez(out/f'root_a{a:.9f}.npz',points=q.points,U=U,L=L)
            np.savez(out/'latest.npz',points=q.points,U=U,L=L)
        rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print(dict(amplitude=a,root=report['status'],spectrum=row.get('spectrum'),rms={k:row['reference'][k]['rms_difference'] for k in ('u','v')} if 'reference' in row else None),flush=True)
        if report['status']!='converged':break
    # Zero physical boundary and zero affine datum is the homogeneous control.
    q,j=transformed(p,1.,radius)
    q.boundary=lambda t:np.zeros(q.B.shape[1])
    checks=dict(homogeneous_pressure=all_pressure_controls(q),analytic_corner_divergence_inf=float(np.max(abs(j['dx'][:,0]+j['dy'][:,1]))))
    basecorner=with_corner(p);ours,_=transformed(p,1.,.25)
    checks['with_corner_parity']=max(float(abs(getattr(basecorner,k)-getattr(ours,k)).max()) for k in ('B','PB','DxB','DyB'))
    # Independent finite differences validate analytic jets at both cores and
    # transition annuli; these artificial validation sites are not Ghia data.
    # Preserve placement relative to each top corner when changing radius.
    test=np.array([[.03,.04],[.14,.09],[.03,.04],[.14,.09]])*(radius/.25)
    test[:,1]=1-test[:,1];test[2:,0]=1-test[2:,0]
    f=evaluate(test,radius=radius);h=2e-6
    for axis,d in enumerate(('dx','dy')):
        shift=np.zeros_like(test);shift[:,axis]=h
        plus=evaluate(test+shift,radius=radius);minus=evaluate(test-shift,radius=radius)
        checks[d+'_fd_error_inf']=float(np.max(abs((plus['velocity']-minus['velocity'])/(2*h)-f[d])))
        checks['gradp_'+str(axis)+'_fd_error_inf']=float(np.max(abs((plus['pressure']-minus['pressure'])/(2*h)-f['gradpressure'][:,axis])))
    h=2e-5;lap=np.zeros_like(f['velocity'])
    for axis in (0,1):
        shift=np.zeros_like(test);shift[:,axis]=h
        lap+=(evaluate(test+shift,radius=radius)['velocity']-2*f['velocity']+evaluate(test-shift,radius=radius)['velocity'])/h**2
    checks['lap_fd_relative_inf']=float(np.max(abs(lap-f['lap']))/np.max(abs(f['lap'])))
    checks['force_identity_inf']=float(np.max(abs(f['stokes_force']+.01*f['lap']-f['gradpressure'])))
    # Homogeneous pressure increments about the nonzero affine cavity background.
    xy=np.column_stack((p.points[:,1],p.points[:,0])).ravel()
    checks['pressure_increment_constraint_inf']=float(np.max(abs(p.M@xy)))
    checks['pressure_increment_gradient_error_inf']=float(np.max(abs(p.PL@xy-xy)))
    checks['pressure_increment_derivative_inf']=float(max(np.max(abs(p.DxL@xy)),np.max(abs(p.DyL@xy))))
    (out/'checks.json').write_text(json.dumps(checks,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--radius',type=float,default=.25);ap.add_argument('--maximum',type=float,default=1.);ap.add_argument('--steps',type=int,default=20);args=ap.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.radius,args.maximum,args.steps)
