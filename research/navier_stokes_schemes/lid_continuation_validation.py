"""Bounded full-lid Newton recovery, preserving the fixed optimized LHI maps."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .pressure_lifting import load_problem
from .cavity_candidates import steady,reference_at_nodal_lines
from .nullspace_dynamics import linearized
from .coupled import evolve


def run(cells,source,start_path,output,end=.05):
    output.mkdir(parents=True,exist_ok=False)
    p=load_problem(cells,source)
    saved=np.load(start_path)
    q=copy.copy(p);q.initial=lambda:(saved['U'].copy(),saved['L'].copy())
    report=dict(cells=cells,source=str(source),startup_initialization=str(start_path),
        scope='bounded nodal full-lid root recovery; fixed maps and 37-site common DF field',stages=[],dynamics=[])
    def save():
        (output/'results.json').write_text(json.dumps(report,indent=2))
    result,U,L=steady(q)
    report['startup_initialized_newton']=result
    np.savez(output/'startup_initialized_newton.npz',points=p.points,U=U,L=L)
    save()
    if result['status']!='converged':
        amplitude=0.;step=.1
        U=np.zeros_like(p.U0);L=U.copy()
        for stage in range(20):
            trial=min(1.,amplitude+step)
            q=copy.copy(p)
            previous_U,previous_L=U.copy(),L.copy()
            q.boundary=lambda t,a=trial:a*p.boundary(t)
            q.initial=lambda:(previous_U.copy(),previous_L.copy())
            result,next_U,next_L=steady(q)
            report['stages'].append(dict(stage=stage,amplitude=trial,**result))
            np.savez(output/f'stage{stage:02d}_lid{trial:.6f}.npz',points=p.points,U=next_U,L=next_L)
            save()
            print(report['stages'][-1],flush=True)
            if result['status']=='converged':
                amplitude=trial;U,L=next_U,next_L
                if trial>=1.-1e-12:
                    break
            else:
                step*=.5
                if step<1e-3:
                    break
        report['achieved_lid_amplitude']=amplitude
        if amplitude<1.-1e-12:
            report['status']='full_lid_root_not_recovered'
            save();return report
    report['full_lid_steady']=result
    report['reference']=reference_at_nodal_lines(p.points,U)
    np.savez(output/'steady.npz',points=p.points,U=U,L=L)
    report['cavity_linearization'],mode,lam=linearized(p,U,L,output)
    save()
    for control in ('cavity_hold','cavity_perturbed_hold'):
        q=copy.copy(p)
        if control=='cavity_hold':
            q.initial=lambda:(U.copy(),L.copy())
        else:
            q.initial=lambda:(U+mode[:len(U)].real,L+mode[len(U):].real)
        for scheme in ('oseen','oseen_newton'):
            for dt in (.00025,.000125):
                row=dict(control=control,scheme=scheme,dt=dt,T=end)
                try:
                    state=evolve(q,scheme,dt,round(end/dt))
                    change=float(np.max(abs(state['U']-U)))
                    row.update(status='completed',velocity_change_inf=change,diagnostics=state['diagnostics'])
                    if control=='cavity_perturbed_hold':
                        row.update(velocity_amplification_inf=change/1e-7,
                            linear_phase_aware_amplification=float(np.max(abs((np.exp(lam*end)*mode[:len(U)]).real))/1e-7))
                    np.savez(output/f'{control}_{scheme}_{dt}.npz',points=p.points,U=state['U'],L=state['L'],gradient=state['gradient'])
                except Exception as exc:
                    row.update(status='failed',error=str(exc))
                report['dynamics'].append(row);save()
                print({k:v for k,v in row.items() if k!='diagnostics'},flush=True)
    report['status']='completed_bounded_full_lid_validation';save()
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells',type=int,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--startup',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.cells,args.source,args.startup,args.output)
