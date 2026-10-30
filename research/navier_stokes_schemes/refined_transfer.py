"""Transfer common-E quadratic-pressure null corrections without fitting cavity data."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import LinearNDInterpolator
from threadpoolctl import threadpool_limits
import rbflab as r
from .cavity_nullspace_lifting import CavityLifting,coupled_spectrum
from .pressure_lifting import load_problem,all_pressure_controls
from .metric_lifting import spectrum,save_maps
from .cavity_candidates import steady,reference_at_nodal_lines
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry

def observations(p):
    unique,records,_=geometry(r.geometry.unit_box_grid(p.cells))
    return [(np.round((unique[rec['job']]['points']-unique[rec['job']]['target'])*p.cells,10),np.asarray(unique[rec['job']]['codes'])) for rec in records]

def transfer(output):
    output.mkdir(parents=True,exist_ok=True)
    src=CavityLifting(load_problem(12,Path('outputs/ns_cavity_candidates_20261029_native/nu2')))
    dst=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    source=np.load('outputs/ns_equilibrium_nullspace_lifting_20261029/parameters.npy')
    x=np.zeros(dst.size)
    oldobs,newobs=observations(src.p),observations(dst.p)
    # Choose nearest wall-layer class, preserving wall direction. No rotations.
    def features(p):
        ij=np.round(p.points*p.cells)
        return np.minimum(np.column_stack((ij,p.cells-ij)),4)
    oldf,newf=features(src.p),features(dst.p)
    mapping=[]
    for j,(xy,codes) in enumerate(newobs):
        feature=np.sum((oldf-newf[j])**2,axis=1)
        near=np.flatnonzero(feature==feature.min())
        best=None
        for k in near:
            ox,oc=oldobs[k]
            cost=np.sum((xy[:,None,:]-ox[None,:,:])**2,axis=2)+1e6*(codes[:,None]!=oc[None,:])
            rows,assigned=linear_sum_assignment(cost)
            cols=np.argmin(cost,axis=1);cols[rows]=assigned
            loss=cost[np.arange(len(xy)),cols].sum()
            if best is None or loss<best[0]:best=(loss,k,cols)
        loss,k,cols=best
        z=source[src.offsets[k]:src.offsets[k+1]].reshape(3,-1)
        # Express actual old nullspace action in target observation order;
        # target local() applies its own exact polynomial-null projector.
        z=z@src.data[k]['P']
        x[dst.offsets[j]:dst.offsets[j+1]]=z[:,cols].ravel()
        mapping.append(dict(owner=j,source_owner=int(k),assignment_squared_distance=float(loss),wall_layer_distance=float(feature[k])))
    np.save(output/'parameters_transfer.npy',x)
    (output/'transfer.json').write_text(json.dumps(mapping,indent=2))
    state=np.load('outputs/ns_equilibrium_nullspace_lifting_20261029/steady.npz')
    # Interpolate only initial guess; all accepted fields solve target equations.
    U=LinearNDInterpolator(src.p.points,state['U'].reshape(-1,2))(dst.p.points)
    L=LinearNDInterpolator(src.p.points,state['L'].reshape(-1,2))(dst.p.points)
    # nearest boundary-interior extrapolation fills edge points beyond source convex hull.
    from scipy.spatial import cKDTree
    nearest=cKDTree(src.p.points).query(dst.p.points)[1]
    for field,name in ((U,'U'),(L,'L')):
        bad=~np.isfinite(field).all(axis=1);field[bad]=state[name].reshape(-1,2)[nearest[bad]]
    np.savez(output/'initial.npz',U=U.ravel(),L=L.ravel())
    print(dict(stage='transfer',size=dst.size,maximum_assignment_loss=max(m['assignment_squared_distance'] for m in mapping)),flush=True)

def evaluate(output,scale=1.):
    model=CavityLifting(load_problem(16,Path('outputs/ns_cavity_candidates_20261029_cells16/nu2')))
    x=np.load(output/'parameters_transfer.npy')*scale
    p=model.matrices(x,full=True)
    initial=np.load(output/'initial.npz');p.initial=lambda:(initial['U'].copy(),initial['L'].copy())
    rr,U,L=steady(p)
    row=dict(scale=scale,root=rr,rest=spectrum(p))
    if rr['status']=='converged':
        row.update(cavity=coupled_spectrum(p,U,L),reference=reference_at_nodal_lines(p.points,U),pressure=all_pressure_controls(p))
    tag=str(scale).replace('.','p')
    np.savez(output/f'state_{tag}.npz',U=U,L=L,points=p.points)
    checks=model.checks(x)
    row['polynomial_error_inf']=max(c['polynomial_error_inf'] for c in checks)
    (output/f'evaluation_{tag}.json').write_text(json.dumps(row,indent=2))
    print(row,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('transfer','evaluate'),required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scale',type=float,default=1.)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        transfer(args.output) if args.stage=='transfer' else evaluate(args.output,args.scale)

