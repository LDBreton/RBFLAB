"""Public LocalApproximation cavity trial; mixed Oseen, 37 sites, no FE space."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from .problem import Problem
from .coupled import evolve

class Cavity(Problem):
    def _manufactured_fields(self):
        self.U0 = np.zeros(2*len(self.points))
        self.L0 = self.U0.copy()
        self.g0 = np.zeros((len(self.boundary_points), 2))
        x,y = self.boundary_points.T
        self.g0[(y == 1)&(x > 0)&(x < 1),0] = 1
        self.g0 = self.g0.ravel()
    def force(self,t):
        return np.zeros_like(self.U0)
    def boundary(self,t):
        return self.g0.copy()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=int,default=12)
    parser.add_argument('--end',type=float,default=.01)
    parser.add_argument('--ranks',type=int,default=8)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    config=dict(vars(args),nu=.01,Re=100,stencil=37,augmentation='6/1',q='cells/44',pressure_amplitude='q^4',corner_subtraction=False,initial='rest; impulsive lid; corner nodes stationary',scheme='BE/BDF2 Oseen',error_scope='nodal diagnostics only; no exact cavity solution')
    (root/'configuration.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter()
    p=Cavity(cells=args.cells,nu=.01,ranks=args.ranks,reuse_stencils=True,weight_folder=root/'weights',linear_solver='gmres_scaled',global_ordering='MMD_AT_PLUS_A')
    print('weights seconds',time.perf_counter()-start,flush=True)
    results=[]
    for dt in (.00025,.000125):
        row=dict(dt=dt,end=args.end)
        try:
            with threadpool_limits(limits=1):
                state=evolve(p,'oseen',dt,round(args.end/dt))
            speed=np.linalg.norm(state['U'].reshape(-1,2),axis=1)
            row.update(status='completed',max_nodal_speed=float(speed.max()),kinetic_mean=float(.5*np.mean(speed**2)),linear_solver=state['linear_solver'],diagnostics=state['diagnostics'])
            np.savez(root/f'fields_dt{dt}.npz',points=p.points,U=state['U'],L=state['L'],gradient=state['gradient'])
        except Exception as error:
            row.update(status='failed',error=str(error),traceback=traceback.format_exc())
        results.append(row)
        (root/'results.json').write_text(json.dumps(results,indent=2))
        print({k:v for k,v in row.items() if k not in ('diagnostics','linear_solver','traceback')},flush=True)
