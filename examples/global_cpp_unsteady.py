"""Unsteady hybrid Stokes through the native public API.

Run: python -m examples.global_cpp_unsteady --digits 40
"""
import argparse
import json
import numpy as np
import sympy as sp
import rbflab as rbf


def run(digits=None,cells=4,steps=10,dt='0.01',power=9):
    model=rbf.SymbolicStokes(2,transient=True)
    U,p=model.fields;x,y=model.coordinates;t=model.time
    psi=sp.exp(-t)*sp.sin(2*x)*sp.cos(3*y)/3
    velocity=sp.ImmutableMatrix([sp.diff(psi,y),-sp.diff(psi,x)])
    pressure=sp.exp(-t)*(sp.sin(x+2*y)+sp.cos(2*x-y)/2)
    forcing=sp.diff(velocity,t)-model.laplacian(velocity)+model.gradient(pressure)
    problem=model.evolution([sp.Eq(sp.diff(U,t)-model.laplacian(U)+model.gradient(p),forcing)],
        initial={U:velocity.subs(t,0)},boundary=[model.bc('boundary',sp.Eq(U,velocity))])
    hybrid=lambda k:rbf.Hybrid(rbf.IMQ(1),rbf.PHS(k),1)
    method=rbf.GlobalCollocation(spaces={U:rbf.DivergenceFreeSpace(hybrid(power)),
        p:rbf.PressureSpace(hybrid(power-4))},precision=rbf.Precision(global_digits=digits),
        backend=rbf.CppGlobalBackend(threads=2,compute_condition=False))
    query=np.array([[.3,.4],[.7,.2],[.5,.6]])
    with method.assemble(problem,rbf.geometry.unit_box_grid(cells)) as system:
        trajectory=system.solve(dt=dt,steps=steps,scheme='bdf2')
        time=sp.Rational(dt)*steps
        u_true=np.asarray([model.data(v.subs(t,time))(query) for v in velocity]).T
        grad_true=np.asarray([model.data(v.subs(t,time))(query) for v in model.gradient(pressure)]).T
        result=dict(trajectory.diagnostics,
            unknowns=system.size,velocity_max_error=float(np.max(np.abs(trajectory.final.velocity(query)-u_true))),
            pressure_gradient_max_error=float(np.max(np.abs(trajectory.final.pressure_gradient(query)-grad_true))),
            divergence_max=float(np.max(np.abs(trajectory.final.divergence(query)))))
        print(json.dumps(result,indent=2))
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--digits',type=int)
    parser.add_argument('--cells',type=int,default=4)
    parser.add_argument('--power',type=int,choices=[7,9],default=9)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--dt',default='0.01')
    run(**vars(parser.parse_args()))
