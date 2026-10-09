"""Steady creeping Couette flow between concentric rotating cylinders.

This is Stokes flow (no convective term), with constant pressure. It is a
velocity/zero-pressure-gradient check, not a nontrivial pressure benchmark.
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import solve, report


def run(interior=70, seed=42, plot=None):
    """Solve in a divergence-free kernel space, with inner angular speed one."""
    domain=geometry.Annulus(.5,1.)
    cloud=meshgen.generate(domain,interior=interior,boundary={"inner":32,"outer":48},seed=seed)
    model=rbf.SymbolicStokes(2,vector_fields=("U",),scalar_fields=("p",))
    U,p=model.fields;x,y=model.coordinates
    # u_theta = A*r+B/r; u_theta(a)=a, u_theta(b)=0.
    factor=(1/(x*x+y*y)-1)/3
    velocity=sp.ImmutableMatrix([-y*factor,x*factor])
    momentum=-model.laplacian(U)+model.gradient(p)
    problem=model.stationary([sp.Eq(momentum,sp.zeros(2,1))],
        boundary=[model.bc("boundary",sp.Eq(U,velocity))])
    method=rbf.GlobalCollocation(spaces={U:rbf.DivergenceFreeSpace(rbf.IMQ(3),2),
                                         p:rbf.PressureSpace(rbf.IMQ(3),1)})
    solution,timing=solve(problem,cloud,method)
    query=meshgen.generate(domain,interior=100,boundary=36,seed=981).interior
    def exact(q):
        f=(1/np.sum(q*q,axis=1)-1)/3
        return np.column_stack((-q[:,1]*f,q[:,0]*f))
    error={"nodal_velocity_max":float(np.max(np.abs(solution.velocity(cloud.points)-exact(cloud.points)))),
           "off_node_velocity_max":float(np.max(np.abs(solution.velocity(query)-exact(query)))),
           "pressure_gradient_max":float(np.max(np.abs(solution.pressure_gradient(query)))),
           "divergence_max":float(np.max(np.abs(solution.divergence(query))))}
    metrics=report("Stokes annulus/global/python",cloud,timing,error)
    if plot:
        from rbflab import viz
        fig,_=viz.plot_velocity(solution,domain=domain,title="Stokes flow / rotating inner cylinder")
        fig.savefig(plot,dpi=160)
    return domain,cloud,solution,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interior",type=int,default=70);p.add_argument("--seed",type=int,default=42);p.add_argument("--plot")
    run(**vars(p.parse_args()))
