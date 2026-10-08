"""A nonpolynomial 3D manufactured Poisson solution inside a ball."""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import local_method, solve, errors, report


def run(interior=300, seed=42, backend="python", plot=None):
    """Solve -Laplacian(u)=-3u/4, u=exp((x+y+z)/2), with exact boundary values."""
    domain=geometry.Sphere()
    cloud=meshgen.generate(domain,interior=interior,boundary=max(40,round(180*(interior/300)**(2/3))),seed=seed)
    model=rbf.SymbolicScalar(3);u=model.field;x,y,z=model.coordinates
    exact=sp.exp((x+y+z)/2);lhs=-model.laplacian(u)
    problem=model.stationary(sp.Eq(lhs,lhs.subs(u,exact).doit()),
                              boundary=[model.bc("boundary",sp.Eq(u,exact))])
    solution,timing=solve(problem,cloud,local_method(backend,stencil=55))
    metrics=report("3D ball/"+backend,cloud,timing,
                   errors(solution,cloud,domain,lambda p:np.exp(p.sum(axis=1)/2)))
    if plot:
        from rbflab import viz
        fig,_=viz.plot_slice(solution,domain,title="Poisson in a ball / z = 0")
        fig.savefig(plot,dpi=160)
    return domain,cloud,solution,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interior",type=int,default=300);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--backend",choices=("python","cpp","torch"),default="python");p.add_argument("--plot")
    run(**vars(p.parse_args()))
