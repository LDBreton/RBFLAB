"""Dirichlet, Neumann and Robin conditions on labeled arcs of an ellipse."""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import local_method, solve, errors, report


def run(interior=240, seed=42, backend="python", plot=None):
    """Manufacture u=exp(x/2)*cos(y) and all boundary data symbolically."""
    domain=geometry.Ellipse(labels=("value","flux","robin"),angle=.25)
    cloud=meshgen.generate(domain,interior=interior,boundary=max(30,round(120*np.sqrt(interior/240))),seed=seed)
    model=rbf.SymbolicScalar(2);u=model.field;x,y=model.coordinates
    exact=sp.exp(x/2)*sp.cos(y);lhs=-model.laplacian(u)+u
    dn=model.normal_derivative(u);robin=u+dn
    problem=model.stationary(sp.Eq(lhs,lhs.subs(u,exact).doit()),boundary=[
        model.bc("value",sp.Eq(u,exact)),
        model.bc("flux",sp.Eq(dn,dn.subs(u,exact).doit())),
        model.bc("robin",sp.Eq(robin,robin.subs(u,exact).doit()))])
    solution,timing=solve(problem,cloud,local_method(backend))
    metrics=report("mixed ellipse/"+backend,cloud,timing,
                   errors(solution,cloud,domain,lambda p:np.exp(p[:,0]/2)*np.cos(p[:,1])))
    if plot:
        from rbflab import viz
        fig,_=viz.plot_scalar(solution,domain=domain,title="Mixed boundary conditions / ellipse")
        fig.savefig(plot,dpi=160)
    return domain,cloud,solution,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interior",type=int,default=240);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--backend",choices=("python","cpp","torch"),default="python");p.add_argument("--plot")
    run(**vars(p.parse_args()))
