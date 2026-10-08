"""Laplace on an annulus: compare global, LHI and RBF-FD against log(r/a)/log(b/a).

Run: python -m examples.annulus --backend cpp
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import local_method, solve, errors, report


def run(interior=240, seed=42, method="rbffd", backend="python", plot=None):
    """Solve the same labeled problem with selectable spatial discretization."""
    domain=geometry.Annulus(.4, 1.)
    cloud=meshgen.generate(domain, interior=interior, boundary={"inner":max(12,round(48*np.sqrt(interior/240))),"outer":max(24,round(96*np.sqrt(interior/240)))}, seed=seed)
    model=rbf.SymbolicScalar(2); u=model.field
    problem=model.stationary(sp.Eq(-model.laplacian(u),0), boundary=[
        model.bc("inner",sp.Eq(u,0)), model.bc("outer",sp.Eq(u,1))])
    if method != "rbffd" and backend != "python":
        raise ValueError("This example exposes C++/Torch through RBF-FD; use --method rbffd")
    methods={"rbffd":lambda:local_method(backend),
             "global":lambda:rbf.GlobalCollocation(rbf.PHS(5),scheme="asymmetric",polynomial_degree=3),
             "lhi":lambda:rbf.LHI(rbf.PHS(5),35,polynomial_degree=3)}
    solution,timing=solve(problem,cloud,methods[method]())
    exact=lambda p:np.log(np.linalg.norm(p,axis=1)/.4)/np.log(2.5)
    metrics=report("annulus/"+method+"/"+backend,cloud,timing,errors(solution,cloud,domain,exact))
    if plot:
        from rbflab import viz
        fig,_=viz.plot_scalar(solution,domain=domain,title="Harmonic field / annulus")
        fig.savefig(plot,dpi=160)
    return domain,cloud,solution,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interior",type=int,default=240);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--method",choices=("rbffd","global","lhi"),default="rbffd")
    p.add_argument("--backend",choices=("python","cpp","torch"),default="python")
    p.add_argument("--plot")
    run(**vars(p.parse_args()))
