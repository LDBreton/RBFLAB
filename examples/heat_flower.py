"""Symbolic manufactured heat equation on a five-petal domain, with BDF2.

Run: python -m examples.heat_flower --backend cpp --animation flower.gif
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen
from examples._curved import local_method, solve, errors, report


def run(interior=250, seed=42, backend="python", dt=.025, steps=20, animation=None):
    """Solve u_t - 0.15 Laplacian(u)=f with exact nonzero boundary data."""
    domain=geometry.Flower()
    cloud=meshgen.generate(domain,interior=interior,boundary=max(30,round(120*np.sqrt(interior/250))),seed=seed)
    model=rbf.SymbolicScalar(2,transient=True)
    u,t=model.field,model.time; x,y=model.coordinates
    exact=sp.exp(-2*t)*(sp.cos(x)*sp.cos(y)+sp.Rational(1,4)*sp.sin(2*x))
    lhs=sp.diff(u,t)-sp.Rational(15,100)*model.laplacian(u)
    problem=model.evolution(sp.Eq(lhs,lhs.subs(u,exact).doit()), initial=exact.subs(t,0),
        boundary=[model.bc("boundary",sp.Eq(u,exact))])
    trajectory,timing=solve(problem,cloud,local_method(backend),dt=dt,steps=steps)
    target=sp.lambdify((x,y),exact.subs(t,dt*steps),"numpy")
    metrics=report("flower heat/"+backend,cloud,timing,
                   errors(trajectory.final,cloud,domain,lambda p:target(*p.T)))
    if animation:
        from rbflab import viz
        viz.animate_scalar(trajectory,animation,domain=domain,resolution=55,title="Heat diffusion / flower")
    return domain,cloud,trajectory,metrics


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interior",type=int,default=250);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--backend",choices=("python","cpp","torch"),default="python")
    p.add_argument("--dt",type=float,default=.025);p.add_argument("--steps",type=int,default=20)
    p.add_argument("--animation")
    run(**vars(p.parse_args()))
