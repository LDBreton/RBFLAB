"""Scalar heat through native global assembly, data and timestep algebra."""
import sympy as sp
import rbflab as r


def run(digits=60,steps=10):
    model=r.SymbolicScalar(2,transient=True);u=model.field;x,y=model.coordinates;t=model.time
    exact=sp.exp(-t)*(1+x*x+y*y)
    problem=model.evolution(sp.Eq(sp.diff(u,t)-model.laplacian(u),sp.diff(exact,t)-model.laplacian(exact)),
        initial=exact.subs(t,0),boundary=[model.bc('boundary',sp.Eq(u,exact))])
    method=r.GlobalCollocation(space=r.ScalarSpace(r.IMQ(),2),collocation_scheme="symmetric",precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(threads=4))
    with method.assemble(problem,r.geometry.unit_box_grid(3)) as system:
        trajectory=system.solve(dt='0.01',steps=steps,time_scheme='bdf2')
        print(r.execution_info(trajectory))
        print(trajectory.diagnostics)
        print(trajectory.final.evaluate_mp([[.3,.4]]))
        return trajectory.diagnostics


if __name__=='__main__':run()
