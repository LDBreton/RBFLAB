"""Scalar, independent multi-output and solenoidal native interpolants."""
import numpy as np
import sympy as sp
import rbflab as r


def run():
    points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.3,.4]])
    backend=r.CppGlobalBackend(threads=4)
    precision=r.Precision(global_digits=60)
    model=r.SymbolicScalar();x,y=model.coordinates
    with r.interpolate(space=r.ScalarSpace(r.PHS(3),1),centers=points,data=model.data(sp.pi*x+y),
        precision=precision,backend=backend) as scalar:
        print('Scalar:',scalar.evaluate_mp([[.2,.3]]))
    # Independent components: no divergence constraint is imposed.
    values=np.column_stack([points[:,0],points[:,1]])
    with r.interpolate(space=r.ScalarSpace(r.IMQ(),1),centers=points,data=values,precision=precision,backend=backend) as vector:
        print('Independent vector:',vector.evaluate_mp([[.2,.3]]))
    # A coupled solenoidal kernel and exact solenoidal polynomial space.
    data=(model.data(y),model.data(-x))
    with r.interpolate(space=r.DivergenceFreeSpace(r.IMQ(),1),centers=points,data=data,precision=precision,backend=backend) as solenoidal:
        print('Divergence-free vector:',solenoidal.evaluate_mp([[.2,.3]]))


if __name__=='__main__':run()
