"""Independent differentiation and algebra checks for the experiment."""
import json
import numpy as np
import sympy as sp
from scipy.linalg import lstsq
from run import gram, Patch, MU

x,y=sp.symbols('x y');e=2.;psi=sp.exp(-e*e*(x*x+y*y))
kernel=sp.Matrix([[-sp.diff(psi,y,2),sp.diff(psi,x,y),0],
                  [sp.diff(psi,x,y),-sp.diff(psi,x,2),0],[0,0,psi]])
def action(v,t,source=False):
    if t<3:return v[t]
    i=t-3
    return -MU*(sp.diff(v[i],x,2)+sp.diff(v[i],y,2))+(-1 if source else 1)*sp.diff(v[2],(x,y)[i])
expected=np.zeros((5,5))
for i in range(5):
    for j in range(5):
        v=sp.Matrix([action(kernel.row(k),j,True) for k in range(3)])
        expected[i,j]=float(action(v,i).subs({x:.17,y:-.23}))
actual=gram(np.tile([.17,-.23],(5,1)),np.arange(5),np.zeros((5,2)),np.arange(5),e)
assert np.allclose(actual,expected,rtol=2e-12,atol=2e-12)
grid=np.linspace(0,1,5);pts=np.array([(a,b) for b in grid for a in grid]);mask=np.all((pts>0)&(pts<1),axis=1)
p=Patch(0,pts,pts[mask],pts[~mask],17,2.);p.prepare(75.,.001)
rng=np.random.default_rng(1);d=rng.normal(size=len(p.x));q=(p.typ==2).astype(float)
assert np.max(abs(p.weights(p.x,p.typ)-np.eye(len(p.x))))<1e-7
A=gram(p.x,p.typ,p.x,p.typ,p.eps);alpha=p.C@d
assert np.isclose(np.linalg.norm(p.E@d)**2,alpha@A@alpha,rtol=1e-7)
assert np.max(abs(p.E@q))<1e-7
# Full least squares versus local elimination; boundary columns known.
F=np.column_stack((p.F[:,:p.ng],p.F[:,-p.nq:]));gauge=np.flatnonzero(p.glob[:p.ng]==2*len(pts[mask]))[0]
F=np.delete(F,gauge,axis=1);rhs=rng.normal(size=len(F))
full=lstsq(F,rhs)[0];reduced=np.delete(p.reduced,gauge,axis=1)
z=lstsq(reduced,p.N@rhs)[0]
assert np.allclose(full[:p.ng-1],z,rtol=1e-6,atol=1e-6)
print(json.dumps({'symbolic_gram_max_error':float(np.max(abs(actual-expected))),
                 'interpolation_max_error':p.interpolation_defect,
                 'condensation_solution_max_difference':float(np.max(abs(full[:p.ng-1]-z))),
                 'checks':'passed'},indent=2))
