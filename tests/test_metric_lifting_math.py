"""Physical derivative and descriptor checks for the bounded research driver."""
from types import SimpleNamespace
import numpy as np
import sympy as sp
from scipy.sparse import diags
from research.navier_stokes_schemes.metric_lifting import enriched_moments, spectrum


def test_df7_enrichment_matches_physical_symbolic_fields():
    x,y=sp.symbols('x y')
    h=sp.Rational(1,12);nu=sp.Rational(1,100)
    xy=np.array([[.5,-.25],[-1.,2.],[.3,.7],[-.8,.6]])
    codes=np.arange(4)
    C,T=enriched_moments(xy,codes,float(h),float(nu),7)
    fields=[]
    for d in range(1,9):
        for a in range(d+1):
            psi=h*(x/h)**a*(y/h)**(d-a)
            fields.append((sp.Matrix([sp.diff(psi,y),-sp.diff(psi,x)]),0*x))
    for d in range(1,3):
        for a in range(d+1):
            fields.append((sp.zeros(2,1),h*(x/h)**a*(y/h)**(d-a)))
    assert C.shape==(4,49)
    for j,(u,p) in enumerate(fields):
        assert sp.simplify(sp.diff(u[0],x)+sp.diff(u[1],y))==0
        gradp=sp.Matrix([sp.diff(p,x),sp.diff(p,y)])
        ell=-nu*(u.diff(x,2)+u.diff(y,2))+gradp
        targets=list(ell)+list(gradp)+list(u.diff(x))+list(u.diff(y))
        np.testing.assert_allclose(T[:,j],[float(e.subs({x:0,y:0})) for e in targets],atol=1e-12)
        observations=list(u)+list(ell)
        for i,(site,code) in enumerate(zip(xy,codes)):
            expected=float(observations[code].subs({x:float(h)*site[0],y:float(h)*site[1]}))
            np.testing.assert_allclose(C[i,j],expected,rtol=1e-12,atol=1e-12)


def test_descriptor_diagnostic_retains_positive_dynamic_mode(tmp_path):
    p=SimpleNamespace(A=diags([1.,-2.,3.,4.]),M=diags([1.,1.,0.,0.]))
    path=tmp_path/'spectrum.npz'
    result=spectrum(p,path)
    assert result['positive_count']==1
    assert result['resolved_count']==2
    assert result['homogeneous_near_infinite_count']==2
    np.testing.assert_allclose(result['real'],2.)
    raw=np.load(path)
    assert len(raw['alpha'])==4
    np.testing.assert_allclose(np.sort(raw['eigenvalues'][raw['resolved']].real),[-1.,2.])
