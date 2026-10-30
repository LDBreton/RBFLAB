"""Independent algebra checks for the experimental common field lifting."""
import numpy as np
import sympy as sp

from research.navier_stokes_schemes.pressure_lifting import moments


def test_mixed_polynomial_moments_match_symbolic_physical_derivatives():
    x,y=sp.symbols('x y')
    h=sp.Rational(1,12);nu=sp.Rational(1,100)
    xy=np.array([[.5,-.25],[-1.,2.],[0.,0.]])
    codes=np.array([0,3,2])
    C,T=moments(xy,codes,float(h),float(nu))
    fields=[]
    for d in range(1,8):
        for a in range(d+1):
            psi=h*(x/h)**a*(y/h)**(d-a)
            fields.append((sp.Matrix([sp.diff(psi,y),-sp.diff(psi,x)]),0*x))
    for d in range(1,3):
        for a in range(d+1):
            fields.append((sp.zeros(2,1),h*(x/h)**a*(y/h)**(d-a)))
    for j,(u,p) in enumerate(fields):
        gradp=sp.Matrix([sp.diff(p,x),sp.diff(p,y)])
        ell=-nu*(u.diff(x,2)+u.diff(y,2))+gradp
        expected=list(ell)+list(gradp)+list(u.diff(x))+list(u.diff(y))
        target=np.array([float(e.subs({x:0,y:0})) for e in expected])
        np.testing.assert_allclose(T[:,j],target,rtol=1e-13,atol=1e-12)
        observed=list(u)+list(ell)
        for i,(site,code) in enumerate(zip(xy,codes)):
            value=float(observed[code].subs({x:float(h)*site[0],y:float(h)*site[1]}))
            np.testing.assert_allclose(C[i,j],value,rtol=1e-13,atol=1e-12)
        assert sp.simplify(sp.diff(u[0],x)+sp.diff(u[1],y))==0


def test_common_lifting_preserves_cardinality_and_reproduces_extended_trial():
    # Source evaluation of F0 is identity. Field values at other functionals
    # are arbitrary here; this verifies common reconstruction algebra.
    rng=np.random.default_rng(29)
    C=rng.normal(size=(54,40))
    E=np.linalg.pinv(C)
    W=rng.normal(size=(8,54));T=rng.normal(size=(8,40))
    lifted=W+(T-W@C)@E
    np.testing.assert_allclose(lifted@C,T,atol=1e-12)
    source=np.eye(54)+(C-np.eye(54)@C)@E
    np.testing.assert_allclose(source,np.eye(54),atol=1e-13)
