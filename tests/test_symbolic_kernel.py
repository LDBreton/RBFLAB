import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r
from rbflab.symbolic_kernel import expressions

s=sp.Symbol('s',nonnegative=True)
c,g=sp.symbols('c g',positive=True)

def test_runtime_parameters_and_origin():
    family=r.Kernel((1+c*s)**sp.Rational(-1,2)+g*s**sp.Rational(7,2),s,(c,g))
    q=np.array([[0.,0.],[1e-8,2e-8],[.2,.3]])
    for value in ('0.5','2'):
        kernel=family(c=value,g='0.01');ref=r.Hybrid(r.IMQ(value),r.PHS(7),'0.01')
        for alpha in [(0,0),(1,1),(4,2)]:np.testing.assert_allclose(kernel.derivative(q,alpha),ref.derivative(q,alpha),rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError):family(c=-1,g=1)
    with pytest.raises(ValueError):family(c=1)
    with pytest.raises(ValueError,match='continuous'):expressions(r.Kernel(s**sp.Rational(3,2),s),(4,0))

@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='GCC/WSL opt-in')
@pytest.mark.parametrize('arithmetic',['float64','mpfr'])
def test_compiled_scalar_runtime_and_3d(tmp_path,arithmetic):
    family=r.Kernel(sp.exp(-c*s),s,(c,))
    artifact=family.compile(dimension=3,derivative_order=2,arithmetic=arithmetic,cache_dir=tmp_path)
    stamp=(artifact.directory/'kernel').stat().st_mtime_ns
    for value in ('0.7','2'):
        q=[[0,0,0],[.2,.3,.1]];alpha=(1,0,1)
        actual=artifact.evaluate(q,(value,),alpha)
        np.testing.assert_allclose(np.asarray(actual,dtype=float),r.Gaussian(value).derivative(q,alpha),atol=1e-14,rtol=1e-13)
    if arithmetic=='mpfr':
        import mpmath
        ctx=mpmath.mp.clone();ctx.dps=100
        actual=artifact.evaluate([['1e-10','0','0']],('0.7',),(0,0,0),digits=100)[0]
        assert abs(actual-ctx.exp(-ctx.mpf('0.7')*ctx.mpf('1e-20')))<ctx.mpf('1e-95')
        assert actual!=1
    again=family.compile(dimension=3,derivative_order=2,arithmetic=arithmetic,cache_dir=tmp_path)
    assert again.directory==artifact.directory
    assert (again.directory/'kernel').stat().st_mtime_ns==stamp

@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='GCC/WSL opt-in')
@pytest.mark.parametrize('digits',[None,45])
def test_space_compiler_matches_builtin(tmp_path,digits):
    from examples.tutorials.stokes_spaces import make_problem
    problem,spaces=make_problem(True);U,p=tuple(spaces);cloud=r.unit_box_grid(3)
    family=r.Kernel((1+c*s)**sp.Rational(-1,2),s,(c,))
    outputs=[];weights=[]
    for custom in (False,True):
        method=r.LHI(spaces={U:r.DivergenceFreeSpace(family(c='2') if custom else r.IMQ('2'),None),p:r.ScalarSpace(family(c='1') if custom else r.IMQ('1'),None)},stencil_size=12,precision=r.Precision(local_digits=digits),local_backend=r.CppBackend(threads=4,compute_condition=False,cache_dir=str(tmp_path)))
        if custom:method.prepare(problem,dimension=2,cache_dir=str(tmp_path))
        system=method.assemble(problem,cloud);solution=system.solve(dt='.01',steps=2).final
        q=np.vstack([cloud.interior[:2],[[.37,.43]]])
        outputs.append(np.hstack([solution.velocity(q),solution.pressure_gradient(q)]))
        weights.append(np.asarray(system.base.reconstruction_weights,dtype=float))
    np.testing.assert_allclose(*weights,rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(*outputs,rtol=1e-8,atol=1e-9)

@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='GCC/WSL opt-in')
def test_compiled_interpolation_fd_and_symmetric(tmp_path,monkeypatch):
    family=r.Kernel((1+c*s)**sp.Rational(-1,2),s,(c,))
    artifact=family.compile(derivative_order=4,cache_dir=tmp_path)
    import rbflab.kernel_compiler as compiler
    monkeypatch.setattr(compiler,'header',lambda *args:pytest.fail('Warm cache must skip symbolic generation'))
    assert family.compile(derivative_order=4,cache_dir=tmp_path).directory==artifact.directory
    kernel=artifact(c='2');points=np.random.default_rng(10).uniform(-1,1,(10,2));target=np.array([[.2,-.1]])
    data=lambda q:1+q[:,0]**2+q[:,1]
    solution=r.interpolate(kernel,points,data,polynomial_degree=2)
    np.testing.assert_allclose(solution.evaluate(target),data(target),atol=1e-10)
    w=r.rbf_fd_weights(kernel,points,target[0],r.Laplacian(),2)
    assert abs(w@data(points)-2)<1e-10
    left=[r.Laplacian()]*len(points)
    from rbflab.assembly import functional_matrix
    matrix=functional_matrix(kernel,points,left,points,left)
    np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
    np.testing.assert_allclose(matrix,functional_matrix(r.IMQ('2'),points,left,points,left),rtol=1e-12,atol=1e-12)

@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='GCC/WSL opt-in')
def test_nonbuiltin_space_kernel(tmp_path):
    from examples.tutorials.stokes_spaces import make_problem
    problem,spaces=make_problem(False);U,p=tuple(spaces);cloud=r.unit_box_grid(3)
    family=r.Kernel(sp.exp(-c*s)*(1+g*s),s,(c,g))
    outputs=[]
    for backend in (r.PythonBackend(compute_condition=False),r.CppBackend(threads=4,compute_condition=False,cache_dir=str(tmp_path))):
        method=r.LHI(spaces={U:r.DivergenceFreeSpace(family(c='2',g='.01'),None),p:r.ScalarSpace(family(c='1',g='.02'),None)},stencil_size=12,precision=r.Precision(local_digits=40),local_backend=backend)
        system=method.assemble(problem,cloud);sol=system.solve()
        q=np.array([[.37,.43]])
        outputs.append(np.hstack([sol.velocity(q),sol.pressure_gradient(q)]))
    np.testing.assert_allclose(*outputs,rtol=1e-10,atol=1e-10)


def test_preparation_preserves_polynomial_metadata():
    from rbflab.kernel_compiler import as_bound
    original=r.Hybrid(r.IMQ('2'),r.PHS(7),'0.01')
    bound=as_bound(original)
    assert bound.minimum_degree==original.minimum_degree
    assert r.DivergenceFreeSpace(bound).degree()==r.DivergenceFreeSpace(original).degree()
    assert r.ScalarSpace(bound).degree()==r.ScalarSpace(original).degree()


@pytest.mark.parametrize("power,order", [(3,2),(7,6)])
def test_pure_phs_zero_origin_jet(power,order):
    family=r.Kernel(s**sp.Rational(power,2),s)
    kernel=family()
    points=np.array([[0.,0.],[.2,.3]])
    for degree in range(order+1):
        for xorder in range(degree+1):
            alpha=(xorder,degree-xorder)
            expr,origin=expressions(family,alpha)
            assert origin==0
            np.testing.assert_allclose(kernel.derivative(points,alpha),r.PHS(power).derivative(points,alpha),rtol=1e-12,atol=1e-12)
