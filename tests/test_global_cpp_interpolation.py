"""Opt-in native interpolation through the existing public function."""
import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r
from rbflab.symbolic import SymbolicData

pytestmark = pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')


@pytest.mark.parametrize('digits', [None, 40])
def test_interpolation_native_parity_and_derivatives(digits, tmp_path):
    centers = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, .3], [.8, .9]])
    samples = np.sin(centers[:, 0])+centers[:, 1]
    precision = r.Precision(global_digits=digits)
    reference = r.interpolate(r.IMQ(), centers, samples, precision)
    with r.interpolate(r.IMQ(), centers, samples, precision, backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        for operator in (None, r.Derivative(0), r.Laplacian()):
            assert np.allclose(fitted.evaluate(centers, operator), reference.evaluate(centers, operator), atol=1e-10)
            if digits:
                actual = fitted.evaluate_mp(centers, operator)
                assert actual.ctx.norm(actual-actual.ctx.matrix(reference.evaluate_mp(centers, operator).tolist()), 'inf') < actual.ctx.mpf('1e-32')
        assert np.allclose(fitted.evaluate(centers), samples, atol=1e-10)
        assert fitted.diagnostics['data_backend'] == 'python'
    with pytest.raises(RuntimeError):
        fitted.evaluate(centers)


def test_native_symbolic_interpolation_and_polynomials(tmp_path):
    x, y = sp.symbols('x y')
    centers = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .4]])
    data = SymbolicData(sp.pi*x+y, (x, y))
    with r.interpolate(r.PHS(3), centers, data, r.Precision(global_digits=40), 1,
                       backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        actual = fitted.evaluate_mp(centers)
        ctx = actual.ctx
        assert max(abs(actual[i]-ctx.pi*ctx.mpf(float(point[0]))-ctx.mpf(float(point[1]))) for i, point in enumerate(centers)) < ctx.mpf('1e-32')
        assert fitted.diagnostics['data_backend'] == 'cpp_symbolic'


def test_invalid_samples_and_duplicates_before_compilation(tmp_path):
    backend = r.CppGlobalBackend(cache_dir=tmp_path)
    with pytest.raises(ValueError, match='distinct'):
        r.interpolate(r.IMQ(), [[0, 0], [0, 0]], [1, 2], backend=backend)
    with pytest.raises(ValueError):
        r.interpolate(r.IMQ(), [[0, 0], [1, 0]], [[1, 2], [3, 4], [5, 6]], backend=backend)


def test_scalar_space_interpolation_3d(tmp_path):
    centers=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
    data=centers[:,0]+2*centers[:,1]-centers[:,2]
    space=r.ScalarSpace(r.PHS(3),1)
    reference=r.interpolate(space,centers,data)
    with r.interpolate(space,centers,data,backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        assert np.allclose(fitted.evaluate(centers),reference.evaluate(centers),atol=1e-11)
        assert np.allclose(fitted.evaluate(centers,r.Derivative(0,dimension=3)),1,atol=1e-11)


def test_scalar_space_mplapack_60_digits(tmp_path):
    prefix=os.environ.get('RBFLAB_MPLAPACK_DIR')
    if not prefix:pytest.skip('Requires optional MPLAPACK prefix')
    centers=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.3,.4]])
    samples=centers[:,0]+centers[:,1]
    space=r.ScalarSpace(r.PHS(3),1)
    with r.interpolate(space,centers,samples,r.Precision(global_digits=60),
        backend=r.CppGlobalBackend(cache_dir=tmp_path,linear_solver='mplapack',mplapack_dir=prefix)) as fitted:
        actual=fitted.evaluate_mp(centers)
        assert max(abs(actual[i]-actual.ctx.mpf(float(value))) for i,value in enumerate(samples))<actual.ctx.mpf('1e-50')
