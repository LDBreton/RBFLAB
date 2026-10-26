"""Vector interpolation keeps ordinary and solenoidal spaces distinct."""
import itertools
import os
import numpy as np
import pytest
import rbflab as r


def cloud(d):
    return np.asarray(list(itertools.product((0., 1.), repeat=d))+[(.3,)*d])


@pytest.mark.parametrize('dimension', [2, 3])
def test_python_divergence_free_reproduction(dimension):
    points=cloud(dimension)
    samples=np.zeros_like(points);samples[:,0]=points[:,1];samples[:,1]=-points[:,0]
    fitted=r.interpolate(r.DivergenceFreeSpace(r.IMQ(),1),points,samples)
    assert np.allclose(fitted.evaluate(points),samples,atol=1e-11)
    divergence=sum(fitted.evaluate(points,r.Derivative(c,dimension))[:,c] for c in range(dimension))
    assert np.max(abs(divergence))<1e-11
    assert fitted.evaluate(np.empty((0,dimension))).shape==(0,dimension)


def test_python_componentwise_unconstrained_and_validation():
    points=cloud(2);samples=np.column_stack((points[:,0],points[:,1],points[:,0]+points[:,1]))
    fitted=r.interpolate(r.ScalarSpace(r.IMQ(),1),points,samples)
    assert fitted.evaluate(points).shape==(len(points),3)
    assert np.allclose(fitted.evaluate(points),samples,atol=1e-11)
    # Independent components retain this nonzero divergence.
    divergence=fitted.evaluate(points,r.Derivative(0))[:,0]+fitted.evaluate(points,r.Derivative(1))[:,1]
    assert np.allclose(divergence,2,atol=1e-11)
    with pytest.raises(ValueError,match='shape'):
        r.interpolate(r.DivergenceFreeSpace(r.IMQ()),points,np.ones((len(points),3)))


def test_tuple_rows_are_node_major_when_node_count_equals_dimension():
    points=np.array([[0.,0.],[1.,1.]])
    rows=((1.,2.),(3.,4.))
    fitted=r.interpolate(r.DivergenceFreeSpace(r.IMQ(),None),points,rows)
    assert np.allclose(fitted.evaluate(points),rows,atol=1e-11)


def test_componentwise_public_coefficients_control_evaluation():
    points=cloud(2);samples=np.column_stack((points[:,0],points[:,1]))
    fitted=r.interpolate(r.ScalarSpace(r.IMQ(),1),points,samples)
    before=fitted.evaluate(points)
    fitted.coefficients[:,0]*=2
    after=fitted.evaluate(points)
    assert np.allclose(after[:,0],2*before[:,0],atol=1e-11)
    assert np.allclose(after[:,1],before[:,1],atol=1e-11)


def test_solenoidal_degree_two_has_adequate_vector_dofs():
    points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.3,.4]])
    samples=np.column_stack((points[:,1]**2,-points[:,0]**2))
    fitted=r.interpolate(r.DivergenceFreeSpace(r.IMQ(),2),points,samples)
    assert np.allclose(fitted.evaluate(points),samples,atol=1e-11)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')
@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('digits',[None,60])
def test_native_divergence_free_parity(dimension,digits,tmp_path):
    points=cloud(dimension);samples=np.zeros_like(points);samples[:,0]=points[:,1];samples[:,1]=-points[:,0]
    precision=r.Precision(global_digits=digits);space=r.DivergenceFreeSpace(r.IMQ(),1)
    reference=r.interpolate(space,points,samples,precision)
    with r.interpolate(space,points,samples,precision,backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        for operator in (None,r.Derivative(0,dimension),r.Laplacian(dimension)):
            assert np.allclose(fitted.evaluate(points,operator),reference.evaluate(points,operator),atol=1e-10)
            if digits:
                actual=fitted.evaluate_mp(points,operator);expected=reference.evaluate_mp(points,operator)
                assert actual.ctx.mnorm(actual-actual.ctx.matrix(expected.tolist()),'inf')<actual.ctx.mpf('1e-48')
        assert fitted.evaluate(np.empty((0,dimension))).shape==(0,dimension)
        with pytest.raises(ValueError,match='dimensions'):
            fitted.evaluate(points,r.Derivative(0,3 if dimension==2 else 2))
        divergence=sum(fitted.evaluate(points,r.Derivative(c,dimension))[:,c] for c in range(dimension))
        assert np.max(abs(divergence))<1e-10
    with pytest.raises(RuntimeError):fitted.evaluate(points)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')
def test_native_componentwise_single_system(tmp_path):
    points=cloud(2);samples=np.column_stack((points[:,0],points[:,1],points[:,0]+points[:,1]))
    with r.interpolate(r.ScalarSpace(r.IMQ(),1),points,samples,r.Precision(global_digits=60),backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        assert all(solution.system is fitted.system for solution in fitted.solutions)
        assert np.allclose(fitted.evaluate(points),samples,atol=1e-11)
        assert fitted.evaluate_mp(np.empty((0,2))).cols==3


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')
def test_native_vector_degree_two_and_tuple_rows(tmp_path):
    points=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.3,.4]])
    samples=np.column_stack((points[:,1]**2,-points[:,0]**2))
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(),2),points,samples,backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        assert np.allclose(fitted.evaluate(points),samples,atol=1e-11)
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(),None),points[:2],((1.,2.),(3.,4.)),backend=r.CppGlobalBackend(cache_dir=tmp_path)) as fitted:
        assert np.allclose(fitted.evaluate(points[:2]),((1.,2.),(3.,4.)),atol=1e-11)
