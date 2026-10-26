"""Planar potential blending must keep the vector field solenoidal."""
import numpy as np
import pytest
import os

import rbflab as r


def cloud():
    x=np.array(np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5),indexing='ij')).reshape(2,-1).T
    cover=r.PatchCover([[-.42,0],[.42,0]],[1.7,1.7])
    return x,cover


def test_polynomial_streamfunction_and_refit():
    x,cover=cloud()
    # psi=x^2+y^2+3x; velocity=(-2y,2x+3).
    v=np.column_stack((-2*x[:,1],2*x[:,0]+3))
    fit=r.interpolate(r.DivergenceFreeSpace(r.IMQ(),1),x,v,patches=cover)
    q=np.array([[.13,.19],[-.38,.21],[.4,-.2]])
    np.testing.assert_allclose(fit.evaluate(q),np.column_stack((-2*q[:,1],2*q[:,0]+3)),atol=2e-8)
    np.testing.assert_allclose(fit.divergence(q),0,atol=2e-8)
    assert fit.diagnostics['glue_edges']==1
    ids=[id(p['factor']) for p in fit.patches]
    fit.refit(2*v)
    assert [id(p['factor']) for p in fit.patches]==ids
    np.testing.assert_allclose(fit.evaluate(q),2*np.column_stack((-2*q[:,1],2*q[:,0]+3)),atol=4e-8)
    before=fit.evaluate(q)
    with pytest.raises(ValueError):fit.refit(np.ones((len(x),3)))
    np.testing.assert_allclose(fit.evaluate(q),before)
    with pytest.raises(NotImplementedError):fit.evaluate(q,r.Laplacian())


def test_overlap_curl_matches_potential_finite_difference():
    x,cover=cloud()
    v=np.column_stack((-np.cos(x[:,1])-x[:,0],np.cos(x[:,0])+x[:,1]))
    fit=r.interpolate(r.DivergenceFreeSpace(r.IMQ()),x,v,patches=cover)
    q=np.array([[.11,.09],[-.21,.13]])
    h=1e-5
    dx=np.array([h,0]);dy=np.array([0,h])
    expected=np.column_stack((-(fit.potential(q+dy)-fit.potential(q-dy))/(2*h),
                               (fit.potential(q+dx)-fit.potential(q-dx))/(2*h)))
    np.testing.assert_allclose(fit.evaluate(q),expected,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(fit.divergence(q),0,atol=1e-8)


def test_mp_preserves_streamfunction_digits():
    x,cover=cloud()
    v=np.column_stack((-2*x[:,1],2*x[:,0]+3))
    fit=r.interpolate(r.DivergenceFreeSpace(r.IMQ(),1),x,v,
                      precision=r.Precision(global_digits=60),patches=cover)
    q=np.array([[.13,.19]])
    ctx=fit.arithmetic.ctx
    expected=[-2*ctx.mpf(float(q[0,1])),2*ctx.mpf(float(q[0,0]))+3]
    actual=fit.evaluate_mp(q)
    assert max(abs(actual[0,j]-expected[j]) for j in range(2))<ctx.mpf('1e-45')
    assert abs(fit.divergence(q,extended=True)[0])<ctx.mpf('1e-45')
    report=r.execution_info(fit)
    assert report['locations']['glue']=='python'
    assert report['glue_arithmetic']=='mpmath'


def test_polynomial_degree_must_be_configured_on_the_space():
    x,cover=cloud()
    with pytest.raises(ValueError,match='DivergenceFreeSpace only'):
        r.interpolate(r.DivergenceFreeSpace(r.IMQ(),1),x,np.zeros_like(x),
                      patches=cover,polynomial_degree=2)


@pytest.mark.parametrize('digits,solver',[(None,'eigen'),(60,'eigen'),(60,'mplapack')])
def test_native_dfpu_parity_refit_and_lifecycle(digits,solver):
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('native worker opt-in')
    prefix=os.environ.get('RBFLAB_MPLAPACK_DIR')
    if solver=='mplapack' and not prefix:pytest.skip('optional MPLAPACK prefix unavailable')
    x,cover=cloud()
    # Nonlinear streamfunction makes local fits differ across the overlap.
    data=np.column_stack((-np.cos(x[:,1])-x[:,0],np.cos(x[:,0])+x[:,1]))
    space=r.DivergenceFreeSpace(r.IMQ(),1)
    precision=r.Precision(global_digits=digits) if digits else r.Precision()
    backend=r.CppGlobalBackend(threads=1,compute_condition=False,linear_solver=solver,
                               mplapack_dir=prefix if solver=='mplapack' else None)
    q=np.array([[.11,.09],[-.21,.13],[.34,-.16]])
    with r.interpolate(space,x,data,precision,patches=cover) as reference:
        with r.interpolate(space,x,data,precision,patches=cover,backend=backend) as native:
            tolerance=1e-44 if digits else 2e-8
            for multiplier in (1,2):
                if multiplier==2:
                    reference.refit(2*data)
                    native.refit(2*data)
                for op in (None,r.Derivative(0),r.Derivative(1)):
                    if digits:
                        a=native.evaluate_mp(q,op);b=reference.evaluate_mp(q,op)
                        assert max(abs(a[i,j]-b[i,j]) for i in range(a.rows) for j in range(2))<reference.arithmetic.ctx.mpf('1e-44')
                    else:
                        np.testing.assert_allclose(native.evaluate(q,op),reference.evaluate(q,op),rtol=tolerance,atol=tolerance)
                if digits:
                    a=native.potential_mp(q);b=reference.potential_mp(q)
                    assert max(abs(a[i]-b[i]) for i in range(len(q)))<reference.arithmetic.ctx.mpf('1e-44')
                else:
                    np.testing.assert_allclose(native.potential(q),reference.potential(q),rtol=tolerance,atol=tolerance)
                assert native.diagnostics['factorizations']==len(cover.radii)
            before=native.evaluate(q)
            with pytest.raises(ValueError):native.refit(np.ones((len(x),3)))
            np.testing.assert_allclose(native.evaluate(q),before)
            assert native.evaluate(np.empty((0,2))).shape==(0,2)
            with pytest.raises(ValueError,match='cover'):
                native.evaluate([[20,20]])
            report=r.execution_info(native)
            assert report['locations']['glue']=='python'
            assert report['locations']['evaluation']=='cpp'
            assert report['glue_arithmetic']==('mpmath' if digits else 'float64')
        with pytest.raises(RuntimeError,match='closed'):
            native.evaluate(q)
    # A pooled process must reset all retained patch state before the next fit.
    with r.interpolate(space,x,data,precision,patches=cover,backend=backend) as second:
        np.testing.assert_allclose(second.evaluate(q),
                                   r.interpolate(space,x,data,precision,patches=cover).evaluate(q),
                                   rtol=2e-8,atol=2e-8)
