import numpy as np
import pytest
import rbflab as r


def small():
    x=np.array([[0.,0.],[0.5,0.],[1.,0.],[0.,0.5],[0.5,0.5],[1.,0.5]])
    cover=r.PatchCover([[0.25,0.25],[0.75,0.25]],[0.8,0.8])
    return x,cover


def test_cover_and_query_validation():
    x,cover=small()
    with pytest.raises(ValueError):r.PatchCover([[0,0]],[0])
    with pytest.raises(ValueError):r.PatchCover([[0,0]],[np.inf])
    with pytest.raises(ValueError):r.interpolate(r.Gaussian(1),x,np.ones(len(x)),patches=r.PatchCover([[10,10]],[1]))
    with pytest.raises(ValueError):r.interpolate(r.Gaussian(1),np.r_[x,x[:1]],np.ones(len(x)+1),patches=cover)
    fit=r.interpolate(r.Gaussian(1),x,np.sin(x[:,0]),patches=cover)
    with pytest.raises(ValueError):fit.evaluate(np.array([[2.,2.]]))
    with pytest.raises(ValueError):fit.refit(np.r_[np.ones(len(x)-1),np.nan])
    np.testing.assert_allclose(fit.evaluate(x),np.sin(x[:,0]),atol=1e-8)
    fit.close()
    with pytest.raises(RuntimeError):fit.evaluate(x)
    with pytest.raises(RuntimeError):fit.refit(np.ones(len(x)))


def test_input_snapshot_and_boundary():
    x,cover=small()
    y=x[:,0]**2+x[:,1]
    fit=r.interpolate(r.Gaussian(1),x,y,patches=cover)
    x[:]=100
    fit.refit(lambda points:points[:,0]**2+points[:,1])
    q=np.array([[0.25,0.25]])
    assert np.isfinite(fit.evaluate(q)).all()
    assert fit.points.flags.writeable is False


def test_exact_support_boundary_is_excluded_and_other_patch_covers():
    x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    cover=r.PatchCover([[0.,0.],[0.5,0.5]],[1.,2.])
    rows=cover.memberships(x)
    assert 1 not in rows[0] and 2 not in rows[0]
    assert set(rows[1])==set(range(len(x)))
    fit=r.interpolate(r.Gaussian(1),x,np.sin(x[:,0]+x[:,1]),patches=cover)
    np.testing.assert_allclose(fit.evaluate(x),np.sin(x[:,0]+x[:,1]),atol=1e-8)
    assert np.isfinite(fit.evaluate(np.array([[1.,0.]]),r.Laplacian())).all()


def test_rank_deficient_patch_rejected_before_native_build(monkeypatch):
    from rbflab import global_backend
    x=np.array([[0.,0.],[0.5,0.],[1.,0.]])
    cover=r.PatchCover([[0.5,0.]],[2.])
    with pytest.raises(np.linalg.LinAlgError):
        r.interpolate(r.Gaussian(1),x,np.ones(len(x)),polynomial_degree=1,patches=cover)
    def forbidden(*args,**kwargs):raise AssertionError('native compilation started before geometry validation')
    monkeypatch.setattr(global_backend,'_compiled',forbidden)
    with pytest.raises(np.linalg.LinAlgError):
        r.interpolate(r.Gaussian(1),x,np.ones(len(x)),polynomial_degree=1,
                      patches=cover,backend=r.CppGlobalBackend())


def test_empty_mp_query_shapes_and_immutable_memberships():
    x,cover=small()
    precision=r.Precision(global_digits=40)
    scalar=r.interpolate(r.Gaussian(1),x,np.sin(x[:,0]),precision=precision,patches=cover)
    vector=r.interpolate(r.Gaussian(1),x,np.column_stack((x[:,0],x[:,1])),precision=precision,patches=cover)
    assert (scalar.evaluate_mp(np.empty((0,2))).rows,scalar.evaluate_mp(np.empty((0,2))).cols)==(0,1)
    assert (vector.evaluate_mp(np.empty((0,2))).rows,vector.evaluate_mp(np.empty((0,2))).cols)==(0,2)
    assert all(not rows.flags.writeable for rows in scalar.memberships)


def test_python_failed_solve_keeps_old_coefficients(monkeypatch):
    x,cover=small()
    y=np.sin(x[:,0]+x[:,1])
    fit=r.interpolate(r.Gaussian(1),x,y,patches=cover)
    q=np.array([[0.3,0.2]])
    before=fit.evaluate(q).copy()
    monkeypatch.setattr(fit.patches[1]['factor'],'solve',
                        lambda rhs:np.full(len(rhs),np.inf))
    with pytest.raises(FloatingPointError,match='Nonfinite patch coefficients'):
        fit.refit(2*y)
    np.testing.assert_allclose(fit.evaluate(q),before,rtol=0,atol=0)


def test_native_refit_atomic_condition_flag_and_pooled_precision():
    import os
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('native worker opt-in')
    x,cover=small()
    y=np.sin(x[:,0]+x[:,1])
    backend=r.CppGlobalBackend(threads=1,reuse_workers=True,compute_condition=False)
    previous_worker=None
    for digits in (40,60):
        precision=r.Precision(global_digits=digits)
        reference=r.interpolate(r.Gaussian(1),x,y,precision=precision,patches=cover)
        with r.interpolate(r.Gaussian(1),x,y,precision=precision,
                           patches=cover,backend=backend) as fit:
            if previous_worker is not None:
                assert fit.worker._worker is previous_worker
            previous_worker=fit.worker._worker
            q=np.array([[0.3,0.2]])
            before=fit.evaluate_mp(q)
            expected=reference.evaluate_mp(q)
            assert abs(before[0]-expected[0])<fit.arithmetic.ctx.mpf(10)**(-digits+10)
            empty=fit.evaluate_mp(np.empty((0,2)))
            assert (empty.rows,empty.cols)==(0,1)
            assert all(value is None for value in fit.diagnostics['scaled_conditions'])
            with pytest.raises(ValueError):fit.refit(np.r_[y[:-1],np.nan])
            after=fit.evaluate_mp(q)
            assert abs(before[0]-after[0])<fit.arithmetic.ctx.mpf('1e-45')
            assert all(not rows.flags.writeable for rows in fit.memberships)


def test_native_does_no_python_query_weight_or_matrix(monkeypatch):
    import os
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('native worker opt-in')
    from rbflab import patches
    from rbflab.nodal import NodalBasis
    x,cover=small()
    y=np.sin(x[:,0]+x[:,1])
    def forbidden(*args,**kwargs):raise AssertionError('Python bulk numerical path used')
    monkeypatch.setattr(NodalBasis,'system_matrix',forbidden)
    with r.interpolate(r.Gaussian(1),x,y,patches=cover,backend=r.CppGlobalBackend(threads=1)) as fit:
        monkeypatch.setattr(patches,'_normalized_weights',forbidden)
        assert np.isfinite(fit.evaluate(np.array([[0.3,0.2]]),r.Laplacian())).all()


def test_native_worker_pool_clears_patch_collection():
    import os
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('native worker opt-in')
    x,cover=small()
    backend=r.CppGlobalBackend(threads=1,reuse_workers=True,derivative_order=2)
    y=np.sin(x[:,0]+x[:,1])
    with r.interpolate(r.Gaussian(1),x,y,patches=cover,backend=backend) as fit:
        first_worker=fit.worker._worker
        np.testing.assert_allclose(fit.evaluate(x),y,atol=1e-8)
    with r.interpolate(r.Gaussian(1),x,y,backend=backend) as global_fit:
        assert global_fit.system.worker._worker is first_worker
        np.testing.assert_allclose(global_fit.evaluate(x),y,atol=1e-8)
    with r.interpolate(r.Gaussian(1),x,y,patches=cover,backend=backend) as fit:
        assert fit.worker._worker is first_worker
        np.testing.assert_allclose(fit.evaluate(x),y,atol=1e-8)
