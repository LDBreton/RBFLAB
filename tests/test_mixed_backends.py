"""Native mixed contractions and MPI scheduling of the same local requests."""
import copy
import os
import numpy as np
import pytest
import rbflab as r


def _native():
    if os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')


def _scalar_request(tmp_path, execution=None):
    I, D = r.Identity(), r.Derivative(0)
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 1),
                          'b': r.ScalarSpace(r.Gaussian(1), 1)})
    A, B = r.FieldOperator({'a': [[I]]}), r.FieldOperator({'b': [[I]]})
    T = r.FieldOperator({'a': [[D]], 'b': [[I]]})
    x = r.geometry.unit_box_grid(2).points
    source = {'a': r.Samples(x, A), 'b': r.Samples(x, B)}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        backend=r.CppBackend(compute_condition=False, cache_dir=tmp_path),
        execution=execution)
    maps = local.operators(targets=[[.3, .4], [.6, .7], [.3, .4]], operators={'T': T})
    return maps, x


def test_mixed_mp80_native_six_targets_and_readonly_inspection(tmp_path):
    _native()
    I, Dx, Dy, Lap = r.Identity(), r.Derivative(0), r.Derivative(1), r.Laplacian()
    space = r.MixedSpace({
        'u': r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(2), r.PHS(7), '0.1'), 2),
        'p': r.ScalarSpace(r.Gaussian(1), 1),
    })
    U = r.FieldOperator({'u': [[I, 0], [0, I]]})
    P = r.FieldOperator({'p': [[I]]})
    L = r.FieldOperator({'u': [[-Lap, 0], [0, -Lap]], 'p': [[Dx], [Dy]]})
    GP, LP = r.FieldOperator({'p': [[Dx], [Dy]]}), r.FieldOperator({'p': [[Lap]]})
    x = r.geometry.unit_box_grid(2).points
    source = {'u': r.Samples(x, U), 'p': r.Samples(x, P),
              'momentum': r.Samples(x, L, target='exclude',
                   indices=[[i for i in range(len(x)) if i != 4]])}
    query = dict(targets=[[.5, .5]], operators={'L': L, 'P': P, 'GP': GP, 'LP': LP})
    kwargs = dict(source=source, trial=space.representers(source),
                  precision=r.Precision(local_digits=80, global_dtype='mpmath'))
    py = r.LocalApproximation(**kwargs, backend=r.PythonBackend(compute_condition=False)).operators(**query)
    cpp = r.LocalApproximation(**kwargs, backend=r.CppBackend(compute_condition=False,
                          cache_dir=tmp_path)).operators(**query)
    ctx = cpp.L._owner.arithmetic.ctx
    for name in query['operators']:
        p, c = py[name].local(0).weights, cpp[name].local(0).weights
        difference = max(abs(p[i, j]-c[i, j]) for i in range(p.rows) for j in range(p.cols))
        scale = max(abs(v) for v in p)
        assert difference/(scale or 1) < ctx.mpf('1e-65')
    report = copy.deepcopy(cpp.diagnostics['execution'])
    before = copy.deepcopy(cpp.L._owner.execution_info)
    native, reference = cpp.L.reconstruct_local(0), py.L.reconstruct_local(0)
    for key in ('matrix', 'rhs'):
        a, b = getattr(native, key), getattr(reference, key)
        difference = max(abs(a[i, j]-b[i, j]) for i in range(a.rows) for j in range(a.cols))
        assert difference < ctx.mpf('1e-70')
    assert cpp.L._owner.execution_info == before
    assert cpp.diagnostics['execution'] == report
    assert cpp.diagnostics['rhs_per_stencil'] == 6


def test_mixed_real_mpi_order_reuse_and_idle_rank(tmp_path):
    _native()
    if os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in real MPI integration')
    # Three ranks, two unique jobs: rank two is idle, not silently removed.
    serial, x = _scalar_request(tmp_path)
    mpi, _ = _scalar_request(tmp_path, r.LocalExecution(scheduler='mpi', ranks=3, reuse='exact'))
    np.testing.assert_allclose(mpi.T.matrix.toarray(), serial.T.matrix.toarray(), atol=2e-12)
    actual = mpi.T @ {'a': x[:, 0]+x[:, 1], 'b': 2+x[:, 0]}
    np.testing.assert_allclose(actual, [[3.3], [3.6], [3.3]], atol=2e-12)
    report = mpi.diagnostics['execution']
    assert report['ranks'] == 3
    assert report['requested_jobs'] == 3 and report['unique_jobs'] == 2
    assert mpi.diagnostics['factorizations'] == 2
    assert mpi.diagnostics['requested_stencils'] == 3
    assert report['cache_hits'] == 1
    assert [w['rank'] for w in report['workers']] == [0, 1, 2]
    assert [w['jobs'] for w in report['workers']] == [1, 1, 0]
    assert all(w['returncode'] == 0 and w['size'] == 3 for w in report['workers'])


def test_mixed_native_3d_local_scaling_zero_and_repeated_targets(tmp_path):
    _native()
    I, Lap = r.Identity(3), r.Laplacian(3)
    derivatives = [r.Derivative(j, 3) for j in range(3)]
    U = r.FieldOperator({'u': [[I, 0, 0], [0, I, 0], [0, 0, I]]})
    P = r.FieldOperator({'p': [[I]]})
    L = r.FieldOperator({'u': [[-Lap, 0, 0], [0, -Lap, 0], [0, 0, -Lap]],
                         'p': [[op] for op in derivatives]})
    zero, repeated = r.FieldOperator({'p': [[0]]}), r.FieldOperator({'p': [[I], [I]]})
    space = r.MixedSpace({'u': r.DivergenceFreeSpace(r.Gaussian(3), 1),
                          'p': r.ScalarSpace(r.Gaussian(2), 1)})
    x = np.array([[i, j, k] for i in (0., 1.) for j in (0., 1.) for k in (0., 1.)])
    source = {'u': r.Samples(x, U), 'p': r.Samples(x, P)}
    kwargs = dict(source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=40, global_dtype='mpmath'),
        stencil_policy=r.StencilPolicy(scaling='local'))
    query = dict(targets=[[.3, .4, .6]], operators={'L': L, 'zero': zero, 'repeat': repeated})
    py = r.LocalApproximation(**kwargs, backend=r.PythonBackend(compute_condition=False)).operators(**query)
    cpp = r.LocalApproximation(**kwargs, backend=r.CppBackend(compute_condition=False,
                    cache_dir=tmp_path)).operators(**query)
    values = {'u': np.c_[x[:, 1], -x[:, 0], np.ones(len(x))],
              'p': 2*x[:, 0]+3*x[:, 1]-x[:, 2]}
    ctx = cpp.L._owner.arithmetic.ctx
    for name in query['operators']:
        a, b = py[name].local(0).weights, cpp[name].local(0).weights
        difference = max(abs(a[i, j]-b[i, j]) for i in range(a.rows) for j in range(a.cols))
        assert difference < ctx.mpf('1e-32')
    np.testing.assert_allclose(np.asarray((cpp.L @ values).tolist(), float), [[2, 3, -1]], atol=1e-13)
    np.testing.assert_allclose(np.asarray((cpp.zero @ values).tolist(), float), [[0]], atol=0)
    np.testing.assert_allclose(np.asarray((cpp.repeat @ values).tolist(), float), [[1.2, 1.2]], atol=1e-13)


def test_execution_configuration_rejects_unsupported_scheduling():
    with pytest.raises(ValueError, match='Multiple ranks'):
        r.LocalExecution(ranks=2)
    with pytest.raises(ValueError, match='reuse'):
        r.LocalExecution(reuse='translated_approximate')
    from rbflab.mixed_backends import validate_execution
    with pytest.raises(NotImplementedError, match='CppBackend'):
        validate_execution(r.PythonBackend(), r.LocalExecution(scheduler='mpi', ranks=2))
    with pytest.raises(ValueError, match='require scheduler'):
        r.LocalExecution(launcher=('mpiexec',))
    with pytest.raises(ValueError, match='require scheduler'):
        r.LocalExecution(wsl_distribution='Ubuntu')


@pytest.mark.parametrize('execution', [False, 0, {}, 'serial'])
def test_falsey_and_untyped_execution_are_rejected(execution):
    from rbflab.mixed_backends import validate_execution
    with pytest.raises(TypeError, match='LocalExecution'):
        validate_execution(r.PythonBackend(), execution)
