"""Quality growth must screen the joint Hermite polynomial observations."""
import os
import numpy as np
import pytest

import rbflab as r


def _scalar(points, *, size=3, limit=4, indices=None, target='allow', policy=None):
    space = r.MixedSpace({'a': r.ScalarSpace(r.Gaussian(2), 1)})
    value = r.FieldOperator({'a': [[r.Identity()]]})
    source = {'values': r.Samples(points, value, size=None if indices is not None else size,
                                  indices=indices, target=target)}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        stencil_policy=policy or r.StencilPolicy(selection='quality', max_size=limit),
        backend=r.PythonBackend(compute_condition=False))
    return local, value


def test_quality_grows_past_collinear_nearest_neighbors():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [0., 1.]])
    local, value = _scalar(x)
    maps = local.operators(targets=x[:1], operators={'dy': r.FieldOperator({'a': [[r.Derivative(1)]]})})
    np.testing.assert_array_equal(maps.dy.local(0).groups['values'], [0, 1, 2, 3])
    np.testing.assert_allclose(maps.dy @ {'values': 1+2*x[:, 0]+3*x[:, 1]}, [[3]], atol=1e-12)
    nearest, _ = _scalar(x, policy=r.StencilPolicy())
    with pytest.raises(np.linalg.LinAlgError, match='not unisolvent'):
        nearest.operators(targets=x[:1], operators={'value': value})


def test_quality_respects_limits_and_fixed_memberships():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [0., 1.]])
    limited, value = _scalar(x, limit=3)
    with pytest.raises(np.linalg.LinAlgError, match='Mixed stencil quality failed.*max_size'):
        limited.operators(targets=x[:1], operators={'value': value})
    fixed, _ = _scalar(x, indices=[[0, 1, 2]])
    with pytest.raises(np.linalg.LinAlgError, match='not unisolvent'):
        fixed.operators(targets=x[:1], operators={'value': value})


def test_quality_respects_owner_exclusion_and_eligible_limit():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.], [0., 1.]])
    local, value = _scalar(x, target='exclude')
    maps = local.operators(targets=x[:1], operators={'value': value})
    np.testing.assert_array_equal(maps.value.local(0).groups['values'], [1, 2, 3, 4])
    invalid, _ = _scalar(x, limit=5, target='exclude')
    with pytest.raises(ValueError, match='eligible point count'):
        invalid.operators(targets=x[:1], operators={'value': value})


def test_quality_uses_joint_velocity_and_momentum_polynomials():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [0., 1.]])
    I, Lap, Dx, Dy = r.Identity(), r.Laplacian(), r.Derivative(0), r.Derivative(1)
    U = r.FieldOperator({'u': [[I, 0], [0, I]]})
    L = r.FieldOperator({'u': [[-Lap, 0], [0, -Lap]], 'p': [[Dx], [Dy]]})
    space = r.MixedSpace({'u': r.DivergenceFreeSpace(r.Gaussian(2), 1),
                          'p': r.PressureSpace(r.Gaussian(1), 1)})
    # The single momentum site has no independently unisolvent velocity-value
    # polynomial set; its two rows supply exactly the pressure gradients.
    source = {'velocity': r.Samples(x, U, size=3),
              'momentum': r.Samples(x, L, indices=[[1]])}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        stencil_policy=r.StencilPolicy(selection='quality', max_size=4),
        backend=r.PythonBackend(compute_condition=False))
    maps = local.operators(targets=x[:1], operators={'u': U, 'L': L})
    np.testing.assert_array_equal(maps.u.local(0).groups['velocity'], [0, 1, 2, 3])
    np.testing.assert_array_equal(maps.u.local(0).groups['momentum'], [1])
    values = {'velocity': np.c_[1+x[:, 0]+2*x[:, 1], 2+3*x[:, 0]-x[:, 1]],
              'momentum': np.tile([3., -2.], (len(x), 1))}
    np.testing.assert_allclose(maps.u @ values, [[1, 2]], atol=1e-11)
    np.testing.assert_allclose(maps.L @ values, [[3, -2]], atol=1e-11)


def test_quality_accepts_complementary_hermite_rows_at_one_site():
    x = np.array([[0., 0.]])
    space = r.MixedSpace({'a': r.ScalarSpace(r.Gaussian(2), 1)})
    value = r.FieldOperator({'a': [[r.Identity()]]})
    gradient = r.FieldOperator({'a': [[r.Derivative(0)], [r.Derivative(1)]]})
    source = {'value': r.Samples(x, value, size=1),
              'gradient': r.Samples(x, gradient, size=1)}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        stencil_policy=r.StencilPolicy(selection='quality'),
        backend=r.PythonBackend(compute_condition=False))
    maps = local.operators(targets=x, operators={'gradient': gradient})
    np.testing.assert_allclose(maps.gradient @ {'value': [1.], 'gradient': [[2., 3.]]}, [[2, 3]])


def test_quality_enforces_joint_threshold_and_physical_separation():
    x = np.array([[0., 0.], [1., 1.], [.5, .5+1e-6], [0., 2.]])
    strict = r.StencilPolicy(selection='quality', max_size=4,
                             min_polynomial_ratio=.1, min_separation_ratio=1e-8)
    local, value = _scalar(x, policy=strict)
    # Use a nodal target that makes the nearly collinear triangle nearest.
    maps = local.operators(targets=x[[1]], operators={'value': value})
    assert len(maps.value.local(0).groups['values']) == 4
    x = np.array([[0., 0.], [1., 0.], [1., 1e-6], [0., 1.]])
    crowded, _ = _scalar(x, policy=r.StencilPolicy(selection='quality', max_size=4,
                                                   min_separation_ratio=.01))
    with pytest.raises(np.linalg.LinAlgError, match='separation_ratios'):
        crowded.operators(targets=x[[1]], operators={'value': value})


def test_loose_quality_threshold_still_grows_until_numerically_unisolvent():
    x = np.array([[0., 0.], [1., 1.], [.5, .5+1e-16], [0., 2.]])
    local, value = _scalar(x, policy=r.StencilPolicy(selection='quality', max_size=4,
                                                     min_polynomial_ratio=1e-30))
    maps = local.operators(targets=x[[1]], operators={'value': value})
    assert len(maps.value.local(0).groups['values']) == 4


def test_group_nearest_override_does_not_grow():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [0., 1.]])
    space = r.MixedSpace({'a': r.ScalarSpace(r.Gaussian(2), 1)})
    value = r.FieldOperator({'a': [[r.Identity()]]})
    source = {'value': r.Samples(x, value, size=3, policy=r.StencilPolicy(selection='nearest'))}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        stencil_policy=r.StencilPolicy(selection='quality', max_size=4))
    with pytest.raises(np.linalg.LinAlgError, match='not unisolvent'):
        local.operators(targets=x[:1], operators={'value': value})


@pytest.mark.parametrize('scheduler', ['serial', 'mpi'])
def test_quality_growth_native_and_mpi_parity(scheduler, tmp_path):
    if os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')
    if scheduler == 'mpi' and os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in real MPI integration')
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [0., 1.]])
    local, _ = _scalar(x)
    local.precision = r.Precision(local_digits=40, global_dtype='mpmath')
    query = dict(targets=x[:2], operators={'dy': r.FieldOperator({'a': [[r.Derivative(1)]]})})
    reference = local.operators(**query)
    local.backend = r.CppBackend(compute_condition=False, cache_dir=tmp_path)
    local.execution = r.LocalExecution(scheduler=scheduler, ranks=3 if scheduler == 'mpi' else 1)
    native = local.operators(**query)
    ctx = native.dy.matrix.ctx
    for i in range(2):
        a, b = reference.dy.local(i), native.dy.local(i)
        np.testing.assert_array_equal(a.groups['values'], b.groups['values'])
        assert len(b.groups['values']) == 4
        assert max(abs(a.weights[j, 0]-b.weights[j, 0]) for j in range(4)) < ctx.mpf('1e-30')
    np.testing.assert_allclose(np.asarray((native.dy @ {'values': 1+2*x[:, 0]+3*x[:, 1]}).tolist(), float),
                               [[3], [3]], atol=1e-12)
    assert native.diagnostics['execution']['scheduler'] == scheduler
