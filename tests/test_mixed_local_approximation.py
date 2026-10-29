"""Generic mixed Hermite contracts, independent of a fluid PDE solver."""
from fractions import Fraction
import os

import numpy as np
import pytest
import rbflab as r


def array(value):
    return np.asarray(value.tolist() if hasattr(value, 'rows') else value, dtype=float)


def scalar_recipe(digits=None, backend=None, execution=None):
    x = r.geometry.unit_box_grid(2).points
    I, dx, lap = r.Identity(), r.Derivative(0), r.Laplacian()
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 2),
                          'b': r.ScalarSpace(r.Gaussian(1), 2)})
    # Invertibly mixed observations exercise coupling inside the local Gram
    # matrix; independent a and b weight calculations cannot be substituted.
    total = r.FieldOperator({'a': [[I]], 'b': [[I]]})
    contrast = r.FieldOperator({'a': [[I]], 'b': [[-2*I]]})
    source = {'total': r.Samples(x, operator=total),
              'contrast': r.Samples(x, operator=contrast)}
    options = dict(source=source, trial=space.representers(source),
                   backend=backend or r.PythonBackend(compute_condition=False),
                   precision=r.Precision(local_digits=digits,
                       global_dtype='mpmath' if digits else 'float64'))
    if execution is not None:
        options['execution'] = execution
    local = r.LocalApproximation(**options)
    targets = np.array([[.2, .3], [.7, .4]])
    ops = local.operators(targets=targets, operators={
        'values': r.FieldOperator({'a': [[I], [0]], 'b': [[0], [I]]}),
        'coupled': r.FieldOperator({'a': [[lap]], 'b': [[dx]]}),
        'repeated': r.FieldOperator({'a': [[I], [I]]}),
        'zero': r.FieldOperator({'a': [[0]]}),
    })
    a, b = x[:, 0]**2+2*x[:, 1], 1+3*x[:, 0]-x[:, 1]**2
    return local, ops, x, targets, {'total': a+b, 'contrast': a-2*b}


@pytest.mark.parametrize('digits', [None, 40])
def test_coupled_scalar_observations_and_different_target_arities(digits):
    _, ops, x, target, data = scalar_recipe(digits)
    exact = np.c_[target[:, 0]**2+2*target[:, 1],
                  1+3*target[:, 0]-target[:, 1]**2]
    np.testing.assert_allclose(array(ops['values'] @ data), exact, atol=2e-10)
    np.testing.assert_allclose(array(ops.coupled @ data), 5*np.ones((2, 1)), atol=2e-9)
    np.testing.assert_allclose(array(ops.repeated @ data), exact[:, [0, 0]], atol=2e-10)
    np.testing.assert_allclose(array(ops.zero @ data), np.zeros((2, 1)), atol=2e-12)
    assert ops['values'].matrix.shape == (4, 2*len(x))
    assert ops.coupled.matrix.shape == (2, 2*len(x))
    for op in ops['values'], ops.coupled:
        state, weights = op.reconstruct_local(0), op.local(0)
        w = np.vstack([array(weights.weights), array(weights.multipliers)])
        np.testing.assert_allclose(array(state.matrix).T @ w, array(state.rhs),
                                   atol=2e-9, rtol=2e-9)
        assert len(weights.groups['total']) == len(x)
    # This block is nonzero precisely because observations involve both fields.
    G = array(ops['values'].reconstruct_local(0).matrix)
    assert np.max(abs(G[:len(x), len(x):2*len(x)])) > .1


@pytest.mark.parametrize('digits', [None, 40])
def test_df_vector_scalar_sources_preserve_arity_and_polynomials(digits):
    x = r.geometry.unit_box_grid(2).points
    I, dx, dy, lap = r.Identity(), r.Derivative(0), r.Derivative(1), r.Laplacian()
    U = r.FieldOperator({'u': [[I, 0], [0, I]]})
    P = r.FieldOperator({'p': [[I]]})
    L = r.FieldOperator({'u': [[-Fraction(1, 10)*lap, 0],
                              [0, -Fraction(1, 10)*lap]],
                         'p': [[dx], [dy]]})
    space = r.MixedSpace({'u': r.DivergenceFreeSpace(r.IMQ(2), 1),
                          'p': r.ScalarSpace(r.Gaussian(1), 1)})
    source = {'u': r.Samples(x, operator=U),
              'p': r.Samples(x, operator=P),
              'equation': r.Samples(x[[1, 3, 5, 7]], operator=L)}
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        backend=r.PythonBackend(compute_condition=False),
        precision=r.Precision(local_digits=digits,
                             global_dtype='mpmath' if digits else 'float64'))
    y = np.array([[.3, .2], [.65, .8]])
    ops = local.operators(targets=y, operators={'p': P, 'u': U, 'L': L})
    values = {'u': np.c_[1+x[:, 0], 2-x[:, 1]],
              'p': 1+2*x[:, 0]-3*x[:, 1],
              'equation': np.tile([2., -3.], (4, 1))}
    np.testing.assert_allclose(array(ops.p @ values),
        (1+2*y[:, 0]-3*y[:, 1])[:, None], atol=2e-9)
    np.testing.assert_allclose(array(ops.u @ values), np.c_[1+y[:, 0], 2-y[:, 1]], atol=2e-9)
    np.testing.assert_allclose(array(ops.L @ values), np.tile([2., -3.], (2, 1)), atol=2e-9)
    assert ops.p['u'].matrix.shape == (2, 18)
    assert ops.p['p'].matrix.shape == (2, 9)
    assert ops.u['p'].matrix.shape == (4, 9)
    # A column-shaped scalar source must not trigger broadcast to n by n.
    shaped = dict(values, p=values['p'][:, None])
    np.testing.assert_allclose(array(ops.p @ shaped), array(ops.p @ values), atol=1e-12)
    assert array(ops.p @ shaped).shape == (2, 1)


def test_field_operator_validation_and_explicit_mixed_binding():
    I = r.Identity()
    for blocks in ({}, {'a': [[I, 0], [I]]}, {'a': [[I]], 'b': [[I], [I]]}):
        with pytest.raises((ValueError, TypeError)):
            r.FieldOperator(blocks)
    x = r.geometry.unit_box_grid(2).points
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 1)})
    for operator in (None, I):
        source = {'a': r.Samples(x, operator=operator)}
        with pytest.raises((ValueError, TypeError)):
            r.LocalApproximation(source=source, trial=space.representers(source))
    for functional in (r.FieldOperator({'missing': [[I]]}),
                       r.FieldOperator({'a': [[I, I]]})):
        with pytest.raises((ValueError, TypeError)):
            source = {'a': r.Samples(x, operator=functional)}
            r.LocalApproximation(source=source, trial=space.representers(source)).operators(
                targets=x[:1], operators={'value': functional})
    with pytest.raises(NotImplementedError):
        space.translates(x)


def test_same_site_degenerate_source_rows_rejected_and_target_exclusion_kept():
    x = r.geometry.unit_box_grid(2).points
    A = r.FieldOperator({'a': [[r.Identity()]]})
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 1)})
    source = {'first': r.Samples(x, operator=A), 'duplicate': r.Samples(x, operator=A)}
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        r.LocalApproximation(source=source, trial=space.representers(source)).operators(
            targets=x[:1], operators={'value': A})
    source = {'a': r.Samples(x, operator=A, size=8, target='exclude')}
    op = r.LocalApproximation(source=source, trial=space.representers(source)).operators(
        targets=x[:1], operators={'value': A}).value
    assert 0 not in op.local(0).groups['a']


def test_mixed_normal_functionals_bind_geometry_and_asymmetric_trials_reject():
    from types import SimpleNamespace
    x = r.geometry.unit_box_grid(2).points
    z = np.array([[.2, .3], [.7, .6]])
    normals = np.array([[1., 0.], [0., 1.]])
    boundary = SimpleNamespace(points=z, normals=normals)
    I = r.Identity()
    A = r.FieldOperator({'a': [[I]]})
    B = r.FieldOperator({'b': [[I]]})
    normal = r.FieldOperator({'a': [[r.NormalDerivative()]], 'b': [[I]]})
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 2),
                          'b': r.ScalarSpace(r.Gaussian(1), 2)})
    source = {'a': r.Samples(x, operator=A), 'b': r.Samples(x, operator=B),
              'normal': r.Samples(boundary, operator=normal)}
    op = r.LocalApproximation(source=source, trial=space.representers(source)).operators(
        targets=boundary, operators={'normal': normal}).normal
    az = 2*z[:, 0]*normals[:, 0]+2*normals[:, 1]
    bz = 1+3*z[:, 0]-z[:, 1]**2
    data = {'a': x[:, 0]**2+2*x[:, 1],
            'b': 1+3*x[:, 0]-x[:, 1]**2, 'normal': az+bz}
    np.testing.assert_allclose(array(op @ data), (az+bz)[:, None], atol=2e-8)
    shifted = dict(source)
    shifted['a'] = r.Samples(x, operator=r.FieldOperator({'a': [[I+r.Derivative(0)]]}))
    with pytest.raises(NotImplementedError):
        r.LocalApproximation(source=source, trial=space.representers(shifted))


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')
@pytest.mark.parametrize('digits', [None, 40])
def test_mixed_cpp_matches_python(digits):
    _, reference, _, _, data = scalar_recipe(digits)
    _, native, _, _, _ = scalar_recipe(digits,
        backend=r.CppBackend(threads=1, compute_condition=False))
    for name in reference:
        np.testing.assert_allclose(native[name].to_scipy().toarray(),
                                   reference[name].to_scipy().toarray(),
                                   rtol=3e-8, atol=3e-8)
        np.testing.assert_allclose(array(native[name] @ data),
                                   array(reference[name] @ data), atol=3e-8)


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in real MPI integration')
def test_mixed_mpi_restores_multi_target_maps():
    _, reference, _, _, data = scalar_recipe(40,
        backend=r.CppBackend(threads=1, compute_condition=False))
    _, distributed, _, _, _ = scalar_recipe(40,
        backend=r.CppBackend(threads=1, compute_condition=False),
        execution=r.LocalExecution(scheduler='mpi', ranks=2))
    for name in reference:
        np.testing.assert_allclose(distributed[name].to_scipy().toarray(),
                                   reference[name].to_scipy().toarray(),
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(array(distributed[name] @ data),
                                   array(reference[name] @ data), atol=2e-12)

