"""Public fixed-stencil selection and LocalApproximation scaling names."""
import os

import numpy as np
import pytest

import rbflab as r
from rbflab.samples import sample_groups, select_groups


def test_stencil_alias_and_fixed_size_selection():
    points = r.geometry.unit_box_grid(8).points
    stencil = r.Stencil(points, size=37)
    assert r.SharedStencil is r.Stencil
    assert len(stencil.select([.43, .51])) == 37

    groups = {
        'inside': r.Samples(points, within=stencil),
        'wall': r.Samples(points, within=stencil),
    }
    # The same physical selection can be partitioned across sample groups;
    # the stencil itself does not grow under a quality-selection policy.
    groups = sample_groups(groups)
    selected = select_groups(groups, [.43, .51], 0,
        r.StencilPolicy(selection='quality'), 1)
    assert len(selected['inside']) == 37


def test_stencil_explicit_indices_are_fixed():
    points = r.geometry.unit_box_grid(3).points
    stencil = r.Stencil(points, indices=[[8, 0, 4]])
    np.testing.assert_array_equal(stencil.select([.5, .5]), [8, 0, 4])
    groups = sample_groups({'u': r.Samples(points, within=stencil)})
    selected = select_groups(groups, [.5, .5], 0,
        r.StencilPolicy(selection='quality'), 1)
    np.testing.assert_array_equal(selected['u'], [8, 0, 4])


def _scalar_matrices(scaling, backend):
    points = r.geometry.unit_box_grid(3).points
    source = {'u': r.Samples(points)}
    space = r.ScalarSpace(r.Gaussian(1.3), 2)
    local = r.LocalApproximation(source=source, space=space,
        stencil_size=len(points), kernel_scaling=scaling, backend=backend,
        precision=r.Precision(local_digits=50))
    return local.operators(targets=points[[4, 5]], operators={
        'value': r.Identity(), 'dx': r.Derivative(0), 'lap': r.Laplacian()})


@pytest.mark.parametrize('scaling', ['physical', None, 'local'])
def test_scalar_new_scaling_name_matches_legacy_policy(scaling):
    points = r.geometry.unit_box_grid(3).points
    policy = r.StencilPolicy(scaling='local' if scaling == 'local' else 'physical')
    source = {'u': r.Samples(points)}
    space = r.ScalarSpace(r.Gaussian(1.3), 2)
    legacy = r.LocalApproximation(source=source, space=space,
        stencil_size=len(points), stencil_policy=policy).operators(
            targets=points[[4, 5]], operators={'value': r.Identity(),
                'dx': r.Derivative(0), 'lap': r.Laplacian()})
    modern = r.LocalApproximation(source=source, space=space,
        stencil_size=len(points), kernel_scaling=scaling).operators(
            targets=points[[4, 5]], operators={'value': r.Identity(),
                'dx': r.Derivative(0), 'lap': r.Laplacian()})
    for name in legacy:
        np.testing.assert_allclose(modern[name].matrix.toarray(),
            legacy[name].matrix.toarray(), atol=2e-12, rtol=2e-12)


def test_scaling_conflicts_and_invalid_values():
    points = r.geometry.unit_box_grid(3).points
    source = {'u': r.Samples(points)}
    space = r.ScalarSpace(r.Gaussian(1), 2)
    with pytest.raises(ValueError):
        r.LocalApproximation(source=source, space=space,
            stencil_policy=r.StencilPolicy(scaling='local'),
            kernel_scaling='physical')
    with pytest.raises(ValueError):
        r.LocalApproximation(source=source, space=space, kernel_scaling='bad')


@pytest.mark.parametrize('scaling', ['physical', 'local'])
@pytest.mark.parametrize('backend_name', ['python', 'cpp', 'mpi'])
def test_mixed_new_scaling_name_matches_legacy_policy(scaling, backend_name, tmp_path):
    if backend_name != 'python' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native mixed integration')
    if backend_name == 'mpi' and os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in real MPI integration')
    points = r.geometry.unit_box_grid(2).points
    I, dx, lap = r.Identity(), r.Derivative(0), r.Laplacian()
    space = r.MixedSpace({'a': r.ScalarSpace(r.IMQ(2), 2),
                          'b': r.ScalarSpace(r.Gaussian(1), 2)})
    total = r.FieldOperator({'a': [[I]], 'b': [[I]]})
    contrast = r.FieldOperator({'a': [[I]], 'b': [[-2*I]]})
    source = {'total': r.Samples(points, operator=total),
              'contrast': r.Samples(points, operator=contrast)}
    target = np.array([[.2, .3], [.7, .4]])
    operators = {'value': r.FieldOperator({'a': [[I], [0]], 'b': [[0], [I]]}),
                 'coupled': r.FieldOperator({'a': [[lap]], 'b': [[dx]]})}
    matrices = []
    backend = (r.PythonBackend() if backend_name == 'python' else
               r.CppBackend(cache_dir=tmp_path))
    execution = (r.LocalExecution(scheduler='mpi', ranks=2)
                 if backend_name == 'mpi' else None)
    for kwargs in ({'stencil_policy': r.StencilPolicy(scaling=scaling)},
                   {'kernel_scaling': scaling}):
        local = r.LocalApproximation(source=source,
            trial=space.representers(source), backend=backend,
            execution=execution, **kwargs)
        matrices.append(local.operators(targets=target, operators=operators))
    for name in operators:
        dense = lambda op: (op.to_scipy().toarray() if hasattr(op, 'to_scipy')
                            else op.matrix.toarray())
        np.testing.assert_allclose(dense(matrices[1][name]),
            dense(matrices[0][name]), atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize('backend_name', ['python', 'cpp'])
@pytest.mark.parametrize('scaling', ['physical', 'local'])
def test_scalar_optional_cpp_scaling_parity(backend_name, scaling, tmp_path):
    if backend_name == 'cpp' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')
    backend = (r.PythonBackend() if backend_name == 'python'
               else r.CppBackend(cache_dir=tmp_path))
    modern = _scalar_matrices(scaling, backend)
    reference = _scalar_matrices(scaling, r.PythonBackend())
    for name in modern:
        np.testing.assert_allclose(modern[name].to_scipy().toarray(),
            reference[name].to_scipy().toarray(), atol=2e-12, rtol=2e-12)
