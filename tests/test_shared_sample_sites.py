import numpy as np
import pytest
import rbflab as r
from rbflab.samples import sample_groups, select_groups


def test_shared_sites_filter_without_refilling_and_keep_source_order():
    points = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    groups = sample_groups({
        'child': r.Samples(points[[0, 2, 3]], within='parent', target='exclude'),
        'parent': r.Samples(points, size=3),
    })
    selected = select_groups(groups, points[0], 0, r.StencilPolicy(), 0)
    assert list(selected) == ['child', 'parent']
    np.testing.assert_array_equal(selected['child'], [1])
    np.testing.assert_array_equal(selected['parent'], [0, 1, 2])


@pytest.mark.parametrize('kwargs', [{'size': 3}, {'indices': [[0]]}, {'policy': r.StencilPolicy()}])
def test_shared_sites_reject_conflicting_selection(kwargs):
    with pytest.raises(ValueError, match='within'):
        r.Samples([[0., 0.]], within='a', **kwargs)


def test_shared_sites_validate_references():
    with pytest.raises(ValueError, match='Unknown'):
        sample_groups({'a': r.Samples([[0., 0.]], within='missing')})
    with pytest.raises(ValueError, match='Cyclic'):
        sample_groups({'a': r.Samples([[0., 0.]], within='b'),
                       'b': r.Samples([[0., 0.]], within='a')})


def test_geometric_shared_stencil_partitions_without_refill():
    cloud = r.geometry.unit_box_grid(8)
    stencil = r.SharedStencil(cloud.points, size=37)
    groups = sample_groups({
        'inside': r.Samples(cloud.interior, within=stencil),
        'wall': r.Samples(cloud.points[cloud.boundary_indices], within=stencil),
        'pde': r.Samples(cloud.interior, within=stencil, target='exclude'),
    })
    for target in cloud.interior[[0, 24]]:
        chosen = select_groups(groups, target, 0, r.StencilPolicy(), 0)
        physical = np.concatenate([groups[k].points[chosen[k]] for k in ('inside', 'wall')])
        assert len(physical) == 37
        assert set(map(tuple, physical)) == set(map(tuple, stencil.points[stencil.select(target)]))
        assert len(chosen['pde']) == len(chosen['inside'])-1
    with pytest.raises(ValueError):
        r.SharedStencil(cloud.points, size=82)
    with pytest.raises(ValueError):
        r.SharedStencil([[0, 0], [0, 0]], size=1)
    with pytest.raises(ValueError):
        r.Samples([[0, 0, 0]], within=stencil)


def test_mixed_shared_sites_match_explicit_memberships():
    points = r.geometry.unit_box_grid(2).points
    target = points[[4]]
    I, dx, dy, lap = r.Identity(), r.Derivative(0), r.Derivative(1), r.Laplacian()
    value = r.FieldOperator({'u': [[I, 0], [0, I]]})
    momentum = r.FieldOperator({'u': [[-lap, 0], [0, -lap]], 'p': [[dx], [dy]]})
    space = r.MixedSpace({'u': r.DivergenceFreeSpace(r.Gaussian(1), 2),
                          'p': r.PressureSpace(r.Gaussian(1), 1)})
    sources = [
        {'velocity': r.Samples(points, operator=value),
         'momentum': r.Samples(points, operator=momentum, within='velocity', target='exclude')},
        {'velocity': r.Samples(points, operator=value, indices=[np.arange(9)]),
         'momentum': r.Samples(points, operator=momentum, indices=[np.delete(np.arange(9), 4)])},
    ]
    matrices = []
    for source in sources:
        local = r.LocalApproximation(source=source, trial=space.representers(source))
        maps = local.operators(targets=target, operators={'L': momentum})
        matrices.append(maps.L.to_scipy().toarray())
    np.testing.assert_array_equal(*matrices)


def test_scalar_shared_sites_match_explicit_memberships():
    points = r.geometry.unit_box_grid(2).points
    space = r.ScalarSpace(r.Gaussian(1), polynomial_degree=2)
    matrices = []
    for shared in (True, False):
        selection = dict(within='value', target='exclude') if shared else dict(
            indices=[np.delete(np.arange(9), 4)])
        source = {'derivative': r.Samples(points, operator=r.Derivative(0), **selection),
                  'value': r.Samples(points)}
        local = r.LocalApproximation(source=source, trial=space.representers(source))
        maps = local.operators(targets=points[[4]], operators={'L': r.Laplacian()})
        matrices.append(maps.L.to_scipy().toarray())
    np.testing.assert_array_equal(*matrices)
