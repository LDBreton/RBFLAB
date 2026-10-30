"""Fast independent checks of coupling equations; no local weights are built."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, vstack

from research.shared_pressure_lhi.coupling_study import (
    GROUPS, OPERATORS, Equations, LeastSquares, build_maps, evaluate, exact_fields, select,
)


@pytest.fixture
def synthetic_maps():
    rng = np.random.default_rng(516)
    nodes = np.array([[.25, .25], [.65, .55]])
    boundary = np.array([[0., 0.], [1., 0.], [0., 1.]])
    targets = np.vstack([nodes, rng.uniform(.1, .9, (8, 2)), [[.45, .4], [.45, .4]]])
    geo = dict(nodes=nodes, boundary=boundary, targets=targets,
               owners=np.array([0, 1]+[0]*4+[1]*4+[0, 1]),
               auxiliary=np.arange(2, 10), overlap=np.array([[10, 11]]))
    columns = dict(velocity=4, pressure=2, boundary=6, momentum=4)
    maps = {op: {g: csr_matrix(rng.normal(size=(len(targets)*(1 if op == 'pressure' else 2), columns[g])))
                 for g in GROUPS} for op in OPERATORS}
    return maps, geo


@pytest.mark.parametrize('dt', [None, .125])
@pytest.mark.parametrize('variant', [
    'baseline', 'pde2', 'pde4', 'velocity', 'pressure', 'traction',
    'pde4_velocity', 'pde4_velocity_pressure', 'pde4_traction',
])
def test_global_residual_matches_direct_patch_evaluation(synthetic_maps, dt, variant):
    maps, geo = synthetic_maps
    eq = Equations(maps, geo, dt, variant)
    rng = np.random.default_rng(99)
    previous = {g: rng.normal(size=maps['velocity'][g].shape[1]) for g in GROUPS}
    values = rng.normal(size=6)
    rhs, fields, wall, rvalue = eq.rhs('polynomial', .3, previous)
    data = dict(velocity=values[:4], pressure=values[4:], boundary=wall,
                momentum=rvalue-eq.alpha*values[:4])
    expected = [evaluate(select(maps, 'momentum', [0, 1]), data)-data['momentum'],
                values[4:]-evaluate(select(maps, 'pressure', [0, 1]), data)]
    for kind, block, ids, old_velocity in eq.extras:
        residual = evaluate(block, data)
        if kind == 'pde':
            test = exact_fields('polynomial', geo['targets'][ids], .3)
            residual -= (test['momentum'] if dt is None else test['force']).ravel()
            if dt is not None:
                residual -= eq.alpha*evaluate(old_velocity, previous)
        expected.append(residual)
    np.testing.assert_allclose(eq.matrix@values-rhs, np.concatenate(expected), rtol=1e-13, atol=1e-12)


def test_traction_uses_symmetric_gradient_and_same_oblique_normal(synthetic_maps):
    maps, geo = synthetic_maps
    eq = Equations(maps, geo, None, 'traction')
    rng = np.random.default_rng(91)
    data = {g: rng.normal(size=maps['velocity'][g].shape[1]) for g in GROUPS}
    dx = evaluate(select(maps, 'dx', [10, 11]), data).reshape(2, 2)
    dy = evaluate(select(maps, 'dy', [10, 11]), data).reshape(2, 2)
    pressure = evaluate(select(maps, 'pressure', [10, 11]), data)
    stresses = [.01*np.array([[2*dx[i, 0], dy[i, 0]+dx[i, 1]],
                             [dy[i, 0]+dx[i, 1], 2*dy[i, 1]]])-pressure[i]*np.eye(2)
                for i in range(2)]
    normal = geo['nodes'][1]-geo['nodes'][0]
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(evaluate(eq.extras[0][1], data), (stresses[0]-stresses[1])@normal)


def test_least_squares_gauge_and_weighted_incompatible_residual():
    # Four velocities, a pressure difference, and a redundant pressure row.
    matrix = np.zeros((6, 6))
    matrix[:4, :4] = np.diag([1., 2., 3., 4.])
    matrix[4:, 4:] = [[1., -1.], [-1., 1.]]
    equations = SimpleNamespace(n=2, matrix=csr_matrix(matrix), K=csr_matrix(matrix),
                                H=csr_matrix((0, 6)))
    factor = LeastSquares(equations)
    assert factor.full_rank
    exact = np.array([.2, -.3, .5, .9, .7, -.7])
    np.testing.assert_allclose(factor.solve(matrix@exact), exact, atol=1e-14)
    rhs = matrix@exact+np.array([0., 0., 0., 0., .5, .5])
    solution = factor.solve(rhs)
    assert abs(solution[4:].mean()) < 1e-14
    residual = matrix@solution-rhs
    assert np.linalg.norm(residual) > .1
    # Weighted normal-equation stationarity verifies the stated objective.
    np.testing.assert_allclose(factor.basis.T@matrix.T@(factor.row_scale**2*residual), 0., atol=1e-14)


def test_extra_equation_detects_missing_velocity_without_regularization():
    base = np.diag([0., 1., 1., 1., 0., 0.])
    base[4:, 4:] = [[1., -1.], [-1., 1.]]
    missing = SimpleNamespace(n=2, K=csr_matrix(base), matrix=csr_matrix(base),
                              H=csr_matrix((0, 6)))
    factor = LeastSquares(missing)
    assert not factor.full_rank
    with pytest.raises(ValueError, match='refusing pseudoinverse'):
        factor.solve(np.zeros(6))
    extra = csr_matrix([[1., 0., 0., 0., 0., 0.]])
    repaired = LeastSquares(SimpleNamespace(n=2, K=missing.K, H=extra,
                                            matrix=vstack([missing.K, extra])))
    assert repaired.full_rank
    assert repaired.diagnostics['base_weak_dimension'] == 1
    assert repaired.diagnostics['extra_on_weak_singular_values'][0] > .9


def test_smooth_fields_satisfy_momentum_and_time_derivative_independently():
    points = np.array([[.27, .34], [.61, .72]])
    time, step = .4, 2e-5
    fields = exact_fields('smooth', points, time)
    laplacian = np.zeros_like(points)
    divergence = np.zeros(len(points))
    for axis in range(2):
        offset = np.zeros_like(points)
        offset[:, axis] = step
        plus = exact_fields('smooth', points+offset, time)['velocity']
        minus = exact_fields('smooth', points-offset, time)['velocity']
        laplacian += (plus-2*fields['velocity']+minus)/step**2
        divergence += (plus[:, axis]-minus[:, axis])/(2*step)
    np.testing.assert_allclose(divergence, 0., atol=2e-8)
    np.testing.assert_allclose(fields['momentum'], -.01*laplacian+fields['gradient'], atol=1e-7)
    derivative = (exact_fields('smooth', points, time+step)['velocity']-
                  exact_fields('smooth', points, time-step)['velocity'])/(2*step)
    np.testing.assert_allclose(fields['force']-fields['momentum'], derivative, atol=1e-10)


def test_auxiliary_targets_reuse_owner_37_site_memberships(monkeypatch, tmp_path):
    from research.shared_pressure_lhi import coupling_study as study

    captured = {}

    class FakeBlock:
        def __init__(self, rows, columns):
            self.matrix = csr_matrix((rows, columns))

        def to_scipy(self):
            return self.matrix

    class FakeResult(dict):
        diagnostics = {'execution': None}

    class FakeApproximation:
        def __init__(self, **kwargs):
            captured['source'] = kwargs['source']

        def operators(self, *, targets, operators):
            captured['targets'] = targets.copy()
            return FakeResult({op: {g: FakeBlock(len(targets)*(1 if op == 'pressure' else 2),
                    len(samples.points)*(1 if g == 'pressure' else 2))
                for g, samples in captured['source'].items()} for op in operators})

    monkeypatch.setattr(study.r, 'LocalApproximation', FakeApproximation)
    _, geo, metadata = build_maps(tmp_path/'weights', cells=6, ranks=0)
    source, owners = captured['source'], geo['owners']
    assert metadata['stencil'] == 37
    assert not metadata['auxiliary_sites_are_observations']
    for target, owner in enumerate(owners):
        for group in GROUPS:
            np.testing.assert_array_equal(source[group].indices[target], source[group].indices[owner])
        locations = {tuple(point) for group in ('velocity', 'boundary')
                     for point in source[group].points[source[group].indices[target]]}
        assert len(locations) == 37
        assert owner not in source['pressure'].indices[target]
        assert owner not in source['momentum'].indices[target]
        assert owner in source['velocity'].indices[target]
    for target in geo['auxiliary']:
        for group in GROUPS:
            assert not np.any(np.all(source[group].points == geo['targets'][target], axis=1))
