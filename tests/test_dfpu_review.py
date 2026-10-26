"""Independent mathematical checks for shifted-potential divergence-free PU."""
import os

import numpy as np
import pytest

import rbflab as r


def _irregular_cloud():
    points = np.array([[0., 0.], [.2, .1], [0., .5], [.3, .7], [.6, .2],
                       [.8, 0.], [.7, .8], [1., .5], [1., 1.]])
    cover = r.PatchCover([[.15, .2], [.75, .3], [.5, .85]], [.75, .7, .68])
    data = np.column_stack((-np.cos(points[:, 1]) - points[:, 0],
                           np.cos(points[:, 0]) + points[:, 1]))
    return points, cover, data


def _independent_local_imq_potential(ctx, patch, coordinates):
    terms = []
    for j, node in enumerate(patch['basis'].points):
        x, y = [value - ctx.mpf(float(origin))
                for value, origin in zip(coordinates, node)]
        # Each patch may own a separate mpmath context. Convert operands
        # into the differentiating context before its temporary precision
        # increase; otherwise another context rounds away diff's steps.
        coefficient = [ctx.mpf(patch['coefficients'][2*j+k]) for k in range(2)]
        terms.append((coefficient[1]*x - coefficient[0]*y) /
                     (1+x*x+y*y)**ctx.mpf('1.5'))
    return ctx.fsum(terms)


def _independent_imq_potential(fit, coordinates):
    """Direct IMQ formula, independent of production derivative/weight code."""
    ctx = fit.arithmetic.ctx
    values, weights = [], []
    for patch, center, radius, shift in zip(
            fit.patches, fit.cover.centers, fit.cover.radii, fit.shifts):
        values.append(_independent_local_imq_potential(ctx, patch, coordinates) + shift)
        distance = ctx.sqrt(ctx.fsum((value-ctx.mpf(float(origin)))**2
                                    for value, origin in zip(coordinates, center)))
        t = distance/ctx.mpf(float(radius))
        weights.append((1-t)**4*(4*t+1) if t < 1 else ctx.zero)
    return ctx.fsum(weight*value for weight, value in zip(weights, values))/ctx.fsum(weights)


def test_nonpolynomial_velocity_and_jacobian_are_derivatives_of_actual_potential():
    points, cover, data = _irregular_cloud()
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ()), points, data,
                       precision=r.Precision(global_digits=60), patches=cover) as fit:
        ctx = fit.arithmetic.ctx
        assert len(fit.edges) == 3  # Includes an inconsistent glue cycle.
        queries = np.array([[.47, .43], [.37, .36]])
        for query in queries:
            position = tuple(ctx.mpf(float(value)) for value in query)
            scalar = lambda x, y: _independent_imq_potential(fit, (x, y))
            velocity = fit.evaluate_mp(query[None, :])
            dx = fit.evaluate_mp(query[None, :], r.Derivative(0))
            dy = fit.evaluate_mp(query[None, :], r.Derivative(1))
            expected = [(-ctx.diff(scalar, position, (0, 1)),
                         ctx.diff(scalar, position, (1, 0))),
                        (-ctx.diff(scalar, position, (1, 1)),
                         ctx.diff(scalar, position, (2, 0))),
                        (-ctx.diff(scalar, position, (0, 2)),
                         ctx.diff(scalar, position, (1, 1)))]
            for actual, reference in zip((velocity, dx, dy), expected):
                assert max(abs(actual[0, component]-reference[component])
                           for component in range(2)) < ctx.mpf('1e-48')
            # Verify the derivatives themselves, not merely divergence().
            assert abs(dx[0, 0]+dy[0, 1]) < ctx.mpf('1e-48')
        gradient = [ctx.zero]*len(fit.patches)
        for (i, j, point, _), weight in zip(fit.edges, fit.weights):
            coordinates = tuple(ctx.mpf(float(value)) for value in point)
            mismatch = (fit.shifts[i]-fit.shifts[j]
                        - _independent_local_imq_potential(ctx, fit.patches[j], coordinates)
                        + _independent_local_imq_potential(ctx, fit.patches[i], coordinates))
            gradient[i] += weight**2*mismatch
            gradient[j] -= weight**2*mismatch
        assert max(abs(value) for value in gradient) < ctx.mpf('1e-48')


def _quadratic_velocity(points):
    x, y = points[:, 0]-7, points[:, 1]+4
    return np.column_stack((-2*x*x+6*x*y-1.5*y*y+3,
                            3*x*x+4*x*y-3*y*y+2))


def test_degree_two_solenoidal_polynomials_with_shifted_scaled_patch_coordinates():
    grid = np.array(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5),
                               indexing='ij')).reshape(2, -1).T
    points = grid + [7., -4.]
    cover = r.PatchCover(np.array([[-.55, 0], [.55, 0], [0, .55]]) + [7., -4.],
                         [1.35]*3)
    queries = np.array([[7.125, -3.875], [6.375, -4.25], [7.625, -3.5]])
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(), 2), points,
                       _quadratic_velocity(points), precision=r.Precision(global_digits=50),
                       patches=cover) as fit:
        ctx = fit.arithmetic.ctx
        x, y = queries[:, 0]-7, queries[:, 1]+4
        references = (_quadratic_velocity(queries),
                      np.column_stack((-4*x+6*y, 6*x+4*y)),
                      np.column_stack((6*x-3*y, 4*x-6*y)))
        for operator, reference in zip((None, r.Derivative(0), r.Derivative(1)), references):
            actual = fit.evaluate_mp(queries, operator)
            assert max(abs(actual[i, j]-ctx.mpf(float(reference[i, j])))
                       for i in range(len(queries)) for j in range(2)) < ctx.mpf('1e-37')


def test_disconnected_cover_and_singleton_have_independent_constant_gauges():
    grid = np.array(np.meshgrid([-.3, 0., .3], [-.3, 0., .3],
                               indexing='ij')).reshape(2, -1).T
    locations = np.array([[-3., 0.], [3., 0.], [0., 4.]])
    points = np.concatenate([grid+location for location in locations])
    centers = [[-3.15, 0], [-2.85, 0], [2.85, 0], [3.15, 0], [0, 4]]
    cover = r.PatchCover(centers, [.6]*5)
    data = np.column_stack((-2*points[:, 1], 2*points[:, 0]+3))
    queries = locations + [.125, -.125]
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(), 1), points, data,
                       precision=r.Precision(global_digits=50), patches=cover) as fit:
        expected = np.column_stack((-2*queries[:, 1], 2*queries[:, 0]+3))
        np.testing.assert_allclose(fit.evaluate(queries), expected, rtol=0, atol=1e-12)
        baseline = fit.evaluate_mp(queries)
        # Each disconnected component admits its own global additive constant.
        fit.shifts = [shift+offset for shift, offset in zip(fit.shifts, [7, 7, -11, -11, 13])]
        shifted = fit.evaluate_mp(queries)
        assert max(abs(shifted[i, j]-baseline[i, j])
                   for i in range(len(queries)) for j in range(2)) < fit.arithmetic.ctx.mpf('1e-38')


def test_refit_reuses_all_factors_and_failure_leaves_field_and_shifts_unchanged(monkeypatch):
    from rbflab.nodal import Arithmetic

    points, cover, data = _irregular_cloud()
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ()), points, data, patches=cover) as fit:
        queries = np.array([[.47, .43], [.37, .36]])
        factors = [id(patch['factor']) for patch in fit.patches]

        def no_factorization(*args, **kwargs):
            raise AssertionError('Refit must reuse local and glue factors')

        with monkeypatch.context() as patcher:
            patcher.setattr(Arithmetic, 'factor', no_factorization)
            fit.refit(2*data)
        assert [id(patch['factor']) for patch in fit.patches] == factors
        baseline = fit.evaluate(queries).copy()
        shifts = list(fit.shifts)
        diagnostics = dict(fit.diagnostics)

        def failed_solve(*args, **kwargs):
            raise np.linalg.LinAlgError('review injected later patch failure')

        for factor in (fit.patches[-1]['factor'], fit.glue_factor['factor']):
            with monkeypatch.context() as patcher:
                patcher.setattr(factor, 'solve', failed_solve)
                with pytest.raises(np.linalg.LinAlgError, match='review injected|Weighted glue'):
                    fit.refit(3*data)
            np.testing.assert_array_equal(fit.evaluate(queries), baseline)
            assert fit.shifts == shifts
            assert fit.diagnostics == diagnostics


def test_unresolvable_float64_glue_weight_is_an_explicit_precision_error():
    points = np.array(np.meshgrid([0., .5, 1.], [-.25, 0., .25],
                                 indexing='ij')).reshape(2, -1).T
    cover = r.PatchCover([[0, 0], [1e-6, 0], [1, 0], [1.000001, 0]], [1.2]*4)
    data = np.column_stack((-2*points[:, 1], 2*points[:, 0]+3))
    # Two tight groups create paper weights that underflow in Float64. The
    # fitter must diagnose this, rather than silently change the objective.
    with pytest.raises((ValueError, np.linalg.LinAlgError), match='(?i)glue|weight|precision|underflow'):
        r.interpolate(r.DivergenceFreeSpace(r.IMQ(), 1), points, data, patches=cover)


def test_sixty_digit_glue_preserves_weights_smaller_than_float64_resolution():
    points = np.array(np.meshgrid([0., .5, 1.], [-.25, 0., .25],
                                 indexing='ij')).reshape(2, -1).T
    cover = r.PatchCover([[0, 0], [.25, 0], [1, 0]], [1.2]*3)
    data = np.column_stack((-2*points[:, 1], 2*points[:, 0]+3))
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(), 1), points, data,
                       precision=r.Precision(global_digits=60), patches=cover) as fit:
        ctx = fit.arithmetic.ctx
        assert min(fit.weights) < ctx.mpf('1e-15')
        query = np.array([[.125, .125]])
        actual = fit.evaluate_mp(query)
        assert abs(actual[0, 0]+ctx.mpf('.25')) < ctx.mpf('1e-45')
        assert abs(actual[0, 1]-ctx.mpf('3.25')) < ctx.mpf('1e-45')


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='native worker opt-in')
def test_native_sixty_digit_refit_keeps_sub_float64_vector_perturbation():
    from fractions import Fraction

    points = np.array(np.meshgrid([-.5, 0., .5], [-.5, 0., .5],
                                 indexing='ij')).reshape(2, -1).T
    cover = r.PatchCover([[-.2, 0], [.2, 0]], [1., 1.])
    data = np.asarray([[Fraction(-2)*Fraction(float(y)),
                        Fraction(2)*Fraction(float(x))+3] for x, y in points], dtype=object)
    delta = Fraction(1, 10**40)
    changed = np.asarray([[u+delta, v-2*delta] for u, v in data], dtype=object)
    queries = np.array([[.125, .25], [-.25, -.125]])
    with r.interpolate(r.DivergenceFreeSpace(r.IMQ(), 1), points, data,
                       precision=r.Precision(global_digits=60), patches=cover,
                       backend=r.CppGlobalBackend(threads=1, compute_condition=False)) as fit:
        ctx = fit.arithmetic.ctx
        baseline = fit.evaluate_mp(queries)
        fit.refit(changed)
        result = fit.evaluate_mp(queries)
        for i in range(len(queries)):
            assert abs(result[i, 0]-baseline[i, 0]-ctx.mpf('1e-40')) < ctx.mpf('1e-52')
            assert abs(result[i, 1]-baseline[i, 1]+ctx.mpf('2e-40')) < ctx.mpf('1e-52')
