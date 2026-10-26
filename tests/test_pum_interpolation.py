import numpy as np
import pytest

import rbflab as r


def setup(dimension=2):
    grid=np.array(np.meshgrid(*([np.linspace(-1,1,4)]*dimension), indexing='ij')).reshape(dimension,-1).T
    centers=np.array([[-0.45]+[0]*(dimension-1),[0.45]+[0]*(dimension-1)])
    return grid,r.PatchCover(centers,[1.65,1.65])


@pytest.mark.parametrize('dimension',[2,3])
def test_nodal_componentwise_and_refit(dimension):
    x,cover=setup(dimension)
    y=np.column_stack((np.sin(x[:,0])+x[:,1]**2,np.cos(x[:,0])-x[:,1]))
    fit=r.interpolate(r.Gaussian(1),x,y,patches=cover)
    np.testing.assert_allclose(fit.evaluate(x),y,atol=2e-9)
    factors=[id(p['factor']) for p in fit.patches]
    fit.refit(2*y)
    assert [id(p['factor']) for p in fit.patches]==factors
    np.testing.assert_allclose(fit.evaluate(x),2*y,atol=4e-9)
    with pytest.raises(ValueError):fit.refit(np.ones((len(x),3)))
    np.testing.assert_allclose(fit.evaluate(x),2*y,atol=4e-9)


def test_one_patch_matches_global_and_empty():
    x,_=setup()
    cover=r.PatchCover([[0,0]],[10])
    y=np.sin(x[:,0]*x[:,1])+x[:,0]
    fit=r.interpolate(r.Gaussian(1),x,y,patches=cover)
    global_fit=r.interpolate(r.Gaussian(1),x,y)
    q=np.array([[0.14,-0.23],[0.77,0.33]])
    for op in (None,r.Derivative(0),r.Laplacian()):
        np.testing.assert_allclose(fit.evaluate(q,op),global_fit.evaluate(q,op),atol=1e-7)
    assert fit.evaluate(np.empty((0,2))).shape==(0,)


def test_overlap_derivative_finite_difference_and_validation():
    x,cover=setup()
    y=np.sin(2*x[:,0]+x[:,1])+x[:,0]*x[:,1]
    fit=r.interpolate(r.Gaussian(1),x,y,patches=cover)
    q=np.array([[0.11,0.09],[-0.21,0.13]])
    h=1e-5
    delta=np.array([h,0.])
    fd=(fit.evaluate(q+delta)-fit.evaluate(q-delta))/(2*h)
    np.testing.assert_allclose(fit.evaluate(q,r.Derivative(0)),fd,rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError,match='cover'):
        fit.evaluate(np.array([[20.,20.]]))
    with pytest.raises(NotImplementedError):
        fit.evaluate(q,r.Derivative(0)@r.Derivative(0)@r.Derivative(0))
    with pytest.raises(NotImplementedError, match='planar 2D'):
        r.interpolate(r.DivergenceFreeSpace(r.Gaussian(1)),
                      np.column_stack((x,np.zeros(len(x)))),
                      np.ones((len(x),3)),
                      patches=r.PatchCover(np.column_stack((cover.centers,np.zeros(2))),cover.radii))


def test_native_reference_double_parity_and_reuse():
    pytest.importorskip('rbflab.global_backend')
    import os
    if os.environ.get('RBFLAB_CPP_TESTS')!='1':pytest.skip('native worker opt-in')
    x,cover=setup()
    y=np.column_stack((np.sin(x[:,0])+x[:,1]**2,np.cos(x[:,1])))
    precision=r.Precision()
    ref=r.interpolate(r.Gaussian(1),x,y,precision=precision,patches=cover)
    with r.interpolate(r.Gaussian(1),x,y,precision=precision,patches=cover,
                       backend=r.CppGlobalBackend(threads=1)) as native:
        q=np.array([[0.08,0.11],[-0.24,0.18]])
        for op in (None,r.Derivative(0),r.Laplacian()):
            np.testing.assert_allclose(native.evaluate(q,op),ref.evaluate(q,op),rtol=1e-7,atol=1e-7)
        native.refit(2*y)
        ref.refit(2*y)
        np.testing.assert_allclose(native.evaluate(q),ref.evaluate(q),rtol=1e-7,atol=1e-7)
        assert native.diagnostics['factorizations']==len(cover.radii)


@pytest.mark.parametrize('dimension', [2, 3])
def test_mp_derivatives_against_independent_blended_value(dimension):
    """Different local fits make missing weight derivatives observable."""
    from rbflab.operators import Operator

    points = np.array([[0, 0], [.2, .1], [0, .5], [.3, .7], [.6, .2],
                       [.8, 0], [.7, .8], [1, .5], [1, 1]], dtype=float)
    centers = np.array([[.2, .3], [.8, .6]])
    query = np.array([[.49, .37]])
    if dimension == 3:
        points = np.column_stack((points, [0, .15, .3, .1, .6, .2, .5, .65, .4]))
        centers = np.column_stack((centers, [.2, .4]))
        query = np.column_stack((query, [.28]))
    cover = r.PatchCover(centers, [.85, .9])
    data = np.sin(2 * points[:, 0]) + np.cos(3 * points[:, 1])
    if dimension == 3:
        data += np.sin(points[:, 2] + points[:, 0])
    with r.interpolate(r.IMQ(), points, data, r.Precision(global_digits=60),
                       patches=cover) as fit:
        ctx = fit.arithmetic.ctx
        position = tuple(ctx.mpf(float(v)) for v in query[0])

        def blended_value(*coordinates):
            # Deliberately evaluate only the scalar formula; no production
            # weight jets, derivative operators or local evaluation helpers.
            weights, local_values = [], []
            for patch, center, radius in zip(fit.patches, cover.centers, cover.radii):
                offset = [x - ctx.mpf(float(c)) for x, c in zip(coordinates, center)]
                t = ctx.sqrt(ctx.fsum(x * x for x in offset)) / ctx.mpf(float(radius))
                weights.append((1 - t)**4 * (4 * t + 1) if t < 1 else ctx.mpf(0))
                terms = []
                for i, node in enumerate(patch['basis'].centers):
                    square = ctx.fsum((x - ctx.mpf(float(v)))**2
                                      for x, v in zip(coordinates, node))
                    terms.append(patch['coefficients'][i, 0] / ctx.sqrt(1 + square))
                local_values.append(ctx.fsum(terms))
            return ctx.fsum(w * v for w, v in zip(weights, local_values)) / ctx.fsum(weights)

        alphas = [tuple([0] * dimension), tuple([1] + [0] * (dimension - 1)),
                  tuple([2] + [0] * (dimension - 1)),
                  tuple([1, 1] + [0] * (dimension - 2))]
        if dimension == 3:
            alphas += [(0, 0, 2), (1, 0, 1), (0, 1, 1)]
        for alpha in alphas:
            expected = ctx.diff(blended_value, position, alpha)
            actual = fit.evaluate_mp(query, Operator(((alpha, 1),), dimension))[0]
            assert abs(actual - expected) < ctx.mpf('1e-50'), alpha


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('linear_solver', ['eigen', 'mplapack'])
def test_native_mp_parity_preserves_digits_and_mixed_derivatives(dimension, linear_solver, monkeypatch):
    import os
    import json
    if os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('native worker opt-in')
    prefix = os.environ.get('RBFLAB_MPLAPACK_DIR')
    if linear_solver == 'mplapack' and not prefix:
        pytest.skip('Requires optional MPLAPACK prefix')

    points = np.array([[0, 0, 0], [.2, .1, .15], [0, .5, .3], [.3, .7, .1],
                       [.6, .2, .6], [.8, 0, .2], [.7, .8, .5], [1, .5, .65],
                       [1, 1, .4]], dtype=float)[:, :dimension]
    centers = np.array([[.2, .3, .2], [.8, .6, .4]])[:, :dimension]
    cover = r.PatchCover(centers, [.85, .9])
    data = np.column_stack((np.sin(2 * points[:, 0]) + np.cos(3 * points[:, 1]),
                            np.cos(points[:, 0]) - points[:, 1]**2))
    queries = np.array([[.49, .37, .28], [.2, .3, .2], [1.05, .3, .2]])[:, :dimension]
    precision = r.Precision(global_digits=60)
    backend = r.CppGlobalBackend(threads=1, compute_condition=False, linear_solver=linear_solver,
                                 mplapack_dir=prefix if linear_solver == 'mplapack' else None)
    operators = [None, r.Derivative(0, dimension), r.Laplacian(dimension),
                 r.Derivative(0, dimension) @ r.Derivative(0, dimension),
                 r.Derivative(0, dimension) @ r.Derivative(1, dimension)]
    if dimension == 3:
        operators.append(r.Derivative(0, dimension) @ r.Derivative(2, dimension))
    with r.interpolate(r.IMQ(), points, data, precision, patches=cover) as reference:
        with r.interpolate(r.IMQ(), points, data, precision, patches=cover,
                           backend=backend) as native:
            ctx = reference.arithmetic.ctx

            def assert_mp_close(actual, expected):
                assert actual.rows == expected.rows and actual.cols == expected.cols
                error = max(abs(ctx.mpf(actual[i, j]) - expected[i, j])
                            for i in range(actual.rows) for j in range(actual.cols))
                assert error < ctx.mpf('1e-47'), (dimension, linear_solver, ctx.nstr(error))

            for values in (data, 2 * data + .125):
                if values is not data:
                    assert native.refit(values) is native
                    assert reference.refit(values) is reference
                nodal = ctx.matrix([[ctx.mpf(float(v)) for v in row] for row in values])
                assert_mp_close(native.evaluate_mp(points), nodal)
                assert_mp_close(reference.evaluate_mp(points), nodal)
                for operator in operators:
                    expected = reference.evaluate_mp(queries, operator)
                    actual = native.evaluate_mp(queries, operator)
                    assert_mp_close(actual, expected)
                assert actual.rows == len(queries) and actual.cols == 2
                assert native.diagnostics['factorizations'] == len(cover.radii)

            def forbidden_request(*args, **kwargs):
                raise AssertionError('execution_info must not request native work')

            with monkeypatch.context() as patcher:
                patcher.setattr(native.worker, 'request', forbidden_request)
                report = r.execution_info(native)
                assert report['supported'] and report['evidence'] == 'implementation'
                assert report['backend'] == 'cpp_pum'
                assert report['arithmetic'] == 'mpfr' and report['digits'] == 60
                assert report['linear_solver'] == linear_solver
                assert report['locations']['evaluation'] == 'cpp'
                assert report['locations']['data'] == 'python'
                assert report['patches'] == len(cover.radii)
                json.dumps(report)


@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('backend_name', ['python', 'native'])
def test_overlap_quadratic_reproduction_in_extended_arithmetic(dimension, backend_name):
    import os
    if backend_name == 'native' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('native worker opt-in')

    points = np.array(np.meshgrid(*([[-1., 0., 1.]] * dimension), indexing='ij')).reshape(dimension, -1).T
    centers = np.zeros((2, dimension))
    centers[:, 0] = [-.45, .45]
    cover = r.PatchCover(centers, [1.6 if dimension == 2 else 1.85] * 2)
    memberships = cover.memberships(points)
    assert set(memberships[0]) != set(memberships[1])

    def polynomial(*coordinates):
        x, y = coordinates[:2]
        value = 1 + 2*x - 3*y + x*x + 2*x*y + 3*y*y
        if len(coordinates) == 3:
            z = coordinates[2]
            value += 4*z - 2*x*z + y*z + 5*z*z
        return value

    data = r.PrecisionData(lambda q: polynomial(*q.T),
                           lambda ctx, *coordinates: polynomial(*coordinates))
    backend = r.CppGlobalBackend(threads=1, compute_condition=False) if backend_name == 'native' else None
    queries = np.array([[-.2, .13, .1], [.24, -.17, -.09], [-.45, 0, 0]])[:, :dimension]
    space = r.ScalarSpace(r.PHS(5), polynomial_degree=2)
    with r.interpolate(space, points, data, r.Precision(global_digits=60),
                       patches=cover, backend=backend) as fit:
        ctx = fit.arithmetic.ctx
        expected = [polynomial(*(ctx.mpf(float(v)) for v in row)) for row in queries]
        actual = fit.evaluate_mp(queries)
        assert max(abs(actual[i] - expected[i]) for i in range(len(queries))) < ctx.mpf('1e-50')
        mixed = fit.evaluate_mp(queries, r.Derivative(0, dimension) @ r.Derivative(1, dimension))
        assert max(abs(value - 2) for value in mixed) < ctx.mpf('1e-50')
        laplacian = fit.evaluate_mp(queries, r.Laplacian(dimension))
        assert max(abs(value - (8 if dimension == 2 else 18)) for value in laplacian) < ctx.mpf('1e-50')
        if dimension == 3:
            mixed_xz = fit.evaluate_mp(queries, r.Derivative(0, 3) @ r.Derivative(2, 3))
            assert max(abs(value + 2) for value in mixed_xz) < ctx.mpf('1e-50')
