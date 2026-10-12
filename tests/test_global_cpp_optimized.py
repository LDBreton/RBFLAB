"""Numerical regression coverage for optimized double kernel generation."""
import os

import numpy as np
import pytest
import rbflab as r


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')
@pytest.mark.parametrize('scheme', ['symmetric', 'asymmetric'])
@pytest.mark.parametrize('dimension', [2, 3])
def test_nonpolynomial_double_matrix_and_derivatives(scheme, dimension):
    cloud = r.geometry.unit_box_grid(2, dimension)
    identity = r.Identity(dimension)
    laplacian = r.Laplacian(dimension)
    problem = r.LinearPDE(identity - laplacian, 2., (r.Dirichlet(1.),))
    common = dict(kernel=r.IMQ(.7), scheme=scheme)
    python = r.GlobalCollocation(**common).assemble(problem, cloud)
    backend = r.CppGlobalBackend(cache_dir='outputs/global_optimized_test_cache', threads=2)
    with r.GlobalCollocation(**common, backend=backend).assemble(problem, cloud) as native:
        np.testing.assert_allclose(native.matrix.to_numpy(), python.matrix, rtol=2e-13, atol=2e-13)
        points = np.random.default_rng(417).uniform(.1, .9, (13, dimension))
        actual, expected = native.solve(), python.solve()
        for op in (identity, r.Derivative(0, dimension=dimension), laplacian):
            np.testing.assert_allclose(actual.evaluate(points, op), expected.evaluate(points, op),
                                       rtol=2e-8, atol=2e-8)
