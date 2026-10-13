"""Precision changes and varied operators exercise native prepared evaluation."""
import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r

pytestmark = pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')


def test_literal_cache_follows_reused_worker_precision(tmp_path):
    s = sp.Symbol('s', nonnegative=True)
    # Non-dyadic rationals and a large integer exercise literal initialization.
    kernel = r.Kernel((sp.Rational(1, 3) + sp.Rational(2**40+1, 2**42+3))*sp.exp(-s), s)()
    problem = r.LinearPDE(r.Identity()+r.Laplacian(), 2., (r.Dirichlet(1.),))
    cloud = r.geometry.unit_box_grid(2)
    backend = r.CppGlobalBackend(threads=2, compute_condition=False, cache_dir=tmp_path)
    worker = None
    for digits in (35, 70, 35):
        common = dict(kernel=kernel, precision=r.Precision(global_digits=digits))
        expected = r.GlobalCollocation(**common).assemble(problem, cloud)
        with r.GlobalCollocation(**common, backend=backend).assemble(problem, cloud) as actual:
            if worker is not None:
                assert actual.worker._worker is worker
            worker = actual.worker._worker
            ctx = actual.ctx
            tolerance = ctx.mpf(10)**(-(digits-8))
            assert ctx.mnorm(actual.matrix.to_mpmath()-ctx.matrix(expected.matrix.tolist()), 'inf') < tolerance
            q = np.array([[.23, .41], [.62, .19]])
            a, b = actual.solve(), expected.solve()
            for op in (r.Identity(), r.Derivative(0), r.Laplacian()):
                assert ctx.norm(a.evaluate_mp(q, op)-ctx.matrix(b.evaluate_mp(q, op).tolist()), 'inf') < tolerance


@pytest.mark.parametrize('intervals', [2, 7])
def test_variable_operator_coefficients_and_plan_fallback(tmp_path, intervals):
    model = r.SymbolicScalar()
    u = model.field
    x, y = model.coordinates
    exact = 1+x*x+2*y*y
    lhs = (1+x+sp.Rational(1, 97)*y)*u - model.laplacian(u)/7
    problem = model.stationary(sp.Eq(lhs, lhs.subs(u, exact).doit()),
                              boundary=[model.bc('boundary', sp.Eq(u, exact))])
    cloud = r.geometry.unit_box_grid(intervals)
    common = dict(kernel=r.Gaussian(), scheme='asymmetric', polynomial_degree=2,
                  precision=r.Precision(global_digits=40))
    expected = r.GlobalCollocation(**common).assemble(problem, cloud)
    # The larger cloud has >32 distinct interior coefficients, selecting the
    # bounded-memory fallback. The smaller cloud uses prepared operator pairs.
    backend = r.CppGlobalBackend(threads=2, compute_condition=False, cache_dir=tmp_path)
    with r.GlobalCollocation(**common, backend=backend).assemble(problem, cloud) as actual:
        ctx = actual.ctx
        assert ctx.mnorm(actual.matrix.to_mpmath()-ctx.matrix(expected.matrix.tolist()), 'inf') < ctx.mpf('1e-32')
        q = np.array([[.17, .29], [.43, .61]])
        a, b = actual.solve(), expected.solve()
        assert ctx.norm(a.evaluate_mp(q)-ctx.matrix(b.evaluate_mp(q).tolist()), 'inf') < ctx.mpf('1e-28')


@pytest.mark.parametrize('digits', [None, 45])
def test_shared_compact_origin_near_and_support(tmp_path, digits):
    problem = r.LinearPDE(r.Identity(), 2., (r.Dirichlet(1.),))
    common = dict(kernel=r.Wendland(smoothness=4, radius=1),
                  precision=r.Precision(global_digits=digits))
    cloud = r.geometry.unit_box_grid(2)
    expected = r.GlobalCollocation(**common).assemble(problem, cloud).solve()
    backend = r.CppGlobalBackend(threads=2, compute_condition=False, cache_dir=tmp_path)
    with r.GlobalCollocation(**common, backend=backend).assemble(problem, cloud) as actual:
        # Includes zero displacement, the stable near-origin branch, exact
        # support boundary, exterior, and more than eight repeated targets.
        q = np.array([[0., 0.], [1e-12, 0.], [1., 0.], [2., 2.], [.23, .41]]*3)
        solution = actual.solve()
        for op in (r.Identity(), r.Derivative(0), r.Laplacian()):
            if digits:
                ctx = actual.ctx
                assert ctx.norm(solution.evaluate_mp(q, op)-ctx.matrix(expected.evaluate_mp(q, op).tolist()), 'inf') < ctx.mpf('1e-36')
            else:
                np.testing.assert_allclose(solution.evaluate(q, op), expected.evaluate(q, op), atol=1e-10, rtol=1e-10)
