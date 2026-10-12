"""Opt-in native scalar global collocation parity and precision tests."""
import os
import numpy as np
import pytest
import rbflab as r

pytestmark=pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS')!='1',reason='Opt-in native integration')


@pytest.fixture(scope='module')
def cache_dir(tmp_path_factory):return tmp_path_factory.mktemp('global_cpp_cache')


@pytest.mark.parametrize('scheme',['symmetric','asymmetric'])
@pytest.mark.parametrize('digits',[None,40])
def test_native_matches_python(scheme,digits,cache_dir):
    cloud=r.geometry.unit_box_grid(2,2)
    problem=r.LinearPDE(r.Identity()+r.Laplacian(),2.0,(r.Dirichlet(1.0),))
    common=dict(kernel=r.IMQ(),scheme=scheme,precision=r.Precision(global_digits=digits))
    native=r.GlobalCollocation(**common,backend=r.CppGlobalBackend(cache_dir=cache_dir)).assemble(problem,cloud)
    try:
        python=r.GlobalCollocation(**common).assemble(problem,cloud)
        x=np.array([[.23,.41],[.62,.19]])
        assert np.allclose(native.solve().evaluate(x),python.solve().evaluate(x),atol=1e-9)
        reference=np.asarray(python.matrix.tolist(),dtype=float) if digits else np.asarray(python.matrix)
        assert np.allclose(np.asarray(native.matrix),reference,atol=1e-9)
        assert np.allclose(native.solve().evaluate(x,r.Derivative(0)),python.solve().evaluate(x,r.Derivative(0)),atol=1e-8)
        if digits:
            matrix = native.matrix.to_mpmath()
            ctx = matrix.ctx
            # mpmath matrices from private contexts need explicit context conversion.
            assert ctx.mnorm(matrix - ctx.matrix(python.matrix.tolist()), 'inf') < ctx.mpf('1e-34')
            for operator in (r.Identity(), r.Derivative(0), r.Laplacian()):
                actual = native.solve().evaluate_mp(x, operator)
                expected = python.solve().evaluate_mp(x, operator)
                assert ctx.norm(actual - ctx.matrix(expected.tolist()), 'inf') < ctx.mpf('1e-30')
    finally:native.close()


def test_rhs_and_matrix_edit(cache_dir):
    cloud=r.geometry.unit_box_grid(2,2)
    problem=r.LinearPDE(r.Identity(),1.0,(r.Dirichlet(1.0),))
    with r.GlobalCollocation(r.IMQ(),backend=r.CppGlobalBackend(cache_dir=cache_dir)).assemble(problem,cloud) as system:
        a=system.solve().evaluate(np.array([[.5,.5]]))[0]
        system.rhs[:]=[2.0]*len(system.rhs)
        b=system.solve().evaluate(np.array([[.5,.5]]))[0]
        assert np.isclose(b,2*a)
        old=system.matrix[0,0]
        system.matrix[0,0]=old+0.1
        assert np.isclose(system.matrix[0,0],old+0.1)
        assert not np.isclose(system.solve().evaluate(np.array([[.5,.5]]))[0],b)


def test_prepare_keeps_native_backend(cache_dir):
    cloud=r.geometry.unit_box_grid(2,2)
    problem=r.LinearPDE(r.Identity(),1.0,(r.Dirichlet(1.0),))
    method=r.GlobalCollocation(r.IMQ(),backend=r.CppGlobalBackend(cache_dir=cache_dir))
    prepared=method.prepare(problem,dimension=2)
    assert prepared is not method and prepared.backend==method.backend
    with prepared.assemble(problem,cloud) as system:
        assert np.allclose(system.solve().evaluate(np.array([[.5,.5]])),1.0,atol=1e-9)
    space_method=r.GlobalCollocation(spaces={'u':r.ScalarSpace(r.IMQ())},backend=method.backend)
    prepared_space=space_method.prepare(problem,dimension=2)
    assert prepared_space.kernel is not None and prepared_space.spaces is None
    with prepared_space.assemble(problem,cloud) as system:
        assert np.allclose(system.solve().evaluate(np.array([[.5,.5]])),1.0,atol=1e-9)


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('scheme',['symmetric','asymmetric'])
@pytest.mark.parametrize('digits',[None,40])
def test_polynomial_mixed_boundaries(dimension,scheme,digits,cache_dir):
    from test_symbolic import model_problem
    model,problem,exact=model_problem(dimension)
    cloud=r.geometry.unit_box_grid(2,dimension)
    backend=r.CppGlobalBackend(cache_dir=cache_dir)
    method=r.GlobalCollocation(r.PHS(5),scheme=scheme,polynomial_degree=2,
        precision=r.Precision(global_digits=digits),backend=backend)
    with method.assemble(problem,cloud) as system:
        solution=system.solve()
        q=np.full((2,dimension),.25)
        expected=model.data(exact)
        assert np.allclose(solution.evaluate(q),expected(q),atol=2e-9)
        if digits:
            actual=solution.evaluate_mp(q)
            assert max(abs(a-b) for a,b in zip(actual,expected.mp_values(actual.ctx,q)))<actual.ctx.mpf('1e-30')


