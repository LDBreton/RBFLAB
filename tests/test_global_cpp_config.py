"""Native backend validation without requiring the compiler."""
import pytest
import rbflab as r


def test_backend_configuration():
    for kwargs in ({'threads':0},{'timeout':float('nan')},{'derivative_order':-1}):
        with pytest.raises((ValueError,TypeError)):r.CppGlobalBackend(**kwargs)
    with pytest.raises(NotImplementedError,match='Ubuntu'):
        r.CppGlobalBackend(distribution='Other')


def test_scalar_only_preflight():
    method=r.GlobalCollocation(r.IMQ(),backend=r.CppGlobalBackend())
    problem=r.LinearPDE(r.Identity(),1.0,(r.Dirichlet(1.0),))
    method.preflight(problem)
    with pytest.raises(NotImplementedError,match='scalar'):
        method.preflight(object())
    with pytest.raises(TypeError,match='CppGlobalBackend'):
        r.GlobalCollocation(r.IMQ(),backend=object()).preflight(problem)


@pytest.mark.parametrize('scheme', ['symmetric', 'asymmetric'])
def test_extended_callback_requires_precision_data_before_compilation(scheme, monkeypatch):
    import rbflab.global_backend as backend_module

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid data must be rejected before compiling')

    monkeypatch.setattr(backend_module, '_compiled', forbidden)
    problem = r.LinearPDE(r.Identity(), lambda points: 1, (r.Dirichlet(1),))
    method = r.GlobalCollocation(
        r.IMQ(), scheme=scheme, backend=r.CppGlobalBackend(),
        precision=r.Precision(global_digits=40),
    )
    with pytest.raises(TypeError, match='PrecisionData'):
        method.assemble(problem, r.geometry.unit_box_grid(2))


def test_transient_and_coupled_problems_reject_native_backend():
    import sympy as sp

    model = r.SymbolicScalar(transient=True)
    u = model.field
    transient = model.evolution(
        sp.Eq(sp.diff(u, model.time) - model.laplacian(u), 0),
        initial=0, boundary=[model.bc('boundary', sp.Eq(u, 0))],
    )
    with pytest.raises(NotImplementedError, match='stationary scalar'):
        r.GlobalCollocation(r.IMQ(), backend=r.CppGlobalBackend()).preflight(transient)

    block = r.SymbolicSystem(fields=('a', 'b'))
    a, b = block.fields
    problem = block.stationary(
        [sp.Eq(a + b, 0), sp.Eq(a - b, 0)],
        boundary=[block.bc('boundary', [sp.Eq(a, 0), sp.Eq(b, 0)])],
    )
    method = r.GlobalCollocation(
        spaces={a: r.ScalarSpace(r.IMQ()), b: r.ScalarSpace(r.IMQ())},
        backend=r.CppGlobalBackend(),
    )
    with pytest.raises(NotImplementedError, match='stationary scalar'):
        method.preflight(problem)
