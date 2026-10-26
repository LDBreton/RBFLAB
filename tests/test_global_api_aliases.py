"""Additive aliases preserve canonical storage and numerical behavior."""
from dataclasses import fields, replace
from types import SimpleNamespace
import os
import numpy as np
import pytest
import sympy as sp
import rbflab as r


def scalar_problem():
    model = r.SymbolicScalar(2, transient=True)
    u = model.field
    x, y = model.coordinates
    t = model.time
    exact = sp.exp(-t) * (1 + x*x + y*y)
    return model.evolution(sp.Eq(sp.diff(u, t) - model.laplacian(u),
                                sp.diff(exact, t) - model.laplacian(exact)),
                           initial=exact.subs(t, 0),
                           boundary=[model.bc('boundary', sp.Eq(u, exact))])


def test_interpolation_space_alias_preserves_scalar_and_componentwise_data():
    centers = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .4]])
    space = r.ScalarSpace(r.IMQ(), 1)
    for data in [centers[:, 0] + 2*centers[:, 1], centers]:
        old = r.interpolate(space, centers, data)
        new = r.interpolate(space=space, centers=centers, data=data)
        np.testing.assert_array_equal(old.coefficients, new.coefficients)
        np.testing.assert_array_equal(old.evaluate([[.2, .3]]), new.evaluate([[.2, .3]]))
    with pytest.raises(ValueError, match='kernel or space'):
        r.interpolate(space, centers, centers[:, 0], space=space)
    with pytest.raises(TypeError, match='centers, data'):
        r.interpolate(space=space)


def test_interpolation_space_alias_preserves_solenoidal_vectors():
    centers = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .4]])
    space = r.DivergenceFreeSpace(r.IMQ(), 1)
    data = np.column_stack([centers[:, 1], -centers[:, 0]])
    old = r.interpolate(space, centers, data)
    new = r.interpolate(space=space, centers=centers, data=data)
    np.testing.assert_array_equal(old.evaluate([[.2, .3]]), new.evaluate([[.2, .3]]))
    np.testing.assert_allclose(sum(new.evaluate([[.2, .3]], r.Derivative(c))[:, c] for c in range(2)), 0, atol=1e-13)


def test_global_aliases_canonical_dataclass_storage_and_positional_compatibility():
    kernel = r.IMQ()
    precision = r.Precision()
    old = r.GlobalCollocation(kernel, precision, 'asymmetric', 1, None, None)
    new = r.GlobalCollocation(space=r.ScalarSpace(kernel, 1), precision=precision,
                             collocation_scheme='asymmetric')
    assert old == new
    assert [f.name for f in fields(new)] == ['kernel', 'precision', 'scheme', 'polynomial_degree', 'spaces', 'backend']
    copied = replace(new, scheme='symmetric', polynomial_degree=2)
    assert copied.scheme == 'symmetric' and copied.polynomial_degree == 2
    assert copied.kernel is kernel
    assert r.GlobalCollocation().scheme == 'symmetric'
    assert r.GlobalCollocation().precision is not r.GlobalCollocation().precision
    mapping = {'u': r.ScalarSpace(kernel, 1)}
    assert r.GlobalCollocation(spaces=mapping).spaces is mapping


def test_scalar_space_prepare_keeps_canonical_storage(monkeypatch):
    import rbflab.kernel_compiler as compiler
    method = r.GlobalCollocation(space=r.ScalarSpace(r.IMQ(), 2), collocation_scheme='asymmetric')
    original_kernel = method.kernel
    seen = []
    def prepare(copied, problem, dimension, cache_dir):
        seen.append(copied)
        assert copied is not method
        assert copied.scheme == 'asymmetric' and copied.polynomial_degree == 2
        return replace(copied, kernel=r.Gaussian())
    monkeypatch.setattr(compiler, 'prepare_method', prepare)
    prepared = method.prepare(scalar_problem(), dimension=2)
    assert len(seen) == 1 and method.kernel is original_kernel
    assert prepared.kernel != original_kernel
    assert prepared.scheme == 'asymmetric' and prepared.polynomial_degree == 2


@pytest.mark.parametrize('kwargs', [
    {'kernel': r.IMQ()}, {'spaces': {'u': r.ScalarSpace(r.IMQ())}},
    {'polynomial_degree': 1},
])
def test_global_space_conflicts(kwargs):
    with pytest.raises(ValueError, match='Specify'):
        r.GlobalCollocation(space=r.ScalarSpace(r.IMQ()), **kwargs)


@pytest.mark.parametrize('space', [r.DivergenceFreeSpace(r.IMQ()), r.PressureSpace(r.IMQ()), r.IMQ()])
def test_global_scalar_shorthand_rejects_field_spaces(space):
    with pytest.raises(TypeError, match='ScalarSpace'):
        r.GlobalCollocation(space=space)


def test_global_scheme_conflict():
    with pytest.raises(ValueError, match='scheme or collocation_scheme'):
        r.GlobalCollocation(r.IMQ(), scheme='symmetric', collocation_scheme='symmetric')


@pytest.mark.parametrize('collocation', ['symmetric', 'asymmetric'])
@pytest.mark.parametrize('time', ['bdf2', 'backward_euler'])
def test_scalar_evolution_alias_parity_and_default(collocation, time):
    problem = scalar_problem()
    cloud = r.geometry.unit_box_grid(3)
    old = r.GlobalCollocation(r.IMQ(), scheme=collocation, polynomial_degree=2).assemble(problem, cloud)
    new = r.GlobalCollocation(space=r.ScalarSpace(r.IMQ(), 2), collocation_scheme=collocation).assemble(problem, cloud)
    np.testing.assert_array_equal(old.matrix, new.matrix)
    np.testing.assert_array_equal(old.mass, new.mass)
    expected = old.solve('.01', 3, scheme=time)
    actual = new.solve('.01', 3, time_scheme=time)
    for a, b in zip(actual.states, expected.states):
        np.testing.assert_array_equal(a.unknowns, b.unknowns)
    if time == 'bdf2':
        np.testing.assert_array_equal(old.solve('.01', 3).final.unknowns, expected.final.unknowns)


def test_time_alias_conflicts_precede_numerical_work_on_all_systems():
    from rbflab.evolution import EvolutionSystem
    from rbflab.global_evolution_backend import CppScalarEvolutionSystem
    from rbflab.global_flow_backend import CppGlobalFlowSystem
    from rbflab.space_stokes import GlobalFlowSystem
    from rbflab.lhi_stokes import LHIStokesSystem
    from rbflab.unsteady_stokes import UnsteadyStokesSystem
    for cls in [EvolutionSystem, CppScalarEvolutionSystem, CppGlobalFlowSystem,
                GlobalFlowSystem, LHIStokesSystem, UnsteadyStokesSystem]:
        with pytest.raises(ValueError, match='scheme or time_scheme'):
            cls.solve(SimpleNamespace(), '.01', 1, scheme='bdf2', time_scheme='bdf2')


def test_python_stokes_time_alias_parity():
    from test_global_cpp_unsteady import problem_and_spaces
    problem, spaces = problem_and_spaces()
    method = r.GlobalCollocation(spaces=spaces)
    system = method.assemble(problem, r.geometry.unit_box_grid(2))
    for scheme in ['bdf2', 'backward_euler']:
        old = system.solve('.01', 3, scheme=scheme)
        new = system.solve('.01', 3, time_scheme=scheme)
        np.testing.assert_array_equal(old.final.coefficients, new.final.coefficients)


native = pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')


@native
@pytest.mark.parametrize('digits', [None, 60])
def test_native_scalar_alias_parity(tmp_path, digits):
    precision = r.Precision(global_digits=digits)
    backend = r.CppGlobalBackend(cache_dir=tmp_path, threads=1, compute_condition=False)
    centers = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.3, .4]])
    space = r.ScalarSpace(r.IMQ(), 1)
    data = centers[:, 0] + centers[:, 1]
    query = [[.2, .3], [.61, .27]]
    operators = [None, r.Derivative(0)]
    def evaluate(fit, op):
        value = fit.evaluate_mp(query, op) if digits else fit.evaluate(query, op)
        return value.tolist() if digits else value.copy()
    def equal(value, expected):
        if digits:
            # Compare native extended values without a binary64 conversion.
            assert value == expected
        else:
            np.testing.assert_array_equal(value, expected)
    with r.interpolate(space, centers, data, precision=precision, backend=backend) as old:
        expected = [evaluate(old, op) for op in operators]
    with r.interpolate(space=space, centers=centers, data=data, precision=precision, backend=backend) as new:
        for op, value in zip(operators, expected):
            equal(evaluate(new, op), value)
    problem = scalar_problem()
    cloud = r.geometry.unit_box_grid(3)
    def snapshot(trajectory):
        return [(state.unknowns.tolist() if digits else state.unknowns.copy(),
                 [evaluate(state, op) for op in operators]) for state in trajectory.states]
    with r.GlobalCollocation(r.IMQ(), precision, 'asymmetric', 2, backend=backend).assemble(problem, cloud) as old:
        expected = {scheme: snapshot(old.solve('.01', 3, scheme=scheme))
                    for scheme in ['bdf2', 'backward_euler']}
    with r.GlobalCollocation(space=r.ScalarSpace(r.IMQ(), 2), precision=precision,
                            collocation_scheme='asymmetric', backend=backend).assemble(problem, cloud) as new:
        info = r.execution_info(new)
        assert info['backend'] == 'cpp_global'
        assert info['arithmetic'] == ('mpfr' if digits else 'float64')
        assert info['digits'] == digits
        for scheme, states in expected.items():
            actual = snapshot(new.solve('.01', 3, time_scheme=scheme))
            for (coefficients, fields), (old_coefficients, old_fields) in zip(actual, states):
                equal(coefficients, old_coefficients)
                for value, expected_value in zip(fields, old_fields):
                    equal(value, expected_value)


@native
def test_native_stokes_time_alias_parity(tmp_path):
    from test_global_cpp_unsteady import problem_and_spaces
    problem, spaces = problem_and_spaces()
    backend = r.CppGlobalBackend(cache_dir=tmp_path, threads=1, compute_condition=False)
    with r.GlobalCollocation(spaces=spaces, backend=backend).assemble(problem, r.geometry.unit_box_grid(2)) as system:
        for scheme in ['bdf2', 'backward_euler']:
            old = system.solve('.01', 3, scheme=scheme).final.velocity([[.2, .3]])
            new = system.solve('.01', 3, time_scheme=scheme).final.velocity([[.2, .3]])
            np.testing.assert_array_equal(old, new)
