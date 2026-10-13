"""Native divergence-free global field parity, precision and lifecycle."""
import os

import numpy as np
import pytest
import sympy as sp
import rbflab as r


@pytest.mark.parametrize('digits', [None, 40])
def test_opt_in_eval_timers_preserve_fields(digits, tmp_path):
    if os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native integration')
    _, problem, spaces, _, _ = field_problem()
    method = r.GlobalCollocation(spaces=spaces,
        precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(threads=2, compute_condition=False, cache_dir=tmp_path))
    q = np.array([[.23, .41], [.62, .19]])
    with method.assemble(problem, r.geometry.unit_box_grid(2)) as system:
        solution = system.solve()
        def values():
            return [getattr(solution, name)(q, extended=bool(digits)).tolist()
                    for name in ('velocity', 'pressure', 'pressure_gradient')]
        reference = values()
        with pytest.raises(RuntimeError, match='No profiled EVAL'):
            system.worker.request('EVAL_TIMING')
        assert system.worker.request('EVAL_PROFILE 1') == []
        assert values() == reference
        timings = list(map(float, system.worker.request('EVAL_TIMING')))
        assert len(timings) == 3 and all(np.isfinite(t) and t >= 0 for t in timings)
        assert timings[1] > 0
        system.worker.request('EVAL_PROFILE 0')
        assert values() == reference
        with pytest.raises(RuntimeError, match='No profiled EVAL'):
            system.worker.request('EVAL_TIMING')


def field_problem(dimension=2, augmented=True, transient=False):
    model = r.SymbolicStokes(dimension, transient=transient)
    u, p = model.fields
    x = model.coordinates
    velocity = sp.ImmutableMatrix([x[(j+1) % dimension]**2 for j in range(dimension)])
    pressure = sum((j+1)*v for j, v in enumerate(x))
    lhs = -sp.Rational(7, 10)*model.laplacian(u)+model.gradient(p)
    if transient:
        lhs += sp.diff(u, model.time)
    forcing = lhs.subs(dict(zip(u, velocity)) | {p: pressure}).doit()
    boundary = [model.bc('boundary', sp.Eq(u, velocity))]
    problem = (model.evolution([sp.Eq(lhs, forcing)], initial={u: velocity}, boundary=boundary)
               if transient else model.stationary([sp.Eq(lhs, forcing)], boundary=boundary))
    spaces = {u: r.DivergenceFreeSpace(r.IMQ('.7'), 2 if augmented else None),
              p: r.PressureSpace(r.Gaussian('1.3'), 1 if augmented else None)}
    return model, problem, spaces, velocity, pressure


def test_field_preflight_and_unsupported_combinations():
    model, problem, spaces, _, _ = field_problem()
    method = r.GlobalCollocation(spaces=spaces, backend=r.CppGlobalBackend())
    method.preflight(problem)
    with pytest.raises(NotImplementedError, match='symmetric'):
        r.GlobalCollocation(spaces=spaces, scheme='asymmetric', backend=method.backend).preflight(problem)
    _, transient, tspaces, _, _ = field_problem(transient=True)
    r.GlobalCollocation(spaces=tspaces, backend=method.backend).preflight(transient)
    with pytest.raises(NotImplementedError, match='derivative_order'):
        r.GlobalCollocation(spaces=spaces, backend=r.CppGlobalBackend(derivative_order=3)).preflight(problem)
    with pytest.raises(ValueError, match='spaces'):
        r.GlobalCollocation(r.IMQ(), backend=method.backend).preflight(problem)
    bad = dict(spaces)
    bad[model.velocity] = r.ScalarSpace(r.IMQ())
    with pytest.raises(TypeError, match='DivergenceFreeSpace'):
        r.GlobalCollocation(spaces=bad, backend=method.backend).preflight(problem)


def test_field_callback_and_geometry_validation_before_compilation(monkeypatch):
    from dataclasses import replace
    import rbflab.global_flow_backend as flow
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid input reached compilation')
    monkeypatch.setattr(flow, '_artifact', forbidden)
    _, problem, spaces, _, _ = field_problem()
    invalid = replace(problem, forcing=(lambda q: np.ones(len(q)), 1))
    method = r.GlobalCollocation(spaces=spaces, precision=r.Precision(global_digits=40),
                                backend=r.CppGlobalBackend())
    with pytest.raises(TypeError, match='PrecisionData'):
        method.assemble(invalid, r.geometry.unit_box_grid(2))
    with pytest.raises(ValueError, match='dimensions'):
        method.assemble(problem, r.geometry.unit_box_grid(2, 3))
    with pytest.raises(ValueError, match='Interior'):
        method.assemble(problem, r.geometry.PointCloud([[0, 0], [1, 0], [0, 1]],
                                                      {'boundary': [0, 1, 2]}, {}))


native = pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')


@pytest.fixture(scope='module')
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('global_fields')


@native
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('digits', [None, 40])
def test_augmented_fields_native_matches_python(dimension, digits, cache_dir):
    model, problem, spaces, velocity, pressure = field_problem(dimension)
    cloud = r.geometry.unit_box_grid(2, dimension)
    method = r.GlobalCollocation(spaces=spaces, precision=r.Precision(global_digits=digits))
    expected = method.assemble(problem, cloud)
    method.backend = r.CppGlobalBackend(cache_dir=cache_dir, threads=2, compute_condition=False)
    prepared = method.prepare(problem, dimension=dimension)
    assert prepared is not method and prepared.spaces == method.spaces
    with prepared.assemble(problem, cloud) as actual:
        q = np.random.default_rng(73).uniform(.1, .9, (3, dimension))
        # Repeat/shuffle targets across the eight-row sharing bound, and include
        # an exact node and a nearby point to exercise origin handling.
        q = np.vstack([q, np.tile(q[0], (10, 1)), np.zeros((1, dimension)),
                       np.full((1, dimension), 1e-12)])[::-1].copy()
        assert 'polynomial_columns' in actual.dof_map
        assert actual.reconstruction['divergence'] == 'analytic'
        a, b = actual.solve(), expected.solve()
        np.testing.assert_allclose(actual.matrix.to_numpy(), np.asarray(expected.matrix.tolist(), dtype=float),
                                   rtol=3e-12, atol=3e-12)
        if digits:
            ctx = actual.ctx
            assert ctx.mnorm(actual.matrix.to_mpmath()-ctx.matrix(expected.matrix.tolist()), 'inf') < ctx.mpf('1e-32')
        for operation in ('velocity', 'pressure_gradient', 'divergence', 'residuals'):
            av, bv = getattr(a, operation)(q, extended=bool(digits)), getattr(b, operation)(q, extended=bool(digits))
            if digits:
                assert ctx.norm(av-ctx.matrix(bv.tolist()), 'inf') < ctx.mpf('1e-28')
            else:
                np.testing.assert_allclose(av, bv, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(a.velocity(q), np.column_stack([model.data(v)(q) for v in velocity]), atol=2e-9)
        np.testing.assert_allclose(a.pressure_gradient(q), np.tile(np.arange(1, dimension+1), (len(q), 1)), atol=2e-9)
        assert np.max(np.abs(a.divergence(q))) < 2e-10
        gauged = a.pressure(q, reference=(np.zeros(dimension), '0'), extended=bool(digits))
        truth = model.data(pressure)
        if digits:
            assert ctx.norm(gauged-ctx.matrix(truth.mp_values(ctx, q)), 'inf') < ctx.mpf('1e-28')
        else:
            np.testing.assert_allclose(gauged, truth(q), atol=2e-9)
        assert a.velocity(np.empty((0, dimension))).shape == (0, dimension)
        if digits:
            assert a.velocity(np.empty((0, dimension)), extended=True).rows == 0
        with pytest.raises(ValueError, match='time-step'):
            actual.solve('.1', 2)


@native
@pytest.mark.parametrize('digits', [None, 60])
def test_unaugmented_fields_native_ownership_and_mp_changes(digits, cache_dir, monkeypatch):
    _, problem, spaces, _, _ = field_problem(augmented=False)
    cloud = r.geometry.unit_box_grid(2)
    precision = r.Precision(global_digits=digits)
    reference = r.GlobalCollocation(spaces=spaces, precision=precision).assemble(problem, cloud)
    expected = reference.solve()
    q = [[.27, .43], [.71, .29]]
    target = expected.velocity(q, extended=bool(digits))
    target_gradient = expected.pressure_gradient(q, extended=bool(digits))
    from rbflab.nodal import Arithmetic
    import rbflab.space_stokes as flow
    def forbidden(*args, **kwargs):
        raise AssertionError('Native field path called Python dense assembly/factor/evaluation')
    method = r.GlobalCollocation(spaces=spaces, precision=precision,
        backend=r.CppGlobalBackend(cache_dir=cache_dir, compute_condition=True))
    monkeypatch.setattr(Arithmetic, 'matrix', forbidden)
    monkeypatch.setattr(Arithmetic, 'factor', forbidden)
    monkeypatch.setattr(flow, '_blocks', forbidden)
    with method.assemble(problem, cloud) as system:
        first = system.solve()
        values = first.velocity(q, extended=bool(digits))
        gradient = first.pressure_gradient(q, extended=bool(digits))
        divergence = first.divergence(q, extended=bool(digits))
        if digits:
            ctx = system.ctx
            assert ctx.norm(values-ctx.matrix(target.tolist()), 'inf') < ctx.mpf('1e-48')
            assert ctx.norm(gradient-ctx.matrix(target_gradient.tolist()), 'inf') < ctx.mpf('1e-48')
            assert ctx.norm(divergence, 'inf') < ctx.mpf('1e-48')
            multiplier = 1+ctx.mpf('1e-40')
        else:
            np.testing.assert_allclose(values, target, atol=1e-9)
            np.testing.assert_allclose(gradient, target_gradient, atol=1e-9)
            assert np.max(np.abs(divergence)) < 1e-10
            multiplier = 2
        assert first.diagnostics['scaled_condition'] > 1
        system.rhs = [v*multiplier for v in system.rhs]
        second = system.solve().velocity(q, extended=bool(digits))
        if digits:
            assert second != values
            assert ctx.norm(second-values*multiplier, 'inf') < ctx.mpf('1e-50')
        else:
            np.testing.assert_allclose(second, values*multiplier, atol=1e-10)
        saved = system.matrix.to_mpmath() if digits else system.matrix.to_numpy()
        system.matrix = saved*2
        third = system.solve().velocity(q, extended=bool(digits))
        if digits:
            assert ctx.norm(third-second/2, 'inf') < ctx.mpf('1e-50')
        else:
            np.testing.assert_allclose(third, second/2, atol=1e-10)
        np.testing.assert_allclose(first.velocity(q), np.asarray(values.tolist() if digits else values, float), atol=1e-12)
    with method.assemble(problem, cloud) as reused:
        with pytest.raises(RuntimeError, match='closed'):
            first.velocity(q)
        assert reused.worker._worker is system.worker._worker


@native
@pytest.mark.parametrize('digits', [None, 40])
def test_phs_velocity_and_scalar_field_worker_isolation(cache_dir, digits):
    model, problem, spaces, velocity, _ = field_problem()
    spaces[model.velocity] = r.DivergenceFreeSpace(r.PHS(7), 2)
    spaces[model.pressure] = r.PressureSpace(r.PHS(3), 1)
    backend = r.CppGlobalBackend(cache_dir=cache_dir, compute_condition=False)
    method = r.GlobalCollocation(spaces=spaces, backend=backend, precision=r.Precision(global_digits=digits))
    cloud = r.geometry.unit_box_grid(2)
    scalar_problem = r.LinearPDE(r.Identity(), 1., (r.Dirichlet(1.),))
    with method.assemble(problem, cloud) as system, \
         r.GlobalCollocation(r.IMQ(), backend=backend).assemble(scalar_problem, cloud) as scalar:
        assert system.worker._worker is not scalar.worker._worker
        q = np.array([[.31, .57], [.72, .21]])
        np.testing.assert_allclose(system.solve().velocity(q),
            np.column_stack([model.data(v)(q) for v in velocity]), atol=2e-10)
        assert np.max(np.abs(system.solve().divergence(q))) < 1e-10
        if digits:
            result=system.solve().velocity(q, extended=True)
            ctx=system.ctx
            expected=ctx.matrix([[model.data(v).mp_values(ctx,q)[i] for v in velocity] for i in range(len(q))])
            assert ctx.norm(result-expected, 'inf') < ctx.mpf('1e-28')
        np.testing.assert_allclose(scalar.solve().evaluate([[.5, .5]]), [1.], atol=1e-10)


@native
def test_field_parameters_rebind_without_recompilation(cache_dir):
    model, problem, spaces, _, _ = field_problem(augmented=False)
    backend = r.CppGlobalBackend(cache_dir=cache_dir, compute_condition=False)
    cloud = r.geometry.unit_box_grid(2)
    previous = None
    with r.GlobalCollocation(spaces=spaces, backend=backend).assemble(problem, cloud) as first:
        worker = first.worker._worker
        previous = first.solve().pressure_gradient([[.31, .43]])
    spaces[model.pressure] = r.PressureSpace(r.Gaussian('2.7'), None)
    with r.GlobalCollocation(spaces=spaces, backend=backend).assemble(problem, cloud) as second:
        assert second.worker._worker is worker
        actual = second.solve().pressure_gradient([[.31, .43]])
        reference = r.GlobalCollocation(spaces=spaces).assemble(problem, cloud).solve()
        np.testing.assert_allclose(actual, reference.pressure_gradient([[.31, .43]]), atol=1e-9)
        assert not np.allclose(actual, previous, atol=1e-10, rtol=1e-10)
