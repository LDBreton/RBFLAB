"""Checkpoint parity on a nonlinear mixed system with time-varying inputs."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from research.navier_stokes_schemes.coupled import evolve as uninterrupted
from research.navier_stokes_schemes.equilibrium_segment_engine import evolve as checkpointed


def nonlinear_problem():
    rng = np.random.default_rng(20261029)
    n = 4
    dense = lambda scale: csr_matrix(scale * rng.normal(size=(n, n)))
    a = csr_matrix(np.diag([1.2, 1.5, 1.1, 1.4]))
    m = csr_matrix(np.diag([.8, 1.1, .9, 1.2]))
    boundary_map = dense(.12)
    boundary = lambda t: np.array([.4 + .3*t, -.2*np.cos(t), .1 + .2*t*t, .3*np.sin(2*t)])
    force = lambda t: np.array([.7*np.cos(t), -.3 + t, .4*np.sin(t) + .2, .5 - .1*t*t])
    u0 = np.array([.3, -.2, .15, .25])
    l0 = np.linalg.solve(m.toarray(), a@u0 + boundary_map@boundary(0))
    return SimpleNamespace(
        A=a, M=m, B=boundary_map,
        PY=dense(.3), PB=dense(.25), PL=dense(.4),
        DxU=dense(.35), DxB=dense(.3), DxL=dense(.25),
        DyU=dense(.3), DyB=dense(.25), DyL=dense(.35),
        boundary=boundary, force=force,
        initial=lambda: (u0.copy(), l0.copy()),
        linear_solver='direct', global_ordering='COLAMD',
    )


@pytest.mark.parametrize('scheme', ['oseen', 'oseen_newton'])
def test_checkpoint_preserves_nonlinear_fields_gradient_and_absolute_time(scheme):
    p = nonlinear_problem()
    dt, steps, cut = .08, 13, 5
    options = dict(rtol=1e-12, atol=1e-14)
    reference = uninterrupted(p, scheme, dt, steps, **options)
    whole = checkpointed(p, scheme, dt, steps, **options)
    first = checkpointed(p, scheme, dt, cut, **options)
    p.initial = lambda: (first['U'].copy(), first['L'].copy())
    resumed = checkpointed(p, scheme, dt, steps-cut, _history=first['_history'], **options)

    for result in (whole, resumed):
        for field in ('U', 'L', 'gradient'):
            np.testing.assert_allclose(result[field], reference[field], rtol=0, atol=3e-14)
    assert first['_history'][2] == cut + 1
    assert resumed['_history'][2] == steps + 1
    # The exported pressure gradient includes the boundary at the final
    # absolute time, so resetting time on resume cannot pass this assertion.
    expected_gradient = p.PY@resumed['U'] + p.PB@p.boundary(dt*steps) + p.PL@resumed['L']
    np.testing.assert_allclose(resumed['gradient'], expected_gradient, rtol=0, atol=3e-14)

    combined = first['diagnostics'] + resumed['diagnostics']
    assert [d['step'] for d in combined] == list(range(1, steps+1))
    for actual, expected in zip(combined, reference['diagnostics']):
        for name in expected:
            if name in ('step', 'newton_iterations'):
                assert actual[name] == expected[name]
            else:
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=3e-14)
    for actual, expected in zip(whole['diagnostics'], reference['diagnostics']):
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=3e-14)

    if scheme == 'oseen_newton':
        assert sum(d['newton_iterations'] for d in reference['diagnostics']) >= steps
        assert all(d['nonlinear_residual_inf'] <= d['tolerance'] for d in combined)
        # Ensure this is a nonlinear correction test, rather than a linear
        # limit where the Oseen predictor already solves the full problem.
        p = nonlinear_problem()
        predictor = uninterrupted(p, 'oseen', dt, steps, **options)
        assert np.max(abs(predictor['U']-reference['U'])) > 1e-5


def test_resumed_krylov_checks_the_actual_final_absolute_step():
    p = nonlinear_problem()
    p.linear_solver = 'gmres_scaled'
    dt, cut, remaining = .05, 4, 3
    first = checkpointed(p, 'oseen', dt, cut)
    p.initial = lambda: (first['U'].copy(), first['L'].copy())
    resumed = checkpointed(p, 'oseen', dt, remaining, _history=first['_history'])
    assert resumed['diagnostics'][-1]['step'] == cut + remaining
    for name in ('direct_check_velocity_max', 'direct_check_gradient_max'):
        assert name in resumed['diagnostics'][-1]
        assert resumed['diagnostics'][-1][name] < 1e-11
    assert all('direct_check_velocity_max' not in d for d in resumed['diagnostics'][:-1])
