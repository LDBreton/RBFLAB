"""First-order characteristic algebra, physical exits and exact chunk history."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes.characteristic_cavity import DepartureOutside, evolve


def problem():
    zero = csr_matrix((2, 2))
    p = SimpleNamespace(points=np.array([[.5, .5]]), bounds=np.array([[0., 1.], [0., 1.]]),
        A=eye(2), M=csr_matrix(np.diag([1., 0.])), B=zero,
        PY=zero, PB=zero, PL=eye(2),
        DxU=zero, DyU=zero, DxB=zero, DyB=zero, DxL=zero, DyL=zero,
        initial=lambda: (np.array([.1, 0.]), np.array([.1, 0.])),
        boundary=lambda t: np.zeros(2), force=lambda t: np.zeros(2))
    return p


class LinearField:
    def __init__(self):
        self.calls = []

    def evaluate(self, U, L, g, feet):
        self.calls.append((U.copy(), L.copy(), g.copy(), feet.copy()))
        return np.column_stack((U[0]*feet[:, 0]/.5, np.zeros(len(feet))))


def test_first_order_foot_and_nonlinear_transport_same_mixed_state():
    p, field, dt = problem(), LinearField(), .02
    result = evolve(p, field, dt, 1)
    # u(x)=U*x/.5 gives u(x-dt*U)=U*(1-dt*U/.5).
    expected = .1*(1-dt*.1/.5)/(1+dt)
    np.testing.assert_allclose(result['U'], [expected, 0.], atol=1e-15)
    np.testing.assert_allclose(result['L'], result['U'], atol=1e-15)
    np.testing.assert_allclose(field.calls[0][3], [[.498, .5]], atol=1e-15)
    np.testing.assert_allclose(field.calls[0][1], [.1, 0.])
    assert result['factorizations'] == 1
    assert result['diagnostics'][0]['solved_residual_inf'] < 1e-13
    assert result['diagnostics'][0]['constraint_residual_inf'] < 1e-13


def test_characteristic_first_order_with_nonlinear_analytic_transport():
    # u(x,t)=a(t)*x, a'+a+a^2=0; evaluate old field exactly.
    errors = []
    for dt in (.01, .005):
        result = evolve(problem(), LinearField(), dt, round(.2/dt))
        a0 = .2
        exact = .5*a0*np.exp(-.2)/(1+a0*(1-np.exp(-.2)))
        errors.append(abs(result['U'][0]-exact))
    assert 1.98 < errors[0]/errors[1] < 2.02


def test_departure_exit_fails_before_evaluator_and_retains_actual_feet():
    p, field = problem(), LinearField()
    p.points = np.array([[.001, .5]])
    with pytest.raises(DepartureOutside) as error:
        evolve(p, field, .02, 1)
    assert not field.calls
    assert error.value.record['owners'] == [0]
    np.testing.assert_allclose(error.value.record['feet'], [[-.001, .5]])


@pytest.mark.parametrize('scheme', ['characteristic', 'be_oseen'])
def test_chunk_restart_matches_whole_history_and_time_dependent_boundary(scheme):
    p = problem()
    p.B = eye(2)
    p.boundary = lambda t: np.array([.01*t, 0.])
    whole = evolve(p, LinearField(), .01, 7, scheme=scheme)
    first = evolve(p, LinearField(), .01, 2, scheme=scheme)
    p.initial = lambda: (first['U'], first['L'])
    second = evolve(p, LinearField(), .01, 5, scheme=scheme, start_step=first['next_step'])
    for name in ('U', 'L', 'gradient'):
        np.testing.assert_allclose(second[name], whole[name], atol=1e-15)
    assert [r['step'] for r in first['diagnostics']+second['diagnostics']] == list(range(1, 8))
    assert second['diagnostics'][-1]['time'] == .07
    assert second['factorizations'] == (1 if scheme == 'characteristic' else 5)


def test_pressure_xy_homogeneous_control_no_spurious_velocity():
    p = problem()
    p.M = csr_matrix((2, 2))
    force = np.array([.5, .5])
    p.force = lambda t: force.copy()
    p.initial = lambda: (np.zeros(2), force.copy())
    result = evolve(p, LinearField(), .01, 3)
    np.testing.assert_allclose(result['U'], 0., atol=1e-15)
    np.testing.assert_allclose(result['gradient'], force, atol=1e-15)


def test_speed_bound_calls_monitor_to_preserve_last_finite_fields():
    p = problem()
    accepted = []
    with pytest.raises(RuntimeError, match='speed bound'):
        evolve(p, LinearField(), .01, 1, speed_bound=.01,
               on_step=lambda r,u,l: accepted.append((r,u,l)))
    assert len(accepted) == 1
    assert accepted[0][0]['step'] == 1


def test_driver_disk_checkpoint_preserves_first_order_history(monkeypatch, tmp_path):
    from research.navier_stokes_schemes import characteristic_cavity as driver
    from research.navier_stokes_schemes import cavity_accuracy_verify, cavity_candidates
    def fixture(*args):
        p = problem()
        p.g0 = np.zeros(2)
        p.nu = .01
        return p
    monkeypatch.setattr(cavity_accuracy_verify, 'problem', fixture)
    monkeypatch.setattr(cavity_candidates, 'reference_at_nodal_lines', lambda *args: {})
    initial = tmp_path/'initial.npz'
    np.savez(initial, U=np.array([.1, 0.]), L=np.array([.1, 0.]))
    source = tmp_path/'source'; (source/'maps').mkdir(parents=True)
    baseline = tmp_path/'raw'; baseline.mkdir()
    for name in ('A','B','SL','PY','PB','PL','DxU','DxB','DxL','DyU','DyB','DyL'):
        (source/'maps'/(name+'.npz')).write_bytes(b'corrected')
        (baseline/(name+'.npz')).write_bytes(b'raw')
    parameters = tmp_path/'parameters.npy'; np.save(parameters, np.zeros(1))
    options = dict(source=source, cells=2, initial=initial,
        baseline=baseline, parameters=parameters, cache=tmp_path/'cache',
        scheme='be_oseen', dt=.01, control='hold', end=.07)
    driver.advance(output=tmp_path/'whole', steps=7, **options)
    driver.advance(output=tmp_path/'split', steps=2, **options)
    row = driver.advance(output=tmp_path/'split', steps=5, **options)
    name = 'hold_be_oseen_0.01'
    whole = np.load(tmp_path/'whole'/name/'checkpoint.npz')
    split = np.load(tmp_path/'split'/name/'checkpoint.npz')
    np.testing.assert_allclose(split['U'], whole['U'], atol=1e-15)
    np.testing.assert_allclose(split['L'], whole['L'], atol=1e-15)
    assert int(split['next_step']) == 8
    ds = json.loads((tmp_path/'split'/name/'diagnostics.json').read_text())
    assert [d['step'] for d in ds] == list(range(1,8))
    assert row['status'] == 'completed'
    trace = np.load(tmp_path/'split'/name/'trajectory.npz')
    np.testing.assert_array_equal(trace['steps'], np.arange(8))
    np.testing.assert_allclose(trace['U'][-1], split['U'], atol=1e-15)
    with pytest.raises(ValueError, match='configuration mismatch'):
        driver.advance(output=tmp_path/'split', steps=1, **{**options, 'end': .08})
    for changed in (source/'maps'/'A.npz', baseline/'DxL.npz', parameters, initial):
        original = changed.read_bytes()
        if changed == initial:
            np.savez(changed, U=np.array([.2, 0.]), L=np.array([.2, 0.]))
        elif changed == parameters:
            np.save(changed, np.ones(1))
        else:
            changed.write_bytes(b'changed')
        with pytest.raises(ValueError, match='configuration mismatch'):
            driver.advance(output=tmp_path/'split', steps=1, **options)
        changed.write_bytes(original)


def test_paired_trajectory_removes_common_root_drift(tmp_path):
    from research.navier_stokes_schemes.characteristic_cavity import paired_difference
    config = dict(source='same', cells=2, nu=.01, reynolds=100., bounds=[[0.,1.],[0.,1.]],
                  scheme='characteristic', dt=.01,
                  end=.02, input_sha256={'initial': 'same'})
    drift = np.array([[0., 0.], [.1, .1], [.2, .2]])
    delta = np.array([[.001, 0.], [.0009, 0.], [.0008, 0.]])
    for name, U in [('hold', drift), ('perturb', drift+delta)]:
        case = tmp_path/name; case.mkdir()
        (case/'configuration.json').write_text(json.dumps({**config, 'control':name}))
        np.savez(case/'trajectory.npz', steps=np.arange(3), U=U, L=U)
    report = paired_difference(tmp_path/'hold', tmp_path/'perturb', tmp_path/'paired')
    assert report['matched_steps'] == 3
    assert report['hold_root_distance_inf'] == .2
    np.testing.assert_allclose(report['final_gain_inf'], .8, atol=1e-13)
    np.testing.assert_allclose(report['peak_gain_inf'], 1., atol=1e-13)
