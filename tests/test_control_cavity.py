"""Small driver checks: nodal comparisons, restarts and rejected iterates."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye

from research.navier_stokes_schemes import control_cavity as driver


def fixture_problem():
    zero = csr_matrix((2, 2))
    return SimpleNamespace(
        cells=4, nu=.01, points=np.array([[.25, .25]]),
        boundary_points=np.array([[0., 0.]]),
        A=eye(2, format="csr"), M=eye(2, format="csr"), B=.1*eye(2, format="csr"),
        DxU=zero, DxB=zero, DxL=zero, DyU=zero, DyB=zero, DyL=zero,
        PY=zero, PB=zero, PL=eye(2, format="csr"),
        boundary=lambda t: np.array([.1+t, 0.]),
        force=lambda t: np.array([.2, .1*t]),
        initial=lambda: (np.zeros(2), np.zeros(2)))


def inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "load_maps", lambda _: fixture_problem())
    folder = tmp_path/"maps"
    folder.mkdir()
    np.savez(folder/"map.npz", x=np.array([1.]))
    return folder


def test_missing_ghia_centerline_never_interpolates_computed_solution(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Ghia interpolation called without centerline nodes")
    monkeypatch.setattr(driver, "reference_at_nodal_lines", forbidden)
    result = driver.ghia_at_nodes(np.array([[.25, .25], [.75, .75]]), np.zeros(4))
    assert result["status"] == "unavailable"
    # Having one centerline but not the other is also unavailable.
    assert driver.ghia_at_nodes(np.array([[.5, .25]]), np.zeros(2))["status"] == "unavailable"


def test_existing_ghia_centerlines_pass_only_existing_nodes(monkeypatch):
    points = np.array([[.5, .25], [.25, .5]])
    velocity = np.arange(4.)
    calls = []
    def reference(p, u):
        calls.append((p.copy(), u.copy()))
        return {"u": {"rms_difference": .1}, "v": {"rms_difference": .2}}
    monkeypatch.setattr(driver, "reference_at_nodal_lines", reference)
    assert driver.ghia_at_nodes(points, velocity)["status"] == "available"
    np.testing.assert_array_equal(calls[0][0], points)
    np.testing.assert_array_equal(calls[0][1], velocity)


def test_chunk_restart_matches_uninterrupted_nonzero_boundary(monkeypatch, tmp_path):
    maps = inputs(monkeypatch, tmp_path)
    driver.run(maps, tmp_path/"whole", .01, end=.04)
    driver.run(maps, tmp_path/"split", .01, end=.04, max_steps=2)
    driver.run(maps, tmp_path/"split", .01, end=.04)
    with np.load(tmp_path/"whole"/"fields.npz") as whole, np.load(tmp_path/"split"/"fields.npz") as split:
        for key in ("U", "L", "gradient", "controls", "time"):
            np.testing.assert_array_equal(whole[key], split[key])
    report = json.loads((tmp_path/"split"/"result.json").read_text())
    assert report["status"] == "completed"
    assert report["completed_steps"] == 4
    assert report["ghia"]["status"] == "unavailable"
    assert report["global_factorizations_this_process"] == 1


def test_changed_map_bytes_refuse_restart(monkeypatch, tmp_path):
    maps = inputs(monkeypatch, tmp_path)
    output = tmp_path/"run"
    driver.run(maps, output, .01, end=.04, max_steps=2)
    np.savez(maps/"map.npz", x=np.array([2.]))
    with pytest.raises(ValueError, match="different numerical inputs"):
        driver.run(maps, output, .01, end=.04)


def test_mixed_checkpoint_cohort_is_rejected(monkeypatch, tmp_path):
    maps = inputs(monkeypatch, tmp_path)
    output = tmp_path/"run"
    driver.run(maps, output, .01, end=.04, max_steps=2)
    (output/"history.json").write_text("[]")
    with pytest.raises(ValueError, match="same step"):
        driver.run(maps, output, .01, end=.04)


def test_unresolved_control_candidate_does_not_replace_accepted_state(monkeypatch, tmp_path):
    maps = inputs(monkeypatch, tmp_path)
    original = driver.StokesControl.solve
    calls = 0
    def fail_second(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            result["converged"] = False
            result["residual_inf"] = .1
        return result
    monkeypatch.setattr(driver.StokesControl, "solve", fail_second)
    output = tmp_path/"run"
    driver.run(maps, output, .01, end=.04)
    report = json.loads((output/"result.json").read_text())
    assert report["status"] == "failed"
    assert report["completed_steps"] == 1
    assert report["failure"]["reason"] == "unresolved_control_residual"
    assert (output/"failed_candidate.npz").exists()
    with np.load(output/"checkpoint.npz") as checkpoint, np.load(output/"fields.npz") as fields:
        assert checkpoint["completed"] == 1
        np.testing.assert_array_equal(checkpoint["U"], fields["U"])


def test_speed_bound_records_triggering_accepted_state(monkeypatch, tmp_path):
    maps = inputs(monkeypatch, tmp_path)
    def strong_problem(_):
        p = fixture_problem()
        p.force = lambda t: np.array([2000., 0.])
        return p
    monkeypatch.setattr(driver, "load_maps", strong_problem)
    output = tmp_path/"run"
    driver.run(maps, output, .01, end=.04)
    report = json.loads((output/"result.json").read_text())
    assert report["status"] == "failed"
    assert report["completed_steps"] == 1
    assert report["failure"]["reason"] == "speed_bound_exceeded"
    assert report["peak_nodal_speed"] > 10


@pytest.mark.parametrize("failure_kind", ["exception", "nonfinite"])
def test_solver_failure_checkpoints_last_accepted_step_and_is_not_silently_retried(
        monkeypatch, tmp_path, failure_kind):
    maps = inputs(monkeypatch, tmp_path)
    original = driver.StokesControl.solve
    calls = 0
    def failure(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2 and failure_kind == "exception":
            raise RuntimeError("Injected numerical failure")
        result = original(self, *args, **kwargs)
        if calls == 2:
            result['states'][0, 0] = np.nan
        return result
    monkeypatch.setattr(driver.StokesControl, "solve", failure)
    output = tmp_path/"run"
    driver.run(maps, output, .01, end=.04)
    report = json.loads((output/"result.json").read_text())
    assert report['status'] == 'failed'
    assert report['completed_steps'] == 1
    assert report['failure']['reason'] == 'solver_exception'
    with np.load(output/'checkpoint.npz') as state:
        assert state['completed'] == 1
        assert np.isfinite(state['U']).all()
    with pytest.raises(ValueError, match='Prior run failed'):
        driver.run(maps, output, .01, end=.04)


@pytest.mark.parametrize("changes", [dict(dt=0), dict(end=-1), dict(max_steps=0),
                                    dict(dt=.03), dict(method='unknown')])
def test_invalid_run_controls_are_rejected_before_output_creation(monkeypatch, tmp_path, changes):
    maps = inputs(monkeypatch, tmp_path)
    arguments = dict(dt=.01, end=.04)
    arguments.update(changes)
    output = tmp_path/"invalid"
    with pytest.raises(ValueError):
        driver.run(maps, output, **arguments)
    assert not output.exists()
