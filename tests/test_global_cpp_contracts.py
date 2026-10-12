"""Native ownership, precision boundaries and public system lifecycle."""
import os

import numpy as np
import pytest
import sympy as sp

import rbflab as r


def native_system(*, digits=None, identity=False, condition=False):
    if os.environ.get("RBFLAB_CPP_TESTS") != "1":
        pytest.skip("Opt-in native integration")
    cloud = r.geometry.unit_box_grid(2)
    model = r.SymbolicScalar()
    u = model.field
    x, y = model.coordinates
    exact = 1 + x*x + 2*y*y
    lhs = u if identity else -model.laplacian(u) + u
    problem = model.stationary(
        sp.Eq(lhs, lhs.subs(u, exact).doit()),
        boundary=[model.bc("boundary", sp.Eq(u, exact))],
    )
    method = r.GlobalCollocation(
        r.PHS(5), scheme="asymmetric", polynomial_degree=2,
        precision=r.Precision(global_digits=digits),
        backend=r.CppGlobalBackend(
            threads=2, compute_condition=condition,
            cache_dir="outputs/global_contract_cache",
        ),
    )
    return method.assemble(problem, cloud)


@pytest.mark.parametrize("digits", [None, 60])
def test_native_assembly_solve_and_evaluation_do_not_use_python_algebra(monkeypatch, digits):
    from rbflab.assembly import Factor
    from rbflab.precision import MPFactor
    from rbflab.nodal import Arithmetic

    def forbidden(*args, **kwargs):
        raise AssertionError("Native path called Python dense assembly or factorization")

    monkeypatch.setattr(Factor, "__init__", forbidden)
    monkeypatch.setattr(MPFactor, "__init__", forbidden)
    monkeypatch.setattr(Arithmetic, "matrix", forbidden)
    with native_system(digits=digits) as system:
        result = system.solve()
        np.testing.assert_allclose(result.evaluate([[.25, .5]]), [1.5625], atol=1e-11)


@pytest.mark.parametrize("digits", [None, 60])
def test_repeated_rhs_keeps_prior_solution_and_worker_alive(digits):
    query = [[.25, .5]]
    with native_system(digits=digits) as system:
        first = system.solve()
        expected = first.evaluate(query)
        system.rhs = [2*v for v in system.rhs]
        second = system.solve()
        np.testing.assert_allclose(second.evaluate(query), 2*expected, atol=1e-11)
        np.testing.assert_allclose(first.evaluate(query), expected, atol=1e-11)
    with pytest.raises(RuntimeError):
        first.evaluate(query)
    system.close()  # Explicit cleanup is idempotent.


def test_replacing_exposed_matrix_invalidates_native_factors():
    with native_system() as system:
        first = system.solve().evaluate([[.25, .5]])
        matrix = np.asarray(system.matrix)
        system.matrix = matrix * 2
        second = system.solve().evaluate([[.25, .5]])
        np.testing.assert_allclose(second, first/2, atol=1e-11)
        with pytest.raises(ValueError):
            system.matrix = np.eye(2)


def test_mp_rhs_change_below_double_precision_survives_solve_and_evaluation():
    with native_system(digits=60) as system:
        first = system.solve().evaluate_mp([[.25, .5]])
        ctx = first.ctx
        multiplier = 1 + ctx.mpf("1e-40")
        system.rhs = [v*multiplier for v in system.rhs]
        second = system.solve().evaluate_mp([[.25, .5]])
        assert second[0] != first[0]
        assert abs(second[0] - first[0]*multiplier) < ctx.mpf("1e-52")


def test_derivative_evaluation_for_value_only_equations():
    with native_system(identity=True) as system:
        solution = system.solve()
        np.testing.assert_allclose(solution.evaluate([[.25, .5]], r.Laplacian()), [6], atol=1e-10)
        np.testing.assert_allclose(solution.evaluate([[.25, .5]], r.Derivative(1)), [2], atol=1e-10)


@pytest.mark.parametrize("digits", [None, 60])
def test_nonfinite_rhs_is_rejected_without_poisoning_worker(digits):
    with native_system(digits=digits) as system:
        saved = list(system.rhs)
        system.rhs[0] = "nan" if digits else float("nan")
        with pytest.raises((ValueError, FloatingPointError)):
            system.solve()
        system.rhs = saved
        np.testing.assert_allclose(system.solve().evaluate([[.25, .5]]), [1.5625], atol=1e-11)


def test_mp_matrix_replacement_preserves_digits_and_is_atomic():
    with native_system(digits=60) as system:
        matrix = system.matrix.to_mpmath()
        ctx = matrix.ctx
        matrix[0, 0] += ctx.mpf("1e-40")
        system.matrix = matrix
        assert abs(system.matrix[0, 0] - matrix[0, 0]) < ctx.mpf("1e-55")
        system.matrix = system.matrix
        assert abs(system.matrix[0, 0] - matrix[0, 0]) < ctx.mpf("1e-55")
        before = system.matrix[0, 0]
        matrix[0, 0] += 1
        matrix[matrix.rows-1, matrix.cols-1] = ctx.nan
        with pytest.raises((ValueError, FloatingPointError)):
            system.matrix = matrix
        assert system.matrix[0, 0] == before


def test_independent_workers_keep_their_requested_precision():
    with native_system(digits=40) as low, native_system(digits=80) as high:
        for system, digits in [(low, 40), (high, 80), (low, 40), (high, 80)]:
            values = system.solve().evaluate_mp([[.25, .5]])
            ctx = values.ctx
            assert ctx.dps == digits
            assert abs(values[0] - ctx.mpf("1.5625")) < ctx.mpf(10)**(-digits + 12)


@pytest.mark.parametrize("digits", [None, 40])
def test_condition_estimates_equilibrated_matrix_one_norm(digits):
    with native_system(digits=digits, condition=True) as system:
        solution = system.solve()
        matrix = system.matrix.to_numpy()
        scale = 1 / np.sqrt(np.max(np.abs(matrix), axis=1))
        balanced = scale[:, None]*matrix*scale[None, :]
        reference = np.linalg.cond(balanced, 1)
        estimate = solution.diagnostics["scaled_condition"]
        assert reference / 20 < estimate < reference * 1.01


def test_normal_solve_and_evaluation_do_not_export_dense_matrix(monkeypatch):
    from rbflab.global_backend import _Worker
    original = _Worker.request
    commands = []

    def record(self, command):
        commands.append(command.split()[0])
        return original(self, command)

    monkeypatch.setattr(_Worker, "request", record)
    with native_system() as system:
        system.solve().evaluate([[.25, .5]])
        assert "EXPORT" not in commands
        assert "GET" not in commands


def test_failed_factorization_can_be_recovered_by_replacing_matrix():
    with native_system() as system:
        saved = system.matrix.to_numpy()
        system.matrix = np.zeros_like(saved)
        with pytest.raises(RuntimeError, match="Zero matrix row"):
            system.solve()
        system.matrix = saved
        np.testing.assert_allclose(system.solve().evaluate([[.25, .5]]), [1.5625], atol=1e-11)


@pytest.mark.parametrize("payload_size", [5, 2**20])
def test_worker_timeout_closes_process_even_when_input_pipe_is_blocked(payload_size):
    import subprocess
    import sys
    import threading
    from rbflab.global_backend import _Worker

    # Exercise the real pipe/timeout code without requiring WSL or compilation.
    worker = _Worker.__new__(_Worker)
    worker.process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    worker.timeout = .1
    worker.lock = threading.RLock()
    try:
        with pytest.raises(TimeoutError):
            worker.request("X" * payload_size)
        assert worker.process.poll() is not None
    finally:
        worker.close(force=True)
