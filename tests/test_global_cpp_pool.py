"""Worker reuse and lease isolation without requiring a native compiler."""
from types import SimpleNamespace
import os
import numpy as np
import pytest
import rbflab.global_backend as gb


class FakeWorker:
    def __init__(self, executable, backend):
        self.dead = False
        self.commands = []
        self.timeout = backend.timeout
        self.process = SimpleNamespace(poll=lambda: 0 if self.dead else None)

    def request(self, command):
        if self.dead: raise RuntimeError('dead worker')
        self.commands.append(command)
        return ['1']

    def close(self, force=False):
        self.dead = True


@pytest.fixture
def pool(monkeypatch, tmp_path):
    gb._close_workers()
    monkeypatch.setattr(gb, '_Worker', FakeWorker)
    executable = tmp_path / 'kernel'
    executable.write_text('worker')
    yield executable
    gb._close_workers()


def test_reuse_resets_and_invalidates_old_lease(pool):
    backend = gb.CppGlobalBackend()
    first = gb._acquire_worker(pool, backend)
    worker = first._worker
    first.close()
    assert worker.commands == ['RESET']
    second = gb._acquire_worker(pool, gb.CppGlobalBackend(timeout=3))
    assert second._worker is worker
    assert worker.timeout == 3
    with pytest.raises(RuntimeError, match='closed'): first.request('GET 0 0')
    first.close()
    assert not worker.dead
    second.close()


def test_active_workers_are_exclusive_and_pool_is_bounded(pool):
    leases = [gb._acquire_worker(pool, gb.CppGlobalBackend()) for _ in range(6)]
    assert len({id(lease._worker) for lease in leases}) == 6
    for lease in leases: lease.close()
    assert len(gb._idle_workers) == gb._IDLE_WORKER_LIMIT
    assert sum(lease._worker.dead for lease in leases) == 2


def test_opt_out_and_reset_failure_discard(pool):
    lease = gb._acquire_worker(pool, gb.CppGlobalBackend(reuse_workers=False))
    lease.close()
    assert lease._worker.dead and not gb._idle_workers
    lease = gb._acquire_worker(pool, gb.CppGlobalBackend())
    def fail(command): raise RuntimeError('failed reset')
    lease._worker.request = fail
    lease.close()
    assert lease._worker.dead and not gb._idle_workers


def test_executable_replacement_and_dead_idle_worker(pool):
    lease = gb._acquire_worker(pool, gb.CppGlobalBackend())
    lease.close()
    pool.write_text('new executable identity')
    replacement = gb._acquire_worker(pool, gb.CppGlobalBackend())
    assert replacement._worker is not lease._worker
    replacement.close()
    replacement._worker.dead = True
    fresh = gb._acquire_worker(pool, gb.CppGlobalBackend())
    assert fresh._worker is not replacement._worker
    fresh.close()


def test_closed_system_matrix_solution_and_empty_evaluation(pool):
    lease = gb._acquire_worker(pool, gb.CppGlobalBackend())
    system = gb.CppGlobalSystem(lease, np.zeros((1, 2)), [], [], [1], None, 17,
                               gb.CppGlobalBackend(), 2)
    matrix = system.matrix
    solution = gb.CppGlobalSolution(system, np.ones(1), {})
    system.close()
    for action in (system.solve, lambda: system.matrix, lambda: matrix[0, 0],
                   lambda: matrix.shape, lambda: solution.evaluate(np.empty((0, 2)))):
        with pytest.raises(RuntimeError, match='closed'): action()


def test_reuse_flag_validation():
    with pytest.raises(TypeError, match='reuse_workers'): gb.CppGlobalBackend(reuse_workers=1)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')
def test_native_reuse_resets_precision_matrix_and_threads(tmp_path):
    import rbflab as r
    gb._close_workers()
    worker = None
    old_solution = None
    problem = r.LinearPDE(r.Identity(), 1.0, (r.Dirichlet(1.0),))
    try:
        for digits, nodes, threads, rhs in ((40, 2, 1, 1), (80, 3, 2, 2), (40, 2, 1, 3)):
            cloud = r.geometry.unit_box_grid(nodes, 2)
            backend = r.CppGlobalBackend(cache_dir=tmp_path, threads=threads, compute_condition=False)
            system = r.GlobalCollocation(r.IMQ(), precision=r.Precision(global_digits=digits),
                                         backend=backend).assemble(problem, cloud)
            try:
                if worker is not None:
                    assert system.worker._worker is worker
                    with pytest.raises(RuntimeError, match='closed'):
                        old_solution.evaluate(np.array([[.3, .4]]))
                worker = system.worker._worker
                # A reused process must discard edited entries and factors.
                assert abs(system.matrix[0, 0] - 1) < system.ctx.mpf('1e-35')
                system.rhs[:] = [system.ctx.mpf(rhs)] * system.size
                solution = system.solve()
                query = np.array([[.3, .4], [.7, .6]])
                reference = r.GlobalCollocation(r.IMQ(), precision=r.Precision(global_digits=digits)).assemble(problem, cloud)
                reference.rhs = reference.matrix.ctx.matrix([rhs] * system.size)
                expected = reference.solve().evaluate_mp(query)
                actual = solution.evaluate_mp(query)
                assert system.ctx.norm(actual - system.ctx.matrix(expected.tolist()), 'inf') < system.ctx.mpf(10) ** (-(digits-10))
                assert system.backend.threads == threads
                system.matrix[0, 0] = 5
                system.solve()
                old_solution = solution
            finally:
                system.close()
    finally:
        gb._close_workers()
