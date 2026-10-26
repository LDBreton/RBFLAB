"""Execution reporting inspects metadata without numerical dispatch."""
import json
from types import SimpleNamespace

from rbflab import (CppGlobalBackend, GlobalCollocation, IMQ, Precision,
                    ScalarSpace, execution_info, interpolate)
from rbflab.global_backend import CppGlobalSystem, CppGlobalSolution


def test_configuration_and_unknown_are_distinct():
    report = execution_info(GlobalCollocation(IMQ(), precision=Precision(global_digits=40)))
    assert report['evidence'] == 'configured'
    assert (report['arithmetic'], report['digits']) == ('mpmath', 40)
    assert execution_info(SimpleNamespace(backend=CppGlobalBackend()))['supported'] is False
    json.dumps(report)


def test_python_scalar_componentwise_and_mp_fits():
    points = [[0., 0.], [1., 0.], [0., 1.]]
    for precision in (Precision(), Precision(global_digits=35)):
        fit = interpolate(ScalarSpace(IMQ()), points, [[1, 2], [2, 3], [3, 4]], precision=precision)
        report = execution_info(fit)
        assert report['supported'] and len(report['components']) == 2
        assert report['backend'] == 'python'
        assert report['digits'] == precision.global_digits
        json.dumps(report)


def test_native_metadata_never_requests_worker_or_exports_matrix():
    class ForbiddenWorker:
        def request(self, *args):
            raise AssertionError('reporting must not invoke native work')
    system = object.__new__(CppGlobalSystem)
    system.worker = ForbiddenWorker()
    system.backend = CppGlobalBackend()
    system.ctx = None
    system.digits = 17
    fit = CppGlobalSolution(system, [], {'data_backend': 'python', 'linear_solver': 'eigen'})
    report = execution_info(fit)
    assert report['locations']['evaluation'] == 'cpp'
    assert report['locations']['data'] == 'python'
    assert report['digits'] is None
    assert report['recorded']['linear_solver'] == 'eigen'
    json.dumps(report)


def test_local_and_scalar_trajectory_metadata():
    import rbflab as r
    import sympy as sp
    local = r.LocalApproximation(source=[[0., 0.], [1., 0.], [0., 1.]],
                                 space=r.ScalarSpace(r.IMQ()), stencil_size=3)
    report = execution_info(local)
    assert report['locations']['solve'] is None
    assert report['evidence'] == 'configured'
    json.dumps(report)
    model = r.SymbolicScalar(2, transient=True)
    u, t = model.field, model.time
    problem = model.evolution(sp.Eq(sp.diff(u, t) - model.laplacian(u), 0),
                              initial=1, boundary=[model.bc('boundary', sp.Eq(u, 1))])
    system = r.GlobalCollocation(space=r.ScalarSpace(r.IMQ(), 1)).assemble(problem, r.geometry.unit_box_grid(2))
    trajectory = system.solve(dt=.01, steps=1)
    for obj in (system, trajectory.states[0], trajectory):
        report = execution_info(obj)
        assert report['locations']['time_stepping'] == 'python'
        json.dumps(report)


def test_stationary_and_transient_global_flow_metadata():
    import rbflab as r
    import sympy as sp
    for transient in (False, True):
        model = r.SymbolicStokes(2, transient=transient)
        u, p = model.fields
        lhs = -model.laplacian(u) + model.gradient(p)
        spaces = {u: r.DivergenceFreeSpace(r.IMQ(), 1), p: r.PressureSpace(r.IMQ(), 1)}
        boundary = [model.bc('boundary', sp.Eq(u, sp.ImmutableMatrix([1, 1])))]
        if transient:
            problem = model.evolution([sp.Eq(sp.diff(u, model.time) + lhs, sp.zeros(2, 1))],
                                      initial={u: sp.ImmutableMatrix([1, 1])}, boundary=boundary)
        else:
            problem = model.stationary([sp.Eq(lhs, sp.zeros(2, 1))], boundary=boundary)
        system = r.GlobalCollocation(spaces=spaces).assemble(problem, r.geometry.unit_box_grid(2))
        report = execution_info(system)
        assert report['locations']['time_stepping'] == ('python' if transient else None)
        json.dumps(report)
        if transient:
            trajectory = system.solve(dt=.01, steps=1)
            json.dumps(execution_info(trajectory))
