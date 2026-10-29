"""Public Stokes scheduling and read-only execution reporting."""
import json
import os
import pytest
import sympy as sp

import rbflab as r


def problem():
    model = r.SymbolicStokes(2)
    u, p = model.fields
    equation = -model.laplacian(u)+model.gradient(p)
    case = model.stationary([sp.Eq(equation, sp.zeros(2, 1))],
        boundary=[model.bc('boundary', sp.Eq(u, sp.ImmutableMatrix([1, 1])))])
    spaces = {u: r.DivergenceFreeSpace(r.IMQ(2), 1), p: r.PressureSpace(r.IMQ(2), 1)}
    return case, spaces


def test_stokes_calls_common_engine_and_reports_actual_route(monkeypatch):
    case, spaces = problem()
    original = r.LocalApproximation.operators
    calls = []
    def traced(self, **kwargs):
        calls.append(tuple(kwargs['operators']))
        return original(self, **kwargs)
    monkeypatch.setattr(r.LocalApproximation, 'operators', traced)
    system = r.LHI(spaces=spaces, stencil_size=12,
        local_backend=r.PythonBackend(compute_condition=False),
        execution=r.LocalExecution()).assemble(case, r.geometry.unit_box_grid(3))
    assert calls and calls[0] == ('Lx', 'Ly', 'px', 'py')
    monkeypatch.setattr(r.LocalApproximation, 'operators', lambda *a, **k: pytest.fail('metadata must not assemble'))
    info = r.execution_info(system)
    assert info['supported'] and info['local_weight_engine'] == 'LocalApproximation'
    assert info['locations']['solve'] == 'python'
    assert info['global_arithmetic'] == 'float64'
    assert info['local_execution'][0]['scheduler'] == 'serial'
    assert system.recipe['execution']['scheduler'] == 'serial'
    assert system.recipe['backend_diagnostics']['local_weight_engine'] == 'LocalApproximation'
    json.dumps(info)


def test_stokes_mpi_rejects_wrong_backend_and_scalar_execution():
    case, spaces = problem()
    with pytest.raises(NotImplementedError, match='require CppBackend'):
        r.LHI(spaces=spaces, execution=r.LocalExecution(scheduler='mpi', ranks=2)).preflight(case)
    scalar = r.LinearPDE(-r.Laplacian(), 0, [r.Dirichlet(0)])
    with pytest.raises(NotImplementedError, match='Stokes LHI only'):
        r.LHI(r.IMQ(2), execution=r.LocalExecution()).preflight(scalar)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')
@pytest.mark.parametrize('scaling', ['physical', 'local'])
def test_stokes_native_preparation_uses_generic_worker(scaling, monkeypatch, tmp_path):
    case, spaces = problem()
    import rbflab.kernel_compiler as compiler
    monkeypatch.setattr(compiler, 'compile_stokes',
                        lambda *a, **k: pytest.fail('legacy LU worker must not be prepared'))
    method = r.LHI(spaces=spaces, stencil_size=12,
                   stencil_policy=r.StencilPolicy(scaling=scaling),
                   local_backend=r.CppBackend(compute_condition=False))
    prepared = method.prepare(case, dimension=2, cache_dir=tmp_path)
    assert prepared.compiled_artifact.is_file()
    system = prepared.assemble(case, r.geometry.unit_box_grid(3))
    assert r.execution_info(system)['local_weight_engine'] == 'LocalApproximation'


@pytest.mark.parametrize('scheduler', ['serial', 'mpi'])
def test_public_cpp_local_scaling_preflight(scheduler):
    case, spaces = problem()
    method = r.LHI(spaces=spaces, stencil_size=12,
        stencil_policy=r.StencilPolicy(scaling='local'),
        local_backend=r.CppBackend(compute_condition=False),
        execution=r.LocalExecution(scheduler=scheduler, ranks=3 if scheduler == 'mpi' else 1))
    method.preflight(case, r.geometry.unit_box_grid(3))


@pytest.mark.parametrize('route', ['svd', 'custom'])
def test_public_specialized_cpp_local_scaling_rejected(route):
    case, spaces = problem()
    method = r.LHI(spaces=spaces, stencil_policy=r.StencilPolicy(scaling='local'),
        local_backend=r.CppBackend(executable='unused-legacy-executable' if route == 'custom' else None),
        local_solver=r.LocalSolver('svd' if route == 'svd' else 'lu'))
    with pytest.raises(NotImplementedError, match='require physical kernel coordinates'):
        method.preflight(case)


@pytest.mark.parametrize('shape_rule', ['hardy', 'radius_scaled', 'separation_scaled'])
@pytest.mark.parametrize('backend', [r.PythonBackend, r.CppBackend])
def test_stokes_geometry_shape_rules_require_physical_coordinates(shape_rule, backend):
    case, spaces = problem()
    method = r.LHI(spaces=spaces,
        stencil_policy=r.StencilPolicy(scaling='local', shape_rule=shape_rule),
        local_backend=backend())
    with pytest.raises(NotImplementedError, match='shape rules require physical kernel coordinates'):
        method.preflight(case)
