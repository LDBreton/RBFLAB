"""Stokes equation adapter over generic LocalApproximation Hermite weights.

The historical stencil/data layout and compact equation assembly stay in the
Stokes adapter. Kernel contractions, augmentation and local weight solves use
the same public mixed approximation path as other applications.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
import time

import numpy as np


def _functional(row):
    from .mixed import FieldOperator
    return FieldOperator({name: [[row.get(component, 0) for component in components]]
                          for name, components in [('u', (0, 1)), ('p', (2,))]
                          if any(component in row for component in components)})


def _space(task):
    from .spaces import MixedSpace, DivergenceFreeSpace, PressureSpace
    return MixedSpace({'u': DivergenceFreeSpace(task['vk'], task['degree']),
                       'p': PressureSpace(task['pk'], task.get('pressure_degree'))})


def _numerical_backend(backend):
    from .lhi_backends import PythonBackend
    from .configuration import LocalSolver
    common = replace(backend, workers=1) if isinstance(backend, PythonBackend) else replace(backend)
    # The adapter already bound the selected shape rule into each kernel.
    object.__setattr__(common, '_shape_rule', 'fixed')
    object.__setattr__(common, '_solver', LocalSolver('lu'))
    return common


def _targets(mu):
    from .operators import Laplacian, Derivative
    return {name: _functional(row) for name, row in zip(
        ('Lx', 'Ly', 'px', 'py'),
        [{c: -Laplacian()*mu, 2: Derivative(c)} for c in (0, 1)]
        + [{2: Derivative(c)} for c in (0, 1)])}


def _extract(maps, index, arithmetic):
    encode = lambda value: arithmetic.ctx.nstr(value, arithmetic.ctx.dps+5) if arithmetic.ctx else float(value)
    rows = [maps[name].local(index) for name in ('Lx', 'Ly', 'px', 'py')]
    return dict(weights=[[encode(value) for value in local.weights[:, 0]] for local in rows],
                condition=rows[0].diagnostics['scaled_condition'],
                residual=rows[0].diagnostics['weight_residual'])


def _one_python_task(task):
    """Process-pool entry point; the numerical solve still uses the public API."""
    from .local_approximation import LocalApproximation
    from .samples import Samples
    from .precision import Precision
    from .stencils import StencilPolicy
    from .lhi_backends import PythonBackend
    source = {'observations': Samples(task['points'], operator=[_functional(row) for row in task['ops']])}
    local = LocalApproximation(source=source, trial=_space(task).representers(source),
        backend=PythonBackend(compute_condition=task['condition']),
        precision=Precision(local_digits=task['digits'], global_dtype='mpmath' if task['digits'] else 'float64'),
        stencil_policy=StencilPolicy(scaling='local' if task['scale'] is not None else 'physical'))
    maps = local.operators(targets=np.asarray(task['origin'])[None, :], operators=_targets(task['mu']))
    return _extract(maps, 0, maps.Lx._owner.arithmetic)


def assemble(method, problem, cloud, backend):
    """Preserve historical Stokes stencils, replacing only their weight engine."""
    from .lhi_backends import _prepare, _finish, _init_worker, PythonBackend
    from .local_approximation import LocalApproximation
    from .samples import Samples
    from .precision import Precision
    from .operators import Identity, Laplacian, Derivative
    from .mixed_backends import LocalExecution

    start = time.perf_counter()
    if backend.local_solver != 'lu':
        raise NotImplementedError('The common Stokes LHI adapter supports LU; select the explicit supported SVD backend route')
    a, tasks, stencils = _prepare(method, problem, cloud, backend.shape_rule)
    from .stokes_polynomials import rank_ratio
    for stencil in stencils:
        if stencil.polynomial_basis is not None:
            ratio = rank_ratio(stencil.polynomial_basis.matrix(stencil.points, stencil.functionals))
            if ratio < 1e-12:
                raise ValueError('Polynomial functionals are rank deficient; enlarge stencil')
            stencil.geometry_diagnostics['functional_polynomial_ratio'] = ratio
    for task in tasks:
        task['condition'] = backend.compute_condition
    execution = getattr(method, 'execution', None)
    if execution is not None and not isinstance(execution, LocalExecution):
        raise TypeError('execution must be LocalExecution')
    results, executions = [None]*len(tasks), []
    if isinstance(backend, PythonBackend) and backend.workers > 1:
        if execution is not None and (execution.scheduler != 'serial' or execution.reuse != 'none'):
            raise NotImplementedError('MPI/reuse requires CppBackend')
        with ProcessPoolExecutor(max_workers=min(backend.workers, len(tasks)),
                mp_context=multiprocessing.get_context('spawn'), initializer=_init_worker) as pool:
            results = list(pool.map(_one_python_task, tasks, chunksize=1))
        executions.append(dict(scheduler='processes', workers=min(backend.workers, len(tasks)),
                               requested_jobs=len(tasks), unique_jobs=len(tasks)))
    else:
        batches = defaultdict(list)
        for i, task in enumerate(tasks):
            batches[repr((task['vk'], task['pk'], task['degree'], task.get('pressure_degree')))].append(i)
        ni, nb = len(cloud.interior_indices), len(cloud.boundary_indices)
        functions = [{c: Identity()} for c in (0, 1)]
        momentum = [{c: -Laplacian()*problem.viscosity, 2: Derivative(c)} for c in (0, 1)]
        specifications = [(group, c, size, points, op)
            for group, size, points, ops in [('solution', ni, cloud.interior, functions),
                ('boundary', nb, cloud.points[cloud.boundary_indices], functions),
                ('pde', ni, cloud.interior, momentum)]
            for c, op in enumerate(ops)]
        common_backend = _numerical_backend(backend)
        for ids in batches.values():
            source = {}
            for group, component, size, points, op in specifications:
                memberships = [[col % size for name, col in stencils[i].slots
                    if name == group and col // size == component] for i in ids]
                source[f'{group}_{component}'] = Samples(points, operator=_functional(op),
                    indices=memberships, target='exclude' if group == 'pde' else 'allow')
            representative = tasks[ids[0]]
            local = LocalApproximation(source=source, trial=_space(representative).representers(source),
                backend=common_backend,
                precision=Precision(local_digits=method.precision.local_digits,
                    global_dtype='mpmath' if a.ctx else 'float64'),
                stencil_policy=replace(method.stencil_policy, shape_rule='fixed'), execution=execution)
            maps = local.operators(targets=cloud.points[[stencils[i].center for i in ids]],
                                   operators=_targets(problem.viscosity))
            executions.append(maps.diagnostics.get('execution', {}))
            for row, i in enumerate(ids):
                results[i] = _extract(maps, row, a)
    native = not isinstance(backend, PythonBackend)
    diagnostics = dict(backend=('cpp_mpfr' if a.ctx else 'cpp_float64') if native else 'python',
        local_weight_engine='LocalApproximation', local_solver='lu', shape_rule=backend.shape_rule,
        workers=min(backend.workers, len(tasks)) if not native else None,
        threads=getattr(backend, 'threads', None), compute_condition=backend.compute_condition,
        local_digits=method.precision.local_digits, off_node_backend='lazy Python',
        condition_norm='infinity' if a.ctx or native else '2', local_execution=executions)
    if native:
        diagnostics['cpp_process_seconds'] = sum(item.get('seconds', 0) for item in executions)
    system = _finish(method, problem, cloud, a, tasks, stencils, results, diagnostics)
    system.backend_diagnostics['assembly_seconds'] = time.perf_counter()-start
    return system
