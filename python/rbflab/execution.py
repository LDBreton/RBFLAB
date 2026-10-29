"""Read-only execution descriptions; never dispatch work or export matrices."""


def execution_info(obj: object) -> dict:
    """Describe a supported object's configured or implemented execution route.

    Args:
        obj: Global method, system, fit, evolution state/trajectory, or local
            approximation to inspect without evaluation or worker requests.

    Returns:
        JSON-compatible execution metadata.

    ``None`` means unknown or not applicable; ``supported=False`` means this
    reporting helper does not cover the type. Implementation describes code
    paths, not runtime profiling. ``recorded`` contains selected existing
    diagnostics only.
    """
    from .methods import GlobalCollocation, GlobalSystem, GlobalSolution
    from .reference import MPGlobalSystem, MPGlobalSolution
    from .nodal import DenseNodalSystem, NodalSolution
    from .global_backend import CppGlobalBackend, CppGlobalSystem, CppGlobalSolution
    from .unsteady_stokes import StokesTrajectory
    from .global_evolution_backend import CppScalarEvolutionSystem
    from .global_vector_interpolation_backend import (
        ComponentwiseInterpolant, VectorInterpolant, CppVectorInterpolant)
    from .evolution import EvolutionSystem, EvolutionState, EvolutionTrajectory
    from .space_stokes import GlobalFlowSystem, GlobalFlowSolution, LocalFlowSystem
    from .local_approximation import LocalApproximation
    from .patches import PatchInterpolant, NativePatchInterpolant
    from .dfpu import DivergenceFreePatchInterpolant, NativeDivergenceFreePatchInterpolant

    result = dict(supported=False, object_type=type(obj).__name__, evidence=None,
                  backend=None, arithmetic=None, digits=None, linear_solver=None,
                  geometry=None, locations=dict.fromkeys(
                      ('assembly', 'solve', 'evaluation', 'data', 'time_stepping')),
                  recorded={})
    if isinstance(obj,(DivergenceFreePatchInterpolant,NativeDivergenceFreePatchInterpolant)):
        native=isinstance(obj,NativeDivergenceFreePatchInterpolant)
        digits=obj.arithmetic.ctx.dps if obj.arithmetic.ctx else None
        result.update(supported=True,evidence='implementation',geometry='float64',
                      backend='cpp_dfpu' if native else 'python_dfpu',
                      arithmetic=('mpfr' if native else 'mpmath') if digits else 'float64',
                      digits=digits,
                      linear_solver=obj.backend.linear_solver if native else 'mpmath LU' if digits else 'scipy LU')
        result['locations'].update(assembly='cpp' if native else 'python',
                                   solve='cpp' if native else 'python',
                                   evaluation='cpp' if native else 'python',
                                   data='python',glue='python')
        result['glue_arithmetic']='mpmath' if digits else 'float64'
        result['operator_combination']='python'
        result['patches']=len(obj.memberships)
        _record(result,obj)
        return result
    if isinstance(obj,(PatchInterpolant,NativePatchInterpolant)):
        native=isinstance(obj,NativePatchInterpolant)
        digits=obj.arithmetic.ctx.dps if obj.arithmetic.ctx else None
        result.update(supported=True,evidence='implementation',geometry='float64',
                      backend='cpp_pum' if native else 'python_pum',
                      arithmetic=('mpfr' if native else 'mpmath') if digits else 'float64',
                      digits=digits,
                      linear_solver=obj.backend.linear_solver if native else 'mpmath LU' if digits else 'scipy LU')
        result['locations'].update(assembly='cpp' if native else 'python',
                                   solve='cpp' if native else 'python',
                                   evaluation='cpp' if native else 'python',
                                   data='python')
        result['patches']=len(obj.memberships)
        _record(result,obj)
        return result
    if isinstance(obj, (EvolutionTrajectory, StokesTrajectory)):
        if obj.states:
            result = execution_info(obj.states[-1])
            result['object_type'] = type(obj).__name__
        _record(result, obj)
        return result
    if isinstance(obj, ComponentwiseInterpolant):
        reports = [execution_info(fit) for fit in obj.solutions]
        if reports:
            from copy import deepcopy
            result = deepcopy(reports[0])
            for key in ('backend', 'arithmetic', 'digits', 'linear_solver'):
                if any(report[key] != result[key] for report in reports):
                    result[key] = 'mixed' if key != 'digits' else None
            for key in result['locations']:
                if any(report['locations'][key] != result['locations'][key] for report in reports):
                    result['locations'][key] = 'mixed'
        result['object_type'] = type(obj).__name__
        result['components'] = reports
        _record(result, obj)
        return result
    fits = (GlobalSolution, MPGlobalSolution, NodalSolution, CppGlobalSolution,
            CppVectorInterpolant, GlobalFlowSolution, EvolutionState)
    if isinstance(obj, fits):
        result = execution_info(obj.system)
        result['object_type'] = type(obj).__name__
        _record(result, obj)
        return result
    locations = result['locations']
    if isinstance(obj, LocalFlowSystem):
        import copy
        info = getattr(obj.base, 'backend_diagnostics', {})
        native_weights = str(info.get('backend', '')).startswith('cpp')
        local = obj.base.local_arithmetic
        digits = local.ctx.dps if local.ctx else None
        result.update(supported=True, evidence='implementation', geometry='float64',
            backend=info.get('backend', 'python'), digits=digits,
            arithmetic=('mpfr' if native_weights else 'mpmath') if digits else 'float64',
            linear_solver='mpmath sparse LU' if obj.base.arithmetic.ctx else 'scipy sparse LU',
            local_weight_engine=info.get('local_weight_engine', 'specialized Stokes'),
            local_execution=copy.deepcopy(info.get('local_execution', [])),
            global_arithmetic='mpmath' if obj.base.arithmetic.ctx else 'float64')
        locations.update(assembly='cpp local weights; python global assembly' if native_weights
                         else 'python local weights and global assembly',
                         solve='python', evaluation='python patchwise reconstruction',
                         data='python', time_stepping='python' if obj.problem.transient else None)
        return result
    if isinstance(obj, LocalApproximation):
        result.update(supported=True, evidence='configured', geometry='float64',
                      backend=type(obj.backend).__name__, digits=obj.precision.local_digits)
        backend = result['backend']
        result['arithmetic'] = ('mpfr' if backend == 'CppBackend' else 'mpmath') if result['digits'] else 'float64'
        locations.update(assembly='cpp local weights; python sparse maps' if backend == 'CppBackend'
                         else 'torch local weights and maps' if backend == 'TorchBackend'
                         else 'python local weights and sparse maps')
        return result
    native = isinstance(obj, CppGlobalSystem) or (
        isinstance(obj, GlobalCollocation) and isinstance(obj.backend, CppGlobalBackend))
    systems = (GlobalSystem, MPGlobalSystem, DenseNodalSystem, GlobalFlowSystem)
    if not (native or isinstance(obj, (GlobalCollocation, VectorInterpolant) + systems)
            or isinstance(obj, EvolutionSystem) and obj.kind == 'global'):
        return result
    configured = isinstance(obj, GlobalCollocation)
    if configured and obj.backend is not None and not native:
        return result
    result.update(supported=True, evidence='configured' if configured else 'implementation',
                  geometry='float64', backend='cpp_global' if native else 'python')
    basis = getattr(obj, 'basis', None)
    arithmetic = getattr(obj, 'arithmetic', None) or getattr(basis, 'arithmetic', None) or getattr(basis, 'a', None)
    ctx = getattr(obj, 'ctx', None) or getattr(arithmetic, 'ctx', None)
    if isinstance(obj, MPGlobalSystem):
        ctx = obj.backend.ctx
    digits = obj.precision.global_digits if configured else (obj.digits if native and ctx else ctx.dps if ctx else None)
    result.update(digits=digits, arithmetic=('mpfr' if native else 'mpmath') if digits else 'float64',
                  linear_solver=obj.backend.linear_solver if native else 'mpmath LU' if digits else 'scipy LU')
    locations.update(assembly='cpp' if native else 'python', solve='cpp' if native else 'python',
                     evaluation='cpp' if native else 'python', data=None if native else 'python')
    if isinstance(obj, (EvolutionSystem, CppScalarEvolutionSystem)):
        locations['time_stepping'] = 'python orchestration; cpp step algebra' if native else 'python'
    elif (isinstance(obj, GlobalFlowSystem) or type(obj).__name__ == 'CppGlobalFlowSystem') and obj.problem.transient:
        locations['time_stepping'] = 'python orchestration/history; cpp mass and shifted solve' if native else 'python'
    data = getattr(obj, 'data_backend', None)
    if data is not None:
        locations['data'] = data
    _record(result, obj)
    return result


def _record(result, obj):
    diagnostics = getattr(obj, 'diagnostics', {})
    for key in ('backend', 'linear_solver', 'data_backend', 'data_fallback_reason',
                'scheme', 'steps', 'global_digits', 'digits'):
        if key in diagnostics:
            value = diagnostics[key]
            result['recorded'][key] = value if value is None or isinstance(value, (str, bool, int, float)) else str(value)
    if 'data_backend' in result['recorded']:
        result['locations']['data'] = result['recorded']['data_backend']
