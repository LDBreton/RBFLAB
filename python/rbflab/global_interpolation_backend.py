"""Owning native scalar interpolants through the existing interpolation API."""
from types import SimpleNamespace
import numpy as np
from .global_backend import CppGlobalBackend, CppGlobalSolution
from .global_scalar_engine import assemble_scalar_native
from .global_data_codegen import native_data_bundle
from .nodal import dense_arithmetic
from .operators import Identity


class CppGlobalInterpolant(CppGlobalSolution):
    def close(self):
        self.system.close()

    def __enter__(self):
        self.system.worker.check_open()
        return self

    def __exit__(self, *_):
        self.close()


def interpolate_cpp(kernel, centers, data, precision, polynomial_degree, backend):
    from .methods import _query
    if not isinstance(backend, CppGlobalBackend):
        raise TypeError('interpolate backend must be CppGlobalBackend or None')
    centers = _query(centers)
    if not len(centers) or len(np.unique(centers, axis=0)) != len(centers):
        raise ValueError('Expected nonempty distinct centers')
    a = dense_arithmetic(kernel, precision)
    bundle = native_data_bundle((data,), centers.shape[1], transient=False)
    plan, ids = bundle if bundle is not None else (None, None)
    samples = [0]*len(centers) if plan is not None else a.data(data, centers)
    if len(samples) != len(centers):
        raise ValueError('One scalar sample required per center')
    method = SimpleNamespace(kernel=kernel, precision=precision, polynomial_degree=polynomial_degree, backend=backend)
    operators = [Identity(centers.shape[1])]*len(centers)
    system = assemble_scalar_native(method, centers, operators, operators, samples, data_plan=plan, arithmetic=a)
    try:
        if plan is not None:
            system.worker.request('DATA_INIT '+str(len(centers))+' '+' '.join([str(ids[0])]*len(centers)))
            result = system.worker.request('RHS_DATA '+system.encode(0))
            if not result or int(result[0]) != system.size or len(result) != system.size+1:
                raise RuntimeError('Invalid native interpolation data result')
            system.rhs = [system.number(value) for value in result[1:]]
        solution = system.solve()
        solution.diagnostics['data_backend'] = 'cpp_symbolic' if plan is not None else 'python'
        return CppGlobalInterpolant(system, solution.coefficients, solution.diagnostics)
    except Exception:
        system.close()
        raise
