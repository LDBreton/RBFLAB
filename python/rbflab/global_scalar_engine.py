"""Native scalar assembly shared by stationary, transient and interpolation APIs."""
import numpy as np
from .global_backend import CppGlobalSystem, _compiled, _acquire_worker, _op_tokens
from .nodal import NodalBasis, dense_arithmetic
from .operators import bind_operators


def assemble_scalar_native(method, centers, target_ops, source_ops, rhs, *, data_plan=None,
                           factor_now=True, system_class=None, arithmetic=None):
    from .methods import _query
    centers = _query(centers)
    if not len(centers):
        raise ValueError('Expected nonempty centers')
    d = centers.shape[1]
    a = arithmetic if arithmetic is not None else dense_arithmetic(method.kernel, method.precision)
    target_ops = bind_operators(list(target_ops), centers)
    source_ops = bind_operators(list(source_ops), centers)
    if len(target_ops) != len(centers) or len(source_ops) != len(centers) or any(op.dimension != d for op in target_ops+source_ops):
        raise ValueError('One matching scalar functional required per center')
    basis = NodalBasis(a, centers, method.polynomial_degree, source_operators=source_ops)
    if basis.powers:
        P = basis.polynomials(centers, source_ops)
        P = np.asarray(P.tolist() if a.ctx else P, dtype=float)
        row = np.max(abs(P), axis=1)
        check = P / np.where(row > 0, row, 1)[:, None]
        col = np.max(abs(check), axis=0)
        check /= np.where(col > 0, col, 1)[None, :]
        if np.linalg.matrix_rank(check) < len(basis.powers):
            raise np.linalg.LinAlgError('Polynomial functionals are not unisolvent')
    source_order = max((sum(alpha) for op in source_ops for alpha, _ in op.terms), default=0)
    target_order = max((sum(alpha) for op in target_ops for alpha, _ in op.terms), default=0)
    order = max(method.backend.derivative_order, target_order) + source_order
    if order > 6:
        raise NotImplementedError('C++ global collocation supports total kernel derivative order up to 6')
    supplied = list(rhs)
    if len(supplied) == len(centers):
        supplied += [0]*len(basis.powers)
    if len(supplied) != len(centers)+len(basis.powers):
        raise ValueError('One scalar datum required per center')
    executable, bound = _compiled(method.kernel, d, order, 'mpfr' if a.ctx else 'float64', method.backend, data_plan=data_plan)
    worker = _acquire_worker(executable, method.backend)
    cls = system_class or CppGlobalSystem
    system = cls(worker, centers, source_ops, basis.powers, supplied, a.ctx,
                 method.precision.global_digits or 17, method.backend, order)
    system.arithmetic = a
    system.target_operators = target_ops
    encode = system.encode
    symmetric = target_ops == source_ops
    tokens = ['INIT', str(system.digits), str(method.backend.threads), str(int(symmetric)),
              str(len(centers)), str(len(basis.powers)), encode(1)]
    tokens.extend(encode(v) for v in bound.values)
    tokens.extend(encode(v) for v in centers.flat)
    tokens.extend(encode(v) for v in centers.flat)
    tokens.extend(encode(v) for v in basis.origin)
    tokens.extend(encode(basis.scale) for _ in range(d))
    tokens.extend(str(v) for power in basis.powers for v in power)
    for op in target_ops+source_ops:
        tokens.extend(_op_tokens(op, encode))
    try:
        worker.request(' '.join(tokens))
        worker.request('BUILD' if factor_now else 'BUILD_MOL')
    except Exception:
        worker.close(force=True)
        raise
    return system
