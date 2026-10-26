"""Global interpolation in an explicitly divergence-free vector space."""
import numpy as np
from .discrete_operators import _Basis
from .nodal import dense_arithmetic
from .operators import Identity, bind_operators
from .global_data_codegen import native_data_bundle
from .global_backend import CppGlobalSystem, _compiled, _acquire_worker
from .global_flow_backend import _field_tokens


class ComponentwiseInterpolant:
    """Independent scalar components, sharing one native factorization."""
    def __init__(self, solutions, system=None):
        self.solutions, self.system = solutions, system
        ctx = system.ctx if system is not None else solutions[0].system.basis.arithmetic.ctx
        self.ctx = ctx
        columns = [list(solution.coefficients) for solution in solutions]
        self.coefficients = ctx.matrix([[column[i] for column in columns] for i in range(len(columns[0]))]) if ctx else np.column_stack(columns)
        self.diagnostics = {'components': len(solutions), 'ordering': 'node-major', 'backend': 'cpp_global' if system is not None else 'python',
                            'component_diagnostics': [solution.diagnostics for solution in solutions]}

    def _sync_coefficients(self):
        expected=(len(self.solutions[0].coefficients),len(self.solutions))
        actual=(self.coefficients.rows,self.coefficients.cols) if self.ctx else np.shape(self.coefficients)
        if actual!=expected:raise ValueError('Coefficient matrix dimensions changed')
        for c,solution in enumerate(self.solutions):
            solution.coefficients=self.ctx.matrix([self.coefficients[i,c] for i in range(expected[0])]) if self.ctx else np.asarray(self.coefficients[:,c])

    def evaluate(self, points, operator=None):
        self._sync_coefficients()
        return np.column_stack([solution.evaluate(points, operator) for solution in self.solutions])

    def evaluate_mp(self, points, operator=None):
        if not self.ctx:raise ValueError('evaluate_mp requires extended precision')
        self._sync_coefficients()
        columns = [solution.evaluate_mp(points, operator) for solution in self.solutions]
        return self.ctx.matrix([[column[i] for column in columns] for i in range(columns[0].rows)]) if columns[0].rows else self.ctx.matrix(0, len(columns))

    def close(self):
        if self.system is not None:self.system.close()
    def __enter__(self):
        if self.system is not None:self.system.worker.check_open()
        return self
    def __exit__(self, *_):self.close()


def interpolate_componentwise(kernel, centers, data, precision, degree, backend=None):
    from .methods import _query
    from .nodal import interpolation
    from .global_scalar_engine import assemble_scalar_native
    from .global_backend import CppGlobalBackend
    from types import SimpleNamespace
    points = _query(centers)
    array = np.asarray(data.tolist() if hasattr(data, 'rows') else data, dtype=object)
    if array.ndim != 2 or array.shape[0] != len(points) or array.shape[1] < 2:
        raise ValueError('Multiple output samples require shape (N, C) with C > 1')
    if backend is None:
        return ComponentwiseInterpolant([interpolation(kernel, points, array[:, c], precision, degree) for c in range(array.shape[1])])
    if not isinstance(backend, CppGlobalBackend):raise TypeError('interpolate backend must be CppGlobalBackend or None')
    if not len(points) or len(np.unique(points, axis=0)) != len(points):raise ValueError('Expected nonempty distinct centers')
    a = dense_arithmetic(kernel, precision)
    values = [a.data(array[:, c], points) for c in range(array.shape[1])]
    method = SimpleNamespace(kernel=kernel, precision=precision, polynomial_degree=degree, backend=backend)
    ops = [Identity(points.shape[1])]*len(points)
    system = assemble_scalar_native(method, points, ops, ops, values[0], arithmetic=a)
    try:
        solutions = []
        for column in values:
            system.rhs = list(column)+[0]*len(system.powers)
            solutions.append(system.solve())
        return ComponentwiseInterpolant(solutions, system)
    except Exception:
        system.close()
        raise


def _samples(data, points, arithmetic):
    d = points.shape[1]
    if _component_tuple(data,d):
        components = [arithmetic.data(value, points) for value in data]
        return [components[c][i] for i in range(len(points)) for c in range(d)]
    if callable(data):
        if arithmetic.ctx:
            if not hasattr(data, 'mp_values'):
                raise TypeError('Extended vector callbacks require PrecisionData components')
            raw = data.mp_values(arithmetic.ctx, points)
        else:
            raw = data(points)
    else:
        raw = data
    array = np.asarray(raw.tolist() if hasattr(raw, 'rows') else raw, dtype=object)
    if array.shape != (len(points), d):
        raise ValueError('Vector samples must have shape (N, dimension) in node-major order')
    values = [arithmetic.number(value) for value in array.flat]
    if not all(arithmetic.ctx.isfinite(value) if arithmetic.ctx else np.isfinite(value) for value in values):
        raise ValueError('Nonfinite vector samples')
    return values


def _component_tuple(data,dimension):
    return isinstance(data,tuple) and len(data)==dimension and all(callable(value) or hasattr(value,'mp_values') or np.ndim(value)==0 for value in data)


class VectorInterpolant:
    def __init__(self, basis, coefficients, diagnostics):
        self.basis, self.coefficients, self.diagnostics = basis, coefficients, diagnostics

    def _values(self, points, operator=None):
        from .methods import _query
        b = self.basis
        points = _query(points, b.d)
        if not len(points):
            return []
        ops = bind_operators([operator or Identity(b.d)]*len(points), points)
        K = b.kernel_matrix(points, ops, b.points)
        P = b.polynomial(points, ops)
        Q = b.a.zeros(len(points)*b.d, b.n+b.m)
        Q[:, :b.n] = K
        if b.m:Q[:, b.n:] = P
        result = Q*self.coefficients if b.a.ctx else Q@self.coefficients
        return list(result)

    def evaluate(self, points, operator=None):
        return np.asarray([float(value) for value in self._values(points, operator)]).reshape(-1, self.basis.d)

    def evaluate_mp(self, points, operator=None):
        if not self.basis.a.ctx:
            raise ValueError('evaluate_mp requires extended precision')
        values = self._values(points, operator)
        d = self.basis.d
        return self.basis.a.ctx.matrix([values[i:i+d] for i in range(0, len(values), d)]) if values else self.basis.a.ctx.matrix(0, d)

    def close(self):pass
    def __enter__(self):return self
    def __exit__(self, *_):self.close()


class CppVectorInterpolant:
    def __init__(self, system, coefficients, diagnostics, dimension):
        self.system, self.coefficients, self.diagnostics, self.dimension = system, coefficients, diagnostics, dimension

    def _values(self, points, operator=None):
        from .methods import _query
        s, d = self.system, self.dimension
        s.worker.check_open()
        points = _query(points, d)
        if not len(points):return []
        ops = bind_operators([operator or Identity(d)]*len(points), points)
        if any(op.dimension!=d for op in ops):
            raise ValueError('Evaluation operator and point dimensions must match')
        query_order = max((sum(alpha) for op in ops for alpha, _ in op.terms), default=0)
        if query_order+2 > s.prepared_order:
            raise ValueError('Evaluation derivative exceeds compiled order; increase CppGlobalBackend(derivative_order=...)')
        rows = [{c: op} for op in ops for c in range(d)]
        repeated = np.repeat(points, d, axis=0)
        tokens = ['EVAL', str(len(rows)), *(s.encode(value) for value in repeated.flat)]
        for row in rows:tokens.extend(_field_tokens(row, s.encode))
        tokens.extend(s.encode(value) for value in self.coefficients)
        reply = s.worker.request(' '.join(tokens))
        if not reply or int(reply[0]) != len(rows) or len(reply) != len(rows)+1:
            raise RuntimeError('Invalid native vector evaluation result')
        return [s.number(value) for value in reply[1:]]

    def evaluate(self, points, operator=None):
        return np.asarray([float(value) for value in self._values(points, operator)]).reshape(-1, self.dimension)

    def evaluate_mp(self, points, operator=None):
        if not self.system.ctx:raise ValueError('evaluate_mp requires extended precision')
        values = self._values(points, operator)
        d = self.dimension
        return self.system.ctx.matrix([values[i:i+d] for i in range(0, len(values), d)]) if values else self.system.ctx.matrix(0, d)

    def close(self):self.system.close()
    def __enter__(self):self.system.worker.check_open();return self
    def __exit__(self, *_):self.close()


def interpolate_vector(space, centers, data, precision, backend=None):
    from .methods import _query
    from .global_backend import CppGlobalBackend
    points = _query(centers)
    if not len(points) or len(np.unique(points, axis=0)) != len(points):
        raise ValueError('Expected nonempty distinct centers')
    degree = space.degree()
    dense_arithmetic(space.kernel, precision)  # validate public precision policy
    b = _Basis(space.kernel, precision.global_digits, points, degree, True, 'physical')
    ids = [Identity(b.d)]*len(points)
    P = b.polynomial(points, ids)
    if b.m:
        check = np.asarray(P.tolist() if b.a.ctx else P, float)
        if np.linalg.matrix_rank(check) < b.m:
            raise np.linalg.LinAlgError('Polynomial space is not unisolvent')
    if backend is None:
        samples = _samples(data, points, b.a)
        G = b.a.zeros(b.n+b.m, b.n+b.m)
        G[:b.n, :b.n] = b.kernel_matrix(points, ids, points)
        if b.m:G[:b.n, b.n:] = P;G[b.n:, :b.n] = P.T
        rhs = b.a.vector(samples+[0]*b.m)
        factor = b.a.factor(G)
        coefficients = factor.solve(rhs)
        residual = b.a.norm(G*coefficients-rhs) if b.a.ctx else b.a.norm(G@coefficients-rhs)
        return VectorInterpolant(b, coefficients, {'backend': 'python', 'divergence': 'analytic', 'ordering': 'node-major', 'residual': float(residual)})
    if not isinstance(backend, CppGlobalBackend):raise TypeError('Vector interpolation backend must be CppGlobalBackend or None')
    order = max(2, backend.derivative_order+2)
    if order > 6:raise NotImplementedError('Divergence-free interpolation supports query derivative order up to 4')
    bundle = native_data_bundle(data, b.d) if _component_tuple(data,b.d) else None
    plan, program_ids = bundle if bundle is not None else (None, None)
    samples = [0]*b.n if plan is not None else _samples(data, points, b.a)
    executable, bound = _compiled(space.kernel, b.d, order, 'mpfr' if b.a.ctx else 'float64', backend, vector_only=True, data_plan=plan)
    worker = _acquire_worker(executable, backend)
    repeated = np.repeat(points, b.d, axis=0)
    rows = [{c: Identity(b.d)} for _ in points for c in range(b.d)]
    system = CppGlobalSystem(worker, repeated, rows, b.columns, samples+[0]*b.m, b.a.ctx, precision.global_digits or 17, backend, order)
    encode = system.encode
    tokens = ['INIT', str(system.digits), str(backend.threads), '1', str(b.n), str(b.m), encode(1)]
    tokens.extend(encode(value) for value in bound.values)
    tokens.extend(encode(value) for value in repeated.flat)
    tokens.extend(encode(value) for value in repeated.flat)
    tokens.extend(encode(value) for value in b.poly.origin)
    tokens.extend([encode(b.poly.scale)]*b.d)
    powers = b.poly.powers
    for column in b.columns:
        terms = [(c, coefficient, power) for c in range(b.d) for j, power in enumerate(powers) if (coefficient := column[c*len(powers)+j])]
        tokens.append(str(len(terms)))
        for component, coefficient, power in terms:tokens.extend([str(component), encode(coefficient), *(str(value) for value in power)])
    for row in rows+rows:tokens.extend(_field_tokens(row, encode))
    try:
        worker.request(' '.join(tokens))
        if plan is not None:
            worker.request('DATA_INIT '+str(b.n)+' '+' '.join(str(program_ids[c]) for _ in points for c in range(b.d)))
            reply = worker.request('RHS_DATA '+encode(0))
            if not reply or int(reply[0]) != system.size or len(reply) != system.size+1:raise RuntimeError('Invalid vector data response')
            system.rhs = [system.number(value) for value in reply[1:]]
        worker.request('BUILD')
        solved = system.solve()
        solved.diagnostics.update(divergence='analytic', ordering='node-major', data_backend='cpp_symbolic' if plan else 'python')
        return CppVectorInterpolant(system, solved.coefficients, solved.diagnostics, b.d)
    except Exception:
        worker.close(force=True)
        raise
