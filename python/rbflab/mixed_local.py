"""Mixed functional requests and maps for the common local approximation API.

Rows are scalar linear functionals, even when a public group samples a vector.
No PDE, forcing, gauge or equation assembly is assigned here.
"""
from collections.abc import Mapping
from types import SimpleNamespace
import copy
from math import factorial
import numpy as np
from scipy.sparse import csr_matrix
from .mixed import FieldOperator
from .spaces import MixedSpace, DivergenceFreeSpace, PressureSpace
from .samples import Samples, KernelTrial, sample_groups, points_array
from .nodal import Arithmetic
from .operators import Derivative, Laplacian
from .stencils import polynomial_powers, geometry_quality
from .discrete_operators import solenoidal_columns, OperatorSet
from .sparse_precision import MPSparseMatrix
from .lhi_backends import PythonBackend, CppBackend
from .torch_backend import TorchBackend


def initialize(local):
    if local.trial is not None and local.space is not None:
        raise ValueError('Specify trial or space, not both')
    if local.trial is None:
        raise ValueError('Mixed samples require explicit trial=space.representers(source)')
    trial = local.trial
    if not isinstance(trial, KernelTrial) or not isinstance(trial.space, MixedSpace):
        raise TypeError('Expected a MixedSpace representer trial')
    if trial.kind != 'representers':
        raise NotImplementedError('Mixed local approximation currently supports representers only')
    source = sample_groups(local.source)
    linked = list(source) == list(trial.source) and all(source[k] is trial.source[k] for k in source)
    if not linked:
        # Equal mathematical observations are accepted, but arbitrary mixed
        # source/trial pairing is intentionally not introduced in this milestone.
        if list(source) != list(trial.source) or any(
            not np.array_equal(source[k].points, trial.source[k].points)
            or source[k]._operators != trial.source[k]._operators
            or source[k].size != trial.source[k].size
            or source[k].target != trial.source[k].target
            or source[k].policy != trial.source[k].policy
            or repr(source[k].indices) != repr(trial.source[k].indices)
            for k in source):
            raise NotImplementedError('Mixed local approximation currently requires matching source and trial samples')
        linked = True
    local.source, local.trial = copy.deepcopy((source, trial))
    local._linked = linked
    preflight(local)


def preflight(local):
    from .precision import Precision
    from .configuration import LocalSolver
    from .stencils import StencilPolicy
    from .mixed_backends import validate_execution
    if not isinstance(local.backend, (PythonBackend, CppBackend)):
        if isinstance(local.backend, TorchBackend):
            raise NotImplementedError('Mixed local approximation does not support Torch yet')
        raise TypeError('Expected PythonBackend or CppBackend for mixed approximation')
    if not isinstance(local.precision, Precision) or local.precision.global_digits is not None:
        raise ValueError('Local approximation uses local_digits and global_dtype')
    if not isinstance(local.local_solver, LocalSolver) or local.local_solver.method != 'lu':
        raise NotImplementedError('Mixed local approximation currently supports LU')
    if not isinstance(local.stencil_policy, StencilPolicy) or local.stencil_policy.shape_rule != 'fixed':
        raise NotImplementedError('Mixed local approximation uses fixed kernel parameters')
    if local.backend.shape_rule != 'fixed' or local.backend.local_solver != 'lu':
        raise NotImplementedError('Mixed local approximation uses fixed parameters and LU')
    validate_execution(local.backend, getattr(local, 'execution', None))
    d = next(iter(local.source.values())).points.shape[1]
    for group in local.source.values():
        ops = group._operators or ((group.operator,) if isinstance(group.operator, FieldOperator) else ())
        if not ops or any(not isinstance(op, FieldOperator) for op in ops):
            raise ValueError('Mixed samples require explicit FieldOperator field binding')
        if len({op.arity for op in ops}) != 1:
            raise ValueError('Each sample group must have a fixed output arity')
    for space in local.trial.space.spaces.values():
        space.degree()
    return dict(method='LocalApproximation', backend=type(local.backend).__name__,
                trial='representers', groups=tuple(local.source),
                fields=tuple(local.trial.space.spaces), precision=local.precision,
                dimension=d)


def _fields(local, d):
    result, master = [], None
    for name, space in local.trial.space.spaces.items():
        a = Arithmetic(space.kernel, local.precision.local_digits)
        if master is None:
            master = a
        elif master.ctx:
            a.ctx = master.ctx
            a.backend.ctx = master.ctx
        vector = isinstance(space, DivergenceFreeSpace)
        result.append(SimpleNamespace(name=name, space=copy.deepcopy(space), kernel=copy.deepcopy(space.kernel),
            arithmetic=a, vector=vector, components=d if vector else 1, degree=space.degree(),
            quotient_pressure=isinstance(space, PressureSpace)))
    return result, master


def _columns(fields, d):
    columns = []
    for f, field in enumerate(fields):
        if field.degree is None:
            continue
        powers = polynomial_powers(d, field.degree)
        if not field.vector:
            if getattr(field, 'quotient_pressure', False):
                powers = [power for power in powers if any(power)]
            columns.extend([(f, [(0, 1, power)]) for power in powers])
        elif d == 2:
            # Curls of normalized stream-function monomials span all 2D
            # solenoidal vector polynomials, including the two translations.
            # Keep the canonical stream-monomial order so old local methods
            # can reuse their polynomial coordinates without a basis change.
            for x, y in polynomial_powers(2, field.degree+1)[1:]:
                terms = []
                if y:
                    terms.append((0, y, (x, y-1)))
                if x:
                    terms.append((1, -x, (x-1, y)))
                columns.append((f, terms))
        else:
            n = len(powers)
            for column in solenoidal_columns(d, field.degree):
                columns.append((f, [(c, column[c*n+j], power)
                    for c in range(d) for j, power in enumerate(powers) if column[c*n+j]]))
    return columns


def polynomial(a, points, rows, columns, origin, radius):
    out = a.zeros(len(rows), len(columns))
    for i, (point, row) in enumerate(zip(points, rows)):
        z = [(a.number(v)-a.number(o))/radius for v, o in zip(point, origin)]
        for j, (field, terms) in enumerate(columns):
            for component, coefficient, power in terms:
                op = row.get((field, component))
                if op is None:
                    continue
                for alpha, value in op.terms:
                    if any(k > p for k, p in zip(alpha, power)):
                        continue
                    v = a.number(coefficient)*a.number(value)/radius**sum(alpha)
                    for x, p, k in zip(z, power, alpha):
                        v *= factorial(p)//factorial(p-k)*x**(p-k)
                    out[i, j] += v
    return out


def kernel(owner, x, left, y, right, scale):
    """Contract scalar mixed functionals with the direct-sum field kernel."""
    out = owner.arithmetic.zeros(len(left), len(right))
    d = owner.dimension
    for f, field in enumerate(owner.fields):
        for c in range(field.components):
            ii = [i for i, row in enumerate(left) if (f, c) in row]
            if not ii:
                continue
            for b in range(field.components):
                jj = [j for j, row in enumerate(right) if (f, b) in row]
                if not jj:
                    continue
                lhs = [left[i][f, c] for i in ii]
                if field.vector:
                    transform = Derivative(c, d) @ Derivative(b, d)
                    if c == b:
                        transform = transform - Laplacian(d)
                    lhs = [op @ transform for op in lhs]
                block = field.arithmetic.matrix(x[ii], lhs, y[jj], [right[j][f, b] for j in jj], scale)
                for i, ir in enumerate(ii):
                    for j, jc in enumerate(jj):
                        out[ir, jc] += block[i, j]
    return out


def arrays(owner, job):
    """Build physical augmented matrix and all target right-hand sides."""
    a = owner.arithmetic
    n, m = job['size'], job['polynomials']
    k = len(job['target_rows'])
    G, Q = a.zeros(n+m, n+m), a.zeros(n+m, k)
    G[:n, :n] = kernel(owner, job['source_points'], job['source_rows'],
                      job['trial_points'], job['trial_rows'], job['scale'])
    Q[:n, :] = kernel(owner, job['target_points'], job['target_rows'],
                     job['trial_points'], job['trial_rows'], job['scale']).T
    if m:
        G[:n, n:] = job['top']
        G[n:, :n] = job['bottom'].T
        Q[n:, :] = job['target_poly'].T
    return G, Q


def _signature(point, row):
    return tuple(point), tuple((field, comp, op.terms) for (field, comp), op in sorted(row.items()))


def _flatten(groups, selected, fields, layout):
    points, rows, ids = [], [], []
    for name, group in groups.items():
        arity = layout[name]['components']
        for node in selected[name]:
            op = group._operators[int(node)]
            r = op.scalar_rows(fields)
            for component, scalar in enumerate(r):
                points.append(group.points[node])
                rows.append(scalar)
                ids.append(layout[name]['column_slice'].start+int(node)*arity+component)
    points = np.asarray(points, dtype=float).reshape(-1, next(iter(groups.values())).points.shape[1])
    signatures = [_signature(p, row) for p, row in zip(points, rows)]
    if any(not row for row in rows):
        raise ValueError('A zero functional cannot be used as a source or trial sample')
    if len(set(signatures)) != len(signatures):
        raise ValueError('Duplicate sampled functional in a local stencil')
    return points, rows, np.asarray(ids, dtype=int)


def _smoothness(fields, left, right, targets):
    from .kernels import PHS, Hybrid
    from .symbolic_kernel import BoundKernel, validate_support
    for f, field in enumerate(fields):
        order = lambda rows: max((sum(alpha) for row in rows for (ff, _), op in row.items()
                                 if ff == f for alpha, coeff in op.terms), default=0)
        required = max(order(left), order(targets)) + order(right) + (2 if field.vector else 0)
        phs = field.kernel.phs if isinstance(field.kernel, Hybrid) else field.kernel if isinstance(field.kernel, PHS) else None
        if required > 6:
            raise NotImplementedError('Mixed kernels currently support derivatives through order six')
        if phs is not None and phs.power-1 < required:
            raise ValueError(f'Kernel requires continuous coincident derivatives through order {required}')
        if isinstance(field.kernel, BoundKernel):
            validate_support(field.kernel.family, required)


def _quotient_functionals(fields, *collections):
    """Reject any functional that changes under a quotient-field constant."""
    quotient = {f for f, field in enumerate(fields)
                if getattr(field, 'quotient_pressure', False)}
    for rows in collections:
        for row in rows:
            for (field, component), op in row.items():
                if field in quotient and any(value and not any(alpha)
                                             for alpha, value in op.terms):
                    raise ValueError(
                        f'PressureSpace field {fields[field].name!r} represents pressure '
                        'modulo constants; source, trial and target functionals must '
                        'annihilate constants. Use ScalarSpace for absolute pressure values.')


def _functional_ratio(matrix, columns):
    """Float64 geometry screening of the actual mixed polynomial functionals."""
    if not columns:
        return 1.
    p = np.asarray(matrix.tolist() if hasattr(matrix, 'rows') else matrix, dtype=float)
    if p.shape[0] < columns or not np.isfinite(p).all():
        return 0.
    row = np.max(abs(p), axis=1)
    p = p/np.where(row > 0, row, 1)[:, None]
    col = np.linalg.norm(p, axis=0)
    p = p/np.where(col > 0, col, 1)[None, :]
    singular = np.linalg.svd(p, compute_uv=False)
    return float(singular[-1]/singular[0]) if singular[0] else 0.


def _select_joint(local, target, index, fields, layout, columns, arithmetic):
    """Grow only quality-controlled groups against their joint polynomial map.

    A group's functionals need not independently span a scalar polynomial
    space: velocity and momentum observations can supply complementary rows.
    Fixed memberships, all-point groups and nearest groups are never enlarged.
    """
    from .local_approximation import _rank
    selected, controlled, candidates = {}, {}, {}
    for name, group in local.source.items():
        policy = group.policy or local.stencil_policy
        if policy.selection != 'quality' or group.indices is not None:
            selected[name] = group.select(target, index, local.stencil_policy, 0)
            continue
        controlled[name] = policy
        coincident = set(group._locations.get(tuple(target), ()))
        excluded = coincident if group.target == 'exclude' else set()
        available = len(group.points)-len(excluded)
        size = available if group.size is None else group.size
        if size > available:
            raise ValueError('Sample size exceeds eligible points after target filtering')
        if group.size is None or size == 0:
            selected[name] = group.select(target, index, local.stencil_policy, 0)
            continue
        limit = policy.max_size if policy.max_size is not None else min(available, 2*size)
        if not size <= limit <= available:
            raise ValueError('max_size must lie between sample size and eligible point count')
        # Include every distance tie at the upper bound, then use source index
        # as the stable secondary key. Excluded owners cannot consume a slot.
        distance = np.atleast_1d(group._tree.query(target, k=limit+len(excluded))[0])[-1]
        ids = np.asarray(group._tree.query_ball_point(target, np.nextafter(float(distance), np.inf)), dtype=int)
        ids = np.asarray([j for j in ids if int(j) not in excluded], dtype=int)
        distances = np.linalg.norm(group.points[ids]-target, axis=1)
        candidates[name] = ids[np.lexsort((ids, distances))][:limit]
        selected[name] = candidates[name][:size]
        if group.target == 'require' and not any(int(j) in coincident for j in selected[name]):
            raise ValueError('Required target is absent from selected samples')
    threshold = max((policy.min_polynomial_ratio for policy in controlled.values()), default=0.)
    sqrt = arithmetic.ctx.sqrt if arithmetic.ctx else np.sqrt
    while True:
        points, rows, global_ids = _flatten(local.source, selected, fields, layout)
        if not len(rows):
            raise ValueError('Each local system needs nonzero source and trial functional counts')
        radius = max(sqrt(sum((arithmetic.number(v)-arithmetic.number(o))**2
                             for v, o in zip(p, target))) for p in points)
        radius = radius or arithmetic.number(1)
        matrix = polynomial(arithmetic, points, rows, columns, target, radius)
        if not controlled:
            return selected, points, rows, global_ids, radius, matrix
        ratio = _functional_ratio(matrix, len(columns))
        try:
            _rank(matrix, len(columns))
            unisolvent = True
        except np.linalg.LinAlgError:
            unisolvent = False
        separation = {}
        for name in controlled:
            sites = np.unique(local.source[name].points[selected[name]], axis=0)
            # Coincident, distinct Hermite observations are legitimate; a
            # single physical site has no pair-separation constraint.
            separation[name] = (geometry_quality(sites, target, 0)['separation_ratio']
                                if len(sites) > 1 else float('inf'))
        if unisolvent and ratio >= threshold and all(separation[name] >= policy.min_separation_ratio
                                      for name, policy in controlled.items()):
            return selected, points, rows, global_ids, radius, matrix
        growing = [name for name, ids in candidates.items() if len(selected[name]) < len(ids)]
        if not growing:
            limits = {name: len(ids) for name, ids in candidates.items()}
            raise np.linalg.LinAlgError(
                f'Mixed stencil quality failed within max_size limits {limits}: '
                f'functional_polynomial_ratio={ratio}, separation_ratios={separation}')
        for name in growing:
            selected[name] = candidates[name][:len(selected[name])+1]


def operators(local, targets, operators):
    from .local_approximation import _rank
    from .mixed_backends import execute
    preflight(local)
    if not isinstance(operators, Mapping) or not operators or any(not isinstance(k, str) or not k for k in operators):
        raise ValueError('operators must be a nonempty named mapping')
    if any(not isinstance(op, FieldOperator) for op in operators.values()):
        raise ValueError('Mixed targets require explicit FieldOperator field binding')
    d = next(iter(local.source.values())).points.shape[1]
    query = points_array(targets, d)
    fields, a = _fields(local, d)
    layout, offset = {}, 0
    for name, group in local.source.items():
        arity = (group._operators[0] if group._operators else group.operator).arity
        layout[name] = dict(points=group.points.copy(), components=arity, arity=arity,
                            column_slice=slice(offset, offset+len(group.points)*arity), ordering='node-major')
        offset += len(group.points)*arity
        if group.indices is not None and len(group.indices) != len(query):
            raise ValueError('One membership row is required per target')
    target_layout, rhs_offset = {}, 0
    bound_targets = {name: Samples(targets, operator=op)._operators for name, op in operators.items()}
    for name, op in operators.items():
        target_layout[name] = dict(components=op.arity, arity=op.arity,
            column_slice=slice(rhs_offset, rhs_offset+op.arity), ordering='node-major')
        rhs_offset += op.arity
    owner = SimpleNamespace(fields=fields, arithmetic=a, dimension=d, source=copy.deepcopy(local.source),
        precision=copy.deepcopy(local.precision), backend=copy.deepcopy(local.backend),
        execution=copy.deepcopy(getattr(local, 'execution', None)),
        targets=query.copy(), layout=layout, target_layout=target_layout, jobs=[], indices=[],
        memberships=[], trial_memberships=[], diagnostics=[])
    columns = _columns(fields, d)
    for i, target in enumerate(query):
        ids, x, rows, global_ids, radius, P = _select_joint(
            local, target, i, fields, layout, columns, a)
        target_rows = []
        for name in operators:
            target_rows.extend(bound_targets[name][i].scalar_rows(fields))
        _quotient_functionals(fields, rows, target_rows)
        _smoothness(fields, rows, rows, target_rows)
        _rank(P, len(columns))
        tx = np.repeat(target[None, :], len(target_rows), axis=0)
        TP = polynomial(a, tx, target_rows, columns, target, radius)
        owner.jobs.append(dict(source_points=x, source_rows=rows, trial_points=x.copy(), trial_rows=copy.deepcopy(rows),
            target_points=tx, target_rows=target_rows, top=P, bottom=P.copy(), target_poly=TP,
            size=len(rows), polynomials=len(columns),
            scale=radius if local.stencil_policy.scaling == 'local' else a.number(1),
            polynomial_origin=target.copy(), polynomial_scale=radius))
        owner.indices.append(global_ids)
        owner.memberships.append(copy.deepcopy(ids))
        owner.trial_memberships.append(copy.deepcopy(ids))
    results = execute(owner)
    owner.weights = [item['weights'] for item in results]
    owner.diagnostics = [dict(weight_residual=item['residual'], scaled_condition=item['condition'], factorizations=1,
        geometry=geometry_quality(np.unique(job['source_points'], axis=0), target, 1))
        for item, job, target in zip(results, owner.jobs, query)]
    result = {}
    for name, out_layout in target_layout.items():
        arity, selection = out_layout['arity'], out_layout['column_slice']
        data, rr, cc = [], [], []
        for i, (ids, weights) in enumerate(zip(owner.indices, owner.weights)):
            for j, col in enumerate(ids):
                for c in range(arity):
                    rr.append(i*arity+c); cc.append(int(col)); data.append(weights[j, selection.start+c])
        if local.precision.global_dtype == 'mpmath':
            sparse_rows = [{} for _ in range(len(query)*arity)]
            for row, col, value in zip(rr, cc, data):
                sparse_rows[row][col] = sparse_rows[row].get(col, 0)+value
            matrix = MPSparseMatrix(a.ctx, sparse_rows, ncols=offset)
        else:
            values = np.asarray([float(v) for v in data])
            if not np.isfinite(values).all():
                raise FloatingPointError('Weights overflow Float64 storage')
            matrix = csr_matrix((values, (rr, cc)), shape=(len(query)*arity, offset))
        blocks = {key: MixedSampleMap(_slice(matrix, value['column_slice']), value['components'],
                    group.points, arity, query) for (key, value), group in zip(layout.items(), local.source.values())}
        result[name] = MixedFunctionalOperator(matrix, blocks, owner, name)
    diagnostics = dict(backend=type(local.backend).__name__,
        factorizations=getattr(owner, 'execution_info', {}).get('unique_jobs', len(query)),
        requested_stencils=len(query), rhs_per_stencil=rhs_offset,
        fields=tuple(local.trial.space.spaces), groups=tuple(local.source), ordering='node-major',
        local_digits=local.precision.local_digits, global_dtype=local.precision.global_dtype,
        input_arities={name: item['arity'] for name, item in layout.items()},
        output_arities={name: item['arity'] for name, item in target_layout.items()})
    if hasattr(owner, 'execution_info'):
        diagnostics['execution'] = owner.execution_info
    return OperatorSet(result, diagnostics)


def _slice(matrix, columns):
    from .local_approximation import _slice_columns
    return _slice_columns(matrix, columns)


class MixedSampleMap:
    """One input group with independent input and output component counts."""
    def __init__(self, matrix, components, points, output_components, targets):
        self.matrix, self.shape = matrix, matrix.shape
        self.components = self.input_components = components
        self.output_components = output_components
        self.points, self.targets = points.copy(), targets.copy()

    def __matmul__(self, values):
        mp = isinstance(self.matrix, MPSparseMatrix)
        v = np.asarray(values.tolist() if hasattr(values, 'rows') else values, dtype=object if mp else float)
        n = len(self.points)
        if v.shape == (n, self.input_components):
            v = v.reshape(-1)
        if v.ndim != 1 or v.shape[0] != self.shape[1]:
            raise ValueError('Input must match source group DOFs')
        out = self.matrix @ v
        shape = (len(self.targets), self.output_components)
        if mp:
            return self.matrix.ctx.matrix(np.asarray(out.tolist(), object).reshape(shape).tolist())
        return np.asarray(out).reshape(shape)

    def to_scipy(self):
        from .local_approximation import SampleMap
        return SampleMap(self.matrix, self.components, self.points).to_scipy()


class MixedFunctionalOperator(MixedSampleMap, Mapping):
    def __init__(self, matrix, blocks, owner, name):
        points = np.concatenate([g.points for g in owner.source.values()])
        super().__init__(matrix, 1, points, owner.target_layout[name]['arity'], owner.targets)
        self._blocks, self._owner, self._name = blocks, owner, name
        self.input_layout = copy.deepcopy(owner.layout)
        self.output_layout = dict(arity=self.output_components, components=self.output_components,
                                  points=owner.targets.copy(), ordering='node-major')

    def __getitem__(self, name):
        return self._blocks[name]

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __matmul__(self, values):
        if not isinstance(values, Mapping):
            return super().__matmul__(values)
        if set(values) != set(self._blocks):
            raise ValueError('Supply exactly one value array per named source group')
        outputs = [block @ values[name] for name, block in self._blocks.items()]
        out = outputs[0]
        for value in outputs[1:]:
            out = out+value
        return out

    def local(self, i):
        o = self._owner
        if type(i) is not int or not 0 <= i < len(o.jobs):
            raise IndexError('Target index out of range')
        s = o.target_layout[self._name]['column_slice']
        n = o.jobs[i]['size']
        return SimpleNamespace(indices=o.indices[i].copy(), target=o.targets[i].copy(),
            weights=o.weights[i][:n, s].copy(), multipliers=o.weights[i][n:, s].copy(),
            groups=copy.deepcopy(o.memberships[i]), trial_groups=copy.deepcopy(o.trial_memberships[i]),
            diagnostics=copy.deepcopy(o.diagnostics[i]), ordering='node-major',
            global_dtype=o.precision.global_dtype)

    def reconstruct_local(self, i):
        from .mixed_backends import reconstruct
        self.local(i)
        o = self._owner
        G, Q = reconstruct(o, i)
        Q = Q[:, o.target_layout[self._name]['column_slice']]
        if o.arithmetic.ctx:
            ctx = o.arithmetic.ctx
            scale = [1/ctx.sqrt(max(abs(G[j, k]) for k in range(G.cols))) for j in range(G.rows)]
            scaled = ctx.matrix([[scale[j]*G[j, k]*scale[k] for k in range(G.cols)] for j in range(G.rows)])
            rhs = ctx.matrix([[scale[j]*Q[j, k] for k in range(Q.cols)] for j in range(Q.rows)])
        else:
            scale = 1/np.sqrt(np.max(abs(G), axis=1))
            scaled, rhs = scale[:, None]*G*scale[None, :], scale[:, None]*Q
        return SimpleNamespace(matrix=G, rhs=Q, scaled_matrix=scaled, scaled_rhs=rhs, scale=scale,
            kernel_scale=o.jobs[i]['scale'], polynomial_origin=o.jobs[i]['polynomial_origin'].copy(),
            polynomial_scale=o.jobs[i]['polynomial_scale'], equation='matrix.T @ augmented_weights = rhs',
            backend=type(o.backend).__name__, local_digits=o.precision.local_digits,
            source_indices=o.indices[i].copy(), groups=copy.deepcopy(o.memberships[i]))
