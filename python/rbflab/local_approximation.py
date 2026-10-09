"""Matrix-first local functional approximation shared by RBF-FD and LHI.

The engine fits functionals to an explicit kernel trial basis and returns
named sparse maps. It does not assign PDE roles, impose equations, or choose
time integration. Python, C++, and Torch consume the same local requests.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
import copy
import numpy as np
from scipy.sparse import csr_matrix
from .samples import Samples, KernelTrial, sample_groups, points_array
from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace
from .operators import Identity, bind_operators
from .precision import Precision
from .configuration import LocalSolver
from .stencils import StencilPolicy, polynomial_powers, geometry_quality
from .nodal import Arithmetic, NodalBasis
from .discrete_operators import OperatorSet, solenoidal_columns
from .sparse_precision import MPSparseMatrix
from .lhi_backends import PythonBackend, CppBackend
from .torch_backend import TorchBackend


def _tensor(value):
    return type(value).__module__.startswith('torch')


def _copy(value):
    return value.clone() if _tensor(value) else value.copy()


def _rank(matrix, columns):
    if not columns:
        return
    p = np.asarray(matrix.tolist() if hasattr(matrix, 'rows') else matrix, dtype=float)
    row = np.max(abs(p), axis=1)
    p = p / np.where(row > 0, row, 1)[:, None]
    col = np.max(abs(p), axis=0)
    p = p / np.where(col > 0, col, 1)[None, :]
    if not np.isfinite(p).all() or np.linalg.matrix_rank(p) < columns:
        raise np.linalg.LinAlgError('Polynomial functionals are not unisolvent; change or enlarge sample groups')


def _polynomials(owner, points):
    # Reuse shifted monomials without assuming distinct *geometric* centers:
    # Hermite functionals may occupy the same point.
    from .kernels import IMQ
    a = owner.arithmetic
    poly = NodalBasis(Arithmetic(IMQ(), owner.precision.local_digits), points,
                      source_operators=[Identity(owner.dimension)] * len(points))
    poly.powers = [] if owner.degree is None else polynomial_powers(owner.dimension, owner.degree)
    if a.ctx:
        poly.arithmetic.ctx = a.ctx
        poly.arithmetic.backend.ctx = a.ctx
    columns = solenoidal_columns(owner.dimension, owner.degree) if owner.vector else None
    m = len(columns) if owner.vector else len(poly.powers)
    def evaluate(x, ops):
        p = poly.polynomials(x, ops)
        if not owner.vector:
            return p
        out = a.zeros(len(x)*owner.components, m)
        count = len(poly.powers)
        for i in range(len(x)):
            for c in range(owner.components):
                for k, column in enumerate(columns):
                    out[i*owner.components+c, k] = sum(
                        p[i, j]*a.number(column[c*count+j]) for j in range(count))
        return out
    return evaluate, m, poly.kernel_scale


def _flat(groups, selected, *, translates=False):
    points, ops = [], []
    for name, group in groups.items():
        ids = selected[name]
        points.extend(group.points[ids])
        ops.extend([Identity(group.points.shape[1])] * len(ids) if translates
                   else [group._operators[int(i)] for i in ids])
    return np.asarray(points, dtype=float), ops


def _check_distinct(points, ops):
    keys = [(tuple(x), op.terms) for x, op in zip(points, ops)]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate sampled functional in a local stencil')
    if any(not op.terms for op in ops):
        raise ValueError('A zero functional cannot be used as a source or trial sample')


class SampleMap:
    """One named sparse block mapping a source group to target quantities.

    matrix retains backend storage: SciPy CSR, Torch sparse COO, or the
    arbitrary-precision MPSparseMatrix. to_scipy() is an explicit Float64
    export (and detaches Torch tensors). Vector values use node-major order.
    """
    def __init__(self, matrix, components, points):
        self.matrix, self.components, self.points = matrix, components, points.copy()
        self.shape = matrix.shape

    def __matmul__(self, values):
        c, n = self.components, len(self.points)
        if _tensor(self.matrix):
            import torch
            v = values if _tensor(values) else torch.as_tensor(values, dtype=self.matrix.dtype, device=self.matrix.device)
            shaped = c > 1 and tuple(v.shape) == (n, c)
            if shaped:
                v = v.reshape(-1)
            if v.ndim not in (1, 2) or v.shape[0] != self.shape[1]:
                raise ValueError('Input must match source DOFs')
            out = self.matrix @ v
            return out.reshape(-1, c) if shaped else out
        mp = isinstance(self.matrix, MPSparseMatrix)
        v = np.asarray(values.tolist() if hasattr(values, 'rows') else values,
                       dtype=object if mp else float)
        shaped = c > 1 and v.shape == (n, c)
        if shaped:
            v = v.reshape(-1)
        if v.ndim not in (1, 2) or v.shape[0] != self.shape[1]:
            raise ValueError('Input must match source DOFs')
        out = self.matrix @ v
        if shaped:
            if mp:
                return self.matrix.ctx.matrix(np.asarray(out.tolist(), object).reshape(-1, c).tolist())
            return out.reshape(-1, c)
        return out

    def to_scipy(self):
        """Explicitly export a detached/rounded Float64 CSR matrix."""
        if _tensor(self.matrix):
            m = self.matrix.coalesce().detach().cpu()
            ids = m.indices().numpy()
            return csr_matrix((m.values().numpy(), (ids[0], ids[1])), shape=self.shape)
        if isinstance(self.matrix, MPSparseMatrix):
            triples = [(i, j, float(v)) for i, row in enumerate(self.matrix.rows) for j, v in row.items()]
            return csr_matrix(([v for i, j, v in triples],
                               ([i for i, j, v in triples], [j for i, j, v in triples])), shape=self.shape)
        return self.matrix.copy()


class FunctionalOperator(SampleMap, Mapping):
    """A target operator with named source blocks and inspectable local algebra."""
    def __init__(self, matrix, blocks, owner, column):
        points = np.concatenate([g.points for g in owner.source.values()])
        super().__init__(matrix, owner.components, points)
        self._blocks, self._owner, self._column = blocks, owner, column
        self.input_layout = copy.deepcopy(owner.layout)
        self.targets = owner.targets.copy()

    def __getitem__(self, name):
        return self._blocks[name]

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __matmul__(self, values):
        if isinstance(values, Mapping):
            if set(values) != set(self._blocks):
                raise ValueError('Supply exactly one value array per named source group')
            outputs = [block @ values[name] for name, block in self._blocks.items()]
            out = outputs[0]
            for item in outputs[1:]:
                out = out + item
            return out
        return super().__matmul__(values)

    def local(self, i):
        """Copy local weights, multipliers, group membership and diagnostics."""
        o = self._owner
        if type(i) is not int or not 0 <= i < len(o.jobs):
            raise IndexError('Target index out of range')
        job = o.jobs[i]
        c = o.components
        k = self._column*c
        w = o.weights[i]
        n = job['size']
        return SimpleNamespace(indices=o.indices[i].copy(), target=o.targets[i].copy(),
            weights=_copy(w[:n, k:k+c]), multipliers=_copy(w[n:, k:k+c]),
            groups=copy.deepcopy(o.memberships[i]), trial_groups=copy.deepcopy(o.trial_memberships[i]),
            diagnostics=copy.deepcopy(o.diagnostics[i]), ordering='node-major',
            global_dtype=o.precision.global_dtype)

    def reconstruct_local(self, i):
        """Rebuild A and q for A.T @ augmented_weights = q, on demand.

        Physical and equilibrated matrices are returned separately. Kernel
        coordinate scaling is already included in A; scale below is algebraic.
        """
        from .functional_backends import reconstruct
        self.local(i)
        o = self._owner
        g, q = reconstruct(o, i)
        k, c = self._column*o.components, o.components
        q = q[:, k:k+c]
        if _tensor(g):
            s = g.abs().amax(dim=1).rsqrt()
            scaled, rhs = s[:, None]*g*s[None, :], s[:, None]*q
        elif o.arithmetic.ctx:
            ctx = o.arithmetic.ctx
            s = [1/ctx.sqrt(max(abs(g[j, l]) for l in range(g.cols))) for j in range(g.rows)]
            scaled = ctx.matrix([[s[j]*g[j, l]*s[l] for l in range(g.cols)] for j in range(g.rows)])
            rhs = ctx.matrix([[s[j]*q[j, l] for l in range(q.cols)] for j in range(q.rows)])
        else:
            s = 1/np.sqrt(np.max(abs(g), axis=1))
            scaled, rhs = s[:, None]*g*s[None, :], s[:, None]*q
        return SimpleNamespace(matrix=g, rhs=q, scaled_matrix=scaled, scaled_rhs=rhs,
            scale=s, kernel_scale=o.jobs[i]['scale'], equation='matrix.T @ augmented_weights = rhs',
            backend=type(o.backend).__name__, local_digits=o.precision.local_digits,
            source_indices=o.indices[i].copy(), groups=copy.deepcopy(o.memberships[i]))


@dataclass
class LocalApproximation:
    """Construct functional interpolation weights and named sparse maps.

    Args:
        source: Named Samples groups, a Samples object, or raw points.
        trial: space.representers(groups) or space.translates(points).
            Mixed-functional sources require an explicit trial description.
        space: Shortcut for ordinary value interpolation; omit with trial.
        stencil_size: Neighbor count for a raw point source (default 15).
            Named Samples own their sizes; None means all eligible points.
        backend: PythonBackend, CppBackend or TorchBackend.
        precision: Local arithmetic and sparse storage policy.
        stencil_policy: Default selection and common kernel scaling.
        local_solver: Local factorization (currently LU).

    This object has no forcing or PDE solve. Use operators() or weights(), then
    assemble your own equations. Full MP sparse maps preserve precision;
    Torch results stay tensors. Geometry and membership are nondifferentiable
    Float64 data, and kernel parameter autograd is not advertised here.
    """
    source: object
    trial: object = None
    space: object = None
    stencil_size: int | None = 15
    backend: object = field(default_factory=PythonBackend)
    precision: Precision = field(default_factory=Precision)
    stencil_policy: StencilPolicy = field(default_factory=StencilPolicy)
    local_solver: LocalSolver = field(default_factory=LocalSolver)

    def __post_init__(self):
        if self.trial is not None and self.space is not None:
            raise ValueError('Specify trial or space, not both')
        raw = not isinstance(self.source, (Mapping, Samples))
        size = self.stencil_size
        if raw and size is not None:
            p = points_array(self.source)
            if type(size) is not int or size < 1:
                raise ValueError('stencil_size must be positive or None')
            size = min(size, len(p))
        source = sample_groups(self.source, size=size if raw else None)
        trial = self.trial
        if trial is None:
            if not isinstance(self.space, ScalarSpace):
                raise TypeError('Provide an explicit trial or a ScalarSpace')
            if any(op != Identity(g.points.shape[1]) for g in source.values() for op in g._operators):
                raise ValueError('Functional samples require an explicit trial, e.g. space.representers(source)')
            trial = KernelTrial(self.space, source)
        if not isinstance(trial, KernelTrial) or not isinstance(trial.space, ScalarSpace):
            raise TypeError('trial must come from space.representers() or space.translates()')
        # Preserve shared memberships for matching Hermite trial/data layouts.
        linked = list(source) == list(trial.source) and all(source[k] is trial.source[k] for k in source)
        if trial.kind == 'translates' and len(source) == len(trial.source) == 1:
            tg = next(iter(trial.source.values()))
            linked = (tg.size is None and tg.indices is None and tg.policy is None and tg.target == 'allow'
                      and np.array_equal(next(iter(source.values())).points, tg.points))
        self.source, self.trial = copy.deepcopy((source, trial))
        self._linked = linked
        self.preflight()

    def preflight(self):
        """Validate configuration before kernel compilation or matrix assembly."""
        if not isinstance(self.backend, (PythonBackend, CppBackend, TorchBackend)):
            raise TypeError('Expected PythonBackend, CppBackend or TorchBackend')
        if not isinstance(self.precision, Precision) or self.precision.global_digits is not None:
            raise ValueError('Local approximation uses local_digits and global_dtype')
        if not isinstance(self.stencil_policy, StencilPolicy) or self.stencil_policy.shape_rule != 'fixed':
            raise NotImplementedError('Use fixed kernel parameters; Hardy policies remain on compatibility Stokes LHI')
        if not isinstance(self.local_solver, LocalSolver) or self.local_solver.method != 'lu':
            raise NotImplementedError('Functional approximation currently supports LU local solves')
        if isinstance(self.backend, TorchBackend) and (self.precision.local_digits is not None or self.precision.global_dtype != 'float64'):
            raise NotImplementedError('Torch functional approximation requires Float64')
        self.trial.space.degree()
        if next(iter(self.source.values())).points.shape[1] != next(iter(self.trial.source.values())).points.shape[1]:
            raise ValueError('Source and trial dimensions differ')
        return dict(method='LocalApproximation', backend=type(self.backend).__name__,
                    trial=self.trial.kind, groups=tuple(self.source), precision=self.precision)

    def weights(self, *, target, operator):
        """Build one target row using configured sample memberships.

        Use Samples(size=None) for an explicit stencil containing all supplied
        centers. Returned data expose weights, multipliers, groups, and a
        zero-argument reconstruct_local().
        """
        op = self.operators(targets=np.asarray(target)[None, :], operators={'value': operator}).value
        result = op.local(0)
        result.reconstruct_local = lambda: op.reconstruct_local(0)
        return result

    def operators(self, *, targets, operators):
        """Build named source-block maps, sharing one factorization per target.

        operators maps user names to differential operators. Normal-dependent
        target operators require targets to carry matching normals. Members
        are chosen once per target and shared by every requested operator.
        """
        from .functional_backends import execute
        self.preflight()
        if not isinstance(operators, Mapping) or not operators or any(not isinstance(k, str) or not k for k in operators):
            raise ValueError('operators must be a nonempty named mapping')
        d = next(iter(self.source.values())).points.shape[1]
        query = points_array(targets, d)
        descriptor = self.trial.space
        degree = descriptor.degree()
        if isinstance(descriptor, PressureSpace) and degree is None:
            degree = 0
        vector = isinstance(descriptor, DivergenceFreeSpace)
        c = d if vector else 1
        owner = SimpleNamespace(kernel=copy.deepcopy(descriptor.kernel), precision=copy.deepcopy(self.precision),
            backend=copy.deepcopy(self.backend), arithmetic=Arithmetic(descriptor.kernel, self.precision.local_digits),
            dimension=d, components=c, vector=vector, degree=degree, source=copy.deepcopy(self.source),
            targets=query.copy(), jobs=[], memberships=[], trial_memberships=[], indices=[], layout={})
        offset = 0
        for name, group in self.source.items():
            owner.layout[name] = dict(points=group.points.copy(), components=c,
                                      column_slice=slice(offset*c, (offset+len(group.points))*c))
            offset += len(group.points)
        for g in list(self.source.values()) + list(self.trial.source.values()):
            if g.indices is not None and len(g.indices) != len(query):
                raise ValueError('One membership row is required per target')
        target_ops = {name: Samples(targets, operator=op)._operators for name, op in operators.items()}
        for row, target in enumerate(query):
            # Geometric screening of separate derivative groups is not a test
            # of functional unisolvency. Check the assembled polynomial blocks.
            geometric_degree = degree if len(self.source) == 1 and all(op == Identity(d) for op in next(iter(self.source.values()))._operators) else 0
            ids = {name: g.select(target, row, self.stencil_policy, geometric_degree) for name, g in self.source.items()}
            if self._linked:
                trial_ids = dict(zip(self.trial.source, ids.values()))
            else:
                trial_ids = {name: g.select(target, row, self.stencil_policy, 0) for name, g in self.trial.source.items()}
            x, left = _flat(self.source, ids)
            z, right = _flat(self.trial.source, trial_ids, translates=self.trial.kind == 'translates')
            if not len(x) or len(x) != len(z):
                raise ValueError('Each local system needs equal nonzero source and trial functional counts')
            _check_distinct(x, left)
            _check_distinct(z, right)
            req = [target_ops[name][row] for name in operators]
            order = max(sum(alpha) for op in left+req for alpha, coeff in op.terms) + max(sum(alpha) for op in right for alpha, coeff in op.terms) + (2 if vector else 0)
            from .kernels import PHS, Hybrid
            phs = descriptor.kernel.phs if isinstance(descriptor.kernel, Hybrid) else descriptor.kernel if isinstance(descriptor.kernel, PHS) else None
            if phs is not None and phs.power-1 < order:
                raise ValueError(f'Kernel requires continuous coincident derivatives through order {order}')
            poly, m, h = _polynomials(owner, np.concatenate([x, z]))
            top, bottom = poly(x, left), poly(z, right)
            _rank(top, m)
            _rank(bottom, m)
            target_points = np.repeat(target[None, :], len(req), axis=0)
            owner.jobs.append(dict(source_points=x, source_ops=left, trial_points=z, trial_ops=right,
                target_points=target_points, target_ops=req, top=top, bottom=bottom,
                target_poly=poly(target_points, req), scale=h if self.stencil_policy.scaling == 'local' else owner.arithmetic.number(1),
                size=len(x)*c, polynomials=m))
            owner.memberships.append({name: indices.copy() for name, indices in ids.items()})
            owner.trial_memberships.append({name: indices.copy() for name, indices in trial_ids.items()})
            global_ids, offset = [], 0
            for name, group in self.source.items():
                global_ids.extend(offset+ids[name]); offset += len(group.points)
            owner.indices.append(np.asarray(global_ids, dtype=int))
        results = execute(owner)
        owner.weights = [r['weights'] for r in results]
        owner.diagnostics = [dict(weight_residual=r['residual'], scaled_condition=r['condition'], factorizations=1,
            geometry=geometry_quality(np.unique(j['source_points'], axis=0), t, degree if degree is not None else 1))
            for r, j, t in zip(results, owner.jobs, query)]
        out = {}
        for column, name in enumerate(operators):
            matrix = _sparse_map(owner, column)
            blocks = {key: SampleMap(_slice_columns(matrix, layout['column_slice']), c, self.source[key].points)
                      for key, layout in owner.layout.items()}
            out[name] = FunctionalOperator(matrix, blocks, owner, column)
        return OperatorSet(out, dict(backend=type(owner.backend).__name__, factorizations=len(query),
            rhs_per_stencil=len(operators)*c, components=c, ordering='node-major',
            groups=tuple(self.source), local_digits=self.precision.local_digits, global_dtype=self.precision.global_dtype))


def _sparse_map(o, k):
    c = o.components
    shape = (len(o.targets)*c, sum(len(g.points) for g in o.source.values())*c)
    rows, columns, values = [], [], []
    for i, (ids, weights) in enumerate(zip(o.indices, o.weights)):
        for a in range(c):
            for j, node in enumerate(ids):
                for b in range(c):
                    rows.append(i*c+a); columns.append(int(node)*c+b)
                    values.append(weights[j*c+b, k*c+a])
    if isinstance(o.backend, TorchBackend):
        import torch
        v = torch.stack(values)
        return torch.sparse_coo_tensor(torch.tensor([rows, columns], dtype=torch.long, device=v.device), v, shape, check_invariants=True).coalesce()
    if o.precision.global_dtype == 'mpmath':
        data = [{} for _ in range(shape[0])]
        for i, j, v in zip(rows, columns, values):
            data[i][j] = data[i].get(j, 0)+v
        return MPSparseMatrix(o.arithmetic.ctx, data, ncols=shape[1])
    v = np.array([float(x) for x in values])
    if not np.isfinite(v).all():
        raise FloatingPointError('Weights overflow Float64 storage')
    return csr_matrix((v, (rows, columns)), shape=shape)


def _slice_columns(matrix, columns):
    if _tensor(matrix):
        import torch
        m = matrix.coalesce(); ids = m.indices()
        mask = (ids[1] >= columns.start) & (ids[1] < columns.stop)
        selected = torch.stack((ids[0, mask], ids[1, mask]-columns.start))
        return torch.sparse_coo_tensor(selected, m.values()[mask], (matrix.shape[0], columns.stop-columns.start), check_invariants=True).coalesce()
    if isinstance(matrix, MPSparseMatrix):
        return MPSparseMatrix(matrix.ctx,
            [{j-columns.start: v for j, v in row.items() if columns.start <= j < columns.stop} for row in matrix.rows],
            ncols=columns.stop-columns.start)
    return matrix[:, columns]


class _NodalCompatibilityOperator(FunctionalOperator):
    """Old RBFFD exports NumPy/SciPy explicitly; new maps retain Torch storage."""
    def local(self, i):
        result = super().local(i)
        for name in ('weights', 'multipliers'):
            value = getattr(result, name)
            if _tensor(value):
                setattr(result, name, value.detach().cpu().numpy().copy())
        return result

    def reconstruct_local(self, i):
        result = super().reconstruct_local(i)
        for name in ('matrix', 'rhs', 'scaled_matrix', 'scaled_rhs', 'scale'):
            value = getattr(result, name)
            if _tensor(value):
                setattr(result, name, value.detach().cpu().numpy().copy())
        return result


def nodal_compatibility(operators):
    """Keep the historical RBFFD result storage while sharing its new engine."""
    out = {}
    for name, op in operators.items():
        if _tensor(op.matrix):
            blocks = {key: SampleMap(block.to_scipy(), block.components, block.points)
                      for key, block in op.items()}
            op = _NodalCompatibilityOperator(op.to_scipy(), blocks, op._owner, op._column)
        out[name] = op
    return OperatorSet(out, operators.diagnostics)
