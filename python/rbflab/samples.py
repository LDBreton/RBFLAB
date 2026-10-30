"""Functional samples and explicit kernel trial descriptions.

Samples describe mathematics, not whether data are known or unknown.  Arrays
are copied so a built approximation is independent of subsequent user edits.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .operators import Identity, NormalDerivative, Robin, bind_operators
from .stencils import StencilPolicy


def points_array(value, dimension=None, *, empty=False):
    points = np.array(getattr(value, 'points', value), dtype=float, copy=True)
    if (points.ndim != 2 or points.shape[1] not in (2, 3)
            or (dimension is not None and points.shape[1] != dimension)
            or (not empty and not len(points)) or not np.isfinite(points).all()):
        raise ValueError('Expected finite Nx2 or Nx3 points of matching dimension')
    return points


@dataclass(frozen=True)
class Stencil:
    """A fixed nearest-site selection shared by multiple sample groups.

    ``Stencil(points, size=37)`` selects physical sites before groups
    intersect them with their own points and apply target exclusion. No
    observations or unknowns are introduced by this geometric object.
    Coordinates match exactly; missing/excluded sites are not replaced.
    Selection uses the same nearest-neighbor convention as Samples(size=...).
    This fixed selection does not grow under a quality stencil policy.

    Alternatively, pass ``indices=memberships`` instead of size: one sequence
    of distinct point indices per target. Explicit ordering is preserved.
    """
    points: object
    size: int | None = None
    indices: object = None

    def __post_init__(self):
        points = points_array(self.points)
        if self.indices is None:
            if type(self.size) is not int or not 1 <= self.size <= len(points):
                raise ValueError('Stencil size must be between 1 and its point count')
        elif self.size is not None:
            raise ValueError('Stencil indices cannot be combined with size')
        else:
            members = []
            for row in self.indices:
                ids = np.asarray(row)
                if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
                    raise ValueError('Stencil membership rows must contain integer indices')
                ids = ids.astype(int, copy=True)
                if len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(points)):
                    raise ValueError('Invalid or repeated stencil indices')
                ids.flags.writeable = False
                members.append(ids)
            object.__setattr__(self, 'indices', tuple(members))
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError('SharedStencil requires distinct physical sites')
        points.flags.writeable = False
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, '_tree', cKDTree(points))

    def __eq__(self, other):
        return (isinstance(other, Stencil) and self.size == other.size
                and np.array_equal(self.points, other.points)
                and ((self.indices is None and other.indices is None) or
                     (self.indices is not None and other.indices is not None
                      and len(self.indices) == len(other.indices)
                      and all(np.array_equal(a, b) for a, b in zip(self.indices, other.indices)))))

    def select(self, target, row=0):
        """Return nearest physical-site indices, ordered by distance and index."""
        if self.indices is not None:
            if row >= len(self.indices):
                raise ValueError('One stencil membership row is required per target')
            return self.indices[row].copy()
        distance, ids = self._tree.query(target, k=self.size)
        distance, ids = np.atleast_1d(distance), np.atleast_1d(ids)
        return ids[np.lexsort((ids, distance))]


# Preserve the original public name and isinstance behavior.
SharedStencil = Stencil


@dataclass(frozen=True)
class Samples:
    """Points carrying values or differential-functional samples.

    Args:
        points: A point cloud or Nx2/Nx3 array. Geometry is Float64.
        operator: One differential operator, or one per point; None is value.
        size: Neighbors per target; None uses all eligible samples.
        target: 'allow', 'exclude', or 'require' geometric target coincidence.
        indices: Explicit group-local memberships, one integer row per target.
            Cannot be combined with size or a selection policy.
        policy: Optional geometric neighbor-selection policy for this group.
            Kernel scaling belongs to LocalApproximation instead.
        normals: Normals for normal-dependent boundary operators.
        within: SharedStencil or name of a source group whose physical sites restrict
            this group. Uses exact coordinate matching, then target filtering;
            does not refill excluded sites. Cannot combine with size, indices
            or policy. The referenced group may appear later in the mapping.

    Groups may overlap geometrically when their functionals differ. No PDE
    role, forcing, boundary data, or unknown values are stored here.
    """
    points: object
    operator: object = None
    size: int | None = None
    target: str = 'allow'
    indices: object = None
    policy: object = None
    normals: object = None
    within: str | SharedStencil | None = None

    def __post_init__(self):
        original = self.points
        p = points_array(original, empty=True)
        from .field_expressions import FieldExpression, as_expression
        import sympy as sp
        if isinstance(self.operator, (sp.Expr, sp.MatrixBase)):
            object.__setattr__(self, 'operator', as_expression(self.operator))
        if isinstance(self.operator, FieldExpression):
            if not self.operator.requires_binding:
                object.__setattr__(self, 'operator', self.operator.compile(p.shape[1]))
        if self.within is not None:
            if not isinstance(self.within, (str, SharedStencil)) or (isinstance(self.within, str) and not self.within):
                raise ValueError('within must name a sample group or be a SharedStencil')
            if isinstance(self.within, SharedStencil) and self.within.points.shape[1] != p.shape[1]:
                raise ValueError('SharedStencil and sample dimensions must agree')
            if self.size is not None or self.indices is not None or self.policy is not None:
                raise ValueError('within cannot be combined with size, indices or policy')
        if self.target not in ('allow', 'exclude', 'require'):
            raise ValueError('target must be allow, exclude, or require')
        if self.size is not None and (type(self.size) is not int or self.size < 0):
            raise ValueError('size must be a nonnegative integer')
        if self.indices is not None and (self.size is not None or self.policy is not None):
            raise ValueError('Explicit indices cannot be combined with size or policy')
        if self.policy is not None and (not isinstance(self.policy, StencilPolicy)
                or self.policy.scaling != 'physical' or self.policy.shape_rule != 'fixed'):
            raise ValueError('Group policy only selects points; configure scaling on the approximation')
        normals = self.normals
        if normals is None and any(isinstance(op, (NormalDerivative, Robin)) or getattr(op, 'requires_normals', False) for op in (self.operator if isinstance(self.operator, (list, tuple)) else [self.operator])):
            normals = getattr(original, 'normals', None)
        if normals is not None:
            normals = np.array(normals, dtype=float, copy=True)
            if normals.shape != p.shape or not np.isfinite(normals).all():
                raise ValueError('Provide one finite normal per sample')
            normals.flags.writeable = False
        ops = self.operator
        if ops is None:
            ops = [Identity(p.shape[1])] * len(p)
        elif not isinstance(ops, (list, tuple)):
            ops = [ops] * len(p)
        if len(ops) != len(p):
            raise ValueError('Provide one operator per sample')
        bound = []
        for i, op in enumerate(ops):
            from .field_expressions import FieldExpression
            if isinstance(op, (sp.Expr, sp.MatrixBase)):
                op = as_expression(op)
            if isinstance(op, FieldExpression):
                if op.requires_binding:
                    bound.append(op)
                    continue
                op = op.compile(p.shape[1])
            from .mixed import FieldOperator
            if isinstance(op, FieldOperator):
                bound.append(op.at_point(p[i], None if normals is None else normals[i]))
                continue
            if isinstance(op, (NormalDerivative, Robin)):
                if normals is None:
                    raise ValueError('Normal-dependent samples require normals')
                op = op.at(normals[i])
            op = bind_operators([op], p[i:i+1])[0]
            if not hasattr(op, 'terms') or op.dimension != p.shape[1]:
                raise ValueError('Sample operator dimension mismatch')
            bound.append(op)
        members = None
        if self.indices is not None:
            members = []
            for row in self.indices:
                ids = np.asarray(row)
                if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
                    raise ValueError('Membership rows must contain integer indices')
                ids = ids.astype(int, copy=True)
                if len(set(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(p)):
                    raise ValueError('Invalid or repeated sample indices')
                ids.flags.writeable = False
                members.append(ids)
            members = tuple(members)
        p.flags.writeable = False
        object.__setattr__(self, 'points', p)
        object.__setattr__(self, 'normals', normals)
        object.__setattr__(self, 'indices', members)
        object.__setattr__(self, '_operators', tuple(bound))
        object.__setattr__(self, '_tree', cKDTree(p))
        locations = {}
        for i, point in enumerate(p):
            locations.setdefault(tuple(point), []).append(i)
        object.__setattr__(self, '_locations', locations)

    def select(self, target, row, policy, degree):
        """Return group-local membership; functionals are rank-checked jointly."""
        coincident = set(self._locations.get(tuple(target), ()))
        policy = self.policy or policy
        if self.indices is not None:
            if row >= len(self.indices):
                raise ValueError('One membership row is required per target')
            ids = self.indices[row].copy()
        else:
            excluded = coincident if self.target == 'exclude' else set()
            available = len(self.points)-len(excluded)
            size = available if self.size is None else self.size
            if size > available:
                raise ValueError('Sample size exceeds eligible points after target filtering')
            if size == 0:
                ids = np.array([], dtype=int)
            elif self.size is None:
                ids = np.array([i for i in range(len(self.points)) if i not in excluded], dtype=int)
            elif policy.selection == 'nearest':
                distance, candidates = self._tree.query(target, k=size+len(excluded))
                distance, candidates = np.atleast_1d(distance), np.atleast_1d(candidates)
                ordered = candidates[np.lexsort((candidates, distance))]
                ids = np.array([i for i in ordered if int(i) not in excluded][:size], dtype=int)
            elif not excluded:
                ids = policy.select(self._tree, target, size, degree)
            else:
                # Adaptive geometric selection is opt-in. Exclusion changes
                # its candidate set; the common nearest path reuses one tree.
                eligible = np.array([i for i in range(len(self.points)) if i not in excluded])
                ids = eligible[policy.select(cKDTree(self.points[eligible]), target, size, degree)]
        if self.target == 'exclude' and any(int(i) in coincident for i in ids):
            raise ValueError('Membership violates target exclusion')
        if self.target == 'require' and not any(int(i) in coincident for i in ids):
            raise ValueError('Required target is absent from selected samples')
        return ids



def sample_groups(source, *, size=None):
    if isinstance(source, Mapping):
        if not source or any(not isinstance(k, str) or not k for k in source):
            raise ValueError('Source groups need nonempty string names')
        groups = {k: v if isinstance(v, Samples) else Samples(v) for k, v in source.items()}
    else:
        groups = {'u': source if isinstance(source, Samples) else Samples(source, size=size)}
    dimensions = {g.points.shape[1] for g in groups.values()}
    if len(dimensions) != 1 or not sum(len(g.points) for g in groups.values()):
        raise ValueError('Sample groups need matching dimensions and at least one point')
    for name in groups:
        visited = set()
        current = name
        while current is not None:
            if isinstance(current, SharedStencil):
                break
            if current not in groups:
                raise ValueError(f'Unknown within sample group: {current}')
            if current in visited:
                raise ValueError('Cyclic within sample groups')
            visited.add(current)
            current = groups[current].within
    return groups


def restrict_groups(groups, selected, target, row=0):
    """Resolve shared-site groups after independent selections (or growth)."""
    shared = {}
    def resolve(name):
        group = groups[name]
        if group.within is None:
            return selected[name]
        if isinstance(group.within, SharedStencil):
            parent = group.within
            key = id(parent)
            if key not in shared:
                shared[key] = parent.select(target, row)
            parent_ids = shared[key]
        else:
            parent_ids = resolve(group.within)
            parent = groups[group.within]
        ids, seen = [], set()
        for j in parent_ids:
            for i in group._locations.get(tuple(parent.points[j]), ()):
                if i not in seen and not (group.target == 'exclude'
                                         and np.array_equal(group.points[i], target)):
                    ids.append(i)
                    seen.add(i)
        if group.target == 'require' and not any(np.array_equal(group.points[i], target) for i in ids):
            raise ValueError('Required target is absent from selected samples')
        selected[name] = np.asarray(ids, dtype=int)
        return selected[name]
    for name in groups:
        resolve(name)
    return {name: selected[name] for name in groups}


def select_groups(groups, target, row, policy, degree):
    selected = {name: group.select(target, row, policy, degree)
                for name, group in groups.items() if group.within is None}
    return restrict_groups(groups, selected, target, row)


@dataclass(frozen=True)
class KernelTrial:
    """Explicit trial basis made by space.representers() or .translates().

    The polynomial side constraints use the trial functionals applied to the
    space polynomials. This is the representer construction, including when
    the fitted data functionals differ from the trial functionals.
    """
    space: object
    source: object
    kind: str = 'representers'

    def __post_init__(self):
        if self.kind not in ('representers', 'translates'):
            raise ValueError('Unknown trial construction')
        object.__setattr__(self, 'source', sample_groups(self.source))
