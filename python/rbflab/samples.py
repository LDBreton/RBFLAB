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

    def __post_init__(self):
        original = self.points
        p = points_array(original, empty=True)
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
        if normals is None and any(isinstance(op, (NormalDerivative, Robin)) for op in (self.operator if isinstance(self.operator, (list, tuple)) else [self.operator])):
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
    return groups


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
