"""Stationary scalar linear PDE and tagged boundary data."""
from dataclasses import dataclass
import numpy as np
from .operators import Operator, Identity, SpatialOperator


def values(data, points):
    result = np.asarray(data(points) if callable(data) else data, dtype=float)
    if result.shape == (len(points), 1):
        result = result[:, 0]
    try:
        result = np.broadcast_to(result, (len(points),)).copy()
    except ValueError as exc:
        raise ValueError("Data must return a scalar or shape (N,)") from exc
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite PDE/boundary data")
    return result


@dataclass(frozen=True)
class BoundaryCondition:
    """A linear boundary row attached to a cloud label or boundary selection.

    Args:
        on: Boundary label or selection understood by the cloud.
        operator: Boundary functional applied to the unknown.
        rhs: Prescribed scalar values or a callable evaluated at boundary points."""
    on: object
    operator: object
    rhs: object
    _value_condition: bool = False


@dataclass(frozen=True)
class LinearPDE:
    """Stationary scalar problem `operator(u) = rhs` with boundary rows.

    Call `solve(cloud, method)` or assemble with a selected numerical method
    to inspect the matrix before solving."""
    operator: Operator
    rhs: object
    boundary: tuple

    def solve(self, cloud, method):
        """Assemble and solve using a selected scalar discretization."""
        return method.assemble(self, cloud).solve()

    def __post_init__(self):
        if not isinstance(self.operator, (Operator, SpatialOperator)) or not self.operator.terms:
            raise TypeError("Expected a nonzero scalar Operator or SpatialOperator")
        object.__setattr__(self, "boundary", tuple(self.boundary))


def Dirichlet(value=0.0, on="boundary"):
    """Create a value boundary condition `u = value` on a label."""
    return BoundaryCondition(on, Identity(), value, _value_condition=True)


def boundary_data(problem, cloud, ctx=None):
    """First boundary condition wins at shared corner nodes (explicit list priority)."""
    if problem.operator.dimension != cloud.dimension:
        raise ValueError("PDE operator and cloud dimensions must match")
    result = {}
    for bc in problem.boundary:
        labels = list(cloud.boundary) if bc.on == "boundary" else (
            [bc.on] if isinstance(bc.on, str) else list(bc.on))
        for label in labels:
            if label not in cloud.boundary:
                raise ValueError(f"Unknown boundary label: {label}")
            ids = cloud.boundary[label]
            data = bc.rhs
            if getattr(data, "requires_normals", False):
                if label not in cloud.normals:
                    raise ValueError(f"Normals required on {label}")
                data = data.bind_normals(cloud.normals[label])
            if ctx is None:
                vals = values(data, cloud.points[ids])
            else:
                from .precision import mp_values
                vals = mp_values(data, cloud.points[ids], ctx)
            for j, node in enumerate(ids):
                if int(node) in result:
                    continue
                if bc._value_condition:
                    op = Identity(cloud.dimension)
                elif isinstance(bc.operator, Operator):
                    op = bc.operator
                elif hasattr(bc.operator, "at_point"):
                    if bc.operator.requires_normals and label not in cloud.normals:
                        raise ValueError(f"Normals required on {label}")
                    normal = cloud.normals[label][j] if label in cloud.normals else None
                    op = bc.operator.at_point(cloud.points[node], normal)
                elif hasattr(bc.operator, "at"):
                    if label not in cloud.normals:
                        raise ValueError(f"Normals required on {label}")
                    op = bc.operator.at(cloud.normals[label][j])
                else:
                    raise TypeError("Unsupported boundary operator")
                if op.dimension != cloud.dimension:
                    raise ValueError("Boundary operator and cloud dimensions must match")
                if not op.terms:
                    raise ValueError("Boundary operator cannot be zero")
                result[int(node)] = (op, vals[j])
    if set(result) != set(cloud.boundary_indices):
        raise ValueError("Every boundary node must have a boundary condition")
    if not result:
        raise ValueError("This milestone requires boundary constraints")
    has_value = lambda op: any(not any(alpha) for alpha, _ in op.terms)
    if (not getattr(problem, "_evolution_slice", False)
            and not has_value(problem.operator) and not any(has_value(op) for op, _ in result.values())):
        raise NotImplementedError("Constant nullspace is unfixed; add a value condition or future gauge constraint")
    return result
