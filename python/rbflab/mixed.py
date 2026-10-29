"""Explicit linear functionals on named scalar and vector fields."""
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from .operators import bind_operators


@dataclass(frozen=True, init=False)
class FieldOperator:
    """A rectangular differential block for each named field.

    ``FieldOperator({'u': [[Lap, 0], [0, Lap]], 'p': [[Dx], [Dy]]})``
    has two outputs. Blocks share a row count; their columns are the field
    components. Only literal zero is accepted in place of an operator.
    Spatial dimensions and field widths are validated when bound.
    """
    blocks: tuple
    arity: int

    def __init__(self, blocks):
        if not isinstance(blocks, Mapping) or not blocks:
            raise ValueError('FieldOperator needs a nonempty field-to-block mapping')
        normalized, arity = [], None
        for name, block in blocks.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Field names must be nonempty strings')
            rows = tuple(tuple(row) for row in block)
            if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError('Field operator blocks must be nonempty rectangular matrices')
            if arity is None:
                arity = len(rows)
            if len(rows) != arity:
                raise ValueError('Field operator blocks need the same output arity')
            for row in rows:
                for op in row:
                    if isinstance(op, (int, float, np.number)) and op == 0:
                        continue
                    if not (hasattr(op, 'terms') or hasattr(op, 'at_point') or hasattr(op, 'at')):
                        raise TypeError('Use scalar differential operators or literal zero in field blocks')
            normalized.append((name, rows))
        object.__setattr__(self, 'blocks', tuple(normalized))
        object.__setattr__(self, 'arity', arity)

    @property
    def requires_normals(self):
        """Whether an unbound scalar entry needs a surface normal."""
        from .operators import NormalDerivative, Robin
        return any(isinstance(op, (NormalDerivative, Robin)) or getattr(op, 'requires_normals', False)
                   for _, rows in self.blocks for row in rows for op in row)

    def at_point(self, point, normal=None):
        """Bind spatially varying scalar entries without selecting a field."""
        d = len(point)
        blocks = {}
        for name, rows in self.blocks:
            bound = []
            for row in rows:
                values = []
                for op in row:
                    if isinstance(op, (int, float, np.number)) and op == 0:
                        values.append(0)
                    else:
                        if hasattr(op, 'at_point'):
                            op = op.at_point(point, normal)
                        elif hasattr(op, 'at') and not hasattr(op, 'terms'):
                            if normal is None:
                                raise ValueError('Normal-dependent field entries require normals')
                            op = op.at(normal)
                        op = bind_operators([op], np.asarray(point)[None, :])[0]
                        if op.dimension != d:
                            raise ValueError('Field operator spatial dimension mismatch')
                        values.append(op)
                bound.append(values)
            blocks[name] = bound
        return FieldOperator(blocks)

    def scalar_rows(self, fields):
        """Normalize to scalar rows indexed by field and component."""
        lookup = {field.name: (i, field.components) for i, field in enumerate(fields)}
        result = [{} for _ in range(self.arity)]
        for name, block in self.blocks:
            if name not in lookup:
                raise ValueError(f'Unknown field {name!r}')
            field, width = lookup[name]
            if len(block[0]) != width:
                raise ValueError(f'Field {name!r} needs {width} operator columns')
            for i, row in enumerate(block):
                for component, op in enumerate(row):
                    if isinstance(op, (int, float, np.number)) and op == 0:
                        continue
                    if not hasattr(op, 'terms'):
                        raise ValueError('Field operators must be bound to geometry')
                    if op.terms:
                        result[i][field, component] = op
        return result
