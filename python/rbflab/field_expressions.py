"""Dimension-independent linear field expressions compiled to FieldOperator."""
from dataclasses import dataclass
import sympy as sp
from .mixed import FieldOperator
from .operators import Identity, Derivative, Laplacian, _coefficient


@dataclass(frozen=True)
class FieldExpression:
    """Linear scalar/vector expression; dimensions are supplied by sample geometry."""
    kind: str
    args: tuple
    vector: bool
    _op_priority = 1000

    def __add__(self, other):
        try:
            other = as_expression(other)
        except TypeError:
            return NotImplemented
        if self.vector != other.vector:
            raise ValueError('Cannot add scalar and vector field expressions')
        return FieldExpression('add', (self, other), self.vector)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return as_expression(other) + self

    def __rsub__(self, other):
        return as_expression(other) - self

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, coefficient):
        return FieldExpression('scale', (self, _coefficient(coefficient)), self.vector)

    __rmul__ = __mul__

    def __getitem__(self, component):
        if not self.vector:
            raise ValueError('Component indexing requires a vector expression')
        if type(component) is not int or component < 0:
            raise ValueError('Component must be a nonnegative integer')
        return FieldExpression('component', (self, component), False)

    @property
    def requires_binding(self):
        return self.kind == "symbolic" or any(isinstance(a, FieldExpression) and a.requires_binding for a in self.args)

    def compile(self, dimension, resolver=None):
        """Return the equivalent explicit constant-coefficient FieldOperator."""
        if self.kind == 'symbolic':
            if resolver is None:
                raise ValueError('Symbolic fields must be bound through MixedSpace')
            return resolver(self.args[0], dimension)
        if self.kind == 'field':
            width = dimension if self.vector else 1
            return FieldOperator({self.args[0]: [[Identity(dimension) if i == j else 0
                                                 for j in range(width)] for i in range(width)]})
        base = self.args[0].compile(dimension, resolver)
        if self.kind == 'component':
            if self.args[1] >= base.arity:
                raise ValueError('Component index exceeds expression output size')
            return FieldOperator({name: [rows[self.args[1]]] for name, rows in base.blocks})
        if self.kind == 'vector':
            compiled = [base] + [expr.compile(dimension, resolver) for expr in self.args[1:]]
            widths = {}
            for item in compiled:
                for name, rows in item.blocks:
                    width = len(rows[0])
                    if name in widths and widths[name] != width:
                        raise ValueError('Conflicting component counts for the same field name')
                    widths[name] = width
            return FieldOperator({name: [dict(item.blocks).get(name, ((0,)*width,))[0]
                                         for item in compiled] for name, width in widths.items()})
        if self.kind == 'derivative':
            op = Identity(dimension)
            for _ in range(self.args[2]):
                op = Derivative(self.args[1], dimension) @ op
            # Validate the axis even for the zeroth derivative.
            Derivative(self.args[1], dimension)
            return FieldOperator({name: [[op @ entry if entry != 0 else 0 for entry in row]
                                         for row in rows] for name, rows in base.blocks})
        if self.kind == 'add':
            right = self.args[1].compile(dimension, resolver)
            if base.arity != right.arity:
                raise ValueError('Field expression output sizes must agree')
            blocks = dict(base.blocks)
            for name, rows in right.blocks:
                if name in blocks:
                    if len(blocks[name][0]) != len(rows[0]):
                        raise ValueError('Conflicting component counts for the same field name')
                    rows = tuple(tuple(a+b for a, b in zip(x, y))
                                 for x, y in zip(blocks[name], rows))
                blocks[name] = rows
            return FieldOperator(blocks)
        if self.kind == 'scale':
            return FieldOperator({name: [[op*self.args[1] if op != 0 else 0 for op in row] for row in rows]
                                  for name, rows in base.blocks})
        if self.kind == 'laplacian':
            return FieldOperator({name: [[Laplacian(dimension) @ op if op != 0 else 0
                                         for op in row] for row in rows]
                                  for name, rows in base.blocks})
        if self.kind == 'gradient':
            return FieldOperator({name: [[Derivative(axis, dimension) @ op if op != 0 else 0
                                         for op in rows[0]] for axis in range(dimension)]
                                  for name, rows in base.blocks})
        raise ValueError('Unknown field expression')


def laplacian(expression):
    """Apply the Laplacian componentwise to a scalar or vector field expression."""
    expression = as_expression(expression)
    return FieldExpression('laplacian', (expression,), expression.vector)


def gradient(expression):
    """Map a scalar field expression to its spatial gradient in 2D or 3D."""
    expression = as_expression(expression)
    if expression.vector:
        raise ValueError('gradient expects a scalar expression; tensor outputs are not supported')
    return FieldExpression('gradient', (expression,), True)


def vector(*components):
    """Stack scalar field expressions into an ordered vector of outputs.

    Output length is independent of spatial dimension. Each component must
    be a scalar expression; use indexing to extract vector components.
    """
    components = tuple(as_expression(e) for e in components)
    if not components:
        raise TypeError('vector requires one or more scalar field expressions')
    if any(e.vector for e in components):
        raise ValueError('vector components must be scalar expressions')
    return FieldExpression('vector', tuple(components), True)


def derivative(expression, axis, order=1):
    """Differentiate a scalar/vector expression along a zero-based Cartesian axis.

    Order is a nonnegative integer. Nest calls for mixed derivatives.
    Spatial dimension and axis bounds are checked when bound to geometry.
    """
    expression = as_expression(expression)
    if type(axis) is not int or axis < 0 or type(order) is not int or order < 0:
        raise ValueError('axis and order must be nonnegative integers')
    return FieldExpression('derivative', (expression, axis, order), expression.vector)


def dx(expression, order=1):
    """Differentiate a field expression along coordinate 0 (x)."""
    return derivative(expression, 0, order)


def dy(expression, order=1):
    """Differentiate a field expression along coordinate 1 (y)."""
    return derivative(expression, 1, order)


def dz(expression, order=1):
    """Differentiate a field expression along coordinate 2 (z); requires 3D."""
    return derivative(expression, 2, order)


def as_expression(value):
    """Keep linear expressions or defer symbolic fields until space binding."""
    if isinstance(value, FieldExpression):
        return value
    if isinstance(value, (sp.Expr, sp.MatrixBase)):
        if isinstance(value, sp.MatrixBase) and value.cols != 1:
            raise ValueError('Local vector expressions must be column vectors')
        return FieldExpression('symbolic', (value,), isinstance(value, sp.MatrixBase))
    raise TypeError('Expected a field expression or symbolic model field')
