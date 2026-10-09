"""Coupled stationary symbolic fields compiled into block differential rows."""
from dataclasses import dataclass
import sympy as sp
from sympy.core.function import AppliedUndef
from .symbolic import SymbolicScalar, _PointBoundaryOperator, _expr


@dataclass(frozen=True)
class BlockEquation:
    operators: dict
    rhs: object


@dataclass(frozen=True)
class BlockBoundary:
    on: object
    slot: int
    operators: dict
    rhs: object


@dataclass(frozen=True)
class PointConstraint:
    field: int
    point: tuple
    value: object
    equation: int


@dataclass(frozen=True)
class BlockPDE:
    """Compiled coupled stationary linear PDE with block boundary rows.

    Produced by `SymbolicSystem.stationary`. Field order determines block
    ordering; point constraints can fix a scalar nullspace such as pressure."""
    fields: tuple
    dimension: int
    equations: tuple
    boundary: tuple
    constraints: tuple

    def field_index(self,field):
        if field in self.fields:return self.fields.index(field)
        matches=[i for i,f in enumerate(self.fields) if str(f.func)==field]
        if len(matches)==1:return matches[0]
        raise ValueError(f"Unknown field: {field}")

    def solve(self,cloud,method):return method.assemble(self,cloud).solve()


class SymbolicSystem:
    """Named scalar fields, one stationary equation slot per field, in 2D/3D.

    A boundary row replaces the equation with the same slot at boundary sites.
    Interior equations are retained in every slot. LHI pressure constraints
    use the specified equation slot for an explicit compatibility multiplier.
    """
    def __init__(self,dimension=2,*,fields=('u','v','p')):
        self._scalar=SymbolicScalar(dimension)
        if not fields or len(set(fields))!=len(fields) or any(not isinstance(f,str) or not f for f in fields):
            raise ValueError("Expected distinct nonempty field names")
        self.dimension=dimension;self.coordinates=self._scalar.coordinates
        self.normal=self._scalar.normal
        self.fields=tuple(sp.Function(f)(*self.coordinates) for f in fields)

    def data(self,expression,*,parameters=None):return self._scalar.data(expression,parameters=parameters)
    def laplacian(self,expression):return self._scalar.laplacian(expression)
    def normal_derivative(self,expression):return self._scalar.normal_derivative(expression)
    def divergence(self,vector):
        if len(vector)!=self.dimension:raise ValueError("Vector length must equal spatial dimension")
        return sum(sp.diff(f,x) for f,x in zip(vector,self.coordinates))

    def bc(self,on,equations,*,slots=None):
        equations=tuple(equations) if isinstance(equations,(tuple,list)) else (equations,)
        if slots is None:
            inferred=[]
            for equation in equations:
                fields=set(equation.atoms(AppliedUndef)) if isinstance(equation,sp.Basic) else set()
                if len(fields)!=1 or not fields.issubset(set(self.fields)):
                    raise ValueError("Coupled boundary rows need explicit slots, e.g. slots=(0,1)")
                inferred.append(self.fields.index(next(iter(fields))))
            slots=tuple(inferred)
        if len(slots)!=len(equations) or any(type(i) is not int or not 0<=i<len(self.fields) for i in slots):
            raise ValueError("One valid equation slot is required per boundary row")
        return (on,equations,tuple(slots))

    def constrain(self,field,*,point,value=0,equation=None):
        if field not in self.fields:raise ValueError("Constraint field must be declared")
        index=self.fields.index(field)
        equation=index if equation is None else equation
        if type(equation) is not int or not 0<=equation<len(self.fields):raise ValueError("Invalid constraint equation slot")
        if len(point)!=self.dimension:raise ValueError("Constraint point dimension mismatch")
        return (index,tuple(point),value,equation)

    def _linear(self,equation,mapping):
        expression=equation.lhs-equation.rhs if isinstance(equation,sp.Equality) else _expr(equation)
        expression=sp.expand(expression.xreplace(mapping).doit())
        if expression.has(sp.Integral,sp.Sum,sp.Product):raise NotImplementedError("Only local differential equations are supported")
        if expression.atoms(AppliedUndef)-set(self.fields):raise ValueError("Equation contains undeclared fields")
        derivatives=sorted(expression.atoms(sp.Derivative),key=sp.default_sort_key)
        slots=list(self.fields)+derivatives
        descriptors=[(i,(0,)*self.dimension) for i in range(len(self.fields))]
        for derivative in derivatives:
            if derivative.expr not in self.fields:raise ValueError("Unsupported field derivative")
            alpha=[0]*self.dimension
            for x,order in derivative.variable_count:
                if x not in self.coordinates:raise NotImplementedError("Coupled equations currently support spatial derivatives only")
                alpha[self.coordinates.index(x)]+=int(order)
            descriptors.append((self.fields.index(derivative.expr),tuple(alpha)))
        unknowns=tuple(sp.Dummy() for _ in slots)
        try:poly=sp.Poly(expression.xreplace(dict(zip(slots,unknowns))),*unknowns)
        except sp.PolynomialError as exc:raise ValueError("Equation must be linear in all fields") from exc
        if poly.total_degree()>1:raise ValueError("Equation must be linear in all fields")
        terms={}
        for symbol,(field,alpha) in zip(unknowns,descriptors):
            c=sp.simplify(poly.coeff_monomial(symbol))
            if c!=0:terms.setdefault(field,[]).append((alpha,c))
        if not terms:raise ValueError("Equation has no nonzero field operator")
        return terms,-poly.coeff_monomial(1)

    def stationary(self,equations,*,boundary,constraints=(),parameters=None):
        equations=tuple(equations)
        if len(equations)!=len(self.fields):raise ValueError("Supply one equation slot per field")
        scalar=self._scalar;mapping=scalar._parameters(parameters);compiled=[]
        for eq in equations:
            terms,rhs=self._linear(eq,mapping)
            compiled.append(BlockEquation({f:scalar._operator(t) for f,t in terms.items()},scalar._data(rhs)))
        bcs=[]
        for on,rows,slots in boundary:
            for row,slot in zip(rows,slots):
                terms,rhs=self._linear(row,mapping);ops={}
                for field,terms_i in terms.items():
                    if any(c.free_symbols-set(self.coordinates+self.normal) for _,c in terms_i):
                        raise ValueError("Unbound boundary coefficients")
                    ops[field]=_PointBoundaryOperator(tuple(terms_i),self.coordinates,self.normal)
                bcs.append(BlockBoundary(on,slot,ops,scalar._data(rhs,normals=True)))
        fixed=[]
        for field,point,value,equation in constraints:
            fixed.append(PointConstraint(field,point,scalar._data(_expr(value).xreplace(mapping)),equation))
        # An absent value functional leaves a scalar constant mode undetermined.
        for field in range(len(self.fields)):
            operators=[row.operators[field] for row in compiled+bcs if field in row.operators]
            if not any(any(not any(alpha) for alpha,_ in op.terms) for op in operators) and not any(c.field==field for c in fixed):
                raise ValueError(f"Field {self.fields[field]} has an unfixed constant mode; add a point constraint")
        if len({c.field for c in fixed})!=len(fixed):raise ValueError("Only one point constraint per field is supported")
        if len({c.equation for c in fixed})!=len(fixed):raise ValueError("Constraint equation slots must be distinct")
        return BlockPDE(self.fields,self.dimension,tuple(compiled),tuple(bcs),tuple(fixed))
