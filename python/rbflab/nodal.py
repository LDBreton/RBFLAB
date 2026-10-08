"""Reusable nodal basis, polynomial augmentation and dense collocation."""
import math
import numpy as np
from .assembly import functional_matrix, Factor, relative_residual
from .precision import Precision, MPBackend, MPFactor, mp_number, mp_values
from .operators import Identity, bind_operators
from .stencils import polynomial_powers
from .kernels import validate_polynomial_degree
from .problems import values, boundary_data


class Arithmetic:
    """Uniform dense assembly/factor interface; no implicit MP-to-float conversion."""
    def __init__(self, kernel, digits=None):
        self.kernel = kernel
        self.backend = MPBackend(kernel, digits) if digits else None
        self.ctx = self.backend.ctx if self.backend else None

    def number(self, value):
        return mp_number(self.ctx, value) if self.ctx else float(value)

    def zeros(self, n, m):
        return self.ctx.matrix(n,m) if self.ctx else np.zeros((n,m))

    def vector(self, data):
        return self.ctx.matrix(list(data)) if self.ctx else np.asarray(data, dtype=float)

    def matrix(self, x, left, y, right, length_scale=None):
        left, right = bind_operators(left,x), bind_operators(right,y)
        dimensions = {len(point) for point in list(x)+list(y)}
        if len(dimensions) != 1 or next(iter(dimensions)) not in (2,3):
            raise ValueError("Point dimensions must match")
        dimension = next(iter(dimensions))
        if len(left) != len(x) or len(right) != len(y) or any(op.dimension != dimension for op in list(left)+list(right)):
            raise ValueError("Functional counts and dimensions must match the points")
        if length_scale is not None:
            from .scaling import scaled_matrix
            return scaled_matrix(self,x,left,y,right,length_scale)
        return self.backend.matrix(x,left,y,right) if self.ctx else functional_matrix(self.kernel,x,left,y,right)

    def factor(self, matrix, *, compute_condition=True):
        return MPFactor(self.backend,matrix,general=True,compute_condition=compute_condition) if self.ctx else Factor(matrix,general=True,compute_condition=compute_condition)

    def data(self, data, points):
        if self.ctx and callable(data) and not hasattr(data,"mp_values"):
            raise TypeError("Extended nodal methods require PrecisionData callbacks")
        return mp_values(data,points,self.ctx) if self.ctx else values(data,points)

    def norm(self, vector):
        return self.ctx.norm(vector,"inf") if self.ctx else np.linalg.norm(vector,ord=np.inf)


class NodalBasis:
    """Kernel/source-functional basis plus shifted/scaled monomials."""
    def __init__(self, arithmetic, centers, degree=None, source_operators=None, kernel_scaling="physical"):
        from .methods import _query
        self.arithmetic = arithmetic
        self.centers = _query(centers)
        if not len(self.centers) or (source_operators is None and len(np.unique(self.centers,axis=0)) != len(self.centers)):
            raise ValueError("Expected nonempty distinct centers")
        if degree is not None and (type(degree) is not int or degree < 0):
            raise ValueError("polynomial_degree must be None or a nonnegative integer")
        validate_polynomial_degree(arithmetic.kernel,degree)
        self.powers = [] if degree is None else polynomial_powers(self.centers.shape[1], degree)
        if len(self.powers)>len(self.centers):
            raise ValueError("Too few centers for polynomial basis")
        # Input geometry is Float64; subsequent normalization uses selected arithmetic.
        self.origin = [arithmetic.number(v) for v in self.centers[0]]
        self.scale = max(abs(arithmetic.number(v)-self.origin[j]) for p in self.centers for j,v in enumerate(p)) or arithmetic.number(1)
        if kernel_scaling not in ("physical","local"):
            raise ValueError("kernel_scaling must be physical or local")
        self.kernel_scaling=kernel_scaling
        sqrt=arithmetic.ctx.sqrt if arithmetic.ctx else math.sqrt
        self.kernel_scale=max(sqrt(sum((arithmetic.number(v)-self.origin[j])**2
            for j,v in enumerate(point))) for point in self.centers) or arithmetic.number(1)
        self.identities = [Identity(self.centers.shape[1])]*len(self.centers)
        self.source_operators = bind_operators(list(source_operators),self.centers) if source_operators is not None else self.identities
        if len(self.source_operators)!=len(self.centers):
            raise ValueError("One source functional required per center")

    def polynomials(self, points, operators):
        operators = bind_operators(operators,points)
        a=self.arithmetic
        out=a.zeros(len(points),len(self.powers))
        for i,(point,op) in enumerate(zip(points,operators)):
            if len(point) != len(self.origin) or op.dimension != len(self.origin):
                raise ValueError("Polynomial point and operator dimensions must match centers")
            z=[(a.number(v)-self.origin[j])/self.scale for j,v in enumerate(point)]
            for j,power in enumerate(self.powers):
                for alpha,coefficient in op.terms:
                    if all(d<=p for d,p in zip(alpha,power)):
                        term=a.number(coefficient)/self.scale**sum(alpha)
                        for value,p,d in zip(z,power,alpha):
                            term *= (math.factorial(p)//math.factorial(p-d))*value**(p-d)
                        out[i,j]+=term
        return out

    def evaluation(self, points, operators):
        a=self.arithmetic;n=len(self.centers);m=len(self.powers)
        kernel=a.matrix(points,operators,self.centers,self.source_operators,
            self.kernel_scale if self.kernel_scaling=="local" else None)
        if not m:
            return kernel
        out=a.zeros(len(points),n+m)
        out[:,:n]=kernel
        out[:,n:]=self.polynomials(points,operators)
        return out

    def system_matrix(self, points, operators):
        n=len(self.centers);m=len(self.powers)
        if len(points)!=n:
            raise ValueError("Square collocation needs one equation per center")
        out=self.arithmetic.zeros(n+m,n+m)
        out[:n,:]=self.evaluation(points,operators)
        if m:
            # P_ij = source_functional_i(p_j), including Hermite derivatives.
            polynomial=self.polynomials(self.centers,self.source_operators)
            check=np.array(polynomial.tolist() if self.arithmetic.ctx else polynomial,dtype=float)
            # Geometry is Float64 even in MP mode; screen numerical unisolvency.
            row_scale=np.max(abs(check),axis=1)
            check=check/np.where(row_scale>0,row_scale,1)[:,None]
            column_scale=np.max(abs(check),axis=0)
            check=check/np.where(column_scale>0,column_scale,1)[None,:]
            if np.linalg.matrix_rank(check)<m:
                raise np.linalg.LinAlgError(
                    "Polynomial functionals are not unisolvent at Float64 geometry tolerance; enlarge or change the stencil")
            out[n:,:n]=polynomial.T
        return out

    def padded(self, data):
        return self.arithmetic.vector(list(data)+[0]*len(self.powers))


class DenseNodalSystem:
    def __init__(self,basis,points,operators,data):
        self.basis=basis
        self.matrix=basis.system_matrix(points,operators)
        self.rhs=basis.padded(data)
        self.factor=basis.arithmetic.factor(self.matrix)

    def solve(self):
        a=self.basis.arithmetic
        coefficients=self.factor.solve(self.rhs)
        residual=self.factor.residual(coefficients,self.rhs,transpose=False) if a.ctx else relative_residual(self.matrix,coefficients,self.rhs)
        return NodalSolution(self,coefficients,{"relative_residual":float(residual),
            "digits":a.ctx.dps if a.ctx else None,"scaled_condition":self.factor.condition})


class NodalSolution:
    def __init__(self,system,coefficients,diagnostics):
        self.system,self.coefficients,self.diagnostics=system,coefficients,diagnostics

    def _evaluate(self,points,operator=None):
        from .methods import _query
        points=_query(points, self.system.basis.centers.shape[1])
        a=self.system.basis.arithmetic
        if not len(points):
            return a.vector([])
        matrix=self.system.basis.evaluation(points,[operator or Identity(points.shape[1])]*len(points))
        return matrix*self.coefficients if a.ctx else matrix@self.coefficients

    def evaluate_mp(self,points,operator=None):
        if not self.system.basis.arithmetic.ctx:
            raise ValueError("evaluate_mp requires extended precision")
        return self._evaluate(points,operator)

    def evaluate(self,points,operator=None):
        return np.array([float(v) for v in self._evaluate(points,operator)])


def dense_arithmetic(kernel,precision):
    if not isinstance(precision,Precision):
        raise TypeError("precision must be Precision")
    if precision.local_digits is not None or precision.global_dtype!="float64":
        raise ValueError("Dense nodal methods use Precision(global_digits=...)")
    return Arithmetic(kernel,precision.global_digits)


def interpolation(kernel,centers,data,precision,degree):
    a=dense_arithmetic(kernel,precision)
    basis=NodalBasis(a,centers,degree)
    return DenseNodalSystem(basis,basis.centers,basis.identities,a.data(data,basis.centers)).solve()


def collocation_data(problem,cloud,arithmetic):
    """One PDE/boundary equation per node, shared by Kansa and RBF-FD."""
    a=arithmetic
    if a.ctx:
        for data in [problem.rhs]+[bc.rhs for bc in problem.boundary]:
            if callable(data) and not hasattr(data,"mp_values"):
                raise TypeError("Extended nodal methods require PrecisionData callbacks")
    bd=boundary_data(problem,cloud,ctx=a.ctx)
    ops=[bd[i][0] if i in bd else problem.operator for i in range(len(cloud.points))]
    forcing=a.data(problem.rhs,cloud.points)
    data=[bd[i][1] if i in bd else forcing[i] for i in range(len(cloud.points))]
    return ops,data


def assemble_global(kernel,problem,cloud,precision,degree=None,scheme="asymmetric"):
    a=dense_arithmetic(kernel,precision)
    ops,data=collocation_data(problem,cloud,a)
    basis=NodalBasis(a,cloud.points,degree,source_operators=ops if scheme=="symmetric" else None)
    return DenseNodalSystem(basis,cloud.points,ops,data)


class LocalNodalStencil:
    """PDE-independent interpolation system reused for any target functional."""
    def __init__(self,arithmetic,points,degree,kernel_scaling="physical"):
        self.basis=NodalBasis(arithmetic,points,degree,kernel_scaling=kernel_scaling)
        self.factor=arithmetic.factor(self.basis.system_matrix(points,self.basis.identities))

    def weights(self,point,operator):
        a=self.basis.arithmetic
        q=self.basis.evaluation(np.asarray(point).reshape(1,-1),[operator]).T
        if not a.ctx:
            q=q[:,0]
        result=self.factor.solve(q,transpose=True)
        return result[:len(self.basis.centers)]


def rbf_fd_weights(kernel,centers,target,operator,polynomial_degree=2,precision=None,*,kernel_scaling="physical"):
    """Local derivative weights independent of a PDE or boundary description.

    Returns Float64 weights, or unrounded mpmath weights when local_digits is set.
    The global_dtype setting applies only to subsequent global assembly.
    """
    precision=precision or Precision()
    if not isinstance(precision,Precision):
        raise TypeError("precision must be Precision")
    if precision.global_digits is not None:
        raise ValueError("Local weights use local_digits")
    a=Arithmetic(kernel,precision.local_digits)
    return LocalNodalStencil(a,centers,polynomial_degree,kernel_scaling).weights(target,operator)
