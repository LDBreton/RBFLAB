"""Correctness-first sparse arbitrary-precision assembly and pivoted LU.

Dictionary rows preserve sparsity; elimination may introduce substantial fill-in.
No Float64 matrix, SciPy solve, numerical dropping or dense conversion is used.
"""
import numpy as np
from .precision import mp_number, mp_values
from .problems import boundary_data


class MPSparseMatrix:
    def __init__(self, ctx, rows, ncols=None):
        if ncols is not None and (type(ncols) is not int or ncols<0):raise ValueError("Invalid sparse column count")
        self.ctx=ctx
        self.rows=[{int(j):mp_number(ctx,v) for j,v in row.items() if v != 0}
                   for row in rows]
        self.shape=(len(rows),len(rows) if ncols is None else ncols)
        if any(j<0 or j>=self.shape[1] for row in self.rows for j in row):
            raise ValueError("Invalid sparse column")
        if any(not ctx.isfinite(v) for row in self.rows for v in row.values()):
            raise ValueError("Nonfinite sparse matrix")

    @property
    def nnz(self):
        return sum(len(row) for row in self.rows)

    def matvec(self, vector):
        if len(vector)!=self.shape[1]:raise ValueError("Sparse input dimension mismatch")
        vector=[mp_number(self.ctx,v) for v in vector]
        return self.ctx.matrix([self.ctx.fsum(v*vector[j] for j,v in row.items())
                                for row in self.rows])


    def __matmul__(self, values):
        if isinstance(values, MPSparseMatrix):
            if self.shape[1] != values.shape[0]:
                raise ValueError('Sparse product dimension mismatch')
            self._same_precision(values)
            rows = []
            for row in self.rows:
                result = {}
                for k, left in row.items():
                    for j, right in values.rows[k].items():
                        result[j] = result.get(j, self.ctx.zero) + left*mp_number(self.ctx, right)
                rows.append(result)
            return MPSparseMatrix(self.ctx, rows, ncols=values.shape[1])
        data=np.asarray(values.tolist() if hasattr(values,"rows") else values,dtype=object)
        if data.ndim==1:return self.matvec(data)
        if data.ndim!=2 or data.shape[0]!=self.shape[1]:raise ValueError("Sparse input dimension mismatch")
        out=self.ctx.matrix(self.shape[0],data.shape[1])
        for j in range(data.shape[1]):out[:,j]=self.matvec(data[:,j])
        return out


    def _same_precision(self, other):
        if self.ctx.dps != other.ctx.dps:
            raise ValueError('Sparse arithmetic requires matching precision contexts')

    def copy(self):
        """Copy sparse coefficients without rounding."""
        return MPSparseMatrix(self.ctx, self.rows, ncols=self.shape[1])

    @classmethod
    def eye(cls, ctx, size):
        """Construct an identity matrix in an explicit mpmath context."""
        if type(size) is not int or size < 0:
            raise ValueError('Identity size must be a nonnegative integer')
        return cls(ctx, [{i:ctx.one} for i in range(size)], ncols=size)

    @property
    def T(self):
        """Sparse transpose retaining the arithmetic context."""
        rows = [{} for _ in range(self.shape[1])]
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                rows[j][i] = value
        return MPSparseMatrix(self.ctx, rows, ncols=self.shape[0])

    def __add__(self, other):
        if not isinstance(other, MPSparseMatrix):
            if np.isscalar(other) and other == 0:
                return self.copy()
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError('Sparse addition dimension mismatch')
        self._same_precision(other)
        rows = [row.copy() for row in self.rows]
        for row, source in zip(rows, other.rows):
            for j, value in source.items():
                row[j] = row.get(j, self.ctx.zero) + mp_number(self.ctx, value)
        return MPSparseMatrix(self.ctx, rows, ncols=self.shape[1])

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, scalar):
        if isinstance(scalar, MPSparseMatrix):
            return NotImplemented
        value = mp_number(self.ctx, scalar)
        return MPSparseMatrix(self.ctx,
            [{j:v*value for j,v in row.items()} for row in self.rows], ncols=self.shape[1])

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (self.ctx.one/mp_number(self.ctx, scalar))


class MPSparseLU:
    """Row-equilibrated sparse LU with partial row pivoting and retained fill."""
    def __init__(self, matrix):
        if matrix.shape[0]!=matrix.shape[1]:raise ValueError("Sparse LU requires a square matrix")
        self.matrix,self.ctx=matrix,matrix.ctx
        ctx=self.ctx
        n=matrix.shape[0]
        self.scale=[]
        for row in matrix.rows:
            largest=max((abs(v) for v in row.values()),default=ctx.zero)
            if largest==0:
                raise np.linalg.LinAlgError("Empty/zero equation in sparse system")
            self.scale.append(1/largest)
        self.rows=[{j:v*self.scale[i] for j,v in row.items()}
                   for i,row in enumerate(matrix.rows)]
        self.permutation=list(range(n))
        for k in range(n):
            pivot=max(range(k,n),key=lambda i:abs(self.rows[i].get(k,ctx.zero)))
            if self.rows[pivot].get(k,ctx.zero)==0:
                raise np.linalg.LinAlgError("Singular sparse system at requested precision")
            if pivot!=k:
                self.rows[k],self.rows[pivot]=self.rows[pivot],self.rows[k]
                self.permutation[k],self.permutation[pivot]=self.permutation[pivot],self.permutation[k]
            upper=[(j,v) for j,v in self.rows[k].items() if j>k]
            for i in range(k+1,n):
                if k not in self.rows[i]:
                    continue
                multiplier=self.rows[i][k]/self.rows[k][k]
                self.rows[i][k]=multiplier
                for j,v in upper:
                    updated=self.rows[i].get(j,ctx.zero)-multiplier*v
                    if updated==0:
                        self.rows[i].pop(j,None)
                    else:
                        self.rows[i][j]=updated
        self.factor_nnz=sum(len(row) for row in self.rows)

    def solve(self, rhs):
        ctx=self.ctx;n=len(self.rows)
        rhs=list(rhs)
        if len(rhs)!=n:
            raise ValueError("RHS length mismatch")
        b=[mp_number(ctx,v)*self.scale[i] for i,v in enumerate(rhs)]
        y=[ctx.zero]*n
        for i,row in enumerate(self.rows):
            y[i]=b[self.permutation[i]]-ctx.fsum(v*y[j] for j,v in row.items() if j<i)
        x=[ctx.zero]*n
        for i in range(n-1,-1,-1):
            row=self.rows[i]
            x[i]=(y[i]-ctx.fsum(v*x[j] for j,v in row.items() if j>i))/row[i]
        return ctx.matrix(x)

    def residual(self, x, rhs):
        numerator=self.ctx.norm(self.matrix.matvec(x)-rhs,"inf")
        denominator=self.ctx.norm(rhs,"inf")
        return numerator/denominator if denominator else numerator

    def condition_inf(self):
        """Original-system infinity norm, inverse row sums streamed one column at a time."""
        ctx=self.ctx;n=len(self.rows)
        sums=[ctx.zero]*n
        for j in range(n):
            rhs=ctx.matrix(n,1);rhs[j]=1
            col=self.solve(rhs)
            for i in range(n):
                sums[i]+=abs(col[i])
        norm=max(ctx.fsum(abs(v) for v in row.values()) for row in self.matrix.rows)
        return norm*max(sums)


def assemble_sparse_system(kernel, cloud, problem, stencils):
    backend=stencils[0].factor.backend
    ctx=backend.ctx
    # No silently promoted Float64 callback values in the full-precision mode.
    for data in [problem.rhs]+[bc.rhs for bc in problem.boundary]:
        if callable(data) and not hasattr(data,"mp_values"):
            raise TypeError("Full-precision LHI data callbacks must use PrecisionData")
    bd=boundary_data(problem,cloud,ctx=ctx)
    forcing=mp_values(problem.rhs,cloud.points,ctx)
    ii=cloud.interior_indices
    mapping={int(j):i for i,j in enumerate(ii)}
    rows=[{} for _ in ii]
    rhs=ctx.matrix([forcing[j] for j in ii])
    for row,s in enumerate(stencils):
        weights=s.weights_mp
        for j,w in zip(s.solution_indices,weights[:len(s.solution_indices)]):
            col=mapping[int(j)]
            rows[row][col]=rows[row].get(col,ctx.zero)+w
        known=s.known_data if s.known_data is not None else [bd[int(j)][1] for j in s.boundary_indices]+[forcing[j] for j in s.pde_indices]
        rhs[row]-=ctx.fsum(weights[len(s.solution_indices)+j]*v for j,v in enumerate(known))
    return MPSparseLHISystem(kernel,cloud,problem,bd,forcing,stencils,
                            MPSparseMatrix(ctx,rows),rhs)


class MPSparseLHISystem:
    def __init__(self,kernel,cloud,problem,bd,forcing,stencils,matrix,rhs):
        self.kernel,self.cloud,self.problem=kernel,cloud,problem
        self.boundary_data,self.forcing,self.stencils=bd,forcing,stencils
        self.matrix,self.rhs,self.factor=matrix,rhs,None

    def solve(self):
        from .reference import MPLHIReferenceSolution
        if self.factor is None:
            self.factor=MPSparseLU(self.matrix)
        u=self.factor.solve(self.rhs)
        ctx=self.matrix.ctx
        residual=self.factor.residual(u,self.rhs)
        diagnostics={
            "local_digits":ctx.dps,"global_digits":ctx.dps,
            "global_solver":"mpmath sparse dictionary LU",
            "matrix_nnz":self.matrix.nnz,"factor_nnz":self.factor.factor_nnz,
            "relative_residual":float(residual),"residual_decimal":ctx.nstr(residual,12),
            "max_local_weight_residual":max(s.weight_residual for s in self.stencils),
            "max_scaled_local_condition":max(s.factor.condition for s in self.stencils),
            "local_condition_norm":"infinity",
        }
        return MPLHIReferenceSolution(self,u,self.boundary_data,self.forcing,diagnostics)

    def solve_reference(self):
        """Independent dense solve of the same unrounded matrix; only for small tests."""
        from .precision import MPFactor
        from .reference import MPLHIReferenceSolution
        if self.matrix.shape[0]>200:
            raise ValueError("Dense reference limited to 200 unknowns")
        ctx=self.matrix.ctx
        dense=ctx.matrix(self.matrix.shape[0])
        for i,row in enumerate(self.matrix.rows):
            for j,v in row.items():
                dense[i,j]=v
        factor=MPFactor(self.stencils[0].factor.backend,dense)
        u=factor.solve(self.rhs)
        return MPLHIReferenceSolution(self,u,self.boundary_data,self.forcing,
                                       {"global_digits":ctx.dps,"global_solver":"dense reference"})
