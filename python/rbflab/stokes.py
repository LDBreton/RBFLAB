"""Stationary 2D Stokes with global divergence-free Hermite collocation.

- nu * Laplacian(u) + grad(p) = f, div(u) = 0.
Velocity Dirichlet data on the entire boundary. Pressure fixed at one point.
A constant pressure basis and its side constraint preserve the pressure gauge.
"""
from dataclasses import dataclass,field
import numpy as np
from .operators import Identity,Derivative,Laplacian
from .kernels import ScalarKernel
from .precision import Precision
from .nodal import Arithmetic,dense_arithmetic
from .methods import _query


@dataclass(frozen=True)
class StokesProblem:
    forcing: tuple
    boundary_velocity: tuple
    viscosity: object = 1
    pressure_point: tuple = (0.,0.)
    pressure_value: object = 0

    def __post_init__(self):
        from .operators import _coefficient
        if len(self.forcing)!=2 or len(self.boundary_velocity)!=2:
            raise ValueError("Expected two scalar forcing and boundary components")
        if _coefficient(self.viscosity)<=0:
            raise ValueError("viscosity must be positive")
        _query([self.pressure_point], dimension=2)


class StokesArithmetic(Arithmetic):
    """Separate scalar potentials for velocity and pressure block kernels."""
    def __init__(self, velocity_kernel, pressure_kernel=None, digits=None):
        super().__init__(velocity_kernel,digits)
        self.pressure_arithmetic = self if pressure_kernel is None else Arithmetic(pressure_kernel,digits)
        if self.ctx and self.pressure_arithmetic is not self:
            # mpmath matrices belong to their context; both blocks must share it.
            self.pressure_arithmetic.ctx = self.ctx
            self.pressure_arithmetic.backend.ctx = self.ctx


def _blocks(a,x,left,y,right,length_scale=None):
    """Apply vector functionals to diag(divergence-free K, scalar K)."""
    dimension=np.asarray(x).shape[1]
    derivatives=[Derivative(i,dimension) for i in range(dimension)]
    lap=Laplacian(dimension)
    kernel_ops={(i,j):derivatives[i]@derivatives[j]-(lap if i==j else 0*Identity(dimension))
                for i in range(dimension) for j in range(dimension)}
    kernel_ops[(dimension,dimension)]=Identity(dimension)
    out=a.zeros(len(x),len(y))
    for (component,source),kernel_op in kernel_ops.items():
        rows=[i for i,op in enumerate(left) if component in op]
        cols=[j for j,op in enumerate(right) if source in op]
        if not rows or not cols:
            continue
        block_arithmetic=getattr(a,"pressure_arithmetic",a) if component==dimension else a
        block=block_arithmetic.matrix(x[rows],[left[i][component]@kernel_op for i in rows],
                       y[cols],[right[j][source] for j in cols],length_scale=length_scale)
        if a.ctx:
            for i,row in enumerate(rows):
                for j,col in enumerate(cols):out[row,col]+=block[i,j]
        else:
            out[np.ix_(rows,cols)]+=block
    return out


@dataclass
class GlobalStokes:
    kernel: object
    precision: Precision = field(default_factory=Precision)

    def assemble(self,problem,cloud):
        if cloud.dimension != 2:
            raise NotImplementedError("Stokes solvers currently support 2D only")
        if not isinstance(problem,StokesProblem):
            raise TypeError("Expected StokesProblem")
        if not isinstance(self.kernel,ScalarKernel):
            raise NotImplementedError("Stokes currently supports smooth IMQ/Gaussian kernels; vector PHS augmentation is not implemented")
        a=dense_arithmetic(self.kernel,self.precision)
        ii,bi=cloud.interior_indices,cloud.boundary_indices
        if not len(ii) or not len(bi):
            raise ValueError("Stokes requires interior and boundary points")
        points=[];functionals=[];data=[]
        for component in (0,1):
            points.extend(cloud.points[ii])
            functionals.extend([{component:-(Laplacian()*problem.viscosity),2:Derivative(component)} for _ in ii])
            data.extend(a.data(problem.forcing[component],cloud.points[ii]))
        for component in (0,1):
            points.extend(cloud.points[bi])
            functionals.extend([{component:Identity()} for _ in bi])
            data.extend(a.data(problem.boundary_velocity[component],cloud.points[bi]))
        points.append(problem.pressure_point);functionals.append({2:Identity()})
        data.append(a.number(problem.pressure_value))
        points=np.asarray(points,dtype=float)
        n=len(points)
        matrix=a.zeros(n+1,n+1)
        matrix[:n,:n]=_blocks(a,points,functionals,points,functionals)
        # Only the pressure-value functional sees the constant pressure mode.
        matrix[n-1,n]=matrix[n,n-1]=a.number(1)
        rhs=a.vector(data+[a.number(0)])
        return StokesSystem(a,problem,cloud,points,functionals,matrix,rhs)


class StokesSystem:
    def __init__(self,a,problem,cloud,points,functionals,matrix,rhs):
        self.arithmetic,self.problem,self.cloud=a,problem,cloud
        self.points,self.functionals=points,functionals
        self.matrix,self.rhs,self.factor=matrix,rhs,None

    def solve(self):
        a=self.arithmetic
        if self.factor is None:self.factor=a.factor(self.matrix)
        coefficients=self.factor.solve(self.rhs)
        defect=(self.matrix*coefficients if a.ctx else self.matrix@coefficients)-self.rhs
        denominator=a.norm(self.rhs)
        residual=a.norm(defect)/(denominator or a.number(1))
        return StokesSolution(self,coefficients,{
            "relative_residual":float(residual),
            "scaled_condition":self.factor.condition,
            "condition_norm":"infinity" if a.ctx else "2",
            "global_digits":a.ctx.dps if a.ctx else None,
            "unknowns":len(coefficients),"pressure_gauge":"point value plus constant mode"})


class StokesSolution:
    def __init__(self,system,coefficients,diagnostics):
        self.system,self.coefficients,self.diagnostics=system,coefficients,diagnostics

    def evaluate(self,points,operator=None,*,extended=False):
        """Return (N,3) columns u_x, u_y, p, or their requested derivative."""
        points=_query(points, dimension=2);op=operator or Identity();a=self.system.arithmetic
        if extended and not a.ctx:raise ValueError("extended requires global_digits")
        out=a.zeros(len(points),3)
        if not len(points):return out if extended else np.empty((0,3))
        n=len(self.system.points)
        for component in (0,1,2):
            block=_blocks(a,points,[{component:op} for _ in points],
                          self.system.points,self.system.functionals)
            if a.ctx:
                values=block*a.ctx.matrix([self.coefficients[i] for i in range(n)])
            else:values=block@self.coefficients[:n]
            constant=sum(a.number(c) for alpha,c in op.terms if sum(alpha)==0)
            for i in range(len(points)):
                out[i,component]=values[i]+(constant*self.coefficients[n] if component==2 else 0)
        if extended:return out
        return np.array(out.tolist(),dtype=float) if a.ctx else out

    def residuals(self,points,*,extended=False):
        """Return (N,3): momentum-x, momentum-y, divergence, in physical units."""
        points=_query(points, dimension=2);a=self.system.arithmetic
        if extended and not a.ctx:raise ValueError("extended requires global_digits")
        use_mp=bool(a.ctx)
        lap=self.evaluate(points,Laplacian(),extended=use_mp)
        dx=self.evaluate(points,Derivative(0),extended=use_mp)
        dy=self.evaluate(points,Derivative(1),extended=use_mp)
        fx=a.data(self.system.problem.forcing[0],points)
        fy=a.data(self.system.problem.forcing[1],points)
        out=a.zeros(len(points),3);nu=a.number(self.system.problem.viscosity)
        for i in range(len(points)):
            out[i,0]=-nu*lap[i,0]+dx[i,2]-fx[i]
            out[i,1]=-nu*lap[i,1]+dy[i,2]-fy[i]
            out[i,2]=dx[i,0]+dy[i,1]
        if extended:return out
        return np.array(out.tolist(),dtype=float) if a.ctx else out
