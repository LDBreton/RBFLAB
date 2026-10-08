"""Stationary block Hermite collocation and local Hermite assembly.

Fields use independent scalar kernels. Differential coupling lives in explicit
functional rows, including incompressibility; it is not built into the kernels.
"""
from .symbolic_kernel import BoundKernel
import warnings
from dataclasses import dataclass,field
import numpy as np
from scipy.spatial import cKDTree
from .operators import Identity
from .kernels import ScalarKernel, PHS, Hybrid
from .nodal import Arithmetic,NodalBasis
from .precision import Precision
from .stencils import StencilPolicy
from .symbolic_system import BlockPDE
from .evolution import _sparse,_factor,_mv


def _kernels(selection,problem):
    if not isinstance(selection,dict):return [selection]*len(problem.fields)
    resolved={}
    for key,kernel in selection.items():
        keys=key if isinstance(key,tuple) else (key,)
        for item in keys:
            index=problem.field_index(item)
            if index in resolved:raise ValueError("Kernel assigned more than once to a field")
            resolved[index]=kernel
    if len(resolved)!=len(problem.fields):raise ValueError("Supply a kernel for every field")
    return [resolved[i] for i in range(len(problem.fields))]


def _arithmetic(kernels,digits):
    if any(not isinstance(kernel,(ScalarKernel,PHS,Hybrid,BoundKernel)) for kernel in kernels):
        raise TypeError("Block fields require scalar IMQ, Gaussian, PHS or Hybrid kernels")
    arithmetics=[Arithmetic(kernel,digits) for kernel in kernels]
    master=arithmetics[0]
    if master.ctx:
        for a in arithmetics[1:]:a.ctx=master.ctx;a.backend.ctx=master.ctx
    return arithmetics


def _validate(problem,cloud):
    if not isinstance(problem,BlockPDE):raise TypeError("Expected a compiled BlockPDE")
    if cloud.dimension!=problem.dimension:raise ValueError("Cloud and block problem dimensions must match")
    if not len(cloud.interior):raise ValueError("Block PDEs require interior nodes")
    for c in problem.constraints:
        if not np.isfinite(np.asarray(c.point,dtype=float)).all():raise ValueError("Constraint points must be finite")


def _boundary(problem,cloud,a):
    result={}
    for bc in problem.boundary:
        labels=list(cloud.boundary) if bc.on=='boundary' else ([bc.on] if isinstance(bc.on,str) else list(bc.on))
        for label in labels:
            if label not in cloud.boundary:raise ValueError(f"Unknown boundary label: {label}")
            ids=cloud.boundary[label];data=bc.rhs
            if getattr(data,'requires_normals',False):
                if label not in cloud.normals:raise ValueError(f"Normals required on {label}")
                data=data.bind_normals(cloud.normals[label])
            vals=a.data(data,cloud.points[ids])
            for j,node in enumerate(ids):
                key=(int(node),bc.slot)
                if key in result:continue
                normal=cloud.normals[label][j] if label in cloud.normals else None
                ops={}
                for f,op in bc.operators.items():
                    if getattr(op,'requires_normals',False) and normal is None:raise ValueError(f"Normals required on {label}")
                    ops[f]=op.at_point(cloud.points[node],normal)
                if not any(op.terms for op in ops.values()):raise ValueError("Zero boundary functional")
                result[key]=(ops,vals[j])
    if {node for node,slot in result}!=set(cloud.boundary_indices):raise ValueError("Every boundary node needs a block boundary condition")
    return result


class BlockBasis:
    """Hermite representers of block functionals plus a polynomial tail per field."""
    def __init__(self,arithmetics,points,functionals,degree,scaling='physical'):
        self.arithmetics=arithmetics;self.a=arithmetics[0]
        self.points=np.asarray(points,dtype=float);self.functionals=functionals
        self.n=len(points);self.bases=[];self.offsets=[];offset=self.n
        for f,a in enumerate(arithmetics):
            ids=[i for i,row in enumerate(functionals) if f in row]
            if not ids:raise ValueError(f"Field {f} has no source functionals")
            basis=NodalBasis(a,self.points[ids],degree,source_operators=[functionals[i][f] for i in ids],kernel_scaling=scaling)
            self.bases.append((ids,basis));self.offsets.append(offset);offset+=len(basis.powers)
        self.size=offset

    def evaluation(self,points,functionals):
        points=np.asarray(points,dtype=float);out=self.a.zeros(len(points),self.size)
        for f,(sources,basis) in enumerate(self.bases):
            targets=[i for i,row in enumerate(functionals) if f in row]
            if not targets:continue
            block=basis.evaluation(points[targets],[functionals[i][f] for i in targets])
            cols=sources+list(range(self.offsets[f],self.offsets[f]+len(basis.powers)))
            if self.a.ctx:
                for i,row in enumerate(targets):
                    for j,col in enumerate(cols):out[row,col]+=block[i,j]
            else:out[np.ix_(targets,cols)]+=block
        return out

    def matrix(self):
        out=self.a.zeros(self.size,self.size)
        out[:self.n,:]=self.evaluation(self.points,self.functionals)
        P=out[:self.n,self.n:]
        if self.size>self.n:
            # Check polynomial unisolvency before numerical factorization.
            check=np.array(P.tolist() if self.a.ctx else P,dtype=float)
            scale=np.max(abs(check),axis=0)
            check=check/np.where(scale>0,scale,1)
            if np.linalg.matrix_rank(check)<self.size-self.n:
                raise np.linalg.LinAlgError("Block polynomial functionals are not unisolvent; increase the stencil or change the polynomial degree")
            out[self.n:,:self.n]=P.T
        return out

    def padded(self,data):return self.a.vector(list(data)+[self.a.number(0)]*(self.size-self.n))


@dataclass
class BlockGlobal:
    """Symmetric block Hermite collocation; one scalar kernel or a field mapping."""
    kernels: object
    polynomial_degree: int | None = None
    precision: Precision = field(default_factory=Precision)

    def assemble(self,problem,cloud):
        _validate(problem,cloud)
        if not isinstance(self.precision,Precision):raise TypeError("Expected Precision")
        if self.precision.local_digits is not None or self.precision.global_dtype!='float64':raise ValueError("BlockGlobal uses global_digits")
        kernels=_kernels(self.kernels,problem);arithmetics=_arithmetic(kernels,self.precision.global_digits);a=arithmetics[0]
        points=[];ops=[];rhs=[]
        for eq in problem.equations:
            points.extend(cloud.interior);ops.extend([eq.operators]*len(cloud.interior));rhs.extend(a.data(eq.rhs,cloud.interior))
        for (node,slot),(row,value) in _boundary(problem,cloud,a).items():
            points.append(cloud.points[node]);ops.append(row);rhs.append(value)
        for c in problem.constraints:
            point=np.asarray(c.point,dtype=float)[None,:]
            points.append(point[0]);ops.append({c.field:Identity(problem.dimension)});rhs.extend(a.data(c.value,point))
        basis=BlockBasis(arithmetics,points,ops,self.polynomial_degree)
        system=BlockGlobalSystem(problem,basis,basis.matrix(),basis.padded(rhs))
        return system


class BlockGlobalSystem:
    def __init__(self,problem,basis,matrix,rhs):
        self.problem,self.basis,self.matrix,self.rhs=problem,basis,matrix,rhs
        self.arithmetic=basis.a;self.factor=None

    def solve(self):
        if self.factor is None:self.factor=self.arithmetic.factor(self.matrix)
        z=self.factor.solve(self.rhs);a=self.arithmetic
        diagnostics={'relative_residual':float(a.norm(_mv(self.matrix,z)-self.rhs)/(a.norm(self.rhs) or a.number(1))),
                     'unknowns':len(z),'scaled_condition':self.factor.condition,'method':'block_global',
                     'digits':a.ctx.dps if a.ctx else None}
        return BlockSolution(self,z,diagnostics)


@dataclass
class BlockLHI:
    """Experimental block LHI, interior values and local PDE/BC constraints.

    A point gauge adds a bordered row and a compatibility multiplier in its
    specified equation slot. Inspect constraint_multipliers in diagnostics.
    """
    kernels: object
    stencil_size: int = 25
    polynomial_degree: int | None = None
    precision: Precision = field(default_factory=Precision)
    stencil_policy: StencilPolicy = field(default_factory=StencilPolicy)

    def assemble(self,problem,cloud):
        _validate(problem,cloud)
        if not isinstance(self.precision,Precision):raise TypeError("Expected Precision")
        if self.precision.global_digits is not None:raise ValueError("BlockLHI uses local_digits and global_dtype")
        if type(self.stencil_size) is not int or not 3<=self.stencil_size<=len(cloud.points):raise ValueError("Invalid stencil_size")
        if not isinstance(self.stencil_policy,StencilPolicy):raise TypeError("Expected StencilPolicy")
        kernels=_kernels(self.kernels,problem);arithmetics=_arithmetic(kernels,self.precision.local_digits)
        local=arithmetics[0];a=local if self.precision.global_dtype=='mpmath' else Arithmetic(kernels[0])
        bd=_boundary(problem,cloud,local);ni=len(cloud.interior);nf=len(problem.fields);nc=len(problem.constraints)
        interior={int(node):i for i,node in enumerate(cloud.interior_indices)}
        rhs_data=[local.data(eq.rhs,cloud.points) for eq in problem.equations]
        size=nf*ni+nc;rows=[{} for _ in range(size)];rhs=[a.number(0)]*size
        tree=cKDTree(cloud.points);stencils=[];identity=Identity(problem.dimension)
        for center in cloud.interior_indices:
            neighbors=self.stencil_policy.select(tree,cloud.points[center],self.stencil_size,self.polynomial_degree)
            selected=set(int(j) for j in neighbors);sc=[int(j) for j in neighbors if int(j) in interior]
            points=[];ops=[];unknowns=[];known=[]
            for f in range(nf):
                for node in sc:
                    points.append(cloud.points[node]);ops.append({f:identity})
                    unknowns.append(f*ni+interior[node]);known.append(local.number(0))
            ns=len(unknowns)
            for (node,slot),(row,value) in bd.items():
                if node in selected:points.append(cloud.points[node]);ops.append(row);known.append(value)
            for f,eq in enumerate(problem.equations):
                for node in sc:
                    if node!=center:
                        points.append(cloud.points[node]);ops.append(eq.operators);known.append(rhs_data[f][node])
            basis=BlockBasis(arithmetics,points,ops,self.polynomial_degree,self.stencil_policy.scaling)
            factor=local.factor(basis.matrix())
            q=basis.evaluation(np.repeat(cloud.points[[center]],nf,axis=0),[eq.operators for eq in problem.equations]).T
            weights=factor.solve(q,transpose=True).T[:,:basis.n]
            stencil=dict(basis=basis,factor=factor,unknowns=unknowns,known=known,ns=ns)
            stencils.append(stencil)
            for f in range(nf):
                row=f*ni+interior[int(center)];rhs[row]=a.number(rhs_data[f][center])
                for j,col in enumerate(unknowns):rows[row][col]=a.number(weights[f,j])
                rhs[row]-=sum(a.number(weights[f,j])*a.number(known[j]) for j in range(ns,len(known)))
        owners=cKDTree(cloud.interior)
        for k,c in enumerate(problem.constraints):
            point=np.asarray(c.point,dtype=float)[None,:];_,owner=owners.query(point[0]);s=stencils[int(owner)]
            q=s['basis'].evaluation(point,[{c.field:identity}]).T
            w=s['factor'].solve(q,transpose=True)
            row=nf*ni+k
            for j,col in enumerate(s['unknowns']):rows[row][col]=a.number(w[j,0])
            rhs[row]=a.number(local.data(c.value,point)[0])-sum(a.number(w[j,0])*a.number(s['known'][j]) for j in range(s['ns'],len(s['known'])))
            for i in range(ni):rows[c.equation*ni+i][row]=a.number(1)
        return BlockLHISystem(problem,cloud,a,local,_sparse(rows,a),a.vector(rhs),stencils,owners,nf*ni)


class BlockLHISystem:
    def __init__(self,problem,cloud,a,local,matrix,rhs,stencils,tree,physical_unknowns):
        self.problem,self.cloud,self.arithmetic,self.local=problem,cloud,a,local
        self.matrix,self.rhs,self.stencils,self.tree=matrix,rhs,stencils,tree
        self.physical_unknowns=physical_unknowns;self.factor=None

    def solve(self):
        a=self.arithmetic
        if self.factor is None:self.factor=_factor(self.matrix,a)
        condition=None
        if not a.ctx and len(self.rhs)<=400:
            condition=float(np.linalg.cond(self.matrix.toarray()))
            if condition>1e12:
                warnings.warn("Ill-conditioned mixed LHI system; a small algebraic residual does not establish a unique accurate field",RuntimeWarning,stacklevel=2)
        z=self.factor.solve(self.rhs)
        multipliers=[float(v) for v in z[self.physical_unknowns:]]
        if any(abs(v)>1e-6*(1+float(a.norm(self.rhs))) for v in multipliers):
            warnings.warn("Mixed LHI has non-negligible constraint multipliers: the original PDE rows are not satisfied exactly; inspect residuals",RuntimeWarning,stacklevel=2)
        diagnostics={'experimental':True,'global_condition':condition,'relative_residual':float(a.norm(_mv(self.matrix,z)-self.rhs)/(a.norm(self.rhs) or a.number(1))),
            'unknowns':len(z),'method':'block_lhi','local_digits':self.local.ctx.dps if self.local.ctx else None,
            'global_digits':a.ctx.dps if a.ctx else None,
            'constraint_multipliers':multipliers,
            'max_local_condition':max(s['factor'].condition for s in self.stencils)}
        return BlockSolution(self,z,diagnostics)


class BlockSolution:
    def __init__(self,system,unknowns,diagnostics):
        self.system,self.unknowns,self.diagnostics=system,unknowns,diagnostics
        if isinstance(system,BlockLHISystem):
            self.a=system.local;self.coefficients=[]
            for s in system.stencils:
                data=[self.a.number(unknowns[j]) for j in s['unknowns']]+s['known'][s['ns']:]
                self.coefficients.append(s['factor'].solve(s['basis'].padded(data)))
        else:self.a=system.arithmetic

    def _rows(self,points,functionals):
        from .methods import _query
        points=_query(points,self.system.problem.dimension)
        if len(functionals)!=len(points):raise ValueError("One functional per query point required")
        if isinstance(self.system,BlockGlobalSystem):
            return _mv(self.system.basis.evaluation(points,functionals),self.unknowns)
        result=self.a.vector([0]*len(points))
        if not len(points):return result
        _,owners=self.system.tree.query(points)
        for owner in np.unique(owners):
            ids=np.flatnonzero(owners==owner);s=self.system.stencils[int(owner)]
            block=s['basis'].evaluation(points[ids],[functionals[int(i)] for i in ids])
            values=_mv(block,self.coefficients[int(owner)])
            for j,i in enumerate(ids):result[int(i)]=values[j]
        return result

    def evaluate(self,points,*,field=None,operator=None,extended=False):
        from .methods import _query
        points=_query(points,self.system.problem.dimension)
        if extended and not self.a.ctx:raise ValueError("Extended evaluation requires extended arithmetic")
        p=self.system.problem;fields=range(len(p.fields)) if field is None else [p.field_index(field)]
        op=operator or Identity(p.dimension);out=self.a.zeros(len(points),len(fields))
        for j,f in enumerate(fields):
            values=self._rows(points,[{f:op}]*len(points))
            for i,v in enumerate(values):out[i,j]=v
        if extended:return out
        result=np.asarray(out.tolist() if self.a.ctx else out,dtype=float).reshape(len(points),len(fields))
        return result[:,0] if field is not None else result

    def residuals(self,points,*,extended=False):
        if extended and not self.a.ctx:raise ValueError("Extended evaluation requires extended arithmetic")
        p=self.system.problem;out=self.a.zeros(len(points),len(p.equations))
        for j,eq in enumerate(p.equations):
            vals=self._rows(points,[eq.operators]*len(points));rhs=self.a.data(eq.rhs,points)
            for i,v in enumerate(vals):out[i,j]=v-rhs[i]
        return out if extended else np.asarray(out.tolist() if self.a.ctx else out,dtype=float)
