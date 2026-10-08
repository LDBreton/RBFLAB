"""Reusable nodal differentiation, including matrix-valued solenoidal spaces."""
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
from types import SimpleNamespace
import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from .nodal import Arithmetic, NodalBasis
from .operators import Identity, Derivative, Laplacian, bind_operators
from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace
from .stencils import polynomial_powers, geometry_quality
from .sparse_precision import MPSparseMatrix


def resolve_space(method, space=None):
    if method.spaces is None:
        if space is not None: raise ValueError('space requires an RBFFD spaces mapping')
        if method.kernel is None: raise ValueError('Specify kernel or spaces')
        return ScalarSpace(method.kernel, method.polynomial_degree)
    if method.kernel is not None: raise ValueError('Specify kernel or spaces, not both')
    if not isinstance(method.spaces, Mapping) or not method.spaces:
        raise ValueError('spaces must be a nonempty field-to-space mapping')
    if space is None:
        if len(method.spaces)!=1: raise ValueError('Select a field with space= when multiple spaces are declared')
        descriptor=next(iter(method.spaces.values()))
    else: descriptor=method.spaces[space]
    if not isinstance(descriptor, ScalarSpace): raise TypeError('Expected ScalarSpace, PressureSpace or DivergenceFreeSpace')
    return descriptor


def scalar_method(method):
    descriptor=resolve_space(method)
    if isinstance(descriptor,DivergenceFreeSpace):
        raise NotImplementedError('Vector spaces are supported by RBFFD.operators; mixed PDE assembly is separate')
    degree=descriptor.degree()
    if isinstance(descriptor,PressureSpace) and degree is None:degree=0
    return replace(method,kernel=descriptor.kernel,polynomial_degree=degree,spaces=None)


@lru_cache(None)
def solenoidal_columns(dimension,degree):
    """Exact nullspace of polynomial divergence, independent of the cloud."""
    import sympy as sp
    if degree is None:return ()
    powers=polynomial_powers(dimension,degree)
    lower=polynomial_powers(dimension,degree-1) if degree else []
    D=sp.zeros(len(lower),dimension*len(powers))
    for c in range(dimension):
        for j,p in enumerate(powers):
            if p[c]:
                q=list(p);q[c]-=1
                D[lower.index(tuple(q)),c*len(powers)+j]=p[c]
    return tuple(tuple(int(v) if v.q==1 else Fraction(int(v.p),int(v.q)) for v in col) for col in D.nullspace())


class _Basis:
    def __init__(self,kernel,digits,points,degree,vector,scaling):
        self.a=Arithmetic(kernel,digits);self.points=points;self.vector=vector
        self.d=points.shape[1];self.components=self.d if vector else 1
        # Vector potential CPD degree differs from the scalar potential degree.
        # Geometry and monomial evaluation do not depend on the kernel.
        from .kernels import IMQ
        self.poly=NodalBasis(Arithmetic(IMQ(),digits),points,degree,kernel_scaling=scaling)
        if self.a.ctx:
            self.poly.arithmetic.ctx=self.a.ctx;self.poly.arithmetic.backend.ctx=self.a.ctx
        self.h=self.poly.kernel_scale if scaling=='local' else self.a.number(1)
        self.columns=solenoidal_columns(self.d,degree) if vector else None
        self.m=len(self.columns) if vector else len(self.poly.powers)
        self.n=len(points)*self.components

    def polynomial(self,points,ops):
        P=self.poly.polynomials(points,ops)
        if not self.vector:return P
        out=self.a.zeros(len(points)*self.d,self.m);s=len(self.poly.powers)
        for i in range(len(points)):
            for c in range(self.d):
                for k,col in enumerate(self.columns):
                    out[i*self.d+c,k]=sum(P[i,j]*self.a.number(col[c*s+j]) for j in range(s))
        return out

    def kernel_matrix(self,x,ops,y,torch=False):
        a=self.a;d=self.d;identities=[Identity(d)]*len(y)
        def evaluate(left):
            if torch:
                from .scalar_backends import _torch_matrix
                return _torch_matrix(a.kernel,x,left,y,identities,self.h)
            return a.matrix(x,left,y,identities,self.h)
        if not self.vector:return evaluate(ops)
        if torch:
            from .torch_backend import _torch
            out=_torch().zeros((len(x)*d,len(y)*d),dtype=_torch().float64)
        else:out=a.zeros(len(x)*d,len(y)*d)
        for c in range(d):
            for b in range(d):
                transform=Derivative(c,d)@Derivative(b,d)
                if c==b:transform=transform-Laplacian(d)
                block=evaluate([op@transform for op in ops])
                for i in range(len(x)):
                    for j in range(len(y)):out[i*d+c,j*d+b]=block[i,j]
        return out

    def arrays(self,target,ops,torch=False):
        a=self.a;n=self.n;m=self.m;ids=[Identity(self.d)]*len(self.points)
        P=self.polynomial(self.points,ids)
        check=np.asarray(P.tolist() if a.ctx else P,float)
        if m and np.linalg.matrix_rank(check)<m:raise np.linalg.LinAlgError('Polynomial space is not unisolvent; enlarge/change stencil')
        targets=np.repeat(target[None,:],len(ops),axis=0)
        T=self.polynomial(targets,ops)
        if torch:
            from .torch_backend import _torch
            t=_torch();G=t.zeros((n+m,n+m),dtype=t.float64)
            G[:n,:n]=self.kernel_matrix(self.points,ids,self.points,True)
            G[:n,n:]=t.as_tensor(P);G[n:,:n]=t.as_tensor(P.T)
            Q=t.cat((self.kernel_matrix(targets,ops,self.points,True),t.as_tensor(T)),dim=1).T
        else:
            G=a.zeros(n+m,n+m);G[:n,:n]=self.kernel_matrix(self.points,ids,self.points)
            G[:n,n:]=P;G[n:,:n]=P.T
            Q=a.zeros(n+m,len(ops)*self.components)
            Q[:n,:]=self.kernel_matrix(targets,ops,self.points).T;Q[n:,:]=T.T
        return G,Q


class OperatorSet(Mapping):
    """Named discrete operators with item and attribute access."""
    def __init__(self,operators,diagnostics):self._operators=operators;self.diagnostics=diagnostics
    def __getitem__(self,name):return self._operators[name]
    def __iter__(self):return iter(self._operators)
    def __len__(self):return len(self._operators)
    def __getattr__(self,name):
        try:return self._operators[name]
        except KeyError:raise AttributeError(name) from None


class DiscreteOperator:
    """Sparse weights; vector spaces use node-major component ordering."""
    def __init__(self,matrix,owner,column):
        self.matrix=matrix;self._owner=owner;self._column=column
        self.shape=matrix.shape;self.components=owner.components
    def __matmul__(self,values):
        mp=isinstance(self.matrix,MPSparseMatrix)
        data=np.asarray(values.tolist() if hasattr(values,'rows') else values,dtype=object if mp else float)
        c=self.components;n=len(self._owner.source)
        shaped=c>1 and data.shape==(n,c)
        if shaped:data=data.reshape(-1)
        if data.ndim not in (1,2) or data.shape[0]!=n*c:raise ValueError('Input must match source DOFs (node-major for vectors)')
        if mp:
            result=self.matrix@data
            if shaped:return np.array(result.tolist(),object).reshape(-1,c)
        else:
            result=self.matrix@data
            if shaped:return result.reshape(-1,c)
        return result
    def local(self,i):
        o=self._owner
        if type(i) is not int or not 0<=i<len(o.targets):raise IndexError('Target index out of range')
        w=o.weights[i];c=o.components;k=self._column*c;n=len(o.ids[i])*c
        return SimpleNamespace(indices=o.ids[i].copy(),target=o.targets[i].copy(),
            weights=w[:n,k:k+c].copy(),multipliers=w[n:,k:k+c].copy(),
            diagnostics=copy.deepcopy(o.diagnostics[i]),ordering='node-major',global_dtype=o.precision.global_dtype)
    def reconstruct_local(self,i):
        self.local(i) # validate index
        o=self._owner;G,Q=o.reconstruct(i);a=o.arithmetic;c=o.components;k=self._column*c
        if a.ctx:
            s=[1/a.ctx.sqrt(max(abs(G[j,l]) for l in range(G.cols))) for j in range(G.rows)]
            scaled=a.ctx.matrix([[s[j]*G[j,l]*s[l] for l in range(G.cols)] for j in range(G.rows)])
            q=Q[:,k:k+c];scaled_rhs=a.ctx.matrix([[s[j]*q[j,l] for l in range(c)] for j in range(G.rows)])
        else:
            s=1/np.sqrt(np.max(abs(G),axis=1));scaled=s[:,None]*G*s[None,:]
            q=Q[:,k:k+c];scaled_rhs=s[:,None]*q
        return SimpleNamespace(matrix=G,rhs=q,scaled_matrix=scaled,scaled_rhs=scaled_rhs,scale=s,
            equation='matrix.T @ augmented_weights = rhs',backend=type(o.backend).__name__,local_digits=o.precision.local_digits)


def build_operators(method,*,source,targets=None,operators,space=None):
    """Build all requested componentwise differential operators in one local solve."""
    from .operator_backends import execute,reconstruct
    from .lhi_backends import PythonBackend
    from .precision import Precision
    from .stencils import StencilPolicy
    if not isinstance(method.precision,Precision):raise TypeError('precision must be Precision')
    if not isinstance(method.stencil_policy,StencilPolicy):raise TypeError('stencil_policy must be StencilPolicy')
    if method.scheme!='standard':raise NotImplementedError('operators currently uses standard nodal RBF-FD; Hermite schemes need explicit source functionals')
    if not isinstance(operators,Mapping) or not operators:raise ValueError('operators must be a nonempty name-to-operator mapping')
    if any(not isinstance(k,str) or not k for k in operators):raise ValueError('Operator names must be nonempty strings')
    descriptor=resolve_space(method,space);degree=descriptor.degree()
    if isinstance(descriptor,PressureSpace) and degree is None:degree=0
    vector=isinstance(descriptor,DivergenceFreeSpace)
    source=np.array(getattr(source,'points',source),dtype=float,copy=True)
    targets=np.array(source if targets is None else getattr(targets,'points',targets),dtype=float,copy=True)
    if source.ndim!=2 or source.shape[1] not in (2,3) or not len(source) or not np.isfinite(source).all():raise ValueError('Expected finite nonempty Nx2 or Nx3 sources')
    if targets.ndim!=2 or targets.shape[1]!=source.shape[1] or not len(targets) or not np.isfinite(targets).all():raise ValueError('Expected matching nonempty targets')
    if len(np.unique(source,axis=0))!=len(source):raise ValueError('Sources must be distinct')
    if type(method.stencil_size) is not int or not 2<=method.stencil_size<=len(source):raise ValueError('Invalid stencil_size')
    if method.precision.global_digits is not None:raise ValueError('Local operators use local_digits')
    tree=cKDTree(source);ids=[];jobs=[];geometry=[]
    for target in targets:
        selected=method.stencil_policy.select(tree,target,method.stencil_size,degree)
        ops=[bind_operators([op],target[None,:])[0] for op in operators.values()]
        if any(op.dimension!=source.shape[1] for op in ops):raise ValueError('Operator dimension mismatch')
        ids.append(selected.copy());jobs.append(dict(points=source[selected].copy(),target=target.copy(),ops=ops))
        geometry.append(geometry_quality(source[selected],target,degree if degree is not None else 1))
    owner=SimpleNamespace(source=source,targets=targets,ids=ids,jobs=jobs,kernel=copy.deepcopy(descriptor.kernel),degree=degree,vector=vector,
        components=source.shape[1] if vector else 1,precision=copy.deepcopy(method.precision),scaling=method.stencil_policy.scaling,
        backend=copy.deepcopy(method.local_backend or PythonBackend()),arithmetic=Arithmetic(descriptor.kernel,method.precision.local_digits))
    results=execute(owner);owner.weights=[r['weights'] for r in results]
    owner.diagnostics=[dict(geometry=g,scaled_condition=r['condition'],weight_residual=r['residual'],factorizations=1) for g,r in zip(geometry,results)]
    owner.reconstruct=lambda i:reconstruct(owner,i)
    c=owner.components;out={};a=owner.arithmetic;full=method.precision.global_dtype=='mpmath'
    if full and not a.ctx:raise ValueError('mpmath sparse weights require local_digits')
    for k,name in enumerate(operators):
        rows=[]
        for selected,w in zip(ids,owner.weights):
            for component in range(c):
                rows.append({int(node)*c+b:w[j*c+b,k*c+component] for j,node in enumerate(selected) for b in range(c)})
        if full:matrix=MPSparseMatrix(a.ctx,rows,ncols=len(source)*c)
        else:
            ii=[];jj=[];vv=[]
            for i,row in enumerate(rows):
                for j,v in row.items():ii.append(i);jj.append(j);vv.append(float(v))
            matrix=csr_matrix((vv,(ii,jj)),shape=(len(targets)*c,len(source)*c))
            if not np.isfinite(matrix.data).all():raise FloatingPointError('Sparse weights overflow Float64')
        out[name]=DiscreteOperator(matrix,owner,k)
    return OperatorSet(out,dict(backend=type(owner.backend).__name__,factorizations=len(targets),rhs_per_stencil=len(operators)*c,
        components=c,ordering='node-major',local_digits=method.precision.local_digits,global_dtype=method.precision.global_dtype))
