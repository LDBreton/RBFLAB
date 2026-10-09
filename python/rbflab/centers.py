"""Independent scalar LHI functional centers and local membership."""
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from scipy.spatial import cKDTree
from .operators import Identity, bind_operators
from .stencils import StencilPolicy
from .problems import values

@dataclass(frozen=True)
class CenterGroup:
    """A named set of LHI data functionals.

    points: Nx2/Nx3 coordinates. Solution points must map to cloud interior DOFs.
    role: solution, pde, boundary, or data. Only solution values are unknown.
    operator/data: PDE defaults to the problem operator/forcing. Boundary/data
        groups must supply both; solution uses Identity and has no data.
    size: nearest-neighbor count after target filtering; None selects all.
    indices: explicit per-target rows of group-local indices, preserving order.
        Mutually exclusive with size/policy. One row per interior target.
    target: allow, exclude, or require geometrical coincidence with target.
    policy: independent neighbor-selection policy (not a separate kernel scale).
    normals: optional normals to bind normal-dependent boundary functionals.

    Independent groups currently support stationary scalar Python LHI only.
    """
    points: object
    role: str = "pde"
    operator: object = None
    data: object = None
    size: int | None = None
    indices: object = None
    target: str = "allow"
    policy: object = None
    normals: object = None

    def __post_init__(self):
        points=np.array(getattr(self.points, 'points', self.points),dtype=float,copy=True)
        if points.ndim!=2 or points.shape[1] not in (2,3) or not np.isfinite(points).all():
            raise ValueError("CenterGroup points must be finite Nx2/Nx3")
        if len(np.unique(points,axis=0))!=len(points):
            raise ValueError("Duplicate centers within a functional group")
        if self.role not in ('solution','pde','boundary','data'): raise ValueError('Unknown center role')
        if self.target not in ('allow','exclude','require'): raise ValueError('Unknown target rule')
        if self.size is not None and (type(self.size) is not int or self.size<0): raise ValueError('size must be nonnegative')
        if self.indices is not None and (self.size is not None or self.policy is not None):
            raise ValueError('Explicit indices cannot be combined with size or policy')
        if self.role=='pde' and self.operator is not None and self.data is None:
            raise ValueError('A custom PDE operator requires its corresponding data')
        if self.role=='solution' and (self.operator is not None or self.data is not None):
            raise ValueError('Solution centers carry unknown Identity values')
        if self.role in ('boundary','data') and (self.operator is None or self.data is None):
            raise ValueError('Prescribed groups require operator and data')
        if self.policy is not None and (not isinstance(self.policy,StencilPolicy) or self.policy.scaling!='physical' or self.policy.shape_rule!='fixed'):
            raise ValueError('Group policy selects neighbors; set kernel scaling/shape on the method')
        if self.normals is not None:
            normals=np.array(self.normals,dtype=float,copy=True)
            if normals.shape!=points.shape or not np.isfinite(normals).all():raise ValueError('One finite normal vector is required per center')
            normals.flags.writeable=False
            object.__setattr__(self,'normals',normals)
        points.flags.writeable=False
        object.__setattr__(self,'points',points)

    def select(self,target,row,degree):
        match=np.all(self.points==target,axis=1)
        if self.indices is not None:
            raw=np.asarray(self.indices[row])
            if raw.ndim!=1 or (raw.size and not np.issubdtype(raw.dtype,np.integer)):
                raise ValueError('Explicit membership must be a vector of integer indices')
            ids=raw.astype(int)
            if len(set(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(self.points)):
                raise ValueError('Invalid or repeated group indices')
            if self.target=='exclude' and np.any(match[ids]):raise ValueError('Explicit membership violates target exclusion')
        else:
            candidates=np.flatnonzero(~match if self.target=='exclude' else np.ones(len(match),bool))
            size=len(candidates) if self.size is None else self.size
            if size>len(candidates):raise ValueError('Group size exceeds eligible centers after target filtering')
            ids=candidates if self.size is None else (candidates[(self.policy or StencilPolicy()).select(cKDTree(self.points[candidates]),target,size,degree)] if size else np.array([],int))
        if self.target=='require' and not np.any(match[ids]):raise ValueError('Group must include the target')
        return ids


def select_groups(method,problem,cloud,bd,row,center,neighbors,default_pde,ctx=None):
    """Normalize defaults and explicit groups to the same functional assembly."""
    ii=set(map(int,cloud.interior_indices)); point=cloud.points[center]
    if method.centers is None:
        sc=np.array([j for j in neighbors if j in ii],int)
        bc=np.array([j for j in neighbors if j in bd],int)
        pc=np.asarray(default_pde,int)
        ids=np.r_[sc,bc,pc];points=cloud.points[ids]
        ops=[Identity(cloud.dimension)]*len(sc)+[bd[int(j)][0] for j in bc]+[problem.operator]*len(pc)
        from .precision import mp_values
        f=mp_values(problem.rhs,cloud.points[pc],ctx) if ctx else values(problem.rhs,cloud.points[pc])
        known=list(bd[int(j)][1] for j in bc)+list(f)
        groups={'solution':sc.copy(),'boundary':bc.copy(),'pde':pc.copy()}
        return sc,bc,pc,points,ops,known,groups
    if not isinstance(method.centers,Mapping) or not method.centers:raise ValueError('centers must be a nonempty name-to-CenterGroup mapping')
    lookup={tuple(cloud.points[j]):int(j) for j in ii}
    unknown=[];known_parts=[];meta={};boundary=[];pde=[]
    for name,g in method.centers.items():
        if not isinstance(name,str) or not name or not isinstance(g,CenterGroup):raise TypeError('Expected named CenterGroup objects')
        if g.points.shape[1]!=cloud.dimension:raise ValueError('Group/cloud dimensions differ')
        ids=g.select(point,row,method.polynomial_degree);points=g.points[ids];meta[name]=ids.copy()
        if g.role=='solution':
            try:nodes=[lookup[tuple(x)] for x in points]
            except KeyError:raise ValueError('Solution group points must map to cloud interior unknowns') from None
            unknown.extend((node,x,Identity(cloud.dimension)) for node,x in zip(nodes,points));continue
        op=g.operator if g.operator is not None else problem.operator
        data=g.data if g.data is not None else problem.rhs
        normals=None if g.normals is None else np.asarray(g.normals)[ids]
        if getattr(data,'requires_normals',False):
            if normals is None:raise ValueError('Boundary data require normals')
            data=data.bind_normals(normals)
        ops=[]
        for k,x in enumerate(points):
            normal=None if normals is None else normals[k]
            if hasattr(op,'at_point'):bound=op.at_point(x,normal)
            elif hasattr(op,'at'):
                if normal is None:raise ValueError('Boundary operator requires normals')
                bound=op.at(normal)
            else:bound=bind_operators([op],x[None,:])[0]
            ops.append(bound)
        if not callable(data) and np.ndim(data)==1 and len(data)==len(g.points):data=np.asarray(data)[ids]
        from .precision import mp_values
        vals=mp_values(data,points,ctx) if ctx else values(data,points)
        known_parts.extend(zip(points,ops,list(vals)))
        if g.role=='boundary':boundary.extend(ids)
        if g.role=='pde':pde.extend(ids)
    if not unknown:raise ValueError('Each LHI stencil requires solution centers')
    sc=np.array([x[0] for x in unknown],int)
    points=np.array([x[1] for x in unknown]+[x[0] for x in known_parts])
    ops=[x[2] for x in unknown]+[x[1] for x in known_parts]
    ops=bind_operators(ops,points)
    signatures=[(tuple(x),repr(op.terms)) for x,op in zip(points,ops)]
    if len(set(signatures))!=len(signatures):raise ValueError('Duplicate local functionals across center groups')
    return sc,np.array(boundary,int),np.array(pde,int),points,ops,[x[2] for x in known_parts],meta
