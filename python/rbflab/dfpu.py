"""Planar divergence-free partition of unity via aligned streamfunctions.

Drake, Fuselier and Wright (2021), doi:10.1137/20M1373505, equations 19--24.
"""
import numpy as np
import weakref

from .discrete_operators import _Basis
from .global_vector_interpolation_backend import _samples
from .operators import Derivative, Identity, Operator
from .patches import PatchCover, _normalized_weights


def _potential(patch, point):
    """Canonical local potential with zero polynomial constant at the origin."""
    b, coeff = patch['basis'], patch['coefficients']
    a = b.a
    q = np.asarray([point], dtype=object if a.ctx else float)
    dx = a.matrix(q, [Derivative(0, 2)], b.points, [Identity(2)]*len(b.points), b.h)
    dy = a.matrix(q, [Derivative(1, 2)], b.points, [Identity(2)]*len(b.points), b.h)
    out = sum(coeff[2*j]*dy[0,j]-coeff[2*j+1]*dx[0,j] for j in range(len(b.points)))
    if b.m:
        z = [(a.number(point[k])-b.poly.origin[k])/b.poly.scale for k in range(2)]
        powers = b.poly.powers
        s = len(powers)
        for k, column in enumerate(b.columns):
            amplitude = coeff[b.n+k]*b.poly.scale
            for j, (px, py) in enumerate(powers):
                cy = column[s+j]
                if cy:
                    out += amplitude*a.number(cy)*z[0]**(px+1)*z[1]**py/(px+1)
                if px == 0 and column[j]:
                    out -= amplitude*a.number(column[j])*z[1]**(py+1)/(py+1)
    return out


def _local_values(patch, point, order):
    b, coeff = patch['basis'], patch['coefficients']
    q = np.asarray([point], dtype=object if b.a.ctx else float)
    op = Identity(2) if order == 0 else Derivative(order-1, 2)
    K = b.kernel_matrix(q, [op], b.points)
    P = b.polynomial(q, [op])
    return [sum(K[c,j]*coeff[j] for j in range(b.n)) +
            sum(P[c,j]*coeff[b.n+j] for j in range(b.m)) for c in range(2)]


def _glue_geometry(cover, arithmetic, gamma=4):
    c, r = cover.centers, cover.radii
    edges = []
    for i in range(len(r)):
        for j in range(i+1, len(r)):
            distance = np.linalg.norm(c[i]-c[j])
            if distance < r[i]+r[j]:
                g = (r[j]*c[i]+r[i]*c[j])/(r[i]+r[j])
                near = min(np.linalg.norm(g-c[i]), np.linalg.norm(g-c[j]))
                edges.append((i,j,g,near))
    if not edges:return edges, []
    minimum=min(edge[3] for edge in edges)
    if minimum == 0:
        # Coincident centers make the paper's distance ratio undefined.
        # Uniform positive weights retain the same graph and a defined gauge.
        return edges,[arithmetic.number(1)]*len(edges)
    a=arithmetic
    weights=[(a.ctx.exp(-a.number(gamma)*(1-a.number(e[3])/a.number(minimum))**2)
              if a.ctx else np.exp(-gamma*(1-e[3]/minimum)**2)) for e in edges]
    if any(not (w>0 and (a.ctx.isfinite(w) if a.ctx else np.isfinite(w))) for w in weights):
        raise np.linalg.LinAlgError('Weighted glue graph lost numerical rank; use extended precision or a more balanced cover')
    return edges,weights


def _glue_factor(arithmetic,n,edges,weights):
    a=arithmetic
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,j,_,_ in edges:
        parent[find(j)]=find(i)
    anchors={min(k for k in range(n) if find(k)==find(i)) for i in range(n)}
    free=[i for i in range(n) if i not in anchors]
    positions={index:k for k,index in enumerate(free)}
    if not free:return {'factor':None,'free':free,'positions':positions,'anchors':anchors}
    L=a.zeros(len(free),len(free))
    for (i,j,_,_),w in zip(edges,weights):
        v=w*w
        if i in positions:L[positions[i],positions[i]]+=v
        if j in positions:L[positions[j],positions[j]]+=v
        if i in positions and j in positions:
            L[positions[i],positions[j]]-=v
            L[positions[j],positions[i]]-=v
    try:
        if not a.ctx:
            condition=np.linalg.cond(L)
            if not np.isfinite(condition) or condition>1e14:
                raise np.linalg.LinAlgError('Ill-conditioned weighted glue graph')
        factor=a.factor(L)
    except (np.linalg.LinAlgError,ValueError,ZeroDivisionError) as exc:
        raise np.linalg.LinAlgError('Weighted glue graph lost numerical rank; use extended precision or a more balanced cover') from exc
    return {'factor':factor,'free':free,'positions':positions,'anchors':anchors}


def _shifts(patches, edges, weights, glue=None):
    a=patches[0]['basis'].a
    n=len(patches)
    glue=glue or _glue_factor(a,n,edges,weights)
    positions=glue['positions']
    rhs=a.vector([0]*len(glue['free']))
    for (i,j,g,_),w in zip(edges,weights):
        delta=_potential(patches[j],g)-_potential(patches[i],g)
        if i in positions:rhs[positions[i]]+=w*w*delta
        if j in positions:rhs[positions[j]]-=w*w*delta
    try:solution=glue['factor'].solve(rhs) if glue['free'] else []
    except ZeroDivisionError as exc:
        raise np.linalg.LinAlgError('Weighted glue graph could not be solved; use more digits or a more balanced cover') from exc
    shifts=[a.number(0)]*n
    for index,k in positions.items():shifts[index]=solution[k]
    residual=max((abs(shifts[i]-shifts[j]-(_potential(patches[j],g)-_potential(patches[i],g)))
                  for i,j,g,_ in edges),default=a.number(0))
    if not (all(a.ctx.isfinite(v) for v in shifts+[residual]) if a.ctx
            else np.isfinite(shifts+[residual]).all()):
        raise FloatingPointError('Nonfinite potential alignment')
    return shifts,residual


class DivergenceFreePatchInterpolant:
    """Curl of the blended, aligned local streamfunctions on a 2D cover."""
    def __init__(self, cover, points, memberships, patches, edges, weights, shifts, residual, precision, glue_factor):
        self.cover,self.points,self.memberships=cover,points,memberships
        self.patches,self.edges,self.weights,self.shifts=patches,edges,weights,shifts
        self.arithmetic=patches[0]['basis'].a
        self.precision=precision
        self.glue_factor=glue_factor
        self._closed=False
        self.diagnostics={'backend':'python_dfpu','arithmetic':'mpmath' if self.arithmetic.ctx else 'float64',
                          'patches':len(patches),'occupancy':[len(rows) for rows in memberships],
                          'factorizations':len(patches),'glue_edges':len(edges),
                          'glue_residual':float(residual),'nodal_exact':False,
                          'glue_components':len(glue_factor['anchors']),
                          'gauge':'one_zero_per_component','ordering':'node-major'}

    def _evaluate(self, points, operator):
        from .methods import _query
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        q=_query(points,2)
        if operator is None:operator=Identity(2)
        if not isinstance(operator,Operator) or operator.dimension!=2:
            raise TypeError('DFPU requires a matching constant-coefficient Operator')
        if any(sum(alpha)>1 for alpha,_ in operator.terms):
            raise NotImplementedError('DFPU supports velocity and first derivatives only')
        a=self.arithmetic
        out=[]
        for point in q:
            weights=_normalized_weights(point,self.cover,a.ctx)
            px=py=pxx=pxy=pyy=a.number(0)
            for patch,shift,(w,dw,ddw) in zip(self.patches,self.shifts,weights):
                if w==0:continue
                potential=_potential(patch,point)+shift
                velocity=_local_values(patch,point,0)
                gx,gy=velocity[1],-velocity[0]
                px+=dw[0]*potential+w*gx
                py+=dw[1]*potential+w*gy
                if any(sum(alpha)==1 for alpha,_ in operator.terms):
                    vx=_local_values(patch,point,1)
                    vy=_local_values(patch,point,2)
                    pxx+=ddw[0][0]*potential+2*dw[0]*gx+w*vx[1]
                    pxy+=ddw[0][1]*potential+dw[0]*gy+dw[1]*gx+w*vy[1]
                    pyy+=ddw[1][1]*potential+2*dw[1]*gy-w*vy[0]
            components=[]
            for c in range(2):
                value=a.number(0)
                for alpha,coefficient in operator.terms:
                    if alpha==(0,0):term=(-py,px)[c]
                    elif alpha==(1,0):term=(-pxy,pxx)[c]
                    elif alpha==(0,1):term=(-pyy,pxy)[c]
                    else:raise NotImplementedError('DFPU supports velocity and first derivatives only')
                    value+=a.number(coefficient)*term
                components.append(value)
            out.append(components)
        return out

    def evaluate(self,points,operator=None):
        result=np.asarray([[float(v) for v in row] for row in self._evaluate(points,operator)],float)
        return result.reshape(-1,2)

    def evaluate_mp(self,points,operator=None):
        ctx=self.arithmetic.ctx
        if ctx is None:raise ValueError('evaluate_mp requires extended precision')
        rows=self._evaluate(points,operator)
        return ctx.matrix(rows) if rows else ctx.matrix(0,2)

    def _potential_values(self,points):
        from .methods import _query
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        q=_query(points,2)
        a=self.arithmetic
        return [sum(weight[0]*(_potential(patch,point)+shift)
                    for patch,shift,weight in zip(self.patches,self.shifts,
                                                   _normalized_weights(point,self.cover,a.ctx))
                    if weight[0]>0)
                for point in q]

    def potential(self,points):
        return np.asarray([float(v) for v in self._potential_values(points)],float)

    def potential_mp(self,points):
        ctx=self.arithmetic.ctx
        if ctx is None:raise ValueError('potential_mp requires extended precision')
        values=self._potential_values(points)
        return ctx.matrix(values) if values else ctx.matrix(0,1)

    def divergence(self,points,extended=False):
        dx=Derivative(0,2)
        dy=Derivative(1,2)
        if extended:
            a=self.evaluate_mp(points,dx)
            b=self.evaluate_mp(points,dy)
            return self.arithmetic.ctx.matrix([a[i,0]+b[i,1] for i in range(a.rows)])
        a=self.evaluate(points,dx)
        b=self.evaluate(points,dy)
        return a[:,0]+b[:,1]

    def refit(self,data):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        values=_samples(data,self.points,self.arithmetic)
        replacement=[]
        for patch,rows in zip(self.patches,self.memberships):
            rhs=[values[2*i+c] for i in rows for c in range(2)]+[0]*patch['basis'].m
            coefficient=patch['factor'].solve(patch['basis'].a.vector(rhs))
            finite=all(self.arithmetic.ctx.isfinite(v) for v in coefficient) if self.arithmetic.ctx else np.isfinite(coefficient).all()
            if not finite:raise FloatingPointError('Nonfinite patch coefficients')
            replacement.append({'basis':patch['basis'],'factor':patch['factor'],'coefficients':coefficient})
        shifts,residual=_shifts(replacement,self.edges,self.weights,self.glue_factor)
        self.patches=replacement
        self.shifts=shifts
        self.diagnostics['glue_residual']=float(residual)
        self.diagnostics['refits']=self.diagnostics.get('refits',0)+1
        return self

    def close(self):self._closed=True
    def __enter__(self):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        return self
    def __exit__(self,*_):self.close()


class NativeDivergenceFreePatchInterpolant:
    """Retained native local factors and native potential-blend evaluation."""
    def __init__(self,worker,cover,points,memberships,bases,edges,weights,arithmetic,backend,digits,glue_factor):
        self.worker,self.cover,self.points,self.memberships=worker,cover,points,memberships
        self.bases,self.edges,self.weights=bases,edges,weights
        self.arithmetic,self.backend,self.digits=arithmetic,backend,digits
        self.glue_factor=glue_factor
        self._closed=False
        self._finalizer=weakref.finalize(self,worker.close)
        self.diagnostics={'backend':'cpp_dfpu','arithmetic':'mpfr' if arithmetic.ctx else 'float64',
                          'data_backend':'python','glue_backend':'python',
                          'evaluation_backend':'cpp','patches':len(bases),
                          'factorizations':len(bases),'glue_edges':len(edges),
                          'occupancy':[len(rows) for rows in memberships],
                          'ordering':'node-major','nodal_exact':False,
                          'glue_components':len(glue_factor['anchors']),
                          'gauge':'one_zero_per_component',
                          'linear_solver':backend.linear_solver}

    def _encode(self,value):
        from .precision import mp_number
        ctx=self.arithmetic.ctx
        if ctx:
            v=mp_number(ctx,value)
            if not ctx.isfinite(v):raise ValueError('Expected finite native input')
            return ctx.nstr(v,self.digits+8)
        v=float(value)
        if not np.isfinite(v):raise ValueError('Expected finite native input')
        return format(v,'.17g')

    def _rows(self,points):
        from .methods import _query
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        q=_query(points,2)
        if not len(q):return []
        try:reply=self.worker.request(' '.join(['DFP_EVAL',str(len(q)),*(self._encode(v) for v in q.flat)]))
        except RuntimeError as exc:
            if 'outside positive patch cover' in str(exc):
                raise ValueError('Query point is outside the positive patch cover') from exc
            raise
        if len(reply)!=1+7*len(q) or int(reply[0])!=len(q):
            raise RuntimeError('Invalid native DFPU evaluation result')
        a=self.arithmetic
        return [[a.number(reply[1+7*i+j]) for j in range(7)] for i in range(len(q))]

    def _evaluate(self,points,operator):
        if operator is None:operator=Identity(2)
        if not isinstance(operator,Operator) or operator.dimension!=2:
            raise TypeError('DFPU requires a matching constant-coefficient Operator')
        if any(sum(alpha)>1 for alpha,_ in operator.terms):
            raise NotImplementedError('DFPU supports velocity and first derivatives only')
        out=[]
        for row in self._rows(points):
            pair=[]
            for component in range(2):
                value=self.arithmetic.number(0)
                for alpha,coefficient in operator.terms:
                    if alpha==(0,0):index=1+component
                    elif alpha==(1,0):index=3+component
                    elif alpha==(0,1):index=5+component
                    else:raise NotImplementedError('DFPU supports velocity and first derivatives only')
                    value+=self.arithmetic.number(coefficient)*row[index]
                pair.append(value)
            out.append(pair)
        return out

    def evaluate(self,points,operator=None):
        return np.asarray([[float(v) for v in row] for row in self._evaluate(points,operator)],float).reshape(-1,2)

    def evaluate_mp(self,points,operator=None):
        ctx=self.arithmetic.ctx
        if ctx is None:raise ValueError('evaluate_mp requires extended precision')
        rows=self._evaluate(points,operator)
        return ctx.matrix(rows) if rows else ctx.matrix(0,2)

    def potential(self,points):return np.asarray([float(row[0]) for row in self._rows(points)],float)
    def potential_mp(self,points):
        ctx=self.arithmetic.ctx
        if ctx is None:raise ValueError('potential_mp requires extended precision')
        rows=self._rows(points)
        return ctx.matrix([row[0] for row in rows]) if rows else ctx.matrix(0,1)

    def divergence(self,points,extended=False):
        rows=self._rows(points)
        values=[row[3]+row[6] for row in rows]
        if extended:
            if self.arithmetic.ctx is None:raise ValueError('Extended divergence requires extended precision')
            return self.arithmetic.ctx.matrix(values) if values else self.arithmetic.ctx.matrix(0,1)
        return np.asarray([float(v) for v in values],float)

    def refit(self,data):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        values=_samples(data,self.points,self.arithmetic)
        rhs=[]
        for basis,rows in zip(self.bases,self.memberships):
            rhs.extend(values[2*i+c] for i in rows for c in range(2))
            rhs.extend([0]*basis.m)
        reply=self.worker.request(' '.join(['DFP_PREPARE',*(self._encode(v) for v in rhs)]))
        if not reply or int(reply[0])!=len(self.bases):raise RuntimeError('Invalid native DFPU coefficients')
        cursor=1;candidate=[]
        for basis in self.bases:
            size=int(reply[cursor]);cursor+=1
            if size!=basis.n+basis.m:raise RuntimeError('Invalid native DFPU coefficient count')
            coeff=self.arithmetic.vector([self.arithmetic.number(v) for v in reply[cursor:cursor+size]])
            candidate.append({'basis':basis,'coefficients':coeff})
            cursor+=size
        if cursor!=len(reply):raise RuntimeError('Invalid native DFPU coefficient tail')
        shifts,residual=_shifts(candidate,self.edges,self.weights,self.glue_factor)
        self.worker.request(' '.join(['DFP_COMMIT',*(self._encode(v) for v in shifts)]))
        self.shifts=shifts
        self.diagnostics['glue_residual']=float(residual)
        self.diagnostics['refits']=self.diagnostics.get('refits',0)+1
        return self

    def close(self):
        if not self._closed:self._finalizer();self._closed=True
    def __enter__(self):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        self.worker.check_open();return self
    def __exit__(self,*_):self.close()


def interpolate_dfpu(space, centers, data, precision, cover, backend=None):
    from .methods import _query
    if not isinstance(cover,PatchCover):raise TypeError('patches must be PatchCover')
    if cover.centers.shape[1]!=2:raise NotImplementedError('DFPU supports planar 2D only')
    points=np.array(_query(centers,2),copy=True)
    if not len(points) or len(np.unique(points,axis=0))!=len(points):
        raise ValueError('Expected nonempty distinct centers')
    from .nodal import dense_arithmetic
    arithmetic=dense_arithmetic(space.kernel,precision)
    memberships=cover.memberships(points,arithmetic.ctx)
    for rows in memberships:rows.setflags(write=False)
    if any(not len(rows) for rows in memberships):raise ValueError('Every patch must contain input nodes')
    covered=np.zeros(len(points),bool)
    for rows in memberships:covered[rows]=True
    if not covered.all():raise ValueError('Input point is outside the positive patch cover')
    edges,weights=_glue_geometry(cover,arithmetic)
    samples=_samples(data,points,arithmetic)
    if backend is not None:
        return _interpolate_native_dfpu(space,points,np.asarray(samples,dtype=object).reshape(-1,2),precision,cover,backend,
                                        memberships,edges,weights,arithmetic)
    patches=[]
    for rows in memberships:
        local=points[rows]
        basis=_Basis(space.kernel,precision.global_digits,local,space.degree(),True,'physical')
        ids=[Identity(2)]*len(local)
        P=basis.polynomial(local,ids)
        if basis.m and np.linalg.matrix_rank(np.asarray(P.tolist() if arithmetic.ctx else P,float))<basis.m:
            raise np.linalg.LinAlgError('Polynomial space is not unisolvent; enlarge patch')
        G=basis.a.zeros(basis.n+basis.m,basis.n+basis.m)
        G[:basis.n,:basis.n]=basis.kernel_matrix(local,ids,local)
        if basis.m:G[:basis.n,basis.n:]=P;G[basis.n:,:basis.n]=P.T
        factor=basis.a.factor(G)
        rhs=[samples[2*i+c] for i in rows for c in range(2)]+[0]*basis.m
        coeff=factor.solve(basis.a.vector(rhs))
        if not (all(basis.a.ctx.isfinite(v) for v in coeff) if basis.a.ctx else np.isfinite(coeff).all()):
            raise FloatingPointError('Nonfinite patch coefficients')
        patches.append({'basis':basis,'factor':factor,'coefficients':coeff})
    glue=_glue_factor(patches[0]['basis'].a,len(patches),edges,weights)
    shifts,residual=_shifts(patches,edges,weights,glue)
    points.setflags(write=False)
    return DivergenceFreePatchInterpolant(cover,points,memberships,patches,edges,weights,shifts,residual,precision,glue)


def _interpolate_native_dfpu(space,points,data,precision,cover,backend,memberships,edges,weights,arithmetic):
    from .global_backend import CppGlobalBackend, _compiled, _acquire_worker
    from .global_flow_backend import _field_tokens
    if not isinstance(backend,CppGlobalBackend):
        raise TypeError('DFPU backend must be CppGlobalBackend or None')
    if backend.derivative_order<1:
        raise ValueError('DFPU native evaluation requires derivative_order >= 1')
    bases=[]
    for rows in memberships:
        local=points[rows]
        basis=_Basis(space.kernel,precision.global_digits,local,space.degree(),True,'physical')
        P=basis.polynomial(local,[Identity(2)]*len(local))
        if basis.m and np.linalg.matrix_rank(np.asarray(P.tolist() if arithmetic.ctx else P,float))<basis.m:
            raise np.linalg.LinAlgError('Polynomial space is not unisolvent; enlarge patch')
        bases.append(basis)
    executable,bound=_compiled(space.kernel,2,max(3,backend.derivative_order+2),
                               'mpfr' if arithmetic.ctx else 'float64',backend,vector_only=True)
    glue=_glue_factor(arithmetic,len(bases),edges,weights)
    worker=_acquire_worker(executable,backend)
    digits=precision.global_digits or 17
    fit=NativeDivergenceFreePatchInterpolant(worker,cover,points,memberships,bases,
                                             edges,weights,arithmetic,backend,digits,glue)
    encode=fit._encode
    try:
        worker.request(' '.join(['DFP_INIT',str(len(bases)),'1',str(digits),
                                 *(encode(v) for v in cover.centers.flat),
                                 *(encode(v) for v in cover.radii)]))
        for j,(basis,rows) in enumerate(zip(bases,memberships)):
            local=points[rows]
            repeated=np.repeat(local,2,axis=0)
            field_rows=[{c:Identity(2)} for _ in local for c in range(2)]
            powers=basis.poly.powers
            tokens=['DFP_PATCH',str(j),str(digits),str(backend.threads),'1',
                    str(basis.n),str(basis.m),encode(1),
                    *(encode(v) for v in bound.values),
                    *(encode(v) for v in repeated.flat),
                    *(encode(v) for v in repeated.flat),
                    *(encode(v) for v in basis.poly.origin),
                    *([encode(basis.poly.scale)]*2)]
            for column in basis.columns:
                terms=[(c,coefficient,power) for c in range(2)
                       for k,power in enumerate(powers)
                       if (coefficient:=column[c*len(powers)+k])]
                tokens.append(str(len(terms)))
                for component,coefficient,power in terms:
                    tokens.extend([str(component),encode(coefficient),*(str(v) for v in power)])
            for row in field_rows+field_rows:tokens.extend(_field_tokens(row,encode))
            reply=worker.request(' '.join(tokens))
            if not reply or int(reply[0])!=basis.n+basis.m:
                raise RuntimeError('Invalid native DFPU patch build result')
        fit.refit(data)
        fit.diagnostics['refits']=0
        points.setflags(write=False)
        return fit
    except Exception:
        worker.close(force=True)
        raise
