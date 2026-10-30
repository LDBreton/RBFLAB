"""Explicit compact patch covers and partition-of-unity interpolation."""
from dataclasses import dataclass
from math import comb
import numpy as np
import weakref

from .nodal import NodalBasis, dense_arithmetic
from .operators import Operator, Identity


@dataclass(frozen=True)
class PatchCover:
    """Fixed ball supports for partition-of-unity interpolation.

    ``centers`` has shape (J, 2) or (J, 3); ``radii`` has J positive entries.
    The compact C2 weight is ``(1-t)^4 (4t+1)`` for ``t<1``.
    Support membership uses strict distance less than radius.

    Args:
        centers: Finite patch-center coordinates of shape (J, 2) or (J, 3).
        radii: One finite positive support radius per patch.
    """
    centers: object
    radii: object

    @classmethod
    def from_points(cls, points, *, n_patches=None, points_per_patch=32,
                    overlap=0.25, min_points=None, seed=0):
        """Build reproducible ball supports for a finite 2D or 3D point cloud.

        Every input point has strictly positive support. ``points_per_patch``
        controls the target clustering size before overlap; final membership
        counts can be larger. This does not certify coverage between samples.
        ``min_points`` checks support occupancy, not polynomial rank.

        Args:
            points (array-like): Distinct finite coordinates with shape (N, 2) or (N, 3).
            n_patches (int or None): Exact number of patches, or ``None`` to infer from
                ``points_per_patch``.
            points_per_patch (int): Target clustering size when inferring patch count.
            overlap (float): Nonnegative radius expansion fraction, less than one.
            min_points (int or None): Minimum samples inside every support. Defaults to
                dimension plus one.
            seed (int): Nonnegative integer for reproducible center initialization.
        """
        from .patch_cover_builders import cover_from_points
        return cover_from_points(cls, points, n_patches=n_patches,
                                 points_per_patch=points_per_patch,
                                 overlap=overlap, min_points=min_points,
                                 seed=seed)

    @classmethod
    def from_mesh(cls, mesh, *, n_patches=None, points_per_patch=32,
                  overlap=0.25, min_points=None, seed=0):
        """Cover every triangle of a :class:`TriangleMesh2D` with open balls.

        Each triangle is fully inside at least one patch ball, including its
        edges and vertices. This guarantees coverage of the mesh's geometric
        domain, including disconnected components.

        Args:
            mesh (TriangleMesh2D): Validated :class:`TriangleMesh2D` instance.
            n_patches (int or None): Exact number of patches, or ``None`` to infer from
                ``points_per_patch`` with at least one per mesh component.
            points_per_patch (int): Target clustering size when inferring patch count.
            overlap (float): Nonnegative radius expansion fraction, less than one.
            min_points (int or None): Minimum mesh vertices inside every support. Defaults
                to three. This does not guarantee polynomial rank.
            seed (int): Nonnegative integer for reproducible center initialization.
        """
        from .patch_cover_builders import cover_from_mesh
        return cover_from_mesh(cls, mesh, n_patches=n_patches,
                               points_per_patch=points_per_patch,
                               overlap=overlap, min_points=min_points,
                               seed=seed)

    def __post_init__(self):
        from .methods import _query
        centers = _query(self.centers)
        radii = np.asarray(self.radii, dtype=float)
        if not len(centers) or radii.shape != (len(centers),) or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError('PatchCover requires one finite positive radius per patch center')
        centers = np.array(centers, copy=True)
        radii = np.array(radii, copy=True)
        centers.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, 'centers', centers)
        object.__setattr__(self, 'radii', radii)

    def memberships(self, points, ctx=None):
        from .methods import _query
        from .precision import mp_number
        points = _query(points, self.centers.shape[1])
        number = (lambda value:mp_number(ctx,value)) if ctx else float
        return tuple(np.asarray([i for i, point in enumerate(points)
                                 if sum((number(x)-number(c))**2 for x,c in zip(point,center)) < number(radius)**2], dtype=int)
                     for center, radius in zip(self.centers,self.radii))


def _weight(point, center, radius, ctx):
    """Weight, gradient and Hessian in the selected arithmetic."""
    from .precision import mp_number
    number = (lambda value:mp_number(ctx,value)) if ctx else float
    sqrt = ctx.sqrt if ctx else np.sqrt
    z = [number(x)-number(c) for x, c in zip(point, center)]
    radius = number(radius)
    square = sum(v*v for v in z)
    d = len(z)
    zero = number(0)
    if square >= radius*radius:
        return zero, [zero]*d, [[zero]*d for _ in range(d)]
    distance = sqrt(square)
    t = distance/radius
    # phi(t)=(1-t)^4 (4t+1).
    u = 1-t
    value = u**4*(4*t+1)
    dt = -20*t*u**3
    ddt = -20*u*u*(1-4*t)
    if distance == 0:
        return value, [zero]*d, [[ddt/(radius*radius) if i==j else zero for j in range(d)] for i in range(d)]
    radial = dt/(radius*distance)
    gradient = [radial*v for v in z]
    hessian = [[(ddt/(radius*radius)-radial)*z[i]*z[j]/(distance*distance)
                + (radial if i==j else zero) for j in range(d)] for i in range(d)]
    return value, gradient, hessian


def _normalized_weights(point, cover, ctx):
    raw = [_weight(point, c, r, ctx) for c, r in zip(cover.centers, cover.radii)]
    total = sum(v[0] for v in raw)
    if total <= 0:
        raise ValueError('Query point is outside the positive patch cover')
    d = len(point)
    first = [sum(v[1][i] for v in raw) for i in range(d)]
    second = [[sum(v[2][i][j] for v in raw) for j in range(d)] for i in range(d)]
    out = []
    for value, gradient, hessian in raw:
        w = value/total
        dw = [(gradient[i]-w*first[i])/total for i in range(d)]
        ddw = [[(hessian[i][j]-dw[i]*first[j]-dw[j]*first[i]-w*second[i][j])/total
                for j in range(d)] for i in range(d)]
        out.append((w, dw, ddw))
    return out


def _multiindices(alpha):
    from itertools import product
    return product(*(range(v+1) for v in alpha))


def _weight_derivative(weight, beta):
    order = sum(beta)
    if order == 0: return weight[0]
    axes = [i for i, count in enumerate(beta) for _ in range(count)]
    if order == 1: return weight[1][axes[0]]
    return weight[2][axes[0]][axes[1]]


class PatchInterpolant:
    def __init__(self, cover, patches, memberships, arithmetic, components, backend, points):
        self.cover, self.patches, self.memberships = cover, patches, memberships
        self.arithmetic, self.components, self.backend = arithmetic, components, backend
        self.points = points
        self._closed = False
        self.diagnostics = {'backend': 'python_pum', 'patches': len(patches),
                            'occupancy': [len(v) for v in memberships],
                            'factorizations': len(patches), 'components': components,
                            'ordering': 'node-major',
                            'scaled_conditions':[None if p['factor'].condition is None else float(p['factor'].condition)
                                                 for p in patches]}

    def _evaluate(self, points, operator):
        from .methods import _query
        points = _query(points, self.cover.centers.shape[1])
        d = points.shape[1]
        if operator is None: operator = Identity(d)
        if not isinstance(operator, Operator) or operator.dimension != d:
            raise TypeError('PUM evaluation requires a matching constant-coefficient Operator')
        if any(sum(alpha)>2 for alpha, _ in operator.terms):
            raise NotImplementedError('PUM supports derivatives through order two')
        a = self.arithmetic
        if self._closed: raise RuntimeError('Patch interpolant is closed')
        if not len(points): return []
        zero = a.number(0)
        result = [[zero for _ in range(self.components)] for _ in points]
        query_memberships=self.cover.memberships(points,a.ctx)
        weights=[_normalized_weights(point,self.cover,a.ctx) for point in points]
        needed = {tuple(alpha_i-beta_i for alpha_i,beta_i in zip(alpha,beta))
                  for alpha,_ in operator.terms for beta in _multiindices(alpha)}
        local_ops = {alpha: Operator(((alpha,1),),d) for alpha in needed}
        # Evaluate each local expansion once per derivative on all points in its support.
        for j,(patch,indices) in enumerate(zip(self.patches,self.memberships)):
            query_rows = query_memberships[j]
            if not len(query_rows): continue
            q = points[query_rows]
            cache = {}
            for alpha,op in local_ops.items():
                matrix=patch['basis'].evaluation(q,[op]*len(q))
                cache[alpha] = matrix*patch['coefficients'] if a.ctx else matrix@patch['coefficients']
            for k,row in enumerate(query_rows):
                weight = weights[row][j]
                for alpha, coefficient in operator.terms:
                    for beta in _multiindices(alpha):
                        scalar=a.number(coefficient)*_weight_derivative(weight,beta)
                        for ai,bi in zip(alpha,beta): scalar*=comb(ai,bi)
                        if scalar == 0: continue
                        values=cache[tuple(ai-bi for ai,bi in zip(alpha,beta))]
                        for c in range(self.components):
                            result[row][c]+=scalar*values[k,c]
        return result

    def evaluate(self, points, operator=None):
        result=np.asarray([[float(v) for v in row] for row in self._evaluate(points,operator)],dtype=float)
        if not len(result): result=result.reshape(0,self.components)
        return result[:,0] if self.components==1 else result

    def evaluate_mp(self, points, operator=None):
        if self.arithmetic.ctx is None: raise ValueError('evaluate_mp requires extended precision')
        result=self._evaluate(points,operator)
        if self.components==1:return self.arithmetic.ctx.matrix([row[0] for row in result]) if result else self.arithmetic.ctx.matrix(0,1)
        return self.arithmetic.ctx.matrix(result) if result else self.arithmetic.ctx.matrix(0,self.components)

    def close(self): self._closed=True
    def __enter__(self):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        return self
    def __exit__(self,*_): self.close()

    def refit(self, data):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        values = _data_values(data,self.points,self.arithmetic)
        if values.shape[1] != self.components:
            raise ValueError('Refit must keep the same number of components')
        replacements=[]
        a=self.arithmetic
        for patch,rows in zip(self.patches,self.memberships):
            basis=patch['basis']
            columns=[_checked_coefficients(patch['factor'].solve(basis.padded(values[rows,c])),a)
                     for c in range(self.components)]
            replacements.append(a.ctx.matrix([[column[i] for column in columns] for i in range(len(columns[0]))])
                                if a.ctx else np.column_stack(columns))
        for patch,coeff in zip(self.patches,replacements):patch['coefficients']=coeff
        self.diagnostics['refits']=self.diagnostics.get('refits',0)+1
        return self


def _data_values(data, points, arithmetic):
    raw=np.asarray(data.tolist() if hasattr(data,'rows') else data,dtype=object) if not callable(data) else None
    if raw is not None and raw.ndim==2:
        if raw.shape[0]!=len(points) or raw.shape[1]<1: raise ValueError('Data must have shape (N, C)')
        values=np.column_stack([arithmetic.data(raw[:,c],points) for c in range(raw.shape[1])])
    else:
        values=np.asarray(arithmetic.data(data,points),dtype=object).reshape(-1,1)
    if values.shape[0]!=len(points): raise ValueError('One datum per input point required')
    return values


def _checked_coefficients(column, arithmetic):
    finite=all(arithmetic.ctx.isfinite(v) for v in column) if arithmetic.ctx else np.isfinite(column).all()
    if not finite:raise FloatingPointError('Nonfinite patch coefficients')
    return column


def interpolate_patches(kernel, centers, data, precision, degree, cover, backend=None):
    from .methods import _query
    if not isinstance(cover,PatchCover): raise TypeError('patches must be PatchCover')
    points=np.array(_query(centers,cover.centers.shape[1]),copy=True)
    points.setflags(write=False)
    if not len(points) or len(np.unique(points,axis=0))!=len(points):
        raise ValueError('Expected nonempty distinct centers')
    arithmetic=dense_arithmetic(kernel,precision)
    memberships=cover.memberships(points,arithmetic.ctx)
    for rows in memberships: rows.setflags(write=False)
    if any(not len(rows) for rows in memberships):
        raise ValueError('Every patch must contain input nodes')
    covered=np.zeros(len(points),dtype=bool)
    for rows in memberships:covered[rows]=True
    if not covered.all():raise ValueError('Input point is outside the positive patch cover')
    values=_data_values(data,points,arithmetic)
    if backend is not None:
        return _interpolate_native(kernel,points,values,precision,degree,cover,backend,memberships,arithmetic)
    patches=[]
    for rows in memberships:
        basis=NodalBasis(arithmetic,points[rows],degree)
        matrix=basis.system_matrix(basis.centers,basis.identities)
        factor=arithmetic.factor(matrix)
        columns=[_checked_coefficients(factor.solve(basis.padded(values[rows,c])),arithmetic)
                 for c in range(values.shape[1])]
        coeff=arithmetic.ctx.matrix([[column[i] for column in columns] for i in range(len(columns[0]))]) if arithmetic.ctx else np.column_stack(columns)
        patches.append({'basis':basis,'factor':factor,'coefficients':coeff})
    return PatchInterpolant(cover,patches,memberships,arithmetic,values.shape[1],backend,points)


class NativePatchInterpolant:
    """A bounded collection of patch factors in one persistent native worker."""
    def __init__(self, worker, cover, points, memberships, arithmetic, components, backend, digits, sizes):
        self.worker,self.cover,self.points,self.memberships=worker,cover,points,memberships
        self.arithmetic,self.components,self.backend=arithmetic,components,backend
        self.digits,self.sizes=digits,sizes
        self._closed=False
        self._finalizer=weakref.finalize(self,worker.close)
        self.diagnostics={'backend':'cpp_pum','arithmetic':'mpfr' if arithmetic.ctx else 'float64',
                          'data_backend':'python','patches':len(sizes),'occupancy':[len(x) for x in memberships],
                          'factorizations':len(sizes),'components':components,'ordering':'node-major',
                          'linear_solver':backend.linear_solver}

    def _encode(self,value):
        from fractions import Fraction
        from .precision import mp_number
        import math
        if isinstance(value,Fraction):
            return self._encode(self.arithmetic.number(value.numerator)/self.arithmetic.number(value.denominator))
        ctx=self.arithmetic.ctx
        if ctx:
            number=mp_number(ctx,value)
            if not ctx.isfinite(number):raise ValueError('Expected finite native input')
            return ctx.nstr(number,self.digits+8)
        number=float(value)
        if not math.isfinite(number):raise ValueError('Expected finite native input')
        return format(number,'.17g')

    def _evaluate(self,points,operator):
        from .methods import _query
        from .global_backend import _op_tokens
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        points=_query(points,self.points.shape[1])
        dimension=self.points.shape[1]
        if operator is None:operator=Identity(dimension)
        if not isinstance(operator,Operator) or operator.dimension!=dimension:
            raise TypeError('PUM evaluation requires a matching constant-coefficient Operator')
        if any(sum(alpha)>2 for alpha,_ in operator.terms):
            raise NotImplementedError('PUM supports derivatives through order two')
        if any(sum(alpha)>self.backend.derivative_order for alpha,_ in operator.terms):
            raise ValueError('Increase CppGlobalBackend(derivative_order=...) for this query')
        if not len(points):return []
        tokens=['PUM_EVAL',str(len(points)),*(self._encode(v) for v in points.flat),
                *_op_tokens(operator,self._encode)]
        try:reply=self.worker.request(' '.join(tokens))
        except RuntimeError as exc:
            if 'outside positive patch cover' in str(exc):
                raise ValueError('Query point is outside the positive patch cover') from exc
            raise
        if len(reply)!=2+len(points)*self.components or int(reply[0])!=len(points) or int(reply[1])!=self.components:
            raise RuntimeError('Invalid native PUM evaluation result')
        number=self.arithmetic.number
        return [[number(reply[2+i*self.components+c]) for c in range(self.components)] for i in range(len(points))]

    def evaluate(self,points,operator=None):
        result=np.asarray([[float(v) for v in row] for row in self._evaluate(points,operator)],dtype=float)
        if not len(result):result=result.reshape(0,self.components)
        return result[:,0] if self.components==1 else result

    def evaluate_mp(self,points,operator=None):
        ctx=self.arithmetic.ctx
        if ctx is None:raise ValueError('evaluate_mp requires extended precision')
        result=self._evaluate(points,operator)
        if self.components==1:return ctx.matrix([row[0] for row in result]) if result else ctx.matrix(0,1)
        return ctx.matrix(result) if result else ctx.matrix(0,self.components)

    def refit(self,data):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        values=_data_values(data,self.points,self.arithmetic)
        if values.shape[1]!=self.components:raise ValueError('Refit must keep the same number of components')
        tokens=['PUM_REFIT']
        for rows,size in zip(self.memberships,self.sizes):
            for c in range(self.components):
                tokens.extend(self._encode(v) for v in values[rows,c])
                tokens.extend(['0']*(size-len(rows)))
        self.worker.request(' '.join(tokens))
        self.diagnostics['refits']=self.diagnostics.get('refits',0)+1
        reply=self.worker.request('PUM_DIAG '+str(int(self.backend.compute_condition)))
        if not reply or int(reply[0])!=len(self.sizes) or len(reply)!=1+2*len(self.sizes):
            raise RuntimeError('Invalid native PUM diagnostics')
        self.diagnostics['relative_residuals']=[None if reply[1+2*j]=='none' else float(reply[1+2*j]) for j in range(len(self.sizes))]
        self.diagnostics['scaled_conditions']=[None if reply[2+2*j]=='none' else float(reply[2+2*j]) for j in range(len(self.sizes))]
        return self

    def close(self):
        if not self._closed:
            self._finalizer()
            self._closed=True
    def __enter__(self):
        if self._closed:raise RuntimeError('Patch interpolant is closed')
        self.worker.check_open()
        return self
    def __exit__(self,*_):self.close()


def _interpolate_native(kernel,points,values,precision,degree,cover,backend,memberships,arithmetic):
    from .global_backend import CppGlobalBackend,_compiled,_acquire_worker,_op_tokens
    if not isinstance(backend,CppGlobalBackend):raise TypeError('interpolate backend must be CppGlobalBackend or None')
    dimension=points.shape[1]
    digits=precision.global_digits or 17
    bases=[]
    for rows in memberships:
        basis=NodalBasis(arithmetic,points[rows],degree)
        # Screen the small polynomial block before compilation and factorization.
        if basis.powers:
            polynomial=basis.polynomials(basis.centers,basis.identities)
            check=np.asarray(polynomial.tolist() if arithmetic.ctx else polynomial,dtype=float)
            row=np.max(abs(check),axis=1)
            check/=np.where(row>0,row,1)[:,None]
            column=np.max(abs(check),axis=0)
            check/=np.where(column>0,column,1)[None,:]
            if np.linalg.matrix_rank(check)<len(basis.powers):
                raise np.linalg.LinAlgError('Polynomial functionals are not unisolvent')
        bases.append(basis)
    executable,bound=_compiled(kernel,dimension,min(2,backend.derivative_order),
                               'mpfr' if arithmetic.ctx else 'float64',backend)
    worker=_acquire_worker(executable,backend)
    fit=NativePatchInterpolant(worker,cover,points,memberships,arithmetic,values.shape[1],backend,digits,[])
    encode=fit._encode
    tokens=['PUM_INIT',str(len(memberships)),str(values.shape[1]),str(digits),
            *(encode(v) for v in cover.centers.flat),*(encode(v) for v in cover.radii)]
    try:
        worker.request(' '.join(tokens))
        for j,(rows,basis) in enumerate(zip(memberships,bases)):
            ops=[Identity(dimension)]*len(rows)
            size=len(rows)+len(basis.powers)
            tokens=['PUM_PATCH',str(j),str(digits),str(backend.threads),'1',str(len(rows)),
                    str(len(basis.powers)),encode(1),*(encode(v) for v in bound.values),
                    *(encode(v) for v in points[rows].flat),*(encode(v) for v in points[rows].flat),
                    *(encode(v) for v in basis.origin),*([encode(basis.scale)]*dimension),
                    *(str(v) for power in basis.powers for v in power)]
            for op in ops+ops:tokens.extend(_op_tokens(op,encode))
            reply=worker.request(' '.join(tokens))
            if not reply or int(reply[0])!=size:raise RuntimeError('Invalid native patch build result')
            fit.sizes.append(size)
        fit.diagnostics['patches']=len(fit.sizes)
        fit.diagnostics['factorizations']=len(fit.sizes)
        fit.refit(values)
        fit.diagnostics['refits']=0
        return fit
    except Exception:
        worker.close(force=True)
        raise
