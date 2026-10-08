"""Selectable local numerical backends for 2D divergence-free LHI Stokes.

Python workers compute independent stencils in spawned processes. The owned C++
backend computes smooth/hybrid stencils with MPFR/OpenMP. Global assembly stays shared.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing,os,subprocess,tempfile,time,json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from .stokes import StokesArithmetic,_blocks
from .nodal import Arithmetic
from .operators import Identity,Derivative,Laplacian
from .kernels import IMQ,ScalarKernel,Hybrid
from .symbolic_kernel import BoundKernel
from .stencils import neighbors,geometry_quality
from .stokes_polynomials import StokesPolynomialBasis,append_columns,rank_ratio
from .legacy_cpp import _linux_path

_WORKER_ARITH={}
_WORKER_THREAD_LIMIT=None
def _init_worker():
    from threadpoolctl import threadpool_limits
    global _WORKER_THREAD_LIMIT
    _WORKER_THREAD_LIMIT=threadpool_limits(limits=1)

def _build(task,a=None):
    if a is None:a=StokesArithmetic(task['vk'],task['pk'],task['digits'])
    p=task['points'];ops=task['ops'];scale=a.number(task['scale']) if task['scale'] is not None else None
    G=_blocks(a,p,ops,p,ops,scale);basis=None
    if task['degree'] is not None:
        basis=StokesPolynomialBasis(a,task['origin'],a.number(task['radius']),task['degree'])
        poly=basis.matrix(p,ops)
        if rank_ratio(poly)<1e-12:raise ValueError('Polynomial functionals are rank deficient; enlarge stencil')
        n=len(p);m=len(basis.columns);A=a.zeros(n+m,n+m);A[:n,:n]=G;A[:n,n:]=poly;A[n:,:n]=poly.T;G=A
    return a,G,basis


def _python_job(task):
    key=(repr(task['vk']),repr(task['pk']),task['digits'])
    if key not in _WORKER_ARITH:_WORKER_ARITH[key]=StokesArithmetic(task['vk'],task['pk'],task['digits'])
    a=_WORKER_ARITH[key]
    a,G,basis=_build(task,a)
    targets=[{j:-Laplacian()*task['mu'],2:Derivative(j)} for j in (0,1)]+[{2:Derivative(j)} for j in (0,1)]
    Q=_blocks(a,np.tile(task['origin'],(4,1)),targets,task['points'],task['ops'],a.number(task['scale']) if task['scale'] is not None else None)
    if basis is not None:Q=append_columns(a,Q,basis.matrix(np.tile(task['origin'],(4,1)),targets))
    factor=a.factor(G,compute_condition=task['condition']);W=factor.solve(Q.T,transpose=True)
    defect=(G.T*W-Q.T) if a.ctx else G.T@W-Q.T
    residual=a.norm(defect)/(a.norm(Q.T) or a.number(1))
    encode=lambda x:a.ctx.nstr(x,a.ctx.dps+5) if a.ctx else float(x)
    return dict(weights=[[encode(W[i,j]) for i in range(len(task['points']))] for j in range(4)],condition=factor.condition,residual=encode(residual))


class _SVDFactor:
    """Lazy Python reconstruction with the C++ equilibration/truncation policy."""
    def __init__(self,G,rcond):
        self.scale=1/np.sqrt(np.max(np.abs(G),axis=1))
        self.u,values,self.vh=np.linalg.svd(self.scale[:,None]*G*self.scale[None,:],full_matrices=False)
        cutoff=len(G)*np.finfo(float).eps if rcond is None else rcond
        self.inverse=np.zeros_like(values)
        keep=values>max(cutoff*values[0],np.finfo(float).tiny)
        self.inverse[keep]=1/values[keep]
    def solve(self,rhs,transpose=False):
        b=np.asarray(rhs);vector=b.ndim==1
        if vector:b=b[:,None]
        left,right=(self.vh.T,self.u.T) if not transpose else (self.u,self.vh)
        result=self.scale[:,None]*(left@(self.inverse[:,None]*(right@(self.scale[:,None]*b))))
        return result[:,0] if vector else result


class _DeferredFactor:
    def __init__(self,a,task,condition):self.a,self.task,self.condition,self.factor=a,task,condition,None
    def solve(self,rhs,transpose=False):
        if self.factor is None:
            _,G,_=_build(self.task,self.a)
            # An already measured condition is retained; do not recompute it on query.
            self.factor=(_SVDFactor(G,self.task.get('svd_rcond')) if self.task.get('local_solver')=='svd' else self.a.factor(G,compute_condition=False))
        return self.factor.solve(rhs,transpose=transpose)


def _hardy_kernel(kernel, factor, arithmetic):
    """Scale only the smooth coefficient; retain the hybrid PHS term verbatim."""
    smooth = kernel.smooth if isinstance(kernel, Hybrid) else kernel
    if not isinstance(smooth, ScalarKernel):
        raise NotImplementedError('Hardy scaling requires an IMQ or Gaussian smooth kernel')
    value = arithmetic.number(smooth.c) * factor
    value = arithmetic.ctx.nstr(value, arithmetic.ctx.dps+5) if arithmetic.ctx else float(value)
    scaled = replace(smooth, c=value)
    return replace(kernel, smooth=scaled) if isinstance(kernel, Hybrid) else scaled


def _prepare(method,problem,cloud,shape_rule):
    from .lhi_stokes import VectorHermiteStencil
    if shape_rule not in ('fixed', 'legacy_hardy', 'radius_scaled', 'separation_scaled'):raise ValueError('Unknown shape rule')
    if shape_rule == 'legacy_hardy':
        if method.stencil_policy.scaling != 'physical':raise ValueError('Hardy shape rule uses physical coordinates')
        if method.polynomial_degree is not None:raise NotImplementedError('Hardy scaling currently requires unaugmented kernels')
        if method.stencil_policy.selection != 'nearest' or method.min_boundary_centers or method.pde_stencil_size is not None:
            raise NotImplementedError('Historical Hardy scaling requires the original nearest-stencil layout')
    pk=method.pressure_kernel if method.pressure_kernel is not None else method.kernel
    a=StokesArithmetic(method.kernel,pk,method.precision.local_digits)
    ii,bi=cloud.interior_indices,cloud.boundary_indices;ni,nb=len(ii),len(bi)
    if not ni or not nb or type(method.stencil_size) is not int or not 3<=method.stencil_size<=len(cloud.points):raise ValueError('Invalid stencil size or cloud')
    if method.min_boundary_centers>nb:raise ValueError('min_boundary_centers exceeds boundary count')
    if method.pde_stencil_size is not None and (type(method.pde_stencil_size) is not int or not 0<=method.pde_stencil_size<ni):raise ValueError('Invalid pde_stencil_size')
    imap={int(j):i for i,j in enumerate(ii)};bmap={int(j):i for i,j in enumerate(bi)}
    tree=cKDTree(cloud.points);ptree=cKDTree(cloud.interior);btree=cKDTree(cloud.points[bi])
    momentum=[{j:-Laplacian()*problem.viscosity,2:Derivative(j)} for j in (0,1)]
    tasks=[];stencils=[]
    for center in ii:
        selected=method.stencil_policy.select(tree,cloud.points[center],method.stencil_size)
        sc=[int(j) for j in selected if int(j) in imap];bc=[int(j) for j in selected if int(j) in bmap]
        if bc and len(bc)<method.min_boundary_centers:
            bc=sorted(set(bc)|set(map(int,bi[neighbors(btree,cloud.points[center],method.min_boundary_centers)])))
        pc=[j for j in sc if j!=center]
        if method.pde_stencil_size is not None:pc=[int(j) for j in ii[neighbors(ptree,cloud.points[center],method.pde_stencil_size+1)] if j!=center][:method.pde_stencil_size]
        if not pc:raise ValueError('Each stencil needs PDE centers')
        points=[];ops=[];slots=[];codes=[]
        for name,ids,mapping,size in [('solution',sc,imap,ni),('boundary',bc,bmap,nb),('pde',pc,imap,ni)]:
            for c in (0,1):
                points.extend(cloud.points[ids]);ops.extend([momentum[c] if name=='pde' else {c:Identity()}]*len(ids))
                slots.extend([(name,mapping[j]+c*size) for j in ids]);codes.extend([c+5 if name=='pde' else c+1]*len(ids))
        points=np.array(points);local=a
        if shape_rule=='legacy_hardy':
            if method.stencil_policy.scaling!='physical':raise ValueError('Hardy shape rule uses physical coordinates')
            radius=float(tree.query(cloud.points[center],k=method.stencil_size)[0][-1]);factor=(a.ctx.sqrt if a.ctx else np.sqrt)(len(sc)+len(bc)+len(pc))*a.number('.815')/a.number(radius)
            local=StokesArithmetic(_hardy_kernel(method.kernel,factor,a),_hardy_kernel(pk,factor,a),method.precision.local_digits)
            if a.ctx:
                local.ctx=a.ctx;local.backend.ctx=a.ctx;local.pressure_arithmetic.ctx=a.ctx;local.pressure_arithmetic.backend.ctx=a.ctx
        sqrt=a.ctx.sqrt if a.ctx else np.sqrt
        radius=max(sqrt(sum((a.number(v)-a.number(w))**2 for v,w in zip(p,cloud.points[center]))) for p in points)
        if shape_rule in ('radius_scaled','separation_scaled'):
            if method.stencil_policy.scaling!='physical':
                raise ValueError('Geometry shape rules require physical coordinates')
            if not all(isinstance(k,ScalarKernel) for k in (method.kernel,pk)):
                raise NotImplementedError('Geometry shape rules currently support smooth scalar potentials only')
            length=radius
            if shape_rule=='separation_scaled':
                unique=np.unique(points,axis=0)
                # Separation here means the minimum pair distance, not half that distance.
                length=a.number(float(cKDTree(unique).query(unique,k=2)[0][:,1].min()))
            if length<=0:raise ValueError('Positive stencil length required')
            factor=a.number(1)/(length*length)
            local=StokesArithmetic(_hardy_kernel(method.kernel,factor,a),_hardy_kernel(pk,factor,a),method.precision.local_digits)
            if a.ctx:
                local.ctx=a.ctx;local.backend.ctx=a.ctx;local.pressure_arithmetic.ctx=a.ctx;local.pressure_arithmetic.backend.ctx=a.ctx
        encode=lambda v:a.ctx.nstr(v,a.ctx.dps+5) if a.ctx else float(v)
        scale=radius if method.stencil_policy.scaling=='local' else None
        task=dict(vk=local.kernel,pk=local.pressure_arithmetic.kernel,digits=method.precision.local_digits,points=points,ops=ops,origin=cloud.points[center],mu=problem.viscosity,scale=encode(scale) if scale is not None else None,radius=encode(radius),degree=method.polynomial_degree,codes=codes)
        basis=StokesPolynomialBasis(a,cloud.points[center],radius,method.polynomial_degree) if method.polynomial_degree is not None else None
        stencil=VectorHermiteStencil(int(center),points,ops,slots,None,scale,geometry_quality(cloud.points[np.unique(sc+bc+pc)],cloud.points[center],1),basis)
        stencil.local_arithmetic=local;tasks.append(task);stencils.append(stencil)
    return a,tasks,stencils


def _finish(method,problem,cloud,a,tasks,stencils,results,diagnostics):
    from .lhi_stokes import LHIStokesSystem,_add
    g=a if method.precision.global_dtype=='mpmath' else Arithmetic(method.kernel);n=len(stencils)
    sy=[{} for _ in range(2*n)];sb=[{} for _ in range(2*n)];sl=[{} for _ in range(2*n)];saved=[None]*(4*n)
    for i,(task,stencil,result) in enumerate(zip(tasks,stencils,results)):
        if result.get('svd'):
            task=dict(task,local_solver='svd',svd_rcond=result['svd']['threshold'])
        stencil.factor=_DeferredFactor(stencil.local_arithmetic,task,result['condition'])
        stencil.geometry_diagnostics['local_solve_residual']=float(result['residual'])
        for target in range(4):
            weights=[a.number(v) for v in result['weights'][target]]
            if len(weights)!=len(stencil.slots):raise RuntimeError('Backend weight count mismatch')
            saved[target*n+i]=weights
            if target<2:
                for (group,col),w in zip(stencil.slots,weights):_add({'solution':sy,'boundary':sb,'pde':sl}[group][i+target*n],col,g.number(w))
    mass=[{i:g.number(1)} for i in range(2*n)]
    for i,row in enumerate(sl):
        for j,v in row.items():_add(mass[i],j,-v)
    system=LHIStokesSystem(problem,cloud,a,g,stencils,sy,sb,sl,mass)
    system.reconstruction_weights=saved;system.lazy_reconstruction=True
    diagnostics['max_local_residual']=max(float(r['residual']) for r in results)
    diagnostics['max_local_condition']=max((r['condition'] for r in results if r['condition'] is not None),default=None)
    system.backend_diagnostics=diagnostics
    return system


@dataclass(frozen=True)
class PythonBackend:
    workers: int = 1
    compute_condition: bool = True
    shape_rule: str = 'fixed'
    def __post_init__(self):
        if self.shape_rule not in ('fixed','legacy_hardy','radius_scaled','separation_scaled'):raise ValueError('Unknown shape rule')
        if type(self.workers) is not int or self.workers<1:raise ValueError('workers must be a positive integer')
        if type(self.compute_condition) is not bool:raise TypeError('compute_condition must be bool')
    def assemble(self,method,problem,cloud):
        start=time.perf_counter();a,tasks,stencils=_prepare(method,problem,cloud,self.shape_rule)
        for task in tasks:task['condition']=self.compute_condition
        if self.workers==1:results=[_python_job(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(self.workers,len(tasks)),mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker) as pool:
                results=list(pool.map(_python_job,tasks,chunksize=1))
        system=_finish(method,problem,cloud,a,tasks,stencils,results,dict(backend='python',shape_rule=self.shape_rule,workers=min(self.workers,len(tasks)),compute_condition=self.compute_condition,local_digits=method.precision.local_digits,off_node_backend='lazy Python',condition_norm='infinity' if a.ctx else '2'))
        system.backend_diagnostics['assembly_seconds']=time.perf_counter()-start
        return system


@dataclass(frozen=True)
class CppBackend:
    threads: int = 1
    compute_condition: bool = True
    shape_rule: str = 'fixed'
    executable: str | None = None
    distribution: str = 'Ubuntu'
    timeout: float = 300
    cache_dir: str | None = None
    local_solver: str = 'lu'
    svd_rcond: float | None = None
    def __post_init__(self):
        if self.local_solver not in ('lu','svd'):raise ValueError('local_solver must be lu or svd')
        if self.svd_rcond is not None and (not np.isfinite(self.svd_rcond) or not 0 <= self.svd_rcond < 1):raise ValueError('svd_rcond must be finite in [0,1)')
        if self.local_solver!='svd' and self.svd_rcond is not None:raise ValueError('svd_rcond requires local_solver=svd')
        if type(self.threads) is not int or self.threads<1:raise ValueError('threads must be positive integer')
        if type(self.compute_condition) is not bool:raise TypeError('compute_condition must be bool')
        if self.shape_rule not in ('fixed','legacy_hardy','radius_scaled','separation_scaled'):raise ValueError('Unknown shape rule')
    def assemble(self,method,problem,cloud):
        start=time.perf_counter();pk=method.pressure_kernel or method.kernel
        if method.stencil_policy.scaling!='physical':
            raise NotImplementedError('Compiled backend uses physical coordinates; use PythonBackend for local scaling')
        for k in (method.kernel,pk):
            if not isinstance(k,(ScalarKernel,Hybrid,BoundKernel)):raise NotImplementedError('Compiled kernels: IMQ, Gaussian, and their supported PHS hybrids')
        native = method.precision.local_digits is None
        if not native and self.local_solver=='svd':raise NotImplementedError('SVD local solver currently requires Float64')
        if self.shape_rule=='legacy_hardy' and (method.stencil_policy.selection!='nearest' or method.min_boundary_centers or method.pde_stencil_size is not None):
            raise NotImplementedError('Historical Hardy scaling requires the original nearest-stencil layout')
        custom=any(isinstance(k,BoundKernel) for k in (method.kernel,pk))
        custom_executable=None
        if custom:
            from .kernel_compiler import compile_stokes
            custom_executable=compile_stokes(method.kernel,pk,'float64' if native else 'mpfr',self.cache_dir)
        executable=custom_executable or (Path(self.executable) if self.executable else Path(__file__).resolve().parents[2]/'cpp'/('lhi_double' if native else 'lhi_mpfr'))
        if not executable.is_file():raise FileNotFoundError('Build the C++ backend first: python examples/build_cpp_backend.py')
        build_info=None
        if self.executable is None and not custom:
            repository=Path(__file__).resolve().parents[2];manifest=executable.parent/('build_manifest_double.json' if native else 'build_manifest.json')
            if not manifest.is_file():raise RuntimeError('Build manifest missing; rerun examples/build_cpp_backend.py')
            build_info=json.loads(manifest.read_text())
            if build_info.get('protocol')!=2:raise RuntimeError('Unsupported C++ build protocol')
            for name,digest in build_info['source_sha256'].items():
                if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('C++ sources changed; rerun examples/build_cpp_backend.py')
            if hashlib.sha256(executable.read_bytes()).hexdigest()!=build_info['binary_sha256']:raise RuntimeError('C++ executable changed; rebuild it')
        a,tasks,stencils=_prepare(method,problem,cloud,self.shape_rule);digits=a.ctx.dps if a.ctx else 17
        num=lambda v:a.ctx.nstr(a.number(v),digits+5) if a.ctx else format(float(v),'.17g')
        lines=[f'{digits} {self.threads} {int(self.compute_condition)} {len(tasks)}']
        for task in tasks:
            task['condition']=self.compute_condition
            basis=None
            if task['degree'] is not None:
                basis=StokesPolynomialBasis(a,task['origin'],a.number(task['radius']),task['degree'])
                P=basis.matrix(task['points'],task['ops'])
                if rank_ratio(P)<1e-12:raise ValueError('Polynomial functionals are rank deficient; enlarge stencil')
                targets=[{j:-Laplacian()*task['mu'],2:Derivative(j)} for j in (0,1)]+[{2:Derivative(j)} for j in (0,1)]
                T=basis.matrix(np.tile(task['origin'],(4,1)),targets)
            def spec(k):
                if custom:
                    from .kernel_compiler import as_bound
                    values=as_bound(k).values
                    return [str(len(values)),*(num(v) for v in values)]
                smooth=k.smooth if isinstance(k,Hybrid) else k
                return [str(int(smooth.family=='gaussian')),str(k.phs.power if isinstance(k,Hybrid) else 0),num(smooth.c),num(k.weight if isinstance(k,Hybrid) else 0)]
            lines.append(' '.join([str(len(task['points'])),str(len(basis.columns) if basis else 0),*spec(task['vk']),*spec(task['pk']),num(task['mu']),*(num(v) for v in task['origin'])]))
            lines.extend(' '.join([str(code),*(num(v) for v in p)]) for code,p in zip(task['codes'],task['points']))
            if basis:
                lines.extend(' '.join(num(P[i,j]) for j in range(P.cols if a.ctx else P.shape[1])) for i in range(P.rows if a.ctx else P.shape[0]))
                lines.extend(' '.join(num(T[k,j]) for k in range(4)) for j in range(T.cols if a.ctx else T.shape[1]))
        with tempfile.TemporaryDirectory(prefix='rbflab_mpfr_') as temporary:
            folder=Path(temporary);source=folder/'input.txt';target=folder/'output.txt';source.write_text('\n'.join(lines)+'\n',encoding='ascii')
            args=[_linux_path(executable),_linux_path(source),_linux_path(target)]
            if self.local_solver=='svd':args += ['svd',str(self.svd_rcond if self.svd_rcond is not None else -1)]
            if os.name=='nt':args=['wsl.exe','-d',self.distribution,'--']+args
            before=time.perf_counter();run=subprocess.run(args,capture_output=True,text=True,timeout=self.timeout);elapsed=time.perf_counter()-before
            if run.returncode:raise RuntimeError('C++ numerical backend: '+run.stderr)
            results=[]
            for expected,line in enumerate(target.read_text().splitlines()):
                items=line.split();index,n=int(items[0]),int(items[1]);condition=None if items[2]=='none' else float(items[2])
                if index!=expected or n!=len(tasks[expected]['points']) or len(items)!=4+4*n+(4 if self.local_solver=='svd' else 0):raise RuntimeError('Malformed C++ output')
                if any(not (a.ctx.isfinite(a.number(v)) if a.ctx else np.isfinite(float(v))) for v in items[3:4+4*n]):raise RuntimeError('Nonfinite C++ weights/residual')
                svd_info={}
                if self.local_solver=='svd':
                    rank,size,cutoff,cond2=items[-4:]
                    svd_info=dict(rank=int(rank),size=int(size),threshold=float(cutoff),condition2=float(cond2))
                    condition=float(cond2) if self.compute_condition else None
                results.append(dict(svd=svd_info,condition=condition,residual=items[3],weights=[items[4+j*n:4+(j+1)*n] for j in range(4)]))
            if len(results)!=len(tasks):raise RuntimeError('Incomplete C++ results')
        system=_finish(method,problem,cloud,a,tasks,stencils,results,dict(backend='cpp_float64' if native else 'cpp_mpfr',local_solver=self.local_solver,svd_rcond=self.svd_rcond,svd_stencils=[r['svd'] for r in results] if self.local_solver=='svd' else None,build=build_info,threads=self.threads,compute_condition=self.compute_condition,shape_rule=self.shape_rule,local_digits=method.precision.local_digits,cpp_process_seconds=elapsed,off_node_backend='lazy Python with matching kernels and '+self.local_solver,condition_norm='2 (untruncated singular values)' if self.local_solver=='svd' else 'infinity'))
        system.backend_diagnostics['assembly_seconds']=time.perf_counter()-start
        return system
