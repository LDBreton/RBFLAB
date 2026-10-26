"""Persistent native dense scalar collocation worker."""
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import math
import os
from .native_toolchain import executable_path
import subprocess
import threading
import weakref
import atexit
import numpy as np

from .precision import mp_number, mp_values


@dataclass(frozen=True)
class CppGlobalBackend:
    """Compile and reuse a C++ scalar or divergence-free Stokes global worker.

    Native assembly, factorization, solves, and evaluation stay in the selected
    arithmetic. Accessing ``system.matrix`` explicitly exports entries lazily.
    Windows uses native MinGW-w64 GCC and directly launches Windows executables.

    Args:
        threads: Thread limit for native assembly, evaluation, matrix products,
            and Eigen's parallel global dense factorization when applicable.
        compute_condition: Estimate the balanced matrix one-norm condition.
        cache_dir: Directory for compiled native artifacts.
        distribution: Compatibility field; retain the default Ubuntu. CPU execution is native.
        timeout: Maximum seconds for each worker command.
        derivative_order: Maximum evaluation derivative order to prepare, in
            addition to source functional derivatives. Total order is at most 6.
        reuse_workers: Keep up to four idle workers for subsequent assemblies.
        linear_solver: Dense factor engine, ``'eigen'`` (default) or the optional
            MPFR-only ``'mplapack'``. Assembly and evaluation are unchanged.
        mplapack_dir: Explicit installed MPLAPACK MPFR/OpenMP dependency prefix.
    """
    threads: int = 1
    compute_condition: bool = True
    cache_dir: object = None
    distribution: str = 'Ubuntu'
    timeout: float = 120.0
    derivative_order: int = 2
    reuse_workers: bool = True
    linear_solver: str = 'eigen'
    mplapack_dir: object = None

    def __post_init__(self):
        if self.linear_solver not in ('eigen','mplapack'):
            raise ValueError("linear_solver must be 'eigen' or 'mplapack'")
        if self.linear_solver=='mplapack' and self.mplapack_dir is None:
            raise ValueError('mplapack_dir is required for the optional MPLAPACK solver')
        if self.linear_solver=='eigen' and self.mplapack_dir is not None:
            raise ValueError('mplapack_dir requires linear_solver=\'mplapack\'')
        if type(self.threads) is not int or self.threads < 1: raise ValueError('threads must be positive')
        if type(self.compute_condition) is not bool: raise TypeError('compute_condition must be bool')
        if type(self.reuse_workers) is not bool: raise TypeError('reuse_workers must be bool')
        if self.distribution != 'Ubuntu': raise NotImplementedError('distribution is a compatibility field; retain Ubuntu')
        if not isinstance(self.timeout,(float,int)) or not math.isfinite(self.timeout) or self.timeout <= 0: raise ValueError('timeout must be positive')
        if type(self.derivative_order) is not int or not 0 <= self.derivative_order <= 6:
            raise ValueError('derivative_order must be between 0 and 6')


# Only fully validated disk artifacts enter this process-local memo. Stat checks
# retain invalidation while avoiding repeated dependency and executable hashing.
_compiled_memo = {}
_compiled_memo_lock = threading.RLock()


def _file_signature(path):
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
                stat.st_ino, stat.st_mode)
    except OSError:
        return None


def _memo_valid(entry):
    return all(signature is not None and _file_signature(path) == signature
               for path, signature in entry[1])


def _remember_compiled(key, directory, index, root, dependency=None):
    from . import kernel_compiler
    dependencies = [Path(__file__), Path(kernel_compiler.__file__),
                    Path(kernel_compiler.__file__).with_name('symbolic_kernel.py'),
                    Path(kernel_compiler.__file__).with_name('native_toolchain.py')]
    flow_bridge = Path(__file__).with_name('global_flow_backend.py')
    if flow_bridge.exists(): dependencies.append(flow_bridge)
    data_codegen=Path(__file__).with_name('global_data_codegen.py')
    if data_codegen.exists():dependencies.append(data_codegen)
    dependencies.extend(root/'cpp'/name for name in
                        ('global_collocation.hpp', 'real.hpp', 'real_double.hpp',
                         'lhi_mpfr.cpp', 'eigen_solver.hpp','global_solver.hpp'))
    if dependency is not None:
        dependencies.append(Path(__file__).with_name('mplapack_build.py'))
        dependencies.extend(dependency.files)
    # Directory stats detect new/removed Eigen headers without walking the tree
    # on warm calls; file stats detect edits to every hashed dependency.
    from .eigen_build import eigen_dependency
    eigen = root/'cpp/deps/usr/include/eigen3/Eigen'
    if not eigen.is_dir():eigen = eigen_dependency()[0]/'Eigen'
    dependencies.extend([eigen, *eigen.rglob('*')])
    manifest = directory/'manifest.json'
    files = json.loads(manifest.read_text())['files']
    dependencies.extend([index, manifest, executable_path(directory/'kernel')])
    dependencies.extend(directory/name for name in files)
    signatures = tuple((path, _file_signature(path)) for path in dependencies)
    if all(signature is not None for _, signature in signatures):
        with _compiled_memo_lock:
            if key not in _compiled_memo and len(_compiled_memo) >= 32:
                _compiled_memo.pop(next(iter(_compiled_memo)))
            _compiled_memo[key] = (executable_path(directory/'kernel'), signatures)


def _compiled(kernel, dimension, order, arithmetic, backend, *, pressure_kernel=None, pressure_order=0,data_plan=None,vector_only=False):
    from .kernel_compiler import as_bound, request_cache, record_request, header, build, ROOT, PRELUDE
    bound=as_bound(kernel)
    if vector_only and pressure_kernel is not None:raise ValueError('Vector-only workers do not have pressure')
    family=bound.family
    base=(Path(backend.cache_dir) if backend.cache_dir else Path.home()/'.cache'/'rbflab').resolve()
    key=(family.expression, family.radial_variable, family.parameters,
         family.dimension, dimension, order, arithmetic, base,backend.linear_solver,
         Path(backend.mplapack_dir).resolve() if backend.mplapack_dir is not None else None,vector_only,
         os.environ.get('RBFLAB_CXX'),os.environ.get('RBFLAB_NATIVE_PREFIX'))
    if backend.linear_solver=='mplapack' and arithmetic!='mpfr':
        raise ValueError('MPLAPACK requires Precision(global_digits=...) with MPFR arithmetic')
    pressure=as_bound(pressure_kernel) if pressure_kernel is not None else None
    families=(family,) if pressure is None else (family,pressure.family)
    if pressure is not None:
        f=pressure.family
        key+=('fields',f.expression,f.radial_variable,f.parameters,f.dimension,pressure_order)
    if data_plan is not None:key+=('data',data_plan.expressions)
    with _compiled_memo_lock:
        entry=_compiled_memo.get(key)
        if entry is not None and _memo_valid(entry):return entry[0],bound
        _compiled_memo.pop(key,None)
    dependency=None
    if backend.linear_solver=='mplapack':
        from .mplapack_build import mplapack_dependency
        dependency=mplapack_dependency(backend.mplapack_dir)
    runtime=(ROOT/'cpp/global_collocation.hpp').read_text()
    adapter=(ROOT/'cpp/global_solver.hpp').read_bytes()
    bridge=Path(__file__).read_bytes()
    if pressure is not None:bridge+=Path(__file__).with_name('global_flow_backend.py').read_bytes()
    data_identity=None
    if data_plan is not None:
        import sympy as sp
        bridge+=Path(__file__).with_name('global_data_codegen.py').read_bytes()
        data_identity=[sp.srepr(expr) for expr in data_plan.expressions]
    if dependency is not None:bridge+=Path(__file__).with_name('mplapack_build.py').read_bytes()
    index,cached=request_cache('global_collocation',families,arithmetic,backend.cache_dir,
        dimension=dimension,order=order,runtime=hashlib.sha256(runtime.encode()).hexdigest(),
        bridge=hashlib.sha256(bridge+adapter).hexdigest(),pressure_order=pressure_order,
        linear_solver=backend.linear_solver,dependency=dependency.identity if dependency else None,
        native_data=data_identity,vector_only=vector_only)
    if cached:
        _remember_compiled(key,cached,index,ROOT,dependency)
        return executable_path(cached/'kernel'),bound
    def generated(family,order,namespace):
        code,alphas=header(family,dimension,order,namespace,arithmetic,shared=True)
        dispatch=f'\ntemplate<class A> int {namespace}_index(const A&a){{\n'
        for i,alpha in enumerate(alphas):
            dispatch+='if('+ ' && '.join(f'a[{j}]=={v}' for j,v in enumerate(alpha))+f') return {i};\n'
        dispatch+='throw std::runtime_error("Unprepared derivative");}\n'
        dispatch+=f'template<class Z> Real {namespace}_prepared(int i,const Z&z,const std::vector<Real>&params){{return {namespace}::eval(i,'+','.join(f'z[{j}]' for j in range(dimension))+',params);}\n'
        dispatch+=f'template<class Z,class A> Real {namespace}_value(const Z&z,const A&a,const std::vector<Real>&params){{return {namespace}_prepared({namespace}_index(a),z,params);}}\n'
        return code+dispatch
    code=generated(bound.family,order,'radial')
    if pressure is not None:
        code=f'\n#define RBFLAB_GLOBAL_FIELDS\n#define RBFLAB_GLOBAL_PRESSURE\nconstexpr int PRESSURE_PARAMETERS={len(pressure.values)};\n'+code
        code+=generated(pressure.family,pressure_order,'pressure')
    elif vector_only:code='\n#define RBFLAB_GLOBAL_FIELDS\n#define RBFLAB_GLOBAL_VECTOR_ONLY\n'+code
    if data_plan is not None:
        from .global_data_codegen import generate_data
        code+=generate_data(data_plan,arithmetic)
    source='#define RBFLAB_GLOBAL_DENSE\n'+PRELUDE+f'\nconstexpr int DIM={dimension},PARAMETERS={len(bound.values)};\n'+code+runtime
    if dependency is not None:source='#define RBFLAB_MPLAPACK\n'+source
    directory=build(source,dict(kind='global_collocation',dimension=dimension,order=order,
                               pressure_order=pressure_order,fields=pressure is not None or vector_only,
                               native_data=data_identity,
                               vector_only=vector_only,
                               linear_solver=backend.linear_solver),arithmetic,backend.cache_dir,
                    dependency=dependency,extra_headers=('global_solver.hpp',))
    record_request(index,directory)
    _remember_compiled(key,directory,index,ROOT,dependency)
    return executable_path(directory/'kernel'),bound


class _Worker:
    def __init__(self, executable, backend):
        from .native_toolchain import native_path as _native_path, executable_path
        prefix=[]
        self.process=subprocess.Popen([*prefix,_native_path(executable)],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self.timeout=float(backend.timeout)
        self.lock=threading.RLock()
        try:self.request('HELLO')
        except Exception:
            self.close(force=True);raise

    def request(self, command):
        with self.lock:
            if self.process.poll() is not None: raise RuntimeError('C++ global worker exited')
            result=[];errors=[]
            def exchange():
                try:
                    self.process.stdin.write(command+'\n')
                    self.process.stdin.flush()
                    result.append(self.process.stdout.readline())
                except (BrokenPipeError,OSError,ValueError) as exc:errors.append(exc)
            thread=threading.Thread(target=exchange,daemon=True);thread.start();thread.join(self.timeout)
            if thread.is_alive():
                self.close(force=True)
                raise TimeoutError('C++ global worker timed out')
            if errors:
                self.close(force=True)
                raise RuntimeError('C++ global worker pipe failed') from errors[0]
            line=result[0].strip()
            if not line:
                self.close(force=True)
                raise RuntimeError('C++ global worker returned no response')
            if line.startswith('ERR '): raise RuntimeError('C++ global worker: '+line[4:])
            if not line.startswith('OK'):
                self.close(force=True)
                raise RuntimeError('Invalid C++ global worker response: '+line[:120])
            return line.split()[1:]

    def close(self,force=False):
        if self.process.poll() is None:
            if not force:
                try:self.request('QUIT')
                except Exception:pass
            if self.process.poll() is None:self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for pipe in (self.process.stdin,self.process.stdout,self.process.stderr):
            if pipe is None:continue
            try:pipe.close()
            except OSError:pass


_IDLE_WORKER_LIMIT = 4
_idle_workers = []
_pool_lock = threading.RLock()
_all_workers = weakref.WeakSet()


def _worker_key(executable, backend):
    path = Path(executable).resolve()
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, backend.distribution)


class _WorkerLease:
    """Exclusive access whose lifetime cannot outlive its assembled system."""
    def __init__(self, worker, key, reuse):
        self._worker, self._key, self._reuse = worker, key, reuse
        self._closed = False
        self._lock = threading.RLock()

    def check_open(self):
        if self._closed: raise RuntimeError('C++ global system is closed')

    def request(self, command):
        with self._lock:
            self.check_open()
            return self._worker.request(command)

    def close(self, force=False):
        with self._lock:
            if self._closed: return
            self._closed = True
            worker = self._worker
            if force or not self._reuse:
                worker.close(force=force)
                return
            try:
                worker.request('RESET')
            except Exception:
                worker.close(force=True)
                return
            with _pool_lock:
                # Evict the oldest idle worker so stale kernel configurations
                # cannot permanently prevent reuse of a newly requested one.
                evicted = _idle_workers.pop(0)[1] if len(_idle_workers) >= _IDLE_WORKER_LIMIT else None
                _idle_workers.append((self._key, worker))
            if evicted is not None: evicted.close()


def _acquire_worker(executable, backend):
    key = _worker_key(executable, backend)
    worker = None
    if backend.reuse_workers:
        with _pool_lock:
            for i in range(len(_idle_workers)-1, -1, -1):
                candidate_key, candidate = _idle_workers[i]
                if candidate_key == key:
                    _idle_workers.pop(i)
                    if candidate.process.poll() is None:
                        worker = candidate
                        break
                    candidate.close(force=True)
    if worker is None:
        worker = _Worker(executable, backend)
        with _pool_lock: _all_workers.add(worker)
    worker.timeout = float(backend.timeout)
    return _WorkerLease(worker, key, backend.reuse_workers)


def _close_workers():
    with _pool_lock:
        workers = list(_all_workers)
        _idle_workers.clear()
    for worker in workers:
        worker.close(force=True)


atexit.register(_close_workers)


class _Matrix:
    """Explicit per-entry native matrix inspection and editing."""
    def __init__(self, system):self.system=system
    @property
    def shape(self):
        self.system.worker.check_open()
        return (self.system.size,self.system.size)
    def __getitem__(self,key):
        if isinstance(key,tuple) and len(key)==2 and all(isinstance(v,(int,np.integer)) for v in key):
            i,j=key;v=self.system.worker.request(f'GET {i} {j}')[0]
            return self.system.number(v)
        raise TypeError('Slice access requires explicit system.matrix.to_numpy() or to_mpmath()')
    def __setitem__(self,key,value):
        if not (isinstance(key,tuple) and len(key)==2 and all(isinstance(v,(int,np.integer)) for v in key)):
            raise TypeError('Set native matrix entries one at a time')
        i,j=key;self.system.worker.request(f'SET {i} {j} {self.system.encode(value)}')
    def __array__(self,dtype=None,copy=None):
        out=self.to_numpy()
        if dtype is not None:out=out.astype(dtype)
        return out.copy() if copy else out
    def _export(self):
        reply=self.system.worker.request('EXPORT')
        n=self.shape[0]
        if not reply or int(reply[0])!=n or len(reply)!=1+n*n:raise RuntimeError('Invalid native matrix export')
        return reply[1:]
    def to_numpy(self):
        n=self.shape[0]
        return np.asarray([float(v) for v in self._export()]).reshape(n,n)
    def to_mpmath(self):
        if self.system.ctx is None:raise ValueError('to_mpmath requires extended precision')
        n=self.shape[0];values=self._export()
        return self.system.ctx.matrix([[self.system.number(values[i*n+j]) for j in range(n)] for i in range(n)])


class CppGlobalSystem:
    _native_global=True
    def __init__(self, worker, centers, source_operators, powers, rhs, ctx, digits, backend, prepared_order):
        self.worker,self.centers,self.source_operators,self.powers=worker,centers,source_operators,powers
        self.size=len(centers)+len(powers)
        self.rhs=rhs
        self.ctx,self.digits,self.backend=ctx,digits,backend
        self.prepared_order=prepared_order
        self.source_order=max((sum(alpha) for row in source_operators
            for op in (row.values() if isinstance(row,dict) else (row,)) for alpha,_ in op.terms),default=0)
        self._matrix=_Matrix(self)
        self._finalizer=weakref.finalize(self,worker.close)

    @property
    def matrix(self):
        self.worker.check_open()
        return self._matrix
    @matrix.setter
    def matrix(self,value):
        self.worker.check_open()
        if isinstance(value, _Matrix):
            value = value.to_mpmath() if self.ctx and value.system.ctx else value.to_numpy()
        array=np.asarray(value.tolist() if hasattr(value,'rows') and hasattr(value,'cols') else value)
        if array.shape!=(self.size,self.size):raise ValueError('Matrix dimensions changed; reassemble the method instead')
        encoded=[self.encode(array[i,j]) for i in range(self.size) for j in range(self.size)]
        self.worker.request(' '.join(['REPLACE',str(self.size),*encoded]))

    def number(self,value):
        result = mp_number(self.ctx,value) if self.ctx else float(value)
        if not (self.ctx.isfinite(result) if self.ctx else math.isfinite(result)):
            raise FloatingPointError('Nonfinite native result')
        return result
    def encode(self,value):
        if isinstance(value,Fraction):return self.encode(value.numerator/self.number(value.denominator))
        if self.ctx:
            v=mp_number(self.ctx,value)
            if not self.ctx.isfinite(v):raise ValueError('Expected finite native input')
            return self.ctx.nstr(v,self.digits+8)
        v=float(value)
        if not math.isfinite(v):raise ValueError('Expected finite native input')
        return format(v,'.17g')

    def solve(self):
        self.worker.check_open()
        data=list(self.rhs)
        if len(data)!=self.size:raise ValueError('RHS dimensions changed; reassemble the method')
        reply=self.worker.request('SOLVE 1 '+' '.join(self.encode(v) for v in data))
        if not reply or int(reply[0])!=1 or len(reply)!=self.size+2:raise RuntimeError('Invalid native solve result')
        coeffs=[self.number(v) for v in reply[1:1+self.size]]
        coeffs=self.ctx.matrix(coeffs) if self.ctx else np.asarray(coeffs,dtype=float)
        native_residual=self.number(reply[1+self.size])
        diagnostics={'relative_residual':float(native_residual),
            'unknowns':self.size,'digits':self.digits if self.ctx else None}
        if self.ctx:diagnostics['relative_residual_mp']=native_residual
        if self.backend.compute_condition:
            result=self.worker.request('DIAG 1')
            diagnostics['scaled_condition']=None if result[-1]=='none' else float(result[-1])
            if self.ctx and result[-1]!='none':diagnostics['scaled_condition_mp']=self.number(result[-1])
            diagnostics['condition_norm']='1'
            diagnostics['condition_kind']='estimate'
        diagnostics['backend']='cpp_global'
        diagnostics['arithmetic']='mpfr' if self.ctx else 'float64'
        diagnostics['linear_solver']=self.backend.linear_solver
        if hasattr(self,'data_backend'):diagnostics['data_backend']=self.data_backend
        return CppGlobalSolution(self,coeffs,diagnostics)

    def close(self):self._finalizer()
    def __enter__(self):
        self.worker.check_open()
        return self
    def __exit__(self,*_):self.close()


class CppGlobalSolution:
    def __init__(self,system,coefficients,diagnostics):
        self.system,self.coefficients,self.diagnostics=system,coefficients,diagnostics

    def _evaluate(self,points,operator):
        from .methods import _query
        from .operators import Identity, bind_operators
        s=self.system;s.worker.check_open();points=_query(points,s.centers.shape[1])
        if not len(points):return []
        operators=bind_operators([operator or Identity(points.shape[1])]*len(points),points)
        if any(getattr(op, 'dimension', None) != points.shape[1] for op in operators):
            raise ValueError('Evaluation operator and point dimensions must match')
        query_order=max((sum(alpha) for op in operators for alpha,_ in op.terms),default=0)
        if query_order+s.source_order>s.prepared_order:
            raise ValueError('Evaluation derivative exceeds compiled order; increase CppGlobalBackend(derivative_order=...)')
        tokens=['EVAL',str(len(points))]
        tokens.extend(s.encode(v) for v in points.flat)
        for op in operators:tokens.extend(_op_tokens(op,s.encode))
        tokens.extend(s.encode(v) for v in self.coefficients)
        result=s.worker.request(' '.join(tokens))
        if not result or int(result[0])!=len(points) or len(result)!=len(points)+1:raise RuntimeError('Invalid native evaluation result')
        return [s.number(v) for v in result[1:]]

    def evaluate_mp(self,points,operator=None):
        if self.system.ctx is None:raise ValueError('evaluate_mp requires extended precision')
        values=self._evaluate(points,operator)
        return self.system.ctx.matrix(values) if values else self.system.ctx.matrix(0,1)

    def evaluate(self,points,operator=None):
        return np.asarray([float(v) for v in self._evaluate(points,operator)],dtype=float)


def _op_tokens(op,encode):
    return [str(len(op.terms)),*(v for alpha,c in op.terms for v in [encode(c),*(str(i) for i in alpha)])]


def prepare_cpp_global(method,problem,dimension,cache_dir):
    """Warm the worker executable while retaining the original bound kernel."""
    from dataclasses import replace
    from .operators import NormalDerivative,Robin
    from .spaces import SpaceStokesProblem
    from .evolution import EvolutionPDE
    if isinstance(problem,SpaceStokesProblem):
        from .global_flow_backend import prepare_cpp_flow
        return prepare_cpp_flow(method,problem,dimension,cache_dir)
    backend=method.backend
    if cache_dir is not None:
        backend=replace(backend,cache_dir=cache_dir)
        method.backend=backend
    def op_order(op):
        if hasattr(op,'terms'):return max((sum(alpha) for alpha,_ in op.terms),default=0)
        if isinstance(op,(NormalDerivative,Robin)):return 1
        raise NotImplementedError('Cannot infer derivative order for this boundary operator')
    equation=max([op_order(problem.operator)]+[op_order(bc.operator) for bc in problem.boundary])
    source=equation if method.scheme=='symmetric' else 0
    order=max(equation+source,backend.derivative_order+source)
    if order>6:raise NotImplementedError('C++ global collocation supports total kernel derivative order up to 6')
    from .global_evolution_backend import scalar_data_bundle
    data_plan=scalar_data_bundle(problem,dimension,problem.initial if isinstance(problem,EvolutionPDE) else None)[0]
    method.compiled_artifact=_compiled(method.kernel,dimension,order,
        'mpfr' if method.precision.global_digits else 'float64',backend,data_plan=data_plan)[0]
    return method


def assemble_cpp_global(method,problem,cloud):
    from .spaces import SpaceStokesProblem
    if isinstance(problem,SpaceStokesProblem):
        from .global_flow_backend import assemble_cpp_flow
        return assemble_cpp_flow(method,problem,cloud)
    from .evolution import EvolutionPDE
    if isinstance(problem,EvolutionPDE):
        from .global_evolution_backend import assemble_cpp_evolution
        return assemble_cpp_evolution(method,problem,cloud)
    from .methods import _query
    from .operators import Identity, bind_operators
    from .problems import boundary_data, values
    from .nodal import dense_arithmetic, collocation_data
    from .global_scalar_engine import assemble_scalar_native
    from .global_evolution_backend import scalar_data_bundle, scalar_data_rows
    d=cloud.dimension
    if d not in (2,3):raise ValueError('C++ global collocation requires 2D or 3D')
    arithmetic=dense_arithmetic(method.kernel,method.precision)
    ctx=arithmetic.ctx
    if ctx:
        for data in [problem.rhs]+[bc.rhs for bc in problem.boundary]:
            if callable(data) and not hasattr(data,'mp_values'):
                raise TypeError('Extended native methods require PrecisionData callbacks')
    plan,program_ids,_=scalar_data_bundle(problem,d)
    node_order=np.r_[cloud.interior_indices,cloud.boundary_indices] if method.scheme=='symmetric' and method.polynomial_degree is None else np.arange(len(cloud.points))
    rows=scalar_data_rows(problem,cloud,program_ids,order=node_order)
    if rows is not None:
        from dataclasses import replace
        template=replace(problem,rhs=0,boundary=tuple(replace(bc,rhs=0) for bc in problem.boundary))
        bd=boundary_data(template,cloud)
        centers=_query(cloud.points[node_order])
        ops=[bd[int(node)][0] if int(node) in bd else problem.operator for node in node_order]
        rhs=[0]*len(centers)
    elif method.scheme=='symmetric' and method.polynomial_degree is None:
        bd=boundary_data(problem,cloud,ctx=ctx)
        node_order=np.r_[cloud.interior_indices,cloud.boundary_indices]
        centers=_query(cloud.points[node_order])
        ops=[problem.operator]*len(cloud.interior_indices)+[bd[int(i)][0] for i in cloud.boundary_indices]
        data=mp_values(problem.rhs,cloud.points[cloud.interior_indices],ctx) if ctx else values(problem.rhs,cloud.points[cloud.interior_indices])
        rhs=list(data)+[bd[int(i)][1] for i in cloud.boundary_indices]
    else:
        node_order=np.arange(len(cloud.points))
        centers=_query(cloud.points)
        ops,rhs=collocation_data(problem,cloud,arithmetic)
    ops=bind_operators(ops,centers)
    source_ops=ops if method.scheme=='symmetric' else [Identity(d)]*len(centers)
    system=assemble_scalar_native(method,centers,ops,source_ops,rhs,data_plan=plan,arithmetic=arithmetic)
    system.data_backend='python'
    try:
        if plan is not None and rows is not None:
            system.worker.request('DATA_INIT '+str(len(rows))+' '+' '.join(map(str,rows)))
            reply=system.worker.request('RHS_DATA '+system.encode(0))
            if not reply or int(reply[0])!=system.size or len(reply)!=system.size+1:
                raise RuntimeError('Invalid native stationary data result')
            system.rhs=[system.number(value) for value in reply[1:]]
            system.data_backend='cpp_symbolic'
    except Exception:
        system.close()
        raise
    return system
