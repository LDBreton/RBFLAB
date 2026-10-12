"""Persistent native dense scalar collocation worker."""
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
import hashlib
import math
import os
import subprocess
import threading
import weakref
import numpy as np

from .precision import mp_number, mp_values


@dataclass(frozen=True)
class CppGlobalBackend:
    """Compile and reuse a C++ scalar global collocation worker.

    Native assembly, factorization, solves, and evaluation stay in the selected
    arithmetic. Accessing ``system.matrix`` explicitly exports entries lazily.
    The compiler toolchain uses Ubuntu WSL on Windows.

    Args:
        threads: OpenMP threads for assembly and evaluation.
        compute_condition: Estimate the balanced matrix one-norm condition.
        cache_dir: Directory for compiled native artifacts.
        distribution: WSL distribution; currently Ubuntu.
        timeout: Maximum seconds for each worker command.
        derivative_order: Maximum evaluation derivative order to prepare, in
            addition to source functional derivatives. Total order is at most 6.
    """
    threads: int = 1
    compute_condition: bool = True
    cache_dir: object = None
    distribution: str = 'Ubuntu'
    timeout: float = 120.0
    derivative_order: int = 2

    def __post_init__(self):
        if type(self.threads) is not int or self.threads < 1: raise ValueError('threads must be positive')
        if type(self.compute_condition) is not bool: raise TypeError('compute_condition must be bool')
        if self.distribution != 'Ubuntu': raise NotImplementedError('Native compilation currently uses Ubuntu WSL')
        if not isinstance(self.timeout,(float,int)) or not math.isfinite(self.timeout) or self.timeout <= 0: raise ValueError('timeout must be positive')
        if type(self.derivative_order) is not int or not 0 <= self.derivative_order <= 6:
            raise ValueError('derivative_order must be between 0 and 6')


def _compiled(kernel, dimension, order, arithmetic, backend):
    from .kernel_compiler import as_bound, request_cache, record_request, header, build, ROOT, PRELUDE
    bound=as_bound(kernel)
    runtime=(ROOT/'cpp/global_collocation.hpp').read_text()
    bridge=Path(__file__).read_bytes()
    index,cached=request_cache('global_collocation',(bound.family,),arithmetic,backend.cache_dir,
        dimension=dimension,order=order,runtime=hashlib.sha256(runtime.encode()).hexdigest(),
        bridge=hashlib.sha256(bridge).hexdigest())
    if cached:return cached/'kernel',bound
    code,alphas=header(bound.family,dimension,order,'radial')
    dispatch='\nReal radial_value(const std::vector<Real>&z,const std::vector<int>&a,const std::vector<Real>&params){\n'
    for i,alpha in enumerate(alphas):
        dispatch+='if('+ ' && '.join(f'a[{j}]=={v}' for j,v in enumerate(alpha))+') return radial::eval('+str(i)+','+','.join(f'z[{j}]' for j in range(dimension))+',params);\n'
    dispatch+='throw std::runtime_error("Unprepared derivative");}\n'
    source=PRELUDE+f'\nconstexpr int DIM={dimension},PARAMETERS={len(bound.values)};\n'+code+dispatch+runtime
    directory=build(source,dict(kind='global_collocation',dimension=dimension,order=order),arithmetic,backend.cache_dir)
    record_request(index,directory)
    return directory/'kernel',bound


class _Worker:
    def __init__(self, executable, backend):
        from .legacy_cpp import _linux_path
        prefix=['wsl.exe','-d',backend.distribution,'--'] if os.name=='nt' else []
        self.process=subprocess.Popen([*prefix,_linux_path(executable)],stdin=subprocess.PIPE,
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


class _Matrix:
    """Explicit per-entry native matrix inspection and editing."""
    def __init__(self, system):self.system=system
    @property
    def shape(self):return (self.system.size,self.system.size)
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
        self.source_order=max((sum(alpha) for op in source_operators for alpha,_ in op.terms),default=0)
        self._matrix=_Matrix(self)
        self._finalizer=weakref.finalize(self,worker.close)

    @property
    def matrix(self):return self._matrix
    @matrix.setter
    def matrix(self,value):
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
        return CppGlobalSolution(self,coeffs,diagnostics)

    def close(self):self._finalizer()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


class CppGlobalSolution:
    def __init__(self,system,coefficients,diagnostics):
        self.system,self.coefficients,self.diagnostics=system,coefficients,diagnostics

    def _evaluate(self,points,operator):
        from .methods import _query
        from .operators import Identity, bind_operators
        s=self.system;points=_query(points,s.centers.shape[1])
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
        return self.system.ctx.matrix(self._evaluate(points,operator))

    def evaluate(self,points,operator=None):
        return np.asarray([float(v) for v in self._evaluate(points,operator)],dtype=float)


def _op_tokens(op,encode):
    return [str(len(op.terms)),*(v for alpha,c in op.terms for v in [encode(c),*(str(i) for i in alpha)])]


def prepare_cpp_global(method,problem,dimension,cache_dir):
    """Warm the worker executable while retaining the original bound kernel."""
    from dataclasses import replace
    from .operators import NormalDerivative,Robin
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
    method.compiled_artifact=_compiled(method.kernel,dimension,order,
        'mpfr' if method.precision.global_digits else 'float64',backend)[0]
    return method


def assemble_cpp_global(method,problem,cloud):
    from .methods import _query
    from .operators import Identity, bind_operators
    from .problems import boundary_data, values
    from .nodal import Arithmetic, NodalBasis, collocation_data
    from .stencils import polynomial_powers
    import mpmath
    p=method.precision;b=method.backend;d=cloud.dimension
    if d not in (2,3):raise ValueError('C++ global collocation requires 2D or 3D')
    ctx=mpmath.mp.clone() if p.global_digits else None
    if ctx:ctx.dps=p.global_digits
    digits=p.global_digits or 17
    if ctx:
        for data in [problem.rhs] + [bc.rhs for bc in problem.boundary]:
            if callable(data) and not hasattr(data, 'mp_values'):
                raise TypeError('Extended native methods require PrecisionData callbacks')
    if method.scheme=='symmetric' and method.polynomial_degree is None:
        bd=boundary_data(problem,cloud,ctx=ctx)
        ids=np.r_[cloud.interior_indices,cloud.boundary_indices]
        centers=_query(cloud.points[ids]);ops=[problem.operator]*len(cloud.interior_indices)+[bd[int(i)][0] for i in cloud.boundary_indices]
        data=mp_values(problem.rhs,cloud.points[cloud.interior_indices],ctx) if ctx else values(problem.rhs,cloud.points[cloud.interior_indices])
        rhs=list(data)+[bd[int(i)][1] for i in cloud.boundary_indices]
    else:
        centers=_query(cloud.points)
        ops,rhs=collocation_data(problem,cloud,Arithmetic(method.kernel,p.global_digits))
    ops=bind_operators(ops,centers)
    source_ops=ops if method.scheme=='symmetric' else [Identity(d)]*len(centers)
    powers=[] if method.polynomial_degree is None else polynomial_powers(d,method.polynomial_degree)
    if len(powers)>len(centers):raise ValueError('Too few centers for polynomial basis')
    if powers:
        basis=NodalBasis(Arithmetic(method.kernel,p.global_digits),centers,method.polynomial_degree,source_operators=source_ops)
        P=np.asarray(basis.polynomials(centers,source_ops).tolist() if ctx else basis.polynomials(centers,source_ops),float)
        row_scale=np.max(abs(P),axis=1)
        check=P/np.where(row_scale>0,row_scale,1)[:,None]
        column_scale=np.max(abs(check),axis=0)
        check=check/np.where(column_scale>0,column_scale,1)[None,:]
        if np.linalg.matrix_rank(check)<len(powers):
            raise np.linalg.LinAlgError('Polynomial functionals are not unisolvent')
        origin=basis.origin;scale=[basis.scale]*d
    else:origin=[0]*d;scale=[1]*d
    source_order=max((sum(alpha) for op in source_ops for alpha,_ in op.terms),default=0)
    equation_order=max((sum(alpha) for op in ops for alpha,_ in op.terms),default=0)
    order=max(b.derivative_order,equation_order)+source_order
    if order>6:raise NotImplementedError('C++ global collocation supports total kernel derivative order up to 6')
    executable,bound=_compiled(method.kernel,d,order,'mpfr' if ctx else 'float64',b)
    worker=_Worker(executable,b)
    system=CppGlobalSystem(worker,centers,source_ops,powers,list(rhs)+[0]*len(powers),ctx,digits,b,order)
    enc=system.encode
    tokens=['INIT',str(digits),str(b.threads),str(int(method.scheme=='symmetric')),str(len(centers)),str(len(powers)),enc(1)]
    tokens.extend(enc(v) for v in bound.values)
    tokens.extend(enc(v) for v in centers.flat)
    tokens.extend(enc(v) for v in centers.flat)
    tokens.extend(enc(v) for v in origin)
    tokens.extend(enc(v) for v in scale)
    tokens.extend(str(v) for power in powers for v in power)
    for op in ops:tokens.extend(_op_tokens(op,enc))
    for op in source_ops:tokens.extend(_op_tokens(op,enc))
    try:
        worker.request(' '.join(tokens))
        worker.request('BUILD')
    except Exception:
        worker.close(force=True);raise
    return system
