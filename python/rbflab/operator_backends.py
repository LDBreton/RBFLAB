"""Multi-RHS local solves. No local systems are retained after construction."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing, os, subprocess, tempfile, hashlib
import numpy as np
from .discrete_operators import _Basis
from .lhi_backends import PythonBackend, CppBackend, _init_worker
from .torch_backend import TorchBackend


def _basis(o,j):return _Basis(o.kernel,o.precision.local_digits,j['points'],o.degree,o.vector,o.scaling)


def _python(payload):
    kernel,digits,degree,vector,scaling,condition,j=payload
    basis=_Basis(kernel,digits,j['points'],degree,vector,scaling);a=basis.a
    G,Q=basis.arrays(j['target'],j['ops']);factor=a.factor(G,compute_condition=condition)
    W=factor.solve(Q,transpose=True)
    defect=G.T*W-Q if a.ctx else G.T@W-Q
    maximum=lambda x:max(abs(v) for v in (x if a.ctx else x.flat))
    # MP matrix contexts are reconstructed by the caller after process transport.
    encode=lambda x:a.ctx.nstr(x,a.ctx.dps+5) if a.ctx else float(x)
    rows=W.rows if a.ctx else len(W);cols=W.cols if a.ctx else W.shape[1]
    return dict(weights=[[encode(W[i,k]) for k in range(cols)] for i in range(rows)],condition=factor.condition,residual=float(maximum(defect)/(maximum(Q) or 1)))


def _compile(o):
    from .kernel_compiler import as_bound,request_cache,record_request,header,build,ROOT,PRELUDE
    bound=as_bound(o.kernel);runtime=(ROOT/'cpp/operators_local.hpp').read_text()
    d=o.source.shape[1];order=max(sum(alpha) for j in o.jobs for op in j['ops'] for alpha,c in op.terms) if any(op.terms for j in o.jobs for op in j['ops']) else 0
    order+=2 if o.vector else 0
    arithmetic='mpfr' if o.precision.local_digits else 'float64'
    index,cached=request_cache('operators_local',(bound.family,),arithmetic,o.backend.cache_dir,dimension=d,order=order,runtime=hashlib.sha256(runtime.encode()).hexdigest(),bridge=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if cached:return cached/'kernel',bound
    code,alphas=header(bound.family,d,order,'radial')
    dispatch='\nReal radial_value(const std::vector<Real>&z,const std::vector<int>&a,const std::vector<Real>&params){\n'
    for i,alpha in enumerate(alphas):
        dispatch+='if('+ ' && '.join(f'a[{j}]=={v}' for j,v in enumerate(alpha))+') return radial::eval('+str(i)+','+','.join(f'z[{j}]' for j in range(d))+',params);\n'
    dispatch+='throw std::runtime_error("Unprepared derivative");}\n'
    source=PRELUDE+f'\nconstexpr int DIM={d},PARAMETERS={len(bound.values)};\n'+code+dispatch+runtime
    directory=build(source,dict(kind='operators_local',dimension=d,order=order),arithmetic,o.backend.cache_dir)
    record_request(index,directory);return directory/'kernel',bound


def _cpp(o,jobs,inspect=False):
    from .legacy_cpp import _linux_path
    from .operators import Identity
    a=o.arithmetic;b=o.backend;digits=o.precision.local_digits or 17
    exe,bound=_compile(o)
    num=lambda v:a.ctx.nstr(a.number(v),digits+5) if a.ctx else format(float(v),'.17g')
    lines=[f'{digits} {b.threads} {int(b.compute_condition)} {int(inspect)} {len(jobs)} {o.components}',' '.join(num(v) for v in bound.values)]
    def op_line(op):return ' '.join([str(len(op.terms))]+[v for alpha,c in op.terms for v in [num(c),*map(str,alpha)]])
    for j in jobs:
        basis=_basis(o,j);P=basis.polynomial(j['points'],[Identity(basis.d)]*len(j['points']))
        # Validate polynomial rank even though kernels are assembled natively.
        check=np.asarray(P.tolist() if a.ctx else P,float)
        if basis.m and np.linalg.matrix_rank(check)<basis.m:raise np.linalg.LinAlgError('Polynomial space is not unisolvent')
        T=basis.polynomial(np.repeat(j['target'][None,:],len(j['ops']),axis=0),j['ops'])
        lines.append(f'{len(j["points"])} {basis.m} {len(j["ops"])} '+num(basis.h))
        lines.extend(' '.join(num(v) for v in point) for point in j['points'])
        lines.append(' '.join(num(v) for v in j['target']))
        lines.extend(op_line(op) for op in j['ops'])
        lines.extend(' '.join(num(P[i,k]) for k in range(basis.m)) for i in range(basis.n))
        lines.extend(' '.join(num(T[i,k]) for k in range(basis.m)) for i in range(len(j['ops'])*o.components))
    with tempfile.TemporaryDirectory(prefix='rbflab_operators_') as tmp:
        src=Path(tmp)/'input';dst=Path(tmp)/'output';src.write_text('\n'.join(lines)+'\n',encoding='ascii')
        prefix=['wsl.exe','-d',b.distribution,'--'] if os.name=='nt' else []
        run=subprocess.run([*prefix,_linux_path(exe),_linux_path(src),_linux_path(dst)],capture_output=True,text=True,timeout=b.timeout)
        if run.returncode:raise RuntimeError('C++ operators: '+run.stderr)
        result=[]
        for index,line in enumerate(dst.read_text().splitlines()):
            data=line.split();jobid,n,k=map(int,data[:3]);basis=_basis(o,jobs[index])
            if jobid!=index or n!=basis.n+basis.m or k!=len(jobs[index]['ops'])*o.components:raise RuntimeError('Invalid operators output shape')
            offset=3 if inspect else 5;values=[a.number(v) for v in data[offset:]]
            if len(values)!=(n*n+n*k if inspect else n*k):raise RuntimeError('Truncated operators output')
            if not all(a.ctx.isfinite(v) if a.ctx else np.isfinite(v) for v in values):raise FloatingPointError('Nonfinite operators output')
            matrix=lambda v,r,c:a.ctx.matrix([v[i*c:(i+1)*c] for i in range(r)]) if a.ctx else np.array(v).reshape(r,c)
            if inspect:result.append((matrix(values[:n*n],n,n),matrix(values[n*n:],n,k)))
            else:result.append(dict(weights=matrix(values,n,k),condition=None if data[3]=='none' else float(data[3]),residual=float(data[4])))
        if len(result)!=len(jobs):raise RuntimeError('Incomplete operators output')
        return result


def execute(o):
    b=o.backend
    if not isinstance(b,(PythonBackend,CppBackend,TorchBackend)):raise TypeError('Unsupported local backend')
    if b.shape_rule!='fixed':raise NotImplementedError('Operator builder uses fixed parameters; choose local stencil scaling for dimensionless kernels')
    if isinstance(b,CppBackend):
        if b.executable is not None:raise ValueError('Operator builder uses its cached native executable')
        if b.local_solver!='lu':raise NotImplementedError('Operator builder currently uses Eigen LU')
        return _cpp(o,o.jobs)
    if isinstance(b,PythonBackend):
        data=[(o.kernel,o.precision.local_digits,o.degree,o.vector,o.scaling,b.compute_condition,j) for j in o.jobs]
        if b.workers==1:result=list(map(_python,data))
        else:
            with ProcessPoolExecutor(max_workers=b.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker) as pool:result=list(pool.map(_python,data))
        for r in result:r['weights']=o.arithmetic.ctx.matrix(r['weights']) if o.arithmetic.ctx else np.array(r['weights'])
        return result
    if o.precision.local_digits is not None:raise ValueError('PyTorch requires Float64 local precision')
    from .torch_backend import _torch
    t=_torch();old=t.get_num_threads();t.set_num_threads(b.threads)
    try:
        result=[None]*len(o.jobs);groups={}
        for i,j in enumerate(o.jobs):groups.setdefault(len(j['points']),[]).append(i)
        for ids in groups.values():
            for start in range(0,len(ids),b.batch_size):
                batch=ids[start:start+b.batch_size]
                arrays=[_basis(o,o.jobs[i]).arrays(o.jobs[i]['target'],o.jobs[i]['ops'],True) for i in batch]
                G=t.stack([x[0] for x in arrays]);Q=t.stack([x[1] for x in arrays])
                maxima=G.abs().amax(-1)
                if (maxima<=0).any():raise np.linalg.LinAlgError('Zero local row')
                scale=maxima.rsqrt();A=scale[:,:,None]*G*scale[:,None,:]
                W=t.linalg.solve(A.transpose(-1,-2),scale[:,:,None]*Q)*scale[:,:,None]
                defect=(G.transpose(-1,-2)@W-Q).abs().amax(dim=(-1,-2));den=Q.abs().amax(dim=(-1,-2))
                defect/=t.where(den>0,den,t.ones_like(den))
                cond=t.linalg.cond(A) if b.compute_condition else None
                if not t.isfinite(W).all():raise FloatingPointError('Nonfinite weights')
                for k,i in enumerate(batch):result[i]=dict(weights=W[k].numpy(),condition=None if cond is None else float(cond[k]),residual=float(defect[k]))
        return result
    finally:t.set_num_threads(old)


def reconstruct(o,i):
    if isinstance(o.backend,CppBackend):return _cpp(o,[o.jobs[i]],True)[0]
    arrays=_basis(o,o.jobs[i]).arrays(o.jobs[i]['target'],o.jobs[i]['ops'],isinstance(o.backend,TorchBackend))
    if isinstance(o.backend,TorchBackend):return tuple(x.numpy() for x in arrays)
    if o.arithmetic.ctx:return tuple(o.arithmetic.ctx.matrix(x.tolist()) for x in arrays)
    return arrays
