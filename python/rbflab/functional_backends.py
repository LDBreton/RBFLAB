"""Shared functional approximation execution for Python, C++ and PyTorch.

Each job contains independent source/trial points and differential functionals,
three polynomial blocks, and several target functionals. Backends solve
``G.T @ weights = Q`` once per local system. Mathematical equation assembly is
intentionally absent: these routines do not distinguish LHI from RBF-FD.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import multiprocessing
import os
import subprocess
import tempfile

import numpy as np

from .lhi_backends import PythonBackend, CppBackend, _init_worker
from .torch_backend import TorchBackend
from .nodal import Arithmetic
from .operators import Derivative, Laplacian


def _kernel(owner, x, left, y, right, scale, torch=False):
    """Apply both source functionals, including signs from the second argument."""
    if torch:
        from .scalar_backends import _torch_matrix
        from .torch_backend import _torch
        evaluate = lambda ops: _torch_matrix(owner.kernel, x, ops, y, right, scale)
        zeros = lambda n,m: _torch().zeros((n,m), dtype=_torch().float64)
    else:
        evaluate = lambda ops: owner.arithmetic.matrix(x, ops, y, right, scale)
        zeros = owner.arithmetic.zeros
    if not owner.vector:
        return evaluate(left)
    d = owner.dimension
    out = zeros(len(x)*d, len(y)*d)
    for a in range(d):
        for b in range(d):
            transform = Derivative(a,d) @ Derivative(b,d)
            if a == b:
                transform = transform - Laplacian(d)
            block = evaluate([op @ transform for op in left])
            for i in range(len(x)):
                for j in range(len(y)):
                    out[i*d+a,j*d+b] = block[i,j]
    return out


def arrays(owner, job, torch=False):
    """Reconstruct the mathematical local arrays without factoring or caching."""
    a = owner.arithmetic
    n, m = job['size'], job['polynomials']
    k = len(job['target_points'])*owner.components
    if torch:
        from .torch_backend import _torch
        t = _torch()
        G = t.zeros((n+m,n+m), dtype=t.float64)
        Q = t.zeros((n+m,k), dtype=t.float64)
        convert = lambda x: t.as_tensor(x, dtype=t.float64)
    else:
        G, Q = a.zeros(n+m,n+m), a.zeros(n+m,k)
        convert = lambda x: a.ctx.matrix(x) if a.ctx else np.asarray(x, dtype=float)
    G[:n,:n] = _kernel(owner, job['source_points'], job['source_ops'],
                      job['trial_points'], job['trial_ops'], job['scale'], torch)
    Q[:n,:] = _kernel(owner, job['target_points'], job['target_ops'],
                     job['trial_points'], job['trial_ops'], job['scale'], torch).T
    if m:
        G[:n,n:] = convert(job['top'])
        G[n:,:n] = convert(job['bottom']).T
        Q[n:,:] = convert(job['target_poly']).T
    return G, Q


def _python_job(payload):
    kernel, digits, dimension, components, vector, condition, job = payload
    a = Arithmetic(kernel, digits)
    owner = SimpleNamespace(kernel=kernel, arithmetic=a, dimension=dimension,
                            components=components, vector=vector)
    job=dict(job)
    job['scale']=a.number(job['scale'])
    G,Q = arrays(owner,job)
    factor = a.factor(G, compute_condition=condition)
    W = factor.solve(Q, transpose=True)
    error = G.T*W-Q if a.ctx else G.T@W-Q
    maximum = lambda v: max((abs(x) for x in (v if a.ctx else v.flat)),default=0)
    residual = float(maximum(error)/(maximum(Q) or 1))
    encode = lambda x: a.ctx.nstr(x,a.ctx.dps+5) if a.ctx else float(x)
    rows, cols = (W.rows,W.cols) if a.ctx else W.shape
    return dict(weights=[[encode(W[i,j]) for j in range(cols)] for i in range(rows)],
                condition=factor.condition, residual=residual)


def _compile(owner):
    from .kernel_compiler import as_bound, request_cache, record_request, header, build, ROOT, PRELUDE
    bound = as_bound(owner.kernel)
    runtime = (ROOT/'cpp/functional_local.hpp').read_text(encoding='utf-8-sig')
    order = lambda op: max((sum(alpha) for alpha,c in op.terms),default=0)
    maximum = max(max(map(order,j['source_ops']+j['target_ops'])) +
                  max(map(order,j['trial_ops'])) for j in owner.jobs)
    maximum += 2 if owner.vector else 0
    d = owner.dimension
    arithmetic = 'mpfr' if owner.precision.local_digits else 'float64'
    index,cached = request_cache('functional_local',(bound.family,),arithmetic,
        owner.backend.cache_dir,dimension=d,order=maximum,
        runtime=hashlib.sha256(runtime.encode()).hexdigest(),
        bridge=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if cached:
        return cached/'kernel',bound
    code,alphas = header(bound.family,d,maximum,'radial')
    dispatch = '\nReal radial_value(const std::vector<Real>&z,const std::vector<int>&a,const std::vector<Real>&params){\n'
    for i,alpha in enumerate(alphas):
        dispatch += 'if('+ ' && '.join(f'a[{j}]=={v}' for j,v in enumerate(alpha))+') return radial::eval('+str(i)+','+','.join(f'z[{j}]' for j in range(d))+',params);\n'
    dispatch += 'throw std::runtime_error("Unprepared functional derivative");}\n'
    source = PRELUDE+f'\nconstexpr int DIM={d},PARAMETERS={len(bound.values)};\n'+code+dispatch+runtime
    directory = build(source,dict(kind='functional_local',dimension=d,order=maximum),
                      arithmetic,owner.backend.cache_dir)
    record_request(index,directory)
    return directory/'kernel',bound


def _cpp(owner, jobs, inspect=False):
    from .legacy_cpp import _linux_path
    a,b = owner.arithmetic,owner.backend
    digits = owner.precision.local_digits or 17
    executable,bound = _compile(owner)
    num = lambda v: a.ctx.nstr(a.number(v),digits+5) if a.ctx else format(float(v),'.17g')
    lines = [f'{digits} {b.threads} {int(b.compute_condition)} {int(inspect)} {len(jobs)} {owner.components}',
             ' '.join(num(v) for v in bound.values)]
    def op_line(op):
        return ' '.join([str(len(op.terms))]+[v for alpha,c in op.terms for v in [num(c),*map(str,alpha)]])
    for job in jobs:
        n,m,k = len(job['source_points']),job['polynomials'],len(job['target_points'])
        if len(job['trial_points']) != n:
            raise ValueError('Functional approximation requires equal source and trial counts')
        lines.append(f'{n} {m} {k} '+num(job['scale']))
        for name in ('source_points','trial_points'):
            lines.extend(' '.join(num(v) for v in p) for p in job[name])
        for name in ('source_ops','trial_ops'):
            lines.extend(op_line(op) for op in job[name])
        lines.extend(' '.join(num(v) for v in p) for p in job['target_points'])
        lines.extend(op_line(op) for op in job['target_ops'])
        for name,count in (('top',job['size']),('bottom',job['size']),('target_poly',k*owner.components)):
            P=job[name]
            lines.extend(' '.join(num(P[i,j]) for j in range(m)) for i in range(count))
    with tempfile.TemporaryDirectory(prefix='rbflab_functional_') as tmp:
        source,target = Path(tmp)/'input',Path(tmp)/'output'
        source.write_text('\n'.join(lines)+'\n',encoding='ascii')
        prefix = ['wsl.exe','-d',b.distribution,'--'] if os.name=='nt' else []
        run = subprocess.run([*prefix,_linux_path(executable),_linux_path(source),_linux_path(target)],
                             capture_output=True,text=True,timeout=b.timeout)
        if run.returncode:
            raise RuntimeError('C++ functional backend: '+run.stderr)
        results=[]
        for index,line in enumerate(target.read_text().splitlines()):
            data=line.split()
            jobid,n,k=map(int,data[:3])
            expected=jobs[index]['size']+jobs[index]['polynomials'] if index<len(jobs) else -1
            if jobid!=index or n!=expected or k!=len(jobs[index]['target_points'])*owner.components:
                raise RuntimeError('Invalid functional output shape')
            offset=3 if inspect else 5
            values=[a.number(v) for v in data[offset:]]
            if len(values)!=(n*n+n*k if inspect else n*k):
                raise RuntimeError('Truncated functional output')
            if not all(a.ctx.isfinite(v) if a.ctx else np.isfinite(v) for v in values):
                raise FloatingPointError('Nonfinite functional output')
            matrix=lambda v,r,c: a.ctx.matrix([v[i*c:(i+1)*c] for i in range(r)]) if a.ctx else np.array(v).reshape(r,c)
            if inspect:
                results.append((matrix(values[:n*n],n,n),matrix(values[n*n:],n,k)))
            else:
                results.append(dict(weights=matrix(values,n,k),
                    condition=None if data[3]=='none' else float(data[3]),residual=float(data[4])))
        if len(results)!=len(jobs):
            raise RuntimeError('Incomplete functional output')
        return results


def _torch(owner):
    from .torch_backend import _torch as load_torch
    t=load_torch();b=owner.backend
    old=t.get_num_threads();t.set_num_threads(b.threads)
    try:
        results=[None]*len(owner.jobs);groups={}
        for i,job in enumerate(owner.jobs):
            key=(job['size']+job['polynomials'],len(job['target_points'])*owner.components)
            groups.setdefault(key,[]).append(i)
        for ids in groups.values():
            for start in range(0,len(ids),b.batch_size):
                batch=ids[start:start+b.batch_size]
                pairs=[arrays(owner,owner.jobs[i],True) for i in batch]
                G=t.stack([x[0] for x in pairs]);Q=t.stack([x[1] for x in pairs])
                maxima=G.abs().amax(-1)
                if (maxima<=0).any():
                    raise np.linalg.LinAlgError('Zero local row')
                scale=maxima.rsqrt()
                A=scale[:,:,None]*G.transpose(-1,-2)*scale[:,None,:]
                W=t.linalg.solve(A,scale[:,:,None]*Q)*scale[:,:,None]
                if not t.isfinite(W).all():
                    raise FloatingPointError('Nonfinite functional weights')
                defect=(G.transpose(-1,-2)@W-Q).abs().amax(dim=(-1,-2))
                den=Q.abs().amax(dim=(-1,-2))
                defect=defect/t.where(den>0,den,t.ones_like(den))
                condition=t.linalg.cond(A) if b.compute_condition else None
                for j,i in enumerate(batch):
                    # Only small diagnostics are detached; weights retain tensors/autograd.
                    results[i]=dict(weights=W[j],condition=None if condition is None else float(condition[j].detach()),
                                    residual=float(defect[j].detach()))
        return results
    finally:
        t.set_num_threads(old)


def execute(owner):
    """Build and solve each local system with all requested right-hand sides."""
    b=owner.backend
    if not isinstance(b,(PythonBackend,CppBackend,TorchBackend)):
        raise TypeError('Expected PythonBackend, CppBackend or TorchBackend')
    if b.shape_rule!='fixed':
        raise NotImplementedError('Functional approximation uses fixed kernel parameters; select local scaling explicitly')
    if b.local_solver!='lu':
        raise NotImplementedError('Functional approximation currently supports LU local solves')
    if not owner.jobs:
        return []
    if isinstance(b,CppBackend):
        if b.executable is not None:
            raise ValueError('Functional approximation uses its compiled cached executable')
        if b.local_solver!='lu':
            raise NotImplementedError('Functional C++ backend currently uses Eigen LU')
        return _cpp(owner,owner.jobs)
    if isinstance(b,TorchBackend):
        if owner.precision.local_digits is not None:
            raise ValueError('PyTorch functional backend requires Float64 local precision')
        return _torch(owner)
    jobs=owner.jobs
    if owner.arithmetic.ctx:
        # mpmath contexts/matrices are not safely picklable across worker processes.
        ctx=owner.arithmetic.ctx;jobs=[]
        for original in owner.jobs:
            job=dict(original)
            job['scale']=ctx.nstr(job['scale'],ctx.dps+5)
            for name in ('top','bottom','target_poly'):
                block=job[name]
                job[name]=[[ctx.nstr(v,ctx.dps+5) for v in row] for row in block.tolist()]
            jobs.append(job)
    payloads=[(owner.kernel,owner.precision.local_digits,owner.dimension,owner.components,
               owner.vector,b.compute_condition,j) for j in jobs]
    if b.workers==1:
        result=list(map(_python_job,payloads))
    else:
        with ProcessPoolExecutor(max_workers=b.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker) as pool:
            result=list(pool.map(_python_job,payloads))
    for item in result:
        item['weights']=owner.arithmetic.ctx.matrix(item['weights']) if owner.arithmetic.ctx else np.asarray(item['weights'])
    return result


def reconstruct(owner,index):
    """Rebuild arrays using the selected arithmetic; no factorization is retained."""
    if isinstance(owner.backend,CppBackend):
        return _cpp(owner,[owner.jobs[index]],True)[0]
    return arrays(owner,owner.jobs[index],isinstance(owner.backend,TorchBackend))
