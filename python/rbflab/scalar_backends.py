"""Shared scalar local functional systems for Python, Eigen and PyTorch.

The C++ path assembles kernel derivatives and solves in the selected arithmetic.
Polynomial geometry/functionals are prepared once with the same arithmetic.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .nodal import Arithmetic,NodalBasis
from .operators import bind_operators
from .lhi_backends import PythonBackend,CppBackend,_init_worker
from .torch_backend import TorchBackend


def _python_job(payload):
    job,kernel,digits,condition,rhs=payload
    a=Arithmetic(kernel,digits)
    basis=NodalBasis(a,job['points'],job['degree'],job['right'],job['scaling'])
    n=len(job['points']);m=len(basis.powers)
    G=a.zeros(n+m,n+m);G[:n,:]=basis.evaluation(job['points'],job['left'])
    if m:G[n:,:n]=basis.polynomials(job['points'],job['moments']).T
    factor=a.factor(G,compute_condition=condition)
    q=basis.evaluation(job['target'].reshape(1,-1),[job['op']]).T if rhs is None else a.vector(rhs)
    if not a.ctx and rhs is None:q=q[:,0]
    w=factor.solve(q,transpose=rhs is None)
    defect=(G.T if rhs is None else G)*w-q if a.ctx else (G.T if rhs is None else G)@w-q
    residual=max(abs(v) for v in defect)/(max(abs(v) for v in q) or 1)
    encode=lambda x:a.ctx.nstr(x,a.ctx.dps+5) if a.ctx else float(x)
    return dict(weights=[encode(v) for v in w],condition=factor.condition,residual=float(residual))


def _cpp_compile(kernel,dimension,order,digits,cache_dir):
    from .kernel_compiler import as_bound,request_cache,record_request,header,build,ROOT,PRELUDE
    bound=as_bound(kernel);runtime=(ROOT/'cpp/scalar_local.hpp').read_text()
    arithmetic='mpfr' if digits else 'float64'
    index,cached=request_cache('scalar_local',(bound.family,),arithmetic,cache_dir,dimension=dimension,order=order,bridge=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime=hashlib.sha256(runtime.encode()).hexdigest())
    if cached:return cached/'kernel',bound
    code,alphas=header(bound.family,dimension,order,'radial')
    dispatch='\nReal radial_value(const std::vector<Real>&z,const std::vector<int>&a,const std::vector<Real>&params){\n'
    for i,alpha in enumerate(alphas):
        dispatch+='if('+ ' && '.join(f'a[{j}]=={v}' for j,v in enumerate(alpha))+') return radial::eval('+str(i)+','+','.join(f'z[{j}]' for j in range(dimension))+',params);\n'
    dispatch+='throw std::runtime_error("Unprepared scalar derivative");}\n'
    source=PRELUDE+f'\nconstexpr int DIM={dimension},PARAMETERS={len(bound.values)};\n'+code+dispatch+runtime
    directory=build(source,dict(kind='scalar_local',dimension=dimension,order=order),arithmetic,cache_dir)
    record_request(index,directory)
    return directory/'kernel',bound


def _torch_matrix(kernel,x,left,y,right,scale):
    from .torch_backend import _torch,TorchKernel
    torch=_torch();x=torch.as_tensor(x,dtype=torch.float64);y=torch.as_tensor(y,dtype=torch.float64)
    z=(x[:,None,:]-y[None,:,:])/float(scale);out=torch.zeros(z.shape[:-1],dtype=torch.float64)
    groups=[]
    for ops in (left,right):
        g={}
        for i,op in enumerate(ops):
            for alpha,c in op.terms:
                if alpha not in g:g[alpha]=torch.zeros(len(ops),dtype=torch.float64)
                g[alpha][i]=float(c)
        groups.append(g)
    evaluator=TorchKernel(kernel);derivatives={}
    for alpha,ca in groups[0].items():
        for beta,cb in groups[1].items():
            order=tuple(i+j for i,j in zip(alpha,beta))
            if order not in derivatives:derivatives[order]=evaluator.derivative(z,order)/float(scale)**sum(order)
            out+=ca[:,None]*cb[None,:]*(-1)**sum(beta)*derivatives[order]
    return out


class ScalarExecutor:
    def __init__(self,kernel,digits,backend,jobs):
        self.kernel,self.digits,self.backend,self.jobs=kernel,digits,backend or PythonBackend(),jobs
        self.arithmetic=Arithmetic(kernel,digits)
        if not isinstance(self.backend,(PythonBackend,CppBackend,TorchBackend)):
            raise TypeError('Scalar local backend must be PythonBackend, CppBackend or TorchBackend')
        if self.backend.shape_rule!='fixed':raise NotImplementedError('Scalar backends use the kernel parameter directly; use StencilPolicy(scaling="local") for dimensionless kernels')
        if isinstance(self.backend,CppBackend):
            if self.backend.executable is not None:raise ValueError('Scalar kernels use the owned cached executable')
            if self.backend.local_solver!='lu':raise NotImplementedError('Scalar C++ local solves currently use Eigen LU')
            order=lambda op:max((sum(a) for a,c in op.terms),default=0)
            maximum=max(max(map(order,j['left']+[j['op']]))+max(map(order,j['right'])) for j in jobs)
            self.executable,self.bound=_cpp_compile(kernel,jobs[0]['points'].shape[1],maximum,digits,self.backend.cache_dir)
        if isinstance(self.backend,TorchBackend) and digits is not None:
            raise ValueError('PyTorch scalar backend requires Float64 local precision')

    def solve(self,rhs=None):
        begin=time.perf_counter();backend=self.backend
        if isinstance(backend,CppBackend):result=self._cpp(rhs)
        elif isinstance(backend,TorchBackend):result=self._torch(rhs)
        else:
            data=[(job,self.kernel,self.digits,backend.compute_condition if rhs is None else False,None if rhs is None else rhs[i]) for i,job in enumerate(self.jobs)]
            if backend.workers==1:result=list(map(_python_job,data))
            else:
                import multiprocessing
                with ProcessPoolExecutor(max_workers=backend.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_init_worker) as pool:result=list(pool.map(_python_job,data))
        self.last_seconds=time.perf_counter()-begin
        return result

    def _cpp(self,rhs):
        from .legacy_cpp import _linux_path
        a=self.arithmetic;digits=self.digits or 17;b=self.backend
        num=lambda v:a.ctx.nstr(a.number(v),digits+5) if a.ctx else format(float(v),'.17g')
        lines=[f'{digits} {b.threads} {int(b.compute_condition and rhs is None)} {int(rhs is not None)} {len(self.jobs)}',' '.join(num(v) for v in self.bound.values)]
        def op_line(op):return ' '.join([str(len(op.terms))]+[v for alpha,c in op.terms for v in [num(c),*map(str,alpha)]])
        for index,j in enumerate(self.jobs):
            basis=NodalBasis(a,j['points'],j['degree'],j['right'],j['scaling']);n=len(j['points']);m=len(basis.powers)
            scale=basis.kernel_scale if j['scaling']=='local' else 1
            lines.append(f'{n} {m} '+num(scale))
            lines.extend(' '.join(num(v) for v in point) for point in j['points'])
            lines.extend(op_line(op) for op in j['left']+j['right'])
            lines.append(' '.join(num(v) for v in j['target']));lines.append(op_line(j['op']))
            for ops in (j['left'],j['moments']):
                P=basis.polynomials(j['points'],ops)
                lines.extend(' '.join(num(P[i,k]) for k in range(m)) for i in range(n))
            P=basis.polynomials(j['target'].reshape(1,-1),[j['op']]);lines.append(' '.join(num(P[0,k]) for k in range(m)))
            if rhs is not None:lines.append(' '.join(num(v) for v in rhs[index]))
        with tempfile.TemporaryDirectory(prefix='rbflab_scalar_') as tmp:
            source=Path(tmp)/'input';target=Path(tmp)/'output';source.write_text('\n'.join(lines)+'\n',encoding='ascii')
            prefix=['wsl.exe','-d',b.distribution,'--'] if os.name=='nt' else []
            run=subprocess.run([*prefix,_linux_path(self.executable),_linux_path(source),_linux_path(target)],capture_output=True,text=True,timeout=b.timeout)
            if run.returncode:raise RuntimeError('C++ scalar backend: '+run.stderr)
            results=[]
            for i,line in enumerate(target.read_text().splitlines()):
                items=line.split();n=len(self.jobs[i]['points'])+len(NodalBasis(a,self.jobs[i]['points'],self.jobs[i]['degree']).powers)
                if int(items[0])!=i or int(items[1])!=n or len(items)!=n+4:raise RuntimeError('Malformed scalar backend output')
                weights=[a.number(v) for v in items[4:]]
                if not all(a.ctx.isfinite(v) if a.ctx else np.isfinite(v) for v in weights):raise FloatingPointError('Nonfinite scalar weights')
                results.append(dict(weights=weights,condition=None if items[2]=='none' else float(items[2]),residual=float(items[3])))
            if len(results)!=len(self.jobs):raise RuntimeError('Incomplete scalar backend output')
            return results

    def _torch(self,rhs):
        from .torch_backend import _torch
        torch=_torch();old_threads=torch.get_num_threads();torch.set_num_threads(self.backend.threads)
        try:
            results=[None]*len(self.jobs);groups={}
            for i,j in enumerate(self.jobs):groups.setdefault((len(j['points']),j['degree']),[]).append(i)
            for ids in groups.values():
                for start in range(0,len(ids),self.backend.batch_size):
                    batch=ids[start:start+self.backend.batch_size];matrices=[];vectors=[]
                    for i in batch:
                        j=self.jobs[i];basis=NodalBasis(self.arithmetic,j['points'],j['degree'],j['right'],j['scaling']);n=len(j['points']);m=len(basis.powers);h=basis.kernel_scale if j['scaling']=='local' else 1
                        G=torch.zeros((n+m,n+m),dtype=torch.float64)
                        G[:n,:n]=_torch_matrix(self.kernel,j['points'],j['left'],j['points'],j['right'],h)
                        if m:
                            G[:n,n:]=torch.as_tensor(basis.polynomials(j['points'],j['left']))
                            G[n:,:n]=torch.as_tensor(basis.polynomials(j['points'],j['moments']).T)
                        if rhs is None:
                            q=torch.cat([_torch_matrix(self.kernel,j['target'].reshape(1,-1),[j['op']],j['points'],j['right'],h).flatten(),torch.as_tensor(basis.polynomials(j['target'].reshape(1,-1),[j['op']])).flatten()])
                        else:q=torch.as_tensor(rhs[i],dtype=torch.float64)
                        matrices.append(G);vectors.append(q)
                    G=torch.stack(matrices);q=torch.stack(vectors).unsqueeze(-1)
                    maxima=G.abs().amax(dim=-1)
                    if (maxima<=0).any():raise np.linalg.LinAlgError('Zero local row')
                    scale=maxima.rsqrt();A=G*scale.unsqueeze(-1)*scale.unsqueeze(-2)
                    if rhs is None:A=A.transpose(-1,-2)
                    w=torch.linalg.solve(A,q*scale.unsqueeze(-1))*scale.unsqueeze(-1)
                    error=((G.transpose(-1,-2) if rhs is None else G)@w-q).abs().amax(dim=(-1,-2));den=q.abs().amax(dim=(-1,-2));error/=torch.where(den>0,den,torch.ones_like(den))
                    cond=torch.linalg.cond(A) if self.backend.compute_condition and rhs is None else None
                    if not torch.isfinite(w).all():raise FloatingPointError('Nonfinite scalar weights')
                    for k,i in enumerate(batch):results[i]=dict(weights=w[k,:,0].numpy(),condition=None if cond is None else float(cond[k]),residual=float(error[k]))
            return results
        finally:torch.set_num_threads(old_threads)


def prepare_scalar_method(method,problem,dimension,cache_dir):
    """Warm the selected backend without replacing the user-facing kernel."""
    from dataclasses import replace
    def order(op):
        if hasattr(op,'terms'):return max((sum(alpha) for alpha,c in op.terms),default=0)
        from .operators import NormalDerivative,Robin
        if isinstance(op,(NormalDerivative,Robin)):return 1
        raise NotImplementedError('Cannot infer this scalar operator order')
    target=max([order(problem.operator)]+[order(b.operator) for b in problem.boundary])
    source=0 if method.scheme=='standard' else (target if method.scheme=='symmetric' else max([0]+[order(b.operator) for b in problem.boundary]))
    backend=method.local_backend
    if isinstance(backend,CppBackend):
        if cache_dir is not None:method.local_backend=backend=replace(backend,cache_dir=cache_dir)
        _cpp_compile(method.kernel,dimension,target+source,method.precision.local_digits,backend.cache_dir)
    elif isinstance(backend,TorchBackend):
        if method.precision.local_digits is not None:raise ValueError('PyTorch scalar backend requires Float64 local precision')
        from .torch_backend import TorchKernel
        TorchKernel(method.kernel).prepare(dimension,target+source)
    return method
