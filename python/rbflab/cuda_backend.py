"""Experimental CUDA Float64 local Stokes weights; global solve stays on CPU."""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import hashlib,json,os,re,subprocess,tempfile,time
import numpy as np
from .kernel_compiler import as_bound,stokes_header,ROOT
from .legacy_cpp import _linux_path
from .lhi_backends import _prepare,_finish

@lru_cache(maxsize=1)
def cuda_toolchain():
    prefix=['wsl.exe','-d','Ubuntu','--'] if os.name=='nt' else []
    nvcc=subprocess.check_output(prefix+['nvcc','--version'],text=True)
    capability=subprocess.check_output(prefix+['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True).splitlines()[0].strip().replace('.','')
    if not capability.isdigit():raise RuntimeError('Cannot determine CUDA architecture')
    return prefix,nvcc,capability


def compile_cuda(velocity,pressure,cache_dir=None):
    prefix,nvcc,arch=cuda_toolchain();v,p=as_bound(velocity),as_bound(pressure)
    dependencies=[ROOT/'cpp/cuda_lhi.cu',Path(__file__),Path(__file__).with_name('kernel_compiler.py'),Path(__file__).with_name('symbolic_kernel.py')]
    identity=dict(velocity=repr(v.family),pressure=repr(p.family),nvcc=nvcc,arch=arch,flags=['-O3','--fmad=false','-ccbin=g++-12'],sources={str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in dependencies})
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();root=Path(cache_dir) if cache_dir else Path.home()/'.cache/rbflab/cuda';folder=root/key
    manifest=folder/'manifest.json'
    if manifest.is_file():
        stored=json.loads(manifest.read_text())
        if not all((folder/n).is_file() and hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in stored['files'].items()):raise RuntimeError('CUDA cache integrity failure')
        return folder/'cuda_lhi'
    if folder.exists():raise RuntimeError('Incomplete CUDA cache; select a fresh cache directory')
    code=stokes_header(v.family,p.family)
    code=re.sub(r'Real\("([^"\n]+)"\)',lambda m:'('+m.group(1)+'.0)' if re.fullmatch(r'-?\d+',m.group(1)) else '('+m.group(1)+')',code)
    code=code.replace('std::vector<Real> params','const double* params').replace('const std::vector<Real>& params','const double* params')
    code=re.sub(r'params.at\((\d+)\)',r'params[\1]',code).replace('Real','double').replace('inline double','__device__ inline double')
    code=re.sub(r'throw std::runtime_error\("[^"\n]+"\);','return nan("");',code)
    source=(ROOT/'cpp/cuda_lhi.cu').read_text().replace('// RBFLAB_GENERATED_SPACE',code)
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-',dir=root) as tmp:
        temp=Path(tmp);(temp/'kernel.cu').write_text(source)
        path=_linux_path(temp)
        result=subprocess.run(prefix+['nvcc','-O3','-std=c++17','--fmad=false','-ccbin','g++-12','-arch=sm_'+arch,path+'/kernel.cu','-lcublas','-o',path+'/cuda_lhi'],capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError('CUDA compile failed: '+result.stderr)
        files={n:hashlib.sha256((temp/n).read_bytes()).hexdigest() for n in ('kernel.cu','cuda_lhi')}
        (temp/'manifest.json').write_text(json.dumps(dict(identity=identity,files=files),indent=2));temp.rename(folder)
    return folder/'cuda_lhi'

@dataclass(frozen=True)
class CudaLHIBackend:
    """Feasibility backend: 2D, physical, unaugmented, Float64; no condition estimate."""
    batch_size: int = 256
    shape_rule: str = 'fixed'
    cache_dir: str | None = None
    timeout: float = 300
    def __post_init__(self):
        if type(self.batch_size) is not int or not 1<=self.batch_size<=4096:raise ValueError('batch_size must be 1..4096')
        if self.shape_rule not in ('fixed','legacy_hardy'):raise ValueError('Unknown shape rule')
    def assemble(self,method,problem,cloud):
        begin=time.perf_counter()
        if method.precision.local_digits is not None or method.precision.global_dtype!='float64':raise ValueError('CUDA prototype requires Float64 local/global arithmetic')
        if cloud.dimension!=2 or method.stencil_policy.scaling!='physical' or method.polynomial_degree is not None:raise NotImplementedError('CUDA prototype requires unaugmented physical-coordinate 2D stencils')
        pk=method.pressure_kernel or method.kernel
        start=time.perf_counter();executable=compile_cuda(method.kernel,pk,self.cache_dir);compile_seconds=time.perf_counter()-start
        a,tasks,stencils=_prepare(method,problem,cloud,self.shape_rule)
        nv=len(as_bound(method.kernel).values);np_=len(as_bound(pk).values)
        num=lambda v:format(float(v),'.17g')
        lines=[f'{len(tasks)} {self.batch_size} {nv} {np_}']
        for task in tasks:
            lines.append(' '.join([str(len(task['points'])),num(task['mu']),*(num(v) for v in task['origin']),*(num(v) for v in as_bound(task['vk']).values),*(num(v) for v in as_bound(task['pk']).values)]))
            lines.extend(' '.join([str(code),*(num(v) for v in point)]) for code,point in zip(task['codes'],task['points']))
        with tempfile.TemporaryDirectory(prefix='rbflab_cuda_') as tmp:
            folder=Path(tmp);source=folder/'input';target=folder/'output';source.write_text('\n'.join(lines)+'\n')
            start=time.perf_counter();prefix=cuda_toolchain()[0]
            runtime_prefix=prefix+['env','LD_LIBRARY_PATH=/usr/lib/wsl/lib'] if os.name=='nt' else prefix
            result=subprocess.run([*runtime_prefix,_linux_path(executable),_linux_path(source),_linux_path(target)],capture_output=True,text=True,timeout=self.timeout);process=time.perf_counter()-start
            if result.returncode:raise RuntimeError('CUDA local backend: '+result.stderr)
            gpu=json.loads(result.stdout);results=[]
            for i,line in enumerate(target.read_text().splitlines()):
                items=line.split();n=len(tasks[i]['points'])
                if int(items[0])!=i or int(items[1])!=n or len(items)!=4+4*n:raise RuntimeError('Invalid CUDA weight output')
                if not all(np.isfinite(float(v)) for v in items[3:]):raise RuntimeError('Nonfinite CUDA output')
                results.append(dict(condition=None,residual=items[3],weights=[items[4+k*n:4+(k+1)*n] for k in range(4)]))
            if len(results)!=len(tasks):raise RuntimeError('Incomplete CUDA result')
        system=_finish(method,problem,cloud,a,tasks,stencils,results,dict(backend='cuda_float64',compile_seconds=compile_seconds,cuda_process_seconds=process,gpu=gpu,batch_size=self.batch_size,local_digits=None,compute_condition=False,off_node_backend='lazy Python',shape_rule=self.shape_rule))
        system.backend_diagnostics['assembly_seconds']=time.perf_counter()-begin
        return system
