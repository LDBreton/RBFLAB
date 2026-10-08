"""Cached C++ Cartesian evaluators shared by scalar methods and space assemblers.

Compilation currently uses GCC on Linux or Ubuntu WSL from a source checkout.
Parameter values never participate in generated source or cache identity.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,itertools
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from .symbolic_kernel import BoundKernel,expressions,compact_parts,near_expression
from .kernels import _COORDINATES
from .legacy_cpp import _linux_path
from .eigen_build import eigen_dependency

ROOT=Path(__file__).resolve().parents[2]

class Printer(C99CodePrinter):
    def _print_Integer(self,e):return 'Real("'+str(e)+'")'
    def _print_Rational(self,e):return '(Real("'+str(e.p)+'")/Real("'+str(e.q)+'"))'
    def _print_Float(self,e):return 'Real("'+str(e)+'")'


def indices(dimension,order):
    return [a for a in itertools.product(range(order+1),repeat=dimension) if sum(a)<=order]


@lru_cache(maxsize=64)
def header(family,dimension,order,namespace):
    alpha=indices(dimension,order);printer=Printer();lines=['namespace '+namespace+' {']
    mapping={x:sp.Symbol('z'+str(i)) for i,x in enumerate(_COORDINATES[:dimension])}
    mapping.update({p:sp.Symbol('p'+str(i)) for i,p in enumerate(family.parameters)})
    args=', '.join('const Real& z'+str(i) for i in range(dimension))+', const std::vector<Real>& params'
    for i,a in enumerate(alpha):
        expr,origin=expressions(family,a);expr=expr.xreplace(mapping);origin=origin.xreplace(mapping)
        lines.append('inline Real f'+str(i)+'('+args+'){')
        lines.extend('const Real& p'+str(j)+'=params.at('+str(j)+');' for j in range(len(family.parameters)))
        _,support=compact_parts(family.expression,family.radial_variable)
        if support is not None:
            squared=sum(sp.Symbol('z'+str(j))**2 for j in range(dimension))
            lines.append('if(!('+printer.doprint(squared)+' < '+printer.doprint(support.xreplace(mapping))+')) return Real(0);')
        lines.append('if('+' && '.join('z'+str(j)+'==Real(0)' for j in range(dimension))+') return '+printer.doprint(origin)+';')
        if support is not None:
            lines.append('if('+printer.doprint(squared)+' < '+printer.doprint(support.xreplace(mapping)/16)+'){')
            near_subs,near_values=sp.cse(near_expression(family,a).xreplace(mapping),symbols=sp.numbered_symbols('near'))
            lines.extend('const Real '+str(k)+'='+printer.doprint(v)+';' for k,v in near_subs)
            lines.append('return '+printer.doprint(near_values[0])+'; }')
        subs,values=sp.cse(expr,symbols=sp.numbered_symbols('tmp'))
        lines.extend('const Real '+str(k)+'='+printer.doprint(v)+';' for k,v in subs)
        lines.append('return '+printer.doprint(values[0])+'; }')
    lines.append('inline Real eval(int i,'+args+'){switch(i){')
    for i in range(len(alpha)):lines.append('case '+str(i)+':return f'+str(i)+'('+','.join('z'+str(j) for j in range(dimension))+',params);')
    lines.append('default:throw std::runtime_error("Unsupported derivative");}} }')
    return '\n'.join(lines),alpha

@lru_cache(maxsize=8)
def toolchain():
    prefix=('wsl.exe','-d','Ubuntu','--') if os.name=='nt' else ()
    version=subprocess.check_output([*prefix,'g++','--version'],text=True).splitlines()[0]
    return prefix,version


def build(source,identity,arithmetic,cache_dir=None):
    if arithmetic not in ('float64','mpfr'):raise ValueError('arithmetic must be float64 or mpfr')
    prefix,version=toolchain();cpp=ROOT/'cpp'
    headers={name:(cpp/name).read_text() for name in ('real.hpp','real_double.hpp','eigen_solver.hpp')}
    identity=dict(identity,eigen_sha256=eigen_dependency()[1],arithmetic=arithmetic,compiler=version,source=hashlib.sha256(source.encode()).hexdigest(),headers={k:hashlib.sha256(v.encode()).hexdigest() for k,v in headers.items()},flags=['-O3','-std=c++17','-fopenmp'])
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    base=Path(cache_dir) if cache_dir else Path.home()/'.cache'/'rbflab';base=base.resolve();base.mkdir(parents=True,exist_ok=True);dest=base/key
    def valid():
        if not (dest/'manifest.json').is_file():return False
        m=json.loads((dest/'manifest.json').read_text())
        return m['identity']==identity and all((dest/n).is_file() and hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in m['files'].items())
    if valid():return dest
    if dest.exists():raise RuntimeError('Compiled cache failed integrity validation; choose a fresh cache directory')
    with tempfile.TemporaryDirectory(prefix='build-',dir=base) as tmp:
        folder=Path(tmp)
        for n,v in headers.items():(folder/n).write_text(v)
        (folder/'kernel.cpp').write_text(source)
        path=_linux_path(folder);flags=['-DRBFLAB_DOUBLE'] if arithmetic=='float64' else ['-I'+_linux_path(cpp/'deps/usr/include'),'-I'+_linux_path(cpp/'deps/usr/include/x86_64-linux-gnu'),'-Wl,-l:libmpfr.so.6','-Wl,-l:libgmp.so.10']
        result=subprocess.run([*prefix,'g++','-O3','-std=c++17','-fopenmp','-I'+_linux_path(cpp/'deps/usr/include/eigen3'),path+'/kernel.cpp',*flags,'-o',path+'/kernel'],capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError('Kernel compilation failed: '+result.stderr)
        files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('kernel.cpp','real.hpp','real_double.hpp','eigen_solver.hpp','kernel')}
        (folder/'manifest.json').write_text(json.dumps(dict(identity=identity,files=files),indent=2))
        try:folder.rename(dest)
        except OSError:
            if not valid():raise
    return dest

PRELUDE='#include "real.hpp"\n#include <vector>\n#include <fstream>\n#include <iostream>\n#include <cmath>\n'
SET_PRECISION='\n#ifndef RBFLAB_DOUBLE\nReal::bits=static_cast<mpfr_prec_t>(std::ceil(digits*std::log2(10.0))+8);\n#endif\n'

@dataclass(frozen=True)
class CompiledKernel:
    """Prepared native evaluator for a symbolic radial-kernel family.

    The compiled family records dimension, derivative multi-indices,
    arithmetic, and cache directory. Numeric parameter values are bound
    later through __call__, so a supported family can reuse compiled
    derivative code at different parameter values.
    """
    family: object
    dimension: int
    alpha: tuple
    arithmetic: str
    directory: Path
    def __call__(self,**parameters):
        bound=self.family(**parameters)
        return CompiledBoundKernel(bound.family,bound.values,self)
    def evaluate(self,points,values,alpha,digits=100):
        if len(values)!=len(self.family.parameters):raise ValueError('Kernel parameter count mismatch')
        self.family(**{p.name:v for p,v in zip(self.family.parameters,values)})
        points=np.asarray(points,dtype=object if self.arithmetic=='mpfr' else float)
        if points.ndim<1 or points.shape[-1]!=self.dimension:raise ValueError('Compiled kernel dimension mismatch')
        if tuple(alpha) not in self.alpha:raise ValueError('Derivative was not prepared')
        if type(digits) is not int or digits<16:raise ValueError('digits must be >=16')
        def number(v):
            e=sp.sympify(v,rational=True)
            if e.free_symbols or e.is_real is not True or e.is_finite is not True:raise ValueError('Expected finite real input')
            return str(e.evalf(digits+5))
        flat=points.reshape(-1,self.dimension)
        text=f'{digits} {len(flat)} {self.alpha.index(tuple(alpha))}\n'+' '.join(map(number,values))+'\n'+'\n'.join(' '.join(map(number,row)) for row in flat)
        with tempfile.TemporaryDirectory(prefix='rbflab_eval_') as tmp:
            source=Path(tmp)/'input';source.write_text(text)
            result=subprocess.run([*toolchain()[0],_linux_path(self.directory/'kernel'),_linux_path(source)],capture_output=True,text=True,check=True,timeout=120)
        data=result.stdout.split()
        if len(data)!=len(flat):raise RuntimeError('Incomplete compiled evaluation')
        if self.arithmetic=='float64':
            values=np.asarray(data,dtype=float)
            if not np.isfinite(values).all():raise ValueError('Nonfinite compiled kernel evaluation')
            return values.reshape(points.shape[:-1])
        import mpmath
        ctx=mpmath.mp.clone();ctx.dps=digits
        values=[ctx.mpf(v) for v in data]
        if not all(ctx.isfinite(v) for v in values):raise ValueError('Nonfinite compiled kernel evaluation')
        return np.asarray(values,dtype=object).reshape(points.shape[:-1])

@dataclass(frozen=True)
class CompiledBoundKernel(BoundKernel):
    compiled: CompiledKernel
    def derivative(self,displacement,alpha=None):
        z=np.asarray(displacement);alpha=(0,)*z.shape[-1] if alpha is None else tuple(alpha)
        return np.asarray(self.compiled.evaluate(z,self.values,alpha),dtype=float)


def compile_kernel(family,dimension=2,derivative_order=6,arithmetic='float64',cache_dir=None):
    if dimension not in (2,3) or type(derivative_order) is not int or not 0<=derivative_order<=6:raise ValueError('Expected dimension 2/3 and derivative order 0..6')
    index,cached=request_cache('scalar',(family,),arithmetic,cache_dir,dimension=dimension,order=derivative_order)
    if cached:return CompiledKernel(family,dimension,tuple(indices(dimension,derivative_order)),arithmetic,cached)
    code,alpha=header(family,dimension,derivative_order,'radial')
    source=PRELUDE+code+"\nint main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error(\"Expected input file\");std::ifstream in(argv[1]);int digits,n,index;in>>digits>>n>>index;"+SET_PRECISION
    source+='std::vector<Real> params('+str(len(family.parameters))+');std::string s;for(auto&v:params){in>>s;v=Real(s);}for(int i=0;i<n;i++){std::vector<Real> z('+str(dimension)+');for(auto&v:z){in>>s;v=Real(s);}if(!in)throw std::runtime_error("Invalid input");std::cout<<radial::eval(index,'+','.join('z['+str(j)+']' for j in range(dimension))+',params).str(digits)<<"\\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what();return 1;}}'
    # Keep a literal C++ newline escape, not a double-escaped output string.
    identity=dict(kind='scalar',expression=sp.srepr(family.expression),parameters=[sp.srepr(p) for p in family.parameters],dimension=dimension,derivative_order=derivative_order)
    directory=build(source,identity,arithmetic,cache_dir);record_request(index,directory)
    return CompiledKernel(family,dimension,tuple(alpha),arithmetic,directory)


def as_bound(kernel):
    from .symbolic_kernel import Kernel
    from .kernels import ScalarKernel,Hybrid,PHS
    if isinstance(kernel,BoundKernel):return kernel
    s=sp.Symbol('s',nonnegative=True);c=sp.Symbol('c',positive=True);g=sp.Symbol('g',real=True)
    if isinstance(kernel,Hybrid):
        smooth=as_bound(kernel.smooth)
        expr=smooth.family.expression.xreplace({smooth.family.radial_variable:s,smooth.family.parameters[0]:c})
        family=Kernel(expr+g*(-1)**((kernel.phs.power+1)//2)*s**sp.Rational(kernel.phs.power,2),s,(c,g),minimum_degree=kernel.minimum_degree)
        return family(c=kernel.smooth.c,g=kernel.weight)
    if isinstance(kernel,PHS):return Kernel((-1)**((kernel.power+1)//2)*s**sp.Rational(kernel.power,2),s,minimum_degree=kernel.minimum_degree)()
    if isinstance(kernel,ScalarKernel):return Kernel((1+c*s)**sp.Rational(-1,2) if kernel.family=='imq' else sp.exp(-c*s),s,(c,))(c=kernel.c)
    raise TypeError('Unsupported kernel family')


@lru_cache(maxsize=32)
def stokes_header(velocity,pressure):
    vcode,va=header(velocity,2,6,'velocity');pcode,pa=header(pressure,2,2,'pressure')
    mu=sp.Symbol('mu');I=[{j:[((0,0),sp.Integer(1))]} for j in (0,1)]
    L=[{j:[((2,0),-mu),((0,2),-mu)],2:[((1,0) if j==0 else (0,1),sp.Integer(1))]} for j in (0,1)]
    source=I+I+L;target=source+[{2:[((1,0),sp.Integer(1))]},{2:[((0,1),sp.Integer(1))]}]
    code=vcode+'\n'+pcode+'\nstruct KernelSpec {std::vector<Real> params;};\n'
    printer=Printer()
    code+='inline Real custom_eval(int source,int target,const Real& x,const Real& y,const Real& mu,const KernelSpec& v,const KernelSpec& p){switch(source*10+target){\n'
    for i,right in enumerate(source,1):
        for j,left in enumerate(target,1):
            terms=[]
            for a in range(3):
                for b in range(3):
                    if a==2 and b==2:base=[((0,0),1)]
                    elif a<2 and b<2:
                        derivative=[0,0];derivative[a]+=1;derivative[b]+=1;base=[(tuple(derivative),1)]
                        if a==b:base += [((2,0),-1),((0,2),-1)]
                    else:continue
                    for l,lc in left.get(a,[]):
                        for r,rc in right.get(b,[]):
                            for d,dc in base:
                                alpha=tuple(x+y+z for x,y,z in zip(l,r,d));role='pressure' if a==2 else 'velocity';indices_=pa if a==2 else va;params='p' if a==2 else 'v'
                                coefficient=lc*rc*dc*(-1)**sum(r)
                                terms.append('('+printer.doprint(coefficient)+')*'+role+'::eval('+str(indices_.index(alpha))+',x,y,'+params+'.params)')
            code+='case '+str(i*10+j)+':return '+('+'.join(terms) or 'Real(0)')+';\n'
    code+='default:throw std::runtime_error("Unsupported Stokes functional");}}\n'
    return code


def compile_stokes(velocity,pressure,arithmetic,cache_dir=None):
    """Space transform + two-sided functionals; scalar derivative compiler is shared."""
    velocity,pressure=as_bound(velocity),as_bound(pressure)
    index,cached=request_cache('stokes_lhi',(velocity.family,pressure.family),arithmetic,cache_dir)
    if cached:return cached/'kernel'
    code=stokes_header(velocity.family,pressure.family)
    runtime=(ROOT/'cpp/lhi_mpfr.cpp').read_text()
    runtime=runtime.replace('#include "generated_imq.hpp"','').replace('#include "generated_hybrid.hpp"','')
    runtime=runtime.replace('using Matrix=',code+'\nusing Matrix=',1)
    first=runtime.index(' if(t.v.family==');last=runtime.index('\n}',first)
    runtime=runtime[:first]+' return custom_eval(s,op,x,y,t.mu,t.v,t.p);'+runtime[last:]
    first=runtime.index('  for(auto*k:{&t.v,&t.p})');last=runtime.index('  t.mu=read(in);',first)
    runtime=runtime[:first]+'  for(auto*k:{&t.v,&t.p}){int count;if(!(in>>count)||count<0)throw std::runtime_error("Invalid parameter count");k->params.resize(count);for(auto&v:k->params)v=read(in);}\n'+runtime[last:]
    ident=dict(kind='stokes_lhi',velocity=sp.srepr(velocity.family.expression),pressure=sp.srepr(pressure.family.expression))
    directory=build(runtime,ident,arithmetic,cache_dir);record_request(index,directory)
    return directory/'kernel'


def prepare_method(method,problem,dimension=2,cache_dir=None):
    from .spaces import SpaceStokesProblem
    from .space_stokes import select_spaces
    from .lhi_backends import CppBackend
    from .cuda_backend import CudaLHIBackend,compile_cuda
    from .torch_backend import TorchBackend
    from dataclasses import replace
    if getattr(method,'spaces',None) is not None and isinstance(problem,SpaceStokesProblem):
        if problem.dimension!=dimension:raise ValueError('Problem dimension mismatch')
        velocity,pressure,_,_=select_spaces(method,problem)
        if isinstance(getattr(method,'local_backend',None),TorchBackend):
            method.local_backend.prepare(velocity.kernel,pressure.kernel,method.precision,dimension)
            return method
        if isinstance(getattr(method,'local_backend',None),CudaLHIBackend):
            if dimension!=2 or method.precision.local_digits is not None or method.precision.global_dtype!='float64':
                raise ValueError('CUDA preparation requires 2D Float64 Stokes')
            cache=cache_dir if cache_dir is not None else method.local_backend.cache_dir
            method.local_backend=replace(method.local_backend,cache_dir=cache)
            method.compiled_artifact=compile_cuda(velocity.kernel,pressure.kernel,cache)
            return method
        if isinstance(getattr(method,'local_backend',None),CppBackend):
            if dimension!=2:raise NotImplementedError('C++ Stokes LHI assembly currently supports 2D')
            method.local_backend=replace(method.local_backend,cache_dir=cache_dir)
            method.compiled_artifact=compile_stokes(velocity.kernel,pressure.kernel,'float64' if method.precision.local_digits is None else 'mpfr',cache_dir)
            return method
        spaces={}
        for field,space in method.spaces.items():
            from .spaces import DivergenceFreeSpace
            order=6 if isinstance(space,DivergenceFreeSpace) else 2
            bound=as_bound(space.kernel);artifact=bound.family.compile(dimension=dimension,derivative_order=order,cache_dir=cache_dir)
            spaces[field]=replace(space,kernel=artifact(**{p.name:v for p,v in zip(bound.family.parameters,bound.values)}))
        method.spaces=spaces
        return method
    # Request only the derivatives used by this scalar discretization.
    if method.kernel is None:raise ValueError('Expected a scalar kernel or Stokes spaces')
    from .operators import NormalDerivative,Robin
    from .methods import LHI,GlobalCollocation
    def operator_order(operator):
        if hasattr(operator,'terms'):return max((sum(alpha) for alpha,_ in operator.terms),default=0)
        if isinstance(operator,(NormalDerivative,Robin)):return 1
        raise NotImplementedError('Cannot infer derivative order for this boundary operator; compile the kernel explicitly')
    order=max([operator_order(problem.operator)]+[operator_order(bc.operator) for bc in problem.boundary])
    if isinstance(method,LHI) or isinstance(method,GlobalCollocation) and method.scheme=='symmetric':order*=2
    bound=as_bound(method.kernel);artifact=bound.family.compile(dimension=dimension,derivative_order=order,cache_dir=cache_dir)
    method.kernel=artifact(**{p.name:v for p,v in zip(bound.family.parameters,bound.values)})
    return method


def request_cache(kind,families,arithmetic,cache_dir,**settings):
    """Check before any symbolic differentiation or C++ generation."""
    dependencies=[Path(__file__),Path(__file__).with_name('symbolic_kernel.py'),ROOT/'cpp/real.hpp',ROOT/'cpp/real_double.hpp',ROOT/'cpp/lhi_mpfr.cpp',ROOT/'cpp/eigen_solver.hpp']
    request=dict(eigen_sha256=eigen_dependency()[1],kind=kind,families=[dict(expression=sp.srepr(f.expression),radial=sp.srepr(f.radial_variable),parameters=[sp.srepr(p) for p in f.parameters],dimension=f.dimension) for f in families],arithmetic=arithmetic,compiler=toolchain()[1],settings=settings,dependencies=[hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies])
    base=Path(cache_dir) if cache_dir else Path.home()/'.cache'/'rbflab';base=base.resolve();base.mkdir(parents=True,exist_ok=True)
    index=base/('request-'+hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()+'.json')
    if index.exists():
        name=json.loads(index.read_text())['artifact']
        if len(name)!=64 or any(c not in '0123456789abcdef' for c in name):raise RuntimeError('Invalid cache index')
        directory=base/name;m=json.loads((directory/'manifest.json').read_text())
        if not all((directory/n).is_file() and hashlib.sha256((directory/n).read_bytes()).hexdigest()==digest for n,digest in m['files'].items()):raise RuntimeError('Compiled cache failed integrity validation')
        return index,directory
    return index,None


def record_request(index,directory):
    with tempfile.NamedTemporaryFile(mode='w',dir=index.parent,delete=False,suffix='.json') as stream:
        json.dump(dict(artifact=directory.name),stream);temporary=Path(stream.name)
    os.replace(temporary,index)
