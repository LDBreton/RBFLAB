"""Experimental bridge to the original Linux/MPFR LHI weight generator.

Only stationary 2D IMQ Dirichlet Stokes is supported. Local weights are generated
in one subprocess batch. Nodal pressure gradients use those weights; arbitrary
query points use lazy Python reconstruction with the identical fixed kernels.
"""
from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
import os, subprocess, tempfile, time, hashlib
import numpy as np
from scipy.spatial import cKDTree
from .kernels import IMQ,ScalarKernel
from .nodal import Arithmetic
from .stokes import StokesArithmetic,_blocks
from .operators import Identity,Derivative,Laplacian
from .stencils import geometry_quality


def _linux_path(path):
    path=Path(path).resolve()
    if os.name!='nt':return str(path)
    if len(path.drive)!=2:raise ValueError('WSL bridge requires a local drive path')
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]


class _LazyFactor:
    def __init__(self,a,points,ops):
        self.a,self.points,self.ops=a,points,ops;self.factor=None
        self.condition=float('nan')  # not computed by legacy LU; never label its placeholder as a condition estimate
    def solve(self,rhs,transpose=False):
        if self.factor is None:
            self.factor=self.a.factor(_blocks(self.a,self.points,self.ops,self.points,self.ops))
            self.condition=self.factor.condition
        return self.factor.solve(rhs,transpose=transpose)


@dataclass(frozen=True)
class LegacyCppLHIBackend:
    """Historical C++ LHI adapter retained for reproducibility experiments.

    Prefer `CppBackend` for the current compiled local-weight path."""
    binary_directory: str
    shape_rule: str = 'fixed'
    threads: int = 1
    distribution: str = 'Ubuntu'
    output_directory: str | None = None
    timeout: float = 1800

    def __post_init__(self):
        if self.shape_rule=='legacy_hardy':
            raise NotImplementedError('The archived Hardy executable fails weight validation; rebuild it from verified source before enabling this rule')
        if self.shape_rule!='fixed':raise ValueError('Unknown legacy shape_rule')
        if type(self.threads) is not int or self.threads<1:raise ValueError('threads must be a positive integer')

    def assemble(self,method,problem,cloud):
        from .lhi_stokes import VectorHermiteStencil,LHIStokesSystem,_add
        start=time.perf_counter()
        pk=method.pressure_kernel or method.kernel
        if cloud.dimension!=2 or any(not isinstance(k,ScalarKernel) or k.family!='imq' for k in (method.kernel,pk)):
            raise NotImplementedError('Legacy C++ bridge supports only 2D IMQ Stokes')
        if method.polynomial_degree is not None or method.min_boundary_centers or method.pde_stencil_size is not None or method.stencil_policy.scaling!='physical' or method.stencil_policy.selection!='nearest':
            raise NotImplementedError('Legacy C++ bridge needs unaugmented nearest stencils in physical coordinates')
        digits=method.precision.local_digits
        if digits is None:raise ValueError('Set local_digits explicitly for the MPFR backend')
        if method.precision.global_digits is not None:raise ValueError('LHI uses local_digits')
        if problem.pressure_value!=0:raise NotImplementedError('No pressure gauge in legacy LHI')
        a=StokesArithmetic(method.kernel,pk,digits);ctx=a.ctx
        g=a if method.precision.global_dtype=='mpmath' else Arithmetic(method.kernel)
        ii,bi=cloud.interior_indices,cloud.boundary_indices;ni,nb=len(ii),len(bi)
        if not ni or not nb or not 3<=method.stencil_size<=len(cloud.points):raise ValueError('Invalid cloud or stencil size')
        imap={int(j):i for i,j in enumerate(ii)};bmap={int(j):i for i,j in enumerate(bi)};tree=cKDTree(cloud.points)
        groups=[];radii=[];stencils=[]
        momentum=[{j:-Laplacian()*problem.viscosity,2:Derivative(j)} for j in (0,1)]
        for center in ii:
            selected=method.stencil_policy.select(tree,cloud.points[center],method.stencil_size)
            sc=[int(j) for j in selected if int(j) in imap];bc=[int(j) for j in selected if int(j) in bmap];pc=[j for j in sc if j!=center]
            if not pc:raise ValueError('Each stencil needs PDE centers')
            groups.append(([imap[j] for j in sc],[bmap[j] for j in bc],[imap[j] for j in pc]))
            # Match the archived MATLAB/knnsearch Float64 radius before conversion.
            radius=float(tree.query(cloud.points[center],k=method.stencil_size)[0][-1]);radii.append(radius)
            local=a
            points=[];ops=[];slots=[]
            for name,ids,mapping,size in [('solution',sc,imap,ni),('boundary',bc,bmap,nb),('pde',pc,imap,ni)]:
                for j in (0,1):
                    points.extend(cloud.points[ids]);ops.extend([momentum[j] if name=='pde' else {j:Identity()}]*len(ids));slots.extend([(name,mapping[k]+j*size) for k in ids])
            points=np.array(points)
            stencil=VectorHermiteStencil(int(center),points,ops,slots,_LazyFactor(local,points,ops),None,geometry_quality(cloud.points[selected],cloud.points[center],1))
            stencil.local_arithmetic=local;stencils.append(stencil)
        root=Path(self.binary_directory).resolve();exe=root/'LHI_Wegths_Save_lu.out';library=root/'LHI_Stokes_pre_divergence.so'
        if not exe.is_file() or not library.is_file():raise FileNotFoundError('Legacy IMQ executable/shared library missing')
        parent=Path(self.output_directory).resolve() if self.output_directory else None
        if parent:parent.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='rbflab_cpp_',dir=parent);folder=Path(temporary.name)
        def write(name,lines):(folder/name).write_text('\n'.join(lines)+'\n',encoding='ascii')
        dec=lambda v:format(Decimal.from_float(float(v)),'.100f')
        write('P_sc.txt',[' '.join(dec(v) for v in p) for p in cloud.points[ii]])
        write('P_fc.txt',[' '.join(dec(v) for v in p) for p in cloud.points[bi]])
        write('distancias.txt',[dec(v) for v in radii])
        for k in range(3):write(f'Indices_sup{k+1}.txt',[' '.join(map(str,row[k])) if row[k] else '-1' for row in groups])
        args=[_linux_path(exe),_linux_path(library),str(digits)]+[_linux_path(folder/name) for name in ['Indices_sup1.txt','Indices_sup2.txt','Indices_sup3.txt','distancias.txt','P_sc.txt','P_fc.txt']]+['10','6','5 6 7 8',str(ni),_linux_path(folder)+'/',str(digits),ctx.nstr(a.number(method.kernel.c),digits),ctx.nstr(a.number(pk.c),digits),ctx.nstr(a.number(problem.viscosity),digits)]
        command=(['wsl.exe','-d',self.distribution,'--','env',f'OMP_NUM_THREADS={self.threads}']+args) if os.name=='nt' else ['env',f'OMP_NUM_THREADS={self.threads}']+args
        prepared=time.perf_counter();completed=subprocess.run(command,capture_output=True,text=True,timeout=self.timeout)
        cpp_seconds=time.perf_counter()-prepared
        if completed.returncode:raise RuntimeError(f'Legacy C++ failed: {completed.stderr}\n{completed.stdout}')
        sy=[{} for _ in range(2*ni)];sb=[{} for _ in range(2*ni)];sl=[{} for _ in range(2*ni)];saved=[]
        with (folder/'pesos_file.txt').open() as stream:
            for index,line in enumerate(stream):
                if index>=4*ni:raise RuntimeError('Unexpected extra C++ weight rows')
                owner=index%ni;target=index//ni;weights=[a.number(v) for v in line.split()]
                if len(weights)!=len(stencils[owner].slots) or any(not ctx.isfinite(v) for v in weights):raise RuntimeError('Invalid C++ weight row')
                saved.append(weights)
                if target<2:
                    for (name,column),w in zip(stencils[owner].slots,weights):_add({'solution':sy,'boundary':sb,'pde':sl}[name][owner+target*ni],column,g.number(w))
        if len(saved)!=4*ni:raise RuntimeError('Missing C++ weight rows')
        mass=[{i:g.number(1)} for i in range(2*ni)]
        for i,row in enumerate(sl):
            for j,v in row.items():_add(mass[i],j,-v)
        system=LHIStokesSystem(problem,cloud,a,g,stencils,sy,sb,sl,mass)
        system.cpp_weights=saved;system.reconstruction_weights=saved;system.lazy_reconstruction=True
        system.backend_diagnostics=dict(backend='legacy_cpp',local_digits=digits,threads=self.threads,shape_rule=self.shape_rule,prepare_seconds=prepared-start,cpp_process_seconds=cpp_seconds,total_assembly_seconds=time.perf_counter()-start,local_condition='not measured',off_node_backend='lazy Python with identical fixed kernels',binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        temporary.cleanup()
        return system
