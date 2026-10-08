"""Build the owned MPFR/OpenMP LHI executable (Linux or Windows with WSL)."""
from pathlib import Path
import os,subprocess,shutil,json,hashlib
import sympy as sp
from rbflab.kernels import _base_expression,_COORDINATES,_C

def main():
 root=Path(__file__).resolve().parents[1];folder=root/'cpp';x,y=_COORDINATES[:2];c,cp,mu=sp.symbols('c cp mu',real=True)
 phi=_base_expression('imq').subs(_C,c);psi=_base_expression('imq').subs(_C,cp)
 K={(i,j):sp.diff(phi,(x,y)[i],(x,y)[j])-(sp.diff(phi,x,2)+sp.diff(phi,y,2) if i==j else 0) for i in (0,1) for j in (0,1)};K[2,2]=psi
 I=[{j:[((0,0),1)]} for j in (0,1)];L=[{j:[((2,0),-mu),((0,2),-mu)],2:[((1,0) if j==0 else (0,1),1)]} for j in (0,1)]
 source=I+I+L;target=source+[{2:[((1,0),1)]},{2:[((0,1),1)]}];lines=['// Generated from rbflab Cartesian IMQ expressions; do not edit.','#pragma once']
 for i,right in enumerate(source,1):
  for j,left in enumerate(target,1):
   expr=0
   for (r,s),kernel in K.items():
    for a,ca in left.get(r,[]):
     for b,cb in right.get(s,[]):expr+=ca*cb*(-1)**sum(b)*sp.diff(kernel,x,a[0]+b[0],y,a[1]+b[1])
   replacements,values=sp.cse(sp.factor(expr),symbols=sp.numbered_symbols('t'))
   lines.append(f'inline Real k{i}{j}(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){{')
   lines.extend(' const Real '+str(k)+' = '+sp.ccode(v)+';' for k,v in replacements)
   lines.append(' return '+sp.ccode(values[0])+';\n}')
 lines+=['using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&,const Real&);','inline Real kernel(int source,int target,const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){',' static Fn table[6][8]={'+','.join('{'+','.join(f'k{i}{j}' for j in range(1,9))+'}' for i in range(1,7))+'};',' return table[source-1][target-1](x,y,c,cp,mu);','}']
 (folder/'generated_imq.hpp').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 from examples.generate_cpp_hybrid import generate
 generate(folder)
 from rbflab.eigen_build import eigen_dependency
 eigen_path,eigen_hash=eigen_dependency()
 from rbflab.legacy_cpp import _linux_path
 src=_linux_path(folder)
 prefix=['wsl.exe','-d','Ubuntu','--'] if os.name=='nt' else []
 compiler=subprocess.check_output(prefix+['g++','--version'],text=True).splitlines()[0]
 sources=['cpp/lhi_mpfr.cpp','cpp/real.hpp','cpp/eigen_solver.hpp','cpp/real_double.hpp','cpp/generated_imq.hpp','cpp/generated_hybrid.hpp','examples/generate_cpp_hybrid.py','python/rbflab/kernels.py','examples/build_cpp_backend.py','python/rbflab/eigen_build.py']
 for native in (False,True):
  name='lhi_double' if native else 'lhi_mpfr'
  flags=['-DRBFLAB_DOUBLE'] if native else ['-I'+src+'/deps/usr/include','-I'+src+'/deps/usr/include/x86_64-linux-gnu','-Wl,-l:libmpfr.so.6','-Wl,-l:libgmp.so.10']
  subprocess.run(prefix+['g++','-O3','-std=c++17','-fopenmp','-I'+src+'/deps/usr/include/eigen3',src+'/lhi_mpfr.cpp',*flags,'-o',src+'/'+name],check=True)
  versions=subprocess.check_output(prefix+[src+'/'+name,'--version'],text=True).strip()
  manifest=dict(eigen_sha256=eigen_hash,protocol=2,arithmetic='float64' if native else 'mpfr',compiler=compiler,runtime=versions,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},binary_sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest())
  (folder/('build_manifest_double.json' if native else 'build_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
  print('Built',folder/name,versions,flush=True)
if __name__=='__main__':main()
