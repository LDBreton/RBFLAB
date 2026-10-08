"""Generate separate velocity/pressure contributions; all constants are exact."""
import sympy as sp
from rbflab.kernels import _base_expression,_COORDINATES,_C

def generate(folder):
 x,y=_COORDINATES[:2];c,mu=sp.symbols('c mu',real=True)
 I=[{j:[((0,0),1)]} for j in (0,1)]
 L=[{j:[((2,0),-mu),((0,2),-mu)],2:[((1,0) if j==0 else (0,1),1)]} for j in (0,1)]
 source=I+I+L;target=source+[{2:[((1,0),1)]},{2:[((0,1),1)]}]
 lines=['// Generated Cartesian derivatives, with analytic coincident limits.','#pragma once']
 for role,families in [('v',('imq','gaussian','phs7','phs9')),('p',('imq','gaussian','phs3','phs5','phs7','phs9'))]:
  for family in families:
   phi=_base_expression(family).subs(_C,c)
   K=({(i,j):sp.diff(phi,(x,y)[i],(x,y)[j])-(sp.diff(phi,x,2)+sp.diff(phi,y,2) if i==j else 0) for i in (0,1) for j in (0,1)} if role=='v' else {(2,2):phi})
   ns=role+'_'+family;lines.append('namespace '+ns+' {')
   for i,right in enumerate(source,1):
    for j,left in enumerate(target,1):
     expr=0
     for (r,s),kernel in K.items():
      for a,ca in left.get(r,[]):
       for b,cb in right.get(s,[]):expr+=ca*cb*(-1)**sum(b)*sp.diff(kernel,x,a[0]+b[0],y,a[1]+b[1])
     replacements,values=sp.cse(sp.factor(expr),symbols=sp.numbered_symbols('t'))
     lines.append(f'inline Real k{i}{j}(const Real& x,const Real& y,const Real& c,const Real& mu){{')
     if family.startswith('phs'):lines.append(' if(x==Real(0) && y==Real(0)) return Real(0);')
     lines.extend(' const Real '+str(k)+' = '+sp.ccode(v)+';' for k,v in replacements)
     lines.append(' return '+sp.ccode(values[0])+';\n}')
   lines+=['using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);','inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){',' static Fn table[6][8]={'+','.join('{'+','.join(f'k{i}{j}' for j in range(1,9))+'}' for i in range(1,7))+'};',' return table[source-1][target-1](x,y,c,mu);','}}']
 lines+=['struct KernelSpec {int family,power;Real c,weight;};',
 'inline Real contribution(bool velocity,const KernelSpec& k,int s,int t,const Real& x,const Real& y,const Real& mu){',
 ' Real value=velocity ? (k.family==0?v_imq::eval(s,t,x,y,k.c,mu):v_gaussian::eval(s,t,x,y,k.c,mu)) : (k.family==0?p_imq::eval(s,t,x,y,k.c,mu):p_gaussian::eval(s,t,x,y,k.c,mu));',
 ' if(k.power) { Real tail=velocity?(k.power==7?v_phs7::eval(s,t,x,y,k.c,mu):v_phs9::eval(s,t,x,y,k.c,mu)):(k.power==3?p_phs3::eval(s,t,x,y,k.c,mu):(k.power==5?p_phs5::eval(s,t,x,y,k.c,mu):(k.power==7?p_phs7::eval(s,t,x,y,k.c,mu):p_phs9::eval(s,t,x,y,k.c,mu)))); value+=k.weight*tail; }',
 ' return value; }']
 (folder/'generated_hybrid.hpp').write_text('\n'.join(lines)+'\n')
