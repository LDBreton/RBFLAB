"""Exact moving-foot evaluation of the saved corrected mixed DF-LHI field.

Research-only native extension: MPI builds reusable MPFR80 LU factors once.
The persistent worker solves coefficients and contracts the actual analytic
kernel representers at every foot. All arithmetic involved in cancellation
stays MPFR80; only incoming observations and final field values are Float64.
No moving nearest-neighbour selection or polynomial surrogate is used.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import time
import numpy as np
import rbflab as r
from .nullspace_lifting import NullLifting
from .pressure_lifting import load_problem
from research.divfree_lhi_energy.public_stokes_baseline_reproduction import geometry


_NATIVE = r'''
#include <iomanip>
#include <memory>
struct StoredFactor {
 Rows lu; std::vector<Real> scale; std::vector<int> perm;
 std::vector<Real> solve(const std::vector<Real>& b) const {
  int n=lu.size();std::vector<Real> y(n); Real term;
  for(int i=0;i<n;++i)y[perm[i]]=b[i]*scale[i];
  for(int i=0;i<n;++i)for(int j=0;j<i;++j){mpfr_mul(term.v,lu[i][j].v,y[j].v,MPFR_RNDN);mpfr_sub(y[i].v,y[i].v,term.v,MPFR_RNDN);}
  for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j){mpfr_mul(term.v,lu[i][j].v,y[j].v,MPFR_RNDN);mpfr_sub(y[i].v,y[i].v,term.v,MPFR_RNDN);}y[i]/=lu[i][i];}
  for(int i=0;i<n;++i)y[i]*=scale[i];return y;
 }
};
struct Lift {int job; double x,y,h; std::vector<int> group,col; std::vector<std::vector<double>> C,E;};
Real mono(int a,int b,const Real& x,const Real& y,int dx=0,int dy=0){
 if(a<dx||b<dy)return Real(0);Real z(1);for(int j=0;j<dx;++j)z*=Real(a-j);for(int j=0;j<dy;++j)z*=Real(b-j);return z*pow(x,a-dx)*pow(y,b-dy);
}
Real velocity_poly(int a,int b,int c,const Real& x,const Real& y,int dx=0,int dy=0){
 if(c==0&&b)return Real(b)*mono(a,b-1,x,y,dx,dy);
 if(c==1&&a)return Real(-a)*mono(a-1,b,x,y,dx,dy);return Real(0);
}
std::vector<Real> qpoly(int code,const Real& x,const Real& y,const Real& h,const Real& nu){
 std::vector<Real> q;int c=code%2;
 for(int d=1;d<=7;++d)for(int a=0;a<=d;++a){int b=d-a;Real z;
  if(code<2)z=velocity_poly(a,b,c,x,y);
  else if(code<4)z=-nu/(h*h)*(velocity_poly(a,b,c,x,y,2,0)+velocity_poly(a,b,c,x,y,0,2));
  else if(code>=6&&code<8)z=velocity_poly(a,b,c,x,y,1,0)/h;
  else if(code>=8)z=velocity_poly(a,b,c,x,y,0,1)/h;q.push_back(z);
 }
 for(int d=1;d<=2;++d)for(int a=0;a<=d;++a)q.push_back(code>=2&&code<6?mono(a,d-a,x,y,1-c,c):Real(0));
 return q;
}
MixedRow outrow(int code,const Real& nu){
 MixedRow result;int c=code%2;
 auto add=[&](int f,int cp,int dx,int dy,Real v){result.push_back(MixedTerm{f,cp,{dx,dy},v});};
 if(code<2)add(0,c,0,0,Real(1));
 else if(code<4){add(0,c,2,0,-nu);add(0,c,0,2,-nu);add(1,0,1-c,c,Real(1));}
 else if(code<6)add(1,0,1-c,c,Real(1));
 else add(0,c,code<8?1:0,code<8?0:1,Real(1));return result;
}
int main(int argc,char** argv){try{
 bool serve=argc==5&&std::string(argv[1])=="serve";
 if(!serve&&argc!=3)throw std::runtime_error("Expected build input output or serve input factors lifts");
 std::ifstream in(argv[serve?2:1]);int digits,threads,condition,inspect,count;in>>digits>>threads>>condition>>inspect>>count;
 auto precision=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);Real::bits=precision;
 std::vector<int> vectors(FIELDS),components(FIELDS);std::vector<std::vector<Real>> params(FIELDS);
 for(int f=0;f<FIELDS;++f){int np;in>>vectors[f]>>components[f]>>np;params[f].resize(np);for(auto& v:params[f])v=mixed_read_real(in);}
 std::vector<MixedJob> jobs(count);
 for(auto& t:jobs){in>>t.n>>t.m>>t.k;t.h=mixed_read_real(in);t.source=Rows(t.n,std::vector<Real>(DIM));t.trial=t.source;t.target=Rows(t.k,std::vector<Real>(DIM));
  for(auto* points:{&t.source,&t.trial})for(auto& p:*points)for(auto& v:p)v=mixed_read_real(in);
  for(int j=0;j<t.n;++j)t.left.push_back(mixed_read_row(in,components));for(int j=0;j<t.n;++j)t.right.push_back(mixed_read_row(in,components));
  for(auto& p:t.target)for(auto& v:p)v=mixed_read_real(in);for(int j=0;j<t.k;++j)t.ops.push_back(mixed_read_row(in,components));
  t.top=Rows(t.n,std::vector<Real>(t.m));t.bottom=t.top;t.target_poly=Rows(t.k,std::vector<Real>(t.m));
  for(auto* block:{&t.top,&t.bottom,&t.target_poly})for(auto& row:*block)for(auto& v:row)v=mixed_read_real(in);
 }
 if(!in)throw std::runtime_error("Truncated job input");
 if(!serve){std::ofstream out(argv[2]);
  for(int id=0;id<count;++id){auto& t=jobs[id];int n=t.n+t.m;Rows G(n,std::vector<Real>(n));
   for(int i=0;i<t.n;++i){for(int j=0;j<t.n;++j)G[i][j]=mixed_entry(t.source[i],t.left[i],t.trial[j],t.right[j],t.h,vectors,params);for(int p=0;p<t.m;++p){G[i][t.n+p]=t.top[i][p];G[t.n+p][i]=t.bottom[i][p];}}
   std::vector<Real> scale(n);for(int i=0;i<n;++i){Real maximum;for(auto& v:G[i])if(maximum<abs(v))maximum=abs(v);scale[i]=Real(1)/sqrt(maximum);}
   for(int i=0;i<n;++i)for(int j=0;j<n;++j)G[i][j]*=scale[i]*scale[j];EigenLocalLU factor(G);
   out<<id<<" "<<n;for(int i=0;i<n;++i)out<<" "<<factor.factor.permutationP().indices()[i];for(auto& s:scale)out<<" "<<s.str(digits);
   for(int i=0;i<n;++i)for(int j=0;j<n;++j)out<<" "<<factor.factor.matrixLU()(i,j).str(digits);out<<"\n";
  }return 0;
 }
 std::ifstream saved(argv[3]);std::vector<StoredFactor> factors(count);
 for(int id=0;id<count;++id){int got,n;saved>>got>>n;if(got!=id)throw std::runtime_error("Wrong factor id");auto& f=factors[id];f.perm.resize(n);f.scale.resize(n);f.lu=Rows(n,std::vector<Real>(n));for(auto& p:f.perm)saved>>p;for(auto& v:f.scale)v=mixed_read_real(saved);for(auto& row:f.lu)for(auto& v:row)v=mixed_read_real(saved);}
 std::ifstream liftfile(argv[4]);int owners,ni,nb,cells;double nuvalue;liftfile>>owners>>ni>>nb>>nuvalue>>cells;Real nu(nuvalue);std::vector<Lift> lifts(owners);
 for(auto& l:lifts){int n;liftfile>>l.job>>n>>l.x>>l.y>>l.h;l.group.resize(n);l.col.resize(n);for(int i=0;i<n;++i)liftfile>>l.group[i]>>l.col[i];l.C=std::vector<std::vector<double>>(n,std::vector<double>(40));l.E=std::vector<std::vector<double>>(40,std::vector<double>(n));for(auto* block:{&l.C,&l.E})for(auto& row:*block)for(auto& v:row)liftfile>>v;}
 if(!liftfile)throw std::runtime_error("Truncated lifts");std::cout<<"READY\n"<<std::flush;
 std::string cmd;
 while(std::cin>>cmd){if(cmd=="QUIT")break;if(cmd!="EVAL")throw std::runtime_error("Bad command");int mode,nrows;std::cin>>mode>>nrows;
  std::vector<double> U(ni),g(nb),L(ni);for(auto* vec:{&U,&g,&L})for(auto& v:*vec)std::cin>>v;
  std::vector<int> rows(nrows);std::vector<std::vector<double>> feet(nrows,std::vector<double>(2));for(int i=0;i<nrows;++i)std::cin>>rows[i]>>feet[i][0]>>feet[i][1];if(!std::cin)throw std::runtime_error("Truncated request");
  std::vector<std::vector<double>> result(nrows,std::vector<double>(2));std::vector<std::string> errors(nrows);
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(int row=0;row<nrows;++row){try{Real::bits=precision;const auto& l=lifts.at(rows[row]);const auto& t=jobs[l.job];int n=t.n+t.m;
   const std::vector<double>* sources[]={&U,&g,&L};std::vector<Real> obs(t.n),e(40),rhs(n);Real term;
   for(int i=0;i<t.n;++i)obs[i]=Real(sources[l.group[i]]->at(l.col[i]));
   for(int p=0;p<40;++p)for(int i=0;i<t.n;++i){mpfr_mul_d(term.v,obs[i].v,l.E[p][i],MPFR_RNDN);e[p]+=term;}
   for(int i=0;i<t.n;++i){rhs[i]=obs[i];for(int p=0;p<40;++p){mpfr_mul_d(term.v,e[p].v,l.C[i][p],MPFR_RNDN);rhs[i]-=term;}}
   auto coefficients=factors[l.job].solve(rhs);Real zx=Real(feet[row][0])-Real(l.x),zy=Real(feet[row][1])-Real(l.y),h(l.h);
   std::vector<Real> point={zx+t.target[0][0],zy+t.target[0][1]};Real h0=Real(1)/Real(cells),x0=zx/h0,y0=zy/h0;zx*=Real(cells);zy*=Real(cells);
   for(int c=0;c<2;++c){int code=mode==0?c:mode==1?6+c:mode==2?8+c:mode==3?2+c:4+c;auto op=outrow(code,nu);Real value;
    for(int i=0;i<t.n;++i)value+=mixed_entry(point,op,t.trial[i],t.right[i],t.h,vectors,params)*coefficients[i];
    // F0 polynomial tail uses the same 35 DF velocity and two pressure coordinates as G.
    auto q0=qpoly(code,x0,y0,h0,nu);auto q=qpoly(code,zx,zy,h,nu);for(int p=0;p<t.m;++p)value+=(p<35?q0[p]:q0[p]/h0)*coefficients[t.n+p];
    for(int p=0;p<40;++p)value+=q[p]*e[p];result[row][c]=mpfr_get_d(value.v,MPFR_RNDN);
   }
  }catch(const std::exception& ex){errors[row]=ex.what();}}
  for(auto& error:errors)if(!error.empty())throw std::runtime_error(error);
  std::cout<<"RESULT "<<nrows;for(auto& row:result)for(auto v:row)std::cout<<" "<<std::setprecision(17)<<v;std::cout<<"\n"<<std::flush;
 }
 return 0;
}catch(const std::exception& ex){std::cerr<<ex.what()<<std::endl;return 1;}}
'''


def _requests(p):
    """Use the original owner geometry and explicit functional rows."""
    from rbflab.mixed_local import _fields, _columns, polynomial
    unique, records, selection = geometry(r.geometry.unit_box_grid(p.cells))
    owner = SimpleNamespace(trial=SimpleNamespace(space=p.space), precision=r.Precision(local_digits=80),
        backend=r.CppBackend(threads=8, timeout=3600, compute_condition=False),
        execution=r.LocalExecution(scheduler='mpi', ranks=8), dimension=2, jobs=[])
    owner.fields, owner.arithmetic = _fields(owner, 2)
    columns = _columns(owner.fields, 2)
    I,lap,dx,dy=r.Identity(),r.Laplacian(),r.Derivative(0),r.Derivative(1)
    observed = [r.FieldOperator({'u':[[I,0]]}),r.FieldOperator({'u':[[0,I]]}),
        r.FieldOperator({'u':[[-p.nu*lap,0]],'p':[[dx]]}),
        r.FieldOperator({'u':[[0,-p.nu*lap]],'p':[[dy]]})]
    scalar = [op.scalar_rows(owner.fields)[0] for op in observed]
    for job in unique:
        points=job['points']; rows=[scalar[code] for code in job['codes']]
        a=owner.arithmetic; radius=a.number(1)/p.cells
        P=polynomial(a,points,rows,columns,job['target'],radius)
        tx=np.repeat(job['target'][None,:],2,axis=0); targetrows=scalar[:2]
        owner.jobs.append(dict(source_points=points,trial_points=points,source_rows=rows,trial_rows=rows,
            target_points=tx,target_rows=targetrows,top=P,bottom=P,target_poly=polynomial(a,tx,targetrows,columns,job['target'],radius),
            size=len(rows),polynomials=len(columns),scale=a.number(1)))
    return owner, unique, records, selection


def _compile(owner, cache_dir):
    from rbflab.kernel_compiler import PRELUDE, ROOT, header, as_bound, build
    from rbflab.native_toolchain import executable_path
    bounds=[as_bound(f.kernel) for f in owner.fields]
    pieces=[PRELUDE,'constexpr int DIM=2,FIELDS=2;']
    alphas=[]
    for i,b in enumerate(bounds):
        code,alpha=header(b.family,2,6 if i==0 else 2,'field%d'%i,arithmetic='mpfr')
        pieces.append(code);alphas.append(alpha)
    dispatch=['Real mixed_radial(int field,const std::vector<Real>& z,const std::vector<int>& a,const std::vector<Real>& params){']
    for i,alpha in enumerate(alphas):
        dispatch.append('if(field==%d){'%i)
        for j,multi in enumerate(alpha):
            dispatch.append('if(a[0]==%d&&a[1]==%d)return field%d::eval(%d,z[0],z[1],params);'%(*multi,i,j))
        dispatch.append('}')
    dispatch.append('throw std::runtime_error("Unsupported derivative");}')
    runtime=(ROOT/'cpp/mixed_functional_local.hpp').read_text().split('int main(')[0]
    pieces+=['\n'.join(dispatch),runtime,_NATIVE]
    folder=build('\n'.join(pieces),dict(kind='corrected_characteristic_reconstruction',version=1),
                 'mpfr',cache_dir=cache_dir)
    return executable_path(folder/'kernel'),bounds


def _cache_text(path, payload):
    """Create immutable text atomically; validate existing content."""
    if path.exists():
        if path.read_text()!=payload:
            raise ValueError('Cached characteristic text failed validation: '+str(path))
        return
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(payload)
    temporary.replace(path)


def _factor_report(factorpath, reportpath):
    """A factor is committed only when its completion report is present."""
    if not factorpath.exists():
        return None
    if not reportpath.is_file():
        raise ValueError('Incomplete characteristic factor cache: missing completion report')
    return json.loads(reportpath.read_text())


def _commit_factors(factorpath, reportpath, lines, report):
    """Publish the report before the final factor completion marker."""
    if factorpath.exists():
        raise FileExistsError('Refusing to replace completed characteristic factors')
    factor_tmp=factorpath.with_name(factorpath.name+'.partial')
    report_tmp=reportpath.with_name(reportpath.name+'.partial')
    factor_tmp.write_text('\n'.join(lines)+'\n')
    report_tmp.write_text(json.dumps(report,indent=2))
    report_tmp.replace(reportpath)
    factor_tmp.replace(factorpath)


class CorrectedCharacteristicReconstruction:
    """Frozen-owner evaluation of F0+(Q-F0 C)E with reusable MPFR80 LU.

    ``model`` is the existing NullLifting and ``parameters`` its saved vector.
    A real MPI launch constructs factors only. A persistent native worker uses
    eight OpenMP threads for coefficient solves and exact kernel evaluations.
    """
    def __init__(self, model, parameters, cache_dir, ranks=8):
        from rbflab.mixed_backends import _serialize, _input, _mpi
        from rbflab.native_toolchain import compiler_environment
        self.model=model; self.parameters=np.asarray(parameters,float);self.p=model.matrices(self.parameters,full=True)
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.owner,self.unique,self.records,selection=_requests(model.p)
        self.owner.execution=r.LocalExecution(scheduler='mpi',ranks=ranks)
        executable,bounds=_compile(self.owner,self.cache_dir/'native')
        fields,chunks=_serialize(self.owner,self.owner.jobs,bounds)
        identity=hashlib.sha256((executable.parent.name+''.join(chunks)+'\n'.join(fields)).encode()).hexdigest()
        self.folder=self.cache_dir/identity;self.folder.mkdir(exist_ok=True)
        inputpath=self.folder/'jobs.in'
        payload=_input(self.owner,fields,chunks,False)
        _cache_text(inputpath,payload)
        factorpath=self.folder/'factors.txt'; reportpath=self.folder/'build.json'
        before=time.perf_counter()
        report=_factor_report(factorpath,reportpath)
        if report is None:
            mpidir=self.folder/'mpi';mpidir.mkdir(exist_ok=True)
            lines,report=_mpi(self.owner,executable,fields,chunks,mpidir,False)
            report.update(selection=selection,local_digits=80,purpose='reusable native MPFR80 LU factors; no timestepping refactorization')
            _commit_factors(factorpath,reportpath,lines,report)
        self.build_report=report
        lines=[f'{len(model.data)} {len(model.p.U0)} {2*len(model.p.boundary_points)} {model.p.nu:.17g} {model.p.cells}']
        for owner,d in enumerate(model.data):
            _,E=model.local(self.parameters,owner);rec=self.records[owner]
            x,y=model.p.points[owner];lines.append(f'{rec["job"]} {len(d["C"])} {x:.17g} {y:.17g} {1/model.p.cells:.17g}')
            groupids=np.empty(len(d['C']),int);columns=np.empty(len(d['C']),int)
            for group,(mask,cols) in enumerate(d['groups']):groupids[mask]=group;columns[mask]=cols
            lines.extend(f'{g} {c}' for g,c in zip(groupids,columns))
            lines.extend(' '.join(format(v,'.17g') for v in row) for block in (d['C'],E) for row in block)
        payload='\n'.join(lines)+'\n'
        liftpath=self.folder/('lifts_'+hashlib.sha256(payload.encode()).hexdigest()+'.txt')
        _cache_text(liftpath,payload)
        self.process=subprocess.Popen([str(executable),'serve',str(inputpath),str(factorpath),str(liftpath)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=compiler_environment())
        ready=self.process.stdout.readline().strip()
        if ready!='READY':raise RuntimeError('Native evaluator startup failed: '+ready+' '+self.process.stderr.read())
        self.evaluations=0;self.seconds=0.;self.setup_seconds=time.perf_counter()-before

    def evaluate(self,U,L,boundary,points,*,rows=None,operator='value'):
        mode={'value':0,'dx':1,'dy':2,'momentum':3,'gradient':4}[operator]
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Expected finite (N,2) feet')
        if rows is None:
            if len(points)!=len(self.records):raise ValueError('Subset queries must identify original owners')
            rows=np.arange(len(points))
        rows=np.asarray(rows,int)
        if rows.shape!=(len(points),) or np.any(rows<0) or np.any(rows>=len(self.records)):raise ValueError('Invalid owner rows')
        arrays=[np.asarray(v,float).ravel() for v in (U,boundary,L)]
        expected=(len(self.p.U0),2*len(self.p.boundary_points),len(self.p.U0))
        if tuple(map(len,arrays))!=expected or not all(np.isfinite(v).all() for v in arrays):raise ValueError('Invalid observation arrays')
        before=time.perf_counter()
        lines=[f'EVAL {mode} {len(rows)}']+[' '.join(format(v,'.17g') for v in a) for a in arrays]
        lines.extend(f'{i} {x:.17g} {y:.17g}' for i,(x,y) in zip(rows,points))
        self.process.stdin.write('\n'.join(lines)+'\n');self.process.stdin.flush()
        output=self.process.stdout.readline().split()
        if len(output)!=2+2*len(rows) or output[:2]!=['RESULT',str(len(rows))]:
            raise RuntimeError('Native field evaluation failed: '+' '.join(output)+' '+self.process.stderr.read())
        result=np.asarray(output[2:],float).reshape(-1,2)
        self.evaluations+=1;self.seconds+=time.perf_counter()-before
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite native field values')
        return result

    def close(self):
        if getattr(self,'process',None) and self.process.poll() is None:
            self.process.stdin.write('QUIT\n');self.process.stdin.flush();self.process.wait(timeout=30)
        self.process=None

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def build_evaluator(cells,source,parameters,cache_dir,ranks=8):
    model=NullLifting(load_problem(cells,Path(source)))
    parameters=np.load(parameters) if isinstance(parameters,(str,Path)) else parameters
    return CorrectedCharacteristicReconstruction(model,parameters,cache_dir,ranks=ranks)

