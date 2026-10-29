// Generic direct-sum functional Hermite runtime. Radial evaluators precede this file.
#include "eigen_solver.hpp"
#include <sstream>
#include <omp.h>
using Rows=EigenLocalLU::Rows;
struct MixedTerm { int field,component; std::vector<int> alpha; Real coefficient; };
using MixedRow=std::vector<MixedTerm>;
struct MixedJob {
 int n,m,k; Real h;
 Rows source,trial,target,top,bottom,target_poly;
 std::vector<MixedRow> left,right,ops;
};
Real mixed_read_real(std::istream& in){std::string s;if(!(in>>s))throw std::runtime_error("Truncated numeric input");return Real(s);}
MixedRow mixed_read_row(std::istream& in,const std::vector<int>& components){
 int count;in>>count;if(!in||count<0||count>100000)throw std::runtime_error("Invalid mixed functional row");
 MixedRow row(count);for(auto& term:row){
  in>>term.field>>term.component;
  if(!in||term.field<0||term.field>=FIELDS||term.component<0||term.component>=components[term.field])throw std::runtime_error("Invalid field/component");
  term.coefficient=mixed_read_real(in);term.alpha.resize(DIM);
  for(auto& a:term.alpha){in>>a;if(!in||a<0||a>6)throw std::runtime_error("Invalid derivative");}
 }return row;
}
Real mixed_entry(const std::vector<Real>& x,const MixedRow& left,const std::vector<Real>& y,const MixedRow& right,const Real& h,const std::vector<int>& vectors,const std::vector<std::vector<Real>>& params){
 std::vector<Real> z(DIM);for(int j=0;j<DIM;++j)z[j]=(x[j]-y[j])/h;
 Real result;
 for(const auto& l:left)for(const auto& r:right){
  if(l.field!=r.field)continue;
  std::vector<int> alpha(DIM);int order=0,sign=0;
  for(int j=0;j<DIM;++j){alpha[j]=l.alpha[j]+r.alpha[j];order+=alpha[j];sign+=r.alpha[j];}
  Real value;
  if(!vectors[l.field])value=mixed_radial(l.field,z,alpha,params[l.field])/pow(h,order);
  else{
   auto beta=alpha;beta[l.component]++;beta[r.component]++;
   value=mixed_radial(l.field,z,beta,params[l.field]);
   if(l.component==r.component)for(int j=0;j<DIM;++j){beta=alpha;beta[j]+=2;value-=mixed_radial(l.field,z,beta,params[l.field]);}
   value/=pow(h,order+2);
  }
  result+=l.coefficient*r.coefficient*Real(sign%2?-1:1)*value;
 }return result;
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("Expected input/output paths");
 std::ifstream in(argv[1]);int digits,threads,condition,inspect,count;
 in>>digits>>threads>>condition>>inspect>>count;
 if(!in||digits<16||threads<1||count<1)throw std::runtime_error("Invalid mixed header");
#ifndef RBFLAB_DOUBLE
 auto precision=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);Real::bits=precision;
#endif
 std::vector<int> vectors(FIELDS),components(FIELDS);
 std::vector<std::vector<Real>> params(FIELDS);
 for(int f=0;f<FIELDS;++f){int np;in>>vectors[f]>>components[f]>>np;
  if(!in||np<0||(vectors[f]!=0&&vectors[f]!=1)||components[f]!=(vectors[f]?DIM:1))throw std::runtime_error("Invalid field descriptor");
  params[f].resize(np);for(auto& v:params[f])v=mixed_read_real(in);
 }
 std::vector<MixedJob> jobs(count);
 for(auto& t:jobs){
  in>>t.n>>t.m>>t.k;if(!in||t.n<1||t.m<0||t.k<1)throw std::runtime_error("Invalid mixed shape");
  t.h=mixed_read_real(in);if(!(Real(0)<t.h))throw std::runtime_error("Invalid kernel scale");
  t.source=Rows(t.n,std::vector<Real>(DIM));t.trial=t.source;t.target=Rows(t.k,std::vector<Real>(DIM));
  for(auto* points:{&t.source,&t.trial})for(auto& p:*points)for(auto& v:p)v=mixed_read_real(in);
  for(int j=0;j<t.n;++j)t.left.push_back(mixed_read_row(in,components));
  for(int j=0;j<t.n;++j)t.right.push_back(mixed_read_row(in,components));
  for(auto& p:t.target)for(auto& v:p)v=mixed_read_real(in);
  for(int j=0;j<t.k;++j)t.ops.push_back(mixed_read_row(in,components));
  t.top=Rows(t.n,std::vector<Real>(t.m));t.bottom=t.top;t.target_poly=Rows(t.k,std::vector<Real>(t.m));
  for(auto* block:{&t.top,&t.bottom,&t.target_poly})for(auto& row:*block)for(auto& v:row)v=mixed_read_real(in);
 }
 std::string trailing;if(in>>trailing)throw std::runtime_error("Trailing mixed input");
 std::vector<std::string> outputs(count),errors(count);
 #pragma omp parallel for num_threads(threads) schedule(dynamic)
 for(int index=0;index<count;++index){try{
#ifndef RBFLAB_DOUBLE
  Real::bits=precision;
#endif
  const auto& t=jobs[index];int n=t.n+t.m,k=t.k;
  Rows G(n,std::vector<Real>(n)),Q(n,std::vector<Real>(k));
  for(int i=0;i<t.n;++i){
   for(int j=0;j<t.n;++j)G[i][j]=mixed_entry(t.source[i],t.left[i],t.trial[j],t.right[j],t.h,vectors,params);
   for(int p=0;p<t.m;++p){G[i][t.n+p]=t.top[i][p];G[t.n+p][i]=t.bottom[i][p];}
   for(int j=0;j<k;++j)Q[i][j]=mixed_entry(t.target[j],t.ops[j],t.trial[i],t.right[i],t.h,vectors,params);
  }
  for(int p=0;p<t.m;++p)for(int j=0;j<k;++j)Q[t.n+p][j]=t.target_poly[j][p];
  std::ostringstream out;out<<index<<" "<<n<<" "<<k;
  if(inspect){for(auto& row:G)for(auto& v:row)out<<" "<<v.str(digits);for(auto& row:Q)for(auto& v:row)out<<" "<<v.str(digits);outputs[index]=out.str();continue;}
  std::vector<Real> scale(n);for(int i=0;i<n;++i){Real maximum;for(auto& v:G[i])if(maximum<abs(v))maximum=abs(v);if(maximum==Real(0))throw std::runtime_error("Zero local row");scale[i]=Real(1)/sqrt(maximum);}
  Rows A=G,B=Q;for(int i=0;i<n;++i){for(int j=0;j<n;++j)A[i][j]=scale[i]*G[j][i]*scale[j];for(int j=0;j<k;++j)B[i][j]*=scale[i];}
  EigenLocalLU factor(A);auto W=factor.solve(B);for(int i=0;i<n;++i)for(int j=0;j<k;++j)W[i][j]*=scale[i];
  Real residual,den;for(int i=0;i<n;++i)for(int j=0;j<k;++j){Real value=-Q[i][j];for(int l=0;l<n;++l)value+=G[l][i]*W[l][j];if(residual<abs(value))residual=abs(value);if(den<abs(Q[i][j]))den=abs(Q[i][j]);}if(!(den==Real(0)))residual/=den;
  std::string cond="none";if(condition){Rows I(n,std::vector<Real>(n));for(int i=0;i<n;++i)I[i][i]=Real(1);auto inverse=factor.solve(I);Real na,ni;for(int i=0;i<n;++i){Real sa,si;for(int j=0;j<n;++j){sa+=abs(A[i][j]);si+=abs(inverse[i][j]);}if(na<sa)na=sa;if(ni<si)ni=si;}cond=(na*ni).str(digits);}
  out<<" "<<cond<<" "<<residual.str(digits);for(auto& row:W)for(auto& v:row)out<<" "<<v.str(digits);outputs[index]=out.str();
 }catch(const std::exception& e){errors[index]=e.what();}}
 for(int i=0;i<count;++i)if(!errors[i].empty())throw std::runtime_error("job "+std::to_string(i)+": "+errors[i]);
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write output");for(auto& s:outputs)out<<s<<"\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
