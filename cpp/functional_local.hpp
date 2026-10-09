// Functional approximation runtime: source and trial functionals are independent.
// Generated radial derivatives and Real precede this source.
#include "eigen_solver.hpp"
#include <sstream>
#include <omp.h>
using Rows=EigenLocalLU::Rows;
struct Term { std::vector<int> alpha; Real coefficient; };
using Op=std::vector<Term>;
struct Job {
 int n,m,k; Real h;
 Rows source,trial,top,bottom,target_poly,target;

 std::vector<Op> left,right,ops;
};
Real read_real(std::istream& in) {
 std::string s; if(!(in>>s))throw std::runtime_error("Truncated input");return Real(s);
}
Op read_op(std::istream& in) {
 int n;in>>n;if(!in||n<0||n>1000)throw std::runtime_error("Invalid operator");
 Op result(n);for(auto& t:result){t.coefficient=read_real(in);t.alpha.resize(DIM);for(auto& a:t.alpha){in>>a;if(a<0)throw std::runtime_error("Invalid derivative");}}
 if(!in)throw std::runtime_error("Truncated operator");return result;
}
Real entry(const std::vector<Real>& x,const Op& left,const std::vector<Real>& y,const Op& right,int a,int b,int components,const Real& h,const std::vector<Real>& params) {
 std::vector<Real> z(DIM);for(int j=0;j<DIM;++j)z[j]=(x[j]-y[j])/h;
 Real result;
 for(const auto& l:left)for(const auto& r:right){
  std::vector<int> alpha(DIM);int order=0,sign=0;
  for(int j=0;j<DIM;++j){alpha[j]=l.alpha[j]+r.alpha[j];order+=alpha[j];sign+=r.alpha[j];}
  Real value;
  if(components==1)value=radial_value(z,alpha,params)/pow(h,order);
  else{
   auto beta=alpha;beta[a]++;beta[b]++;value=radial_value(z,beta,params);
   if(a==b)for(int j=0;j<DIM;++j){beta=alpha;beta[j]+=2;value-=radial_value(z,beta,params);}
   value/=pow(h,order+2);
  }
  result+=l.coefficient*r.coefficient*Real(sign%2?-1:1)*value;
 }
 return result;
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("Expected input/output paths");
 std::ifstream in(argv[1]);int digits,threads,condition,inspect,count,c;
 in>>digits>>threads>>condition>>inspect>>count>>c;
 if(!in||digits<16||threads<1||count<1||(c!=1&&c!=DIM))throw std::runtime_error("Invalid header");
#ifndef RBFLAB_DOUBLE
 auto precision=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);Real::bits=precision;
#endif
 std::vector<Real> params(PARAMETERS);for(auto& v:params)v=read_real(in);
 std::vector<Job> jobs(count);
 for(auto& t:jobs){
  in>>t.n>>t.m>>t.k;if(!in||t.n<1||t.m<0||t.k<1)throw std::runtime_error("Invalid shape");
  t.h=read_real(in);if(!(Real(0)<t.h))throw std::runtime_error("Invalid scale");
  t.source=Rows(t.n,std::vector<Real>(DIM));t.trial=t.source;
  for(auto* points:{&t.source,&t.trial})for(auto& p:*points)for(auto& v:p)v=read_real(in);
  for(int j=0;j<t.n;++j)t.left.push_back(read_op(in));
  for(int j=0;j<t.n;++j)t.right.push_back(read_op(in));
  t.target=Rows(t.k,std::vector<Real>(DIM));for(auto& p:t.target)for(auto& v:p)v=read_real(in);
  for(int j=0;j<t.k;++j)t.ops.push_back(read_op(in));
  t.top=Rows(t.n*c,std::vector<Real>(t.m));t.bottom=t.top;
  t.target_poly=Rows(t.k*c,std::vector<Real>(t.m));
  for(auto* block:{&t.top,&t.bottom,&t.target_poly})for(auto& row:*block)for(auto& v:row)v=read_real(in);
 }
 std::vector<std::string> outputs(count),errors(count);
 #pragma omp parallel for num_threads(threads) schedule(dynamic)
 for(int index=0;index<count;++index){try{
#ifndef RBFLAB_DOUBLE
  Real::bits=precision;
#endif
  const auto& t=jobs[index];int dofs=t.n*c,n=dofs+t.m,k=t.k*c;
  Rows G(n,std::vector<Real>(n)),Q(n,std::vector<Real>(k));
  for(int i=0;i<t.n;++i)for(int a=0;a<c;++a){int row=i*c+a;
   for(int j=0;j<t.n;++j)for(int b=0;b<c;++b)
    G[row][j*c+b]=entry(t.source[i],t.left[i],t.trial[j],t.right[j],a,b,c,t.h,params);
   for(int p=0;p<t.m;++p){G[row][dofs+p]=t.top[row][p];G[dofs+p][row]=t.bottom[row][p];}
   for(int op=0;op<t.k;++op)for(int out=0;out<c;++out)
    Q[row][op*c+out]=entry(t.target[op],t.ops[op],t.trial[i],t.right[i],out,a,c,t.h,params);
  }
  for(int p=0;p<t.m;++p)for(int j=0;j<k;++j)Q[dofs+p][j]=t.target_poly[j][p];
  std::ostringstream out;out<<index<<" "<<n<<" "<<k;
  if(inspect){for(auto& row:G)for(auto& v:row)out<<" "<<v.str(digits);for(auto& row:Q)for(auto& v:row)out<<" "<<v.str(digits);outputs[index]=out.str();continue;}
  std::vector<Real> scale(n);for(int i=0;i<n;++i){Real maximum;for(auto& v:G[i])if(maximum<abs(v))maximum=abs(v);if(maximum==Real(0))throw std::runtime_error("Zero local row");scale[i]=Real(1)/sqrt(maximum);}
  Rows A=G,B=Q;for(int i=0;i<n;++i){for(int j=0;j<n;++j)A[i][j]=scale[i]*G[j][i]*scale[j];for(int j=0;j<k;++j)B[i][j]*=scale[i];}
  EigenLocalLU factor(A);auto W=factor.solve(B);for(int i=0;i<n;++i)for(int j=0;j<k;++j)W[i][j]*=scale[i];
  Real residual,den;for(int i=0;i<n;++i)for(int j=0;j<k;++j){Real value=-Q[i][j];for(int l=0;l<n;++l)value+=G[l][i]*W[l][j];if(residual<abs(value))residual=abs(value);if(den<abs(Q[i][j]))den=abs(Q[i][j]);}
  if(!(den==Real(0)))residual/=den;
  std::string cond="none";
  if(condition){Rows I(n,std::vector<Real>(n));for(int i=0;i<n;++i)I[i][i]=Real(1);auto inverse=factor.solve(I);Real na,ni;
   for(int i=0;i<n;++i){Real sa,si;for(int j=0;j<n;++j){sa+=abs(A[i][j]);si+=abs(inverse[i][j]);}if(na<sa)na=sa;if(ni<si)ni=si;}cond=(na*ni).str(digits);
  }
  out<<" "<<cond<<" "<<residual.str(digits);for(auto& row:W)for(auto& v:row)out<<" "<<v.str(digits);outputs[index]=out.str();
 }catch(const std::exception& e){errors[index]=e.what();}}
 for(auto& e:errors)if(!e.empty())throw std::runtime_error(e);
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write output");for(auto& s:outputs)out<<s<<"\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
