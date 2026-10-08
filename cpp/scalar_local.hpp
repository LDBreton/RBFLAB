// Shared scalar local functional assembly. Generated radial::eval precedes this file.
#include "eigen_solver.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
using Rows=EigenLocalLU::Rows;
struct Term {std::vector<int> alpha;Real coefficient;};
using Op=std::vector<Term>;
struct ScalarJob {
 int n,m;Real scale;std::vector<std::vector<Real>> points;
 std::vector<Op> left,right;std::vector<Real> target;Op target_op;
 Rows top,bottom;std::vector<Real> target_poly,rhs;
};
Real read_real(std::istream& in){std::string x;if(!(in>>x))throw std::runtime_error("Truncated scalar input");return Real(x);}
Op read_op(std::istream& in){int count;in>>count;if(count<0||count>1000)throw std::runtime_error("Invalid operator");Op op(count);for(auto&t:op){t.coefficient=read_real(in);t.alpha.resize(DIM);for(auto&a:t.alpha){in>>a;if(a<0)throw std::runtime_error("Negative derivative");}}return op;}
Real entry(const std::vector<Real>&x,const Op&l,const std::vector<Real>&y,const Op&r,const Real&h,const std::vector<Real>&params){
 std::vector<Real> z(DIM);for(int d=0;d<DIM;++d)z[d]=(x[d]-y[d])/h;
 Real result;
 for(auto&a:l)for(auto&b:r){std::vector<int> alpha(DIM);int order=0,sign=0;for(int d=0;d<DIM;++d){alpha[d]=a.alpha[d]+b.alpha[d];order+=alpha[d];sign+=b.alpha[d];}
 result+=a.coefficient*b.coefficient*Real(sign%2?-1:1)*radial_value(z,alpha,params)/pow(h,order);}
 return result;
}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("Expected scalar input/output paths");
 std::ifstream in(argv[1]);int digits,threads,condition,mode,count;in>>digits>>threads>>condition>>mode>>count;
 if(!in||digits<16||threads<1||count<1||(mode!=0&&mode!=1))throw std::runtime_error("Invalid scalar header");
#ifndef RBFLAB_DOUBLE
 auto precision=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);Real::bits=precision;
#endif
 std::vector<Real> params(PARAMETERS);for(auto&v:params)v=read_real(in);
 std::vector<ScalarJob> jobs(count);
 for(auto&t:jobs){in>>t.n>>t.m;if(t.n<1||t.m<0)throw std::runtime_error("Invalid stencil size");t.scale=read_real(in);
 t.points=Rows(t.n,std::vector<Real>(DIM));for(auto&p:t.points)for(auto&v:p)v=read_real(in);
 for(int i=0;i<t.n;++i)t.left.push_back(read_op(in));for(int i=0;i<t.n;++i)t.right.push_back(read_op(in));
 t.target.resize(DIM);for(auto&v:t.target)v=read_real(in);t.target_op=read_op(in);
 t.top=Rows(t.n,std::vector<Real>(t.m));t.bottom=t.top;
 for(auto&r:t.top)for(auto&v:r)v=read_real(in);for(auto&r:t.bottom)for(auto&v:r)v=read_real(in);
 t.target_poly.resize(t.m);for(auto&v:t.target_poly)v=read_real(in);
 if(mode){t.rhs.resize(t.n+t.m);for(auto&v:t.rhs)v=read_real(in);}}
 std::vector<std::string> outputs(count),errors(count);
 #pragma omp parallel for num_threads(threads) schedule(dynamic)
 for(int k=0;k<count;++k){try{
#ifndef RBFLAB_DOUBLE
 Real::bits=precision;
#endif
 auto&t=jobs[k];int n=t.n+t.m;Rows G(n,std::vector<Real>(n)),q(n,std::vector<Real>(1));
 for(int i=0;i<t.n;++i){
  for(int j=0;j<t.n;++j)G[i][j]=entry(t.points[i],t.left[i],t.points[j],t.right[j],t.scale,params);
  for(int j=0;j<t.m;++j){G[i][t.n+j]=t.top[i][j];G[t.n+j][i]=t.bottom[i][j];}
  q[i][0]=entry(t.target,t.target_op,t.points[i],t.right[i],t.scale,params);
 }
 for(int j=0;j<t.m;++j)q[t.n+j][0]=t.target_poly[j];
 if(mode)for(int i=0;i<n;++i)q[i][0]=t.rhs[i];
 // Weights solve G^T w=q; reconstruction coefficients solve G a=d.
 Rows A=G;if(!mode)for(int i=0;i<n;++i)for(int j=0;j<n;++j)A[i][j]=G[j][i];
 std::vector<Real> scale(n);for(int i=0;i<n;++i){Real maximum;for(auto&v:G[i])if(maximum<abs(v))maximum=abs(v);if(maximum==Real(0))throw std::runtime_error("Zero local row");scale[i]=Real(1)/sqrt(maximum);}
 Rows b=q;for(int i=0;i<n;++i){b[i][0]*=scale[i];for(int j=0;j<n;++j)A[i][j]*=scale[i]*scale[j];}
 EigenLocalLU factor(A);auto w=factor.solve(b);for(int i=0;i<n;++i)w[i][0]*=scale[i];
 Real residual,denom;for(int i=0;i<n;++i){Real v=-q[i][0];for(int j=0;j<n;++j)v+=(mode?G[i][j]:G[j][i])*w[j][0];if(residual<abs(v))residual=abs(v);if(denom<abs(q[i][0]))denom=abs(q[i][0]);}if(!(denom==Real(0)))residual/=denom;
 std::string cond="none";
 if(condition){Rows I(n,std::vector<Real>(n));for(int i=0;i<n;++i)I[i][i]=Real(1);auto inverse=factor.solve(I);Real na,ni;for(int i=0;i<n;++i){Real sa,si;for(int j=0;j<n;++j){sa+=abs(A[i][j]);si+=abs(inverse[i][j]);}if(na<sa)na=sa;if(ni<si)ni=si;}cond=(na*ni).str(digits);}
 std::ostringstream out;out<<k<<" "<<n<<" "<<cond<<" "<<residual.str(digits);for(auto&row:w)out<<" "<<row[0].str(digits);outputs[k]=out.str();
 }catch(const std::exception&e){errors[k]=e.what();}}
 for(auto&e:errors)if(!e.empty())throw std::runtime_error(e);
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot write scalar output");for(auto&s:outputs)out<<s<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what();return 1;}}
