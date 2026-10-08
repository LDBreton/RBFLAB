#include "real.hpp"
#include "eigen_solver.hpp"
#include "generated_imq.hpp"
#include "generated_hybrid.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
using Matrix=std::vector<std::vector<Real>>;
struct Node {int op;Real x,y;};
struct Task {KernelSpec v,p;Real mu,x,y;std::vector<Node> nodes;Matrix poly,targets;};
Real evaluate(const Task&t,int s,int op,const Real&x,const Real&y){
 if(t.v.family==0 && t.p.family==0 && t.v.power==0 && t.p.power==0)return kernel(s,op,x,y,t.v.c,t.p.c,t.mu);
 return contribution(true,t.v,s,op,x,y,t.mu)+contribution(false,t.p,s,op,x,y,t.mu);
}
Real read(std::istream& s){std::string v;if(!(s>>v))throw std::runtime_error("Truncated input");return Real(v);}
Real norm(const Matrix& m){Real result;for(auto&r:m){Real sum;for(auto&v:r)sum+=abs(v);if(result<sum)result=sum;}return result;}
int main(int argc,char**argv){try{
#ifdef RBFLAB_DOUBLE
 if(argc==2 && std::string(argv[1])=="--version"){std::cout<<"Eigen PartialPivLU native Float64 OpenMP "<<_OPENMP<<"\n";return 0;}
#else
 if(argc==2 && std::string(argv[1])=="--version"){std::cout<<"Eigen PartialPivLU MPFR "<<mpfr_get_version()<<" GMP "<<gmp_version<<" OpenMP "<<_OPENMP<<"\n";return 0;}
#endif
 if(argc!=3 && argc!=5)throw std::runtime_error("Usage: lhi_mpfr INPUT OUTPUT [lu|svd CUTOFF]");
 std::string solver=argc==5 ? argv[3] : "lu";
 double cutoff=argc==5 ? std::stod(argv[4]) : -1;
 if((solver!="lu" && solver!="svd") || !std::isfinite(cutoff) || (cutoff!=-1 && (cutoff<0 || cutoff>=1)))throw std::runtime_error("Invalid local solver or relative SVD threshold");
 std::ifstream in(argv[1]);int digits,threads,condition,count;
 if(!(in>>digits>>threads>>condition>>count)||digits<16||threads<1||count<1)throw std::runtime_error("Invalid header");
#ifndef RBFLAB_DOUBLE
 auto precision=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);Real::bits=precision;
#endif
 std::vector<Task> tasks(count);
 for(auto&t:tasks){int n,m;if(!(in>>n>>m)||n<1||m<0)throw std::runtime_error("Invalid stencil size");
  for(auto*k:{&t.v,&t.p}){if(!(in>>k->family>>k->power)||k->family<0||k->family>1||(k->power!=0&&k->power!=3&&k->power!=5&&k->power!=7&&k->power!=9))throw std::runtime_error("Invalid kernel");k->c=read(in);k->weight=read(in);}
  if(t.v.power && t.v.power!=7 && t.v.power!=9)throw std::runtime_error("Velocity needs PHS7 or PHS9");
  t.mu=read(in);t.x=read(in);t.y=read(in);t.nodes.resize(n);t.poly=Matrix(n,std::vector<Real>(m));t.targets=Matrix(m,std::vector<Real>(4));
  for(auto&p:t.nodes){if(!(in>>p.op)||p.op<1||p.op>6)throw std::runtime_error("Invalid source operator");p.x=read(in);p.y=read(in);}
  for(auto&row:t.poly)for(auto&v:row)v=read(in);
  for(auto&row:t.targets)for(auto&v:row)v=read(in);}
 std::vector<std::string> output(count),errors(count);
 #pragma omp parallel for num_threads(threads) schedule(dynamic)
 for(int task=0;task<count;task++){try{
#ifndef RBFLAB_DOUBLE
  Real::bits=precision;
#endif
  auto&t=tasks[task];int physical=t.nodes.size(),n=physical+t.targets.size();Matrix G(n,std::vector<Real>(n)),B(n,std::vector<Real>(4));
  for(int i=0;i<physical;i++){
   auto&p=t.nodes[i];
   for(int j=0;j<=i;j++){auto&q=t.nodes[j];G[i][j]=evaluate(t,q.op,p.op,p.x-q.x,p.y-q.y);G[j][i]=G[i][j];}
   for(int k=0;k<4;k++)B[i][k]=evaluate(t,p.op,k+5,t.x-p.x,t.y-p.y);
  }
  for(int j=0;j<int(t.targets.size());j++){
   for(int i=0;i<physical;i++)G[i][physical+j]=G[physical+j][i]=t.poly[i][j];
   for(int k=0;k<4;k++)B[physical+j][k]=t.targets[j][k];
  }
  std::vector<Real> scale(n);Matrix A=G,R=B;
  for(int i=0;i<n;i++){Real max;for(auto&v:G[i])if(max<abs(v))max=abs(v);if(max==Real(0))throw std::runtime_error("Zero local row");scale[i]=Real(1)/sqrt(max);}
  for(int i=0;i<n;i++){for(int j=0;j<n;j++)A[i][j]=scale[i]*G[i][j]*scale[j];for(int k=0;k<4;k++)R[i][k]*=scale[i];}
  EigenLocalSolver lu(A,solver,cutoff);auto W=lu.solve(R);for(int i=0;i<n;i++)for(auto&v:W[i])v*=scale[i];
  Real residual,denom;for(int i=0;i<n;i++)for(int k=0;k<4;k++){Real d=-B[i][k];for(int j=0;j<n;j++)d+=G[i][j]*W[j][k];if(residual<abs(d))residual=abs(d);if(denom<abs(B[i][k]))denom=abs(B[i][k]);}
  if(!(denom==Real(0)))residual/=denom;
  std::string cond="none";
  if(condition && solver=="lu"){Matrix I(n,std::vector<Real>(n));for(int i=0;i<n;i++)I[i][i]=Real(1);cond=(norm(A)*norm(lu.solve(I))).str(digits);}
  std::ostringstream line;line<<task<<" "<<physical<<" "<<cond<<" "<<residual.str(digits);
  for(int k=0;k<4;k++)for(int i=0;i<physical;i++)line<<" "<<W[i][k].str(digits);
  if(solver=="svd")line<<" "<<lu.rank<<" "<<n<<" "<<std::setprecision(17)<<lu.threshold<<" "<<lu.condition2;
  output[task]=line.str();
 }catch(const std::exception&e){errors[task]=e.what();}}
 for(auto&e:errors)if(!e.empty())throw std::runtime_error(e);
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("Cannot create output");for(auto&s:output)out<<s<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
