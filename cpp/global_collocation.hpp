// Persistent scalar global collocation worker. Generated radial_value, DIM and
// PARAMETERS precede this file; all wire scalars are decimal strings.
#include "eigen_solver.hpp"
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <array>
#include <omp.h>

namespace GlobalWorker {
 using Dense=EigenLocalLU::Dense;
 struct Term { Real c; std::vector<int> alpha; };
 using Op=std::vector<Term>;
 void require_finite(const Real& value) {
#ifdef RBFLAB_DOUBLE
  if (!std::isfinite(value.v)) throw std::runtime_error("Nonfinite native arithmetic result");
#else
  if (!mpfr_number_p(value.v)) throw std::runtime_error("Nonfinite native arithmetic result");
#endif
 }
 Real number(std::istream& in){std::string s;if(!(in>>s))throw std::runtime_error("Truncated command");Real v(s);
#ifdef RBFLAB_DOUBLE
  if(!std::isfinite(v.v))throw std::runtime_error("Nonfinite input");
#else
  if(!mpfr_number_p(v.v))throw std::runtime_error("Nonfinite input");
#endif
  return v;}
 int integer(std::istream& in){int x;if(!(in>>x))throw std::runtime_error("Truncated command");return x;}
 Op read_op(std::istream& in){int n=integer(in);
  if(n<0||n>1000)throw std::runtime_error("Invalid operator size");
  Op op;
  op.reserve(n);
  for(int i=0;i<n;++i){Term t;
   t.c=number(in);
   for(int d=0;d<DIM;++d){int a=integer(in);
    if(a<0||a>1000)throw std::runtime_error("Invalid derivative order");
    t.alpha.push_back(a);
   }
   op.push_back(std::move(t));
  }
  return op;
 }
 void end(std::istream& in){std::string extra;if(in>>extra)throw std::runtime_error("Extra command data");}
 struct Worker {
  int n=0,m=0,digits=17,threads=1;bool symmetric=false,built=false,dirty=true;
  std::unique_ptr<Real> h;
  std::vector<Real> params,source,trial,center,scale;
  std::vector<std::vector<int>> powers;
  std::vector<Op> left,right;
  Dense matrix,balanced;
  std::vector<Real> row_scale;
  std::unique_ptr<Eigen::PartialPivLU<Dense>> lu;
  std::unique_ptr<Real> last_residual,condition_estimate;
  int size()const{return n+m;}
  Real coordinate(const std::vector<Real>& p,int i,int d)const{return p[i*DIM+d];}
  Real polynomial(const std::vector<Real>& p,int i,const Op& op,int col)const{
   Real sum;
   for(const auto& t:op){Real v=t.c;
    for(int d=0;d<DIM;++d){int power=powers[col][d],derivative=t.alpha[d];
     if(derivative>power){v=Real(0);
      break;
     }
     for(int k=0;k<derivative;++k)v*=Real(power-k);
     Real q=(coordinate(p,i,d)-center[d])/scale[d];
     v*=pow(q,power-derivative)/pow(scale[d],derivative);
    }
    sum+=v;
   }
   return sum;
  }
  Real kernel(const std::vector<Real>& x,int i,const Op& l,const std::vector<Real>& y,int j,const Op& r)const{
   std::array<Real,DIM> z;for(int d=0;d<DIM;++d)z[d]=(coordinate(x,i,d)-coordinate(y,j,d))/(*h);
   Real sum;
   for(const auto& a:l)for(const auto& b:r){std::array<int,DIM> alpha;
    int order=0,sign=0;
    for(int d=0;d<DIM;++d){alpha[d]=a.alpha[d]+b.alpha[d];
     order+=alpha[d];
     sign+=b.alpha[d];
    }
    // Global coordinates normally have unit kernel scale; avoid a libm call.
    Real value=a.c*b.c*Real(sign%2?-1:1)*radial_value(z,alpha,params);
    sum+=(*h==Real(1))?value:value/pow(*h,order);
   }
   return sum;
  }
  void init(std::istream& in){
   int requested_digits=integer(in),requested_threads=integer(in),sym=integer(in),nn=integer(in),mm=integer(in);
   if(requested_digits<16||requested_digits>10000||requested_threads<1||requested_threads>1024||(sym!=0&&sym!=1)||nn<1||mm<0||nn>100000||mm>10000||nn+mm>100000)throw std::runtime_error("Invalid INIT header");
#ifndef RBFLAB_DOUBLE
   Real::bits=mpfr_prec_t(std::ceil(requested_digits*std::log2(10.0))+8);
#endif
   digits=requested_digits;threads=requested_threads;symmetric=sym;n=nn;m=mm;
   h=std::make_unique<Real>(number(in));if(!(Real(0)<*h))throw std::runtime_error("Invalid kernel scale");
   params=std::vector<Real>(PARAMETERS);for(auto& v:params)v=number(in);
   source=std::vector<Real>(n*DIM);trial=std::vector<Real>(n*DIM);for(auto* p:{&source,&trial})for(auto& v:*p)v=number(in);
   center=std::vector<Real>(DIM);
   scale=std::vector<Real>(DIM);
   for(auto& v:center)v=number(in);
   for(auto& v:scale){v=number(in);
    if(!(Real(0)<v))throw std::runtime_error("Invalid polynomial scale");
   }
   powers.assign(m,std::vector<int>(DIM));
   for(auto& p:powers)for(auto& a:p){a=integer(in);
    if(a<0||a>1000)throw std::runtime_error("Invalid polynomial power");
   }
   left.resize(n);right.resize(n);for(auto& op:left)op=read_op(in);for(auto& op:right)op=read_op(in);end(in);
   matrix.resize(0,0);balanced.resize(0,0);row_scale.clear();lu.reset();last_residual.reset();condition_estimate.reset();built=false;dirty=true;
  }
  void assemble(){if(n<1)throw std::runtime_error("INIT required");
   built=false;
   dirty=true;
   lu.reset();
   last_residual.reset();
   condition_estimate.reset();
   int N=size();
   matrix=Dense::Zero(N,N);
   std::vector<std::string> errors(n);
   // Each entry has a distinct destination; the MPFR precision is thread local.
#pragma omp parallel for num_threads(threads) schedule(dynamic)
   for(int i=0;i<n;++i){try{
#ifndef RBFLAB_DOUBLE
     Real::bits=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);
#endif
     // Symmetric collocation uses the same source/target functionals. Each
     // lower-triangle entry owns its transpose too (no cross-thread writes).
     for(int j=0;j<(symmetric?i+1:n);++j){
      auto value=eigen_value(kernel(source,i,left[i],trial,j,right[j]));
      matrix(i,j)=value;
      if(symmetric&&j!=i)matrix(j,i)=value;
     }
     for(int p=0;p<m;++p)matrix(i,n+p)=eigen_value(polynomial(source,i,left[i],p));
    }catch(const std::exception& e){errors[i]=e.what();}
   }
   for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
   for(int p=0;p<m;++p)for(int j=0;j<n;++j)matrix(n+p,j)=eigen_value(polynomial(trial,j,right[j],p));
   built=true;dirty=true;factor();
  }
  void factor(){if(!built)throw std::runtime_error("BUILD required");if(!dirty)return;int N=size();balanced=matrix;row_scale.assign(N,Real(1));
   for(int i=0;i<N;++i)for(int j=0;j<N;++j)require_finite(Real(matrix(i,j)));
   for(int i=0;i<N;++i){Real mag;
    if(symmetric&&m==0){mag=abs(Real(matrix(i,i)));if(!(Real(0)<mag))throw std::runtime_error("Nonpositive symmetric diagonal");}
    else {for(int j=0;j<N;++j){Real v=abs(Real(matrix(i,j)));if(mag<v)mag=v;}if(mag==Real(0))throw std::runtime_error("Zero matrix row");}
    row_scale[i]=Real(1)/sqrt(mag);
   }
   for(int i=0;i<N;++i)for(int j=0;j<N;++j){
    Real value=row_scale[i]*Real(matrix(i,j))*row_scale[j];
    require_finite(value);
    balanced(i,j)=eigen_value(value);
   }
   lu=std::make_unique<Eigen::PartialPivLU<Dense>>(balanced);
   for(int i=0;i<N;++i)for(int j=0;j<N;++j)require_finite(Real(lu->matrixLU()(i,j)));
   for(int i=0;i<N;++i)if(lu->matrixLU()(i,i)==EigenScalar(0))throw std::runtime_error("Singular global matrix at selected precision");dirty=false;
  }
  std::pair<std::vector<Real>,Real> solve(const std::vector<Real>& b){factor();
   int N=size();
   Dense rhs(N,1);
   for(int i=0;i<N;++i)rhs(i,0)=eigen_value(row_scale[i]*b[i]);
   Dense x=lu->solve(rhs);
   std::vector<Real> result(N);
   for(int i=0;i<N;++i){
    result[i]=row_scale[i]*Real(x(i,0));
    require_finite(result[i]);
   }
   Real residual,den;
   for(int i=0;i<N;++i){Real value=-b[i];
    for(int j=0;j<N;++j)value+=Real(matrix(i,j))*result[j];
    require_finite(value);
    if(residual<abs(value))residual=abs(value);
    if(den<abs(b[i]))den=abs(b[i]);
   }
   if(!(den==Real(0)))residual/=den;
   require_finite(residual);
   last_residual=std::make_unique<Real>(residual);
   return {result,residual};
  }
  Real condition(){
   factor();
   if(condition_estimate)return *condition_estimate;
   const int N=size();
   Real normA;
   for(int j=0;j<N;++j){Real s;
    for(int i=0;i<N;++i)s+=abs(Real(balanced(i,j)));
    if(normA<s)normA=s;
   }
   Dense x=Dense::Constant(N,1,eigen_value(Real(1)/Real(N)));Real inverse_norm;
   for(int iteration=0;iteration<6;++iteration){Dense y=lu->solve(x);
    Real estimate;
    Dense signs(N,1);
    for(int i=0;i<N;++i){Real v(y(i,0));
     estimate+=abs(v);
     signs(i,0)=eigen_value(v<Real(0)?Real(-1):Real(1));
    }
    if(inverse_norm<estimate)inverse_norm=estimate;
    Dense z=lu->transpose().solve(signs);
    int pivot=0;
    Real maximum;
    for(int i=0;i<N;++i){Real value=abs(Real(z(i,0)));
     if(maximum<value){maximum=value;
      pivot=i;
     }
    }
    Real projection;
    for(int i=0;i<N;++i)projection+=Real(z(i,0))*Real(x(i,0));
    if(!(projection<maximum))break;
    x=Dense::Zero(N,1);
    x(pivot,0)=eigen_value(Real(1));
   }
   condition_estimate=std::make_unique<Real>(normA*inverse_norm);return *condition_estimate;
  }
  void command(const std::string& line,std::ostream& out){std::istringstream in(line);
   std::string cmd;
   in>>cmd;
   if(cmd=="HELLO"){end(in);
    out<<"OK "<<DIM<<" "<<PARAMETERS;
   }
   else if(cmd=="INIT"){init(in);out<<"OK";}
   else if(cmd=="RESET"){end(in);*this=Worker();out<<"OK";}
   else if(cmd=="BUILD"){end(in);assemble();out<<"OK "<<size();}
   else if(cmd=="SOLVE"){int count=integer(in);
    if(count<1||count>100000)throw std::runtime_error("Invalid RHS count");
    std::vector<std::vector<Real>> rhs(count,std::vector<Real>(size()));
    for(auto& b:rhs)for(auto& v:b)v=number(in);
    end(in);
    out<<"OK "<<count;
    for(auto& b:rhs){auto answer=solve(b);
     for(auto& x:answer.first)out<<" "<<x.str(digits);
     out<<" "<<answer.second.str(digits);
    }
   }
   else if(cmd=="EVAL"){
    int count=integer(in);if(count<0||count>100000)throw std::runtime_error("Invalid evaluation count");
    std::vector<Real> points(count*DIM);for(auto& v:points)v=number(in);
    std::vector<Op> ops(count);for(auto& op:ops)op=read_op(in);
    std::vector<Real> coeff(size());for(auto& v:coeff)v=number(in);end(in);
    std::vector<std::string> values(count),errors(count);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for(int i=0;i<count;++i){try{
#ifndef RBFLAB_DOUBLE
      Real::bits=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);
#endif
      Real v;for(int j=0;j<n;++j)v+=coeff[j]*kernel(points,i,ops[i],trial,j,right[j]);
      for(int p=0;p<m;++p)v+=coeff[n+p]*polynomial(points,i,ops[i],p);
      require_finite(v);
      values[i]=v.str(digits);
     }catch(const std::exception& e){errors[i]=e.what();}}
    for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
    out<<"OK "<<count;for(const auto& value:values)out<<" "<<value;
   }
   else if(cmd=="GET"){int i=integer(in),j=integer(in);
    end(in);
    if(!built||i<0||j<0||i>=size()||j>=size())throw std::runtime_error("Invalid GET index");
    out<<"OK "<<Real(matrix(i,j)).str(digits);
   }
   else if(cmd=="EXPORT"){end(in);
    if(!built)throw std::runtime_error("BUILD required");
    out<<"OK "<<size();
    for(int i=0;i<size();++i)for(int j=0;j<size();++j)out<<" "<<Real(matrix(i,j)).str(digits);
   }
   else if(cmd=="SET"){int i=integer(in),j=integer(in);
    Real v=number(in);
    end(in);
    if(!built||i<0||j<0||i>=size()||j>=size())throw std::runtime_error("Invalid SET index");
    matrix(i,j)=eigen_value(v);
    dirty=true;
    last_residual.reset();
    condition_estimate.reset();
    out<<"OK";
   }
   else if(cmd=="REPLACE"){if(!built)throw std::runtime_error("BUILD required");
    int N=integer(in);
    if(N!=size())throw std::runtime_error("Invalid replacement shape");
    Dense replacement(N,N);
    for(int i=0;i<N;++i)for(int j=0;j<N;++j)replacement(i,j)=eigen_value(number(in));
    end(in);
    matrix.swap(replacement);
    dirty=true;
    last_residual.reset();
    condition_estimate.reset();
    out<<"OK";
   }
   else if(cmd=="DIAG"){int cond=integer(in);
    end(in);
    if(cond!=0&&cond!=1)throw std::runtime_error("Invalid condition flag");
    factor();
    out<<"OK "<<(last_residual ? last_residual->str(digits) : "none")<<" ";
    if(cond)out<<condition().str(digits);
    else out<<"none";
   }
   else if(cmd=="QUIT"){end(in);out<<"OK";}
   else throw std::runtime_error("Unknown command");
  }
 };
}
int main(){GlobalWorker::Worker worker;
 std::string line;
 while(std::getline(std::cin,line)){try{std::ostringstream out;
   worker.command(line,out);
   std::cout<<out.str()<<std::endl;
   if(line=="QUIT")break;
  }
  catch(const std::exception& e){std::cout<<"ERR "<<e.what()<<std::endl;
  }
 }
 return 0;
}
