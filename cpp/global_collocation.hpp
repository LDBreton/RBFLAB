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
#include <chrono>
#include <iomanip>
#include <map>

namespace GlobalWorker {
 using ProfileClock=std::chrono::steady_clock;
 double profile_seconds(ProfileClock::time_point start){return std::chrono::duration<double>(ProfileClock::now()-start).count();}
 using Dense=EigenLocalLU::Dense;
 struct Term { Real c; std::vector<int> alpha; int component=0; };
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
#ifdef RBFLAB_GLOBAL_FIELDS
   t.component=integer(in);
   if(t.component<0||t.component>DIM)throw std::runtime_error("Invalid field component");
#endif
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
  bool profile_eval=false,profile_available=false;
  std::array<double,3> eval_seconds{};
  std::unique_ptr<Real> h;
  std::vector<Real> params,source,trial,center,scale;
  std::vector<std::vector<int>> powers;
#ifdef RBFLAB_GLOBAL_FIELDS
  struct PolynomialTerm {int component;Real coefficient;std::vector<int> power;};
  std::vector<std::vector<PolynomialTerm>> columns;
  std::vector<Real> pressure_params;
#endif
  std::vector<Op> left,right;
  struct PreparedTerm {
   Real coefficient,divisor;
   bool pressure=false,negative=false;
   std::vector<int> derivatives;
  };
  using KernelPlan=std::vector<PreparedTerm>;
  struct PlanBank {
   std::vector<int> left_ids,right_ids;
   std::vector<KernelPlan> plans;
   int right_count=0;
   const KernelPlan* find(int i,int j)const{
    return plans.empty()?nullptr:&plans[left_ids[i]*right_count+right_ids[j]];
   }
  };
  PlanBank assembly_plans;
  std::vector<std::vector<int>> trial_groups;
  bool unit_scale=true;
  Dense matrix,balanced;
  Dense stiffness,mass;
  struct ShiftFactor {
   Dense balanced;
   std::vector<Real> scales;
   std::shared_ptr<Eigen::PartialPivLU<Dense>> lu;
  };
  std::map<std::string,ShiftFactor> shift_factors;
  bool has_mass=false;
  std::vector<Real> row_scale;
  std::shared_ptr<Eigen::PartialPivLU<Dense>> lu;
  std::unique_ptr<Real> last_residual,condition_estimate;
  int size()const{return n+m;}
  static bool same_operator(const Op& a,const Op& b){
   if(a.size()!=b.size())return false;
   for(size_t i=0;i<a.size();++i)
    if(a[i].component!=b[i].component||a[i].alpha!=b[i].alpha||!(a[i].c==b[i].c))return false;
   return true;
  }
  static bool group_operators(const std::vector<Op>& ops,std::vector<Op>& unique,std::vector<int>& ids){
   ids.reserve(ops.size());
   for(const auto& op:ops){
    size_t index=0;while(index<unique.size()&&!same_operator(op,unique[index]))++index;
    if(index==unique.size()){
     if(unique.size()==32)return false;
     unique.push_back(op);
    }
    ids.push_back(int(index));
   }
   return true;
  }
  KernelPlan prepare_kernel(const Op& l,const Op& r)const{
   KernelPlan plan;
   for(const auto& a:l)for(const auto& b:r){
    PreparedTerm term;std::array<int,DIM> alpha;
    int order=0,sign=0;
    for(int d=0;d<DIM;++d){alpha[d]=a.alpha[d]+b.alpha[d];order+=alpha[d];sign+=b.alpha[d];}
#ifdef RBFLAB_GLOBAL_FIELDS
    if(a.component==DIM&&b.component==DIM){
     term.pressure=true;term.derivatives.push_back(pressure_index(alpha));
    }else if(a.component<DIM&&b.component<DIM){
     if(a.component==b.component){
      term.negative=true;
      for(int d=0;d<DIM;++d)if(d!=a.component){auto beta=alpha;beta[d]+=2;term.derivatives.push_back(radial_index(beta));}
     }else{auto beta=alpha;beta[a.component]++;beta[b.component]++;term.derivatives.push_back(radial_index(beta));}
     order+=2;
    }else continue;
#else
    term.derivatives.push_back(radial_index(alpha));
#endif
    term.coefficient=a.c*b.c*Real(sign%2?-1:1);
    if(!unit_scale)term.divisor=pow(*h,order);
    plan.push_back(std::move(term));
   }
   return plan;
  }
  PlanBank prepare_plans(const std::vector<Op>& lhs,const std::vector<Op>& rhs)const{
   PlanBank bank;std::vector<Op> ls,rs;
   if(!group_operators(lhs,ls,bank.left_ids)||!group_operators(rhs,rs,bank.right_ids))return {};
   size_t lt=0,rt=0;for(const auto& op:ls)lt+=op.size();for(const auto& op:rs)rt+=op.size();
   if(lt*rt>65536)return {}; // Bound extra storage for varied/large operators.
   bank.right_count=int(rs.size());bank.plans.reserve(ls.size()*rs.size());
   for(const auto& l:ls)for(const auto& r:rs)bank.plans.push_back(prepare_kernel(l,r));
   return bank;
  }
  struct PairValues {
   static std::array<Real,DIM> displacement(const Worker& w,const std::vector<Real>& x,int i,const std::vector<Real>& y,int j){
    std::array<Real,DIM> result;
    for(int d=0;d<DIM;++d){result[d]=w.coordinate(x,i,d)-w.coordinate(y,j,d);if(!w.unit_scale)result[d]/=*w.h;}
    return result;
   }
   std::array<Real,DIM> z;
   radial::Shared velocity;
#ifdef RBFLAB_GLOBAL_FIELDS
   pressure::Shared pressure;
#endif
   PairValues(const Worker& w,const std::vector<Real>& x,int i,const std::vector<Real>& y,int j):
    z(displacement(w,x,i,y,j)),velocity(z,w.params)
#ifdef RBFLAB_GLOBAL_FIELDS
    ,pressure(z,w.pressure_params)
#endif
   {}
  };
  Real planned_kernel(PairValues& pair,const KernelPlan& plan)const{
   Real sum;
   for(const auto& term:plan){
    Real radial;
#ifdef RBFLAB_GLOBAL_FIELDS
    if(term.pressure)radial=pair.pressure.get(term.derivatives[0]);
    else
#endif
    if(term.negative){for(int index:term.derivatives)radial-=pair.velocity.get(index);}
    else radial=pair.velocity.get(term.derivatives[0]);
    Real value=term.coefficient*radial;
    sum+=unit_scale?value:value/term.divisor;
   }
   return sum;
  }
  Real planned_kernel(const std::vector<Real>& x,int i,const std::vector<Real>& y,int j,const KernelPlan& plan)const{
   if(plan.empty())return Real(0);
   PairValues pair(*this,x,i,y,j);return planned_kernel(pair,plan);
  }
  static std::vector<std::vector<int>> group_points(const std::vector<Real>& points,int limit){
   std::vector<int> order(points.size()/DIM);
   for(size_t i=0;i<order.size();++i)order[i]=int(i);
   auto less=[&](int a,int b){for(int d=0;d<DIM;++d){
    if(points[a*DIM+d]<points[b*DIM+d])return true;
    if(points[b*DIM+d]<points[a*DIM+d])return false;
   }return false;};
   std::stable_sort(order.begin(),order.end(),less);
   std::vector<std::vector<int>> groups;
   for(int i:order){
    if(groups.empty()||int(groups.back().size())==limit||less(groups.back()[0],i))groups.push_back({});
    groups.back().push_back(i);
   }
   return groups;
  }
  Real coordinate(const std::vector<Real>& p,int i,int d)const{return p[i*DIM+d];}
  Real polynomial(const std::vector<Real>& p,int i,const Op& op,int col)const{
   Real sum;
#ifdef RBFLAB_GLOBAL_FIELDS
   for(const auto& basis:columns[col])for(const auto& t:op){
    if(t.component!=basis.component)continue;
    Real v=t.c*basis.coefficient;
#else
   for(const auto& t:op){Real v=t.c;
#endif
    for(int d=0;d<DIM;++d){
#ifdef RBFLAB_GLOBAL_FIELDS
     int power=basis.power[d],derivative=t.alpha[d];
#else
     int power=powers[col][d],derivative=t.alpha[d];
#endif
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
#ifdef RBFLAB_GLOBAL_FIELDS
    Real radial;
    if(a.component==DIM&&b.component==DIM){
     radial=pressure_value(z,alpha,pressure_params);
    }else if(a.component<DIM&&b.component<DIM){
     if(a.component==b.component){
      // Hessian_cc - Laplacian cancels derivative_cc exactly. Evaluate
      // only the remaining diagonal derivatives (one radial call in 2D).
      for(int d=0;d<DIM;++d)if(d!=a.component){
       auto beta=alpha;beta[d]+=2;radial-=radial_value(z,beta,params);
      }
     }else{
      auto beta=alpha;beta[a.component]++;beta[b.component]++;
      radial=radial_value(z,beta,params);
     }
     order+=2;
    }else continue; // Independent velocity and pressure blocks of the kernel.
    Real value=a.c*b.c*Real(sign%2?-1:1)*radial;
#else
    Real value=a.c*b.c*Real(sign%2?-1:1)*radial_value(z,alpha,params);
#endif
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
   unit_scale=(*h==Real(1));
   params=std::vector<Real>(PARAMETERS);for(auto& v:params)v=number(in);
#ifdef RBFLAB_GLOBAL_FIELDS
   if(!symmetric)throw std::runtime_error("Field collocation requires symmetric scheme");
   pressure_params=std::vector<Real>(PRESSURE_PARAMETERS);for(auto& v:pressure_params)v=number(in);
#endif
   source=std::vector<Real>(n*DIM);trial=std::vector<Real>(n*DIM);for(auto* p:{&source,&trial})for(auto& v:*p)v=number(in);
   center=std::vector<Real>(DIM);
   scale=std::vector<Real>(DIM);
   for(auto& v:center)v=number(in);
   for(auto& v:scale){v=number(in);
    if(!(Real(0)<v))throw std::runtime_error("Invalid polynomial scale");
   }
#ifdef RBFLAB_GLOBAL_FIELDS
   columns.clear();columns.resize(m);
   for(auto& column:columns){
    int count=integer(in);
    if(count<1||count>10000)throw std::runtime_error("Invalid polynomial column");
    for(int k=0;k<count;++k){PolynomialTerm term;
     term.component=integer(in);
     if(term.component<0||term.component>DIM)throw std::runtime_error("Invalid polynomial component");
     term.coefficient=number(in);term.power.resize(DIM);
     for(auto& a:term.power){a=integer(in);if(a<0||a>1000)throw std::runtime_error("Invalid polynomial power");}
     column.push_back(std::move(term));
    }
   }
#else
   powers.assign(m,std::vector<int>(DIM));
   for(auto& p:powers)for(auto& a:p){a=integer(in);
    if(a<0||a>1000)throw std::runtime_error("Invalid polynomial power");
   }
#endif
   left.resize(n);right.resize(n);for(auto& op:left)op=read_op(in);for(auto& op:right)op=read_op(in);end(in);
   assembly_plans=prepare_plans(left,right);
   trial_groups=group_points(trial,n);
   matrix.resize(0,0);balanced.resize(0,0);row_scale.clear();lu.reset();last_residual.reset();condition_estimate.reset();built=false;dirty=true;
   has_mass=false;stiffness.resize(0,0);mass.resize(0,0);shift_factors.clear();
  }
  void assemble(bool factor_now=true){if(n<1)throw std::runtime_error("INIT required");
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
      const auto* plan=assembly_plans.find(i,j);
      auto value=eigen_value(plan?planned_kernel(source,i,trial,j,*plan):kernel(source,i,left[i],trial,j,right[j]));
      matrix(i,j)=value;
      if(symmetric&&j!=i)matrix(j,i)=value;
     }
     for(int p=0;p<m;++p)matrix(i,n+p)=eigen_value(polynomial(source,i,left[i],p));
    }catch(const std::exception& e){errors[i]=e.what();}
   }
   for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
   for(int p=0;p<m;++p)for(int j=0;j<n;++j)matrix(n+p,j)=eigen_value(polynomial(trial,j,right[j],p));
   built=true;dirty=true;if(factor_now)factor();
  }
  void assemble_mass(const std::vector<Op>& rows){
   if(!built||rows.size()>size_t(n))throw std::runtime_error("Invalid mass rows");
   const int count=int(rows.size());
   Dense next=Dense::Zero(size(),size());
   auto plans=prepare_plans(rows,right);
   std::vector<std::string> errors(count);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
   for(int i=0;i<count;++i){try{
#ifndef RBFLAB_DOUBLE
    Real::bits=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);
#endif
    for(int j=0;j<n;++j){const auto* plan=plans.find(i,j);
     next(i,j)=eigen_value(plan?planned_kernel(source,i,trial,j,*plan):kernel(source,i,rows[i],trial,j,right[j]));}
    for(int p=0;p<m;++p)next(i,n+p)=eigen_value(polynomial(source,i,rows[i],p));
   }catch(const std::exception& e){errors[i]=e.what();}}
   for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
   stiffness=matrix;mass.swap(next);has_mass=true;shift_factors.clear();
  }
  bool shift(const Real& alpha){
   if(!has_mass)throw std::runtime_error("MASS required");
   std::string key=alpha.str(digits+8);
   matrix=stiffness+eigen_value(alpha)*mass;
   last_residual.reset();condition_estimate.reset();
   auto found=shift_factors.find(key);
   if(found!=shift_factors.end()){
    balanced=found->second.balanced;row_scale=found->second.scales;
    lu=found->second.lu;dirty=false;
   }else{
    dirty=true;factor();
    // Bound storage for repeated calls with changing time steps.
    if(shift_factors.size()>=4)shift_factors.erase(shift_factors.begin());
    ShiftFactor saved;saved.balanced=balanced;saved.scales=row_scale;
    saved.lu=lu;
    shift_factors.emplace(key,std::move(saved));
   }
   return found==shift_factors.end();
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
   lu=std::make_shared<Eigen::PartialPivLU<Dense>>(balanced);
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
   else if(cmd=="BUILD_MOL"){end(in);assemble(false);out<<"OK "<<size();}
   else if(cmd=="MASS"){int count=integer(in);
    if(count<0||count>n)throw std::runtime_error("Invalid mass count");
    std::vector<Op> rows(count);for(auto& row:rows)row=read_op(in);end(in);
    assemble_mass(rows);out<<"OK";
   }
   else if(cmd=="SHIFT"){Real alpha=number(in);end(in);bool fresh=shift(alpha);out<<"OK "<<fresh<<" "<<shift_factors.size();}
   else if(cmd=="MASS_MV"){
    if(!has_mass)throw std::runtime_error("MASS required");
    Dense coefficients(size(),1);for(int i=0;i<size();++i)coefficients(i,0)=eigen_value(number(in));end(in);
    Dense values=mass*coefficients;out<<"OK "<<size();
    for(int i=0;i<size();++i)out<<" "<<Real(values(i,0)).str(digits);
   }
   else if(cmd=="MASS_EXPORT"){
    end(in);if(!has_mass)throw std::runtime_error("MASS required");
    out<<"OK "<<size();for(int i=0;i<size();++i)for(int j=0;j<size();++j)out<<" "<<Real(mass(i,j)).str(digits);
   }
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
   else if(cmd=="EVAL_PROFILE"){int enabled=integer(in);end(in);
    if(enabled!=0&&enabled!=1)throw std::runtime_error("EVAL_PROFILE expects 0 or 1");
    profile_eval=enabled;profile_available=false;eval_seconds={};out<<"OK";
   }
   else if(cmd=="EVAL_TIMING"){end(in);
    if(!profile_available)throw std::runtime_error("No profiled EVAL available");
    out<<"OK"<<std::setprecision(17);for(double seconds:eval_seconds)out<<" "<<seconds;
   }
   else if(cmd=="EVAL"){
    auto parse_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
    profile_available=false;
    int count=integer(in);if(count<0||count>100000)throw std::runtime_error("Invalid evaluation count");
    std::vector<Real> points(count*DIM);for(auto& v:points)v=number(in);
    std::vector<Op> ops(count);for(auto& op:ops)op=read_op(in);
    std::vector<Real> coeff(size());for(auto& v:coeff)v=number(in);end(in);
    auto evaluation_plans=prepare_plans(ops,right);
    auto point_groups=group_points(points,8); // Bound scratch to eight rows per thread.
    if(profile_eval)eval_seconds[0]=profile_seconds(parse_start);
    auto compute_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
    std::vector<std::string> values(count),errors(count);
    std::vector<Real> numeric_values(profile_eval?count:0);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for(int g=0;g<int(point_groups.size());++g){try{
#ifndef RBFLAB_DOUBLE
      Real::bits=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);
#endif
      const auto& group=point_groups[g];
      std::vector<Real> contributions;
      if(!evaluation_plans.plans.empty()){
       contributions.resize(group.size()*n);
       for(const auto& sources:trial_groups){
        PairValues pair(*this,points,group[0],trial,sources[0]);
        for(size_t k=0;k<group.size();++k)for(int j:sources)
         contributions[k*n+j]=coeff[j]*planned_kernel(pair,*evaluation_plans.find(group[k],j));
       }
      }
      for(size_t k=0;k<group.size();++k){
       int i=group[k];Real v;
       // Keep the original source summation order despite point grouping.
       for(int j=0;j<n;++j)v+=contributions.empty()?coeff[j]*kernel(points,i,ops[i],trial,j,right[j]):contributions[k*n+j];
       for(int p=0;p<m;++p)v+=coeff[n+p]*polynomial(points,i,ops[i],p);
       require_finite(v);
       if(profile_eval)numeric_values[i]=v;
       else values[i]=v.str(digits);
      }
     }catch(const std::exception& e){errors[point_groups[g][0]]=e.what();}}
    for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
    if(profile_eval)eval_seconds[1]=profile_seconds(compute_start);
    auto format_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
    if(profile_eval){
#pragma omp parallel for num_threads(threads) schedule(dynamic)
     for(int i=0;i<count;++i){try{
#ifndef RBFLAB_DOUBLE
       Real::bits=mpfr_prec_t(std::ceil(digits*std::log2(10.0))+8);
#endif
       values[i]=numeric_values[i].str(digits);
      }catch(const std::exception& e){errors[i]=e.what();}}
     for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
    }
    out<<"OK "<<count;for(const auto& value:values)out<<" "<<value;
    if(profile_eval){eval_seconds[2]=profile_seconds(format_start);profile_available=true;}
   }
   else if(cmd=="GET"){int i=integer(in),j=integer(in);
    end(in);
    if(!built||i<0||j<0||i>=size()||j>=size())throw std::runtime_error("Invalid GET index");
    out<<"OK "<<Real((has_mass?stiffness:matrix)(i,j)).str(digits);
   }
   else if(cmd=="EXPORT"){end(in);
    if(!built)throw std::runtime_error("BUILD required");
    out<<"OK "<<size();
    for(int i=0;i<size();++i)for(int j=0;j<size();++j)out<<" "<<Real((has_mass?stiffness:matrix)(i,j)).str(digits);
   }
   else if(cmd=="SET"){int i=integer(in),j=integer(in);
    Real v=number(in);
    end(in);
    if(!built||i<0||j<0||i>=size()||j>=size())throw std::runtime_error("Invalid SET index");
    matrix(i,j)=eigen_value(v);
    if(has_mass){stiffness(i,j)=eigen_value(v);matrix=stiffness;shift_factors.clear();}
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
    if(has_mass){stiffness=matrix;shift_factors.clear();}
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
