// Persistent scalar global collocation worker. Generated radial_value, DIM and
// PARAMETERS precede this file; all wire scalars are decimal strings.
#include "global_solver.hpp"
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
#include <list>

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
#ifndef RBFLAB_GLOBAL_PRESSURE
   if(t.component==DIM)throw std::runtime_error("Pressure component unavailable");
#endif
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
  bool gram_matrix=false;
  bool profile_eval=false,profile_available=false;
  std::array<double,3> eval_seconds{};
  bool profile_global=false;
  std::array<double,10> global_seconds{};
  std::map<std::string,std::pair<size_t,std::array<double,10>>> global_timings;
#ifdef RBFLAB_GLOBAL_DATA
  std::vector<int> data_rows;
#endif
  ProfileClock::time_point global_start()const{return profile_global?ProfileClock::now():ProfileClock::time_point{};}
  void global_phase(int index,ProfileClock::time_point start){if(profile_global)global_seconds[index]+=profile_seconds(start);}
  std::unique_ptr<Real> h;
  std::vector<Real> params,source,trial,center,scale;
  std::vector<std::vector<int>> powers;
#ifdef RBFLAB_GLOBAL_FIELDS
  struct PolynomialTerm {int component;Real coefficient;std::vector<int> power;};
  std::vector<std::vector<PolynomialTerm>> columns;
#ifdef RBFLAB_GLOBAL_PRESSURE
  std::vector<Real> pressure_params;
#endif
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
  std::vector<Op> mass_rows;
  struct ShiftFactor {
   Dense balanced;
   std::vector<Real> scales;
   std::shared_ptr<GlobalDenseLU> lu;
  };
  std::map<std::string,ShiftFactor> shift_factors;
  std::string active_shift;
  using EvaluationMatrix=Eigen::Matrix<EigenScalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  struct EvaluationCache {std::string key;EvaluationMatrix matrix;size_t bytes;};
  std::list<EvaluationCache> evaluation_cache;
  size_t evaluation_cache_bytes=0,evaluation_cache_hits=0,evaluation_cache_misses=0;
  static constexpr size_t evaluation_cache_limit=size_t(512)*1024*1024;
  bool has_mass=false;
  std::vector<Real> row_scale;
  std::shared_ptr<GlobalDenseLU> lu;
  Dense initial_matrix;
  std::unique_ptr<ShiftFactor> initial_factor;
  std::vector<Real> evolution_previous,evolution_previous2;
  std::unique_ptr<Real> last_residual,condition_estimate;
  int size()const{return n+m;}
  void clear_evaluation_cache(){evaluation_cache.clear();evaluation_cache_bytes=0;}
  template<class Matrix> std::vector<Real> multiply(const Matrix& a,const std::vector<Real>& b)const{
   std::vector<Real> values(a.rows());
#pragma omp parallel for num_threads(threads) schedule(static) if(a.rows()>=128)
   for(int i=0;i<a.rows();++i){Real sum,product;
    for(int j=0;j<a.cols();++j){
#ifdef RBFLAB_DOUBLE
     sum.v+=a(i,j)*b[j].v;
#else
     mpfr_mul(product.v,a(i,j).v,b[j].v,MPFR_RNDN);
     mpfr_add(sum.v,sum.v,product.v,MPFR_RNDN);
#endif
    }
    values[i]=std::move(sum);
   }
   for(const auto& v:values)require_finite(v);
   return values;
  }
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
#ifdef RBFLAB_GLOBAL_PRESSURE
    if(a.component==DIM&&b.component==DIM){
     term.pressure=true;term.derivatives.push_back(pressure_index(alpha));
    }else
#endif
    if(a.component<DIM&&b.component<DIM){
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
#ifdef RBFLAB_GLOBAL_PRESSURE
   pressure::Shared pressure;
#endif
   PairValues(const Worker& w,const std::vector<Real>& x,int i,const std::vector<Real>& y,int j):
    z(displacement(w,x,i,y,j)),velocity(z,w.params)
#ifdef RBFLAB_GLOBAL_PRESSURE
    ,pressure(z,w.pressure_params)
#endif
   {}
  };
  Real planned_kernel(PairValues& pair,const KernelPlan& plan)const{
   Real sum;
   for(const auto& term:plan){
    Real radial;
#ifdef RBFLAB_GLOBAL_PRESSURE
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
#ifdef RBFLAB_GLOBAL_PRESSURE
    if(a.component==DIM&&b.component==DIM){
     radial=pressure_value(z,alpha,pressure_params);
    }else
#endif
    if(a.component<DIM&&b.component<DIM){
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
   Real::global_bits=Real::bits;
#endif
   digits=requested_digits;threads=requested_threads;symmetric=sym;gram_matrix=symmetric;n=nn;m=mm;
   Eigen::setNbThreads(threads);
   GlobalDenseLU::initialize(threads);
   h=std::make_unique<Real>(number(in));if(!(Real(0)<*h))throw std::runtime_error("Invalid kernel scale");
   unit_scale=(*h==Real(1));
   params=std::vector<Real>(PARAMETERS);for(auto& v:params)v=number(in);
#ifdef RBFLAB_GLOBAL_FIELDS
   if(!symmetric)throw std::runtime_error("Field collocation requires symmetric scheme");
#ifdef RBFLAB_GLOBAL_PRESSURE
   pressure_params=std::vector<Real>(PRESSURE_PARAMETERS);for(auto& v:pressure_params)v=number(in);
#endif
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
#ifndef RBFLAB_GLOBAL_PRESSURE
     if(term.component==DIM)throw std::runtime_error("Pressure component unavailable");
#endif
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
   has_mass=false;mass_rows.clear();stiffness.resize(0,0);mass.resize(0,0);shift_factors.clear();active_shift.clear();
   initial_matrix.resize(0,0);initial_factor.reset();evolution_previous.clear();evolution_previous2.clear();
   clear_evaluation_cache();evaluation_cache_hits=evaluation_cache_misses=0;
   profile_global=false;global_seconds={};global_timings.clear();
#ifdef RBFLAB_GLOBAL_DATA
   data_rows.clear();
#endif
  }
  void assemble(bool factor_now=true){if(n<1)throw std::runtime_error("INIT required");
   clear_evaluation_cache();active_shift.clear();shift_factors.clear();
   built=false;
   dirty=true;
   lu.reset();
   last_residual.reset();
   condition_estimate.reset();
   int N=size();
   matrix=Dense::Zero(N,N);
   std::vector<std::string> errors(n);
   // Each entry has a distinct destination; global MPFR precision is immutable
   // for the duration of this command, including Eigen/OpenMP worker threads.
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
   built=true;gram_matrix=symmetric;dirty=true;if(factor_now)factor();
  }
  Dense operator_matrix(const std::vector<Op>& rows,const std::vector<int>& indices={}){
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
     int row=indices.empty()?i:indices[i];
     next(row,j)=eigen_value(plan?planned_kernel(source,row,trial,j,*plan):kernel(source,row,rows[i],trial,j,right[j]));}
    int row=indices.empty()?i:indices[i];
    for(int p=0;p<m;++p)next(row,n+p)=eigen_value(polynomial(source,row,rows[i],p));
   }catch(const std::exception& e){errors[i]=e.what();}}
   for(const auto& e:errors)if(!e.empty())throw std::runtime_error(e);
   return next;
  }
  void assemble_mass(const std::vector<Op>& rows,const std::vector<int>& indices={}){
   Dense next=operator_matrix(rows,indices);
   stiffness=matrix;mass.swap(next);mass_rows=indices.empty()?rows:std::vector<Op>{};has_mass=true;shift_factors.clear();active_shift.clear();
  }
  bool shift(const Real& alpha){
   if(!has_mass)throw std::runtime_error("MASS required");
   std::string key=alpha.str(digits+8);
   if(key==active_shift&&!dirty){last_residual.reset();return false;}
   matrix=stiffness+eigen_value(alpha)*mass;
   last_residual.reset();condition_estimate.reset();
   auto found=shift_factors.find(key);
   const bool fresh=found==shift_factors.end();
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
   active_shift=key;
   return fresh;
  }
  ShiftFactor factor_dense(const Dense& input,bool diagonal){auto phase=global_start();int N=size();ShiftFactor saved;saved.balanced=input;saved.scales.assign(N,Real(1));
   for(int i=0;i<N;++i)for(int j=0;j<N;++j)require_finite(Real(input(i,j)));
   for(int i=0;i<N;++i){Real mag;
    if(diagonal){mag=abs(Real(input(i,i)));if(!(Real(0)<mag))throw std::runtime_error("Nonpositive symmetric diagonal");}
    else {for(int j=0;j<N;++j){Real v=abs(Real(input(i,j)));if(mag<v)mag=v;}if(mag==Real(0))throw std::runtime_error("Zero matrix row");}
    saved.scales[i]=Real(1)/sqrt(mag);
   }
   for(int i=0;i<N;++i)for(int j=0;j<N;++j){
    Real value=saved.scales[i]*Real(input(i,j))*saved.scales[j];
    require_finite(value);
    saved.balanced(i,j)=eigen_value(value);
   }
   global_phase(1,phase);phase=global_start();
   saved.lu=std::make_shared<GlobalDenseLU>(saved.balanced);
   global_phase(2,phase);
   return saved;
  }
  void factor(){if(!built)throw std::runtime_error("BUILD required");if(!dirty)return;
   bool diagonal=gram_matrix&&m==0;
#ifndef RBFLAB_GLOBAL_FIELDS
   if(has_mass)diagonal=false; // Scalar MOL shifts are generally not Gram matrices.
#endif
   auto saved=factor_dense(matrix,diagonal);
   balanced=std::move(saved.balanced);row_scale=std::move(saved.scales);lu=std::move(saved.lu);dirty=false;
  }
  std::vector<Real> initial_solve(const std::vector<Real>& b){
   if(!built||n<1||initial_matrix.rows()!=size())throw std::runtime_error("INITIAL_MATRIX required");
   if(!initial_factor)initial_factor=std::make_unique<ShiftFactor>(factor_dense(initial_matrix,false));
   Dense rhs(size(),1);for(int i=0;i<size();++i)rhs(i,0)=eigen_value(initial_factor->scales[i]*b[i]);
   Dense x=initial_factor->lu->solve(rhs);std::vector<Real> result(size());
   for(int i=0;i<size();++i){result[i]=initial_factor->scales[i]*Real(x(i,0));require_finite(result[i]);}
   return result;
  }
#ifdef RBFLAB_GLOBAL_DATA
  std::vector<Real> data_values(const Real& time,int program=-1){
   if(n<1||(program<0&&int(data_rows.size())!=n))throw std::runtime_error("DATA_INIT required");
   if(program>=DATA_PROGRAMS)throw std::runtime_error("Invalid data expression index");
   std::vector<Real> values(size());std::vector<std::string> errors(n);
#pragma omp parallel for num_threads(threads) schedule(static) if(n>=128)
   for(int i=0;i<n;++i){try{
#ifndef RBFLAB_DOUBLE
    Real::bits=Real::precision();
#endif
    std::vector<Real> point(DIM);for(int d=0;d<DIM;++d)point[d]=source[i*DIM+d];
    values[i]=data_value(program<0?data_rows[i]:program,point,time);require_finite(values[i]);
   }catch(const std::exception& e){errors[i]=e.what();}}
   for(const auto& error:errors)if(!error.empty())throw std::runtime_error(error);
   return values;
  }
#endif
  std::pair<std::vector<Real>,Real> solve(const std::vector<Real>& b){factor();
   auto phase=global_start();
   int N=size();
   Dense rhs(N,1);
   for(int i=0;i<N;++i)rhs(i,0)=eigen_value(row_scale[i]*b[i]);
   global_phase(3,phase);phase=global_start();
   Dense x=lu->solve(rhs);
   global_phase(4,phase);phase=global_start();
   std::vector<Real> result(N);
   for(int i=0;i<N;++i){
    result[i]=row_scale[i]*Real(x(i,0));
    require_finite(result[i]);
   }
   global_phase(5,phase);phase=global_start();
   Real residual,den;
   auto products=multiply(matrix,result);
   for(int i=0;i<N;++i){Real value=products[i]-b[i];
    require_finite(value);
    if(residual<abs(value))residual=abs(value);
    if(den<abs(b[i]))den=abs(b[i]);
   }
   if(!(den==Real(0)))residual/=den;
   require_finite(residual);
   last_residual=std::make_unique<Real>(residual);
   global_phase(6,phase);
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
    Dense z=lu->solve(signs,true);
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
   const bool timed=profile_global&&(cmd=="SOLVE"||cmd=="MASS_MV"||cmd=="SHIFT"||cmd=="RHS_DATA");
   auto command_start=timed?ProfileClock::now():ProfileClock::time_point{};
   if(timed)global_seconds={};
   if(cmd=="HELLO"){end(in);
    out<<"OK "<<DIM<<" "<<PARAMETERS;
   }
   else if(cmd=="INIT"){init(in);out<<"OK";}
   else if(cmd=="RESET"){end(in);*this=Worker();out<<"OK";}
   else if(cmd=="BUILD"){end(in);assemble();out<<"OK "<<size();}
   else if(cmd=="BUILD_MOL"){end(in);assemble(false);out<<"OK "<<size();}
   else if(cmd=="MASS_INDEXED"){
    int count=integer(in);if(count<0||count>n)throw std::runtime_error("Invalid mass count");
    std::vector<int> indices(count);std::vector<bool> seen(n,false);std::vector<Op> rows(count);
    for(int i=0;i<count;++i){indices[i]=integer(in);if(indices[i]<0||indices[i]>=n||seen[indices[i]])throw std::runtime_error("Invalid mass row index");seen[indices[i]]=true;rows[i]=read_op(in);}
    end(in);assemble_mass(rows,indices);out<<"OK";
   }
   else if(cmd=="INITIAL_MATRIX"){
    int count=integer(in);if(count!=n)throw std::runtime_error("Invalid initial row count");
    std::vector<Op> rows(count);for(auto& row:rows)row=read_op(in);end(in);
    initial_matrix=operator_matrix(rows);if(m)initial_matrix.bottomRows(m)=(has_mass?stiffness:matrix).bottomRows(m);
    initial_factor.reset();evolution_previous.clear();evolution_previous2.clear();out<<"OK";
   }
   else if(cmd=="INITIAL_EXPORT"){
    end(in);if(!built||n<1||initial_matrix.rows()!=size())throw std::runtime_error("INITIAL_MATRIX required");
    out<<"OK "<<size();for(int i=0;i<size();++i)for(int j=0;j<size();++j)out<<" "<<Real(initial_matrix(i,j)).str(digits);
   }
   else if(cmd=="INITIAL_REPLACE"){
    if(!built||n<1)throw std::runtime_error("BUILD required");
    int count=integer(in);if(count!=size())throw std::runtime_error("Invalid initial matrix shape");
    Dense next(count,count);for(int i=0;i<count;++i)for(int j=0;j<count;++j)next(i,j)=eigen_value(number(in));end(in);
    initial_matrix.swap(next);initial_factor.reset();evolution_previous.clear();evolution_previous2.clear();out<<"OK";
   }
   else if(cmd=="EVOLVE_INIT"){
    std::vector<Real> b(size());for(auto& value:b)value=number(in);end(in);bool fresh=!initial_factor;
    auto values=initial_solve(b);auto product=multiply(initial_matrix,values);Real residual,den;
    for(int i=0;i<size();++i){residual=std::max(residual,abs(product[i]-b[i]));den=std::max(den,abs(b[i]));}
    if(!(den==Real(0)))residual/=den;
    evolution_previous=values;evolution_previous2.clear();
    out<<"OK "<<fresh<<" "<<size();for(const auto& value:values)out<<" "<<value.str(digits+8);out<<" "<<residual.str(digits+8);
   }
   else if(cmd=="TIME_STEP"){
    if(!has_mass)throw std::runtime_error("MASS required");
    Real time=number(in),alpha=number(in);int count=integer(in);
    if(count<1||count>2||int(evolution_previous.size())!=size()||(count==2&&int(evolution_previous2.size())!=size()))throw std::runtime_error("Invalid evolution history");
    std::vector<Real> weights(count);for(auto& value:weights)value=number(in);int native_data=integer(in);
    if(native_data!=0&&native_data!=1)throw std::runtime_error("Invalid data mode");
    std::vector<Real> rhs(size());
    if(native_data){
#ifdef RBFLAB_GLOBAL_DATA
      rhs=data_values(time);
#else
      throw std::runtime_error("Native data unavailable");
#endif
    }else for(auto& value:rhs)value=number(in);
    end(in);std::vector<Real> history(size());
    for(int i=0;i<size();++i){history[i]=weights[0]*evolution_previous[i];if(count==2)history[i]+=weights[1]*evolution_previous2[i];}
    auto product=multiply(mass,history);for(int i=0;i<size();++i)rhs[i]+=product[i];
    bool fresh=shift(alpha);auto answer=solve(rhs);
    out<<"OK "<<fresh<<" "<<shift_factors.size()<<" "<<size();
    for(const auto& value:answer.first)out<<" "<<value.str(digits+8);
    for(int i=0;i<size();++i){Real derivative=alpha*answer.first[i]-history[i];require_finite(derivative);out<<" "<<derivative.str(digits+8);}
    out<<" "<<answer.second.str(digits+8);
    evolution_previous2=std::move(evolution_previous);evolution_previous=std::move(answer.first);
   }
#ifdef RBFLAB_GLOBAL_DATA
   else if(cmd=="DATA_INIT"){
    int count=integer(in);if(count!=n)throw std::runtime_error("Invalid data row count");
    std::vector<int> next(count);
    for(auto& id:next){id=integer(in);if(id<0||id>=DATA_PROGRAMS)throw std::runtime_error("Invalid data expression index");}
    end(in);data_rows.swap(next);out<<"OK";
   }
   else if(cmd=="RHS_DATA"){
    auto phase=global_start();Real time=number(in);end(in);global_phase(0,phase);
    auto values=data_values(time);
    phase=global_start();out<<"OK "<<size();for(const auto& value:values)out<<" "<<value.str(digits+8);
    global_phase(8,phase);
   }
   else if(cmd=="SAMPLE_DATA"){
    int program=integer(in);if(program<0)throw std::runtime_error("Invalid data expression index");
    Real time=number(in);end(in);auto values=data_values(time,program);
    out<<"OK "<<n;for(int i=0;i<n;++i)out<<" "<<values[i].str(digits+8);
   }
#endif
   else if(cmd=="MASS"){int count=integer(in);
    if(count<0||count>n)throw std::runtime_error("Invalid mass count");
    std::vector<Op> rows(count);for(auto& row:rows)row=read_op(in);end(in);
    assemble_mass(rows);out<<"OK";
   }
   else if(cmd=="SHIFT"){auto phase=global_start();Real alpha=number(in);end(in);global_phase(0,phase);bool fresh=shift(alpha);out<<"OK "<<fresh<<" "<<shift_factors.size();}
   else if(cmd=="MASS_MV"){
    if(!has_mass)throw std::runtime_error("MASS required");
    auto phase=global_start();
    std::vector<Real> coefficients(size());for(auto& v:coefficients)v=number(in);end(in);
    global_phase(0,phase);phase=global_start();
    auto values=multiply(mass,coefficients);global_phase(7,phase);phase=global_start();out<<"OK "<<size();
    for(const auto& v:values)out<<" "<<v.str(digits);
    global_phase(8,phase);
   }
   else if(cmd=="MASS_EXPORT"){
    end(in);if(!has_mass)throw std::runtime_error("MASS required");
    out<<"OK "<<size();for(int i=0;i<size();++i)for(int j=0;j<size();++j)out<<" "<<Real(mass(i,j)).str(digits);
   }
   else if(cmd=="SOLVE"){auto phase=global_start();int count=integer(in);
    if(count<1||count>100000)throw std::runtime_error("Invalid RHS count");
    std::vector<std::vector<Real>> rhs(count,std::vector<Real>(size()));
    for(auto& b:rhs)for(auto& v:b)v=number(in);
    end(in);
    global_phase(0,phase);
    out<<"OK "<<count;
    for(auto& b:rhs){auto answer=solve(b);
     phase=global_start();
     for(auto& x:answer.first)out<<" "<<x.str(digits);
     out<<" "<<answer.second.str(digits);
     global_phase(8,phase);
    }
   }
   else if(cmd=="GLOBAL_PROFILE"){int enabled=integer(in);end(in);
    if(enabled!=0&&enabled!=1)throw std::runtime_error("GLOBAL_PROFILE expects 0 or 1");
    profile_global=enabled;global_seconds={};global_timings.clear();out<<"OK";
   }
   else if(cmd=="GLOBAL_TIMING"){end(in);out<<"OK "<<global_timings.size()<<std::setprecision(17);
    for(const auto& entry:global_timings){out<<" "<<entry.first<<" "<<entry.second.first;
     for(double seconds:entry.second.second)out<<" "<<seconds;
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
   else if(cmd=="EVAL_CACHE_INFO"){end(in);
    out<<"OK "<<evaluation_cache_hits<<" "<<evaluation_cache_misses<<" "<<evaluation_cache_bytes<<" "<<evaluation_cache.size();
   }
   else if(cmd=="RUNTIME_INFO"){end(in);
    out<<"OK "<<Eigen::nbThreads()<<" ";
#ifdef RBFLAB_DOUBLE
    out<<53;
#else
    out<<Real::precision();
#endif
   }
   else if(cmd=="SOLVER_INFO"){end(in);
#ifdef RBFLAB_MPLAPACK
    int count=0,minimum=1000000,maximum=0;
#pragma omp parallel num_threads(threads) reduction(+:count) reduction(min:minimum) reduction(max:maximum)
    {
     mpfrxx::mpfr_class scratch(0);
     int bits=mpfr_get_prec(scratch.get_mpfr_t());
     minimum=std::min(minimum,bits);maximum=std::max(maximum,bits);++count;
    }
    out<<"OK mplapack "<<count<<" "<<minimum<<" "<<maximum;
#else
    out<<"OK eigen "<<Eigen::nbThreads();
#endif
   }
   else if(cmd=="EVAL"){
    auto parse_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
    profile_available=false;
    int count=integer(in);if(count<0||count>100000)throw std::runtime_error("Invalid evaluation count");
    std::vector<Real> points(count*DIM);for(auto& v:points)v=number(in);
    std::vector<Op> ops(count);for(auto& op:ops)op=read_op(in);
    const std::string cache_key=line.substr(0,static_cast<size_t>(in.tellg()));
    std::vector<Real> coeff(size());for(auto& v:coeff)v=number(in);end(in);
    bool mass_query=has_mass&&count==int(mass_rows.size())&&points.size()<=source.size()
      &&std::equal(points.begin(),points.end(),source.begin());
    if(mass_query)for(int i=0;i<count;++i)if(!same_operator(ops[i],mass_rows[i])){mass_query=false;break;}
    auto cached=std::find_if(evaluation_cache.begin(),evaluation_cache.end(),
      [&](const EvaluationCache& item){return item.key==cache_key;});
    size_t entry_bytes=sizeof(EigenScalar);
#ifndef RBFLAB_DOUBLE
    // Include limbs and an allowance for allocator bookkeeping per MPFR value.
    entry_bytes+=sizeof(mp_limb_t)*size_t((Real::precision()+GMP_NUMB_BITS-1)/GMP_NUMB_BITS)+2*sizeof(void*);
#endif
    const size_t cache_bytes=size_t(count)*size_t(size())*entry_bytes+cache_key.size();
    if(mass_query||cached!=evaluation_cache.end()||(count>0&&cache_bytes<=evaluation_cache_limit)){
     if(profile_eval)eval_seconds[0]=profile_seconds(parse_start);
     auto compute_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
     std::vector<Real> values;
     if(mass_query){++evaluation_cache_hits;values=multiply(mass.topRows(count),coeff);}
     else {
      if(cached==evaluation_cache.end()){
       ++evaluation_cache_misses;
       while(!evaluation_cache.empty()&&evaluation_cache_bytes+cache_bytes>evaluation_cache_limit){
        evaluation_cache_bytes-=evaluation_cache.back().bytes;evaluation_cache.pop_back();
       }
       EvaluationMatrix next(count,size());
       auto plans=prepare_plans(ops,right);auto groups=group_points(points,8);
       std::vector<std::string> errors(count);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
       for(int g=0;g<int(groups.size());++g){try{
        const auto& group=groups[g];
        if(!plans.plans.empty())for(const auto& sources:trial_groups){
         PairValues pair(*this,points,group[0],trial,sources[0]);
         for(int i:group)for(int j:sources)next(i,j)=eigen_value(planned_kernel(pair,*plans.find(i,j)));
        }
        else for(int i:group)for(int j=0;j<n;++j)next(i,j)=eigen_value(kernel(points,i,ops[i],trial,j,right[j]));
        for(int i:group)for(int p=0;p<m;++p)next(i,n+p)=eigen_value(polynomial(points,i,ops[i],p));
       }catch(const std::exception& e){errors[groups[g][0]]=e.what();}}
       for(const auto& error:errors)if(!error.empty())throw std::runtime_error(error);
       evaluation_cache.push_front({cache_key,std::move(next),cache_bytes});evaluation_cache_bytes+=cache_bytes;
       cached=evaluation_cache.begin();
      }else{++evaluation_cache_hits;evaluation_cache.splice(evaluation_cache.begin(),evaluation_cache,cached);}
      values=multiply(cached->matrix,coeff);
     }
     if(profile_eval)eval_seconds[1]=profile_seconds(compute_start);
     auto format_start=profile_eval?ProfileClock::now():ProfileClock::time_point{};
     out<<"OK "<<count;for(const auto& value:values)out<<" "<<value.str(digits);
     if(profile_eval){eval_seconds[2]=profile_seconds(format_start);profile_available=true;}
     return;
    }
    ++evaluation_cache_misses;
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
    gram_matrix=false;
    if(has_mass){stiffness(i,j)=eigen_value(v);matrix=stiffness;shift_factors.clear();active_shift.clear();}
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
    gram_matrix=false;
    if(has_mass){stiffness=matrix;shift_factors.clear();active_shift.clear();}
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
   if(timed){global_seconds[9]=profile_seconds(command_start);auto& entry=global_timings[cmd];++entry.first;
    for(size_t i=0;i<global_seconds.size();++i)entry.second[i]+=global_seconds[i];
   }
  }
 };
}
namespace PartitionUnity {
 using namespace GlobalWorker;
 struct Patch {std::unique_ptr<Worker> worker;std::vector<std::vector<Real>> coefficients;};
 struct State {
  std::vector<Patch> patches;
  std::vector<Real> centers,radii;
  std::vector<std::vector<Real>> pending_df;
  std::vector<Real> df_shifts;
  int components=0,digits=17;
  bool df_mode=false;
  void reset(){patches.clear();centers.clear();radii.clear();pending_df.clear();df_shifts.clear();components=0;df_mode=false;}
  bool active()const{return !patches.empty();}
#ifdef RBFLAB_GLOBAL_FIELDS
  Real df_potential(const Worker& patch,const std::vector<Real>& coeff,const std::vector<Real>& point)const{
   if(DIM!=2)throw std::runtime_error("DFPU requires 2D");
   Real result;
   for(int node=0;node<patch.n/2;++node){
    std::array<Real,DIM> z;
    for(int d=0;d<DIM;++d)z[d]=(point[d]-patch.trial[node*2*DIM+d])/(*patch.h);
    std::array<int,DIM> dx{};dx[0]=1;
    std::array<int,DIM> dy{};dy[1]=1;
    result+=(coeff[2*node]*radial_value(z,dy,patch.params)
             -coeff[2*node+1]*radial_value(z,dx,patch.params))/(*patch.h);
   }
   for(int col=0;col<patch.m;++col){
    Real amplitude=coeff[patch.n+col];
    for(const auto& term:patch.columns[col]){
     Real monomial=amplitude*term.coefficient;
     int px=term.power[0],py=term.power[1];
     Real zx=(point[0]-patch.center[0])/patch.scale[0];
     Real zy=(point[1]-patch.center[1])/patch.scale[1];
     if(term.component==1){
      monomial*=patch.scale[0]*pow(zx,px+1)*pow(zy,py)/Real(px+1);
      result+=monomial;
     }else if(term.component==0&&px==0){
      monomial*=patch.scale[1]*pow(zy,py+1)/Real(py+1);
      result-=monomial;
     }
    }
   }
   return result;
  }
  Real df_local(const Worker& patch,const std::vector<Real>& coeff,const std::vector<Real>& point,int component,int axis)const{
   Term term;term.c=Real(1);term.component=component;term.alpha={0,0};
   if(axis>=0)term.alpha[axis]=1;
   Op op={term};Real value;
   for(int k=0;k<patch.n;++k)value+=coeff[k]*patch.kernel(point,0,op,patch.trial,k,patch.right[k]);
   for(int k=0;k<patch.m;++k)value+=coeff[patch.n+k]*patch.polynomial(point,0,op,k);
   return value;
  }
#endif
  void command(const std::string& line,std::ostream& out){
   std::istringstream in(line);std::string cmd;in>>cmd;
   if(cmd=="PUM_INIT"||cmd=="DFP_INIT"){
    int count=integer(in),cols=integer(in),requested_digits=integer(in);
    if(count<1||count>100000||cols<1||cols>100000||requested_digits<16||requested_digits>10000)
     throw std::runtime_error("Invalid PUM header");
#ifndef RBFLAB_DOUBLE
    Real::bits=mpfr_prec_t(std::ceil(requested_digits*std::log2(10.0))+8);
    Real::global_bits=Real::bits;
#endif
    std::vector<Real> next_centers(count*DIM),next_radii(count);
    for(auto& v:next_centers)v=number(in);
    for(auto& v:next_radii){v=number(in);if(!(Real(0)<v))throw std::runtime_error("Invalid patch radius");}
    end(in);reset();patches.resize(count);centers=std::move(next_centers);
    radii=std::move(next_radii);components=cols;digits=requested_digits;
    df_mode=cmd=="DFP_INIT";
    if(df_mode&&(DIM!=2||components!=1))throw std::runtime_error("DFPU requires planar vector mode");
    out<<"OK";
   }else if(cmd=="PUM_PATCH"||cmd=="DFP_PATCH"){
    int index=integer(in);if(index<0||index>=int(patches.size())||patches[index].worker)
     throw std::runtime_error("Invalid PUM patch index");
    std::string payload;std::getline(in,payload);
    auto next=std::make_unique<Worker>();
    std::istringstream init(payload);next->init(init);
    if(next->digits!=digits)throw std::runtime_error("PUM patch precision mismatch");
    next->assemble();
    patches[index].worker=std::move(next);out<<"OK "<<patches[index].worker->size();
   }else if(cmd=="DFP_PREPARE"){
#ifdef RBFLAB_GLOBAL_FIELDS
    if(!df_mode||!active())throw std::runtime_error("DFP_INIT required");
    std::vector<std::vector<Real>> candidate(patches.size());
    for(size_t j=0;j<patches.size();++j){
     if(!patches[j].worker)throw std::runtime_error("DFPU patches must be built");
     candidate[j].resize(patches[j].worker->size());
     for(auto& value:candidate[j])value=number(in);
    }
    end(in);
    for(size_t j=0;j<patches.size();++j)candidate[j]=patches[j].worker->solve(candidate[j]).first;
    pending_df=std::move(candidate);
    out<<"OK "<<pending_df.size();
    for(const auto& patch:pending_df){out<<" "<<patch.size();for(const auto& value:patch)out<<" "<<value.str(digits);}
#else
    throw std::runtime_error("DFPU requires vector worker");
#endif
   }else if(cmd=="DFP_COMMIT"){
    if(!df_mode||pending_df.size()!=patches.size())throw std::runtime_error("DFP_PREPARE required");
    std::vector<Real> shifts(patches.size());for(auto& value:shifts)value=number(in);
    end(in);
    for(size_t j=0;j<patches.size();++j)patches[j].coefficients={std::move(pending_df[j])};
    df_shifts=std::move(shifts);pending_df.clear();out<<"OK";
   }else if(cmd=="DFP_EVAL"){
#ifdef RBFLAB_GLOBAL_FIELDS
    if(!df_mode||df_shifts.size()!=patches.size())throw std::runtime_error("DFP_COMMIT required");
    int count=integer(in);if(count<0||count>100000)throw std::runtime_error("Invalid DFPU query count");
    std::vector<Real> points(count*DIM);for(auto& value:points)value=number(in);
    end(in);out<<"OK "<<count;
    for(int q=0;q<count;++q){
     std::vector<Real> point={points[q*DIM],points[q*DIM+1]};
     struct Weight {Real value;std::array<Real,2> gradient;std::array<std::array<Real,2>,2> hessian;};
     std::vector<Weight> raw(patches.size());Real total;
     std::array<Real,2> first{};std::array<std::array<Real,2>,2> second{};
     for(size_t j=0;j<patches.size();++j){
      std::array<Real,2> delta={point[0]-centers[j*2],point[1]-centers[j*2+1]};
      Real square=delta[0]*delta[0]+delta[1]*delta[1],radius=radii[j];
      if(!(square<radius*radius))continue;
      Real distance=sqrt(square),t=distance/radius,u=Real(1)-t;
      raw[j].value=pow(u,4)*(Real(4)*t+Real(1));total+=raw[j].value;
      Real radial=distance==Real(0)?Real(0):Real(-20)*t*pow(u,3)/(radius*distance);
      Real ddt=Real(-20)*u*u*(Real(1)-Real(4)*t)/(radius*radius);
      for(int a=0;a<2;++a){
       raw[j].gradient[a]=radial*delta[a];first[a]+=raw[j].gradient[a];
       for(int b=0;b<2;++b){
        raw[j].hessian[a][b]=distance==Real(0)?(a==b?ddt:Real(0)):
         (ddt-radial)*delta[a]*delta[b]/square+(a==b?radial:Real(0));
        second[a][b]+=raw[j].hessian[a][b];
       }
      }
     }
     if(!(Real(0)<total))throw std::runtime_error("Query point outside positive patch cover");
     Real potential,px,py,pxx,pxy,pyy;
     for(size_t j=0;j<patches.size();++j){if(!(Real(0)<raw[j].value))continue;
      Real w=raw[j].value/total;
      std::array<Real,2> dw;
      std::array<std::array<Real,2>,2> ddw;
      for(int a=0;a<2;++a)dw[a]=(raw[j].gradient[a]-w*first[a])/total;
      for(int a=0;a<2;++a)for(int b=0;b<2;++b)
       ddw[a][b]=(raw[j].hessian[a][b]-dw[a]*first[b]-dw[b]*first[a]-w*second[a][b])/total;
      const auto& patch=*patches[j].worker;const auto& coeff=patches[j].coefficients.at(0);
      Real psi=df_potential(patch,coeff,point)+df_shifts[j];
      Real ux=df_local(patch,coeff,point,0,-1),uy=df_local(patch,coeff,point,1,-1);
      Real uxx=df_local(patch,coeff,point,0,0),uyx=df_local(patch,coeff,point,1,0);
      Real uxy=df_local(patch,coeff,point,0,1),uyy=df_local(patch,coeff,point,1,1);
      potential+=w*psi;
      px+=dw[0]*psi+w*uy;py+=dw[1]*psi-w*ux;
      pxx+=ddw[0][0]*psi+Real(2)*dw[0]*uy+w*uyx;
      pxy+=ddw[0][1]*psi-dw[0]*ux+dw[1]*uy+w*uyy;
      pyy+=ddw[1][1]*psi-Real(2)*dw[1]*ux-w*uxy;
     }
     std::array<Real,7> result={potential,-py,px,-pxy,pxx,-pyy,pxy};
     for(const auto& value:result){require_finite(value);out<<" "<<value.str(digits);}
    }
#else
    throw std::runtime_error("DFPU requires vector worker");
#endif
   }else if(cmd=="PUM_REFIT"){
    if(!active())throw std::runtime_error("PUM_INIT required");
    std::vector<std::vector<std::vector<Real>>> replacements(patches.size());
    for(size_t j=0;j<patches.size();++j){
     if(!patches[j].worker)throw std::runtime_error("PUM patches must be built");
     int size=patches[j].worker->size();
     replacements[j].assign(components,std::vector<Real>(size));
     for(auto& rhs:replacements[j])for(auto& value:rhs)value=number(in);
    }
    end(in);
    std::vector<Real> residuals(patches.size());
    for(size_t j=0;j<patches.size();++j)
     for(auto& rhs:replacements[j]){
      auto answer=patches[j].worker->solve(rhs);
      if(residuals[j]<answer.second)residuals[j]=answer.second;
      rhs=std::move(answer.first);
     }
    for(size_t j=0;j<patches.size();++j){
     patches[j].coefficients=std::move(replacements[j]);
     patches[j].worker->last_residual=std::make_unique<Real>(residuals[j]);
    }
    out<<"OK "<<patches.size();
   }else if(cmd=="PUM_DIAG"){
    int with_condition=integer(in);end(in);
    if(with_condition!=0&&with_condition!=1)throw std::runtime_error("Invalid PUM condition flag");
    out<<"OK "<<patches.size();
    for(const auto& patch:patches){
     if(!patch.worker)throw std::runtime_error("PUM patches must be built");
     out<<" "<<(patch.worker->last_residual?patch.worker->last_residual->str(digits):"none");
     out<<" ";
     if(with_condition)out<<patch.worker->condition().str(digits);
     else out<<"none";
    }
   }else if(cmd=="PUM_EVAL"){
    int count=integer(in);if(count<0||count>100000||!active())throw std::runtime_error("Invalid PUM evaluation count");
    std::vector<Real> points(count*DIM);for(auto& v:points)v=number(in);
    Op op=read_op(in);end(in);
    for(const auto& term:op){int order=0;for(int a:term.alpha)order+=a;
     if(order>2)throw std::runtime_error("PUM derivatives above order two unsupported");}
    for(const auto& patch:patches)if(!patch.worker||int(patch.coefficients.size())!=components)
     throw std::runtime_error("PUM patches must be built and solved");
    out<<"OK "<<count<<" "<<components;
    for(int q=0;q<count;++q){
     struct Weight {Real value;std::array<Real,DIM> gradient;std::array<std::array<Real,DIM>,DIM> hessian;};
     std::vector<Weight> weights(patches.size());Real total;
     std::array<Real,DIM> gradient_total{};
     std::array<std::array<Real,DIM>,DIM> hessian_total{};
     for(size_t j=0;j<patches.size();++j){
      std::array<Real,DIM> delta;Real square;
      for(int a=0;a<DIM;++a){delta[a]=points[q*DIM+a]-centers[j*DIM+a];square+=delta[a]*delta[a];}
      Real radius=radii[j];if(!(square<radius*radius))continue;
      Real distance=sqrt(square),t=distance/radius,u=Real(1)-t;
      Real value=pow(u,4)*(Real(4)*t+Real(1));
      Real first=Real(-20)*t*pow(u,3);
      Real second=Real(-20)*u*u*(Real(1)-Real(4)*t);
      weights[j].value=value;total+=value;
      for(int a=0;a<DIM;++a){
       weights[j].gradient[a]=distance==Real(0)?Real(0):first*delta[a]/(radius*distance);
       gradient_total[a]+=weights[j].gradient[a];
       for(int b=0;b<DIM;++b){
        Real radial=distance==Real(0)?Real(0):first/(radius*distance);
        weights[j].hessian[a][b]=distance==Real(0)?
         (a==b?second/(radius*radius):Real(0)):
         (second/(radius*radius)-radial)*delta[a]*delta[b]/square+(a==b?radial:Real(0));
        hessian_total[a][b]+=weights[j].hessian[a][b];
       }
      }
     }
     if(!(Real(0)<total))throw std::runtime_error("Query point outside positive patch cover");
     std::vector<Real> result(components);
     for(size_t j=0;j<patches.size();++j){if(!(Real(0)<weights[j].value))continue;
      auto& patch=*patches[j].worker;
      Real w=weights[j].value/total;
      std::array<Real,DIM> dw;
      std::array<std::array<Real,DIM>,DIM> ddw;
      for(int a=0;a<DIM;++a)dw[a]=(weights[j].gradient[a]-w*gradient_total[a])/total;
      for(int a=0;a<DIM;++a)for(int b=0;b<DIM;++b)
       ddw[a][b]=(weights[j].hessian[a][b]-dw[a]*gradient_total[b]-dw[b]*gradient_total[a]-w*hessian_total[a][b])/total;
      for(const auto& term:op){
       std::vector<int> beta(DIM,0);
       auto evaluate_beta=[&](const std::vector<int>& b){
        int order=0;Real factor=term.c;
        for(int a=0;a<DIM;++a){order+=b[a];if(term.alpha[a]==2&&b[a]==1)factor*=Real(2);}
        if(order==0)factor*=w;
        else if(order==1){for(int a=0;a<DIM;++a)if(b[a])factor*=dw[a];}
        else {int first_axis=-1;for(int a=0;a<DIM;++a)for(int k=0;k<b[a];++k){if(first_axis<0)first_axis=a;else factor*=ddw[first_axis][a];}}
        Term local;local.c=Real(1);local.alpha.resize(DIM);
        for(int a=0;a<DIM;++a)local.alpha[a]=term.alpha[a]-b[a];
        Op local_op={local};
        for(int c=0;c<components;++c){
         Real value;
         for(int k=0;k<patch.n;++k)value+=patches[j].coefficients[c][k]*patch.kernel(points,q,local_op,patch.trial,k,patch.right[k]);
         for(int k=0;k<patch.m;++k)value+=patches[j].coefficients[c][patch.n+k]*patch.polynomial(points,q,local_op,k);
         result[c]+=factor*value;
        }
       };
       // Each derivative order is at most two; enumerate beta <= alpha.
       while(true){evaluate_beta(beta);int a=0;
        for(;a<DIM;++a){if(beta[a]<term.alpha[a]){++beta[a];break;}beta[a]=0;}
        if(a==DIM)break;
       }
      }
     }
     for(const auto& value:result){require_finite(value);out<<" "<<value.str(digits);}
    }
   }else throw std::runtime_error("Unknown PUM command");
  }
 };
}
int main(){GlobalWorker::Worker worker;PartitionUnity::State pum;
 std::string line;
 while(std::getline(std::cin,line)){try{std::ostringstream out;
   if(line.rfind("PUM_",0)==0||line.rfind("DFP_",0)==0)pum.command(line,out);
   else {if(line=="RESET"||line.rfind("INIT ",0)==0)pum.reset();worker.command(line,out);}
   std::cout<<out.str()<<std::endl;
   if(line=="QUIT")break;
  }
  catch(const std::exception& e){std::cout<<"ERR "<<e.what()<<std::endl;
  }
 }
 return 0;
}
