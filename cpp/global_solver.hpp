#pragma once
#include "eigen_solver.hpp"
#include <omp.h>
#include <cstdlib>
#ifdef RBFLAB_MPLAPACK
#ifdef RBFLAB_DOUBLE
#error MPLAPACK global solver requires MPFR arithmetic
#endif
#include <mpblas_mpfr.h>
#include <mplapack_mpfr.h>
#endif

// Assembly, balancing and residuals keep RBFLAB's Real/Eigen storage. Only
// the owned dense factor and its normal/transposed solves vary by engine.
struct GlobalDenseLU {
 using Dense=EigenLocalLU::Dense;
#ifdef RBFLAB_MPLAPACK
 using Number=mpfrxx::mpfr_class;
 std::vector<Number> factors;
 std::vector<mplapackint> pivots;
 mplapackint size;
 static void initialize(int threads){
  if(!mpfr_buildopt_tls_p())throw std::runtime_error("MPLAPACK requires MPFR TLS support");
  const auto bits=std::to_string(Real::precision());
#ifdef _WIN32
  if(_putenv_s("MPFRXX_DEFAULT_PRECISION_BITS",bits.c_str()))
#else
  if(setenv("MPFRXX_DEFAULT_PRECISION_BITS",bits.c_str(),1))
#endif
   throw std::runtime_error("Cannot configure MPLAPACK precision defaults");
  omp_set_dynamic(0);omp_set_max_active_levels(1);omp_set_num_threads(threads);
#pragma omp parallel num_threads(threads)
  {
   mpfrxx::set_default_precision_bits(Real::precision());
   mpfrxx::set_default_rounding_mode(MPFR_RNDN);
  }
 }
 explicit GlobalDenseLU(const Dense& a):size(a.rows()){
  factors.resize(size_t(size)*size);pivots.resize(size);
  for(int j=0;j<size;++j)for(int i=0;i<size;++i)
   mpfr_set(factors[i+size_t(j)*size].get_mpfr_t(),a(i,j).v,MPFR_RNDN);
  mplapackint info=0;
  Rgetrf(size,size,factors.data(),size,pivots.data(),info);
  if(info<0)throw std::runtime_error("MPLAPACK Rgetrf rejected an argument");
  if(info>0)throw std::runtime_error("Singular global matrix at selected precision");
  for(const auto& value:factors)
   if(!mpfr_number_p(value.get_mpfr_t()))throw std::runtime_error("Nonfinite native arithmetic result");
 }
 Dense solve(const Dense& b,bool transpose=false){
  std::vector<Number> x(size_t(size)*b.cols());
  for(int j=0;j<b.cols();++j)for(int i=0;i<size;++i)
   mpfr_set(x[i+size_t(j)*size].get_mpfr_t(),b(i,j).v,MPFR_RNDN);
  mplapackint info=0;
  Rgetrs(transpose?"T":"N",size,b.cols(),factors.data(),size,pivots.data(),x.data(),size,info);
  if(info)throw std::runtime_error("MPLAPACK Rgetrs rejected an argument");
  Dense result(size,b.cols());
  for(int j=0;j<b.cols();++j)for(int i=0;i<size;++i)
   mpfr_set(result(i,j).v,x[i+size_t(j)*size].get_mpfr_t(),MPFR_RNDN);
  return result;
 }
#else
 Eigen::PartialPivLU<Dense> factor;
 static void initialize(int){}
 explicit GlobalDenseLU(const Dense& a):factor(a){
  for(int i=0;i<a.rows();++i)for(int j=0;j<a.cols();++j){
#ifdef RBFLAB_DOUBLE
   if(!std::isfinite(factor.matrixLU()(i,j)))throw std::runtime_error("Nonfinite native arithmetic result");
#else
   if(!mpfr_number_p(factor.matrixLU()(i,j).v))throw std::runtime_error("Nonfinite native arithmetic result");
#endif
  }
  for(int i=0;i<a.rows();++i)if(factor.matrixLU()(i,i)==EigenScalar(0))
   throw std::runtime_error("Singular global matrix at selected precision");
 }
 Dense solve(const Dense& b,bool transpose=false){
  if(transpose)return factor.transpose().solve(b);
  return factor.solve(b);
 }
#endif
};
