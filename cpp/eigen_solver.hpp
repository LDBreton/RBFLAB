#pragma once
// Local methods parallelize across stencils. Global workers instead allow
// Eigen to parallelize sufficiently large dense products/factorizations.
#ifndef RBFLAB_GLOBAL_DENSE
#define EIGEN_DONT_PARALLELIZE
#endif
#include "real.hpp"
#ifndef RBFLAB_DOUBLE
inline bool operator!=(const Real&a,const Real&b){return !(a==b);}
inline bool operator<=(const Real&a,const Real&b){return !(a>b);}
inline bool operator>=(const Real&a,const Real&b){return !(a<b);}
inline Real abs2(const Real&a){return a*a;}
inline Real conj(const Real&a){return a;}
inline Real real(const Real&a){return a;}
inline Real imag(const Real&){return Real(0);}
#endif
#include <Eigen/Core>
#ifndef RBFLAB_DOUBLE
namespace Eigen {
template<> struct NumTraits<::Real>:GenericNumTraits<::Real> {
 using Real=::Real; using NonInteger=::Real; using Nested=::Real; using Literal=::Real;
 enum {IsComplex=0,IsInteger=0,IsSigned=1,RequireInitialization=1,ReadCost=10,AddCost=20,MulCost=40};
 static Real epsilon(){Real x(1);mpfr_div_2si(x.v,x.v,::Real::precision()-1,MPFR_RNDN);return x;}
 static Real dummy_precision(){return sqrt(epsilon());}
 static int digits10(){return int((::Real::precision()-1)*0.3010299956639812);}
};
}
#if defined(RBFLAB_GLOBAL_DENSE) && !defined(RBFLAB_MPFR_GEBP_REFERENCE)
// Eigen's scalar panel update otherwise constructs a product and a sum for
// every multiply-add. Reuse its scratch scalar and accumulator instead,
// retaining separate round-to-nearest multiplication and addition.
namespace Eigen { namespace internal {
template<> template<typename LhsPacketType, typename RhsPacketType,
                     typename AccPacketType, typename LaneIdType>
EIGEN_STRONG_INLINE void
gebp_traits<::Real,::Real,false,false,Architecture::Target,GEBPPacketFull>::madd(
    const LhsPacketType& a,const RhsPacketType& b,AccPacketType& c,
    RhsPacketType& tmp,const LaneIdType&) const {
  mpfr_mul(tmp.v,a.v,b.v,MPFR_RNDN);
  mpfr_add(c.v,c.v,tmp.v,MPFR_RNDN);
}
}}
#endif
#endif
#include <Eigen/LU>
#include <vector>
#ifdef RBFLAB_DOUBLE
using EigenScalar=double;
inline double eigen_value(const Real&x){return x.v;}
#else
using EigenScalar=Real;
inline const Real& eigen_value(const Real&x){return x;}
#endif
// Own the factorization once and reuse it for all reconstruction/diagnostic RHS.
struct EigenLocalLU {
 using Dense=Eigen::Matrix<EigenScalar,Eigen::Dynamic,Eigen::Dynamic>;
 using Rows=std::vector<std::vector<Real>>;
 Eigen::PartialPivLU<Dense> factor;
 static Dense dense(const Rows& rows){
  Dense a(rows.size(),rows.at(0).size());
  for(int i=0;i<a.rows();++i)for(int j=0;j<a.cols();++j)a(i,j)=eigen_value(rows[i][j]);
  return a;
 }
 explicit EigenLocalLU(const Rows& a):factor(dense(a)){
  for(int i=0;i<factor.matrixLU().rows();++i)
   if(factor.matrixLU()(i,i)==EigenScalar(0))throw std::runtime_error("Singular local matrix at selected precision");
 }
 Rows solve(const Rows& b)const{
  Dense x=factor.solve(dense(b));
  Rows result(x.rows(),std::vector<Real>(x.cols()));
  for(int i=0;i<x.rows();++i)for(int j=0;j<x.cols();++j)result[i][j]=Real(x(i,j));
  return result;
 }
};

#include <memory>
#ifdef RBFLAB_DOUBLE
#include <Eigen/SVD>
#endif
struct EigenLocalSolver {
 using Rows=EigenLocalLU::Rows;
 std::unique_ptr<EigenLocalLU> lu;
#ifdef RBFLAB_DOUBLE
 Eigen::JacobiSVD<EigenLocalLU::Dense> svd;
#endif
 int size,rank; double threshold=0,condition2=0;
 EigenLocalSolver(const Rows& a,const std::string& solver,double cutoff):size(a.size()),rank(size){
  if(solver=="lu"){lu=std::make_unique<EigenLocalLU>(a);return;}
#ifdef RBFLAB_DOUBLE
  svd.compute(EigenLocalLU::dense(a),Eigen::ComputeThinU|Eigen::ComputeThinV);
  if(svd.info()!=Eigen::Success)throw std::runtime_error("Local SVD failed");
  if(cutoff>=0)svd.setThreshold(cutoff);
  threshold=svd.threshold();rank=svd.rank();
  const auto& values=svd.singularValues();
  condition2=values(size-1)>0 ? values(0)/values(size-1) : std::numeric_limits<double>::infinity();
#else
  throw std::runtime_error("SVD local solver currently requires Float64");
#endif
 }
 Rows solve(const Rows& b)const{
  if(lu)return lu->solve(b);
#ifdef RBFLAB_DOUBLE
  EigenLocalLU::Dense x=svd.solve(EigenLocalLU::dense(b));
  Rows result(x.rows(),std::vector<Real>(x.cols()));
  for(int i=0;i<x.rows();++i)for(int j=0;j<x.cols();++j)result[i][j]=Real(x(i,j));
  return result;
#else
  throw std::runtime_error("SVD local solver currently requires Float64");
#endif
 }
};
