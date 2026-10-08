#ifdef RBFLAB_DOUBLE
#include "real_double.hpp"
#else
#pragma once
#include <mpfr.h>
#include <string>
#include <ostream>
#include <stdexcept>
// Small RAII wrapper around dynamically linked MPFR. No legacy wrapper/code.
class Real {
public:
 static thread_local mpfr_prec_t bits;
 mpfr_t v;
 Real(){mpfr_init2(v,bits);mpfr_set_zero(v,0);}
 Real(int x):Real(){mpfr_set_si(v,x,MPFR_RNDN);}
 Real(double x):Real(){mpfr_set_d(v,x,MPFR_RNDN);}
 Real(const std::string& x):Real(){if(mpfr_set_str(v,x.c_str(),10,MPFR_RNDN))throw std::runtime_error("Invalid decimal input");}
 Real(const Real& x):Real(){mpfr_set(v,x.v,MPFR_RNDN);}
 Real(Real&& x) noexcept:Real(){mpfr_swap(v,x.v);}
 ~Real(){mpfr_clear(v);}
 Real& operator=(const Real& x){mpfr_set(v,x.v,MPFR_RNDN);return *this;}
 Real& operator=(Real&& x) noexcept {mpfr_swap(v,x.v);return *this;}
 Real& operator+=(const Real& x){mpfr_add(v,v,x.v,MPFR_RNDN);return *this;}
 Real& operator-=(const Real& x){mpfr_sub(v,v,x.v,MPFR_RNDN);return *this;}
 Real& operator*=(const Real& x){mpfr_mul(v,v,x.v,MPFR_RNDN);return *this;}
 Real& operator/=(const Real& x){mpfr_div(v,v,x.v,MPFR_RNDN);return *this;}
 std::string str(int digits)const {char* b=nullptr;mpfr_asprintf(&b,"%.*Rg",digits,v);std::string out(b);mpfr_free_str(b);return out;}
};
thread_local mpfr_prec_t Real::bits=340;
inline Real operator+(const Real&a,const Real&b){Real r;mpfr_add(r.v,a.v,b.v,MPFR_RNDN);return r;}
inline Real operator-(const Real&a,const Real&b){Real r;mpfr_sub(r.v,a.v,b.v,MPFR_RNDN);return r;}
inline Real operator*(const Real&a,const Real&b){Real r;mpfr_mul(r.v,a.v,b.v,MPFR_RNDN);return r;}
inline Real operator/(const Real&a,const Real&b){Real r;mpfr_div(r.v,a.v,b.v,MPFR_RNDN);return r;}
inline Real operator-(const Real&a){Real r;mpfr_neg(r.v,a.v,MPFR_RNDN);return r;}
inline bool operator<(const Real&a,const Real&b){return mpfr_less_p(a.v,b.v);}
inline bool operator>(const Real&a,const Real&b){return b<a;}
inline bool operator==(const Real&a,const Real&b){return mpfr_equal_p(a.v,b.v);}
inline Real abs(const Real&a){Real r;mpfr_abs(r.v,a.v,MPFR_RNDN);return r;}
inline Real sqrt(const Real&a){Real r;mpfr_sqrt(r.v,a.v,MPFR_RNDN);return r;}
inline Real pow(const Real&a,int b){Real r;mpfr_pow_si(r.v,a.v,b,MPFR_RNDN);return r;}
inline Real pow(const Real&a,double b){Real e(b),r;mpfr_pow(r.v,a.v,e.v,MPFR_RNDN);return r;}
inline Real pow(const Real&a,const Real&b){Real r;mpfr_pow(r.v,a.v,b.v,MPFR_RNDN);return r;}

inline Real exp(const Real&a){Real r;mpfr_exp(r.v,a.v,MPFR_RNDN);return r;}

inline Real sin(const Real&a){Real r;mpfr_sin(r.v,a.v,MPFR_RNDN);return r;}
inline Real cos(const Real&a){Real r;mpfr_cos(r.v,a.v,MPFR_RNDN);return r;}
inline Real log(const Real&a){Real r;mpfr_log(r.v,a.v,MPFR_RNDN);return r;}
#endif
