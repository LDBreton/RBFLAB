#pragma once
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
// Native hardware double; wrapper keeps the generated kernel interface shared.
class Real {
public:
 double v=0;
 Real()=default;
 Real(double x):v(x){}
 Real(const std::string& x):v(std::stod(x)){}
 Real& operator+=(const Real& x){v+=x.v;return *this;}
 Real& operator-=(const Real& x){v-=x.v;return *this;}
 Real& operator*=(const Real& x){v*=x.v;return *this;}
 Real& operator/=(const Real& x){v/=x.v;return *this;}
 std::string str(int)const {if(!std::isfinite(v))throw std::runtime_error("Nonfinite double result");std::ostringstream s;s<<std::setprecision(17)<<v;return s.str();}
};
inline Real operator+(Real a,const Real& b){return a+=b;}
inline Real operator-(Real a,const Real& b){return a-=b;}
inline Real operator*(Real a,const Real& b){return a*=b;}
inline Real operator/(Real a,const Real& b){return a/=b;}
inline Real operator-(const Real& a){return -a.v;}
inline bool operator<(const Real&a,const Real&b){return a.v<b.v;}
inline bool operator>(const Real&a,const Real&b){return a.v>b.v;}
inline bool operator==(const Real&a,const Real&b){return a.v==b.v;}
inline Real abs(const Real&a){return std::abs(a.v);}
inline Real sqrt(const Real&a){return std::sqrt(a.v);}
inline Real pow(const Real&a,int b){return std::pow(a.v,b);}
inline Real pow(const Real&a,double b){return std::pow(a.v,b);}
inline Real pow(const Real&a,const Real&b){return std::pow(a.v,b.v);}
inline Real exp(const Real&a){return std::exp(a.v);}
inline Real sin(const Real&a){return std::sin(a.v);}
inline Real cos(const Real&a){return std::cos(a.v);}
inline Real log(const Real&a){return std::log(a.v);}
