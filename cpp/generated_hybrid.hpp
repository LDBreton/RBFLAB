// Generated Cartesian derivatives, with analytic coincident limits.
#pragma once
namespace v_imq {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 3);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = pow(c, 2);
 const Real t6 = pow(x, 4);
 const Real t7 = pow(y, 4);
 return 45*pow(mu, 2)*t0*(-11*t0*t1*t7 - 4*t0*t3*t6 + t0*pow(x, 6) - 6*t0*pow(y, 6) + 90*t1*t3*t5 - 4*t2 - 116*t4 - 11*t5*t6 + 101*t5*t7 + 8)/pow(t2 + t4 + 1, 13.0/2.0);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 return 315*pow(c, 4)*pow(mu, 2)*x*y*(2*t0*t2*t4 - 16*t1 - 16*t3 + t4*pow(x, 4) + t4*pow(y, 4) + 16)/pow(t1 + t3 + 1, 13.0/2.0);
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 return 315*pow(c, 4)*pow(mu, 2)*x*y*(2*t0*t2*t4 - 16*t1 - 16*t3 + t4*pow(x, 4) + t4*pow(y, 4) + 16)/pow(t1 + t3 + 1, 13.0/2.0);
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 3);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = pow(c, 2);
 const Real t6 = pow(x, 4);
 const Real t7 = pow(y, 4);
 return -45*pow(mu, 2)*t0*(4*t0*t1*t7 + 11*t0*t3*t6 + 6*t0*pow(x, 6) - t0*pow(y, 6) - 90*t1*t3*t5 + 116*t2 + 4*t4 - 101*t5*t6 + 11*t5*t7 - 8)/pow(t2 + t4 + 1, 13.0/2.0);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace v_gaussian {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = -c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 + t2 - 7*t4 + t5*pow(y, 4) + 2)*exp(t2)*exp(-t4);
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = -c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 - 7*t2 + t4 + t5*pow(x, 4) + 2)*exp(-t2)*exp(t4);
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = -c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 + t2 - 7*t4 + t5*pow(y, 4) + 2)*exp(t2)*exp(-t4);
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = -c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 - 7*t2 + t4 + t5*pow(x, 4) + 2)*exp(-t2)*exp(t4);
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = -c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 + t2 - 7*t4 + t5*pow(y, 4) + 2)*exp(t2)*exp(-t4);
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = -c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 + t2 - 7*t4 + t5*pow(y, 4) + 2)*exp(t2)*exp(-t4);
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 3);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = pow(c, 2);
 const Real t6 = pow(x, 4);
 const Real t7 = pow(y, 4);
 const Real t8 = 2*t0;
 return -32*pow(mu, 2)*t0*(4*t0*t1*t7 - 18*t1*t3*t5 + 6*t2 + t3*t6*t8 + 30*t4 - t5*t6 - 17*t5*t7 + t8*pow(y, 6) - 6)*exp(-t2)*exp(-t4);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 const Real t2 = t0 + t1;
 return 64*pow(c, 4)*pow(mu, 2)*x*y*(t2 - 6)*(t2 - 2)*exp(-t0)*exp(-t1);
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = -c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 - 7*t2 + t4 + t5*pow(x, 4) + 2)*exp(-t2)*exp(t4);
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 return -16*pow(c, 3)*mu*x*y*(t0 + t1 - 3)*exp(-t0)*exp(-t1);
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 2);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = -c*t3;
 const Real t5 = 2*t0;
 return 8*mu*t0*(t1*t3*t5 - 7*t2 + t4 + t5*pow(x, 4) + 2)*exp(-t2)*exp(t4);
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 const Real t1 = c*pow(y, 2);
 const Real t2 = t0 + t1;
 return 64*pow(c, 4)*pow(mu, 2)*x*y*(t2 - 6)*(t2 - 2)*exp(-t0)*exp(-t1);
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(c, 3);
 const Real t1 = pow(x, 2);
 const Real t2 = c*t1;
 const Real t3 = pow(y, 2);
 const Real t4 = c*t3;
 const Real t5 = pow(c, 2);
 const Real t6 = pow(x, 4);
 const Real t7 = pow(y, 4);
 const Real t8 = 2*t0;
 return -32*pow(mu, 2)*t0*(4*t0*t3*t6 - 18*t1*t3*t5 + t1*t7*t8 + 30*t2 + 6*t4 - 17*t5*t6 - t5*t7 + t8*pow(x, 6) - 6)*exp(-t2)*exp(-t4);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace v_phs7 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3675*pow(mu, 2)*(t0 + 2*t1)/sqrt(t0 + t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 3675*pow(mu, 2)*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -735*mu*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 245*mu*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 3675*pow(mu, 2)*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3675*pow(mu, 2)*(2*t0 + t1)/sqrt(t0 + t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace v_phs9 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 19845*pow(mu, 2)*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -59535*pow(mu, 2)*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 2835*mu*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -567*mu*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -59535*pow(mu, 2)*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 19845*pow(mu, 2)*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_imq {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_gaussian {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(x, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(y, 2));
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 return -4*pow(c, 2)*x*y*exp(-c*pow(x, 2))*exp(-c*pow(y, 2));
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 const Real t0 = c*pow(y, 2);
 return -2*c*(2*t0 - 1)*exp(-t0)*exp(-c*pow(x, 2));
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_phs3 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3*(2*t0 + t1)/sqrt(t0 + t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -3*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3*(2*t0 + t1)/sqrt(t0 + t1);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -3*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -3*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3*(t0 + 2*t1)/sqrt(t0 + t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -3*x*y/sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -3*(t0 + 2*t1)/sqrt(t0 + t1);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_phs5 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 5*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 15*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 5*sqrt(t0 + t1)*(4*t0 + t1);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 15*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 15*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 5*sqrt(t0 + t1)*(t0 + 4*t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 15*x*y*sqrt(pow(x, 2) + pow(y, 2));
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 5*sqrt(t0 + t1)*(t0 + 4*t1);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_phs7 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(6*t0 + t1);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return -35*x*y*pow(pow(x, 2) + pow(y, 2), 3.0/2.0);
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return -7*pow(t0 + t1, 3.0/2.0)*(t0 + 6*t1);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
namespace p_phs9 {
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(8*t0 + t1);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 0;
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 return 63*x*y*pow(pow(x, 2) + pow(y, 2), 5.0/2.0);
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& mu){
 if(x==Real(0) && y==Real(0)) return Real(0);
 const Real t0 = pow(x, 2);
 const Real t1 = pow(y, 2);
 return 9*pow(t0 + t1, 5.0/2.0)*(t0 + 8*t1);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&);
inline Real eval(int source,int target,const Real& x,const Real& y,const Real& c,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,mu);
}}
struct KernelSpec {int family,power;Real c,weight;};
inline Real contribution(bool velocity,const KernelSpec& k,int s,int t,const Real& x,const Real& y,const Real& mu){
 Real value=velocity ? (k.family==0?v_imq::eval(s,t,x,y,k.c,mu):v_gaussian::eval(s,t,x,y,k.c,mu)) : (k.family==0?p_imq::eval(s,t,x,y,k.c,mu):p_gaussian::eval(s,t,x,y,k.c,mu));
 if(k.power) { Real tail=velocity?(k.power==7?v_phs7::eval(s,t,x,y,k.c,mu):v_phs9::eval(s,t,x,y,k.c,mu)):(k.power==3?p_phs3::eval(s,t,x,y,k.c,mu):(k.power==5?p_phs5::eval(s,t,x,y,k.c,mu):(k.power==7?p_phs7::eval(s,t,x,y,k.c,mu):p_phs9::eval(s,t,x,y,k.c,mu)))); value+=k.weight*tail; }
 return value; }
