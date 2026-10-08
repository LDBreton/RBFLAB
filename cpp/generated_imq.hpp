// Generated from rbflab Cartesian IMQ expressions; do not edit.
#pragma once
inline Real k11(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k12(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k13(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k14(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k15(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k16(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k17(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k18(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k21(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k22(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k23(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k24(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k25(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k26(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k27(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k28(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k31(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k32(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k33(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(y, 2);
 const Real t1 = c*pow(x, 2) + 1;
 return c*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
inline Real k34(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k35(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k36(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k37(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k38(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k41(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k42(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k43(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 3*pow(c, 2)*x*y/pow(c*pow(x, 2) + c*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k44(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*pow(y, 2) + 1;
 return -c*(2*c*t0 - t1)/pow(c*t0 + t1, 5.0/2.0);
}
inline Real k45(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k46(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k47(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k48(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return 0;
}
inline Real k51(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k52(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k53(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = pow(c, 2);
 const Real t4 = 3*t3;
 return -mu*t4*(27*c*t2 - t0*t2*t4 - 3*t1 + t3*pow(x, 4) - 4*t3*pow(y, 4) - 4)/pow(c*t2 + t1 + 1, 9.0/2.0);
}
inline Real k54(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k55(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = t1 + t3 + 1;
 const Real t5 = cp*t0;
 const Real t6 = cp*t2;
 const Real t7 = t5 + t6 + 1;
 const Real t8 = sqrt(t4);
 const Real t9 = cp*t8;
 const Real t10 = pow(cp, 2);
 const Real t11 = t10*t8;
 const Real t12 = 6*t9;
 const Real t13 = pow(c, 3);
 const Real t14 = pow(mu, 2);
 const Real t15 = sqrt(t7);
 const Real t16 = t14*t15;
 const Real t17 = t13*t16;
 const Real t18 = 360*t17;
 const Real t19 = pow(x, 12);
 const Real t20 = pow(c, 6);
 const Real t21 = t20*t9;
 const Real t22 = pow(y, 12);
 const Real t23 = pow(x, 4);
 const Real t24 = pow(y, 4);
 const Real t25 = 6*t11;
 const Real t26 = 15*t9;
 const Real t27 = pow(c, 2);
 const Real t28 = 20*t13;
 const Real t29 = pow(x, 6);
 const Real t30 = t29*t9;
 const Real t31 = pow(y, 6);
 const Real t32 = t31*t9;
 const Real t33 = pow(x, 8);
 const Real t34 = pow(c, 4);
 const Real t35 = t26*t34;
 const Real t36 = pow(y, 8);
 const Real t37 = pow(x, 10);
 const Real t38 = pow(c, 5);
 const Real t39 = t12*t38;
 const Real t40 = pow(y, 10);
 const Real t41 = t11*t20;
 const Real t42 = 15*t11;
 const Real t43 = t11*t36;
 const Real t44 = t16*t38;
 const Real t45 = 4545*t44;
 const Real t46 = t16*t20;
 const Real t47 = 45*t46;
 const Real t48 = 30*t8;
 const Real t49 = 720*t17;
 const Real t50 = t24*t5;
 const Real t51 = 60*t8;
 const Real t52 = t13*t51;
 const Real t53 = t23*t6;
 const Real t54 = t34*t51;
 const Real t55 = t23*t24;
 const Real t56 = t38*t48;
 const Real t57 = 60*t38;
 const Real t58 = 6*t20*t8;
 const Real t59 = 15*t21;
 const Real t60 = t10*t18;
 const Real t61 = t0*t11;
 const Real t62 = t0*t44;
 return -(6*c*t0*t10*t2*t8 + 12*c*t10*t23*t8 - c*t24*t25 + 10800*cp*t0*t14*t15*t2*t34 + 1530*cp*t0*t14*t15*t20*t31 + 270*cp*t14*t15*t2*t20*t29 + 1350*cp*t14*t15*t20*t23*t24 + 540*cp*t14*t15*t20*t36 + 360*cp*t14*t15*t23*t34 + 10440*cp*t14*t15*t24*t34 + 990*cp*t14*t15*t29*t38 - 9090*cp*t31*t44 - 90*cp*t33*t46 + 1035*t0*t10*t14*t15*t20*t36 + 10620*t0*t10*t14*t15*t24*t34 - t0*t10*t2*t49 + 2*t0*t10*t8 + 495*t0*t14*t15*t20*t24 + 180*t0*t14*t15*t34 - 4*t0*t22*t41 - 30*t0*t34*t43 - t1*t12 + 100*t10*t13*t2*t29*t8 + 60*t10*t13*t23*t24*t8 + 40*t10*t13*t33*t8 + 90*t10*t14*t15*t2*t20*t33 + 5580*t10*t14*t15*t2*t23*t34 + 1440*t10*t14*t15*t20*t23*t31 + 810*t10*t14*t15*t20*t24*t29 + 270*t10*t14*t15*t20*t40 + 180*t10*t14*t15*t29*t34 + 5220*t10*t14*t15*t31*t34 + 495*t10*t14*t15*t33*t38 + 11*t10*t19*t2*t20*t8 + 12*t10*t19*t38*t8 + 45*t10*t2*t23*t27*t8 - 3060*t10*t2*t29*t44 + 105*t10*t2*t33*t34*t8 + 54*t10*t2*t37*t38*t8 + 24*t10*t20*t24*t37*t8 + 10*t10*t20*t29*t36*t8 + 25*t10*t20*t31*t33*t8 + 2*t10*t20*t8*pow(x, 14) + 30*t10*t23*t31*t34*t8 + 120*t10*t24*t29*t34*t8 + 90*t10*t24*t33*t38*t8 + 30*t10*t27*t29*t8 + 60*t10*t29*t31*t38*t8 - 13140*t10*t31*t62 + 30*t10*t34*t37*t8 - t10*t36*t45 - t10*t37*t47 - 12150*t10*t44*t55 - t11*t2 - t12*t3 + 180*t14*t15*t2*t20*t23 + 5220*t14*t15*t2*t34 + 270*t14*t15*t20*t31 + 495*t14*t15*t23*t38 - t18 - t19*t21 - t2*t27*t48*t5 - 4050*t2*t62 - t21*t22 - 20*t21*t29*t31 - t22*t25*t38 - t23*t26*t27 - t23*t32*t57 - t23*t36*t59 - 3*t23*t40*t41 - t23*t60 - t24*t26*t27 - t24*t30*t57 - t24*t33*t59 - t24*t45 - t24*t60 - t27*t31*t42 - t28*t30 - t28*t31*t61 - t28*t32 - t28*t43 - t29*t47 - t29*t54*t6 - t31*t5*t54 - t33*t35 - t33*t56*t6 - t34*t40*t42 - 90*t34*t55*t9 - t35*t36 - t36*t5*t56 - t37*t39 - t37*t58*t6 - 18*t38*t40*t61 - t39*t40 - t40*t5*t58 - t41*pow(y, 14) - 17190*t44*t50 - 7110*t44*t53 - t49*t5 - t49*t6 - t50*t52 - t52*t53 - t9)/(pow(t4, 13.0/2.0)*pow(t7, 5.0/2.0));
}
inline Real k56(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = t1 + t3 + 1;
 const Real t5 = cp*t0;
 const Real t6 = cp*t2;
 const Real t7 = t5 + t6 + 1;
 const Real t8 = pow(cp, 2);
 const Real t9 = sqrt(t4)*t8;
 const Real t10 = pow(c, 4);
 const Real t11 = pow(mu, 2)*sqrt(t7);
 const Real t12 = 1680*t11;
 const Real t13 = t10*t12;
 const Real t14 = 6*t9;
 const Real t15 = pow(c, 6);
 const Real t16 = t15*t9;
 const Real t17 = pow(x, 4);
 const Real t18 = pow(c, 2);
 const Real t19 = 15*t9;
 const Real t20 = t18*t19;
 const Real t21 = pow(y, 4);
 const Real t22 = pow(x, 6);
 const Real t23 = 20*t22;
 const Real t24 = pow(c, 3)*t9;
 const Real t25 = pow(y, 6);
 const Real t26 = pow(x, 8);
 const Real t27 = t10*t19;
 const Real t28 = pow(y, 8);
 const Real t29 = pow(x, 10);
 const Real t30 = pow(c, 5);
 const Real t31 = t14*t30;
 const Real t32 = pow(y, 10);
 const Real t33 = t12*t30;
 const Real t34 = t11*t15;
 const Real t35 = 105*t34;
 const Real t36 = 3360*t11;
 const Real t37 = t10*t36;
 const Real t38 = cp*t30*t36;
 const Real t39 = 210*t34;
 const Real t40 = cp*t39;
 const Real t41 = t0*t2;
 const Real t42 = 30*t9;
 const Real t43 = 60*t24;
 const Real t44 = t0*t21;
 const Real t45 = t17*t2;
 const Real t46 = t13*t8;
 const Real t47 = t0*t25;
 const Real t48 = 60*t9;
 const Real t49 = t10*t48;
 const Real t50 = t17*t21;
 const Real t51 = t2*t22;
 const Real t52 = t33*t8;
 const Real t53 = t30*t42;
 const Real t54 = t30*t48;
 const Real t55 = t35*t8;
 const Real t56 = 6*t16;
 const Real t57 = 15*t16;
 const Real t58 = t11*t30;
 const Real t59 = 630*t34;
 const Real t60 = 5040*t58*t8;
 const Real t61 = 420*t34*t8;
 return -3*x*y*(t0*t28*t53 + t0*t32*t56 + t0*t33 + t1*t14 + 90*t10*t50*t9 - t13 + t14*t3 + t16*t23*t25 + t16*pow(x, 12) + t16*pow(y, 12) + t17*t20 + t17*t25*t54 + t17*t28*t57 - t17*t35 + t17*t38 - t17*t46 - t17*t59*t6 + t18*t41*t42 + t2*t26*t53 + t2*t29*t56 + t2*t33 + 6720*t2*t5*t58 + t20*t21 + t21*t22*t54 + t21*t26*t57 - t21*t35 + t21*t38 - t21*t46 - t21*t5*t59 - t22*t40 + t22*t52 + t23*t24 + 20*t24*t25 - t25*t40 + t25*t52 + t26*t27 - t26*t55 + t27*t28 - t28*t55 + t29*t31 + t31*t32 - t37*t41*t8 - t37*t5 - t37*t6 - t39*t41 + t43*t44 + t43*t45 + t44*t60 + t45*t60 + t47*t49 - t47*t61 + t49*t51 - t50*t59*t8 - t51*t61 + t9)/(pow(t4, 13.0/2.0)*pow(t7, 5.0/2.0));
}
inline Real k57(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = cp*pow(y, 2) + 1;
 return -cp*(2*cp*t0 - t1)/pow(cp*t0 + t1, 5.0/2.0);
}
inline Real k58(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return -3*pow(cp, 2)*x*y/pow(cp*pow(x, 2) + cp*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k61(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k62(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k63(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = c*pow(x, 2) + c*pow(y, 2);
 return -15*pow(c, 3)*mu*x*y*(t0 - 6)/pow(t0 + 1, 9.0/2.0);
}
inline Real k64(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = pow(c, 2);
 const Real t5 = 3*t4;
 return mu*t5*(t0*t2*t5 - 27*t1 + 3*t3 + 4*t4*pow(x, 4) - t4*pow(y, 4) + 4)/pow(t1 + t3 + 1, 9.0/2.0);
}
inline Real k65(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = t1 + t3 + 1;
 const Real t5 = cp*t0;
 const Real t6 = cp*t2;
 const Real t7 = t5 + t6 + 1;
 const Real t8 = pow(cp, 2);
 const Real t9 = sqrt(t4)*t8;
 const Real t10 = pow(c, 4);
 const Real t11 = pow(mu, 2)*sqrt(t7);
 const Real t12 = 1680*t11;
 const Real t13 = t10*t12;
 const Real t14 = 6*t9;
 const Real t15 = pow(c, 6);
 const Real t16 = t15*t9;
 const Real t17 = pow(x, 4);
 const Real t18 = pow(c, 2);
 const Real t19 = 15*t9;
 const Real t20 = t18*t19;
 const Real t21 = pow(y, 4);
 const Real t22 = pow(x, 6);
 const Real t23 = 20*t22;
 const Real t24 = pow(c, 3)*t9;
 const Real t25 = pow(y, 6);
 const Real t26 = pow(x, 8);
 const Real t27 = t10*t19;
 const Real t28 = pow(y, 8);
 const Real t29 = pow(x, 10);
 const Real t30 = pow(c, 5);
 const Real t31 = t14*t30;
 const Real t32 = pow(y, 10);
 const Real t33 = t12*t30;
 const Real t34 = t11*t15;
 const Real t35 = 105*t34;
 const Real t36 = 3360*t11;
 const Real t37 = t10*t36;
 const Real t38 = cp*t30*t36;
 const Real t39 = 210*t34;
 const Real t40 = cp*t39;
 const Real t41 = t0*t2;
 const Real t42 = 30*t9;
 const Real t43 = 60*t24;
 const Real t44 = t0*t21;
 const Real t45 = t17*t2;
 const Real t46 = t13*t8;
 const Real t47 = t0*t25;
 const Real t48 = 60*t9;
 const Real t49 = t10*t48;
 const Real t50 = t17*t21;
 const Real t51 = t2*t22;
 const Real t52 = t33*t8;
 const Real t53 = t30*t42;
 const Real t54 = t30*t48;
 const Real t55 = t35*t8;
 const Real t56 = 6*t16;
 const Real t57 = 15*t16;
 const Real t58 = t11*t30;
 const Real t59 = 630*t34;
 const Real t60 = 5040*t58*t8;
 const Real t61 = 420*t34*t8;
 return -3*x*y*(t0*t28*t53 + t0*t32*t56 + t0*t33 + t1*t14 + 90*t10*t50*t9 - t13 + t14*t3 + t16*t23*t25 + t16*pow(x, 12) + t16*pow(y, 12) + t17*t20 + t17*t25*t54 + t17*t28*t57 - t17*t35 + t17*t38 - t17*t46 - t17*t59*t6 + t18*t41*t42 + t2*t26*t53 + t2*t29*t56 + t2*t33 + 6720*t2*t5*t58 + t20*t21 + t21*t22*t54 + t21*t26*t57 - t21*t35 + t21*t38 - t21*t46 - t21*t5*t59 - t22*t40 + t22*t52 + t23*t24 + 20*t24*t25 - t25*t40 + t25*t52 + t26*t27 - t26*t55 + t27*t28 - t28*t55 + t29*t31 + t31*t32 - t37*t41*t8 - t37*t5 - t37*t6 - t39*t41 + t43*t44 + t43*t45 + t44*t60 + t45*t60 + t47*t49 - t47*t61 + t49*t51 - t50*t59*t8 - t51*t61 + t9)/(pow(t4, 13.0/2.0)*pow(t7, 5.0/2.0));
}
inline Real k66(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = pow(x, 2);
 const Real t1 = c*t0;
 const Real t2 = pow(y, 2);
 const Real t3 = c*t2;
 const Real t4 = t1 + t3 + 1;
 const Real t5 = cp*t0;
 const Real t6 = cp*t2;
 const Real t7 = t5 + t6 + 1;
 const Real t8 = sqrt(t4);
 const Real t9 = cp*t8;
 const Real t10 = pow(cp, 2);
 const Real t11 = t10*t8;
 const Real t12 = t0*t11;
 const Real t13 = 6*t9;
 const Real t14 = pow(c, 3);
 const Real t15 = pow(mu, 2)*sqrt(t7);
 const Real t16 = 360*t15;
 const Real t17 = t14*t16;
 const Real t18 = 2*t11;
 const Real t19 = pow(x, 12);
 const Real t20 = pow(c, 6);
 const Real t21 = t20*t9;
 const Real t22 = pow(y, 12);
 const Real t23 = pow(x, 4);
 const Real t24 = 6*t11;
 const Real t25 = pow(y, 4);
 const Real t26 = 12*t11;
 const Real t27 = pow(c, 2);
 const Real t28 = 15*t9;
 const Real t29 = t27*t28;
 const Real t30 = pow(x, 6);
 const Real t31 = 20*t30;
 const Real t32 = t14*t9;
 const Real t33 = pow(y, 6);
 const Real t34 = pow(x, 8);
 const Real t35 = pow(c, 4);
 const Real t36 = t28*t35;
 const Real t37 = pow(y, 8);
 const Real t38 = pow(x, 10);
 const Real t39 = pow(c, 5);
 const Real t40 = t13*t39;
 const Real t41 = pow(y, 10);
 const Real t42 = t11*t20;
 const Real t43 = 15*t11;
 const Real t44 = 30*t11;
 const Real t45 = t11*t14;
 const Real t46 = t35*t44;
 const Real t47 = t15*t35;
 const Real t48 = 5220*t47;
 const Real t49 = 180*t47;
 const Real t50 = t15*t39;
 const Real t51 = 4545*t50;
 const Real t52 = 495*t50;
 const Real t53 = t15*t20;
 const Real t54 = 270*t53;
 const Real t55 = 45*t53;
 const Real t56 = t5*t8;
 const Real t57 = 30*t56;
 const Real t58 = 720*t14*t15;
 const Real t59 = 60*t25;
 const Real t60 = t6*t8;
 const Real t61 = 60*t23;
 const Real t62 = t23*t47;
 const Real t63 = t25*t35;
 const Real t64 = t33*t35;
 const Real t65 = 60*t30;
 const Real t66 = cp*t50;
 const Real t67 = t37*t39;
 const Real t68 = t39*t9;
 const Real t69 = cp*t53;
 const Real t70 = 6*t20;
 const Real t71 = 15*t21;
 const Real t72 = t10*t17;
 const Real t73 = t10*t30;
 const Real t74 = t10*t33;
 const Real t75 = t11*t23;
 const Real t76 = t10*t34;
 const Real t77 = t10*t37;
 const Real t78 = t11*t39;
 const Real t79 = t0*t2;
 const Real t80 = t0*t25;
 const Real t81 = t2*t53;
 const Real t82 = t23*t25;
 return (c*t23*t24 - c*t25*t26 - cp*t16*t63 - 10440*cp*t62 - t0*t48 + 3060*t0*t50*t74 - 90*t0*t53*t77 + t1*t13 - t1*t2*t24 - 10620*t10*t2*t62 - t10*t38*t54 + t10*t41*t55 - 5580*t10*t47*t80 + 12150*t10*t50*t82 + t10*t58*t79 - 100*t12*t14*t33 - 11*t12*t20*t22 - 45*t12*t25*t27 - 105*t12*t35*t37 - 54*t12*t39*t41 + t12 + t13*t3 + t14*t56*t59 + t14*t60*t61 + t17 - t18*t2 - t18*t20*pow(y, 14) + 4*t19*t2*t42 + t19*t21 + t19*t24*t39 + t2*t27*t57 + t2*t31*t45 + t2*t34*t46 + 18*t2*t38*t78 - 10800*t2*t47*t5 - t2*t49 + 13140*t2*t50*t73 + t21*t22 + t21*t31*t33 - t22*t26*t39 + t23*t29 + t23*t37*t71 - 24*t23*t41*t42 - t23*t45*t59 + 17190*t23*t50*t6 + t23*t51 - 810*t23*t53*t74 + 90*t23*t63*t9 + t23*t72 - 495*t23*t81 + t25*t29 - t25*t30*t46 + t25*t34*t71 + 3*t25*t38*t42 + 7110*t25*t5*t50 - t25*t52 - 1440*t25*t53*t73 + t25*t72 + t27*t30*t43 - t27*t33*t44 - 25*t30*t37*t42 - 1530*t30*t53*t6 - t30*t54 + t30*t59*t68 + 9090*t30*t66 + t31*t32 + 20*t32*t33 - 10*t33*t34*t42 - t33*t5*t54 + t33*t55 + t33*t61*t68 - t33*t65*t78 - 990*t33*t66 + t34*t36 + 30*t34*t39*t60 + 20*t34*t45 - 540*t34*t69 + t35*t38*t43 + t35*t60*t65 + t36*t37 - 40*t37*t45 + 90*t37*t69 + t38*t40 + t38*t60*t70 + t40*t41 - t41*t46 + t41*t56*t70 + t42*pow(x, 14) - t48*t73 - t49*t74 + t5*t58 + 4050*t50*t79 + t51*t76 - t52*t77 - 180*t53*t80 + 60*t56*t64 + t57*t67 + t58*t6 - 120*t64*t75 - 90*t67*t75 - 1350*t69*t82 - 1035*t76*t81 + t9)/(pow(t4, 13.0/2.0)*pow(t7, 5.0/2.0));
}
inline Real k67(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 return -3*pow(cp, 2)*x*y/pow(cp*pow(x, 2) + cp*pow(y, 2) + 1, 5.0/2.0);
}
inline Real k68(const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 const Real t0 = cp*pow(y, 2);
 const Real t1 = cp*pow(x, 2) + 1;
 return cp*(-2*t0 + t1)/pow(t0 + t1, 5.0/2.0);
}
using Fn=Real(*)(const Real&,const Real&,const Real&,const Real&,const Real&);
inline Real kernel(int source,int target,const Real& x,const Real& y,const Real& c,const Real& cp,const Real& mu){
 static Fn table[6][8]={{k11,k12,k13,k14,k15,k16,k17,k18},{k21,k22,k23,k24,k25,k26,k27,k28},{k31,k32,k33,k34,k35,k36,k37,k38},{k41,k42,k43,k44,k45,k46,k47,k48},{k51,k52,k53,k54,k55,k56,k57,k58},{k61,k62,k63,k64,k65,k66,k67,k68}};
 return table[source-1][target-1](x,y,c,cp,mu);
}
