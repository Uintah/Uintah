clear; clc; close all;

dt = [ 0.4, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001 ];
n = length(dt);
err = zeros(n,2);
co = 1.0;
for( i=1:n )
   t = 0:dt(i):1;
   nn = length(t);
   c_rk3 = ssprk3('rhs',co,t);
   c_fe  = forward_euler('rhs',co,t);
   cx = co*exp(-t);
   err(i,1) = abs(c_rk3(nn) - cx(nn));
   err(i,2) = abs(c_fe (nn) - cx(nn));
end

loglog(dt,err);
legend('rk3','fe');

% output from testIntegrator.cpp using SSPRK2
fe = [0.4  0.0851942
   0.2  0.0401994
   0.1  0.0190605
   0.05  0.00939352
   0.025  0.004647
   0.01  0.0018471
   0.005  0.000921619
   0.0025  0.000460327
   0.001  0.000184016
];

% output from testIntegrator.cpp using SSPRK3
rk3=[0.4  0.00132812
   0.2  0.000143957
   0.1  1.65291e-05
   0.05  1.99429e-06
   0.025  2.44345e-07
   0.01  1.54515e-08
   0.005  1.92372e-09
   0.0025  2.39996e-10
   0.001  1.53622e-11];

hold on;
loglog(fe(:,1),fe(:,2),'co');
loglog(rk3(:,1),rk3(:,2),'ro');

