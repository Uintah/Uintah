% This matlab scrips generates the source terms for an arbitrary MMS.

%______________Define the MMS solution
clear all
close all
mms_solution = 2;  % 
BurgersEq    = 0;
plotPress    = false;
 
syms rho u v T P t x y z gamma cv mu


% SAND 2000-1444, pg 45, compressible 
if(mms_solution == 1)
  syms rho0 omg u0 v0 e0 eps
  
  rho = rho0 * (sin((omg * t) + x^2 + y^2)   + 1.5 )
  u   = u0   * (sin((omg * t) + x^2 + y^2)   + eps )
  v   = v0   * (cos((omg * t) + x^2 + y^2)   + eps )
  e   = e0   * (cos((omg * t) + x^2 + y^2 )  + 1.5)
  
  P   = (gamma -1) * rho * e
end

% Randy McDermott's Solution (constant Density, incompressible)
if(mms_solution == 2)    
  syms nu mu rho A
  rho = 1
  u = 1 - A * cos(x-t) * sin(y-t) * exp(-2*mu*t)
  v = 1 + A * sin(x-t) * cos(y-t) * exp(-2*mu*t)
  P = -((A^2)/4)*( cos(2*(x-t)) +  cos(2*(y-t)) ) * exp(-4*mu*t) 
  e = 1
end

%Minion "On the stability of Godunov-Projection Methods for incompressible
%Flow, pg 14, N = 1
if(mms_solution == 3)  
  syms mu rho A     
  rho = 1
  A = 10
  u = 1 - ( A * cos(x-t) * sin(y-t) )* exp(-2*mu*t)
  v = 1 + ( A * sin(x-t) * cos(y-t) )* exp(-2*mu*t)
  P = ((A^2)/2)*( ( (sin(x-t))^2 +  (sin(y-t))^2 )*exp(-4*mu*t) )
  e = 1
end


%______________Define the Governing Equations_________________________
Smass = diff(rho,t) + diff((u*rho),x)  ...
                    + diff((v*rho),y)

% viscous tensor
div_vel= ( diff(u,'x') + diff(v,'y') );
tau_xx = -2 * mu * diff(u,'x') + (2/3) * mu * div_vel;
tau_yy = -2 * mu * diff(v,'y') + (2/3) * mu * div_vel;
tau_xy = -mu * ( diff(u,'y') + diff(v,'x') );

% x-momentum
Su = diff( (rho*u),'t') + diff((rho*(u^2)),'x') + diff((rho*u*v),'y') ....
  + diff(P,'x') + (diff(tau_xx,'x') + diff(tau_xy,'y'));

% y-momentum
Sv = diff( (rho*v),'t') + diff((rho*(u*v)),'x') + diff((rho*(v^2)),'y') ....
  + diff(P,'y') + (diff(tau_xy,'x') + diff(tau_yy,'y'));

% energy
% Se = diff( (rho*e),'t') + diff((rho*e),'x') + diff((rho*e),'y') ...
%      -P * (div_vel)


gradP = diff(P,'x') + diff(P,'y')

x_mom_source = simple(Su)
y_mom_source = simple(Sv)
 
% Plotting 
if (plotPress)
  ezcontour(P)
end
  
%______________Burgers equations
% This plots the difference between SAND2000-1444 pg 61
% and matlab code.  This is simply a sanity check.
if(BurgersEq)
  u0  = 1;
  nu  = 1;
  omg = 0;
  v0  = 1;
  eps = 1;
  t   = 2;
  % define the velocity
  u   = u0   * (sin((omg * t) + x^2 + y^2 ) + eps )
  v   = v0   * (cos((omg * t) + x^2 + y^2)  + eps )

  Su = diff(u,'t') + diff((u^2),'x') + diff((u*v),'y') ....
    - nu * ( diff(u,'x',2) + diff(u,'y',2) )

  Sv = diff(v,'t') + diff((u*v),'x') + diff((v^2),'y') ....
    - nu * ( diff(v,'x',2) + diff(v,'y',2) )

  % equation c5 of SAND2000-1444
  exactSu = u0*( 2 * v0 * y * cos(2 * ( (omg*t) + x^2 + y^2)) ...
    + 2*(2*nu*(x^2) - eps*v0*y + 2*nu*(y^2))*sin((omg*t) + x^2 + y^2) ...
    + ( 4*eps*u0*x - 4*nu + 2*eps*v0*y + 4*u0*x*sin((omg*t) + x^2 + y^2))*cos((omg*t) + x^2 + y^2))


  %equation c6 of rof SAND2000-1444
  exactSv = -v0*( -2 * u0 * x * cos(2 * ( (omg*t) + x^2 + y^2)) ...
    - 2*(eps*u0*x + 2*nu*( (x^2)+(y^2) ))*cos((omg*t) + x^2 + y^2) ...
    + ( 2*eps*u0*x - 4*nu + 4*eps*v0*y + 4*v0*y*cos((omg*t) + x^2 + y^2))*sin((omg*t) + x^2 + y^2))

  % difference between matlab and reference source terms
  % they should be exact
  ezcontour(Su-exactSu)
  figure
  ezcontour(Sv-exactSv)
end
      
