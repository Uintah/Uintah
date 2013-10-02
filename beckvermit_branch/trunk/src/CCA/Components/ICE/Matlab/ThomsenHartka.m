%Thomsen & Hartka Equation of State for Water
clc; clear all; close all;

%In the following, input the known values for two of the variables, and
%a guess for the third.  Save and run the program.  Query the "_calc"
%version of the unknown variable, e.g. r_calc, and replace the value of
%the unknown with that value.  Now, if you run it again, you'll find
%that the "_calc" versions of all variables are the same as those entered.
%Additionally, all of the other variables will be computed with that
%consistent thermodynamic state.
r=999.9756494265155; %(kg·m-3)
T=300.0;             %(K)
P=10132500.0;        %(Pa)

%Constants
vo=1.00008E-03;     %(m3·kg-1)
L=8.E-06;           %(K-2)
To=277;             %(K)
a=2.E-07;           %(K·Pa-1)
ko=5.E-10;          %(Pa-1)
co=4205.7;          %(J·kg-1·K-1)
b=2.6;              %(J·kg-1·K-2)

%Gibbs function
g=(1/2)*b*(T-To)^2+(T-To)*(co+b*To)+(P-(1/2)*ko*P^2)*vo+...
    P*((1/3)*a^2*P^2+a*P*(T-To)+(T-To)^2)*L*vo+T*(-co-b*To)*log(T/To);
    
%Specific volume
v=(1-ko*P+(a*P+T-To)^2*L)*vo;

%Internal energy
u=co*(T-To)-(1/2)*b*(T-To)^2+P*(a*P*(-2*T+To)+2*T*(-T+To))*L*vo+(1/6)*P^2*(3*ko-4*a^2*P*L)*vo;

%Specific heat at constant volume
cv=co+b*(-T+To)+2*P*(-a*P-2*T+To)*L*vo;

%Enthalpy
h=(1/6)*(3*(T-To)*(2*co+b*(-T+To))+P*(6-3*ko*P+2*(a^2*P^2-3*T^2-3*a*P*To+3*To^2)*L)*vo);

%Specific heat at constant pressure
cp=co+b*(-T+To)-2*P*T*L*vo;

%Ratio of Specific heats
y=(co+b*(-T+To)-2*P*T*L*vo)/(co+b*(-T+To)+2*P*(-a*P-2*T+To)*L*vo);

%Density
r_calc=1/((1-ko*P+(a*P+T-To)^2*L)*vo);

%Temperature
T_calc=-a*P+To+sqrt(L*vo*r*(1+(-1+ko*P)*vo*r))/(L*vo*r);

%Pressure
P_calc=(1/(2*a^2*L*vo*r))*((ko+2*a*(-T+To)*L)*vo*r-sqrt(vo*r*(ko^2*vo*r+4*a*L*(a-(a+ko*(T-To))*vo*r))));

%dP/dr at constant temperature
dPdr=(1-ko*P+(a*P+T-To)^2*L)^2*vo/(ko-2*a*(a*P+T-To)*L);

%dP/de at constant volume
dPde=(2*(a*P+T-To)*L)/((ko-2*a*(a*P+T-To)*L)*(co+b*(-T+To)+2*P*(-a*P-2*T+To)*L*vo));

%Constant pressure thermal expansivity
B=2*(a*P+T-To)*L/(1-ko*P+(a*P+T-To)^2*L);
