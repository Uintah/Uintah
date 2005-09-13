rho = 7830;
mu = 80.0e9;
Mkg = 55.845*1.66043998903379e-27;
debye_freq = 3*sqrt(pi)*0.5*(4.0*pi*rho/(3.0*Mkg))^(1.0/3.0)*sqrt(mu/rho)


kb = 1.38e-23;
T = 300;
wd = debye_freq;
c = sqrt(mu/rho);
dislocation_drag = kb*T*wd^2/(pi^2*c^3)

%
% St
%
rho = 0.7e11;
a = 3.48e-10;
b = 2.48e-10;
L = 10^4*b
nu = debye_freq;
w = 24*b
D = 1.0e-4;
C1 = rho*L*a*b^2*nu/(2*w^2)
C2 = D/(rho*b^2)

%
% Cu
%
rho = 1.0e11;
a = 2.86e-10;
b = 2.86e-10;
L = 10^4*b;
nu = 1.0e13;
w = 24*b;
D = 1.0e-4;
C1 = rho*L*a*b^2*nu/(2*w^2)
C2 = D/(rho*b^2)

%
% Al
%
rho = 3.0e11;
a = 3.0e-10;
b = 2.0e-10;
L = 10^4*b;
w = 50*b;
nu = 5.2e13;
D = 1.0e-4;
C1 = rho*L*a*b^2*nu/(2*w^2);
C2 = D/(rho*b^2);

