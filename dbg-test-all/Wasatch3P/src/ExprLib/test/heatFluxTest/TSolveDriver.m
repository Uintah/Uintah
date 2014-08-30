clear; clc; close all;

n = 102;
dx = get_dx(n);

To = zeros(n,1);
for( i=1:n )
   To(i) = 10*exp(-(i*dx-n*dx/2).^2/100);
end

[t,T] = ode45('rhs',[0,10],To);
nt = length(t);
plot(T(nt,:),'r.-'); hold on; plot(To,'b.-'); legend('Tn','To');

TT  = load_T;
TTo = load_T0;

plot(TT,'ro'); plot(TTo,'bo');