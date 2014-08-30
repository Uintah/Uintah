function dTdt = rhs( t, T )
n = length(T);
dx = get_dx(n);
dTdt = zeros(n,1);
lambda = 20.0;
for( i=2:n-1 )
   dTdt(i) = lambda*(T(i-1)-2*T(i)+T(i+1)) / dx^2;
end