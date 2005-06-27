function f = rhs(x,y)

% Smooth force term function.
f = 8*pi*pi*sin(2*pi*x).*sin(2*pi*y);
