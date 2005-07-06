function u = exactSolution(x,y)

u = sin(pi*x).*sin(pi*y);

center = find((abs(x-0.5) <= 0.25) & (abs(y-0.5) <= 0.25));
u(center) = u(center) + (x(center)-0.25).^4.*(x(center)-0.75).^4;

%u = x;
