function u = exactSolution(x,y)

u = sin(pi*x).*sin(pi*y);

if (0)
    center = find((abs(x-0.5) <= 0.25) & (abs(y-0.5) <= 0.25));
    %u(center) = u(center) + (sin(2*pi*(x(center)-0.25)).*sin(2*pi*(y(center)-0.25))).^2;
    %u(center) = u(center) + sin(2*pi*(x(center)-0.25)).^4;
    u(center) = u(center) + (x(center)-0.25).^4.*(x(center)-0.75).^4;

    %u = x;
end
