function u = exactSolution(x,y)

%u = sin(pi*x).*sin(pi*y);
u = zeros(size(x));

if (1)
    %center = find((abs(x-0.5) <= 0.25) & (abs(y-0.5) <= 0.25));
    %u(center) = u(center) + (sin(2*pi*(x(center)-0.25)).*sin(2*pi*(y(center)-0.25))).^2;
    %u(center) = u(center) + sin(2*pi*(x(center)-0.25)).^4;
    %u(center) = u(center) + 100*(x(center)-0.25).^4.*(x(center)-0.75).^4;

    sigma = 0.25;
    u = u + exp(-((x-0.5).^2 + (y-0.5).^2)/sigma^2);
    
    %u = x;
end

%u = log(sqrt((x{1}-0.5).^2 + (x{2}-0.5).^2));
