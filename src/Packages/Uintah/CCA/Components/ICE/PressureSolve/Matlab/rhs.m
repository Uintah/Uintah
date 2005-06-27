function f = rhs(x,y)

[xmat1,xmat2] = ndgrid(x,y);

f = 2*pi*pi*sin(pi*xmat1).*sin(pi*xmat2);

center = find((abs(xmat1-0.5) <= 0.25) & (abs(xmat2-0.5) <= 0.25));
f(center) = f(center) - ...
    (12*(xmat1(center)-0.25).^2.*(xmat1(center)-0.75).^4 + 12*(xmat1(center)-0.25).^4.*(xmat1(center)-0.75).^2 + ...
    2*4*4*(xmat1(center)-0.25).^3.*(xmat1(center)-0.75).^3);

%f = zeros(size(xmat1));

