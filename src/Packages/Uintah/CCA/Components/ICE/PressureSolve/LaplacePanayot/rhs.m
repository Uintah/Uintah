function f = rhs(x)

%[xmat1,xmat2] = ndgrid(x,y);

f = 2*pi*pi*sin(pi*x{1}).*sin(pi*x{2});

if (0)
    center = find((abs(xmat1-0.5) <= 0.25) & (abs(xmat2-0.5) <= 0.25));
    % f(center) = f(center) + 8*pi*pi*(...
    %     sin(4*pi*(xmat1(center)-0.25)).*sin(2*pi*(xmat2(center)-0.25)).^2 + ...
    %     sin(4*pi*(xmat2(center)-0.25)).*sin(2*pi*(xmat1(center)-0.25)).^2 ...
    %     );
    % f(center) = f(center) - ...
    %     16*pi*pi*(3*sin(2*pi*(xmat1(center)-0.25)).^2 - 4*sin(2*pi*(xmat1(center)-0.25)).^4);
    f(center) = f(center) - ...
        (12*(xmat1(center)-0.25).^2.*(xmat1(center)-0.75).^4 + 12*(xmat1(center)-0.25).^4.*(xmat1(center)-0.75).^2 + ...
        2*4*4*(xmat1(center)-0.25).^3.*(xmat1(center)-0.75).^3);

    %f = zeros(size(xmat1));

end
