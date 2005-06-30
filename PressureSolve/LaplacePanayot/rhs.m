function f = rhs(x)

%[xmat1,x{2}] = ndgrid(x,y);

%f = 2*pi*pi*sin(pi*x{1}).*sin(pi*x{2});
f = zeros(size(x{1}));

if (1)
    %center = find((abs(x{1}-0.5) <= 0.25) & (abs(x{2}-0.5) <= 0.25));
    % f(center) = f(center) + 8*pi*pi*(...
    %     sin(4*pi*(x{1}(center)-0.25)).*sin(2*pi*(x{2}(center)-0.25)).^2 + ...
    %     sin(4*pi*(x{2}(center)-0.25)).*sin(2*pi*(x{1}(center)-0.25)).^2 ...
    %     );
    % f(center) = f(center) - ...
    %     16*pi*pi*(3*sin(2*pi*(x{1}(center)-0.25)).^2 - 4*sin(2*pi*(x{1}(center)-0.25)).^4);
    %     f(center) = f(center) - ...
    %         100*((12*(x{1}(center)-0.25).^2.*(x{1}(center)-0.75).^4 + 12*(x{1}(center)-0.25).^4.*(x{1}(center)-0.75).^2 + ...
    %         2*4*4*(x{1}(center)-0.25).^3.*(x{1}(center)-0.75).^3));

    sigma = 0.25;
    f = f - exp(-((x{1}-0.5).^2 + (x{2}-0.5).^2)/sigma^2).*(4*((x{1}-0.5)/sigma).^2 - 2 + ...
        4*((x{2}-0.5)/sigma).^2 - 2)/sigma^2;


end

% Delta function (how to average to cell centered grid?)
% f = zeros(size(x{1}));
% find((abs(x{1}-0.5) <= eps) & (abs(x{2}-0.5) <= eps));