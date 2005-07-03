function f = rhs(x)
%RHS  Right hand side function.
%   F = RHS(X) returns the diffusion equation's right hand side,
%   evaluated at the locations X. X is a Dx1 cell array of coordinates
%   X{1},...,X{D} in D dimensions. The global struct entry
%   PARAM.problemType controls which RHS is output.
%
%   See also: TESTDISC, EXACTSOLUTION, RHSBC.

globalParams;

switch (param.problemType)

    case 'quadratic',
        f       = 2*(x{1}.*(1-x{1}) + x{2}.*(1-x{2}));

    case 'ProblemA',
        f       = 2*pi*pi*sin(pi*x{1}).*sin(pi*x{2});

    case 'ProblemB',
        K       = 1;
        x0      = [0.5 0.5];
        sigma   = [0.05 0.05];
        f       = -exp(-((x{1}-x0(1)).^2/sigma(1)^2 + (x{2}-x0(2)).^2/sigma(2)^2)) .* ( ...
            (4*((x{1}-x0(1))/sigma(1)).^2 - 2)/sigma(1)^2 + ...
            (4*((x{2}-x0(2))/sigma(2)).^2 - 2)/sigma(2)^2);

    otherwise,
        error('Unknown problem type');
end

%f = zeros(size(x{1}));

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

% Delta function (how to average to cell centered grid?)
% f = zeros(size(x{1}));
% find((abs(x{1}-0.5) <= eps) & (abs(x{2}-0.5) <= eps));
