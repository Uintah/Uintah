function u = exactSolution(x)
%EXACTSOLUTION  Exact solution for test problems.
%   U = EXACTSOLUTION(X) returns the exact solution evaluated at the
%   locations X. X is a Dx1 cell array of coordinates X{1},...,X{D} in D
%   dimensions. The global struct entry PARAM.problemType controls which
%   solution is output.
%
%   See also: TESTDISC, RHS, RHSBC, EXACTSOLUTIONAMR.

% Exact solution

globalParams;

switch (param.problemType)

    case 'ProblemA',
        u       = sin(pi*x{1}).*sin(pi*x{2});

    case 'ProblemB',
        K       = 1;
        x0      = [0.5 0.5];
        sigma   = [0.25 0.25];
        u       = exp(-((x{1}-x0(1)).^2/sigma(1)^2 + (x{2}-x0(2)).^2/sigma(2)^2));

    otherwise,
        error('Unknown problem type');
end

%center = find((abs(x-0.5) <= 0.25) & (abs(y-0.5) <= 0.25));
%u(center) = u(center) + (sin(2*pi*(x(center)-0.25)).*sin(2*pi*(y(center)-0.25))).^2;
%u(center) = u(center) + sin(2*pi*(x(center)-0.25)).^4;
%u(center) = u(center) + 100*(x(center)-0.25).^4.*(x(center)-0.75).^4;
%u = x;
%u = log(sqrt((x{1}-0.5).^2 + (x{2}-0.5).^2));
