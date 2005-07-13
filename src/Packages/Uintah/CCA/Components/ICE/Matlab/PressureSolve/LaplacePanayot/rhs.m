function f = rhs(x)
%RHS  Right hand side function for the diffusion equation.
%   F = RHS(X) returns the diffusion equation's right hand side,
%   evaluated at the locations X. X is a Dx1 cell array of coordinates
%   X{1},...,X{D} in D dimensions. The global struct entry
%   PARAM.problemType controls which RHS is output. See "help exactsolution"
%   for a list of possible problems.
%
%   See also: TESTDISC, EXACTSOLUTION, RHSBC, DIFFUSION.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments.

globalParams;

switch (param.problemType)

    case {'linear','Lshaped','jump_linear'}
        f = zeros(size(x{1}));

    case 'quadratic',
        f       = 2*(x{1}.*(1-x{1}) + x{2}.*(1-x{2}));

    case 'sinsin',
        f       = 2*pi*pi*sin(pi*x{1}).*sin(pi*x{2});

    case 'GaussianSource',
        K       = 1;
        x0      = [0.5 0.5];
        sigma   = [0.05 0.05];
        f       = -exp(-((x{1}-x0(1)).^2/sigma(1)^2 + (x{2}-x0(2)).^2/sigma(2)^2)) .* ( ...
            (4*((x{1}-x0(1))/sigma(1)).^2 - 2)/sigma(1)^2 + ...
            (4*((x{2}-x0(2))/sigma(2)).^2 - 2)/sigma(2)^2);

    case 'jump_quadratic',
        f       = -ones(size(x{1}));
    
    case 'diffusion_const',
        f       = -2*ones(size(x{1}));

    case 'diffusion_quadratic',
        f       = -2 - 6*x{1}.^2;

    otherwise,
        error('Unknown problem type');
end
