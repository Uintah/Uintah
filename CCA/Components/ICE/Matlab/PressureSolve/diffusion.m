function a = diffusion(x)
%DIFFUSION  Diffusion coefficient.
%   A = DIFFUSION(X) returns the value of the diffusion coefficient
%   at (X,Y). X is a cell array of size Dx1, where D is the number of
%   dimensions, and A is an array of size size(X{1}).
%   PARAM.problemType selects the problem type. See "help exactsolution"
%   for a list of possible problems.
%
%   See also: EXACTSOLUTION, RHS, RHSBC.

% Revision history:
% 12-JUL-2005    Oren Livne    Created
% 13-JUL-2005    Oren Livne    Moved from a(x,y) to general dimension a(x)

globalParams;

d                   = length(x);
a                   = zeros(size(x{1}));

switch (param.problemType)
    % Problems with a=1 (Laplace operator)
    case {'linear','quad1','quad2','sinsin','GaussianSource','Lshaped'},
        a           = repmat(1.0,size(x{1}));

    case {'jump_linear'},
        % Piecewise constant diffusion coefficient with a big jump at x1=0.5.
        x0          = 0.5;
        aLeft       = 1.0;
        aRight      = 1.0e+6;
        left        = find(x{1} < x0);
        right       = find(x{1} >= x0);
        a(left)     = aLeft;
        a(right)    = aRight;

    case {'jump_quad'},
        % Piecewise constant diffusion coefficient with a big jump at x1=0.5.
        x0          = 0.53;
        aLeft       = 10.0;
        aRight      = 1.0e+6;
        left        = find(x{1} < x0);
        right       = find(x{1} >= x0);
        a(left)     = aLeft;
        a(right)    = aRight;

    case 'diffusion_quad_linear',
        a           = 1 + x{1};

    case 'diffusion_quad_quad',
        a           = 1 + x{1}.^2;

    otherwise,
        error('Unknown problem type');
end
