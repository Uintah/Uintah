function u = exactSolution(x)
%EXACTSOLUTION  Exact solution for test problems.
%   U = EXACTSOLUTION(X) returns the exact solution evaluated at the
%   locations X. X is a Dx1 cell array of coordinates X{1},...,X{D} in D
%   dimensions. The global struct entry param.problemType controls which
%   solution is output. Options are:
%       'quadratic'
%           U = quadratic function with Dirichlet B.C. on the 2D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth.
%       'sinsin'
%           U = sin(pi*x1)*sin(pi*x2) with Dirichlet B.C. on the 2D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth.
%       'GaussianSource'
%           U = is the solution to Laplace's equation with Gaussian right
%           hand side, centered at (0.5,0.5) with standard deviation of
%           (0.05,0.05) on the 2D unit square. Diffusion a=1 (Laplace
%           operator). U is smooth but localized around the source, so at
%           coarse level it is beneficial to locally refine around the
%           center of the domain.
%       'Lshaped'
%           U = r^(2/3)*sin(2*theta/3) is the solution the Laplace's
%           equation with Dirichlet B.C. on the L-shaped domain [0,1]^2 \
%           [0.5,1]^2.Diffusion a=1 (Laplace operator). This is a
%           re-entrant corner problem where U is singular.
%
%   See also: TESTDISC, RHS, RHSBC, EXACTSOLUTIONAMR, DIFFUSION.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments.

globalParams;

switch (param.problemType)

    case 'quadratic',
        u       = x{1}.*(1-x{1}).*x{2}.*(1-x{2});

    case 'sinsin',
        u       = sin(pi*x{1}).*sin(pi*x{2});

    case 'GaussianSource',
        K       = 1;
        x0      = [0.5 0.5];
        sigma   = [0.05 0.05];
        u       = exp(-((x{1}-x0(1)).^2/sigma(1)^2 + (x{2}-x0(2)).^2/sigma(2)^2));
        
    case 'Lshaped',
        x0          = [0.5 0.5];
        r           = sqrt((x{1}-x0(1)).^2+(x{2}-x0(2)).^2);
        t           = atan2(x{2}-x0(2),x{1}-x0(1));
        t           = mod(-t+2*pi,2*pi);
        alpha       = 2/3;
        u           = r.^(alpha).*sin(alpha*t);
        u(find(min(x{1}-x0(1),x{2}-x0(2)) >= 0)) = 0.0;
        
    otherwise,
        error('Unknown problem type');
end
