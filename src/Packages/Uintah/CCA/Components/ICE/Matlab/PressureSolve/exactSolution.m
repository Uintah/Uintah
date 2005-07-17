function u = exactSolution(x)
%EXACTSOLUTION  Exact solution for test problems.
%   U = EXACTSOLUTION(X) returns the exact solution evaluated at the
%   locations X. X is a Dx1 cell array of coordinates X{1},...,X{D} in D
%   dimensions. The global struct entry param.problemType controls which
%   solution is output. Options are:
%       'linear'
%           U = linear function with Dirichlet B.C. on the d-D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth.
%       'quad1'
%           U = quadratic function with Dirichlet B.C. on the d-D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth. U
%           depends only on x1.
%       'quad2'
%           U = quadratic function with Dirichlet B.C. on the d-D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth.
%       'sinsin'
%           U = sin(pi*x1)*sin(pi*x2) with Dirichlet B.C. on the d-D unit
%           square. Diffusion a=1 (Laplace operator). U is smooth.
%       'GaussianSource'
%           U = is the solution to Laplace's equation on the d-D unit square
%           with Gaussian right hand side, centered at (0.5,...,0.5) with
%           standard deviation of (0.05,...,0.05) on the 2D unit square. 
%           Diffusion a=1 (Laplace operator). U is smooth but localized 
%           around the source, so at coarse level it is beneficial to locally
%           refine around the center of the domain.
%       'jump_linear'
%           Piecewise constant diffusion coefficient a on the d-D unit
%           square. a has a big jump at the hyperplane x1=0.5 (a,u depend only on
%           x1; a = aLeft for x1 <= 0.5, a = aRight otherwise). Piecewise linear
%           solution U that solves Laplace's equation with this a.
%       'jump_quad'
%           Like jump_linear, but with a piecewise quadratic solution U
%           that solves Poisson's equation with RHS = -1, this a, and appropriate
%           B.C.
%       'diffusion_quad_quad'
%           a = 1 + x{1} and u = x{1}^2 (d-D; linear diffusion and smooth
%           quadratic solution). Appropriate RHS and Dirichlet BC.
%       'diffusion_quad_quad'
%           a = 1 + x{1}^2 and u = x{1}^2 (d-D; quadratic diffusion and smooth
%           quadratic solution). Appropriate RHS and Dirichlet BC.
%       'Lshaped' (2-D only)
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

    case 'linear',
        % u is a linear function (d-D)        
        u = ones(size(x{1}));
        for d = 1:param.dim
            u = u + x{d};
        end
        
    case 'quad1',
        % Smooth diffusion and smooth solution, depends only on x1 (d-D).
        u           = x{1}.^2;

    case 'quad2',
        % u is a quadratic function that satisfies zero Dirichlet B.C.
        % (d-D).
        u = ones(size(x{1}));
        for d = 1:param.dim
            u = u .* x{d}.*(1-x{d});
        end

    case 'sinsin',
        % u is a smooth function (d-D).
        u = ones(size(x{1}));
        for d = 1:param.dim
            u = u .* sin(pi*x{d});
        end

    case 'GaussianSource',
        % u is smooth, solves Laplace's equation with a Gaussian RHS (d-D).
        K       = 1;
        x0      = repmat(0.5,[1 param.dim]);
        sigma   = repmat(0.05,[1 param.dim]);
        u       = K * ones(size(x{1}));
        for d = 1:param.dim
            u = u .* exp(-((x{d}-x0(d)).^2/sigma(d)^2));
        end
                
    case 'jump_linear',
        % Piecewise constant diffusion coefficient with a big jump at
        % x{1}=x0. Piecewise linear solution (d-D).
        u           = zeros(size(x{1}));
        x0          = 0.5;
        aLeft       = 1.0;
        aRight      = 1.0e+6;
        left        = find(x{1} < x0);
        right       = find(x{1} >= x0);
        u(left)     = (x{1}(left) - x0)/aLeft;
        u(right)    = (x{1}(right) - x0)/aRight;
        
    case 'jump_quad',
        % Piecewise constant diffusion coefficient with a big jump at
        % x{1}=x0. Piecewise quadratic solution (d-D).
        u           = zeros(size(x{1}));
        x0          = 0.5;
        aLeft       = 10.0;
        aRight      = 1.0e+6;
        left        = find(x{1} < x0);
        right       = find(x{1} >= x0);
        u(left)     = (x{1}(left) - x0).^2/(2*aLeft);
        u(right)    = (x{1}(right) - x0).^2/(2*aRight);

    case 'diffusion_quad_linear',
        % Linear diffusion and quadratic solution (d-D).
        u           = x{1}.^2;

    case 'diffusion_quad_quad',
        % Quadratic diffusion and quadratic solution (d-D).
        u           = x{1}.^2;

    case 'Lshaped',
        % L-shaped domain, u has a singularity due to the re-entrant corner
        % at x0 (2-D).
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
