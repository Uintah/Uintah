function u = exactSolution(x)
%EXACTSOLUTION  Exact solution for test problems.
%   U = EXACTSOLUTION(X) returns the exact solution evaluated at the
%   locations X. X is a Dx1 cell array of coordinates X{1},...,X{D} in D
%   dimensions. The global struct entry param.problemType controls which
%   solution is output. See "help inittest" for a list of test cases.
%
%   See also: INITTEST, TESTDISC, RHS, RHSBC, EXACTSOLUTIONAMR, DIFFUSION.

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
        x0          = 0.53;
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
