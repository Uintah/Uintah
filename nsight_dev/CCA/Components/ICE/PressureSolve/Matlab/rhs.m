function f = rhs(x)
%RHS  Right hand side function for the diffusion equation.
%   F = RHS(X) returns the diffusion equation's right hand side,
%   evaluated at the locations X. X is a Dx1 cell array of coordinates
%   X{1},...,X{D} in D dimensions. The global struct entry
%   PARAM.problemType controls which RHS is output. See "help inittest"
%   for a list of possible problems.
%
%   See also: TESTDISC, EXACTSOLUTION, RHSBC, DIFFUSION.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments.

globalParams;

switch (param.problemType)

    case {'linear','Lshaped','jump_linear'}
        f = zeros(size(x{1}));

    case 'quad1',
        f       = -2*ones(size(x{1}));
        
    case 'quad2',
        f       = zeros(size(x{1}));
        for d = 1:param.dim,
            g   = 2*ones(size(x{1}));
            for m = 1:param.dim
                if (m ~= d)
                    g = g .* x{m}.*(1-x{m});
                end
            end
            f   = f + g;
        end

    case 'sinsin',
        f       = param.dim*pi*pi*ones(size(x{1}));
        for d = 1:param.dim
            f = f .* sin(pi*x{d});
        end

    case 'GaussianSource',
        K       = 1;
        x0      = repmat(0.5,[1 param.dim]);
        sigma   = repmat(0.05,[1 param.dim]);
        f       = K * ones(size(x{1}));
        for d = 1:param.dim
            f = f .* exp(-((x{d}-x0(d)).^2/sigma(d)^2));
        end
        g       = zeros(size(x{1}));
        for d = 1:param.dim
            g = g + (-4*((x{d}-x0(d))/sigma(d)).^2 + 2)/sigma(d)^2;
        end
        f       = f .* g;

    case 'jump_quad',
        f       = -ones(size(x{1}));
    
    case 'diffusion_quad_linear',
        f       = -2 - 4*x{1};

    case 'diffusion_quad_quad',
        f       = -2 - 6*x{1}.^2;

    otherwise,
        error('Unknown problem type');
end
