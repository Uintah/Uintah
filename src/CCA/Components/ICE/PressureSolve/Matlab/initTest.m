function initTest
%INITTEST  Initialize parameter structure with test information.
%   INITTEST sets some PARAM fields that are useful for saving the results.
%   Call before starting any runs with this test case.
%   List of test cases:
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
%       'diffusion_linear_quad'
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
%   See also: TESTDISC, RHS, RHSBC, EXACTSOLUTION, DIFFUSION.

% Revision history:
% 17-JUL-2005    Oren Livne    Added comments.

globalParams;

switch (param.problemType)

    case 'linear',
        % u is a linear function (d-D)        
        param.longTitle     = 'Linear Solution';
        param.supportedDims = -1;           % Works in any dimension
        
    case 'quad1',
        param.longTitle     = 'Quadratic Solution $u=u(x_1)$';
        param.supportedDims = -1;           % Works in any dimension

    case 'quad2',
        param.longTitle     = 'Quadratic Solution';
        param.supportedDims = -1;           % Works in any dimension

    case 'sinsin',
        param.longTitle     = 'Smooth Solution';
        param.supportedDims = -1;           % Works in any dimension

    case 'GaussianSource',
        param.longTitle     = 'Localized Gaussian Source';
        param.supportedDims = -1;           % Works in any dimension
                
    case 'jump_linear',
        param.longTitle     = 'Piecewise Constant Diffusion, Piecewise Linear Solution';
        param.supportedDims = -1;           % Works in any dimension
        
    case 'jump_quad',
        param.longTitle     = 'Piecewise Constant Diffusion, Piecewise Quadratic Solution';
        param.supportedDims = -1;           % Works in any dimension

    case 'diffusion_quad_linear',
        param.longTitle     = 'Linear Diffusion, Quadratic Solution';
        param.supportedDims = -1;           % Works in any dimension

    case 'diffusion_quad_quad',
        param.longTitle     = 'Quadratic Diffusion, Quadratic Solution';
        param.supportedDims = -1;           % Works in any dimension

    case 'Lshaped',
        param.longTitle     = 'L-Shaped Reentrant Corner';
        param.supportedDims = 2;            % Works only 2-D

    otherwise,
        error('Unknown problem type');
end
