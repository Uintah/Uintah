function g = rhsBC(x)
%RHSBC  Right hand side function for Dirichlet boundary conditions.
%   G = RHS(X) returns the right hand side for Dirichlet boundaries where
%   we enforce U = G. G is evaluated at the locations X. X is a Dx1 cell
%   array of coordinates X{1},...,X{D} in D dimensions. The global struct
%   entry PARAM.problemType controls which RHS is output. See "help inittest"
%   for a list of possible problems.
%
%   See also: TESTDISC, EXACTSOLUTION, RHS, DIFFUSION.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments.

globalParams;

switch (param.problemType)

    % Problems with Dirichlet B.C. at all boundaries
    case {'linear','quad1','quad2','sinsin','GaussianSource','Lshaped',...
            'jump_linear','jump_quad','diffusion_quad_linear','diffusion_quad_quad'},
        g       = exactSolution(x);

    otherwise,
        error('Unknown problem type');
end
