function g = rhsBC(x)
%RHSBC  Right hand side function for Dirichlet boundary conditions.
%   G = RHS(X) returns the right hand side for Dirichlet boundaries where
%   we enforce U = G. G is evaluated at the locations X. X is a Dx1 cell
%   array of coordinates X{1},...,X{D} in D dimensions. The global struct
%   entry PARAM.problemType controls which RHS is output.
%
%   See also: TESTDISC, EXACTSOLUTION, RHS.

globalParams;

switch (param.problemType)

    case {'ProblemA','ProblemB','quadratic'},
        g       = exactSolution(x);

    otherwise,
        error('Unknown problem type');
end

% g = zeros(size(x{1}));
% sigma = 0.25;
% g = g + exp(-((x{1}-0.5).^2 + (x{2}-0.5).^2)/sigma^2);
% g = log(sqrt((x{1}-0.5).^2 + (x{2}-0.5).^2));
