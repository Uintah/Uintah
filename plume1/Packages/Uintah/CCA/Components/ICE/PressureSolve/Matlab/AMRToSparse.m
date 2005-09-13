function x = AMRToSparse(u,grid,T,convert)
%AMRTOSPARSE  Convert AMR data to sparse system data.
%   X = AMRTOSPARSE(U,GRID,T,CONVERT) converts the function U defined on
%   the AMR hierarchy GRID into a vector X of the composite grid variables.
%   T is a transformation that converts the flux-based ghost points to
%   pointwise values at all patch boundaries, so that X contains only point
%   values of the function. If CONVERT=1, we indeed apply T; if CONVERT=0,
%   we do not (e.g. if U represents residuals / truncation errors).
%
%   See also: TESTADAPTIVE, SPARSETOAMR.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

y = zeros(grid.totalVars,1);

for k = 1:grid.numLevels,
    level = grid.level{k};
    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};
        y(P.cellIndex) = u{k}{q}(:);
    end
end

if (convert)
    x = T*y;                % y contains U values, x contains discretization variables
else
    x = y;
end
