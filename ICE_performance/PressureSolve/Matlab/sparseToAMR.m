function u = sparseToAMR(x,grid,TI,convert)
%SPARSETOAMR  Convert sparse system data to AMR data.
%   U = SPARSETOAMR(X,GRID,TI,CONVERT) converts vector X of the composite grid
%   variables to a function U defined on the AMR hierarchy GRID.
%   TI is the inverse of T, a transformation that converts pointwise values at all patch boundaries
%   to flux-based ghost pointsIf CONVERT=1, we indeed apply TI; if CONVERT=0,
%   we do not (e.g. if X represents residuals / truncation errors).
%
%   See also: TESTADAPTIVE, AMRTOSPARSETO.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (convert)
    y = TI*x;                           % y contains values of U at all gridpoints
else
    y = x;
end

u = cell(grid.numLevels,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    u{k} = cell(level.numPatches,1);
    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};
        u{k}{q} = y(P.cellIndex);
    end
end
