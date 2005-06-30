function x = AMRToSparse(u,grid,T,convert)
% Strip the long vector of composite grid variables x to patch-based data
% in u.

y = zeros(grid.totalVars,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
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
