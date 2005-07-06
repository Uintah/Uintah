function u = sparseToAMR(x,grid,TI,convert)
% Strip the long vector of composite grid variables x to patch-based data
% in u.

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
