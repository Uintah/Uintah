function u = sparseToAMR(x,grid)
% Strip the long vector of composite grid variables x to patch-based data
% in u.
u = cell(grid.numLevels,1);

for k = 1:grid.numLevels,
    u{k} = cell(grid.level{k}.numPatches,1);

    for q = 1:grid.level{k}.numPatches,
        P = grid.level{k}.patch{q};
        u{k}{q} = x(P.cellIndex);
    end

end
