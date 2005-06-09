function [A,b] = setupOperator(grid)

Alist   = zeros(0,3);
b       = zeros(grid.totalVars,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    numPatches  = length(level.numPatches);
    s           = level.stencilOffsets;
    numEntries  = size(s,1);
    h = level.h;

    for q = 1:numPatches,
        P = grid.level{k}.patch{q};
        map = P.cellIndex;

        [AlistPatch,bPatch] = setupOperatorPatch(grid,k,q);
        Alist = [Alist; AlistPatch];
        b(map(:)) = bPatch;
    end
end

A = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
