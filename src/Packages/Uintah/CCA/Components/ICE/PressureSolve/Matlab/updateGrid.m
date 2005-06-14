function grid = updateGrid(grid)
global verboseLevel
% Update grid tree:
% - Update ID and base index for each patch.
% - Update index maps of each patch.

if (verboseLevel >= 1)
    fprintf('--- Updating grid ---\n');
end
baseIndex = 1;
for k = 1:grid.numLevels,
    for q = 1:grid.level{k}.numPatches,
        P = grid.level{k}.patch{q};

        % Update base index
        P.baseIndex = baseIndex;

        % Create cell index map
        i1 = [P.ilower(1)-1:P.iupper(1)+1] + P.offset(1);
        i2 = [P.ilower(2)-1:P.iupper(2)+1] + P.offset(2);
        [mat1,mat2] = ndgrid(i1,i2);
        P.cellIndex = sub2ind(P.size,mat1,mat2) + P.baseIndex - 1;

        grid.level{k}.patch{q} = P;
        baseIndex = baseIndex + prod(P.size);
    end
end
grid.totalVars = baseIndex-1;
