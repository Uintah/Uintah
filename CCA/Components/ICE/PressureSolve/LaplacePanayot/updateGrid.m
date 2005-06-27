function grid = updateGrid(grid)
global verboseLevel
% Update grid tree:
% - Update ID and base index for each patch.
% - Update index maps of each patch.

if (verboseLevel >= 1)
    fprintf('--- Updating grid ---\n');
end
index = 0;
for k = 1:grid.numLevels,
    for q = 1:grid.level{k}.numPatches,
        P           = grid.level{k}.patch{q};

        % Update base index
        P.offsetInd = index;

        % Create cell index map
        sub         = cell(grid.dim,1);
        for d = 1:grid.dim
            sub{d}  = [P.ilower(d)-1:P.iupper(d)+1] + P.offsetSub(d);
        end
        matSub      = cell(grid.dim,1);
        [matSub{:}] = ndgrid(sub{:});
        P.cellIndex = sub2ind(P.size,matSub{:}) + P.offsetInd;

        grid.level{k}.patch{q} = P;
        index       = index + prod(P.size);
    end
end
grid.totalVars      = index;
