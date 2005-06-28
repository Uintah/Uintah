function grid = updateGrid(grid)
%UPDATEGRID  Update AMR grid tree.
%   After adding or deleting a patch, this function is called to update
%   index and subscript offsets of each patch, and update the cellIndex
%   maps of each patch.
%
%   See also: ADDGRIDPATCH.

globalParams;

if (param.verboseLevel >= 1)
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
